"""Process a saved scan in the browser: export with settings, and cut out an object mesh.

    leap2-scanner process [scans/<name>]      (default: the newest scan)

Export: loop closure, mesh resolution, STL size budget, smoothing; progress and a preview.
Object: a box (auto-fitted to what the scan was centred on, or placed with the gizmo) is meshed
at up to 1 mm. If the object rests on a surface, the box floor lies on it: cut the surface off
and close the object there, so the STL is ready to print.
"""

from __future__ import annotations

import threading
import time
import traceback
import webbrowser
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vtf

from .export import Session, read_ply_points
from .fusion import backproject, estimate_up

MAX_DISPLAY_POINTS = 400_000
FLOOR = {"keep everything": "keep", "cut off": "open", "cut off + close": "closed"}  # box floor options
BOX_COLOR = (255, 140, 0)
NOT_MESHES = {"pointcloud.ply", "points3D.ply"}


def _newest_scan(folder: Path) -> Path:
    scans = sorted(p for p in folder.iterdir() if (p / "keyframes.json").exists())
    if not scans:
        raise SystemExit(f"no scans in {folder}/ - scan something first, or pass a scan folder")
    return scans[-1]


def _load_cloud(s: Session, log):
    """The exported point cloud, or (not exported yet) a quick one from the raw keyframe depth."""
    f = s.root / "pointcloud.ply"
    if f.exists():
        return read_ply_points(f)
    log("no pointcloud.ply yet (not exported): showing raw keyframe depth")
    pts, gray = [], []
    for r in s.records:
        left, d = s.load_frame(r)
        X, u, v = backproject(d, s.K, stride=6)
        pts.append(X @ r.T_wc[:3, :3].T + r.T_wc[:3, 3])
        gray.append(left[v, u])
    pts, gray = np.vstack(pts), np.concatenate(gray).astype(np.float32)
    _, keep = np.unique(np.floor(pts / 0.003).astype(np.int64), axis=0, return_index=True)
    return pts[keep], gray[keep]


def run_process(args):
    root = Path(args.session) if args.session else _newest_scan(Path(args.scans))
    s = Session.load(root)
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = False
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print(f"\nProcessing {root}: {url}\n", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    gui = server.gui
    lines: list[str] = []
    st = {"busy": False, "mesh": None, "cloud": None, "pts": None, "show": None, "depths": None}

    status = gui.add_markdown("")
    prog = gui.add_progress_bar(0.0, visible=False)
    log_md = gui.add_markdown("")

    def log(msg: str):
        print(msg, flush=True)
        lines.append(msg)
        log_md.content = "  \n".join(f"`{m}`" for m in lines[-7:])

    def progress(f: float):
        prog.visible = True
        prog.value = float(np.clip(f * 100, 0, 100))

    def describe():
        done = (root / "keyframes_odometry.json").exists()
        meshes = [p.name for p in sorted(root.glob("*.stl"))]
        status.content = (f"**{root.name}** · {len(s.records)} keyframes  \n"
                          f"loop closure: {'done' if done else 'not yet'} · "
                          f"exported: {'yes' if (root / 'pointcloud.ply').exists() else 'no'}  \n"
                          f"STL files: {', '.join(meshes) if meshes else 'none'}")

    # ------------------------------------------------------------------ scene
    pts_all, gray_all = _load_cloud(s, log)
    poses = [r.T_wc for r in s.records]
    up, _ = estimate_up(pts_all, poses)
    server.scene.set_up_direction(tuple(up))
    fwd = np.mean([T[:3, 2] for T in poses], 0)  # the side the scan was made from
    fwd -= up * (fwd @ up)
    fwd /= np.linalg.norm(fwd)

    def show_cloud():
        pts, gray = pts_all, gray_all
        if len(pts) > MAX_DISPLAY_POINTS:
            sel = np.random.default_rng(0).choice(len(pts), MAX_DISPLAY_POINTS, replace=False)
            pts, gray = pts[sel], gray[sel]
        st["pts"] = pts.astype(np.float32)
        g = np.clip(gray, 0, 255).astype(np.uint8)
        st["gray"] = np.stack([g, g, g], 1)
        st["cloud"] = server.scene.add_point_cloud("/cloud", st["pts"], st["gray"], point_size=0.0015,
                                                   visible=cb_cloud.value)

    C = np.array([T[:3, 3] for T in poses])
    path = server.scene.add_line_segments("/path", np.stack([C[:-1], C[1:]], 1), (255, 120, 0), thickness=0.001) \
        if len(C) > 1 else None

    # ------------------------------------------------------------------ GUI
    with gui.add_folder("View"):
        cb_cloud = gui.add_checkbox("Point cloud", True)
        cb_path = gui.add_checkbox("Camera path", True)
        sl_pt = gui.add_slider("Point size (mm)", 0.5, 6.0, 0.5, 1.5)
        dd_mesh = gui.add_dropdown("Mesh", ["(none)"], initial_value="(none)")

    lc_done = (root / "keyframes_odometry.json").exists()
    with gui.add_folder("Export whole scan"):
        cb_lc = gui.add_checkbox("Loop closure", True, hint="Drift-correct keyframe poses (~1-2 min per 100 keyframes)")
        cb_redo = gui.add_checkbox("Redo loop closure", False, visible=lc_done,
                                   hint="Already done for this scan; redo from the original odometry")
        num_vox = gui.add_number("Mesh voxel (mm)", 1.5, min=0.5, max=10.0, step=0.1)
        num_mb = gui.add_number("STL size (MB)", 40.0, min=0.0, max=1000.0, step=5.0,
                                hint="Decimate to about this size (0 = no limit)")
        sl_sm = gui.add_slider("Smoothing", 0, 20, 1, 10)
        cb_only = gui.add_checkbox("Mesh only", (root / "pointcloud.ply").exists(),
                                   hint="Keep the point cloud and splat dataset, only rebuild the mesh")
        b_export = gui.add_button("Run export", color="green")

    with gui.add_folder("Object"):
        b_fit = gui.add_button("Auto-fit box to object",
                               hint="Finds what the scan was centred on, and the surface it rests on if any")
        cb_box = gui.add_checkbox("Show box", True)
        vec_size = gui.add_vector3("Box size (cm)", (20.0, 20.0, 20.0), min=(1.0, 1.0, 1.0), step=0.5)
        num_ovox = gui.add_number("Voxel (mm)", 1.0, min=0.5, max=5.0, step=0.1)
        num_omb = gui.add_number("STL size (MB)", 20.0, min=0.0, max=500.0, step=5.0)
        sl_osm = gui.add_slider("Smoothing ", 0, 20, 1, 5)
        dd_bottom = gui.add_dropdown("Box floor", list(FLOOR), initial_value="keep everything",
                                     hint="Cut off what lies on the box floor (the surface the object rests on) "
                                          "and optionally close the object there. Auto-fit picks this for you.")
        num_cut = gui.add_number("Cut above floor (mm)", 3.0, min=0.0, max=30.0, step=0.5,
                                 hint="Lower for thin objects, higher for a rough surface")
        txt_name = gui.add_text("Name", "object")
        b_obj = gui.add_button("Mesh object", color="orange")

    buttons = (b_export, b_fit, b_obj)

    def busy(on: bool):
        st["busy"] = on
        for b in buttons:
            b.disabled = on
        if not on:
            prog.visible = False

    def mesh_files():
        return [p.name for p in sorted(root.glob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
                if p.name not in NOT_MESHES]

    def refresh_meshes(select: str | None = None):
        names = mesh_files()
        dd_mesh.options = ["(none)"] + names
        if select:
            dd_mesh.value = select
        describe()

    def show_mesh(name: str):
        import trimesh

        if st["mesh"] is not None:
            st["mesh"].remove()
            st["mesh"] = None
        if name == "(none)":
            return
        st["mesh"] = server.scene.add_mesh_trimesh("/mesh", trimesh.load(root / name, process=False))

    dd_mesh.on_update(lambda _: show_mesh(dd_mesh.value))
    cb_cloud.on_update(lambda _: setattr(st["cloud"], "visible", cb_cloud.value))
    if path is not None:
        cb_path.on_update(lambda _: setattr(path, "visible", cb_path.value))

    def look_at_box():
        """Point every open view at the crop box, from where the scan looked at it."""
        c = box_T()[:3, 3]
        d = 1.8 * float(np.linalg.norm(np.asarray(vec_size.value) / 100.0))
        rel = C - c
        dist = np.linalg.norm(rel, axis=1)
        facing = np.einsum("ij,ij->i", -rel / dist[:, None], np.array([T[:3, 2] for T in poses])) > 0.7
        view = (rel[facing] / dist[facing, None]).mean(0) if facing.any() else up * 0.6 - fwd
        view /= np.linalg.norm(view)
        for client in server.get_clients().values():
            client.camera.position = tuple(c + view * d)
            client.camera.look_at = tuple(c)
    sl_pt.on_update(lambda _: setattr(st["cloud"], "point_size", sl_pt.value / 1000))

    # ------------------------------------------------------------------ crop box
    box = {"T": np.eye(4), "size": np.array([0.2, 0.2, 0.2])}
    gizmo = server.scene.add_transform_controls("/box", scale=0.12, depth_test=False)
    solid = server.scene.add_box("/box/solid", color=BOX_COLOR, dimensions=(0.2, 0.2, 0.2), opacity=0.12)
    wire = server.scene.add_box("/box/wire", color=BOX_COLOR, dimensions=(0.2, 0.2, 0.2), wireframe=True)

    def box_T() -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = vtf.SO3(np.asarray(gizmo.wxyz)).as_matrix()
        T[:3, 3] = gizmo.position
        return T

    def highlight():
        """Tint the points inside the box."""
        T = box_T()
        P = (st["pts"] - T[:3, 3]) @ T[:3, :3]
        inside = np.all(np.abs(P) <= np.asarray(vec_size.value) / 200.0, 1)
        col = st["gray"].copy()
        col[inside] = (col[inside] * 0.55 + np.array(BOX_COLOR) * 0.45).astype(np.uint8)
        st["cloud"].colors = col

    def set_box(T: np.ndarray, size: np.ndarray):
        gizmo.position = T[:3, 3]
        gizmo.wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
        vec_size.value = tuple(np.round(size * 100, 1))
        highlight()

    @vec_size.on_update
    def _(_):
        dims = tuple(np.asarray(vec_size.value, np.float64) / 100.0)
        solid.dimensions = dims
        wire.dimensions = dims
        highlight()

    @gizmo.on_update
    def _(event):  # recolour when a drag ends (older viser versions have no phase: every update)
        if getattr(event, "phase", "end") == "end":
            highlight()

    @cb_box.on_update
    def _(_):
        for h in (gizmo, solid, wire):
            h.visible = cb_box.value

    # ------------------------------------------------------------------ actions
    def run(job):
        if st["busy"]:
            return
        busy(True)

        def target():
            try:
                job()
            except Exception as e:
                traceback.print_exc()
                log(f"failed: {e}")
            finally:
                busy(False)

        threading.Thread(target=target, daemon=True).start()

    def depths():
        if st["depths"] is None:
            from .pipeline import load_filtered_depths

            st["depths"] = load_filtered_depths(s, log)
        return st["depths"]

    @b_fit.on_click
    def _(_):
        def job():
            from .objects import fit_object_box

            log("fitting the box to the object...")
            T, size, info, on_surface = fit_object_box(pts_all, poses, depths(), s.K)
            dd_bottom.value = "cut off + close" if on_surface else "keep everything"
            set_box(T, size)
            look_at_box()
            log(info)

        run(job)

    @b_obj.on_click
    def _(_):
        def job():
            from .pipeline import object_mesh

            bottom = FLOOR[dd_bottom.value]
            name = "".join(c if c.isalnum() or c in "-_" else "_" for c in txt_name.value.strip()) or "object"
            mb = float(num_omb.value)
            mesh, fname = object_mesh(s, box_T(), np.asarray(vec_size.value) / 100.0, float(num_ovox.value) / 1000,
                                      int(sl_osm.value), int(mb * 1e6 / 50) if mb > 0 else None, bottom,
                                      cut=float(num_cut.value) / 1000, progress=progress, log=log, name=name)
            if fname:
                cb_cloud.value = False
                st["cloud"].visible = False
                cb_box.value = False
                for h in (gizmo, solid, wire):
                    h.visible = False
                if path is not None:
                    cb_path.value = False
                    path.visible = False
                refresh_meshes(fname + ".ply")
                look_at_box()

        run(job)

    @b_export.on_click
    def _(_):
        def job():
            nonlocal pts_all, gray_all
            from .pipeline import export_session

            mb = float(num_mb.value)
            t0 = time.time()
            mesh = export_session(s, float(num_vox.value) / 1000, progress, log, loop_closure=cb_lc.value,
                                  mesh_only=cb_only.value, smooth=int(sl_sm.value),
                                  max_faces=int(mb * 1e6 / 50) if mb > 0 else None, redo_loop_closure=cb_redo.value)
            st["depths"] = None  # poses / filtered depth may have changed
            pts_all, gray_all = _load_cloud(s, log)
            show_cloud()
            cb_redo.visible = (root / "keyframes_odometry.json").exists()
            cb_only.value = True
            if mesh is not None:
                newest = mesh_files()[0]
                refresh_meshes(newest)
            log(f"export done in {time.time() - t0:.0f} s")

        run(job)

    # ------------------------------------------------------------------ start
    show_cloud()
    refresh_meshes()
    center = np.median(pts_all, 0)
    radius = float(np.percentile(np.linalg.norm(pts_all - center, axis=1), 80))
    T0 = np.eye(4)  # initial box: upright, around the middle of the scan
    T0[:3, 2] = up
    T0[:3, 0] = fwd
    T0[:3, 1] = np.cross(up, fwd)
    T0[:3, 3] = center
    set_box(T0, np.array([0.2, 0.2, 0.2]))
    log("Auto-fit or move the box, then 'Mesh object' - or run a whole-scan export.")

    @server.on_client_connect
    def _(client):
        client.camera.position = tuple(center - fwd * 1.7 * radius + up * 1.3 * radius)
        client.camera.look_at = tuple(center)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
