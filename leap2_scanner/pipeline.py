"""Export pipeline shared by the live app and the offline `export` command."""

from __future__ import annotations

import json
import time
from typing import Callable

import cv2
import numpy as np

from .export import Session, export_splat_dataset, write_mesh, write_ply_points
from .fusion import backproject, cloud_from_depths, filter_depths, fuse_keyframes, robust_bounds


DERIVED = ("pointcloud.ply", "points3D.ply", "transforms.json", "sparse")


def _preserve_previous(root, names) -> str | None:
    """Move earlier export outputs into previous_export_<time>/ instead of overwriting them."""
    existing = [root / n for n in names if (root / n).exists()]
    if not existing:
        return None
    dest = root / time.strftime("previous_export_%Y%m%d_%H%M%S")
    dest.mkdir()
    for f in existing:
        f.rename(dest / f.name)
    return dest.name


def _free_mesh_name(root, base: str) -> str:
    if not (root / f"{base}.stl").exists() and not (root / f"{base}.ply").exists():
        return base
    return f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"


def restereo(session: Session, mode: str = "half", z_min: float = 0.15, z_max: float = 0.8,
             log: Callable[[str], None] = print):
    """Recompute keyframe depth from the saved rectified image pairs (e.g. with a newer matcher).

    The previous depth/ folder must already have been moved aside. No temporal fusion here:
    the in-between frames of a scan are not stored."""
    from types import SimpleNamespace

    from .stereo import DepthEstimator

    K = session.K
    de = DepthEstimator(SimpleNamespace(f=K[0, 0], baseline=session.baseline, K=K), z_min, z_max, mode=mode)
    out = session.root / "depth"
    out.mkdir()
    for rec in session.records:
        L = cv2.imread(str(session.root / "images" / (session.name(rec.idx, "L") + ".jpg")), cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(str(session.root / "images" / (session.name(rec.idx, "R") + ".jpg")), cv2.IMREAD_GRAYSCALE)
        z = de.compute(L, R)
        cv2.imwrite(str(out / (session.name(rec.idx) + ".png")), np.clip(np.round(z * 1000), 0, 65535).astype(np.uint16))
    log(f"recomputed depth of {len(session.records)} keyframes ({mode}-resolution matcher)")


def export_session(session: Session, voxel: float, progress: Callable[[float], None] | None = None,
                   log: Callable[[str], None] = print, max_voxels: int = 350**3, loop_closure: bool = False,
                   mesh_only: bool = False, mesh_name: str = "mesh", smooth: int = 10,
                   restereo_mode: str | None = None, detail_voxel: float = 0.001, max_faces: int | None = 800_000,
                   redo_loop_closure: bool = False):
    """Point cloud, splat dataset and TSDF mesh from the saved keyframes. Returns the mesh.

    Depth maps are multi-view consistency filtered first: single-view stereo mismatches
    otherwise leave ghost copies of edges in the map. With `loop_closure`, keyframe poses
    are drift-corrected first (original odometry kept in keyframes_odometry.json); a scan
    that already has that file is not corrected again unless `redo_loop_closure`, which
    starts over from the original odometry. `max_faces` decimates the meshes (STL size is
    50 bytes per triangle: 800k ~ 40 MB).
    `restereo_mode` recomputes the keyframe depth from the saved images first ("half"/"full").
    `mesh_only` leaves point cloud and splat files untouched. Nothing earlier is overwritten:
    previous outputs move to previous_export_<time>/, new meshes get a time suffix."""
    t0 = time.time()
    session.save_index()
    root = session.root
    keep = ("depth_filtered",) + (() if mesh_only else DERIVED) + (("depth",) if restereo_mode else ())
    moved = _preserve_previous(root, keep)
    if moved:
        log(f"earlier export files moved to {moved}/")
    if restereo_mode:
        restereo(session, restereo_mode, log=log)
    frames = [session.load_frame(r) for r in session.records]
    if not frames:
        raise RuntimeError("No keyframes to export")
    images = [f[0] for f in frames]
    raw = [f[1] for f in frames]

    backup = root / "keyframes_odometry.json"
    if loop_closure and backup.exists() and not redo_loop_closure:
        log("poses are already drift-corrected (keyframes_odometry.json) - skipping loop closure")
    elif loop_closure:
        from .optimize import optimize_poses

        if backup.exists():  # redo: start over from the original odometry
            start = [np.array(k["T_wc"]) for k in json.loads(backup.read_text())["keyframes"]]
        else:  # keep the original odometry once
            backup.write_text((root / "keyframes.json").read_text())
            start = [r.T_wc for r in session.records]
        poses = optimize_poses(images, raw, start, session.K, session.size, session.baseline, log)
        for rec, T in zip(session.records, poses):
            rec.T_wc = T
        session.save_index()
    poses = [r.T_wc for r in session.records]

    depths = filter_depths(raw, poses, session.K)
    kept = sum(int((d > 0).sum()) for d in depths) / max(1, sum(int((d > 0).sum()) for d in raw))
    log(f"depth consistency filter kept {kept*100:.0f}% of depth pixels ({time.time()-t0:.0f} s)")
    (root / "depth_filtered").mkdir()
    for rec, d in zip(session.records, depths):
        cv2.imwrite(str(root / "depth_filtered" / (session.name(rec.idx) + ".png")),
                    np.clip(np.round(d * 1000), 0, 65535).astype(np.uint16))

    pts, gray = cloud_from_depths(depths, images, poses, session.K)
    if len(pts) < 100:
        raise RuntimeError("Too few consistent depth points to export")
    if not mesh_only:
        log(f"writing point cloud ({len(pts):,} points)")
        write_ply_points(root / "pointcloud.ply", pts, gray)
        log("writing Gaussian-splat dataset (COLMAP + transforms.json)")
        export_splat_dataset(session, pts, gray)

    lo, hi = robust_bounds(pts, margin=0.02)
    log(f"TSDF fusion of {len(depths)} keyframes (voxel {voxel*1000:.1f} mm)...")
    vol = fuse_keyframes(zip(depths, images, poses), len(depths), session.K, voxel, (lo, hi), pts, progress,
                         max_voxels)
    if vol.voxel > voxel * 1.01:
        log(f"memory limit (--max-voxels): voxel grown to {vol.voxel*1000:.2f} mm")
    log("extracting mesh (marching cubes)...")
    mesh = vol.extract_mesh(smooth=smooth, max_faces=max_faces)
    if mesh is None or len(mesh.faces) == 0:
        log("no surface found for mesh")
        return None
    mesh_name = _free_mesh_name(root, mesh_name)  # never overwrite an earlier mesh
    write_mesh(root, mesh, mesh_name)
    log(f"mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces -> {mesh_name}.stl "
        f"({(84 + 50 * len(mesh.faces)) / 1e6:.0f} MB) / {mesh_name}.ply ({time.time()-t0:.0f} s)")

    detail = [i for i, r in enumerate(session.records) if r.detail]
    if detail and detail_voxel > 0:
        detail_mesh(session, depths, images, poses, detail, detail_voxel, progress, log, mesh_name + "_detail",
                    max_faces)
    log(f"export took {time.time()-t0:.0f} s")
    return mesh


def detail_mesh(session: Session, depths, images, poses, detail: list[int], voxel: float,
                progress: Callable[[float], None] | None, log: Callable[[str], None], name: str,
                max_faces: int | None = None):
    """High-resolution mesh of the area covered by detail-mode keyframes, fused from all keyframes."""
    K = session.K
    # the subject of a close-up pass: centre of the view, near the typical working distance
    size = session.size
    seen = []
    for i in detail:
        X, u, v = backproject(depths[i], K, stride=4)
        centre = (np.abs(u - size / 2) < 0.3 * size) & (np.abs(v - size / 2) < 0.3 * size)
        if centre.sum() < 20:
            continue
        z_near = min(0.35, 1.3 * float(np.median(X[centre, 2])))
        X = X[centre & (X[:, 2] < z_near)]
        seen.append(X @ poses[i][:3, :3].T + poses[i][:3, 3])
    seen = np.vstack(seen) if seen else np.empty((0, 3))
    if len(seen) < 100:
        log("detail mode: too little depth in the detail keyframes for a detail mesh")
        return None
    lo = np.percentile(seen, 2, axis=0) - 0.01
    hi = np.percentile(seen, 98, axis=0) + 0.01
    log(f"detail mesh: {len(detail)} detail keyframes, region {np.round((hi - lo) * 100, 1)} cm, "
        f"voxel {voxel*1000:.1f} mm...")
    band = []
    for d, T in zip(depths, poses):  # all keyframes that saw the region contribute
        X, _, _ = backproject(d, K, stride=1)
        P = X @ T[:3, :3].T + T[:3, 3]
        band.append(P[np.all((P >= lo) & (P <= hi), 1)])
    band = np.vstack(band)
    band = band[np.unique(np.floor(band / voxel).astype(np.int64), axis=0, return_index=True)[1]]
    vol = fuse_keyframes(zip(depths, images, poses), len(depths), K, voxel, (lo, hi), band, progress, 600**3)
    mesh = vol.extract_mesh(smooth=3, max_faces=max_faces)
    if mesh is None or len(mesh.faces) == 0:
        log("detail mesh: no surface found")
        return None
    name = _free_mesh_name(session.root, name)
    write_mesh(session.root, mesh, name)
    log(f"detail mesh: {len(mesh.faces):,} faces at {vol.voxel*1000:.1f} mm -> {name}.stl")
    return mesh


def load_filtered_depths(session: Session, log: Callable[[str], None] = print):
    """The consistency-filtered keyframe depths of the last export (computed now if there is none)."""
    folder = session.root / "depth_filtered"
    files = [folder / (session.name(r.idx) + ".png") for r in session.records]
    if all(f.exists() for f in files):
        return [cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0 for f in files]
    log("no complete depth_filtered/ (not exported yet?) - filtering the depth now")
    raw = [session.load_frame(r)[1] for r in session.records]
    return filter_depths(raw, [r.T_wc for r in session.records], session.K)


def _drop_fragments(mesh, frac: float = 0.01, min_faces: int = 200):
    """Keep connected pieces with at least `frac` of the largest piece's faces."""
    import trimesh

    if len(mesh.faces) == 0:
        return mesh
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    sizes = np.bincount(labels)
    mesh.update_faces(sizes[labels] >= min(sizes.max(), max(min_faces, frac * sizes.max())))
    mesh.remove_unreferenced_vertices()
    return mesh


def _refine_floor(mesh, floor: np.ndarray, up: np.ndarray, log):
    """If a surface lies on the box floor (what the object rests on), its exact plane from the
    fused mesh - much smoother than the point cloud, which far-away views blur by up to ~1 cm.
    Otherwise the box floor itself."""
    from .fusion import _dominant_plane

    V = np.asarray(mesh.vertices)
    near = V[np.abs((V - floor) @ up) < 0.006]  # only the floor: a big flat face of the object sits higher
    if len(near) < 500:
        return floor, up
    sub = near[np.random.default_rng(0).choice(len(near), min(len(near), 30000), replace=False)]
    n, p, frac = _dominant_plane(sub, iters=300, tol=0.0015)
    if n is None or frac < 0.2 or abs(n @ up) < 0.95:  # no clear surface there, or not parallel to the floor
        return floor, up
    n = n if n @ up > 0 else -n
    p = p + n * float(np.median((sub[np.abs((sub - p) @ n) < 0.0015] - p) @ n))
    log(f"object: supporting surface from the mesh, {((p - floor) @ up) * 1000:+.1f} mm from the box floor, "
        f"tilted {np.degrees(np.arccos(min(1.0, n @ up))):.1f}°")
    return p, n


def object_mesh(session: Session, T_box: np.ndarray, size, voxel: float = 0.001, smooth: int = 5,
                max_faces: int | None = 400_000, bottom: str = "keep", cut: float = 0.003,
                progress: Callable[[float], None] | None = None, log: Callable[[str], None] = print,
                name: str = "object"):
    """Fine mesh of everything inside an oriented box (T_box: box -> world, `size` in metres,
    centred on the box origin).

    The whole-scan mesh has to cover the scene, so its voxel is ~1.5-2 mm; a box around one
    object can use 1 mm. Fused in the box frame from every keyframe that sees the box.
    `bottom` treats the box floor (z = -size/2), where a supporting surface lies if there is one:
    "keep" meshes everything in the box, "open" cuts `cut` above the floor (removing the surface),
    "closed" also closes the cut outline with a flat face (printable).
    Written in world coordinates like mesh.stl, under a new name (nothing is overwritten)."""
    from .fusion import _decimate
    from .objects import cap_bottom

    t0 = time.time()
    K = session.K
    size = np.asarray(size, np.float64)
    lo, hi = -size / 2, size / 2
    below = 0.015 if bottom in ("closed", "open") else 0.0
    lo = lo - np.array([0, 0, below])  # also mesh what lies just below the floor: it gives the exact cut plane
    T_bw = np.linalg.inv(T_box)
    depths = load_filtered_depths(session, log)
    use, band = [], []
    for k, (d, rec) in enumerate(zip(depths, session.records)):
        X, _, _ = backproject(d, K, stride=2)
        T = T_bw @ rec.T_wc
        P = X @ T[:3, :3].T + T[:3, 3]
        inside = np.all((P >= lo) & (P <= hi), 1)
        if inside.sum() > 200:
            use.append(k)
            band.append(P[inside])
    if not use:
        log("object: no keyframe sees depth inside the box")
        return None, None
    band = np.vstack(band)
    band = band[np.unique(np.floor(band / voxel).astype(np.int64), axis=0, return_index=True)[1]]
    log(f"object: {len(use)} keyframes see the box ({np.round(size * 100, 1)} cm), voxel {voxel*1000:.1f} mm...")
    frames = ((depths[k], session.load_frame(session.records[k])[0], T_bw @ session.records[k].T_wc) for k in use)
    vol = fuse_keyframes(frames, len(use), K, voxel, (lo, hi), band, progress, 600**3)
    if vol.voxel > voxel * 1.01:
        log(f"box too large for {voxel*1000:.1f} mm: voxel grown to {vol.voxel*1000:.2f} mm")
    mesh = vol.extract_mesh(smooth=smooth)
    if mesh is None or len(mesh.faces) == 0:
        log("object: no surface found")
        return None, None
    mesh.apply_transform(T_box)
    if bottom in ("closed", "open"):
        up = T_box[:3, 2]
        floor = T_box[:3, 3] - up * size[2] / 2
        floor, up = _refine_floor(mesh, floor, up, log)
        mesh, n_cap = cap_bottom(mesh, floor + up * cut, up, close=bottom == "closed")
        mesh = _drop_fragments(mesh)  # bits of surface texture just above the cut
        if bottom == "closed":
            log(f"object: cut closed with {n_cap:,} faces; watertight: {'yes' if mesh.is_watertight else 'no (holes elsewhere)'}")
    if max_faces and len(mesh.faces) > max_faces:
        mesh = _decimate(mesh, max_faces)
    name = _free_mesh_name(session.root, name)
    write_mesh(session.root, mesh, name)
    log(f"object: {len(mesh.faces):,} faces -> {name}.stl ({(84 + 50 * len(mesh.faces)) / 1e6:.1f} MB) / "
        f"{name}.ply ({time.time()-t0:.0f} s)")
    return mesh, name
