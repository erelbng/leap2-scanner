"""Browse a saved scan in the browser: point cloud, mesh, keyframe cameras and images."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import viser
import viser.transforms as vtf

from .export import Session, read_ply_points
from .fusion import estimate_up
from .stereo import colorize_depth


def run_review(args):
    root = Path(args.session)
    s = Session.load(root)
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = False
    print(f"\nReviewing {root}: http://localhost:{args.port}\n", flush=True)

    pts, gray = read_ply_points(root / "pointcloud.ply")
    up, flipped = estimate_up(pts, [r.T_wc for r in s.records])  # works for upside-down scans too
    server.scene.set_up_direction(tuple(up))
    if len(pts) > args.max_points:  # keep the browser responsive on big scans
        sel = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, gray = pts[sel], gray[sel]
    g = np.clip(gray, 0, 255).astype(np.uint8)
    cloud = server.scene.add_point_cloud("/map", pts.astype(np.float32), np.stack([g, g, g], 1), point_size=0.0015)

    mesh_path = root / "mesh.ply"
    mesh_h = None

    P = np.array([r.T_wc[:3, 3] for r in s.records])
    if len(P) > 1:
        server.scene.add_line_segments("/traj", np.stack([P[:-1], P[1:]], 1), (255, 120, 0), thickness=0.0015)
    fov = 2 * np.arctan(s.size / 2 / s.K[0, 0])
    every = args.frustum_every or max(1, len(s.records) // 40)  # ~40 cameras, however long the scan
    for i, r in enumerate(s.records[::every]):
        server.scene.add_camera_frustum(f"/kf/{i}", fov=float(fov), aspect=1.0, scale=0.02, color=(255, 160, 60),
                                        thickness=0.001, wxyz=vtf.SO3.from_matrix(r.T_wc[:3, :3]).wxyz,
                                        position=r.T_wc[:3, 3])
    cur = server.scene.add_camera_frustum("/current", fov=float(fov), aspect=1.0, scale=0.05, color=(255, 60, 0),
                                          thickness=0.003)

    gui = server.gui
    with gui.add_folder("Scan"):
        gui.add_markdown(f"**{root.name}**  \n{len(s.records)} keyframes · {len(pts):,} points  \n"
                         f"baseline {s.baseline*1000:.1f} mm · {s.size}px images")
        sl = gui.add_slider("Keyframe", 0, len(s.records) - 1, 1, len(s.records) // 2)
        img = gui.add_image(np.zeros((160, 320, 3), np.uint8), label="IR | depth", format="jpeg")
    with gui.add_folder("Display"):
        cb_map = gui.add_checkbox("Point cloud", True)
        cb_mesh = gui.add_checkbox("Mesh", False, disabled=not mesh_path.exists())
        sl_pt = gui.add_slider("Point size (mm)", 0.5, 6.0, 0.5, 1.5)

    def show(k: int):
        r = s.records[k]
        left, depth = s.load_frame(r)
        vis = np.hstack([cv2.cvtColor(left, cv2.COLOR_GRAY2RGB),
                         cv2.cvtColor(colorize_depth(depth, 0.15, 0.8), cv2.COLOR_BGR2RGB)])
        if flipped:
            vis = np.hstack([cv2.rotate(h, cv2.ROTATE_180) for h in np.split(vis, 2, axis=1)])
        img.image = cv2.resize(vis, (640, 320), interpolation=cv2.INTER_AREA)
        cur.wxyz = vtf.SO3.from_matrix(r.T_wc[:3, :3]).wxyz
        cur.position = r.T_wc[:3, 3]

    sl.on_update(lambda _: show(int(sl.value)))
    cb_map.on_update(lambda _: setattr(cloud, "visible", cb_map.value))
    sl_pt.on_update(lambda _: setattr(cloud, "point_size", sl_pt.value / 1000))

    @cb_mesh.on_update
    def _(_):
        nonlocal mesh_h
        if mesh_h is None and cb_mesh.value:  # meshes are large: only send when asked for
            import trimesh

            mesh_h = server.scene.add_mesh_trimesh("/mesh", trimesh.load(mesh_path))
        elif mesh_h is not None:
            mesh_h.visible = cb_mesh.value

    show(len(s.records) // 2)

    center = np.median(pts, 0)
    radius = float(np.percentile(np.linalg.norm(pts - center, axis=1), 80))  # scene size
    fwd = np.mean([r.T_wc[:3, 2] for r in s.records], 0)
    fwd -= up * (fwd @ up)
    fwd /= np.linalg.norm(fwd)

    @server.on_client_connect
    def _(client):  # overview from the side the scan was made from, a bit above
        client.camera.position = tuple(center - fwd * 1.7 * radius + up * 1.3 * radius)
        client.camera.look_at = tuple(center)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
