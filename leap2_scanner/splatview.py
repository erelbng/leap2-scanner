"""View a trained Gaussian splat (standard 3DGS .ply) in the browser with viser."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .fusion import estimate_up

C0 = 0.28209479177387814  # SH band 0


def load_3dgs_ply(path: Path):
    """-> centers (N,3), covariances (N,3,3), rgb (N,3) in [0,1], opacity (N,1)."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode().splitlines()
    n = int(next(l for l in header if l.startswith("element vertex")).split()[-1])
    names = [l.split()[-1] for l in header if l.startswith("property float")]
    arr = np.frombuffer(data[end:], "<f4", count=n * len(names)).reshape(n, len(names))
    col = {k: arr[:, i] for i, k in enumerate(names)}
    xyz = np.stack([col["x"], col["y"], col["z"]], 1)
    rgb = np.clip(np.stack([col["f_dc_%d" % i] for i in range(3)], 1) * C0 + 0.5, 0, 1)
    opacity = 1 / (1 + np.exp(-col["opacity"]))
    scales = np.exp(np.stack([col["scale_%d" % i] for i in range(3)], 1))
    q = np.stack([col["rot_%d" % i] for i in range(4)], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)
    cov = R @ (scales[:, :, None] ** 2 * np.transpose(R, (0, 2, 1)))
    return xyz, cov, rgb, opacity[:, None]


def run_splat_view(path: Path, host: str = "127.0.0.1", port: int = 8081):
    """`path`: a scan folder (uses splat/point_cloud.ply) or a .ply file."""
    import viser

    path = Path(path)
    ply = path / "splat" / "point_cloud.ply" if path.is_dir() else path
    xyz, cov, rgb, op = load_3dgs_ply(ply)
    server = viser.ViserServer(host=host, port=port)
    server.scene.add_gaussian_splats("/splat", centers=xyz, covariances=cov, rgbs=rgb, opacities=op)

    poses_file = ply.parent / "poses_refined.json"
    if poses_file.exists():  # start from a real scan viewpoint, a bit behind the camera
        poses = np.array(json.loads(poses_file.read_text())["T_wc_left"])
        up, _ = estimate_up(xyz[op[:, 0] > 0.5], list(poses))  # table normal, works for upside-down scans
        server.scene.set_up_direction(tuple(up))
        T = poses[len(poses) // 3]
        fwd = T[:3, 2] - up * (T[:3, 2] @ up)
        fwd /= np.linalg.norm(fwd)
        target = T[:3, 3] + T[:3, 2] * 0.3

        @server.on_client_connect
        def _(client):
            client.camera.up_direction = tuple(up)
            client.camera.position = tuple(target - fwd * 0.4 + up * 0.15)
            client.camera.look_at = tuple(target)
    else:
        server.scene.set_up_direction("-y")

    print("%d gaussians - open http://localhost:%d" % (len(xyz), port), flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
