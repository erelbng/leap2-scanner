"""Session storage and exports.

Session layout (also a ready-to-train Gaussian-splatting dataset):
    images/kf_00012_L.jpg   rectified left IR image (pinhole, saved as RGB)
    images/kf_00012_R.jpg   rectified right IR image (same intrinsics)
    depth/kf_00012.png      uint16 depth in millimetres (left view)
    keyframes.json          intrinsics, baseline, per-keyframe camera-to-world poses
    calibration.json        factory calibration used
    transforms.json         nerfstudio / instant-ngp format (OpenGL camera axes)
    sparse/0/*.bin|*.txt    COLMAP model (for original 3DGS, gsplat, ...)
    points3D.ply            sparse init cloud for splats
    pointcloud.ply          full fused point cloud
    mesh.ply / mesh.stl     TSDF mesh (STL in millimetres)
"""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class KeyframeRecord:
    idx: int
    stamp: float
    T_wc: np.ndarray  # rectified-left camera -> world (OpenCV axes)
    sharpness: float
    detail: bool = False  # captured in detail mode (close-up pass -> high-resolution detail mesh)


class Session:
    def __init__(self, root: Path, K: np.ndarray, size: int, baseline: float):
        self.root = Path(root)
        self.K = K
        self.size = size
        self.baseline = baseline
        self.records: list[KeyframeRecord] = []
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "depth").mkdir(exist_ok=True)

    @staticmethod
    def name(idx: int, side: str = "") -> str:
        return f"kf_{idx:05d}" + (f"_{side}" if side else "")

    def add(self, rec: KeyframeRecord, left: np.ndarray, right: np.ndarray, depth: np.ndarray):
        q = [cv2.IMWRITE_JPEG_QUALITY, 95]
        cv2.imwrite(str(self.root / "images" / (self.name(rec.idx, "L") + ".jpg")), cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), q)
        cv2.imwrite(str(self.root / "images" / (self.name(rec.idx, "R") + ".jpg")), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR), q)
        dmm = np.clip(np.round(depth * 1000), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(self.root / "depth" / (self.name(rec.idx) + ".png")), dmm)
        self.records.append(rec)

    def save_index(self):
        data = dict(
            K=self.K.tolist(), size=self.size, baseline=self.baseline,
            keyframes=[dict(idx=r.idx, stamp=r.stamp, T_wc=r.T_wc.tolist(), sharpness=r.sharpness, detail=r.detail)
                       for r in self.records],
        )
        tmp = self.root / "keyframes.json.tmp"
        tmp.write_text(json.dumps(data))
        tmp.replace(self.root / "keyframes.json")

    @staticmethod
    def load(root: Path) -> "Session":
        root = Path(root)
        d = json.loads((root / "keyframes.json").read_text())
        s = Session(root, np.array(d["K"]), d["size"], d["baseline"])
        s.records = [KeyframeRecord(k["idx"], k["stamp"], np.array(k["T_wc"]), k["sharpness"], k.get("detail", False))
                     for k in d["keyframes"]]
        return s

    def load_frame(self, rec: KeyframeRecord):
        left = cv2.imread(str(self.root / "images" / (self.name(rec.idx, "L") + ".jpg")), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(self.root / "depth" / (self.name(rec.idx) + ".png")), cv2.IMREAD_UNCHANGED)
        return left, depth.astype(np.float32) / 1000.0


def free_disk_mb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e6


# ---------------------------------------------------------------------- PLY / mesh
def write_ply_points(path: Path, pts: np.ndarray, gray: np.ndarray):
    col = np.clip(gray, 0, 255).astype(np.uint8)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(len(pts), dt)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr["red"] = arr["green"] = arr["blue"] = col
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())


def read_ply_points(path: Path):
    """Read a PLY written by write_ply_points -> (points, gray)."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    n = int(data[:end].split(b"element vertex ")[1].split(b"\n")[0])
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    a = np.frombuffer(data[end:], dt, count=n)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64), a["red"].astype(np.float32)


def write_mesh(root: Path, mesh, name: str = "mesh"):
    mesh.export(root / f"{name}.ply")
    stl = mesh.copy()
    stl.apply_scale(1000.0)  # STL has no units; slicers assume millimetres
    stl.export(root / f"{name}.stl")


# ---------------------------------------------------------------------- splat datasets
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _views(session: Session):
    """Yield (name, T_wc) for every left and right image (right = left shifted along +x)."""
    T_lr = np.eye(4)
    T_lr[0, 3] = session.baseline
    for r in session.records:
        yield session.name(r.idx, "L"), r.T_wc, session.name(r.idx)
        yield session.name(r.idx, "R"), r.T_wc @ T_lr, None


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def write_nerfstudio(session: Session, init_ply: str | None):
    K = session.K
    frames = []
    # No depth_file_path: nerfstudio requires depth for every frame and right views have none.
    for name, T_wc, _ in _views(session):
        frames.append(dict(file_path=f"images/{name}.jpg", transform_matrix=(T_wc @ _CV_TO_GL).tolist()))
    data = dict(camera_model="OPENCV", fl_x=K[0, 0], fl_y=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                w=session.size, h=session.size, k1=0.0, k2=0.0, p1=0.0, p2=0.0, frames=frames)
    if init_ply:
        data["ply_file_path"] = init_ply
    (session.root / "transforms.json").write_text(json.dumps(data, indent=1))


def write_colmap(session: Session, pts: np.ndarray, gray: np.ndarray):
    out = session.root / "sparse" / "0"
    out.mkdir(parents=True, exist_ok=True)
    K = session.K
    params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    views = list(_views(session))
    imgs = []
    for i, (name, T_wc, _) in enumerate(views, start=1):
        T_cw = np.linalg.inv(T_wc)
        imgs.append((i, _rotmat_to_qvec(T_cw[:3, :3]), T_cw[:3, 3], name + ".jpg"))
    col = np.clip(gray, 0, 255).astype(np.uint8)

    # text
    with open(out / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        f.write(f"1 PINHOLE {session.size} {session.size} " + " ".join(f"{p:.6f}" for p in params) + "\n")
    with open(out / "images.txt", "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, q, t, name in imgs:
            f.write(f"{i} {' '.join(f'{v:.9f}' for v in q)} {' '.join(f'{v:.9f}' for v in t)} 1 {name}\n\n")
    with open(out / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for j, (p, c) in enumerate(zip(pts, col), start=1):
            f.write(f"{j} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c} {c} {c} 0\n")

    # binary
    with open(out / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, session.size, session.size))  # model 1 = PINHOLE
        f.write(struct.pack("<4d", *params))
    with open(out / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for i, q, t, name in imgs:
            f.write(struct.pack("<i4d3di", i, *q, *t, 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    with open(out / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for j, (p, c) in enumerate(zip(pts, col), start=1):
            f.write(struct.pack("<Q3d3BdQ", j, *p.astype(float), c, c, c, 0.0, 0))


def export_splat_dataset(session: Session, pts: np.ndarray, gray: np.ndarray, max_init: int = 150_000):
    if len(pts) > max_init:
        sel = np.random.default_rng(0).choice(len(pts), max_init, replace=False)
        pts, gray = pts[sel], gray[sel]
    write_ply_points(session.root / "points3D.ply", pts, gray)
    write_nerfstudio(session, "points3D.ply")
    write_colmap(session, pts, gray)
