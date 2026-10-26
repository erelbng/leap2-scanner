"""Factory calibration of the Leap 2 and stereo rectification.

Blob layout (version 3, 156 bytes, little endian), found by inspection:
    0   'C','A', version(u8), 0
    4   u32  (id / flags)
    8   2 x f32  (unused, zero)
    16  camera 0 (left):  fx, 0, 0, fy, cx, cy, d0..d4, rot(3), trans(3)   17 x f32
    84  camera 1 (right): same layout; rot = Rodrigues vector, trans in metres,
        mapping left-camera coordinates to right-camera coordinates:
        X_right = R @ X_left + t
    152 u32 checksum
Intrinsics refer to the 512x512 stream. The lenses are ~160 degree, nearly ideal
equidistant fisheyes; d0..d4 are a forward radial polynomial in the incidence angle
(model "poly": r = theta + d0*theta^2 + ... + d4*theta^6). Verified on ~440k stereo matches
from real scans: with the relative rotation refined (the factory one is ~0.2-0.3 deg off),
"poly" aligns the rectified rows best (0.47 px median vs 0.50 equidistant, inverse readings
worse). `leap2-scanner calibrate` refines the extrinsics from live data or saved scans.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .device import device_serial, parse_blob_floats, read_calibration_blob

NATIVE_RES = 512


@dataclass
class FisheyeCam:
    fx: float
    fy: float
    cx: float
    cy: float
    d: np.ndarray  # 5 radial terms (meaning depends on model)
    model: str = "equidistant"

    def angle(self, rd: np.ndarray) -> np.ndarray:
        """Incidence angle for 'inv_*' models, which map normalized radius -> angle."""
        d, r = self.d, rd
        if self.model == "inv_poly":
            return r * (1 + r * (d[0] + r * (d[1] + r * (d[2] + r * (d[3] + r * d[4])))))
        if self.model == "inv_kb5":
            r2 = r * r
            return r * (1 + r2 * (d[0] + r2 * (d[1] + r2 * (d[2] + r2 * (d[3] + r2 * d[4])))))
        if self.model == "inv_kb3last":
            r2 = r * r
            return r * (1 + r2 * (d[2] + r2 * (d[3] + r2 * d[4])))
        raise ValueError(self.model)

    def distort(self, theta: np.ndarray) -> np.ndarray:
        """Distorted normalized radius as function of incidence angle."""
        d, t = self.d, theta
        if self.model.startswith("inv_"):
            r = t.copy()
            for _ in range(15):  # Newton on angle(r) = theta
                e = self.angle(r) - t
                de = (self.angle(r + 1e-6) - self.angle(r)) / 1e-6
                r = r - e / de
            return r
        if self.model == "kb5":
            t2 = t * t
            return t * (1 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * (d[3] + t2 * d[4])))))
        if self.model == "kb3last":
            t2 = t * t
            return t * (1 + t2 * (d[2] + t2 * (d[3] + t2 * d[4])))
        if self.model == "poly":
            return t * (1 + t * (d[0] + t * (d[1] + t * (d[2] + t * (d[3] + t * d[4])))))
        if self.model == "equidistant":
            return t
        raise ValueError(self.model)

    def project(self, rays: np.ndarray) -> np.ndarray:
        """Project 3D rays (N,3) in camera frame to pixels (N,2)."""
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        rd = self.distort(theta)
        s = np.where(r > 1e-12, rd / np.maximum(r, 1e-12), 0.0)
        return np.stack([self.fx * x * s + self.cx, self.fy * y * s + self.cy], -1)

    def unproject(self, px: np.ndarray) -> np.ndarray:
        """Pixels (N,2) to unit rays (N,3)."""
        x = (px[..., 0] - self.cx) / self.fx
        y = (px[..., 1] - self.cy) / self.fy
        rd = np.hypot(x, y)
        if self.model.startswith("inv_"):
            th = np.clip(self.angle(rd), 0, np.pi * 0.95)
            s = np.where(rd > 1e-12, np.sin(th) / np.maximum(rd, 1e-12), 1.0)
            return np.stack([x * s, y * s, np.cos(th)], -1)
        th = rd.copy()
        for _ in range(15):  # Newton on distort(theta) = rd
            e = self.distort(th) - rd
            de = (self.distort(th + 1e-6) - self.distort(th)) / 1e-6
            th = np.clip(th - e / de, 0, np.pi * 0.95)
        s = np.where(rd > 1e-12, np.sin(th) / np.maximum(rd, 1e-12), 1.0)
        return np.stack([x * s, y * s, np.cos(th)], -1)

    def scaled(self, k: float) -> "FisheyeCam":
        # pixel centres: c' = k * (c + 0.5) - 0.5
        return FisheyeCam(self.fx * k, self.fy * k, k * (self.cx + 0.5) - 0.5,
                          k * (self.cy + 0.5) - 0.5, self.d.copy(), self.model)


@dataclass
class StereoCalib:
    left: FisheyeCam
    right: FisheyeCam
    R: np.ndarray  # left -> right rotation
    t: np.ndarray  # left -> right translation (metres)
    res: int = NATIVE_RES
    source: str = "factory"

    @staticmethod
    def from_blob(blob: bytes, model: str = "poly") -> "StereoCalib":
        fl = parse_blob_floats(blob)
        cams = []
        for o in (2, 19):
            c = fl[o : o + 17]
            cams.append(FisheyeCam(c[0], c[3], c[4], c[5], c[6:11].copy(), model))
        r = fl[19 + 11 : 19 + 14]
        t = fl[19 + 14 : 19 + 17]
        R = cv2.Rodrigues(r)[0]
        return StereoCalib(cams[0], cams[1], R, t.copy())

    def for_resolution(self, res: int) -> "StereoCalib":
        k = res / self.res
        return StereoCalib(self.left.scaled(k), self.right.scaled(k), self.R.copy(), self.t.copy(), res, self.source)

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.t))

    def to_json(self) -> dict:
        cam = lambda c: dict(fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy, d=c.d.tolist(), model=c.model)
        return dict(left=cam(self.left), right=cam(self.right), R=self.R.tolist(), t=self.t.tolist(),
                    res=self.res, source=self.source)

    @staticmethod
    def from_json(d: dict) -> "StereoCalib":
        cam = lambda c: FisheyeCam(c["fx"], c["fy"], c["cx"], c["cy"], np.array(c["d"]), c["model"])
        return StereoCalib(cam(d["left"]), cam(d["right"]), np.array(d["R"]), np.array(d["t"]),
                           d["res"], d.get("source", "file"))

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @staticmethod
    def load(path: Path) -> "StereoCalib":
        return StereoCalib.from_json(json.loads(Path(path).read_text()))


@dataclass
class Rectifier:
    """Rectified pinhole stereo pair; x axis runs along the baseline.

    The rectified-left camera frame is the reference "camera frame" of the
    whole scanner (tracking, fusion, export all use it).
    """

    calib: StereoCalib
    fov_deg: float = 100.0
    size: int = 0  # rectified image width = height (0 = auto)
    f: float = 0.0
    R_rect: np.ndarray = field(default=None)  # left cam -> rectified frame
    maps: list = field(default_factory=list)

    def __post_init__(self):
        c = self.calib
        C = -c.R.T @ c.t  # right camera centre in left frame
        ex = C / np.linalg.norm(C)
        z = np.array([0, 0, 1.0]) + c.R.T @ np.array([0, 0, 1.0])
        ez = z - ex * (z @ ex)
        ez /= np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        self.R_rect = np.stack([ex, ey, ez])
        f_raw = 0.5 * (c.left.fx + c.right.fx)
        half = np.tan(np.radians(self.fov_deg / 2))
        if not self.size:
            self.size = int(round(2 * half * f_raw / 8)) * 8
        self.f = self.size / (2 * half)
        self.cx = self.cy = (self.size - 1) / 2
        self.K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]])
        self.baseline = c.baseline
        u, v = np.meshgrid(np.arange(self.size, dtype=np.float64), np.arange(self.size, dtype=np.float64))
        rays = np.stack([(u - self.cx) / self.f, (v - self.cy) / self.f, np.ones_like(u)], -1)
        rays_l = rays @ self.R_rect  # == (R_rect.T @ ray) per pixel
        rays_r = rays_l @ c.R.T
        self.maps = []
        self.float_maps = []  # rectified pixel -> raw pixel, for mapping matches back
        for cam, rr in ((c.left, rays_l), (c.right, rays_r)):
            p = cam.project(rr.reshape(-1, 3)).reshape(self.size, self.size, 2).astype(np.float32)
            self.float_maps.append(p)
            self.maps.append(cv2.convertMaps(p[..., 0], p[..., 1], cv2.CV_16SC2))

    def rectify(self, left: np.ndarray, right: np.ndarray):
        out = []
        for img, (m1, m2) in zip((left, right), self.maps):
            out.append(cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        return out[0], out[1]


_CONFIG_BASE = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
CONFIG_DIR = _CONFIG_BASE / "leap2-scanner"
if not CONFIG_DIR.exists() and (_CONFIG_BASE / "leapscan").exists():
    CONFIG_DIR = _CONFIG_BASE / "leapscan"  # calibration from before the project was renamed


def load_device_calibration(dev: str, model: str = "poly", use_refined: bool = True) -> StereoCalib:
    """Factory calibration from the device (cached on disk), or a refined one if present."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    serial = device_serial(dev)
    refined = CONFIG_DIR / f"{serial}_refined.json"
    if use_refined and refined.exists():
        return StereoCalib.load(refined)
    cache = CONFIG_DIR / f"{serial}_factory.bin"
    try:
        blob = read_calibration_blob(dev)
        if not cache.exists():
            cache.write_bytes(blob)
    except OSError:
        if not cache.exists():
            raise
        blob = cache.read_bytes()
    return StereoCalib.from_blob(blob, model)
