"""Dense depth from the rectified Leap 2 pair (OpenCV SGBM)."""

from __future__ import annotations

import cv2
import numpy as np

from .calib import Rectifier


class DepthEstimator:
    """SGBM on the rectified pair.

    mode "half" (default): match on 2x downsampled images. In dark, noisy IR the averaging makes
      matching far more reliable (about 5x more multi-view confirmed depth than full resolution
      on a desk scan) and it is ~3x faster.
    mode "full": full resolution - finer and sharper at edges, but sparse unless well lit.
    mode "hybrid": full resolution where it finds a match, half resolution fills the rest.
      Used by detail mode (close range, where the IR LEDs light things well).

    `fill` (off by default, needs opencv-contrib): pixels still without depth get it from a
    WLS-filtered (edge-aware, left-right checked) version of the half-resolution match. Matched
    pixels are never changed. In an A/B desk scan this gave ~50% more mesh surface (fewer holes)
    but rougher filled areas, slightly lower multi-view agreement, no better splats, and costs
    ~15 ms per frame. (The WLS result on its own smears small structures like keys into streaks.)
    Can be switched at any time (`fill` attribute) when `can_fill`."""

    def __init__(self, rect: Rectifier, z_min: float = 0.12, z_max: float = 0.9, block: int = 5,
                 mode: str = "half", fill: bool = False):
        self.rect = rect
        self.z_min = z_min
        self.z_max = z_max
        self.mode = mode
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._sgbm = {scale: self._make_sgbm(scale, block) for scale in (1, 2)}
        self.can_fill = hasattr(cv2, "ximgproc")
        self.fill = fill and self.can_fill
        if self.can_fill:
            # separate matcher copies: createDisparityWLSFilter reconfigures the matcher it gets
            # (no left-right check, no speckle filter), which must not touch our own matching
            self._right = cv2.ximgproc.createRightMatcher(self._make_sgbm(2, block))
            self._wls = cv2.ximgproc.createDisparityWLSFilter(self._make_sgbm(2, block))
            self._wls.setLambda(8000.0)
            self._wls.setSigmaColor(1.5)

    def _make_sgbm(self, scale: int, block: int):
        fb = self.rect.f * self.rect.baseline / scale
        nd = int(np.clip(int(np.ceil(fb / self.z_min / 16.0)) * 16, 32, 256))  # range for the closest depth
        return cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=nd,
            blockSize=block,
            P1=8 * block * block,
            P2=(96 if scale == 2 else 32) * block * block,  # smoother at half res: fills dark areas
            disp12MaxDiff=1,
            preFilterCap=31,
            uniquenessRatio=12,
            speckleWindowSize=120,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def normalize(self, img: np.ndarray) -> np.ndarray:
        return self.clahe.apply(img)

    def _disparity(self, l: np.ndarray, r: np.ndarray, scale: int):
        """Disparity in full-resolution pixels. Scale 2 also returns the WLS-filled version (or None)."""
        if scale == 1:
            return self._sgbm[1].compute(l, r).astype(np.float32) / 16.0
        lh, rh = cv2.pyrDown(l), cv2.pyrDown(r)
        raw = self._sgbm[2].compute(lh, rh)

        def up(d):  # /16 fixed point, *2 for the half resolution
            return cv2.resize(d.astype(np.float32) / 8.0, (l.shape[1], l.shape[0]), interpolation=cv2.INTER_NEAREST)

        if not self.fill:
            return up(raw), None
        return up(raw), up(self._wls.filter(raw, lh, disparity_map_right=self._right.compute(rh, lh)))

    def _to_depth(self, disp: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        valid = disp > 0.5
        # Reject flying pixels at depth discontinuities.
        dmax = cv2.dilate(disp, np.ones((3, 3), np.uint8))
        dmin = cv2.erode(disp, np.ones((3, 3), np.uint8))
        valid &= (dmax - dmin) < np.maximum(1.0, 0.04 * disp)
        # Reject saturated / unlit / out-of-lens-circle pixels.
        valid &= (left > 6) & (left < 250) & (right > 6) & (right < 250)
        z = np.zeros_like(disp)
        z[valid] = (self.rect.f * self.rect.baseline) / disp[valid]
        z[(z < self.z_min) | (z > self.z_max)] = 0
        return z

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return metric depth Z (float32, metres, 0 = invalid) in rectified-left frame."""
        l = self.normalize(left)
        r = self.normalize(right)
        filled = None
        if self.mode != "full" or self.fill:
            half, filled = self._disparity(l, r, 2)
        if self.mode == "full":
            z = self._to_depth(self._disparity(l, r, 1), left, right)
        elif self.mode == "half":
            z = self._to_depth(half, left, right)
        else:
            z_full = self._to_depth(self._disparity(l, r, 1), left, right)
            z = np.where(z_full > 0, z_full, self._to_depth(half, left, right))
        if filled is not None:
            z = np.where(z > 0, z, self._to_depth(filled, left, right))
        return z

    def backproject(self, depth: np.ndarray, stride: int = 2):
        """Depth map -> (N,3) points in camera frame and their pixel coords."""
        d = depth[::stride, ::stride]
        v, u = np.nonzero(d > 0)
        z = d[v, u]
        u = u * stride
        v = v * stride
        K = self.rect.K
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        return np.stack([x, y, z], 1).astype(np.float32), u, v


def colorize_depth(depth: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    d = np.clip((depth - z_min) / max(z_max - z_min, 1e-6), 0, 1)
    vis = cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[depth <= 0] = 0
    return vis
