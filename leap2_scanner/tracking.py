"""Stereo visual odometry on the rectified-left image.

Each tracked 2D corner is anchored to a 3D world point (from stereo depth at
the moment it was first seen). Per frame: KLT optical flow -> PnP RANSAC ->
LM refinement. When tracking fails, ORB features of recent keyframes are used
to relocalize.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv_se3(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return se3(R.T, -R.T @ t)


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def sample_depth(depth: np.ndarray, pts: np.ndarray, win: int = 2, rel_spread: float = 0.03) -> np.ndarray:
    """Robust depth at sub-pixel points; 0 where the neighbourhood is empty or straddles an edge."""
    h, w = depth.shape
    out = np.zeros(len(pts), np.float32)
    for i, (x, y) in enumerate(np.round(pts).astype(int)):
        if x < win or y < win or x >= w - win or y >= h - win:
            continue
        patch = depth[y - win : y + win + 1, x - win : x + win + 1]
        v = patch[patch > 0]
        if len(v) < (2 * win + 1) ** 2 // 2:
            continue
        m = np.median(v)
        if v.max() - v.min() < rel_spread * m:
            out[i] = m
    return out


@dataclass
class TrackResult:
    ok: bool
    T_wc: np.ndarray
    n_tracks: int = 0
    n_inliers: int = 0
    status: str = ""


class StereoVO:
    def __init__(self, K: np.ndarray, size: int, target_tracks: int = 400, min_inliers: int = 25):
        self.K = K.astype(np.float64)
        self.size = size
        self.target = target_tracks
        self.min_inliers = min_inliers
        self.reproj = max(1.2, 2.0 * size / 800)
        self.min_dist = max(6, size // 70)
        self.orb = cv2.ORB_create(1200, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.reset()

    def reset(self):
        self.T_wc = np.eye(4)
        self.velocity = np.eye(4)
        self.prev: np.ndarray | None = None
        self.p2d = np.empty((0, 2), np.float32)
        self.p3d = np.empty((0, 3), np.float64)
        self.initialized = False
        self.lost = False
        self.reloc_db: list[tuple[np.ndarray, np.ndarray]] = []  # (descriptors, world points)
        self._reloc_cursor = 0

    # ------------------------------------------------------------------ helpers
    def _backproject(self, pts: np.ndarray, z: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
        K = self.K
        x = (pts[:, 0] - K[0, 2]) / K[0, 0] * z
        y = (pts[:, 1] - K[1, 2]) / K[1, 1] * z
        Xc = np.stack([x, y, z], 1)
        return Xc @ T_wc[:3, :3].T + T_wc[:3, 3]

    def _replenish(self, img: np.ndarray, depth: np.ndarray, T_wc: np.ndarray):
        need = self.target - len(self.p2d)
        if need < self.target * 0.25:
            return
        mask = np.where(depth > 0, 255, 0).astype(np.uint8)
        for x, y in self.p2d.astype(int):
            cv2.circle(mask, (int(x), int(y)), self.min_dist, 0, -1)
        c = cv2.goodFeaturesToTrack(img, maxCorners=need, qualityLevel=0.005,
                                    minDistance=self.min_dist, mask=mask, blockSize=7)
        if c is None:
            return
        c = c.reshape(-1, 2).astype(np.float32)
        z = sample_depth(depth, c)
        keep = z > 0
        if keep.sum() == 0:
            return
        Xw = self._backproject(c[keep].astype(np.float64), z[keep].astype(np.float64), T_wc)
        self.p2d = np.vstack([self.p2d, c[keep]])
        self.p3d = np.vstack([self.p3d, Xw])

    def _pnp(self, X: np.ndarray, x: np.ndarray, T_wc_guess: np.ndarray):
        if len(X) < self.min_inliers:
            return None, None
        T_cw = inv_se3(T_wc_guess)
        rvec = cv2.Rodrigues(T_cw[:3, :3])[0]
        tvec = T_cw[:3, 3].reshape(3, 1).copy()
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            X.astype(np.float64), x.astype(np.float64), self.K, None, rvec, tvec,
            useExtrinsicGuess=True, iterationsCount=150, reprojectionError=self.reproj,
            confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inl is None or len(inl) < self.min_inliers:
            return None, None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(X[inl].astype(np.float64), x[inl].astype(np.float64),
                                          self.K, None, rvec, tvec)
        return inv_se3(se3(cv2.Rodrigues(rvec)[0], tvec)), inl

    # ------------------------------------------------------------------ API
    def add_keyframe(self, img: np.ndarray, depth: np.ndarray, T_wc: np.ndarray):
        """Store ORB features of a keyframe for relocalization."""
        kps, des = self.orb.detectAndCompute(img, (depth > 0).astype(np.uint8) * 255)
        if des is None or len(kps) < 30:
            return
        pts = np.float32([k.pt for k in kps])
        z = sample_depth(depth, pts)
        keep = z > 0
        if keep.sum() < 30:
            return
        Xw = self._backproject(pts[keep].astype(np.float64), z[keep].astype(np.float64), T_wc)
        self.reloc_db.append((des[keep], Xw))

    def _relocalize(self, img: np.ndarray, depth: np.ndarray) -> TrackResult:
        if not self.reloc_db:
            return TrackResult(False, self.T_wc, status="lost (no keyframes)")
        kps, des = self.orb.detectAndCompute(img, None)
        if des is None or len(kps) < 30:
            return TrackResult(False, self.T_wc, status="lost (no features)")
        pts = np.float32([k.pt for k in kps])
        # newest keyframes first, then sweep older ones a few per frame
        n = len(self.reloc_db)
        cands = list(range(n - 1, max(-1, n - 6), -1))
        for _ in range(3):
            self._reloc_cursor = (self._reloc_cursor + 1) % n
            cands.append(self._reloc_cursor)
        best = None
        for ci in dict.fromkeys(cands):
            kdes, kX = self.reloc_db[ci]
            m = self.matcher.knnMatch(des, kdes, k=2)
            good = [a for a, *b in m if b and a.distance < 0.8 * b[0].distance]
            if len(good) < self.min_inliers:
                continue
            X = kX[[g.trainIdx for g in good]]
            x = pts[[g.queryIdx for g in good]]
            T, inl = self._pnp(X, x, self.T_wc)
            if T is not None and len(inl) >= 40 and (best is None or len(inl) > best[1]):
                best = (T, len(inl))
        if best is None:
            return TrackResult(False, self.T_wc, status="lost - move back to a scanned area")
        self.T_wc = best[0]
        self.velocity = np.eye(4)
        self.lost = False
        self.prev = img
        self.p2d = np.empty((0, 2), np.float32)
        self.p3d = np.empty((0, 3), np.float64)
        self._replenish(img, depth, self.T_wc)
        return TrackResult(True, self.T_wc.copy(), len(self.p2d), best[1], "relocalized")

    def track(self, img: np.ndarray, depth: np.ndarray) -> TrackResult:
        """img: rectified-left (contrast normalised) uint8, depth: metres."""
        if not self.initialized:
            self.T_wc = np.eye(4)
            self.prev = img
            self._replenish(img, depth, self.T_wc)
            if len(self.p2d) < self.min_inliers * 2:
                self.p2d = np.empty((0, 2), np.float32)
                self.p3d = np.empty((0, 3), np.float64)
                return TrackResult(False, self.T_wc, status="init: need more texture/depth")
            self.initialized = True
            return TrackResult(True, self.T_wc.copy(), len(self.p2d), len(self.p2d), "initialized")
        if self.lost:
            return self._relocalize(img, depth)

        if len(self.p2d) == 0:
            self.lost = True
            return TrackResult(False, self.T_wc, status="lost")
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev, img, self.p2d.reshape(-1, 1, 2), None, **self.lk)
        p0, st0, _ = cv2.calcOpticalFlowPyrLK(img, self.prev, p1, None, **self.lk)
        p1 = p1.reshape(-1, 2)
        fb = np.linalg.norm(p0.reshape(-1, 2) - self.p2d, axis=1)
        h, w = img.shape
        good = (st1.ravel() == 1) & (st0.ravel() == 1) & (fb < 1.0)
        good &= (p1[:, 0] >= 0) & (p1[:, 1] >= 0) & (p1[:, 0] < w - 1) & (p1[:, 1] < h - 1)
        X, x = self.p3d[good], p1[good]

        T_pred = self.T_wc @ self.velocity
        T_new, inl = self._pnp(X, x, T_pred)
        if T_new is None:
            self.lost = True
            return TrackResult(False, self.T_wc, int(good.sum()), 0, "lost (too few inliers)")
        delta = inv_se3(self.T_wc) @ T_new
        if np.linalg.norm(delta[:3, 3]) > 0.12 or rot_angle_deg(delta[:3, :3]) > 30:
            self.lost = True
            return TrackResult(False, self.T_wc, int(good.sum()), len(inl), "lost (jump rejected)")
        self.velocity = delta
        self.T_wc = T_new
        self.p2d = x[inl].astype(np.float32)
        self.p3d = X[inl]
        self.prev = img
        n_inl = len(inl)
        self._replenish(img, depth, self.T_wc)
        return TrackResult(True, self.T_wc.copy(), len(self.p2d), n_inl, "tracking")
