"""Loop closure: remove drift from keyframe poses after a scan.

Odometry drifts a few cm over a long scan, so when you come back to an object it is fused
again at a shifted place (surfaces get thick, edges appear two or three times). This module
fixes the keyframe poses afterwards:

  1. SIFT features + stereo depth per keyframe,
  2. co-visibility: which keyframes really look at the same geometry (also across revisits),
  3. *guided* matching: a feature is only searched near where the current poses predict it.
     Plain matching fails between visits because the IR LEDs move with the camera, so the
     lighting changes; the position prior makes matching reliable anyway,
  4. one Gauss-Newton problem over all poses: feature reprojection, dense point-to-plane
     distances between the depth maps of co-visible keyframes, and weak odometry priors.
     Features alone leave the surfaces of revisits a few mm apart; depth alone lets keyframes
     slide along flat surfaces. Then re-match with the better poses and repeat.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------- SE(3) helpers
def _exp(x: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(x[:3], np.float64))[0]
    T[:3, 3] = x[3:]
    return T


def _inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _project(X: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.maximum(X[:, 2:3], 1e-6)
    return X[:, :2] / z * [K[0, 0], K[1, 1]] + K[:2, 2]


# ---------------------------------------------------------------------- features
@dataclass
class KFFeatures:
    pts: np.ndarray  # (N,2) pixels
    des: np.ndarray  # (N,128)
    X: np.ndarray  # (N,3) camera-frame points, nan where no reliable depth
    tree: cKDTree


def _depth_at(depth: np.ndarray, pts: np.ndarray, win: int = 2) -> np.ndarray:
    h, w = depth.shape
    out = np.full(len(pts), np.nan)
    for k, (x, y) in enumerate(np.round(pts).astype(int)):
        if win <= x < w - win and win <= y < h - win:
            v = depth[y - win : y + win + 1, x - win : x + win + 1]
            v = v[v > 0]
            if len(v) >= 5 and v.max() - v.min() < 0.03 * np.median(v):
                out[k] = np.median(v)
    return out


def extract(images, depths, K: np.ndarray) -> list[KFFeatures]:
    sift = cv2.SIFT_create(3000, contrastThreshold=0.02)
    feats = []
    for img, depth in zip(images, depths):
        kps, des = sift.detectAndCompute(cv2.GaussianBlur(img, (0, 0), 1.0), None)  # blur: IR sensor noise
        pts = np.array([k.pt for k in kps], np.float64).reshape(-1, 2)
        des = des if des is not None else np.empty((0, 128), np.float32)
        z = _depth_at(depth, pts)
        X = np.stack([(pts[:, 0] - K[0, 2]) / K[0, 0] * z, (pts[:, 1] - K[1, 2]) / K[1, 1] * z, z], 1)
        feats.append(KFFeatures(pts, des, X, cKDTree(pts) if len(pts) else None))
    return feats


# ---------------------------------------------------------------------- co-visibility + matching
def covisible_pairs(depths, poses, K, size: int, seq: int = 3, per_kf: int = 10, tol: float = 0.08):
    """Pairs of keyframes that see the same surfaces (depth-consistent reprojection)."""
    n = len(poses)
    samples = []
    for d, T in zip(depths, poses):
        dd = d[::12, ::12]
        v, u = np.nonzero(dd > 0)
        z = dd[v, u]
        X = np.stack([(u * 12 - K[0, 2]) * z / K[0, 0], (v * 12 - K[1, 2]) * z / K[1, 1], z], 1)
        samples.append(X @ T[:3, :3].T + T[:3, 3])
    score = np.zeros((n, n))
    for j in range(n):
        Tj = _inv(poses[j])
        for i in range(n):
            if i == j or len(samples[i]) == 0:
                continue
            Y = samples[i] @ Tj[:3, :3].T + Tj[:3, 3]
            ok = Y[:, 2] > 0.05
            uv = _project(Y[ok], K).astype(int)
            ins = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < size) & (uv[:, 1] < size)
            if not ins.any():
                continue
            dz = depths[j][uv[ins, 1], uv[ins, 0]]
            zz = Y[ok][ins, 2]
            score[i, j] = np.mean((dz > 0) & (np.abs(dz - zz) < tol * zz)) * ins.mean()
    S = np.minimum(score, score.T)
    pairs = set()
    for i in range(n):
        for j in range(i + 1, min(n, i + seq + 1)):
            pairs.add((i, j))
        cand = [j for j in np.argsort(-S[i]) if abs(int(j) - i) > seq and S[i, j] > 0.15][:per_kf]
        pairs.update((min(i, int(j)), max(i, int(j))) for j in cand)
    return sorted(pairs)


@dataclass
class Match:
    i: int
    j: int
    xi: np.ndarray
    xj: np.ndarray
    Xi: np.ndarray  # nan rows where no depth in i
    Xj: np.ndarray


def _guided(fa: KFFeatures, fb: KFFeatures, T_ba: np.ndarray, K, radius: float, ratio: float = 0.85):
    """Features of a with depth, predicted in b; best descriptor match inside the radius."""
    ia = np.nonzero(~np.isnan(fa.X[:, 2]))[0]
    if len(ia) == 0 or fb.tree is None:
        return np.empty(0, int), np.empty(0, int)
    Y = fa.X[ia] @ T_ba[:3, :3].T + T_ba[:3, 3]
    front = Y[:, 2] > 0.05
    ia, uv = ia[front], _project(Y[front], K)
    qa, qb = [], []
    for a, cand in zip(ia, fb.tree.query_ball_point(uv, radius)):
        if not cand:
            continue
        d = np.linalg.norm(fb.des[cand] - fa.des[a], axis=1)
        o = np.argsort(d)
        if len(o) == 1 or d[o[0]] < ratio * d[o[1]]:
            qa.append(a)
            qb.append(cand[o[0]])
    return np.array(qa, int), np.array(qb, int)


def match_pair(fi: KFFeatures, fj: KFFeatures, Ti, Tj, K, radius: float, reproj: float = 3.0, min_inliers: int = 15):
    T_ji = _inv(Tj) @ Ti
    a1, b1 = _guided(fi, fj, T_ji, K, radius)  # i features with depth -> j
    b2, a2 = _guided(fj, fi, _inv(T_ji), K, radius)  # j features with depth -> i
    qi = np.r_[a1, a2]
    qj = np.r_[b1, b2]
    if len(qi) < min_inliers:
        return None
    pair = np.unique(np.stack([qi, qj], 1), axis=0)
    qi, qj = pair[:, 0], pair[:, 1]
    # verify with PnP (3D in i -> 2D in j) and keep matches consistent with that relative pose
    hi = ~np.isnan(fi.X[qi, 2])
    if hi.sum() < min_inliers:
        return None
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(fi.X[qi[hi]], fj.pts[qj[hi]], K, None, iterationsCount=200,
                                                 reprojectionError=reproj, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return None
    if not ok or inl is None or len(inl) < min_inliers:
        return None
    R = cv2.Rodrigues(rvec)[0]
    t = tvec.ravel()
    keep = np.zeros(len(qi), bool)
    keep[np.nonzero(hi)[0][inl.ravel()]] = True
    hj = ~np.isnan(fj.X[qj, 2]) & ~keep
    if hj.any():
        pi = _project((fj.X[qj[hj]] - t) @ R, K)
        keep[np.nonzero(hj)[0][np.linalg.norm(pi - fi.pts[qi[hj]], axis=1) < reproj]] = True
    qi, qj = qi[keep], qj[keep]
    return Match(-1, -1, fi.pts[qi], fj.pts[qj], fi.X[qi], fj.X[qj])


# ---------------------------------------------------------------------- joint pose refinement
def _skew(a: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,3,3) cross-product matrices."""
    z = np.zeros(len(a))
    return np.stack([np.stack([z, -a[:, 2], a[:, 1]], 1), np.stack([a[:, 2], z, -a[:, 0]], 1),
                     np.stack([-a[:, 1], a[:, 0], z], 1)], 1)


def _point_normal_maps(depth: np.ndarray, K: np.ndarray, step: int = 3):
    """Camera-frame points and normals per pixel (nan where invalid or at a depth discontinuity)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    X = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    X[z <= 0] = np.nan
    s = step
    dx = np.full_like(X, np.nan)
    dy = np.full_like(X, np.nan)
    dx[:, s:-s] = X[:, 2 * s:] - X[:, : -2 * s]
    dy[s:-s] = X[2 * s:] - X[: -2 * s]
    n = np.cross(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        n[np.sum(n * X, -1) > 0] *= -1  # face the camera
        jump = (np.abs(dx[..., 2]) > 0.08 * z + 0.01) | (np.abs(dy[..., 2]) > 0.08 * z + 0.01)
    n[jump] = np.nan
    return X.astype(np.float32), n.astype(np.float32)


def _add(H, g, a: int, b: int, Ja, Jb, w, r):
    """Accumulate residuals r (M,) with Jacobians Ja, Jb (M,6) w.r.t. poses a, b into H, g."""
    A, B = 6 * a, 6 * b
    WJa, WJb = Ja * w[:, None], Jb * w[:, None]
    H[A : A + 6, A : A + 6] += WJa.T @ Ja
    H[B : B + 6, B : B + 6] += WJb.T @ Jb
    Hab = WJa.T @ Jb
    H[A : A + 6, B : B + 6] += Hab
    H[B : B + 6, A : A + 6] += Hab.T
    g[A : A + 6] += WJa.T @ r
    g[B : B + 6] += WJb.T @ r


class _Dense:
    """Point-to-plane terms between co-visible keyframes (depth maps at half resolution)."""

    def __init__(self, depths, K, fb: float, samples: int = 1500, sigma_px: float = 0.5, seed: int = 0):
        self.K = K.copy()
        self.K[:2] /= 2.0  # half resolution: matches the stereo noise, quarter of the memory
        self.maps = [_point_normal_maps(d[::2, ::2], self.K) for d in depths]
        self.shape = depths[0][::2, ::2].shape
        self.src = []
        for X, N in self.maps:
            Xs, Ns = X[::2, ::2].reshape(-1, 3), N[::2, ::2].reshape(-1, 3)
            ok = np.isfinite(Xs).all(1) & np.isfinite(Ns).all(1)
            self.src.append((Xs[ok].astype(np.float64), Ns[ok].astype(np.float64)))
        self.fb, self.samples, self.sigma_px = fb, samples, sigma_px
        self.rng = np.random.default_rng(seed)

    def add(self, H, g, poses, i: int, j: int, gate: float, weight: float):
        """Points of i against the surface of j. Returns normalised residuals."""
        Xi, Ni = self.src[i]
        if len(Xi) == 0:
            return None
        sel = self.rng.choice(len(Xi), min(self.samples, len(Xi)), replace=False)
        p, n_i = Xi[sel], Ni[sel]
        Ti, Tj = poses[i], poses[j]
        Tji = _inv(Tj)
        P = p @ Ti[:3, :3].T + Ti[:3, 3]
        Y = P @ Tji[:3, :3].T + Tji[:3, 3]
        h, w = self.shape
        ok = Y[:, 2] > 0.05
        uv = np.zeros((len(Y), 2), int)
        uv[ok] = np.round(_project(Y[ok], self.K)).astype(int)
        ok &= (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < w) & (uv[:, 1] < h)
        if ok.sum() < 30:
            return None
        Xj, Nj = self.maps[j]
        q = Xj[uv[ok, 1], uv[ok, 0]].astype(np.float64)
        nq = Nj[uv[ok, 1], uv[ok, 0]].astype(np.float64)
        with np.errstate(invalid="ignore"):
            good = np.isfinite(q).all(1) & np.isfinite(nq).all(1)
            good &= np.sum((n_i[ok] @ (Tji[:3, :3] @ Ti[:3, :3]).T) * nq, 1) > 0.7
            good &= np.abs(np.sum(nq * (Y[ok] - q), 1)) < gate
        if good.sum() < 30:
            return None
        Q = q[good] @ Tj[:3, :3].T + Tj[:3, 3]
        nw = nq[good] @ Tj[:3, :3].T
        Pg = P[ok][good]
        r = np.sum(nw * (Pg - Q), 1)
        # stereo depth noise grows with z^2 (constant disparity noise)
        sig = np.hypot(p[ok][good][:, 2] ** 2, q[good][:, 2] ** 2) * self.sigma_px / self.fb
        wr = weight * np.minimum(1.0, 2.0 * sig / np.maximum(np.abs(r), 1e-12)) / sig**2  # Huber at 2 sigma
        Ji = np.hstack([np.cross(Pg, nw), nw])
        Jj = np.hstack([np.cross(nw, Pg - Q) - np.cross(Q, nw), -nw])
        _add(H, g, i, j, Ji, Jj, wr, r)
        return np.abs(r) / sig


def _add_features(H, g, poses, m: Match, K, sigma: float):
    """Reprojection terms of one feature link (3D point from depth in a, observed pixel in b)."""
    errs = []
    for a, b, Xa, xb in ((m.i, m.j, m.Xi, m.xj), (m.j, m.i, m.Xj, m.xi)):
        h = ~np.isnan(Xa[:, 2])
        if not h.any():
            continue
        Ta, Rb = poses[a], poses[b][:3, :3]
        P = Xa[h] @ Ta[:3, :3].T + Ta[:3, 3]
        Y = (P - poses[b][:3, 3]) @ Rb
        z = np.maximum(Y[:, 2], 1e-6)
        res = _project(Y, K) - xb[h]
        Jp = np.zeros((len(Y), 2, 3))
        Jp[:, 0, 0] = K[0, 0] / z
        Jp[:, 0, 2] = -K[0, 0] * Y[:, 0] / z**2
        Jp[:, 1, 1] = K[1, 1] / z
        Jp[:, 1, 2] = -K[1, 1] * Y[:, 1] / z**2
        dP = np.concatenate([-_skew(P), np.broadcast_to(np.eye(3), (len(P), 3, 3))], 2)  # world point / pose a
        Ja = (Jp @ Rb.T @ dP).reshape(-1, 6)  # pose b moves the point the opposite way: Jb = -Ja
        e = np.linalg.norm(res, axis=1)
        wr = np.repeat(np.minimum(1.0, 2.0 * sigma / np.maximum(e, 1e-12)), 2) / sigma**2
        _add(H, g, a, b, Ja, -Ja, wr, res.ravel())
        errs.append(e)
    return errs


def _add_odometry(H, g, poses, k: int, vp, vp_next, sigma: float):
    """Weak prior on the odometry's relative pose k -> k+1, as virtual points in front of the camera."""
    A = vp @ poses[k][:3, :3].T + poses[k][:3, 3]
    B = vp_next @ poses[k + 1][:3, :3].T + poses[k + 1][:3, 3]
    I3 = np.broadcast_to(np.eye(3), (len(vp), 3, 3))
    Ja = np.concatenate([-_skew(A), I3], 2).reshape(-1, 6)
    Jb = -np.concatenate([-_skew(B), I3], 2).reshape(-1, 6)
    _add(H, g, k, k + 1, Ja, Jb, np.full(3 * len(vp), 1.0 / sigma**2), (A - B).ravel())


def _reproj_stats(matches, poses, K, loops_only=False):
    errs = []
    for m in matches:
        if loops_only and m.j - m.i <= 3:
            continue
        T_ji = _inv(poses[m.j]) @ poses[m.i]
        h = ~np.isnan(m.Xi[:, 2])
        if h.any():
            errs.append(np.linalg.norm(_project(m.Xi[h] @ T_ji[:3, :3].T + T_ji[:3, 3], K) - m.xj[h], axis=1))
    e = np.concatenate(errs) if errs else np.array([np.nan])
    return float(np.median(e)), float(np.percentile(e, 90))


def optimize_poses(images, depths, poses, K, size: int, baseline: float, log: Callable[[str], None] = print,
                   radii=(40.0, 20.0, 10.0), iters: int = 4, w_dense: float = 0.25):
    """Return drift-corrected camera->world poses (first keyframe fixed).

    `depths` are the raw keyframe depths. One Gauss-Newton problem over all poses with
    analytic Jacobians: feature reprojection (fixes sliding along flat surfaces), dense
    point-to-plane distances between the consistency-filtered depth maps of co-visible
    keyframes (aligns the surfaces themselves, weighted by stereo noise), and weak odometry
    priors. Each round re-matches features with the improved poses; each iteration
    re-associates the dense points with a shrinking gate."""
    from .fusion import filter_depths

    t0 = time.time()
    n = len(poses)
    poses = [np.array(T, np.float64) for T in poses]
    original = [T.copy() for T in poses]
    vp = np.array([[-0.1, -0.1, 0.35], [0.1, -0.1, 0.35], [0.1, 0.1, 0.35], [-0.1, 0.1, 0.35]])
    vp_next = [vp @ _inv(_inv(poses[k]) @ poses[k + 1])[:3, :3].T + _inv(_inv(poses[k]) @ poses[k + 1])[:3, 3]
               for k in range(n - 1)]
    feats = extract(images, depths, K)
    pairs = covisible_pairs(depths, poses, K, size)
    dense = _Dense(filter_depths(depths, poses, K), K, K[0, 0] * baseline) if w_dense > 0 else None
    dirs = pairs + [(j, i) for i, j in pairs]
    log(f"loop closure: {len(pairs)} co-visible keyframe pairs ({time.time()-t0:.0f} s)")
    step = 0
    for r, radius in enumerate(radii):
        matches = []
        for i, j in pairs:
            m = match_pair(feats[i], feats[j], poses[i], poses[j], K, radius)
            if m is not None:
                m.i, m.j = i, j
                matches.append(m)
        before = _reproj_stats(matches, poses, K, loops_only=True)
        for _ in range(iters):
            gate = max(0.03 * 0.6**step, 0.006)  # metres
            step += 1
            H = np.zeros((6 * n, 6 * n))
            g = np.zeros(6 * n)
            chi = []
            if dense is not None:
                for i, j in dirs:
                    e = dense.add(H, g, poses, i, j, gate, w_dense)
                    if e is not None:
                        chi.append(e)
            for m in matches:
                _add_features(H, g, poses, m, K, sigma=1.0)
            for k in range(n - 1):
                _add_odometry(H, g, poses, k, vp, vp_next[k], sigma=0.002)
            Hk = H[6:, 6:]  # keyframe 0 fixes the gauge
            Hk += 1e-6 * np.eye(len(Hk)) + 1e-3 * np.diag(np.diag(Hk))
            dx = np.r_[np.zeros(6), -np.linalg.solve(Hk, g[6:])]
            poses = [_exp(dx[6 * k : 6 * k + 6]) @ poses[k] for k in range(n)]
        after = _reproj_stats(matches, poses, K, loops_only=True)
        dense_txt = f", depth residual median {np.median(np.concatenate(chi)):.2f} sigma" if chi else ""
        log(f"  round {r+1}: {len(matches)} links ({sum(m.j - m.i > 3 for m in matches)} revisits), revisit "
            f"reprojection error {before[0]:.2f} -> {after[0]:.2f} px (p90 {before[1]:.1f} -> {after[1]:.1f})"
            f"{dense_txt} ({time.time()-t0:.0f} s)")
    moved = [np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(original, poses)]
    log(f"loop closure done: keyframes moved median {np.median(moved)*1000:.0f} mm, "
        f"max {np.max(moved)*1000:.0f} mm ({time.time()-t0:.0f} s)")
    return poses


def optimize_session(session, log: Callable[[str], None] = print):
    """Drift-correct a Session's keyframe poses in place (caller saves). Returns (old, new) poses."""
    frames = [session.load_frame(r) for r in session.records]
    old = [r.T_wc.copy() for r in session.records]
    new = optimize_poses([f[0] for f in frames], [f[1] for f in frames], old, session.K, session.size,
                         session.baseline, log)
    for rec, T in zip(session.records, new):
        rec.T_wc = T
    return old, new
