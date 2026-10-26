"""Map building: voxel-averaged point cloud (live) and TSDF -> mesh (export)."""

from __future__ import annotations

from typing import Callable, Iterable

import cv2
import numpy as np

_OFF = 1 << 20  # voxel index offset so packed keys stay positive


class VoxelCloud:
    """Point cloud averaged per voxel; noise shows up as low-count voxels."""

    def __init__(self, voxel: float = 0.003):
        self.voxel = voxel
        self.keys = np.empty(0, np.int64)
        self.xyz = np.empty((0, 3), np.float64)  # sums
        self.val = np.empty(0, np.float64)  # intensity sums
        self.cnt = np.empty(0, np.int64)
        self.version = 0

    def _pack(self, pts: np.ndarray) -> np.ndarray:
        q = np.floor(pts / self.voxel).astype(np.int64) + _OFF
        return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]

    def add(self, pts: np.ndarray, intensity: np.ndarray):
        if len(pts) == 0:
            return
        keys = np.concatenate([self.keys, self._pack(pts)])
        uk, inv = np.unique(keys, return_inverse=True)
        n = len(uk)
        xyz = np.concatenate([self.xyz, pts.astype(np.float64)])
        val = np.concatenate([self.val, intensity.astype(np.float64)])
        cnt = np.concatenate([self.cnt, np.ones(len(pts), np.int64)])
        self.xyz = np.stack([np.bincount(inv, xyz[:, i], n) for i in range(3)], 1)
        self.val = np.bincount(inv, val, n)
        self.cnt = np.bincount(inv, cnt, n).astype(np.int64)
        self.keys = uk
        self.version += 1

    def points(self, min_count: int = 2):
        m = self.cnt >= min_count
        c = self.cnt[m][:, None]
        return (self.xyz[m] / c).astype(np.float32), (self.val[m] / c[:, 0]).astype(np.float32)

    def __len__(self):
        return len(self.keys)


def backproject(depth: np.ndarray, K: np.ndarray, stride: int = 1):
    """Depth map -> (N,3) camera-frame points and their pixel coordinates."""
    d = depth[::stride, ::stride]
    v, u = np.nonzero(d > 0)
    z = d[v, u].astype(np.float64)
    u, v = u * stride, v * stride
    X = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    return X, u, v


def consistency_check(depth: np.ndarray, T_wc: np.ndarray, others, K: np.ndarray, tol: float = 0.02) -> np.ndarray:
    """Per pixel of `depth`: in how many of `others` [(depth, T_wc), ...] the same surface shows up.

    SGBM mismatches (repetitive keys/pegboard, glossy or dark surfaces, edges along the
    baseline) depend on the viewpoint, so they rarely agree with neighbouring keyframes,
    while real surfaces do. Returns an int count image (0 where depth is invalid)."""
    X, u, v = backproject(depth, K)
    agree = np.zeros(len(X), np.int32)
    h, w = depth.shape
    for d_o, T_o in others:
        T = np.linalg.inv(T_o) @ T_wc
        Y = X @ T[:3, :3].T + T[:3, 3]
        z = np.maximum(Y[:, 2], 1e-6)
        uo = np.round(Y[:, 0] / z * K[0, 0] + K[0, 2]).astype(int)
        vo = np.round(Y[:, 1] / z * K[1, 1] + K[1, 2]).astype(int)
        ok = (Y[:, 2] > 0.05) & (uo >= 0) & (vo >= 0) & (uo < w) & (vo < h)
        dz = np.zeros(len(X))
        dz[ok] = d_o[vo[ok], uo[ok]]
        agree += (dz > 0) & (np.abs(dz - Y[:, 2]) < tol * Y[:, 2])
    out = np.zeros(depth.shape, np.int32)
    out[v, u] = agree
    return out


def fuse_depth_window(depth: np.ndarray, T_wc: np.ndarray, frames, K: np.ndarray, tol: float = 0.02,
                      min_obs: int = 3) -> np.ndarray:
    """Temporal depth fusion for one keyframe.

    `frames`: [(T_wc, X), ...] camera-frame depth points of the frames around the keyframe
    (~20 fps vs. a keyframe every few cm). Every observation is warped into the keyframe.
    Pixels the keyframe measured itself are averaged with the observations within `tol` of
    their depth (kept unchanged if nothing agrees - the keyframe-level consistency check
    decides later); pixels it missed are filled with the mean of the observations within
    `tol` of their median when `min_obs` agree, which recovers depth where matching flickers
    (dark, low-texture IR). Never removes depth. Runs live once per keyframe, so it avoids
    sorting the observations of measured pixels (the bulk of them)."""
    frames = [f for f in frames if len(f[1])]
    min_obs = min(min_obs, 1 + len(frames))  # few neighbours (first/last keyframe): degrade gracefully
    h, w = depth.shape
    T_kw = np.linalg.inv(T_wc)
    pix, zs = [], []
    for T_j, X in frames:
        P = (K @ (T_kw @ T_j)[:3]).astype(np.float32)  # -> (u*z, v*z, z) in the keyframe
        Y = cv2.transform(X.reshape(-1, 1, 3).astype(np.float32, copy=False), P).reshape(-1, 3)
        z = Y[:, 2]
        ok = (z > 0.05) & (z < 3.2)
        Y, z = Y[ok], z[ok]
        uu = np.floor(Y[:, 0] / z + 0.5).astype(np.int32)
        vv = np.floor(Y[:, 1] / z + 0.5).astype(np.int32)
        ins = (uu >= 0) & (vv >= 0) & (uu < w) & (vv < h)
        pix.append(vv[ins] * w + uu[ins])
        zs.append(z[ins])
    own = depth.ravel().astype(np.float32)
    out = own.copy()
    if not pix:
        return out.reshape(h, w)
    p = np.concatenate(pix)
    z = np.concatenate(zs)
    # measured pixels: average with the observations that agree
    ref = own[p]
    m = ref > 0
    inl = m & (np.abs(z - ref) < tol * ref)
    n_inl = np.bincount(p[inl], minlength=h * w)
    s_inl = np.bincount(p[inl], weights=z[inl], minlength=h * w)
    upd = n_inl > 0
    out[upd] = (own[upd] + s_inl[upd]) / (1 + n_inl[upd])
    # unmeasured pixels: one sort by pixel, then depth -> per-pixel median and its inliers
    p, z = p[~m], z[~m]
    if len(p):
        q = 20000.0  # depth quantisation for the sort key: 0.05 mm steps, < 2^16 up to 3.2 m
        k = np.sort((p.astype(np.int64) << 16) | np.minimum(z * q, 65535).astype(np.int64))
        f = k >> 16
        zq = (k & 0xFFFF).astype(np.float32) / q
        start = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
        counts = np.diff(np.r_[start, len(f)])
        med = np.repeat(zq[start + counts // 2], counts)
        inl = np.abs(zq - med) < tol * med
        n_i = np.add.reduceat(inl.astype(np.int32), start)
        s_i = np.add.reduceat(np.where(inl, zq, 0), start)
        fill = n_i >= min_obs
        out[f[start][fill]] = s_i[fill] / n_i[fill]
    return out.reshape(h, w)


def filter_depths(depths, poses, K: np.ndarray, neighbours: int = 3, min_agree: int = 2, tol: float = 0.02):
    """Keep only depth pixels confirmed by >= min_agree of the +-`neighbours` keyframes."""
    out = []
    n = len(depths)
    for i in range(n):
        others = [(depths[j], poses[j]) for j in range(max(0, i - neighbours), min(n, i + neighbours + 1)) if j != i]
        need = min(min_agree, len(others))
        out.append(np.where(consistency_check(depths[i], poses[i], others, K, tol) >= need, depths[i], 0).astype(np.float32))
    return out


def cloud_from_depths(depths, images, poses, K: np.ndarray, voxel: float = 0.002, stride: int = 2):
    """Voxel-averaged world point cloud (points, IR intensity) from depth maps."""
    cloud = VoxelCloud(voxel)
    for d, img, T in zip(depths, images, poses):
        X, u, v = backproject(d, K, stride)
        if len(X):
            cloud.add(X @ T[:3, :3].T + T[:3, 3], img[v, u])
    return cloud.points(min_count=1)


def _dominant_plane(pts: np.ndarray, iters: int = 300, tol: float = 0.006, seed: int = 0):
    """RANSAC plane with the most support -> (normal, point on plane, inlier fraction)."""
    rng = np.random.default_rng(seed)
    best, best_n, best_p = 0, None, None
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        cnt = int((np.abs((pts - a) @ n) < tol).sum())
        if cnt > best:
            best, best_n, best_p = cnt, n, a
    return best_n, best_p, best / len(pts)


def estimate_up(points: np.ndarray, poses) -> tuple[np.ndarray, bool]:
    """World "up" for display: normal of the dominant plane (usually the table) on the cameras' side.

    Falls back to the cameras' mean up axis. Also returns whether the device was held
    upside down (camera images need a 180 degree rotation to look upright)."""
    C = np.array([T[:3, 3] for T in poses])
    cam_up = -np.mean([T[:3, 1] for T in poses], 0)  # OpenCV cameras: y points down
    cam_up /= np.linalg.norm(cam_up)
    sub = points[np.random.default_rng(0).choice(len(points), min(len(points), 30000), replace=False)]
    n, p0, frac = _dominant_plane(sub) if len(sub) >= 3 else (None, None, 0.0)
    if n is None or frac < 0.15 or abs(n @ cam_up) < 0.5:  # no clear horizontal plane
        return cam_up, False
    up = n if np.mean((C - p0) @ n) > 0 else -n
    return up, bool(up @ cam_up < 0)


def robust_bounds(pts: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(pts, 0.5, axis=0) - margin
    hi = np.percentile(pts, 99.5, axis=0) + margin
    return lo, hi


class TSDFVolume:
    """Sparse narrow-band TSDF.

    Only voxels near `band_points` (the fused cloud) are stored, in cells of 3^3 voxels, and the
    mesh is extracted chunk by chunk. So memory follows the surface area, not the bounding box,
    and a whole desk scene fits at 1.5 mm. `max_voxels` caps the stored voxels (~18 bytes each);
    above it the voxel grows."""

    CELL = 3

    def __init__(self, lo: np.ndarray, hi: np.ndarray, voxel: float, trunc_vox: float = 4.0,
                 max_voxels: int = 350**3, band_points: np.ndarray | None = None):
        from scipy.ndimage import binary_dilation

        self.lo = lo.astype(np.float64)
        ext = np.asarray(hi, np.float64) - self.lo
        c = self.CELL
        reach = int(np.ceil((trunc_vox + 1) / c))  # band half-width in cells
        while True:
            dims = np.maximum(np.ceil(ext / voxel).astype(int), 2)
            cdims = -(-dims // c)
            if np.prod(cdims.astype(np.float64)) > 3e8:  # even the coarse occupancy grid is too big
                voxel *= 1.25
                continue
            if band_points is None:
                occ = np.ones(cdims, bool)
            else:
                occ = np.zeros(cdims, bool)
                q = np.floor((band_points - self.lo) / (voxel * c)).astype(np.int64)
                q = q[np.all((q >= 0) & (q < cdims), 1)]
                occ[q[:, 0], q[:, 1], q[:, 2]] = True
                occ = binary_dilation(occ, np.ones((3, 3, 3), bool), iterations=reach)
            n = int(occ.sum()) * c**3
            if n <= max_voxels:
                break
            voxel *= 1.02 * (n / max_voxels) ** 0.5  # band voxels scale with surface area ~ 1/voxel^2
        cells = np.argwhere(occ)
        offs = np.stack(np.meshgrid(*[np.arange(c)] * 3, indexing="ij"), -1).reshape(-1, 3)
        itype = np.int16 if dims.max() < 32000 else np.int32
        # voxel k of cell i is stored at i * 27 + k (cells may stick out of the box by < 3 voxels)
        self.idx = (cells[:, None, :] * c + offs[None]).reshape(-1, 3).astype(itype)
        self.cells = cells.astype(itype)
        self.cell_centers = (self.lo + (cells * c + c / 2) * voxel).astype(np.float32)
        self.voxel = float(voxel)
        self.dims = dims
        self.trunc = trunc_vox * voxel
        n = len(self.idx)
        self.tsdf = np.ones(n, np.float32)
        self.weight = np.zeros(n, np.float32)
        self.val = np.zeros(n, np.float32)

    def _visible_voxels(self, R, t, K, depth: np.ndarray) -> np.ndarray:
        """Indices of the voxels in cells this depth map can update.

        Per cell of 3^3 voxels: inside the view frustum, some depth measured in its footprint,
        and not more than the truncation distance behind all of it (occluded voxels get no
        update anyway). Cheap at cell level, and it skips most of a wide-angle view."""
        h, w = depth.shape
        B = 8  # footprint blocks: max depth per 8x8 pixels, spread to the neighbouring blocks
        hb, wb = -(-h // B), -(-w // B)
        pad = np.zeros((hb * B, wb * B), np.float32)
        pad[:h, :w] = depth
        dmax = pad.reshape(hb, B, wb, B).max(axis=(1, 3))
        dmax = cv2.dilate(dmax, np.ones((3, 3), np.uint8))
        Pc = self.cell_centers @ R.T + t
        z = Pc[:, 2]
        rad = np.float32(0.87 * self.CELL * self.voxel)  # cell half-diagonal
        zs = np.maximum(z, rad)
        fx = np.float32(K[0, 0])
        u = Pc[:, 0] / zs * fx + np.float32(K[0, 2])
        v = Pc[:, 1] / zs * np.float32(K[1, 1]) + np.float32(K[1, 2])
        m = rad / zs * fx + 1  # cell radius in pixels
        ok = (z > -rad) & (u > -m) & (u < w + m) & (v > -m) & (v < h + m)
        small = ok & (m < B)  # footprint within the 3x3 blocks around its centre: depth test
        bu = np.clip((u[small] // B).astype(np.int32), 0, wb - 1)
        bv = np.clip((v[small] // B).astype(np.int32), 0, hb - 1)
        dm = dmax[bv, bu]
        ok[np.nonzero(small)[0][(dm <= 0) | (z[small] - rad > dm + self.trunc)]] = False
        n = self.CELL**3
        return (np.nonzero(ok)[0][:, None].astype(np.int64) * n + np.arange(n)).ravel()

    def integrate(self, depth: np.ndarray, intensity: np.ndarray, K: np.ndarray, T_wc: np.ndarray):
        T_cw = np.linalg.inv(T_wc)
        R = T_cw[:3, :3].astype(np.float32)
        d = depth.astype(np.float32)
        h, w = depth.shape
        fx, fy, cx, cy = (np.float32(K[0, 0]), np.float32(K[1, 1]), np.float32(K[0, 2]), np.float32(K[1, 2]))
        z_far = np.float32(d.max() + self.trunc)
        vis = self._visible_voxels(R, T_cw[:3, 3].astype(np.float32), K, d)
        # voxel centres are (idx + 0.5) * voxel + lo: fold that into the transform
        t = (T_cw[:3, :3] @ (self.lo + 0.5 * self.voxel) + T_cw[:3, 3]).astype(np.float32)
        Rv = R * np.float32(self.voxel)
        chunk = 2_000_000
        for s0 in range(0, len(vis), chunk):
            sel = vis[s0 : s0 + chunk]
            P = self.idx[sel].astype(np.float32) @ Rv.T + t
            z = P[:, 2]
            cand = np.nonzero((z > 1e-3) & (z < z_far))[0]  # cheap depth cull first
            if len(cand) == 0:
                continue
            z = z[cand]
            u = np.rint(P[cand, 0] / z * fx + cx).astype(np.int32)
            v = np.rint(P[cand, 1] / z * fy + cy).astype(np.int32)
            inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
            cand, z, u, v = cand[inside], z[inside], u[inside], v[inside]
            dz = d[v, u]
            sdf = dz - z
            upd = (dz > 0) & (sdf > -self.trunc)
            if not upd.any():
                continue
            gi = sel[cand[upd]]
            dz = dz[upd]
            tsdf_new = np.minimum(sdf[upd] / self.trunc, 1.0)
            wn = 0.01 / np.maximum(dz, 0.05) ** 2  # trust near measurements more
            W = self.weight[gi]
            Wt = W + wn
            self.tsdf[gi] = (self.tsdf[gi] * W + tsdf_new * wn) / Wt
            self.val[gi] = (self.val[gi] * W + intensity[v[upd], u[upd]] * wn) / Wt
            self.weight[gi] = np.minimum(Wt, 50.0)

    def _chunks(self, size: int = 160):
        """Dense (tsdf, weight, val, origin) blocks of `size`^3 cells; neighbours share one voxel plane."""
        c, n = self.CELL, self.CELL**3
        cmin = self.cells.astype(np.int32) * c
        for o in np.unique(cmin // size, axis=0) * size:
            near = np.nonzero(np.all((cmin + c - 1 >= o) & (cmin <= o + size), 1))[0]  # cells touching the chunk
            cand = (near[:, None] * n + np.arange(n)).ravel()
            ci = self.idx[cand].astype(np.int32)
            keep = np.all((ci >= o) & (ci <= o + size), 1)
            sel, loc = cand[keep], ci[keep] - o
            shape = loc.max(0) + 1
            if np.any(shape < 2):
                continue
            tsdf = np.ones(shape, np.float32)
            wgt = np.zeros(shape, np.float32)
            val = np.zeros(shape, np.float32)
            tsdf[loc[:, 0], loc[:, 1], loc[:, 2]] = self.tsdf[sel]
            wgt[loc[:, 0], loc[:, 1], loc[:, 2]] = self.weight[sel]
            val[loc[:, 0], loc[:, 1], loc[:, 2]] = self.val[sel]
            yield tsdf, wgt, val, o

    def extract_mesh(self, min_weight: float = 0.0, keep_frac: float = 0.002, min_faces: int = 60,
                     smooth: int = 10, max_faces: int | None = None):
        """Marching cubes -> trimesh (metres, vertex colours = IR intensity).

        `max_faces`: quadric decimation down to this many triangles (STL size is 50 bytes per
        triangle). Flat areas collapse into few large triangles, detail keeps its triangles."""
        from skimage.measure import marching_cubes
        import trimesh

        V, F, G = [], [], []
        nv = 0
        for tsdf, wgt, val, o in self._chunks():
            obs = wgt > max(min_weight, 1e-6)
            # Require all 8 corners of a cube observed, otherwise fake surfaces appear where
            # observed voxels meet unobserved ones. skimage tests a cube against the mask at
            # its *far* corner (i+1, j+1, k+1), so the all-corners mask is shifted by one.
            full = obs.copy()
            full[:-1] &= obs[1:]
            full[:, :-1] &= full[:, 1:].copy()
            full[:, :, :-1] &= full[:, :, 1:].copy()
            mask = np.zeros_like(obs)
            mask[1:, 1:, 1:] = full[:-1, :-1, :-1]
            if mask.sum() < 8 or tsdf[obs].min() > 0 or tsdf[obs].max() < 0:
                continue
            try:
                verts, faces, _, _ = marching_cubes(tsdf, level=0.0, mask=mask, allow_degenerate=False)
            except (ValueError, RuntimeError):  # no surface crossing in this chunk
                continue
            vi = np.clip(np.round(verts).astype(int), 0, np.array(tsdf.shape) - 1)
            G.append(val[vi[:, 0], vi[:, 1], vi[:, 2]])
            V.append(verts + o)
            F.append(faces + nv)
            nv += len(verts)
        if not F:
            return None
        verts = np.vstack(V) * self.voxel + self.lo + 0.5 * self.voxel
        col = np.clip(np.concatenate(G), 0, 255).astype(np.uint8)
        colors = np.stack([col, col, col, np.full_like(col, 255)], 1)
        mesh = trimesh.Trimesh(verts, np.vstack(F), vertex_colors=colors, process=True)  # merges chunk seams
        if keep_frac > 0 and len(mesh.faces) > 0:
            # drop small floating fragments (noise); label components in one scipy pass,
            # trimesh's split() builds a mesh object per component and crawls on noisy scans
            labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
            sizes = np.bincount(labels)
            mesh.update_faces(sizes[labels] >= max(min_faces, keep_frac * sizes.max()))
            mesh.remove_unreferenced_vertices()
        if smooth > 0 and len(mesh.faces):
            # Taubin smoothing removes stereo/voxel noise without shrinking the surface
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth)
        if max_faces and len(mesh.faces) > max_faces:
            mesh = _decimate(mesh, max_faces)
        return mesh


def _decimate(mesh, max_faces: int):
    """Quadric edge-collapse decimation; vertex colours carried over from the nearest old vertex."""
    import trimesh
    from scipy.spatial import cKDTree

    small = mesh.simplify_quadric_decimation(face_count=int(max_faces))
    near = cKDTree(mesh.vertices).query(small.vertices)[1]
    return trimesh.Trimesh(small.vertices, small.faces, vertex_colors=mesh.visual.vertex_colors[near], process=False)


def fuse_keyframes(frames: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]], n: int, K: np.ndarray,
                   voxel: float, bounds: tuple[np.ndarray, np.ndarray], band_points: np.ndarray | None = None,
                   progress: Callable[[float], None] | None = None, max_voxels: int = 350**3):
    """frames yields (depth, gray image, T_wc); streamed so all keyframes never sit in RAM."""
    vol = TSDFVolume(bounds[0], bounds[1], voxel, max_voxels=max_voxels, band_points=band_points)
    for i, (d, g, T) in enumerate(frames):
        vol.integrate(d, g, K, T)
        if progress:
            progress((i + 1) / max(n, 1))
    return vol
