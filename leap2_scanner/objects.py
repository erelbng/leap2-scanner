"""Object cut-out: fit a box around the object a scan was centred on, close its open side.

The box is the unit of everything here: T_box (box -> world) and its size, centred on the box
origin. If the object rests on a surface (table, floor, shelf, wall, ...), the box's z axis is
that surface's normal and its floor lies on the surface, so "cut a few mm above the floor"
removes the surface and leaves the object's open side, which `cap_bottom` closes. Nothing
assumes which way is up: the Leap has no gravity sensor, and objects don't have to stand on
anything.
"""

from __future__ import annotations

import numpy as np

from .fusion import _dominant_plane


def focus_points(depths, poses, K: np.ndarray) -> np.ndarray:
    """Where the scan looked: the 3D point at the centre of every keyframe (median depth)."""
    pts = []
    for d, T in zip(depths, poses):
        h, w = d.shape
        win = d[h // 2 - 30 : h // 2 + 30, w // 2 - 30 : w // 2 + 30]
        win = win[win > 0]
        if len(win) < 50:
            continue
        z = float(np.median(win))
        X = np.array([(w / 2 - K[0, 2]) / K[0, 0] * z, (h / 2 - K[1, 2]) / K[1, 1] * z, z])
        pts.append(T[:3, :3] @ X + T[:3, 3])
    if not pts:
        return np.array([T[:3, 3] for T in poses])
    return np.array(pts)


def support_plane(points: np.ndarray, poses, focus: np.ndarray, radius: float):
    """The surface the object rests on: the dominant plane around `focus` that lies behind it
    as seen from the cameras -> (normal towards the cameras, point on the plane), or None.

    Only the neighbourhood counts: a big surface is not flat to the millimetre in the
    reconstruction (residual drift bends it), but locally around one object it is."""
    near = points[np.linalg.norm(points - focus, axis=1) < radius]
    if len(near) < 1000:
        return None
    sub = near[np.random.default_rng(0).choice(len(near), min(len(near), 50000), replace=False)]
    C = np.array([T[:3, 3] for T in poses])

    def plane(pts):
        n, p0, frac = _dominant_plane(pts, iters=400)
        if n is None:
            return None
        inl = pts[np.abs((pts - p0) @ n) < 0.006]
        c = inl.mean(0)
        n = np.linalg.svd(inl - c, full_matrices=False)[2][2]  # least-squares refit on the inliers
        n = n if np.mean((C - c) @ n) > 0 else -n
        return n, c, len(inl) / len(sub), np.abs((pts - c) @ n) < 0.006

    first = plane(sub)
    if first is None or first[2] < 0.2:
        return None
    n, c, _, inl = first
    # The object's own big flat face (a lid, a keyboard deck) can outweigh the surface below it:
    # a sizeable parallel plane further from the cameras is the real support.
    rest = sub[~inl]
    if len(rest) > 500:
        second = plane(rest)
        if second is not None and second[2] >= 0.1 and abs(second[0] @ n) > 0.95 and (second[1] - c) @ n < -0.005:
            n, c = second[0], second[1]
    if (focus - c) @ n < -0.01:  # in front of what the scan looked at: not what it rests on
        return None
    return n, c


def _cluster(P: np.ndarray, focus: np.ndarray, within: float, cell: float = 0.005):
    """Points of the connected cluster (cells with >= 3 points, 26-neighbourhood) nearest to
    `focus`, if that is closer than `within`; lone noise points would otherwise bridge separate
    things. The whole cluster is returned, however far it extends. None if there is none."""
    from scipy import ndimage

    if len(P) < 100:
        return None
    qi = np.floor((P - P.min(0)) / cell).astype(int)
    cnt = np.zeros(qi.max(0) + 1, np.int32)
    np.add.at(cnt, (qi[:, 0], qi[:, 1], qi[:, 2]), 1)
    lab, n = ndimage.label(cnt >= 3, structure=np.ones((3, 3, 3), bool))
    pl = lab[qi[:, 0], qi[:, 1], qi[:, 2]]
    sizes = np.bincount(pl, minlength=n + 1)
    best, best_d = 0, np.inf
    for c in range(1, n + 1):  # nearest sizeable cluster; a near tie goes to the bigger one
        if sizes[c] < max(50, 0.02 * len(P)):
            continue
        d = np.min(np.linalg.norm(P[pl == c] - focus, axis=1))
        if d < best_d - 0.01 or (abs(d - best_d) <= 0.01 and sizes[c] > sizes[best]):
            best, best_d = c, d
    if best == 0 or best_d > within:
        return None
    return P[pl == best]


def _basis(n: np.ndarray):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def _footprint(O: np.ndarray, n: np.ndarray, origin: np.ndarray, margin: float):
    """Tightest rotated rectangle around the points projected along n (x = long side)
    -> (axes A1, A2, centre on the plane through origin, size)."""
    import cv2

    e1, e2 = _basis(n)
    xy = np.stack([(O - origin) @ e1, (O - origin) @ e2], 1)
    c2 = np.median(xy, 0)
    r = np.linalg.norm(xy - c2, axis=1)
    ang = np.radians(cv2.minAreaRect(xy[r <= np.percentile(r, 99.5)].astype(np.float32))[2])
    a1 = np.array([np.cos(ang), np.sin(ang)])
    for _ in range(2):
        a2 = np.array([-a1[1], a1[0]])
        u = (xy - c2) @ np.stack([a1, a2], 1)
        lo, hi = np.percentile(u, 0.3, 0) - margin, np.percentile(u, 99.7, 0) + margin
        if hi[0] - lo[0] >= hi[1] - lo[1]:
            break
        a1 = a2
    mid = c2 + ((lo + hi) / 2) @ np.stack([a1, a2])
    A1 = a1[0] * e1 + a1[1] * e2
    return A1, np.cross(n, A1), origin + mid[0] * e1 + mid[1] * e2, hi - lo


def _box(A1, A2, n, centre, size):
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = A1, A2, n
    T[:3, 3] = centre
    return T, np.asarray(size, np.float64)


def _box_on_surface(O: np.ndarray, n: np.ndarray, p0: np.ndarray, margin: float):
    """Box standing on the plane (n, p0), around the object's footprint."""
    A1, A2, c, wh = _footprint(O, n, p0, margin)
    top = float(np.percentile((O - p0) @ n, 99.5)) + margin
    return _box(A1, A2, n, c + n * top / 2, [wh[0], wh[1], top])


def _box_free(O: np.ndarray, margin: float):
    """Box around a cluster with nothing to stand on: z = its thinnest direction, footprint
    = tightest rectangle across it (principal axes alone are arbitrary for round or square shapes)."""
    c = np.median(O, 0)
    n = np.linalg.eigh(np.cov((O - c).T))[1][:, 0]
    A1, A2, mid, wh = _footprint(O, n, c, margin)
    t = (O - c) @ n
    lo, hi = np.percentile(t, 0.3) - margin, np.percentile(t, 99.7) + margin
    return _box(A1, A2, n, mid + n * (lo + hi) / 2, [wh[0], wh[1], hi - lo])


def fit_object_box(points: np.ndarray, poses, depths, K: np.ndarray, margin: float = 0.01,
                   max_radius: float = 1.0):
    """Box around the object the scan was centred on -> (T_box, size, description, on_surface).

    Focus = median of the points at the keyframes' image centres. The object is the cluster
    nearest to it, within 1.5x the spread of those points (an orbit around a mug looks at the
    mug, a sweep over a desk at the desk) - the whole cluster, however far it extends.
    If the object rests on a surface, the box stands on it (z = surface normal, floor on the
    surface) around the nearest cluster above it. Otherwise - free-standing, held, hanging,
    or the "surface" was the object's own big face - the nearest cluster gets a box along its
    principal axes."""
    centres = focus_points(depths, poses, K)
    focus = np.median(centres, 0)
    spread = float(np.percentile(np.linalg.norm(centres - focus, axis=1), 80))
    radius = float(np.clip(1.5 * spread, 0.05, max_radius))
    reach = min(max_radius, max(0.1, 2 * radius))  # the object may extend beyond the focus spread, not into the next room
    sup = support_plane(points, poses, focus, radius + 0.1)
    if sup is not None:
        n, p0 = sup
        rel = points - focus
        lateral = np.linalg.norm(rel - np.outer(rel @ n, n), axis=1)
        h = (points - p0) @ n
        # how far above the surface counts as object: as far as the surface's own noise reaches
        # below the plane, where only surface points are (far views and slightly misaligned
        # passes blur a surface by several mm, with long tails)
        below = -h[(lateral < reach) & (h < 0) & (h > -0.05)]
        h_min = max(0.005, float(np.percentile(below, 99))) if len(below) > 200 else 0.005
        cand = (h > h_min) & (h < reach) & (lateral < reach)
        O = _cluster(points[cand], focus, radius)
        if O is None:  # nothing clearly above the surface: a thin object - cover what the scan looked at
            A1, A2, c, wh = _footprint(centres, n, p0, 0.03)
            height = max(0.02, 2 * h_min)
            T, size = _box(A1, A2, n, c + n * height / 2, [wh[0], wh[1], height])
            return T, size, (f"nothing stands out more than {h_min*1000:.0f} mm from the surface: flat box over "
                             f"the area the scan centred on, {np.round(size * 100, 1)} cm - adjust it by hand"), True
        if O is not None:
            T, size = _box_on_surface(O, n, p0, margin)
            # a real support reaches beyond the object; a plane that ends at its footprint is
            # one of the object's own faces (something held or lying with nothing around it)
            on = points[(np.abs(h) < 0.003) & (lateral < reach)]
            u = (on - T[:3, 3]) @ T[:3, :2]
            outside = np.any(np.abs(u) > size[:2] / 2, axis=1).sum()
            if outside >= max(300, 0.2 * len(on)):
                return T, size, (f"object of {len(O):,} points resting on a surface, {np.round(size * 100, 1)} cm "
                                 f"(parts lower than {h_min*1000:.0f} mm can't be told from the surface)"), True
    O = _cluster(points[np.linalg.norm(points - focus, axis=1) < reach], focus, radius)
    if O is None:
        T = np.eye(4)
        T[:3, 3] = focus
        return T, np.full(3, 0.2), "nothing found near the scan's focus: 20 cm box there", False
    T, size = _box_free(O, margin)
    return T, size, f"object of {len(O):,} points (no supporting surface), {np.round(size * 100, 1)} cm", False


# ---------------------------------------------------------------------- closing the bottom
def _ear_clip(xy: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple counter-clockwise polygon (n,2) -> index triples."""
    idx = list(range(len(xy)))
    tris = []
    while len(idx) > 3:
        P = xy[idx]
        prev, nxt = np.roll(P, 1, 0), np.roll(P, -1, 0)
        cross = (P[:, 0] - prev[:, 0]) * (nxt[:, 1] - P[:, 1]) - (P[:, 1] - prev[:, 1]) * (nxt[:, 0] - P[:, 0])
        convex = cross > 1e-14
        R = P[~convex]
        m = len(idx)
        for k in np.nonzero(convex)[0]:
            a, b, c = prev[k], P[k], nxt[k]
            if len(R):  # no reflex vertex inside the (counter-clockwise) candidate ear
                d1 = (b[0] - a[0]) * (R[:, 1] - a[1]) - (b[1] - a[1]) * (R[:, 0] - a[0])
                d2 = (c[0] - b[0]) * (R[:, 1] - b[1]) - (c[1] - b[1]) * (R[:, 0] - b[0])
                d3 = (a[0] - c[0]) * (R[:, 1] - c[1]) - (a[1] - c[1]) * (R[:, 0] - c[0])
                if np.any((d1 > 0) & (d2 > 0) & (d3 > 0)):  # strict: the ear's own corners are not inside
                    continue
            tris.append((idx[(k - 1) % m], idx[k], idx[(k + 1) % m]))
            del idx[k]
            break
        else:  # only degenerate (noisy, self-touching) corners left: clip one anyway
            tris.append((idx[-1], idx[0], idx[1]))
            del idx[0]
    tris.append(tuple(idx))
    return tris


def _boundary_loops(faces: np.ndarray, on_plane: np.ndarray) -> list[list[int]]:
    """Closed loops of boundary edges lying on the cut plane, in the faces' edge direction."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    rev = {(int(b), int(a)) for a, b in e}
    nxt = {}
    for a, b in e:
        a, b = int(a), int(b)
        if (a, b) not in rev and on_plane[a] and on_plane[b]:
            nxt[a] = b
    loops, seen = [], set()
    for s in list(nxt):
        if s in seen:
            continue
        loop, v = [], s
        while v in nxt and v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        if v == s and len(loop) >= 3:  # closed
            loops.append(loop)
    return loops


def cap_bottom(mesh, origin: np.ndarray, up: np.ndarray, close: bool = True):
    """Cut the mesh at the plane (origin, normal `up`), keep the side `up` points to, and close
    each cut outline with a flat face. Outlines inside other outlines (holes) stay open."""
    import trimesh
    from scipy.spatial import cKDTree

    colors = mesh.visual.vertex_colors if mesh.visual.kind == "vertex" else None
    # slice_faces_plane is plain numpy (slice_mesh_plane needs shapely even without capping)
    res = trimesh.intersections.slice_faces_plane(mesh.vertices, mesh.faces, plane_normal=up, plane_origin=origin)
    cut = trimesh.Trimesh(res[0], res[1], process=False)
    cut.merge_vertices()
    cut.remove_unreferenced_vertices()
    if colors is not None and len(cut.vertices):
        cut.visual.vertex_colors = colors[cKDTree(mesh.vertices).query(cut.vertices)[1]]
    if not close or len(cut.faces) == 0:
        return cut, 0
    e1, e2 = _basis(up)
    V = cut.vertices
    on_plane = np.abs((V - origin) @ up) < 1e-6
    loops = _boundary_loops(cut.faces, on_plane)
    polys = [np.stack([(V[lp] - origin) @ e1, (V[lp] - origin) @ e2], 1) for lp in loops]
    new = []
    for lp, xy in zip(loops, polys):
        # skip holes: outlines whose first vertex lies inside another outline
        if any(_inside(xy[0], other) for other in polys if other is not xy and len(other) > len(xy)):
            continue
        # the cap runs against the existing faces' boundary direction
        poly = np.array(lp[::-1])
        pxy = xy[::-1]
        area = 0.5 * np.sum(pxy[:, 0] * np.roll(pxy[:, 1], -1) - np.roll(pxy[:, 0], -1) * pxy[:, 1])
        if area > 0:
            tris = [poly[list(t)] for t in _ear_clip(pxy)]
        else:
            rpoly = poly[::-1]  # triangulate counter-clockwise, then flip back
            tris = [rpoly[list(t)][::-1] for t in _ear_clip(pxy[::-1].copy())]
        new += tris
    if not new:
        return cut, 0
    faces = np.vstack([cut.faces, np.array(new)])
    out = trimesh.Trimesh(cut.vertices, faces, vertex_colors=cut.visual.vertex_colors, process=False)
    return out, len(new)


def _inside(p: np.ndarray, poly: np.ndarray) -> bool:
    """Even-odd point-in-polygon test."""
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = ((y > p[1]) != (yn > p[1])) & (p[0] < (xn - x) * (p[1] - y) / np.where(yn != y, yn - y, 1e-12) + x)
    return bool(np.count_nonzero(cross) % 2)
