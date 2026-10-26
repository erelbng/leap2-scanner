"""Diagnostics and calibration refinement."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .calib import CONFIG_DIR, FisheyeCam, Rectifier, StereoCalib, load_device_calibration
from .device import LeapCamera, device_serial, find_leap_device
from .stereo import DepthEstimator, colorize_depth


def dense_matches(rect: Rectifier, L: np.ndarray, R: np.ndarray, max_pts: int = 4000, seed: int = 0,
                  rectified: bool = False):
    """Stereo correspondences in raw pixel coordinates (left, right) plus their rectified dy.

    With `rectified=True`, L/R are already rectified by `rect` (e.g. images saved in a scan)."""
    l, r = (L, R) if rectified else rect.rectify(L, R)
    cl = cv2.createCLAHE(3.0, (8, 8))
    l2, r2 = cl.apply(l), cl.apply(r)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fl = dis.calc(l2, r2, None)
    bk = dis.calc(r2, l2, None)
    s = rect.size
    xx, yy = np.meshgrid(np.arange(s, dtype=np.float32), np.arange(s, dtype=np.float32))
    bx = cv2.remap(bk, xx + fl[..., 0], yy + fl[..., 1], cv2.INTER_LINEAR)
    cons = np.hypot(fl[..., 0] + bx[..., 0], fl[..., 1] + bx[..., 1]) < 0.3
    grad = np.hypot(cv2.Sobel(l2, cv2.CV_32F, 1, 0), cv2.Sobel(l2, cv2.CV_32F, 0, 1))
    m = cons & (grad > 60) & (l > 15) & (r > 15) & (l < 250) & (r < 250) & (fl[..., 0] < 0.5)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    # spread samples evenly over the image (16x16 cells)
    cell = (ys * 16 // s) * 16 + xs * 16 // s
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xs))
    per = max(1, max_pts // 128)
    keep = []
    counts = {}
    for i in order:
        c = cell[i]
        if counts.get(c, 0) < per:
            counts[c] = counts.get(c, 0) + 1
            keep.append(i)
    keep = np.array(keep)
    ys, xs = ys[keep], xs[keep]
    mapL, mapR = rect.float_maps
    pl = mapL[ys, xs]
    xr = (xs + fl[ys, xs, 0]).astype(np.float32)[None]
    yr = (ys + fl[ys, xs, 1]).astype(np.float32)[None]
    pr = np.stack([cv2.remap(mapR[..., 0], xr, yr, cv2.INTER_LINEAR)[0],
                   cv2.remap(mapR[..., 1], xr, yr, cv2.INTER_LINEAR)[0]], 1)
    ok = (pr[:, 0] > 0) & (pr[:, 1] > 0)
    return pl[ok].astype(np.float64), pr[ok].astype(np.float64), fl[ys, xs, 1][ok]


def epipolar_px(vl, vr, R, t, f):
    t = t / np.linalg.norm(t)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    n = (tx @ R @ vl.T).T
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    vr = vr / np.linalg.norm(vr, axis=1, keepdims=True)
    return np.arcsin(np.clip(np.sum(n * vr, 1), -1, 1)) * f


def check(args):
    dev = find_leap_device()
    print(f"device {dev}  serial {device_serial(dev)}")
    calib = load_device_calibration(dev, args.model, not args.factory_calib)
    c = calib.for_resolution(args.res)
    print(f"calibration source: {calib.source}, model {c.left.model}")
    for name, cam in (("left", c.left), ("right", c.right)):
        print(f"  {name:5s} f=({cam.fx:.2f},{cam.fy:.2f}) c=({cam.cx:.2f},{cam.cy:.2f}) d={np.round(cam.d, 5)}")
    print(f"  baseline {c.baseline*1000:.2f} mm, R(deg)={np.degrees(cv2.Rodrigues(c.R)[0].ravel()).round(3)}")
    cam = LeapCamera(dev, args.res)
    try:
        time.sleep(1.5)
        print(f"camera {cam.fps:.1f} fps at {args.res}x{args.res} per eye")
        _, _, L, R = cam.read()
    finally:
        cam.close()
    rect = Rectifier(c, fov_deg=args.fov)
    de = DepthEstimator(rect, args.zmin, args.zmax, mode=args.stereo, fill=getattr(args, "fill", False))
    t0 = time.time()
    l, r = rect.rectify(L, R)
    z = de.compute(l, r)
    print(f"rectified {rect.size}px f={rect.f:.1f}; depth {1000*(time.time()-t0):.0f} ms, "
          f"valid {100*(z>0).mean():.1f}%" + (f", median {np.median(z[z>0]):.3f} m" if (z > 0).any() else ""))
    pl, pr, dy = dense_matches(rect, L, R)
    if len(dy):
        print(f"stereo row alignment: median |dy| {np.median(np.abs(dy)):.2f}px over {len(dy)} matches "
              "(< ~0.7 good; point at a textured scene for a meaningful number)")
    if args.save:
        h = rect.size
        raw = cv2.resize(np.hstack([L, cv2.rotate(R, cv2.ROTATE_180)]), (2 * h, h))
        rows = np.vstack([cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR),
                          np.hstack([cv2.cvtColor(l, cv2.COLOR_GRAY2BGR), colorize_depth(z, de.z_min, de.z_max)])])
        cv2.imwrite(str(args.save), rows)
        print(f"preview written to {args.save}")


def _fit_refinement(c0: StereoCalib, PL: np.ndarray, PR: np.ndarray, FR: np.ndarray, max_train: int = 80000,
                    intrinsics: bool = False):
    """Robust fit of small corrections - principal points, right/left focal ratio, 2 lens terms per
    camera, relative rotation, baseline direction - to stereo correspondences in raw pixels.
    Left focal length and baseline length stay at the factory values (they set the metric scale).
    Default: extrinsics only (relative rotation + baseline direction). On real scans this removed
    most of the row misalignment with sane values (~0.2-0.6 deg); also fitting intrinsics gained
    only ~0.03 px while driving them to their bounds - with a cropped field of view the epipolar
    error alone is satisfied by physically absurd intrinsics, hence the bounds.
    Returns (refined calib at c0's resolution, held-out error before, after, parameters)."""
    from scipy.optimize import least_squares

    test = (FR % 4) == 3  # hold out every 4th frame
    rng = np.random.default_rng(0)
    train_idx = np.nonzero(~test)[0]
    if len(train_idx) > max_train:
        train_idx = rng.choice(train_idx, max_train, replace=False)
    test_idx = np.nonzero(test)[0]
    if len(test_idx) > max_train:
        test_idx = rng.choice(test_idx, max_train, replace=False)
    f = c0.left.fx
    r0 = np.linalg.norm(c0.t)
    tdir = c0.t / r0
    e1 = np.cross(tdir, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(tdir, e1)

    def cam(c: FisheyeCam, dcx, dcy, d0, d1, scale):
        # keep the base lens model; equidistant becomes kb5 so the two extra terms can act
        model = "kb5" if c.model == "equidistant" else c.model
        return FisheyeCam(c.fx * scale, c.fy * scale, c.cx + dcx, c.cy + dcy, c.d + np.r_[d0, d1, 0, 0, 0], model)

    def build(x):
        L = cam(c0.left, x[0], x[1], x[2], x[3], 1.0)
        R = cam(c0.right, x[4], x[5], x[6], x[7], 1 + x[13])
        Rm = cv2.Rodrigues(x[8:11])[0] @ c0.R
        t = tdir + x[11] * e1 + x[12] * e2
        return L, R, Rm, r0 * t / np.linalg.norm(t)

    def resid(x, idx):
        L, R, Rm, t = build(x)
        return epipolar_px(L.unproject(PL[idx]), R.unproject(PR[idx]), Rm, t, f)

    px = 12.0 * c0.res / 1024  # centre shifts
    lim = np.r_[[px, px], [0.08, 0.08], [px, px], [0.08, 0.08], [np.radians(2.0)] * 3, [0.1, 0.1], [0.02]]
    free = np.arange(14) if intrinsics else np.arange(8, 13)

    def full(xf):
        x = np.zeros(14)
        x[free] = xf
        return x

    before = np.abs(resid(np.zeros(14), test_idx))
    sol = least_squares(lambda xf: resid(full(xf), train_idx), np.zeros(len(free)),
                        bounds=(-lim[free], lim[free]), loss="soft_l1", f_scale=1.0, x_scale="jac", method="trf")
    after = np.abs(resid(full(sol.x), test_idx))
    at_limit = np.abs(sol.x) > 0.98 * lim[free]
    sol.x = full(sol.x)
    if at_limit.any():
        print(f"  note: {int(at_limit.sum())} parameter(s) at their limit - treat the result with care")
    L, R, Rm, t = build(sol.x)
    return StereoCalib(L, R, Rm, t, c0.res, "refined"), before, after, sol.x


def _report_and_save(base: StereoCalib, res: int, refined_r: StereoCalib, before, after, x, serial: str):
    print(f"held-out stereo row misalignment: median {np.median(before):.2f} -> {np.median(after):.2f} px, "
          f"p90 {np.percentile(before, 90):.2f} -> {np.percentile(after, 90):.2f} px")
    print(f"  centre shift L ({x[0]:+.2f},{x[1]:+.2f}) R ({x[4]:+.2f},{x[5]:+.2f}) px, right/left focal "
          f"{1+x[13]:.4f}, lens terms L ({x[2]:+.4f},{x[3]:+.4f}) R ({x[6]:+.4f},{x[7]:+.4f}), "
          f"rotation fix {np.degrees(x[8:11]).round(3)} deg")
    if np.median(after) >= np.median(before) * 0.9:
        print("No meaningful improvement - keeping the current calibration.")
        return None
    k = base.res / res
    refined = StereoCalib(refined_r.left.scaled(k), refined_r.right.scaled(k), refined_r.R, refined_r.t, base.res,
                          "refined")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFIG_DIR / f"{serial}_refined.json"
    if out.exists():  # keep the previous refinement instead of overwriting it
        out.rename(out.with_name(out.stem + time.strftime("_%Y%m%d_%H%M%S") + ".json"))
    refined.save(out)
    print(f"saved refined calibration to {out} (used automatically from now on; --factory-calib to ignore)")
    return refined


def _serial_offline() -> str:
    found = sorted(CONFIG_DIR.glob("*_factory.bin"))
    if len(found) != 1:
        raise SystemExit("Plug in the Leap (or keep exactly one *_factory.bin in the config folder) "
                         "so the refined calibration can be stored for the right device.")
    return found[0].name[: -len("_factory.bin")]


def calibrate(args):
    if args.from_scan:
        return calibrate_from_scans(args)
    dev = find_leap_device()
    base = load_device_calibration(dev, args.model, use_refined=False)
    c0 = base.for_resolution(args.res)
    rect = Rectifier(c0, fov_deg=140)
    cam = LeapCamera(dev, args.res)
    print(f"Move the Leap slowly around textured things at 20-100 cm for {args.seconds:.0f} s "
          "(cover the whole field of view, incl. edges)...")
    frames = []
    try:
        time.sleep(1.0)
        seq, t_end = -1, time.time() + args.seconds
        while time.time() < t_end:
            seq, _, L, R = cam.read(seq)
            frames.append((L, R))
            time.sleep(0.4)
    finally:
        cam.close()
    PL, PR, FR = [], [], []
    for i, (L, R) in enumerate(frames):
        pl, pr, _ = dense_matches(rect, L, R, 2500, seed=i)
        PL.append(pl)
        PR.append(pr)
        FR.append(np.full(len(pl), i))
    PL, PR, FR = np.vstack(PL), np.vstack(PR), np.concatenate(FR)
    print(f"{len(PL)} correspondences from {len(frames)} frames")
    if len(PL) < 3000:
        print("Not enough texture seen - try again with a busier scene.")
        return
    refined_r, before, after, x = _fit_refinement(c0, PL, PR, FR, intrinsics=args.intrinsics)
    _report_and_save(base, args.res, refined_r, before, after, x, device_serial(dev))


def calibrate_from_scans(args):
    """Refine the calibration from saved scans: their rectified image pairs are matched densely and the
    matches mapped back to raw sensor pixels through the calibration each scan was recorded with."""
    import json

    PL, PR, FR = [], [], []
    base_res = None
    frame_id = 0
    for scan in args.from_scan:
        scan = Path(scan)
        calib = StereoCalib.load(scan / "calibration.json")  # how this scan was rectified
        meta = json.loads((scan / "keyframes.json").read_text())
        res = next((r for r in (1024, 512) if Rectifier(calib.for_resolution(r), fov_deg=100).size == meta["size"]),
                   None)
        if res is None:
            print(f"{scan}: unknown rectification (size {meta['size']}), skipped")
            continue
        if base_res is None:
            base_res = res
        rect = Rectifier(calib.for_resolution(res), fov_deg=100)
        kfs = meta["keyframes"][:: args.every]
        for k in kfs:
            name = "kf_%05d" % k["idx"]
            L = cv2.imread(str(scan / "images" / (name + "_L.jpg")), cv2.IMREAD_GRAYSCALE)
            R = cv2.imread(str(scan / "images" / (name + "_R.jpg")), cv2.IMREAD_GRAYSCALE)
            if L is None or R is None:
                continue
            pl, pr, _ = dense_matches(rect, L, R, 3000, seed=frame_id, rectified=True)
            PL.append(pl)
            PR.append(pr)
            FR.append(np.full(len(pl), frame_id))
            frame_id += 1
        print(f"{scan}: {len(kfs)} keyframes")
    if not PL:
        raise SystemExit("no usable scans")
    PL, PR, FR = np.vstack(PL), np.vstack(PR), np.concatenate(FR)
    print(f"{len(PL)} correspondences from {frame_id} keyframes")
    # the matches are raw sensor pixels, so the refinement can start from the factory calibration
    # with any lens model, independent of how the scans were rectified
    serial = args.serial or _serial_offline()
    factory = CONFIG_DIR / f"{serial}_factory.bin"
    base = StereoCalib.from_blob(factory.read_bytes(), args.model) if factory.exists() else calib
    c0 = base.for_resolution(base_res)
    refined_r, before, after, x = _fit_refinement(c0, PL, PR, FR, intrinsics=args.intrinsics)
    _report_and_save(base, base_res, refined_r, before, after, x, serial)
