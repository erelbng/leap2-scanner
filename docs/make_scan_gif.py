"""Make the README "how the scan was done" animation from a saved scan.

    python docs/make_scan_gif.py scans/<name> docs/img/scan.gif [--size 260] [--fps 8] [--every 2]

Three panels per keyframe: IR image, stereo depth, and the map growing along the camera path
(seen from a fixed overview camera). "Up" is taken from the dominant plane (usually the table)
facing the cameras, so scans made with the Leap held upside down still come out upright.
Needs ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leap2_scanner.export import Session  # noqa: E402
from leap2_scanner.fusion import backproject, estimate_up  # noqa: E402
from leap2_scanner.stereo import colorize_depth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=260, help="panel size in pixels")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--every", type=int, default=1, help="render every n-th keyframe (smaller GIF for long scans)")
    args = ap.parse_args()

    s = Session.load(args.scan)
    K = s.K
    W = args.size
    T = [r.T_wc for r in s.records]
    C = np.array([t[:3, 3] for t in T])

    # per keyframe: world points of the depth used for the map (filtered if available)
    pts_list = []
    for r in s.records:
        left, d = s.load_frame(r)
        f = args.scan / "depth_filtered" / (s.name(r.idx) + ".png")
        if f.exists():
            d = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        X, u, v = backproject(d, K, stride=3)
        pts_list.append((X @ r.T_wc[:3, :3].T + r.T_wc[:3, 3], left[v, u]))
    allp = np.vstack([p for p, _ in pts_list])

    up, flip = estimate_up(allp, T)  # Leap held upside down -> rotate the image panels
    print("images %s" % ("rotated 180°" if flip else "as is"))

    fwd = np.mean([t[:3, 2] for t in T], 0)
    fwd -= up * (fwd @ up)
    fwd /= np.linalg.norm(fwd)
    target = np.median(allp, 0)
    eye = target - fwd * 1.0 + up * 0.8
    zc = target - eye
    zc /= np.linalg.norm(zc)
    xc = np.cross(zc, up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R_ov = np.stack([xc, yc, zc])
    Q = (np.vstack([allp, C]) - eye) @ R_ov.T
    Q = Q[Q[:, 2] > 0.05]
    f_ov = 0.46 * W / np.percentile(np.abs(Q[:, :2] / Q[:, 2:3]), 97, axis=0).max()

    def proj(P):
        Q = (P - eye) @ R_ov.T
        return Q[:, 0] / Q[:, 2] * f_ov + W / 2, Q[:, 1] / Q[:, 2] * f_ov + W / 2, Q[:, 2]

    def frustum_lines(Twc, sc=0.05):
        h = sc * np.tan(np.radians(50))
        c = [np.zeros(3), np.array([-h, -h, sc]), np.array([h, -h, sc]), np.array([h, h, sc]), np.array([-h, h, sc])]
        c = [Twc[:3, :3] @ p + Twc[:3, 3] for p in c]
        return [(c[0], c[i]) for i in range(1, 5)] + [(c[i], c[i % 4 + 1]) for i in range(1, 5)]

    font = cv2.FONT_HERSHEY_SIMPLEX
    frames = []
    acc_p, acc_g = np.empty((0, 3)), np.empty(0)
    t0 = s.records[0].stamp
    new_from = 0
    for i, r in enumerate(s.records):
        p, g = pts_list[i]
        acc_p = np.vstack([acc_p, p])
        acc_g = np.concatenate([acc_g, g])
        if i % args.every and i != len(s.records) - 1:
            continue  # the map still grows from every keyframe, only the frame is skipped
        p = acc_p[new_from:]
        new_from = len(acc_p)
        x, y, z = proj(acc_p)
        ok = (z > 0.05) & (x >= 0) & (y >= 0) & (x < W - 1) & (y < W - 1)
        o = np.argsort(-z[ok])
        xi, yi = x[ok][o].astype(int), y[ok][o].astype(int)
        shade = np.clip(acc_g[ok][o] * 0.8 + 40, 0, 255)
        new = np.zeros(len(acc_p), bool)
        if len(p):
            new[-len(p):] = True
        new = new[ok][o]
        col = np.stack([shade, shade, shade], 1)
        col[new] = np.stack([shade[new] * 0.4 + 40, shade[new] * 0.6 + 90, shade[new] * 0.3 + 160], 1)
        view = np.full((W, W, 3), 18, np.uint8)
        view[yi, xi] = col.astype(np.uint8)
        cx, cy, _ = proj(C[: i + 1])
        if i > 0:
            cv2.polylines(view, [np.stack([cx, cy], 1).astype(int).reshape(-1, 1, 2)], False, (0, 140, 255), 1, cv2.LINE_AA)
        for a, b in frustum_lines(r.T_wc):
            ax, ay, _ = proj(a[None])
            bx, by, _ = proj(b[None])
            cv2.line(view, (int(ax[0]), int(ay[0])), (int(bx[0]), int(by[0])), (0, 120, 255), 2, cv2.LINE_AA)

        left, d = s.load_frame(r)
        ir = cv2.cvtColor(cv2.resize(left, (W, W), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        dv = cv2.resize(colorize_depth(d, 0.15, 0.8), (W, W), interpolation=cv2.INTER_NEAREST)
        if flip:
            ir, dv = cv2.rotate(ir, cv2.ROTATE_180), cv2.rotate(dv, cv2.ROTATE_180)
        frame = np.hstack([ir, dv, view])
        for k, txt in enumerate(("IR camera (left)", "stereo depth", "map + camera path")):
            cv2.putText(frame, txt, (6 + k * W, 16), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "t=%4.1fs  keyframe %d" % (r.stamp - t0, i + 1), (2 * W + 6, W - 8), font, 0.38,
                    (200, 200, 200), 1, cv2.LINE_AA)
        frames.append(frame)
    frames += [frames[-1]] * (2 * args.fps)  # hold the final map

    with tempfile.TemporaryDirectory() as tmp:
        for k, fr in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, "%04d.png" % k), fr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i",
                        os.path.join(tmp, "%04d.png"), "-vf",
                        "split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
                        "[b][p]paletteuse=dither=none:diff_mode=rectangle", str(args.out)], check=True)
    print("wrote %s (%d frames, %.1f MB)" % (args.out, len(frames), args.out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
