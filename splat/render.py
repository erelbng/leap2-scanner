"""Render a trained splat to a video: a gentle orbit around the scanned scene, or the scan path.

    python splat/render.py --data scans/<name> [--mode orbit|path] [--size 640] [--gif out.gif]

Uses <data>/splat/point_cloud.ply and poses_refined.json (written by train.py). Needs CUDA + gsplat.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from a repo checkout
from leap2_scanner.fusion import estimate_up  # noqa: E402


def load_splat(path: Path, device: str):
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode().splitlines()
    n = int(next(l for l in header if l.startswith("element vertex")).split()[-1])
    names = [l.split()[-1] for l in header if l.startswith("property float")]
    a = np.frombuffer(data[end:], "<f4", count=n * len(names)).reshape(n, len(names))
    col = {k: i for i, k in enumerate(names)}
    t = lambda keys: torch.tensor(a[:, [col[k] for k in keys]].copy(), device=device)
    n_rest = len([k for k in names if k.startswith("f_rest_")])
    sh0 = t(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :]
    shN = t(["f_rest_%d" % i for i in range(n_rest)]).reshape(n, 3, n_rest // 3).transpose(1, 2)
    return dict(
        means=t(["x", "y", "z"]), quats=t(["rot_0", "rot_1", "rot_2", "rot_3"]),
        scales=torch.exp(t(["scale_0", "scale_1", "scale_2"])),
        opacities=torch.sigmoid(t(["opacity"])[:, 0]),
        colors=torch.cat([sh0, shN], 1), sh_degree=int(round(np.sqrt(n_rest // 3 + 1))) - 1,
    )


def look_at(eye, target, up):
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--mode", choices=["path", "orbit"], default="path",
                    help="path: smoothed scan path (robust); orbit: swing around the mean view (object scans)")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--orbit-deg", type=float, default=35.0, help="orbit swing (degrees)")
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--out", type=Path, default=None, help="default: <data>/splat/renders/<mode>.mp4")
    ap.add_argument("--gif", type=Path, default=None, help="also write a GIF (needs ffmpeg)")
    args = ap.parse_args()

    dev = "cuda"
    sp = args.data / "splat"
    g = load_splat(sp / "point_cloud.ply", dev)
    poses = np.array(json.loads((sp / "poses_refined.json").read_text())["T_wc_left"])
    C = poses[:, :3, 3]
    means = g["means"].cpu().numpy()
    solid = g["opacities"].cpu().numpy() > 0.5
    target = np.median(means[solid], 0)
    up, flipped = estimate_up(means[solid], list(poses))  # table normal; robust to an upside-down Leap
    print("up from %s" % ("dominant plane (scan was upside down)" if flipped else "scene/cameras"))

    if args.mode == "orbit":
        eye0 = C.mean(0)
        r = eye0 - target
        cams = []
        for k in range(args.frames):
            a = np.radians(args.orbit_deg) * np.sin(2 * np.pi * k / args.frames)
            b = 0.15 * np.sin(4 * np.pi * k / args.frames)  # small vertical bob
            Ra = cv2.Rodrigues(up * a)[0]
            eye = target + Ra @ r + up * b * np.linalg.norm(r)
            cams.append(look_at(eye, target, up))
    else:  # scan path, smoothed with a moving average and resampled
        box = np.ones(7) / 7
        smooth = lambda x: np.stack([np.convolve(np.pad(x[:, i], 3, mode="edge"), box, "valid") for i in range(3)], 1)
        Cs, Fs = smooth(C), smooth(poses[:, :3, 2])
        t = np.linspace(0, len(poses) - 1, args.frames)
        interp = lambda X: np.stack([np.interp(t, np.arange(len(X)), X[:, i]) for i in range(3)], 1)
        cams = [look_at(e, e + d, up) for e, d in zip(interp(Cs), interp(Fs))]

    W = H = args.size
    f = 0.5 * W / np.tan(np.radians(args.fov) / 2)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
    out = args.out or (sp / "renders" / ("%s.mp4" % args.mode))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(tempfile.mkdtemp(prefix="splat_render_"))
    with torch.no_grad():
        for idx, T in enumerate(cams):
            view = torch.tensor(np.linalg.inv(T), dtype=torch.float32, device=dev)[None]
            img, _, _ = rasterization(g["means"], g["quats"], g["scales"], g["opacities"], g["colors"], view, K,
                                      W, H, sh_degree=g["sh_degree"], near_plane=0.02)
            frame = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(frames_dir / ("%04d.png" % idx)), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(frames_dir / "%04d.png"),
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", str(out)], check=True)
    print("wrote", out)
    if args.gif:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "15", "-i", str(frames_dir / "%04d.png"),
                        "-vf", "scale=%d:-1,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=none"
                        % min(W, 480), str(args.gif)], check=True)
        print("wrote", args.gif)


if __name__ == "__main__":
    main()
