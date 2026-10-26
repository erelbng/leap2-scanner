"""Render dense depth maps from a trained splat at every keyframe (left camera, refined pose).

    python splat/render_depth.py --data scans/<name>

Stereo depth from the Leap is sparse (dark / textureless IR), but the splat is fitted to every
pixel of every view, so its rendered depth is dense - useful for experiments, e.g. fusing it
into a mesh yourself. Writes <data>/splat/depth_splat/kf_XXXXX.png (uint16 mm, 0 where the splat
is transparent). Needs CUDA + gsplat.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import load_splat  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--min-alpha", type=float, default=0.95, help="drop pixels the splat doesn't cover well")
    args = ap.parse_args()

    dev = "cuda"
    sp = args.data / "splat"
    meta = json.loads((args.data / "keyframes.json").read_text())
    poses = np.array(json.loads((sp / "poses_refined.json").read_text())["T_wc_left"])
    K = torch.tensor(meta["K"], dtype=torch.float32, device=dev)[None]
    size = meta["size"]
    g = load_splat(sp / "point_cloud.ply", dev)
    out = sp / "depth_splat"
    out.mkdir(exist_ok=True)
    with torch.no_grad():
        for kf, T in zip(meta["keyframes"], poses):
            view = torch.tensor(np.linalg.inv(T), dtype=torch.float32, device=dev)[None]
            img, alpha, _ = rasterization(g["means"], g["quats"], g["scales"], g["opacities"], g["colors"], view, K,
                                          size, size, sh_degree=g["sh_degree"], render_mode="RGB+ED",
                                          near_plane=0.02)
            d = img[0, ..., 3].cpu().numpy()
            a = alpha[0, ..., 0].cpu().numpy()
            d[a < args.min_alpha] = 0
            cv2.imwrite(str(out / ("kf_%05d.png" % kf["idx"])), np.clip(np.round(d * 1000), 0, 65535).astype(np.uint16))
    print("wrote %d depth maps to %s" % (len(poses), out))


if __name__ == "__main__":
    main()
