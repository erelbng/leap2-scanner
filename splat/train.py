"""Train a 3D Gaussian splat from a leap2-scanner scan folder (needs CUDA + gsplat >= 1.3).

    python splat/train.py --data scans/20261024_090814

Reads the scan's native format (keyframes.json, images/, depth/, points3D.ply), so no
COLMAP run is needed. On top of plain 3DGS it adds what this IR stereo rig benefits from:
  * depth supervision from the stereo depth maps (left images)
  * pose refinement: one SE(3) correction per keyframe, shared by the rigid left/right pair
  * per-image exposure compensation (IR auto exposure, left/right sensors differ)
Writes <data>/splat/: point_cloud.ply (standard 3DGS format), renders/, poses_refined.json, train.log
Python >= 3.8. gsplat JIT-compiles its CUDA kernels on first use (needs ninja on PATH).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

C0 = 0.28209479177387814  # SH band 0


# ---------------------------------------------------------------------- data
def read_ply_points(path: Path):
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode()
    n = int(header.split("element vertex ")[1].split("\n")[0])
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    a = np.frombuffer(data[end:], dt, count=n)
    pts = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float32)
    rgb = np.stack([a["r"], a["g"], a["b"]], 1).astype(np.float32) / 255.0
    return pts, rgb


class Scan:
    def __init__(self, root: Path, downscale: int, device: str):
        meta = json.loads((root / "keyframes.json").read_text())
        K = np.array(meta["K"], np.float64)
        size = meta["size"]
        self.baseline = meta["baseline"]
        self.W = self.H = size // downscale
        K[:2] /= downscale
        self.K = torch.tensor(K, dtype=torch.float32, device=device)
        kfs = meta["keyframes"]
        self.n_kf = len(kfs)
        imgs, depths, c2w, kf_index, is_right, names = [], [], [], [], [], []
        T_lr = np.eye(4)
        T_lr[0, 3] = self.baseline
        # multi-view filtered depth (written by `leap2-scanner export`) is far cleaner than raw stereo depth
        depth_dir = root / "depth_filtered" if (root / "depth_filtered").exists() else root / "depth"
        self.depth_dir = depth_dir.name
        for i, kf in enumerate(kfs):
            T = np.array(kf["T_wc"], np.float64)
            for side in ("L", "R"):
                name = "kf_%05d_%s" % (kf["idx"], side)
                im = cv2.imread(str(root / "images" / (name + ".jpg")), cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                if downscale > 1:
                    im = cv2.resize(im, (self.W, self.H), interpolation=cv2.INTER_AREA)
                d = None
                if side == "L":
                    d = cv2.imread(str(depth_dir / ("kf_%05d.png" % kf["idx"])), cv2.IMREAD_UNCHANGED)
                    if d is not None:
                        d = d.astype(np.float32) / 1000.0
                        if downscale > 1:
                            d = cv2.resize(d, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                imgs.append(torch.from_numpy(im))
                depths.append(torch.from_numpy(d).half() if d is not None else None)
                c2w.append(T if side == "L" else T @ T_lr)
                kf_index.append(i)
                is_right.append(side == "R")
                names.append(name)
        self.images = torch.stack(imgs).to(device)  # uint8 (N,H,W), grey IR
        self.depths = depths
        self.c2w = torch.tensor(np.stack(c2w), dtype=torch.float32, device=device)
        self.kf_index = torch.tensor(kf_index, device=device)
        self.is_right = is_right
        self.names = names
        self.T_lr = torch.tensor(T_lr, dtype=torch.float32, device=device)
        self.device = device
        # valid region of the rectified image (outside the lens circle is black)
        self.valid = (self.images.float().mean(0) > 2).float()

    def __len__(self):
        return len(self.images)

    def image(self, i):
        return self.images[i].float().div(255.0)[..., None].expand(-1, -1, 3)

    def depth(self, i):
        d = self.depths[i]
        return None if d is None else d.to(self.device).float()


# ---------------------------------------------------------------------- model helpers
def knn_dist(pts: np.ndarray, k: int = 3) -> np.ndarray:
    from scipy.spatial import cKDTree

    d, _ = cKDTree(pts).query(pts, k=k + 1)
    return np.sqrt((d[:, 1:] ** 2).mean(1))


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], -2)


class PoseOpt(torch.nn.Module):
    """Per-keyframe SE(3) correction (zero-initialised), applied to the left camera; right follows rigidly."""

    def __init__(self, n: int):
        super().__init__()
        self.embed = torch.nn.Embedding(n, 9)
        torch.nn.init.zeros_(self.embed.weight)
        self.register_buffer("identity", torch.tensor([1.0, 0, 0, 0, 1, 0]))

    def forward(self, c2w_left: torch.Tensor, kf: torch.Tensor) -> torch.Tensor:
        p = self.embed(kf)
        R = rot6d_to_matrix(p[..., 3:] + self.identity)
        D = torch.eye(4, device=c2w_left.device).repeat(len(kf), 1, 1)
        D[:, :3, :3] = R
        D[:, :3, 3] = p[..., :3]
        return c2w_left @ D


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: (1,C,H,W) in [0,1]."""
    C1, C2 = 0.01**2, 0.03**2
    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5**2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).expand(a.shape[1], 1, 11, 11)
    f = lambda x: F.conv2d(x, w, padding=5, groups=a.shape[1])
    mu_a, mu_b = f(a), f(b)
    s_a = f(a * a) - mu_a**2
    s_b = f(b * b) - mu_b**2
    s_ab = f(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (s_a + s_b + C2))
    return m.mean()


def save_ply(path: Path, params) -> None:
    means = params["means"].detach().cpu().numpy()
    scales = params["scales"].detach().cpu().numpy()
    quats = F.normalize(params["quats"].detach(), dim=-1).cpu().numpy()
    opac = params["opacities"].detach().cpu().numpy()[:, None]
    sh0 = params["sh0"].detach().cpu().numpy().reshape(len(means), -1)
    shN = params["shN"].detach().transpose(1, 2).cpu().numpy().reshape(len(means), -1)  # channel-major like 3DGS
    attrs = ["x", "y", "z", "nx", "ny", "nz"] + ["f_dc_%d" % i for i in range(sh0.shape[1])]
    attrs += ["f_rest_%d" % i for i in range(shN.shape[1])] + ["opacity"]
    attrs += ["scale_%d" % i for i in range(3)] + ["rot_%d" % i for i in range(4)]
    data = np.concatenate([means, np.zeros_like(means), sh0, shN, opac, scales, quats], 1).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(means)
    header += "".join("property float %s\n" % a for a in attrs) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


# ---------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="scan folder")
    ap.add_argument("--out", type=Path, default=None, help="default: <data>/splat")
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--depth-weight", type=float, default=0.1, help="0 disables depth supervision")
    ap.add_argument("--no-pose-opt", action="store_true")
    ap.add_argument("--no-exposure-opt", action="store_true")
    ap.add_argument("--test-every", type=int, default=10, help="hold out every n-th keyframe (0 = none)")
    ap.add_argument("--max-init", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    out = args.out or (args.data / "splat")
    (out / "renders").mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    data = Scan(args.data, args.downscale, dev)
    kf_np = data.kf_index.cpu().numpy()
    test = np.array([args.test_every > 0 and k % args.test_every == args.test_every // 2 for k in kf_np])
    train_ids = np.nonzero(~test)[0]
    test_ids = np.nonzero(test)[0]
    log("%d images (%dx%d), %d keyframes, %d train / %d test, depth from %s/"
        % (len(data), data.W, data.H, data.n_kf, len(train_ids), len(test_ids), data.depth_dir))

    # ---- init gaussians from the scan's point cloud
    ply = args.data / "points3D.ply"
    if not ply.exists():
        ply = args.data / "pointcloud.ply"
    pts, rgb = read_ply_points(ply)
    if len(pts) > args.max_init:
        sel = np.random.choice(len(pts), args.max_init, replace=False)
        pts, rgb = pts[sel], rgb[sel]
    centers = data.c2w[:, :3, 3]
    scene_scale = float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1
    scene_scale = max(scene_scale, 0.1)
    dist = np.clip(knn_dist(pts), 1e-4, None)
    n = len(pts)
    sh_dim = (args.sh_degree + 1) ** 2
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(pts, device=dev)),
        "scales": torch.nn.Parameter(torch.log(torch.tensor(dist, dtype=torch.float32, device=dev))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(F.normalize(torch.randn(n, 4, device=dev), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(((torch.tensor(rgb, device=dev) - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, sh_dim - 1, 3, device=dev)),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15)
                  for k in params}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / args.iters))

    strategy = DefaultStrategy(
        verbose=False,
        refine_start_iter=500,
        refine_stop_iter=int(args.iters * 0.6),
        reset_every=3000,
        refine_every=100,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    pose_opt = PoseOpt(data.n_kf).to(dev)
    # each keyframe gets only ~100 updates, so the lr must allow cm-level corrections
    pose_optim = torch.optim.Adam(pose_opt.parameters(), lr=1e-4, weight_decay=1e-6)
    expo = torch.nn.Embedding(len(data), 2).to(dev)  # per image: log gain, bias
    torch.nn.init.zeros_(expo.weight)
    expo_optim = torch.optim.Adam(expo.parameters(), lr=1e-3)
    log("init %d gaussians, scene scale %.3f m, pose_opt=%s exposure_opt=%s depth_w=%.3f"
        % (n, scene_scale, not args.no_pose_opt, not args.no_exposure_opt, args.depth_weight))

    def camera(i: int, train: bool):
        kf = data.kf_index[i:i + 1]
        c2w_left = data.c2w[i:i + 1] if not data.is_right[i] else data.c2w[i:i + 1] @ torch.inverse(data.T_lr)
        if not args.no_pose_opt and train:
            c2w_left = pose_opt(c2w_left, kf)
        elif not args.no_pose_opt:
            with torch.no_grad():
                c2w_left = pose_opt(c2w_left, kf)
        c2w = c2w_left @ data.T_lr if data.is_right[i] else c2w_left
        return torch.inverse(c2w)

    def render(i: int, sh_deg: int, train: bool):
        viewmat = camera(i, train)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        out_c, out_a, info = rasterization(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=colors, viewmats=viewmat,
            Ks=data.K[None], width=data.W, height=data.H, sh_degree=sh_deg,
            render_mode="RGB+ED", packed=False, near_plane=0.02, far_plane=20.0,
        )
        rgb_out = out_c[0, ..., :3]
        depth_out = out_c[0, ..., 3]
        if not args.no_exposure_opt:
            g = expo(torch.tensor([i], device=dev))[0]
            rgb_out = rgb_out * torch.exp(g[0]) + g[1]
        return rgb_out, depth_out, out_a[0, ..., 0], info

    t0 = time.time()
    order = np.random.permutation(train_ids)
    pos = 0
    for step in range(args.iters):
        if pos >= len(order):
            order = np.random.permutation(train_ids)
            pos = 0
        i = int(order[pos])
        pos += 1
        sh_deg = min(step // 1000, args.sh_degree)
        rgb_out, depth_out, alpha, info = render(i, sh_deg, train=True)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        gt = data.image(i)
        m = data.valid[..., None]
        l1 = (torch.abs(rgb_out - gt) * m).mean()
        ssim_l = 1 - ssim((rgb_out * m).permute(2, 0, 1)[None], (gt * m).permute(2, 0, 1)[None])
        loss = 0.8 * l1 + 0.2 * ssim_l
        dgt = data.depth(i)
        dl = torch.zeros((), device=dev)
        if args.depth_weight > 0 and dgt is not None:
            v = dgt > 0
            if v.any():
                dw = args.depth_weight * max(0.1, 1 - step / args.iters)
                dl = (torch.abs(depth_out[v] - dgt[v]) / dgt[v]).mean()
                loss = loss + dw * dl
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        if not args.no_pose_opt and step > 300:
            pose_optim.step()
        pose_optim.zero_grad(set_to_none=True)
        if not args.no_exposure_opt:
            expo_optim.step()
        expo_optim.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)

        if step % 200 == 0 or step == args.iters - 1:
            with torch.no_grad():
                psnr = -10 * torch.log10(F.mse_loss(rgb_out.clamp(0, 1) * m, gt * m))
            log("step %5d  loss %.4f  l1 %.4f  depth %.4f  psnr %.2f  gaussians %d  %.1f it/s"
                % (step, loss.item(), l1.item(), dl.item(), psnr.item(), len(params["means"]),
                   (step + 1) / (time.time() - t0)))

    # ---- evaluation + outputs
    with torch.no_grad():
        psnrs = []
        for j, i in enumerate(test_ids):
            rgb_out, depth_out, _, _ = render(int(i), args.sh_degree, train=False)
            gt = data.image(int(i))
            psnrs.append(float(-10 * torch.log10(F.mse_loss(rgb_out.clamp(0, 1), gt))))
            if j < 12:
                vis = np.hstack([(gt[..., 0].cpu().numpy() * 255), (rgb_out[..., 0].clamp(0, 1).cpu().numpy() * 255)])
                cv2.imwrite(str(out / "renders" / ("test_%s.jpg" % data.names[int(i)])), vis.astype(np.uint8))
        if psnrs:
            log("held-out PSNR %.2f dB over %d images (no pose refinement on test views)" % (np.mean(psnrs), len(psnrs)))
        save_ply(out / "point_cloud.ply", params)
        refined = []
        for k in range(data.n_kf):
            i = int(np.nonzero(kf_np == k)[0][0])
            refined.append(torch.inverse(camera(i, train=False))[0].cpu().numpy().tolist())
        (out / "poses_refined.json").write_text(json.dumps({"T_wc_left": refined}))
        # fly-through along the (refined) keyframe path
        vw = cv2.VideoWriter(str(out / "renders" / "trajectory.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15,
                             (2 * data.W, data.H), False)
        for i in range(0, len(data), 2):
            rgb_out, _, _, _ = render(i, args.sh_degree, train=False)
            gt = data.image(i)
            frame = np.hstack([gt[..., 0].cpu().numpy(), rgb_out[..., 0].clamp(0, 1).cpu().numpy()])
            vw.write((frame * 255).astype(np.uint8))
        vw.release()
    log("saved %s (%d gaussians), renders/, poses_refined.json in %.1f min"
        % (out / "point_cloud.ply", len(params["means"]), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
