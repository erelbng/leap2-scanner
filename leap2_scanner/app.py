"""Live scanner: Leap 2 -> stereo depth -> visual odometry -> map, browser viewer (viser)."""

from __future__ import annotations

import queue
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import viser
import viser.transforms as vtf

from .calib import Rectifier, load_device_calibration
from .device import LeapCamera, find_leap_device
from .export import KeyframeRecord, Session, free_disk_mb
from .fusion import backproject, consistency_check, fuse_depth_window
from .pipeline import export_session
from .stereo import DepthEstimator, colorize_depth
from .tracking import StereoVO, inv_se3, rot_angle_deg

MIN_FREE_MB = 400  # stop writing keyframes below this much free disk space
MAX_DISPLAY_POINTS = 250_000
GAP_FRAMES = 30  # frames kept between keyframes for temporal depth fusion
DARK, BRIGHT = 25, 250  # IR intensity limits for the exposure statistics / overlay


class Scanner:
    def __init__(self, args):
        self.args = args
        dev = find_leap_device()
        self.calib = load_device_calibration(dev, args.model, not args.factory_calib)
        self.cam = LeapCamera(dev, args.res)
        self.rect = Rectifier(self.calib.for_resolution(args.res), fov_deg=args.fov)
        self.depth = DepthEstimator(self.rect, args.zmin, args.zmax, mode=args.stereo, fill=getattr(args, "fill", False))
        self.vo = StereoVO(self.rect.K, self.rect.size)
        self.out_root = Path(args.out)
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.state = "preview"  # preview | scanning | paused | exporting
        self.lock = threading.Lock()
        self.kf_queue: queue.Queue = queue.Queue()
        self.live: dict = {}
        self.proc_fps = 0.0
        self.message = "Press **Start scan**, then move slowly around the object."
        self.kf_dist = args.kf_dist
        self.kf_angle = args.kf_angle
        self.gain = 15
        self.cam.set_gain(self.gain)
        self.auto_gain = True
        self.detail_mode = False
        self._gain_t = 0.0
        self.quality = dict(depth=0.0, dark=0.0, bright=0.0)
        # pixels inside the lens circle (rectified pixels that map into the raw image)
        m = self.rect.float_maps[0]
        res = self.calib.for_resolution(args.res).res
        self.lens_mask = ((m[..., 0] >= 0) & (m[..., 1] >= 0) & (m[..., 0] < res) & (m[..., 1] < res))[::4, ::4]
        self.map_lock = threading.RLock()  # guards VO + map against start/reset from the UI thread
        self.export_lock = threading.Lock()
        self.cloud_lock = threading.Lock()
        self.generation = 0
        self._new_map()

        self.running = True
        self._seq, self._n, self._t0 = -1, 0, time.time()
        self._threads = [threading.Thread(target=self._guard(self._proc_step), daemon=True),
                         threading.Thread(target=self._guard(self._map_step), daemon=True)]
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------------ state
    def _new_map(self):
        from .fusion import VoxelCloud

        self.generation += 1  # keyframes still queued from an older map get dropped
        self.vo.reset()
        with self.cloud_lock:
            self.cloud = VoxelCloud(self.args.cloud_voxel_mm / 1000.0)
            self.kf_poses: list[np.ndarray] = []
        # last few keyframes (T, image, depth): a keyframe joins the live map only once its
        # neighbours confirm its depth, so stereo mismatches don't leave ghost edges
        self.recent: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        # tracked frames since the last keyframe [(T_wc, depth points)], and the keyframe that
        # waits for the frames after it before its depth is fused (temporal depth fusion)
        self.gap_frames: list[tuple[np.ndarray, np.ndarray]] = []
        self.pending: tuple | None = None
        self.kf_idx = 0
        self.last_kf_T: np.ndarray | None = None
        self.sharp_ema: float | None = None
        self.session: Session | None = None
        self.disk_full = False

    def _guard(self, step):
        """Run step() forever; an exception in one frame is logged instead of killing the thread."""
        def run():
            last_err = 0.0
            while self.running:
                try:
                    step()
                except Exception:
                    if time.time() - last_err > 5:
                        traceback.print_exc()
                        self.message = "Internal error, see terminal."
                    last_err = time.time()
                    time.sleep(0.05)
        return run

    def start(self):
        with self.map_lock:
            self._start()

    def _start(self):
        if self.state == "preview":
            self._drain_queue()
            self._new_map()
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session = Session(self.out_root / name, self.rect.K, self.rect.size, self.rect.baseline)
            self.calib.save(self.session.root / "calibration.json")
            self.message = f"Scanning into `{self.session.root}`"
        elif self.state == "paused":
            self.vo.lost = True  # device moved while paused: relocalize against keyframes
            self.message = "Resumed - point at an already scanned area to relocalize."
        self.state = "scanning"

    def pause(self):
        if self.state == "scanning":
            self.state = "paused"
            self.message = "Paused."

    def reset(self):
        if self.state == "exporting":
            return
        with self.map_lock:
            self.state = "preview"
            self._drain_queue()
            if self.session:
                self.session.save_index()
            self._new_map()
        self.message = "Map cleared (saved files are kept)."

    def _drain_queue(self):
        while True:
            try:
                self.kf_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------ threads
    def _proc_step(self):
        try:
            self._seq, stamp, L, R = self.cam.read(self._seq)
        except TimeoutError:
            self.message = "No frames from camera - replug the Leap?"
            return
        l, r = self.rect.rectify(L, R)
        z = self.depth.compute(l, r)
        self._update_quality(l, z)
        track = None
        tracks = np.empty((0, 2), np.float32)
        with self.map_lock:
            if self.state == "scanning":
                ln = self.depth.normalize(l)
                track = self.vo.track(ln, z)
                if track.ok and not self._maybe_keyframe(stamp, l, r, z, ln, track.T_wc):
                    X, _, _ = backproject(z, self.rect.K)
                    self.gap_frames = (self.gap_frames + [(track.T_wc, X.astype(np.float32))])[-GAP_FRAMES:]
                tracks = self.vo.p2d.copy()
        with self.lock:
            self.live = dict(l=l, z=z, track=track, stamp=stamp, tracks=tracks)
        self._n += 1
        if time.time() - self._t0 > 1.0:
            self.proc_fps = self._n / (time.time() - self._t0)
            self._n, self._t0 = 0, time.time()

    def _update_quality(self, l, z):
        """Exposure / depth statistics inside the lens circle, and auto gain."""
        m = self.lens_mask
        small = l[::4, ::4][m]
        self.quality = dict(depth=float((z[::4, ::4][m] > 0).mean()), dark=float((small < DARK).mean()),
                            bright=float((small >= BRIGHT).mean()))
        if self.auto_gain and self.state != "exporting" and time.time() - self._gain_t > 0.5:
            q, g = self.quality, self.gain
            if q["bright"] > 0.02 and g > 0:
                g -= 1  # blown-out areas carry no stereo or splat information
            elif q["bright"] < 0.005 and q["dark"] > 0.3 and g < 15:
                g += 1
            if g != self.gain:
                self.gain = g
                self.cam.set_gain(g)
                self._gain_t = time.time()

    def set_gain(self, g: int):
        self.gain = int(g)
        self.cam.set_gain(self.gain)

    def _maybe_keyframe(self, stamp, l, r, z, ln, T) -> bool:
        small = cv2.resize(ln, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        sharp = float(cv2.Laplacian(small, cv2.CV_32F).var())
        self.sharp_ema = sharp if self.sharp_ema is None else 0.9 * self.sharp_ema + 0.1 * sharp
        if self.last_kf_T is not None:
            d = inv_se3(self.last_kf_T) @ T
            motion = max(np.linalg.norm(d[:3, 3]) / self.kf_dist, rot_angle_deg(d[:3, :3]) / self.kf_angle)
            if motion < 1.0:
                return False
            if sharp < 0.6 * self.sharp_ema and motion < 2.0:
                return False  # likely motion blur; wait for a sharper frame
        if (z > 0).mean() < 0.005:
            return False
        self.vo.add_keyframe(ln, z, T)
        self.last_kf_T = T.copy()
        self.kf_queue.put((self.generation, (self.kf_idx, stamp, T.copy(), l, r, z, sharp, self.gap_frames,
                                             self.detail_mode)))
        self.gap_frames = []
        self.kf_idx += 1
        return True

    def _map_step(self):
        try:
            gen, item = self.kf_queue.get(timeout=0.2)
        except queue.Empty:
            return
        if gen != self.generation:
            return
        if item[0] == "flush":  # export / quit: fuse the last keyframe with the frames seen since
            if self.pending is not None:
                self._finalize(self.pending, item[1])
                self.pending = None
            return
        with self.cloud_lock:
            self.kf_poses.append(item[2])
        if self.pending is not None:
            self._finalize(self.pending, item[7])  # frames between pending and this keyframe
        self.pending = item

    def _finalize(self, item, frames_after):
        """Fuse a keyframe's depth from the frames around it, then add it to map and session."""
        idx, stamp, T, l, r, z, sharp, frames_before, detail = item
        if self.kf_queue.qsize() > 2:  # falling behind (fast motion): fuse fewer frames to catch up
            frames_before, frames_after = frames_before[-3:], frames_after[:3]
        z = fuse_depth_window(z, T, frames_before + frames_after, self.rect.K)
        self.recent = (self.recent + [(T, l, z)])[-4:]
        if len(self.recent) >= 2:
            # the previous keyframe now has neighbours on both sides: check and insert it
            Tc, lc, zc = self.recent[-2]
            others = [(d, Tn) for k, (Tn, _, d) in enumerate(self.recent) if k != len(self.recent) - 2]
            agree = consistency_check(zc, Tc, others, self.rect.K)
            zf = np.where(agree >= min(2, len(others)), zc, 0)
            pts, u, v = self.depth.backproject(zf, stride=2)
            pts_w = pts @ Tc[:3, :3].T.astype(np.float32) + Tc[:3, 3].astype(np.float32)
            with self.cloud_lock:
                self.cloud.add(pts_w, lc[v, u])
        s = self.session
        if s is None:
            return
        if free_disk_mb(s.root) < MIN_FREE_MB:
            if not self.disk_full:
                self.disk_full = True
                self.message = f"Disk almost full (<{MIN_FREE_MB} MB) - keyframes no longer saved. Export now."
            return
        s.add(KeyframeRecord(idx, stamp, T, sharp, detail), l, r, z)
        s.save_index()

    def backlog(self) -> int:
        """Keyframes captured but not yet fused / saved by the map thread."""
        return self.kf_queue.qsize() + (self.pending is not None)

    def flush(self, stall_timeout: float = 20.0):
        """Finalize every queued keyframe, including the last one (normally waiting for the frames
        after it). Waits as long as the map thread makes progress - after fast motion it can be
        many keyframes behind, and exporting before they are saved silently drops them."""
        self.kf_queue.put((self.generation, ("flush", list(self.gap_frames))))
        last, t_last = -1, time.time()
        while not self.kf_queue.empty() or self.pending is not None:
            left = self.backlog()
            if left != last:
                if last == -1 and left > 2:
                    print(f"finishing {left - 1} queued keyframes...", flush=True)
                last, t_last = left, time.time()
                self.message = f"Finishing {max(left - 1, 1)} queued keyframe(s)..."
            elif time.time() - t_last > stall_timeout:
                print("map thread stalled - continuing without the remaining keyframes", flush=True)
                break
            time.sleep(0.05)
        time.sleep(0.2)

    def export(self, voxel_mm: float, loop_closure: bool, progress, done):
        with self.export_lock:  # one export at a time (a second click used to race the first)
            if self.state == "exporting":
                self.message = "Export already running - please wait."
                return
            if self.session is None or (not self.session.records and self.pending is None):
                self.message = "Nothing to export yet - scan first."
                return
            self.state = "exporting"
        self.message = "Export: finishing the last keyframes..."

        def run():
            mesh = None
            try:
                self.flush()
                stl_mb = getattr(self.args, "stl_mb", 40)
                mesh = export_session(self.session, voxel_mm / 1000.0, progress, self._log,
                                      loop_closure=loop_closure, smooth=self.args.mesh_smooth,
                                      max_faces=int(stl_mb * 1e6 / 50) if stl_mb > 0 else None)
                self.message = (f"Export done: `{self.session.root}`  \n"
                                "Other settings or object meshes: `./scan.sh process`")
            except Exception as e:
                traceback.print_exc()
                self.message = f"Export failed: {e}"
            finally:
                self.state = "paused"
                done(mesh)  # always: hides the progress bar and re-enables the buttons

        threading.Thread(target=run, daemon=True).start()

    def _log(self, msg: str):
        print(msg, flush=True)
        self.message = msg

    def close(self):
        if self.session is not None and self.state != "exporting":
            self.pause()  # no new keyframes while the queue drains
            self.flush()  # don't lose queued keyframes
        self.running = False
        for t in self._threads:
            t.join(timeout=2)
        if self.session:
            self.session.save_index()
        self.cam.close()


# ---------------------------------------------------------------------- UI
def run_app(args):
    sc = Scanner(args)
    if args.autostart:
        sc.start()
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("-y")
    server.scene.world_axes.visible = False
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    fill = "on" if sc.depth.fill else ("off" if sc.depth.can_fill else "unavailable (needs opencv-contrib)")
    print(f"\nLeap scanner viewer: {url}   (stereo: {args.stereo}, hole fill: {fill})\n", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    @server.on_client_connect
    def _(client):
        # start just behind/above the scanner's first pose (world = first camera frame, y down)
        client.camera.position = (0.0, -0.25, -0.45)
        client.camera.look_at = (0.0, 0.0, 0.35)

    gui = server.gui
    with gui.add_folder("Scan"):
        status = gui.add_markdown("starting...")
        b_start = gui.add_button("Start scan", color="green", icon=None)
        b_pause = gui.add_button("Pause")
        b_export = gui.add_button("Finish & export (STL + splat data)", color="blue")
        b_reset = gui.add_button("New scan (clear map)", color="red")
        prog = gui.add_progress_bar(0.0, visible=False)
    with gui.add_folder("Live view"):
        img_h = gui.add_image(np.zeros((200, 400, 3), np.uint8), label="IR left | depth", format="jpeg")
        cb_flip = gui.add_checkbox("Rotate view 180°", False)
        cb_live = gui.add_checkbox("Show live depth points", True)
        cb_map = gui.add_checkbox("Show map", True)
        cb_follow = gui.add_checkbox("Follow camera", False)
        sl_pt = gui.add_slider("Point size (mm)", 0.5, 6.0, 0.5, 1.5)
    with gui.add_folder("Capture"):
        quality_md = gui.add_markdown("")
        cb_expo = gui.add_checkbox("Show over/under-exposure", True,
                                   hint="Red: blown out (no information). Blue: too dark for stereo.")
        cb_auto = gui.add_checkbox("Auto gain", True, hint="Lower the gain when parts blow out, raise it when dark")
        sl_gain = gui.add_slider("Sensor gain", 0, 15, 1, sc.gain)
        cb_detail = gui.add_checkbox("Detail mode (close-up)", False,
                                     hint="Go to 15-25 cm and move slowly: keyframes every 1.5 cm / 3°, sharper "
                                          "stereo, and the export adds a 1 mm mesh of this area (mesh_detail.stl)")
        cb_fill = gui.add_checkbox("Fill holes", sc.depth.fill, disabled=not sc.depth.can_fill,
                                   hint="Fill pixels stereo can't match from a smoothed (WLS) match: fewer holes, "
                                        "rougher surfaces, a bit slower" if sc.depth.can_fill else
                                        "Needs opencv-contrib-python-headless")
    with gui.add_folder("Settings", expand_by_default=False):
        sl_zmax = gui.add_slider("Max depth (m)", 0.25, 2.0, 0.05, float(args.zmax))
        sl_kfd = gui.add_slider("Keyframe every (cm)", 1.0, 10.0, 0.5, args.kf_dist * 100)
        sl_kfa = gui.add_slider("Keyframe every (deg)", 2.0, 20.0, 1.0, float(args.kf_angle))
        num_vox = gui.add_number("Mesh voxel (mm)", float(args.mesh_voxel_mm), min=0.5, max=20.0, step=0.5)
        cb_loop = gui.add_checkbox("Loop closure on export", True,
                                   hint="Drift-correct poses before meshing (~1-2 min per 100 keyframes)")
        cb_mesh = gui.add_checkbox("Show mesh", True)

    @b_start.on_click
    def _(_):
        sc.start()

    @b_pause.on_click
    def _(_):
        sc.pause()

    clear_scene = threading.Event()  # the render loop owns the scene handles and does the removal

    @b_reset.on_click
    def _(_):
        sc.reset()
        clear_scene.set()

    def on_progress(p):
        prog.visible = True
        prog.value = float(np.clip(p * 100, 0, 100))

    def busy(on: bool):
        for b in (b_start, b_pause, b_export, b_reset):
            b.disabled = on

    mesh_handle = {"h": None}

    def on_done(mesh):
        prog.visible = False
        busy(False)
        if mesh is not None and len(mesh.faces):
            mesh_handle["h"] = server.scene.add_mesh_trimesh("/mesh", mesh, visible=cb_mesh.value)

    @b_export.on_click
    def _(_):
        if sc.state != "exporting":
            busy(True)
        sc.export(float(num_vox.value), cb_loop.value, on_progress, on_done)

    @sl_gain.on_update
    def _(_):
        if int(sl_gain.value) != sc.gain:  # moved by the user (auto gain also updates the slider)
            cb_auto.value = False
            sc.auto_gain = False
            sc.set_gain(int(sl_gain.value))

    @cb_auto.on_update
    def _(_):
        sc.auto_gain = cb_auto.value

    normal_settings = {}

    @cb_fill.on_update
    def _(_):
        sc.depth.fill = cb_fill.value and sc.depth.can_fill

    @cb_detail.on_update
    def _(_):
        if cb_detail.value:
            normal_settings.update(kfd=sl_kfd.value, kfa=sl_kfa.value, zmax=sl_zmax.value)
            values = dict(kfd=1.5, kfa=3.0, zmax=0.5)
        else:
            values = normal_settings or dict(kfd=args.kf_dist * 100, kfa=float(args.kf_angle), zmax=float(args.zmax))
        sl_kfd.value, sl_kfa.value, sl_zmax.value = values["kfd"], values["kfa"], values["zmax"]
        sc.kf_dist, sc.kf_angle, sc.depth.z_max = values["kfd"] / 100, values["kfa"], values["zmax"]
        # close up the LEDs light things well: full-resolution matching (half res fills the rest)
        sc.depth.mode = "hybrid" if cb_detail.value else args.stereo
        sc.detail_mode = cb_detail.value

    @sl_zmax.on_update
    def _(_):
        sc.depth.z_max = float(sl_zmax.value)

    @sl_kfd.on_update
    def _(_):
        sc.kf_dist = float(sl_kfd.value) / 100

    @sl_kfa.on_update
    def _(_):
        sc.kf_angle = float(sl_kfa.value)

    @cb_flip.on_update
    def _(_):
        server.scene.set_up_direction("+y" if cb_flip.value else "-y")

    @cb_mesh.on_update
    def _(_):
        h = mesh_handle["h"]
        if h is not None:
            try:
                h.visible = cb_mesh.value
            except RuntimeError:  # removed by "New scan"
                pass

    fov = 2 * np.arctan(sc.rect.size / 2 / sc.rect.f)
    frustum = server.scene.add_camera_frustum("/cam", fov=float(fov), aspect=1.0, scale=0.04,
                                              color=(255, 120, 0), thickness=0.002)
    live_pc = None
    map_pc = None
    traj = None
    map_version = -1
    last_map_t = 0.0
    T_last = np.eye(4)
    try:
        while True:
            time.sleep(0.08)
            if clear_scene.is_set():
                clear_scene.clear()
                for h in (map_pc, traj, mesh_handle["h"]):
                    if h is not None:
                        try:
                            h.remove()
                        except Exception:
                            pass
                map_pc = traj = mesh_handle["h"] = None
                map_version = -1
            with sc.lock:
                live = dict(sc.live)
            if not live:
                continue
            l, z, track = live["l"], live["z"], live["track"]
            T = track.T_wc if (track is not None and track.ok) else T_last
            T_last = T

            # 2D view
            vis_l = cv2.cvtColor(l, cv2.COLOR_GRAY2RGB)
            if cb_expo.value:
                vis_l[l >= BRIGHT] = (255, 40, 40)
                dark = (l < DARK) & (l > 0)
                vis_l[dark] = (vis_l[dark] * 0.4 + np.array([0, 40, 140])).astype(np.uint8)
            for x, y in live["tracks"].astype(int):
                cv2.circle(vis_l, (int(x), int(y)), 2, (0, 255, 0), -1)
            vis_d = cv2.cvtColor(colorize_depth(z, sc.depth.z_min, sc.depth.z_max), cv2.COLOR_BGR2RGB)
            vis = np.hstack([vis_l, vis_d])
            if cb_flip.value:
                vis = np.hstack([cv2.rotate(vis_l, cv2.ROTATE_180), cv2.rotate(vis_d, cv2.ROTATE_180)])
            scale = 640 / vis.shape[1]
            img_h.image = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # camera pose
            wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
            frustum.wxyz = wxyz
            frustum.position = T[:3, 3]
            frustum.color = (255, 120, 0) if (track is None or track.ok) else (255, 0, 0)
            if cb_follow.value:
                for client in server.get_clients().values():
                    back = T[:3, :3] @ np.array([0, -0.05, -0.35]) + T[:3, 3]
                    client.camera.position = back
                    client.camera.look_at = T[:3, :3] @ np.array([0, 0, 0.3]) + T[:3, 3]

            # live depth points
            ps = float(sl_pt.value) / 1000
            if cb_live.value:
                pts, u, v = sc.depth.backproject(z, stride=4)
                pts = pts @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
                col = np.empty((len(pts), 3), np.uint8)
                col[:] = np.array([255, 140, 40], np.uint8)
                col[:, 1] = np.clip(60 + l[v, u] * 0.7, 0, 255).astype(np.uint8)
                if len(pts) == 0:
                    pts = np.zeros((1, 3), np.float32)
                    col = np.zeros((1, 3), np.uint8)
                live_pc = server.scene.add_point_cloud("/live", pts, col, point_size=ps)
            elif live_pc is not None:
                live_pc.visible = False

            # fused map (throttled)
            if sc.cloud.version != map_version and time.time() - last_map_t > 0.8:
                with sc.cloud_lock:
                    pts, g = sc.cloud.points(min_count=1)  # depth is already neighbour-confirmed
                    poses = list(sc.kf_poses)
                    map_version = sc.cloud.version
                last_map_t = time.time()
                if len(pts) > MAX_DISPLAY_POINTS:
                    sel = np.random.default_rng(0).choice(len(pts), MAX_DISPLAY_POINTS, replace=False)
                    pts, g = pts[sel], g[sel]
                if len(pts):
                    gc = np.clip(g, 0, 255).astype(np.uint8)
                    map_pc = server.scene.add_point_cloud("/map", pts, np.stack([gc, gc, gc], 1),
                                                          point_size=ps, visible=cb_map.value)
                if len(poses) > 1:
                    P = np.array([p[:3, 3] for p in poses])
                    seg = np.stack([P[:-1], P[1:]], 1)
                    traj = server.scene.add_line_segments("/traj", seg, (0, 160, 255), thickness=0.001)
            if map_pc is not None:
                map_pc.visible = cb_map.value
                map_pc.point_size = ps

            # capture quality
            q = sc.quality
            if sc.auto_gain and int(sl_gain.value) != sc.gain:
                sl_gain.value = sc.gain
            hints = []
            if q["dark"] > 0.45:
                hints.append("Mostly dark: add light (lamp, daylight) or get closer.")
            if q["bright"] > 0.03:
                hints.append("Blown-out areas: lower the gain or avoid shiny/white surfaces.")
            if q["depth"] < 0.04 and sc.state == "scanning":
                hints.append("Little depth: get closer (15-40 cm) and point at textured things.")
            if sc.backlog() > 4 and sc.state == "scanning":
                hints.append("Map is falling behind: move a bit slower.")
            quality_md.content = (f"**Depth** {q['depth']*100:.0f}% · **dark** {q['dark']*100:.0f}% · "
                                  f"**blown out** {q['bright']*100:.1f}% · **gain** {sc.gain}  \n"
                                  + ("  \n".join(hints) if hints else "Image looks good."))

            # status
            tr = ""
            if track is not None:
                tr = f"{track.status} · inliers {track.n_inliers}"
            disk = free_disk_mb(sc.out_root)
            status.content = (
                f"**State:** {sc.state}  \n"
                f"**Camera:** {sc.cam.fps:.0f} fps · processing {sc.proc_fps:.0f} fps  \n"
                f"**Tracking:** {tr or '-'}  \n"
                f"**Keyframes:** {len(sc.kf_poses)} · map points {len(sc.cloud):,}"
                + (f" · {sc.backlog()} waiting" if sc.backlog() > 1 else "") + "  \n"
                f"**Disk free:** {disk/1000:.1f} GB  \n\n{sc.message}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        sc.close()
        server.stop()
