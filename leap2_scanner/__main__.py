"""leap2-scanner - portable 3D scanner with the Leap Motion Controller 2.

    python -m leap2_scanner                 live scanner + browser viewer
    python -m leap2_scanner view DIR        browse a saved scan in the browser
    python -m leap2_scanner process [DIR]   export with settings, cut out an object mesh (browser UI)
    python -m leap2_scanner splat-view DIR  view a trained Gaussian splat
    python -m leap2_scanner export DIR      re-run mesh / splat export of a saved scan
    python -m leap2_scanner check           camera + calibration sanity check
    python -m leap2_scanner calibrate       refine calibration from live data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_OUT = Path("scans")  # relative to the working directory


def stl_faces(mb: float) -> int | None:
    """Triangle budget for a binary STL of about `mb` megabytes (50 bytes per triangle)."""
    return int(mb * 1e6 / 50) if mb > 0 else None


def main(argv=None):
    p = argparse.ArgumentParser(prog="leap2-scanner", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    def cam_args(sp):
        sp.add_argument("--res", type=int, default=1024, choices=[512, 1024], help="sensor stream resolution")
        sp.add_argument("--fov", type=float, default=100.0, help="rectified field of view in degrees")
        sp.add_argument("--model", default="poly",
                        choices=["equidistant", "kb5", "kb3last", "poly", "inv_poly", "inv_kb5", "inv_kb3last"],
                        help="lens model for the factory coefficients")
        sp.add_argument("--factory-calib", action="store_true", help="ignore a refined calibration")
        sp.add_argument("--zmin", type=float, default=0.15, help="closest depth (m); smaller = slower")
        sp.add_argument("--zmax", type=float, default=0.8, help="farthest depth used (m)")
        sp.add_argument("--stereo", default="half", choices=["half", "full", "hybrid"],
                        help="stereo matching: half is denser in dark IR and fast, full is sharper, "
                             "hybrid = full + half fill (detail mode uses hybrid)")
        sp.add_argument("--fill", action="store_true",
                        help="fill unmatched pixels from a WLS-filtered match: fewer holes, rougher surfaces, "
                             "~15 ms/frame (needs opencv-contrib; also a checkbox in the scanner)")

    s = sub.add_parser("scan", help="live scanner (default)")
    cam_args(s)
    s.add_argument("--out", default=str(DEFAULT_OUT), help="where scan folders are written (default: ./scans)")
    s.add_argument("--host", default="127.0.0.1", help="viewer bind address (0.0.0.0 = whole network)")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--autostart", action="store_true", help="start scanning right away (no click needed)")
    s.add_argument("--kf-dist", type=float, default=0.03, help="keyframe spacing (m)")
    s.add_argument("--kf-angle", type=float, default=6.0, help="keyframe spacing (deg)")
    s.add_argument("--cloud-voxel-mm", type=float, default=2.0, help="live point cloud resolution")
    s.add_argument("--mesh-voxel-mm", type=float, default=1.5, help="TSDF mesh resolution")
    s.add_argument("--stl-mb", type=float, default=40, help="decimate the mesh to about this STL size (0 = off)")
    s.add_argument("--mesh-smooth", type=int, default=10, help="Taubin smoothing iterations (0 = off)")

    e = sub.add_parser("export", help="re-export a saved scan folder")
    e.add_argument("session", type=Path)
    e.add_argument("--mesh-voxel-mm", type=float, default=1.5, help="TSDF mesh resolution")
    e.add_argument("--stl-mb", type=float, default=40,
                   help="decimate the mesh to about this STL size: detail keeps its triangles, flat areas "
                        "lose them (0 = off)")
    e.add_argument("--max-voxels", type=int, default=350**3,
                   help="stored TSDF voxels (~18 bytes each); above it the voxel grows")
    e.add_argument("--no-loop-closure", dest="loop_closure", action="store_false",
                   help="skip drift correction of the keyframe poses (~1-2 min per 100 keyframes)")
    e.add_argument("--redo-loop-closure", action="store_true",
                   help="drift-correct again from the original odometry (normally skipped once done)")
    e.add_argument("--mesh-smooth", type=int, default=10, help="Taubin smoothing iterations (0 = off)")
    e.add_argument("--detail-voxel-mm", type=float, default=1.0,
                   help="resolution of mesh_detail.stl (area of detail-mode keyframes; 0 = off)")
    e.add_argument("--restereo", choices=["half", "full", "hybrid"], default=None,
                   help="recompute keyframe depth from the saved images with this matcher first "
                        "(old depth/ is moved, not deleted)")
    e.add_argument("--mesh-only", action="store_true",
                   help="only (re)build the mesh; point cloud and splat files stay untouched")

    v = sub.add_parser("view", help="browse a saved scan (point cloud, mesh, cameras)")
    v.add_argument("session", type=Path)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8080)
    v.add_argument("--frustum-every", type=int, default=0, help="draw every n-th keyframe camera (0 = ~40 in total)")
    v.add_argument("--max-points", type=int, default=400_000, help="subsample the cloud for the browser")

    pr = sub.add_parser("process", help="export with settings and cut out object meshes (browser UI)")
    pr.add_argument("session", type=Path, nargs="?", help="scan folder (default: the newest in --scans)")
    pr.add_argument("--scans", type=Path, default=Path("scans"), help="folder with the scans")
    pr.add_argument("--host", default="127.0.0.1")
    pr.add_argument("--port", type=int, default=8090)
    pr.add_argument("--no-browser", action="store_true")
    sv = sub.add_parser("splat-view", help="view a trained Gaussian splat in the browser")
    sv.add_argument("path", type=Path, help="scan folder (uses splat/point_cloud.ply) or a 3DGS .ply")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8081)

    c = sub.add_parser("check", help="camera + calibration sanity check")
    cam_args(c)
    c.add_argument("--save", type=Path, default=None, help="write a preview PNG here")

    k = sub.add_parser("calibrate", help="refine calibration from live data")
    cam_args(k)
    k.add_argument("--seconds", type=float, default=30.0)
    k.add_argument("--from-scan", type=Path, nargs="+", default=None,
                   help="refine from saved scans instead of the live camera (works without the Leap)")
    k.add_argument("--every", type=int, default=2, help="with --from-scan: use every n-th keyframe")
    k.add_argument("--serial", default=None, help="device serial for the refined file (auto if unambiguous)")
    k.add_argument("--intrinsics", action="store_true",
                   help="also fit principal points / lens terms (bounded; usually not needed)")

    argv = sys.argv[1:] if argv is None else argv
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["scan", *argv]  # plain `leap2-scanner [options]` starts the scanner
    args = p.parse_args(argv)

    if args.cmd == "scan":
        from .app import run_app

        run_app(args)
    elif args.cmd == "export":
        from .export import Session
        from .pipeline import export_session

        session = Session.load(args.session)
        last = [-1]

        def progress(f):
            pct = int(f * 100)
            if pct // 10 != last[0]:
                last[0] = pct // 10
                print(f"  {pct}%", flush=True)

        export_session(session, args.mesh_voxel_mm / 1000, progress, print, args.max_voxels,
                       loop_closure=args.loop_closure, mesh_only=args.mesh_only, smooth=args.mesh_smooth,
                       max_faces=stl_faces(args.stl_mb), redo_loop_closure=args.redo_loop_closure,
                       restereo_mode=args.restereo, detail_voxel=args.detail_voxel_mm / 1000)
    elif args.cmd == "view":
        from .review import run_review

        run_review(args)
    elif args.cmd == "process":
        from .process import run_process

        run_process(args)
    elif args.cmd == "splat-view":
        from .splatview import run_splat_view

        run_splat_view(args.path, args.host, args.port)
    elif args.cmd == "check":
        from .tools import check

        check(args)
    elif args.cmd == "calibrate":
        from .tools import calibrate

        calibrate(args)


if __name__ == "__main__":
    main()
