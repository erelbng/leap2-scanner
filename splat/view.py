"""View a trained splat in the browser (same as `leap2-scanner splat-view`).

    python splat/view.py scans/<name>                 # or a path to a 3DGS .ply
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from a repo checkout
from leap2_scanner.splatview import run_splat_view  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="scan folder or .ply")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    run_splat_view(a.path, a.host, a.port)
