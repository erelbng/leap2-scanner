#!/usr/bin/env bash
# Fetch a trained splat (point_cloud.ply, renders, logs) back from the training machine.
#
#   splat/fetch.sh <user@gpu-host> <scan_name>
#
# Env: REMOTE_DIR (default: leap2-scanner). Large GIF renders are skipped (MP4s have the same content).
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${1:?usage: splat/fetch.sh <user@gpu-host> <scan_name>}
NAME=${2:?scan folder name, e.g. 20261024_090814}
REMOTE_DIR=${REMOTE_DIR:-leap2-scanner}
mkdir -p "scans/$NAME/splat"
rsync -a --info=progress2 --exclude '*.gif' "$HOST:$REMOTE_DIR/scans/$NAME/splat/" "scans/$NAME/splat/"
echo "view: ./scan.sh splat-view scans/$NAME   (or: python splat/view.py scans/$NAME/splat/point_cloud.ply)"
