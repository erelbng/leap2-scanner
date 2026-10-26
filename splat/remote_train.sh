#!/usr/bin/env bash
# Copy the project + one scan to a CUDA machine and train a splat there (in the background).
#
#   splat/remote_train.sh <user@gpu-host> [scan_dir] [extra train.py args...]
#
# scan_dir defaults to the newest folder in scans/. Environment variables:
#   REMOTE_DIR     project folder on the GPU machine        (default: leap2-scanner)
#   REMOTE_PYTHON  python with torch + gsplat on that machine (default: python3)
#   REMOTE_ENV     optional conda/venv prefix; its bin/ goes on PATH (gsplat needs `ninja`
#                  from there to JIT-compile its CUDA kernels), python defaults to $REMOTE_ENV/bin/python
# Only copies (rsync without --delete); nothing on either side is removed.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${1:?usage: splat/remote_train.sh <user@gpu-host> [scan_dir] [train.py args...]}
SCAN=${2:-$(ls -d scans/*/ | sort | tail -1)}
shift $(( $# < 2 ? $# : 2 ))
REMOTE_DIR=${REMOTE_DIR:-leap2-scanner}
REMOTE_ENV=${REMOTE_ENV:-}
if [ -n "$REMOTE_ENV" ]; then
  REMOTE_PYTHON=${REMOTE_PYTHON:-$REMOTE_ENV/bin/python}
  RPATH="PATH=$REMOTE_ENV/bin:\$PATH"
else
  REMOTE_PYTHON=${REMOTE_PYTHON:-python3}
  RPATH=""
fi
SCAN=${SCAN%/}
NAME=$(basename "$SCAN")

echo "copying project + $SCAN to $HOST:$REMOTE_DIR"
ssh "$HOST" "mkdir -p $REMOTE_DIR/scans"
rsync -a --exclude .venv --exclude scans --exclude __pycache__ --exclude .git ./ "$HOST:$REMOTE_DIR/"
rsync -a "$SCAN/" "$HOST:$REMOTE_DIR/scans/$NAME/"

# subshell + full redirection so ssh returns immediately; nohup keeps training alive after logout
ssh "$HOST" "cd $REMOTE_DIR && mkdir -p scans/$NAME/splat && \
  ($RPATH nohup $REMOTE_PYTHON splat/train.py --data scans/$NAME $* \
   > scans/$NAME/splat/stdout.log 2>&1 < /dev/null &)"
echo "training started on $HOST"
echo "  follow: ssh $HOST tail -f $REMOTE_DIR/scans/$NAME/splat/train.log"
echo "  fetch:  splat/fetch.sh $HOST $NAME"
