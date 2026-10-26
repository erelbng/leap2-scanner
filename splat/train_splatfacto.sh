#!/usr/bin/env bash
# Alternative: train with nerfstudio's splatfacto on the scan's transforms.json (run on the GPU machine).
#   splat/train_splatfacto.sh scans/<name> [extra ns-train args]
# Needs a nerfstudio env on PATH (e.g. conda activate nerfstudio). No depth/pose/exposure extras.
set -euo pipefail
SCAN=${1:?scan folder}
shift
ns-train splatfacto --data "$SCAN" --output-dir "$SCAN/ns" --viewer.quit-on-train-completion True "$@"
CFG=$(ls -t "$SCAN"/ns/*/splatfacto/*/config.yml | head -1)
ns-export gaussian-splat --load-config "$CFG" --output-dir "$SCAN/ns/export"
echo "exported to $SCAN/ns/export/splat.ply"
