#!/usr/bin/env bash
# Launch leap2-scanner from the repo (see README.md). Extra args go to `python -m leap2_scanner`.
# Uses ./.venv if present, otherwise $PYTHON or python3 with leap2-scanner's dependencies installed.
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
exec "$PY" -m leap2_scanner "$@"
