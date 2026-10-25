#!/usr/bin/env bash
# Local CI entrypoint (Harness agents run this instead of GitHub checks).
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then
    uv venv -q .venv && uv pip install -q -p .venv/bin/python -r requirements.txt pytest
  else
    python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt pytest
  fi
fi
.venv/bin/python -m pytest -q test_imprint_cam.py
