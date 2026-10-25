#!/usr/bin/env bash
# Local CI entrypoint (Harness agents run this instead of GitHub checks).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -U pip
  .venv/bin/pip install -q -r requirements.txt pytest
fi
.venv/bin/python -m pytest -q test_imprint_cam.py
