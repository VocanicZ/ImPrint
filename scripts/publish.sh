#!/usr/bin/env bash
# Build the `imprinted` package and upload it to PyPI. Asks for your PyPI API token
# (pypi-...) without echoing it; nothing is saved. Bump __version__ in
# imprinted/__init__.py first: PyPI refuses a version it already has.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -U build twine
rm -rf dist build imprinted.egg-info
.venv/bin/python -m build
.venv/bin/twine check dist/*
read -rsp "PyPI API token (pypi-...): " token
echo
TWINE_USERNAME=__token__ TWINE_PASSWORD="$token" .venv/bin/twine upload dist/*
