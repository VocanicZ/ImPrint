#!/usr/bin/env bash
# Build the `imprinted` package, upload it to PyPI, then tag v<version> and create the GitHub
# release with the same files. Asks for your PyPI API token (pypi-...) without echoing it;
# nothing is saved. Bump __version__ in imprinted/__init__.py and push first: PyPI refuses a
# version it already has, and the release must point at code that is on GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
ver=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' imprinted/__init__.py)
git fetch -q origin
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "commit your changes first"; exit 1; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || { echo "push main first (HEAD != origin/main)"; exit 1; }
! git rev-parse -q --verify "refs/tags/v$ver" >/dev/null || { echo "v$ver is already tagged: bump __version__"; exit 1; }
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -U build twine
rm -rf dist build imprinted.egg-info
.venv/bin/python -m build
.venv/bin/twine check dist/*
read -rsp "PyPI API token (pypi-...): " token
echo
TWINE_USERNAME=__token__ TWINE_PASSWORD="$token" .venv/bin/twine upload dist/*
git tag -a "v$ver" -m "imprinted $ver"
git push -q origin "v$ver"
gh release create "v$ver" dist/* --verify-tag --title "v$ver" --generate-notes
