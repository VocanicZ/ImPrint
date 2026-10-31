#!/usr/bin/env bash
# Push main, build the `imprinted` package, upload it to PyPI, then tag v<version> and create the
# GitHub release with the same files. Asks for your PyPI API token (pypi-...) without echoing it;
# nothing is saved. Bump __version__ in imprinted/__init__.py and commit on main first: PyPI
# refuses a version it already has, and the release must point at code that is on GitHub.
# A new major version (README "Versioning") also keeps the previous major on a branch named after
# its last release (1.0.0 -> branch 0.7.1). The README must be finished first: the script stops
# if README.md or its figure (versus.png, stamped with the version) hasn't changed since the last
# release. --force publishes anyway. Last, scripts/deploy.sh puts this version on
# https://imprinted.rainbowcreation.net/ (asks for the host's password unless you have a key).
set -euo pipefail
cd "$(dirname "$0")/.."
force=0; [ "${1:-}" = "--force" ] && force=1
ver=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' imprinted/__init__.py)
git fetch -q origin
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "commit your changes first"; exit 1; }
[ "$(git branch --show-current)" = main ] || { echo "publish from main"; exit 1; }
git merge-base --is-ancestor origin/main HEAD || { echo "main is behind origin/main: pull first"; exit 1; }
! git rev-parse -q --verify "refs/tags/v$ver" >/dev/null || { echo "v$ver is already tagged: bump __version__"; exit 1; }
[ -x .venv/bin/python ] || python3 -m venv .venv
prev=$(git tag -l 'v*' --sort=-v:refname | head -1); prev=${prev#v}   # the last release before this one
if [ -n "$prev" ] && [ $force = 0 ]; then
  for f in README.md versus.png; do
    git diff --quiet "v$prev" HEAD -- "$f" || continue
    echo "WARNING: $f is unchanged since v$prev: update the README (versus.png: scripts/readme_image.py) first, or run with --force"
    exit 1
  done
fi
git push -q origin main                                            # the release's code on GitHub first
.venv/bin/pip install -q -U build twine
rm -rf dist build imprinted.egg-info
# README links versus.png and the logo relatively (each branch shows its own); PyPI needs full URLs: this tag's
cp README.md README.md.orig; trap 'mv -f README.md.orig README.md' EXIT
raw="https://raw.githubusercontent.com/VocanicZ/Imprinted/v$ver"
sed -i -e "s#](versus.png)#]($raw/versus.png)#" -e "s#src=\"web/logo.png\"#src=\"$raw/web/logo.png\"#" README.md
.venv/bin/python -m build
mv -f README.md.orig README.md; trap - EXIT
.venv/bin/twine check dist/*
read -rsp "PyPI API token (pypi-...): " token
echo
TWINE_USERNAME=__token__ TWINE_PASSWORD="$token" .venv/bin/twine upload dist/*
git tag -a "v$ver" -m "imprinted $ver"
git push -q origin "v$ver"
if [ -n "$prev" ] && [ "${ver%%.*}" != "${prev%%.*}" ] && ! git ls-remote -q --exit-code --heads origin "$prev" >/dev/null; then
  git push -q origin "v$prev^{commit}:refs/heads/$prev"           # new major: keep the old one on its branch
  echo "major ${prev%%.*} kept on branch $prev"
fi
gh release create "v$ver" dist/* --verify-tag --title "v$ver" --generate-notes
scripts/deploy.sh
