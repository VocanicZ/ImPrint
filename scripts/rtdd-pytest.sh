#!/usr/bin/env bash
# pytest for rtdd: same venv + thread caps as scripts/ci-local.sh, plus the
# pytest-cov / pytest-reportlog plugins the rtdd adapter needs. Args pass to pytest.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
PLUGINS=(pytest pytest-cov pytest-reportlog)
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then
    uv venv -q .venv && uv pip install -q -p .venv/bin/python -r requirements.txt "${PLUGINS[@]}"
  else
    python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt "${PLUGINS[@]}"
  fi
elif ! .venv/bin/python -c 'import pytest_cov, pytest_reportlog' 2>/dev/null; then
  if command -v uv >/dev/null; then
    uv pip install -q -p .venv/bin/python "${PLUGINS[@]}"
  else
    .venv/bin/pip install -q "${PLUGINS[@]}"
  fi
fi
exec nice -n "${RTDD_NICE:-10}" .venv/bin/python -m pytest "$@"
