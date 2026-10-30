<!-- BEGIN rtdd (generated from protocol/PROTOCOL.md; do not edit here) -->
## rtdd

`rtdd` reports which tests cover code you changed, and which changed lines nothing covers,
from recorded coverage rather than a static graph. It reports; it never gates.

`rtdd which` prints the ranked tests covering your current changes plus the uncovered
report, and runs nothing. Untracked files count, so a file you just wrote is included.
`rtdd which --json` is the machine-readable form.

`rtdd run` runs the selection and prints the uncovered report. Exit non-zero means a test
failed, and nothing else — an empty selection and an uncovered report are both exit 0.

The uncovered report splits changed lines into covered, uncovered, and import-time.
Import-time lines execute during collection and are attributed to no test — they are
reported separately and are not a coverage gap.

An empty selection is reported as its own outcome, never as a pass.

`--json` carries `selection_fidelity`: `execution-derived` (tests chosen from recorded
coverage), `static` (chosen from declared correspondence and imports, because this
toolchain records nothing), or `none` (nothing narrower than the full suite). A static
selection can miss a test an execution-derived one would have caught, so a passing static
selection is weaker evidence. `rtdd doctor` reports which fidelity this repository can
achieve, and why.

<!-- END rtdd -->

## Versioning

`x.y.z` (`__version__` in `imprinted/__init__.py`):

- **x, major:** heavy change after which pictures made by an older major version no longer decode (format change). All releases within one major version decode each other's pictures.
- **y, minor:** new features or changes; every picture from the same major version still decodes.
- **z, patch:** hotfixes, bug fixes, small changes.

Before bumping `y` or `z`, confirm the previous release's files still decode (`scripts/compare_versions.py`). If they don't, the change needs a new major version.

Branches: `main` = newest major. When a new major lands on `main`, the previous major is kept on a branch named after its last release (1.0.0 moved 0.7.x to branch `0.7.1`; after 2.0.0: `0.7.1`, `1.y.z`, `main`). Fixes for an older major go on its branch, renamed to the new last release.
