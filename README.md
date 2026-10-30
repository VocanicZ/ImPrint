# Imprinted: invisible, camera-robust data in images

![Original, Imprinted, ×10 difference, and a simulated phone photo of a curled print decoding the payload](https://raw.githubusercontent.com/VocanicZ/Imprinted/main/versus.png)

*Made by `scripts/readme_image.py` with the current encoder: kodim23 at default strength, payload `Imprinted`, then printed and photographed through `sim.py`'s curled-page channel (tilt, curl, mirror, clutter, JPEG).*

Imprinted hides a short payload (up to 32 bytes) inside an ordinary picture. The encoded image:

* **looks like the original**: same size, no border, no QR-style markers, no visible pattern (SSIM ≥ 0.955, mean ΔL\* ≤ 1 by default);
* **decodes from the pixels alone**, not from metadata: from the file, a screenshot, a crop, or a **phone photo of a print or a screen**, at an angle, in odd light, printed in grayscale, mirrored, gently curled, with clutter around it.

Typical uses: provenance tags, case or session IDs, leak forensics, linking a printed photo back to its record. The payload is plain bytes (checked by CRC-32, not encrypted or signed); bind a signature or ledger entry to it if you need proof.

> Status: research prototype. The format and API may still change. Simulated results are below; real printers and phones are the final judge.

---

## How it works

The codec is `imprinted/core.py`. It only changes lightness (L\* in CIE Lab), so color shifts, white balance and grayscale printing don't matter.

**Data cells.** The image is split into cells of 4×4 sub-blocks (a sub-block prints at about 0.8 mm). Each cell carries one bit as the correlation of its sub-block means with a fixed ±1 pattern. The patterns are orthogonal to flat color and linear ramps, so lighting gradients and printer drift cancel out. The bit is quantized onto a lattice (QIM): the encoder moves each cell only to the nearest lattice point that reads the right bit, so busy texture costs a small bounded nudge instead of a lost cell.

**Error correction and tiling.** The payload plus CRC-32 is encoded with a rate-½ LDPC code. The codeword fills a T×T-cell tile, and the tile repeats over the whole picture, so any large enough piece decodes. The decoder soft-combines every repeat and runs belief propagation; CRC-32 rejects false reads.

**Hidden sync (the invisible finder).** Instead of corner markers, four faint cosines are locked to the cell grid. Each shows up as one sharp line in the spectrum, so the decoder can find the grid (spacing, rotation, mirror, perspective) anywhere in a photo. The encoder first notches the picture's own content at those frequencies so texture can't drown them. From a spectral seed the decoder tracks the pilot outward probe by probe, fits perspective and curl, then reads the cells. When a steep tilt smears every spectral line, a grid search tries plausible local grids directly and keeps the ones neighbouring probes confirm.

**Sync cells and the quick check.** A fixed pseudo-random quarter of the cells carry known bits instead of payload. The pilot is a lattice, and so is anything repetitive (halftone dots in a print, windows, fabric): those can look like a pilot, but nothing except an Imprinted picture correlates with the known bits. So `detect()` can say yes or no fast: it shrinks the photo to 2048 px, finds candidate grids from the spectrum (or, for steep tilts, a grid search), probes them, tracks the promising ones coarsely and runs a matched filter for the sync cells. It answers in stages: most Imprinted pictures within a second; a picture with nothing promising under way once every candidate source has been probed (the spectrum, then the grid search at full and half scale: a busy picture's faint pilot often shows only there) is ruled out then, usually within a few seconds; the rest are searched up to ten seconds. On the simulated prints and files within the acceptance envelope that was 81% confirmed by 1 s, 96% by 3 s and 98% by 10 s, with no false alarm on 266 pictures without a pattern (200 real photos). `decode()` runs that check first and gives up on a picture without an Imprinted pattern in seconds, instead of searching it for minutes. A picture under 1 MP has too few sync cells for the coarse look to be sure of: `detect()` tracks its candidate grids at full resolution instead and runs the same matched filter through that map (a clean 768×512 file scores far past the threshold), for at most ten seconds in all; `decode()` reads it in full. The decoder treats sync cells as erasures, and they also tell the two 180° turns of the lattice apart.

**Strength.** A perceptual mask (luminance adaptation, contrast sensitivity at the printed pattern frequency, contrast masking) sets how much each pixel may change; clipped pixels (pure black, white, saturated) are never touched. By default the encoder picks the strongest overall strength δ (up to 2.4) that keeps SSIM ≥ 0.955 and mean |ΔL\*| ≤ 0.99. Smooth pictures land near δ 1.2, busy ones near δ 0.8. Pass `delta=` to force a value.

### Measured (simulated)

`sim.py --acceptance`: 15 cm print, 5 trials, default strength, payload `IMPRINT1`. Every acceptance row passes.

| image | δ | SSIM | mean ΔL\* | clean / turned / mirrored | 60% crops | print easy | gray | hard | curled |
|---|---|---|---|---|---|---|---|---|---|
| kodim23 (smooth) | 1.21 | 0.956 | 0.74 | all | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| kodim05 (busy) | 0.83 | 0.976 | 0.99 | all | 5/5 | 5/5 | 5/5 | – | – |
| kodim08 (busy, periodic) | 0.92 | 0.980 | 0.98 | all | 5/5 | 5/5 | 5/5 | – | – |

`hard` is tilt up to ~40°, 0.6 camera px per printed px, JPEG 75. `sim.py` models print (AM halftone, dot gain, grayscale) and photo (3D tilt, curl, cluttered scene, light, white balance, blur, noise, JPEG, mirror). It predicts; confirm on real hardware with the strength ladder page.

Speed (v0.7.1, one desktop CPU, 4 threads): encoding takes about 1.5 s for a 768×512 picture and 25 s for 3000×2000; decoding a clean file takes 2 to 25 s, and a phone photo of a print 10 to 45 s.

---

## Install

Needs Python 3.10+ (tested on 3.12).

```bash
pip install imprinted
```

That installs the `imprinted` library only (numpy, opencv-python-headless, pillow). For the web UI, simulator and tests, clone the repo:

```bash
git clone https://github.com/VocanicZ/Imprinted.git
cd Imprinted
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows use `.venv\Scripts\pip` and `.venv\Scripts\python`.

## Use

**Python:**

```python
import imprinted

imprinted.encode("photo.png", "hello", "photo_imprinted.png")   # save as PNG
imprinted.encode("photo_imprinted.png", "world", "again.png", mode="append")   # now reads "hello world"
imprinted.decode("phone_shot.jpg")                            # -> "hello", or None
imprinted.detect("phone_shot.jpg")                            # -> True if it holds an Imprinted pattern
```

* `encode(image, payload, out=None, *, mode="override", print_width_cm=15.0, strength=None)` takes a path, PIL image or RGB uint8 array, and a `str` or `bytes` payload of 1 to 32 bytes. It returns the encoded PIL image (same size and mode) and saves it to `out` if given. Set `print_width_cm` to how wide the picture will be printed or shown: it sizes the pattern to about 0.8 mm per sub-block. `strength=None` picks the strongest setting inside the visibility budget. Raises `ValueError` when the picture is too small to read back or too clipped to carry the payload.
  * A picture that is already Imprinted is encoded again on its own grid and at its own strength (`print_width_cm` doesn't apply), with its old text taken out first, so exactly one text reads afterwards: `mode="override"` replaces the text, `mode="append"` writes `"<old text> <payload>"` (`ValueError` if that is over 32 bytes, or if the old text doesn't read). The change from the original stays about as small as a single encode's (mean ΔL\* 0.8–0.9 on kodim23, against 1.3–1.5 when the old text was taken out whole). `mode=None` skips the check (about a second, up to eight under 1 MP) for pictures you know are plain. `core.existing`, `core.unembed` and `core.reencode` are the parts.
* `decode(image, raw=False, check=True)` returns the first payload found as a `str`, as `bytes` with `raw=True`, or `None` if nothing is found. It runs `detect()` first and returns `None` right away when that finds no pattern. Pictures made with 0.7.x have no sync cells: read those with `check=False`.
* `detect(image)` returns `True` when the picture holds an Imprinted pattern, without reading the payload (a picture under 1 MP is read). Most Imprinted pictures answer within a second, most others are ruled out at one, none takes much over ten, and repetitive textures don't fool it.

The full codec is `imprinted.core`: `core.encode` also returns stats (strength, SSIM, mean ΔL\*, tiles), `core.decode` returns every payload found in a photo with its bit error rate and outline, and `core.make_ladder` builds the strength-test page.

**Web UI** (from a clone: encode, decode, print a strength ladder; works from a phone on the same network):

```bash
.venv/bin/uvicorn web:app --host 0.0.0.0 --port 8765
# open http://<this-pc-ip>:8765
```

Set the print width when encoding: it picks the sub-block size so the pattern prints at about 0.8 mm. Decode attempts are logged to `results/decode_log.csv`.

**Simulator:**

```bash
nice .venv/bin/python sim.py --trials 2 photo.png    # decode rates by strength x print/photo preset
nice .venv/bin/python sim.py --acceptance            # full acceptance table, exits 0 only on PASS
nice .venv/bin/python sim.py --quick                 # kodim23, 1 trial
```

`--acceptance` writes `acceptance.json`, each encoded image, its ×10 difference view and its strength ladder to `results/acceptance`.

---

## Repository layout

| path | what |
|---|---|
| `imprinted/__init__.py` | the library's simple `encode` / `decode` |
| `imprinted/core.py` | the codec: full `encode`, `decode`, `make_ladder` |
| `pyproject.toml` | pip package (`imprinted`) |
| `web.py`, `web/index.html` | FastAPI web UI |
| `sim.py` | print/photo simulator and acceptance benchmark |
| `test_imprinted.py` | test suite |
| `testdata/` | Kodak test images (kodim05, 08, 23) |
| `scripts/ci-local.sh` | local CI gate |
| `scripts/readme_image.py` | regenerates `versus.png` |
| `docs/reports/` | experiment reports |
| `imprint.py`, `utils/`, `test.py`, `PROTOTYPE.md` | the earlier color-variant (CVP) prototype with visible corner markers, kept for reference |

---

## Contributing

Bug reports and PRs are welcome. Open PRs against `main`.

**Reporting a bug.** Include the input image (or a small one that reproduces it), the exact command or code, the payload, and what you expected vs. what happened. For decode failures, attach the photo and say how it was taken (printer, paper size, phone, distance, angle).

**Making a change.**

1. Branch from `main`.
2. Run the tests that cover your change while you work, then the full gate before you open the PR (below).
3. If you touch the codec, run `sim.py --acceptance` and put the table in the PR. A change that makes any row FAIL needs a strong reason.
4. If the encoder's output changes, regenerate the README figure: `.venv/bin/python scripts/readme_image.py` (exits non-zero if it no longer decodes).
5. Use conventional commit messages (`feat:`, `fix:`, `docs:`, ...) and reference the issue (`closes #N`).

There is no GitHub CI: `scripts/ci-local.sh` is the gate, and it builds `.venv` itself if missing.

### Test loop (rtdd, then ci-local)

The suite is slow (each test encodes and decodes a full image), so don't rerun all of it on every edit. [rtdd](https://github.com/VocanicZ/rtdd) keeps a per-test coverage map (`.rtdd/map.jsonl`, committed) and runs only the tests that cover your change:

```bash
rtdd which --base origin/main           # tests covering your diff + changed lines nothing covers; runs nothing
rtdd run   --base origin/main           # run just those tests, refresh their map rows
scripts/ci-local.sh                      # full gate before you push/merge
```

* `rtdd` runs pytest through `scripts/rtdd-pytest.sh` (adapter: `.rtdd/adapters/python.yaml`). It reuses ci-local's `.venv`, adds `pytest-cov` + `pytest-reportlog`, and runs under `nice -n 10`.
* An empty selection is not green: the map knows nothing about that change, so run `scripts/ci-local.sh`.
* Keep `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` at **4 or lower** (both scripts default to 4) and run sims under `nice`, e.g. `OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python sim.py --trials 2 photo.png`. Unbounded BLAS threads stall everything else on a shared machine.
* After adding or renaming tests, re-seed with `rtdd seed` (one full instrumented run) and commit `.rtdd/map.jsonl`. Never commit `.rtdd/junit.xml`, `.rtdd/meta.json` or `.coverage`.

---

## License

MIT. See [LICENSE](LICENSE). Test images in `testdata/` are from the Kodak Lossless True Color Image Suite, released for unrestricted use.
