# ImPrint: invisible, camera-robust data in images

![Original, ImPrinted, ×10 difference, and a simulated phone photo of a curled print decoding the payload](versus.png)

*Made by `scripts/readme_image.py` with the current encoder: kodim23 at default strength, payload `ImPrint`, then printed and photographed through `sim.py`'s curled-page channel (tilt, curl, mirror, clutter, JPEG).*

ImPrint hides a short payload (up to 32 bytes) inside an ordinary picture. The encoded image:

* **looks like the original**: same size, no border, no QR-style markers, no visible pattern (SSIM ≥ 0.955, mean ΔL\* ≤ 1 by default);
* **decodes from the pixels alone**, not from metadata: from the file, a screenshot, a crop, or a **phone photo of a print or a screen**, at an angle, in odd light, printed in grayscale, mirrored, gently curled, with clutter around it.

Typical uses: provenance tags, case or session IDs, leak forensics, linking a printed photo back to its record. The payload is plain bytes (checked by CRC-32, not encrypted or signed); bind a signature or ledger entry to it if you need proof.

> Status: research prototype. The format and API may still change. Simulated results are below; real printers and phones are the final judge.

---

## How it works

The current codec is `imprint_cam.py`. It only changes lightness (L\* in CIE Lab), so color shifts, white balance and grayscale printing don't matter.

**Data cells.** The image is split into cells of 4×4 sub-blocks (a sub-block prints at about 0.8 mm). Each cell carries one bit as the correlation of its sub-block means with a fixed ±1 pattern. The patterns are orthogonal to flat color and linear ramps, so lighting gradients and printer drift cancel out. The bit is quantized onto a lattice (QIM): the encoder moves each cell only to the nearest lattice point that reads the right bit, so busy texture costs a small bounded nudge instead of a lost cell.

**Error correction and tiling.** The payload plus CRC-32 is encoded with a rate-½ LDPC code. The codeword fills a T×T-cell tile, and the tile repeats over the whole picture, so any large enough piece decodes. The decoder soft-combines every repeat and runs belief propagation; CRC-32 rejects false reads.

**Hidden sync (the invisible finder).** Instead of corner markers, four faint cosines are locked to the cell grid. Each shows up as one sharp line in the spectrum, so the decoder can find the grid (spacing, rotation, mirror, perspective) anywhere in a photo. The encoder first notches the picture's own content at those frequencies so texture can't drown them. From a spectral seed the decoder tracks the pilot outward probe by probe, fits perspective and curl, then reads the cells. When a steep tilt smears every spectral line, a grid search tries plausible local grids directly and keeps the ones neighbouring probes confirm.

**Strength.** A perceptual mask (luminance adaptation, contrast sensitivity at the printed pattern frequency, contrast masking) sets how much each pixel may change; clipped pixels (pure black, white, saturated) are never touched. By default the encoder picks the strongest overall strength δ (up to 2.4) that keeps SSIM ≥ 0.955 and mean |ΔL\*| ≤ 0.99. Smooth pictures land near δ 1.2, busy ones near δ 0.8. Pass `delta=` to force a value.

### Measured (simulated)

`sim.py --acceptance`: 15 cm print, 5 trials, default strength, payload `IMPRINT1`. Every acceptance row passes.

| image | δ | SSIM | mean ΔL\* | clean / turned / mirrored | 60% crops | print easy | gray | hard | curled |
|---|---|---|---|---|---|---|---|---|---|
| kodim23 (smooth) | 1.21 | 0.956 | 0.74 | all | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 |
| kodim05 (busy) | 0.83 | 0.976 | 0.99 | all | 5/5 | 5/5 | 5/5 | – | – |
| kodim08 (busy, periodic) | 0.92 | 0.980 | 0.98 | all | 5/5 | 5/5 | 5/5 | – | – |

`hard` is tilt up to ~40°, 0.6 camera px per printed px, JPEG 75. `sim.py` models print (AM halftone, dot gain, grayscale) and photo (3D tilt, curl, cluttered scene, light, white balance, blur, noise, JPEG, mirror). It predicts; confirm on real hardware with the strength ladder page.

---

## Install

Needs Python 3.12 (other 3.10+ versions likely work, untested).

```bash
git clone https://github.com/VocanicZ/ImPrint.git
cd ImPrint
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows use `.venv\Scripts\pip` and `.venv\Scripts\python`.

## Use

**Web UI** (encode, decode, print a strength ladder; works from a phone on the same network):

```bash
.venv/bin/uvicorn web:app --host 0.0.0.0 --port 8765
# open http://<this-pc-ip>:8765
```

Set the print width when encoding: it picks the sub-block size so the pattern prints at about 0.8 mm. Decode attempts are logged to `results/decode_log.csv`.

**Python:**

```python
import cv2
import imprint_cam as ic

rgb = cv2.cvtColor(cv2.imread("photo.png"), cv2.COLOR_BGR2RGB)
b = ic.subblock_for_print(rgb.shape[1], print_mm=150)        # sized for a 15 cm wide print
enc, stats = ic.encode(rgb, b"hello", subblock_px=b)          # stats: delta, ssim, mean_dL, tiles, ...
cv2.imwrite("photo_imprint.png", cv2.cvtColor(enc, cv2.COLOR_RGB2BGR))   # save lossless (PNG)

photo = cv2.cvtColor(cv2.imread("phone_shot.jpg"), cv2.COLOR_BGR2RGB)
for f in ic.decode(photo)["found"]:
    print(f["payload"], f["raw_ber"])
```

`encode` takes RGB, RGBA or grayscale uint8 and returns the same layout. It raises `ValueError` when the picture is too small to read back or too clipped to carry the payload.

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
| `imprint_cam.py` | the codec: `encode`, `decode`, `make_ladder` |
| `web.py`, `web/index.html` | FastAPI web UI |
| `sim.py` | print/photo simulator and acceptance benchmark |
| `test_imprint_cam.py` | test suite |
| `testdata/` | Kodak test images (kodim05, 08, 23) |
| `scripts/ci-local.sh` | local CI gate |
| `scripts/readme_image.py` | regenerates `versus.png` |
| `docs/reports/` | experiment reports |
| `imprint.py`, `utils/`, `test.py`, `PROTOTYPE.md` | the earlier color-variant (CVP) prototype with visible corner markers, kept for reference |

---

## Contributing

Bug reports and PRs are welcome. Open issues and PRs against `camera-robust` (the default branch).

**Reporting a bug.** Include the input image (or a small one that reproduces it), the exact command or code, the payload, and what you expected vs. what happened. For decode failures, attach the photo and say how it was taken (printer, paper size, phone, distance, angle).

**Making a change.**

1. Branch from `camera-robust`.
2. Run the tests that cover your change while you work, then the full gate before you open the PR (below).
3. If you touch the codec, run `sim.py --acceptance` and put the table in the PR. A change that makes any row FAIL needs a strong reason.
4. If the encoder's output changes, regenerate the README figure: `.venv/bin/python scripts/readme_image.py` (exits non-zero if it no longer decodes).
5. Use conventional commit messages (`feat:`, `fix:`, `docs:`, ...) and reference the issue (`closes #N`).

There is no GitHub CI: `scripts/ci-local.sh` is the gate, and it builds `.venv` itself if missing.

### Test loop (rtdd, then ci-local)

The suite is slow (each test encodes and decodes a full image), so don't rerun all of it on every edit. [rtdd](https://github.com/VocanicZ/rtdd) keeps a per-test coverage map (`.rtdd/map.jsonl`, committed) and runs only the tests that cover your change:

```bash
rtdd which --base origin/camera-robust   # tests covering your diff + changed lines nothing covers; runs nothing
rtdd run   --base origin/camera-robust   # run just those tests, refresh their map rows
scripts/ci-local.sh                      # full gate before you push/merge
```

* `rtdd` runs pytest through `scripts/rtdd-pytest.sh` (adapter: `.rtdd/adapters/python.yaml`). It reuses ci-local's `.venv`, adds `pytest-cov` + `pytest-reportlog`, and runs under `nice -n 10`.
* An empty selection is not green: the map knows nothing about that change, so run `scripts/ci-local.sh`.
* Keep `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` at **4 or lower** (both scripts default to 4) and run sims under `nice`, e.g. `OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python sim.py --trials 2 photo.png`. Unbounded BLAS threads stall everything else on a shared machine.
* After adding or renaming tests, re-seed with `rtdd seed` (one full instrumented run) and commit `.rtdd/map.jsonl`. Never commit `.rtdd/junit.xml`, `.rtdd/meta.json` or `.coverage`.

---

## License

MIT. See [LICENSE](LICENSE). Test images in `testdata/` are from the Kodak Lossless True Color Image Suite, released for unrestricted use.
