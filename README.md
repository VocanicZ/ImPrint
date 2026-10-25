# ImPrint — Perceptual, Camera-Robust Data in Images (WIP)

![](versus.png)

ImPrint is a research prototype for **embedding small amounts of structured data into ordinary images** so that:

* The changes are **hard to notice by humans** under normal viewing.
* The data can be **recovered from screenshots or re-photographs** (not just by reading file metadata).
* The scheme can be extended with **authentication (signatures, timestamping, ledger proofs)** for evidentiary use.

> Status: early prototype. The code intentionally favors clarity over speed. Formats and CLIs are subject to change.

---

## Print/camera-robust prototype (`imprint_cam.py`)

Goal: hide a short payload in an image so the encoded file looks like the original (same size, **no visible border, markers or finder patterns**) and still decodes from a phone photo of a print or screen: any angle, light, grayscale print, mirrored, gently curled, background clutter, even cropped.

* **Data:** each cell (4×4 sub-blocks) carries one bit as the sign of a ±1 sub-block pattern on L\*. Patterns are orthogonal to flat color and linear ramps, so lighting, white balance and printer drift cancel. Informed embedding plus per-pixel texture masking and gamut headroom (pure black/white/saturated pixels are never touched).
* **Hidden sync (the invisible finder):** three faint cosines locked to the cell grid. Each is one spectral line, so the decoder finds the grid anywhere from the spectrum. The encoder notches the image's own content at those frequencies so texture can't drown them. Matched-filter phase gives the cell offset; the decoder then tracks the pilot outward and fits perspective + curl.
* **Payload tiles:** the codeword (payload + CRC-32, rate-½ Reed–Solomon) repeats in T×T-cell tiles over the whole picture, so any large-enough piece decodes. The decoder tries every tile phase.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn web:app --host 0.0.0.0 --port 8765   # web UI: encode / decode / strength ladder
scripts/ci-local.sh                                      # self-check (also what Harness agents run)
.venv/bin/python sim.py --trials 2 photo1.png            # simulated print->photo benchmark
nice .venv/bin/python sim.py --acceptance                # PRD #1 criteria table, PASS/FAIL (--quick: kodim23, 1 trial)
```

`sim.py` simulates print (AM halftone, dot gain, grayscale option) and photo (pinhole 3D tilt, curl, cluttered scene, light, white balance, blur, noise, JPEG, mirror). It predicts; the real printer/phone decides. Current status, open problems and acceptance targets live in the PRD issue on GitHub. `--acceptance` scores every PRD criterion (5 trials over `testdata/`; `--trials`, `--presets` and image paths override) and writes `acceptance.json`, each image's encoded copy, ×10 difference view and strength ladder to `--out` (default `results/acceptance`). It exits 0 only on PASS.

---

## Why another “watermark”?

Most “invisible watermark” systems push bits into frequency coefficients (e.g., JPEG DCT), which is brittle when images are re-shot from a screen, printed and scanned, or run through aggressive filters. ImPrint approaches the problem *from the color side*: we define **sets of color variants that are perceptually the same to a human** (within a ΔE threshold) and use *which* variant is chosen to carry information. Think of it as “**same color, different coordinates**.”

---

## Core Ideas

### 1) Canonical Variant Palette (CVP)

For any input color, we deterministically compute a **voxel** in CIE Lab space (a coarse cell). Inside that voxel we create:

* A **canonical/default color** (index 0) — the perceptual “center”.
* A fixed, ordered list of **alternate variants** (indices 1..K) that stay inside the same voxel and within a small ΔE of the canonical.
  These variants look the same to humans but are numerically different to a computer.

This ordered list is the **CVP**. The **ImPrint Order Index (iOI)** of a pixel is just the index in this CVP (default is “null”; alternates map to 0,1,2,… for carrying symbols).

> Determinism guarantee: no matter which CVP member you start from (or small perturbations around them), you recover the exact same ordered CVP list.

### 2) ImPrint Pixel Squares (iPS)

We work in tiles of size **w×w** pixels (argument: `-w`, often 10–20). Each tile contains many pixels of similar color; we quantize each pixel to the **default** CVP color (normalization), then choose a **single CVP variant index** for the *whole tile* to encode a symbol. Using per-tile symbols provides redundancy and robustness.

### 3) Targetable Affine Corner Finder (TACF)

To survive re-photography, we add three high-contrast **corner markers** (top-left, top-right, bottom-left). They allow the decoder to:

* detect the region-of-interest,
* estimate perspective/rotation/scale,
* recover the iPS grid.

Marker size is controlled by **iCP** (an integer multiplier of iPS): e.g., iPS=14 and iCP=5 → 70×70 corner squares.

### 4) Adaptive Radix per Tile (planned)

Each tile can have different numbers of safe CVP variants (depends on local color & gamut). Instead of forcing a global base (e.g., pure bits), the encoder will choose a **per-tile radix** (3..16 symbols typical) and pack data with FEC (Reed–Solomon) across the stream.

---

## What’s in this repo (prototype modules)

* `utils/cvp.py`
  Deterministic CVP generator and strip IO.

  * **Generate** CVP list from a color, optionally render a strip image.
  * **Read** a CVP strip back and verify invariance.
* `utils/normalizer.py`
  Snap every pixel in an image to the **default** member of its CVP (human-similar baseline).

  * Also provides a **sanity checker** that reports whether an image is fully normalized.
* `utils/tile_grid.py`
  Split an image into **w×w** tiles (as centered as possible) and optionally render grid lines for visualization.
* `utils/tacf.py`
  Draw the three **TACF** corner markers sized by `iCP` and aligned to the grid.
* `test.py`
  Convenience script to render a canvas with TACF and a blank tile grid.
* `imprint.py` (earlier PoC)
  First pass encoder/decoder; kept for reference. Newer workflows prefer the utils above and will bring back encode/decode as we stabilize headers and FEC.

---

## Install

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install numpy pillow opencv-python
# (reedsolomon/FEC will be added when we enable it; currently not required)
```

---

## Quickstart: CVP Basics

Generate the CVP list for a color and render a strip:

```bash
python utils/cvp.py "#66cc99" --out ./example_img/cvp_strip.png --w 40
```

Read the strip back and verify invariance:

```bash
python utils/cvp.py --from_strip ./example_img/cvp_strip.png --w 40
```

Example output (abridged):

```json
{
  "mode": "generate",
  "count": 9,
  "cvps": [
    {"index":"default","rgb":[...],"hex":"#..."},
    {"index":0,"rgb":[...],"hex":"#..."},
    {"index":1,"rgb":[...],"hex":"#..."}
  ],
  "image": "./example_img/cvp_strip.png"
}
```

---

## Normalize an Image (build a “base”)

Normalize every pixel to its **default CVP** (this yields an image perceptually close to the original, but numerically aligned for embedding):

```bash
python utils/normalizer.py ./example_img/demo_in.png ./example_img/demo_base.png \
  --stepL 8 --stepa 12 --stepb 12 --deltaE 1.0
```

Check whether an image is fully normalized:

```bash
python utils/normalizer.py ./example_img/demo_base.png --check
# -> reports "normalized": true/false and (optionally) a map of offending pixels
```

> Tuning invisibility vs. capacity:
>
> * Larger voxel steps (`--stepL/--stepa/--stepb`) and smaller `--deltaE` make the “base” closer to the original (less visible drift), but reduce how many safe variants each tile gets.
> * Start with `--stepL 8 --stepa 12 --stepb 12 --deltaE 1.0`. If you still see visible drift, increase steps a little or drop `deltaE` to `0.8`.

---

## Make a Tile Grid (iPS)

Split an image into **w×w** tiles and (optionally) render grid lines for inspection:

```bash
python utils/tile_grid.py ./example_img/demo_base.png \
  --w 14 \
  --grid_out ./example_img/demo_base_grid.png \
  --json_out ./example_img/demo_base_grid.json
```

* `--w` is your **iPS** size.
* `--grid_out` saves a copy with visible lines (debugging only).
* `--json_out` lists the grid geometry (rows, cols, tile rectangles).

---

## Add TACF Corner Markers (iCP)

Draw the three calibration markers sized by an integer **iCP** multiplier of iPS:

```bash
python utils/tacf.py ./example_img/demo_base.png ./example_img/demo_tacf.png \
  --w 14 --iCP 5 --style classic
```

* Markers are placed at top-left, top-right, bottom-left.
* The decoder will use these to unwarp, align, and recover the grid.

---

## Compose a Canvas with TACF and Grid (demo)

Render a synthetic canvas that contains TACF + grid (no payload yet):

```bash
python test.py \
  --w 14 --iCP 5 \
  --W 1024 --H 768 \
  --out ./example_img/canvas.png
```

---

## Roadmap (next steps)

* **Adaptive radix per tile** (current utils already expose per-tile CVP size; the encoder will pack symbols using as many variants as are safely available for that tile).
* **Header design**: magic + version + layout parameters (w, iCP), content digest; salted and optionally encrypted payload.
* **FEC**: Reed–Solomon across the symbol stream; interleaving across rows/cols to survive localized damage.
* **Robust decoding**:

  * TACF detection → homography → grid recovery.
  * Photometric normalization (white balance, gamma) before CVP decisions.
  * Multi-hypothesis testing on CVP index per tile with confidence reduction.
* **Security** (optional layers):

  * Detached **manifest** (what was embedded) and **signature** (Ed25519).
  * **Timestamp token** (TSA HMAC or RFC 3161).
  * **Transparency log** (Merkle inclusion proof) to prove the ImPrint existed at a past time.
* **Human-factor tuning**:

  * Psychovisual masks to choose tiles where changes are maximally hidden.
  * Spatial dithering to reduce block boundary salience.

---

## Current Limitations

* The “base/normalized” image may still be distinguishable from the original on flat gradients or large smooth areas. Mitigations:

  * Use larger Lab voxel steps and smaller ΔE for the base.
  * Prefer textures; avoid flat fills if strong invisibility is required.
* Decoding from prints/re-photos is not implemented yet; TACF exists, but the end-to-end homography + symbol inference is in progress.
* The legacy `imprint.py` encoder/decoder is experimental and will be replaced by the adaptive-radix pipeline.

---

## Practical Use Cases (target)

* **Provenance & anti-tamper**: bind time, device, and location into pixels; verify even after screenshots/recompression.
* **Chain-of-custody**: couple a photo to a case ID with a verifiable ledger entry.
* **Soft DRM & leak forensics**: encode a short session or user ID robustly.

> Reminder: cryptographic binding (signatures, logs) turns “this image contains bits” into “this image claims X, signed by Y, seen at time T.” The pixels carry the bits; the proofs make them meaningful.

---

## Example End-to-End (today’s prototype)

1. **Normalize** the image:

```bash
python utils/normalizer.py ./example_img/demo_in.png ./example_img/demo_base.png \
  --stepL 8 --stepa 12 --stepb 12 --deltaE 1.0
```

2. **Grid** and **markers**:

```bash
python utils/tile_grid.py ./example_img/demo_base.png --w 14 \
  --grid_out ./example_img/demo_base_grid.png \
  --json_out ./example_img/demo_base_grid.json

python utils/tacf.py ./example_img/demo_base.png ./example_img/demo_tacf.png \
  --w 14 --iCP 5
```

3. **(Soon)** Encode a short message with adaptive radix per tile and export a manifest.
4. **(Soon)** Decode from the TACF + grid, recover message + verify manifest.

---

## Design Choices & Tuning Notes

* **Color space**: CIE Lab for perceptual uniformity.
* **Voxelization**: `(stepL, stepa, stepb)` define buckets; any input in the same bucket yields the same CVP list.
* **ΔE tolerance**: bounds how far variants may wander from the canonical center.
* **Invisibility levers**:

  * Higher `step*` and lower `ΔE` → closer to the original appearance (fewer variants).
  * Lower `step*` and higher `ΔE` → more variants (capacity) but increased risk of visible hue/contrast shifts.
* **Human vision**: ΔE ≈ 1 is near just-noticeable difference under ideal viewing; many displays/printers/phones are noisier than that. Start conservative and test on your target pipeline (screenshot, print, compress).

---

## Troubleshooting

* **“The normalized image still looks different.”**
  Increase `--stepL/--stepa/--stepb` and/or reduce `--deltaE`. For very flat backgrounds, consider texture-preserving masks (coming soon).
* **“Grid won’t fit nicely.”**
  Try a smaller `-w`. The grid aligns itself centered; we pad/clip as needed.
* **“Decoder can’t find markers.”**
  Ensure TACF is present and large enough (`iCP ≥ 4` for small images). Avoid cropping.

---

## Contributing

Open issues with:

* A minimal input image,
* Exact command lines,
* The observed vs. expected behavior.

The code is intentionally modular; PRs that improve determinism, decoding robustness, or human-factors are welcome.

### Test loop (rtdd → ci-local)

The suite is slow (each test encodes/decodes a full image), so don't rerun all of it on every edit. [rtdd](https://github.com/VocanicZ/rtdd) keeps a per-test coverage map (`.rtdd/map.jsonl`, committed) and runs only the tests that cover your change:

```bash
rtdd which --base origin/camera-robust   # tests covering your diff + changed lines nothing covers; runs nothing
rtdd run   --base origin/camera-robust   # run just those tests, refresh their map rows
scripts/ci-local.sh                      # full gate before you push/merge
```

* `rtdd` runs pytest through `scripts/rtdd-pytest.sh` (adapter: `.rtdd/adapters/python.yaml`, which overrides rtdd's built-in `python` adapter). It reuses ci-local's `.venv`, adds `pytest-cov` + `pytest-reportlog`, and runs under `nice -n 10`.
* An empty selection is not green. The map knows nothing about that change, so run `scripts/ci-local.sh`.
* Keep `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` at **4 or lower**. Both scripts default to 4. Run sims under `nice` too, e.g. `OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python sim.py --trials 2 photo1.png`. Several agents share the box, and unbounded BLAS threads stall every other run.
* After adding or renaming tests, re-seed with `rtdd seed` (one full instrumented run) and commit `.rtdd/map.jsonl`. Never commit `.rtdd/junit.xml`, `.rtdd/meta.json` or `.coverage`.

---

## License

This project is under MIT License

---

## Glossary

* **CVP** — Canonical Variant Palette (ordered list of perceptually-equivalent color variants; index 0 is the canonical/default).
* **iOI** — ImPrint Order Index (tile’s chosen CVP index used to encode a symbol; default is “null,” alternates map to 0,1,2,…).
* **iPS** — ImPrint Pixel Square (tile size `w×w` used as one symbol carrier).
* **iCP** — ImPrint Corner Pixel multiplier (marker size in multiples of `w`).
* **TACF** — Targetable Affine Corner Finder (three corner markers for alignment).
* **ΔE** — Perceptual color difference (CIE76 in this prototype).
* **FEC** — Forward Error Correction (Reed–Solomon planned).

---

*This document evolves with the prototype. If something is unclear or you want a worked example tailored to your image set (e.g., medical scans, body cams, receipts), that’s the perfect next experiment.*
