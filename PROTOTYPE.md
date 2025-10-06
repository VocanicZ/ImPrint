# Imprint: Calibrated Color-Constellation Codes for Print/Scan-Robust Embedded Metadata

## Abstract

**Imprint** encodes metadata directly into an image by nudging colors inside human-invisible bounds, arranged in a QR-style macro layout that includes geometric **finders** and **color calibration** chips. Each data tile chooses one of several **perceptually equivalent color variants** of its local base color to represent a symbol (M-ary modulation). Robust decoding rectifies geometry, corrects color, classifies each tile in a perceptual space, and applies forward-error correction (FEC). Phase-1 targets lossless or lightly compressed images; stronger JPEG-robustness is planned as Phase-2.

---

## 1) Terminology (canonical names)

* **Finder (TACF)**: Three large **T**ri-**A**nchor **C**olor **F**inder modules (top-left, top-right, bottom-left). Each has a high-contrast ring for detection plus embedded **Calibration Chips** (mini color patches).
* **Calibration Chips**: Known colors inside each finder used to recover a device-independent color transform (camera/print correction).
* **Tile (iPS)**: A square **I**mprint **P**ixel **S**ize block that carries one symbol. Recommended: 12–16 px for the prototype.
* **Finder Size (iCP)**: A finder is **I**mprint **C**orner **P**ixel sized at `iCP = κ × iPS` (κ≈5).
* **Color Variant Palette (CVP)**: For each tile, an ordered list of N perceptually-close colors around its **Base Color** in OKLab.
* **Chromatic Variant Index (CVI)**: Index into the CVP. It’s the formalized version of your iOI:

  * CVI = `⊘` (null) → use the 1st (default) variant; no data carried.
  * CVI = 0 → use the 2nd variant; CVI = 1 → 3rd variant; etc.
* **Framing Header**: A small header encoded in the first row/column of tiles: version, grid, iPS, iCP, N, ECC, seed.
* **FEC**: Reed–Solomon RS(255,k) over GF(256) on the tile-symbol stream.
* **ΔE**: Perceptual color difference (OKLab/OKLCh ΔE). Visible goal after calibration: ΔE ≤ 2.

---

## 2) System Overview

**Encoder**:

1. Place three TACF finders; lay timing lines between them.
2. Partition payload region into `R×C` tiles of size iPS.
3. For each tile, compute base color (OKLab), build CVP of N variants with tiny ΔE offsets, pick the variant that encodes the next symbol (CVI).
4. Render tiles with **blue-noise dither** toward the target variant (to avoid flat blocks).
5. Write Framing Header + FEC’d payload.
6. Optionally sign a **Manifest** (time/location/hash) in Phase-1b.

**Decoder**:

1. Detect TACFs, estimate homography, **rectify**.
2. Read calibration chips, solve RGB→OKLab correction (3×3 or small 3D LUT).
3. Convert rectified payload to OKLab; sample each tile’s mean.
4. Classify to nearest CVP entry → CVI → symbol; majority vote within the tile if sub-samples are used.
5. Deframe + Reed–Solomon decode; verify header + optional signature.

---

## 3) Color Model & Palette Construction (CVP)

* Work in **OKLab** for perceptual uniformity.
* For a tile with base color `b = (L,a,b)`, construct N variants:

  * Fix lightness `L` (to avoid luminance edges).
  * Generate small hue/chroma offsets on a symmetric star around `b` in **OKLCh**:

    * `ΔC ∈ {±c1, ±c2}`, `Δh ∈ {±h1, ±h2}` (tunable); clamp to gamut.
  * Order variants by predicted **post-correction** ΔE; first is “default” (CVI=⊘), then 2nd→CVI=0, 3rd→CVI=1, etc.
* Prototype palette sizes: **N=8 (3 bits/tile)** or **N=4 (2 bits/tile)** for extra margin.

**Visibility budget**: pre-correction ΔE up to ~4 (cap 6), post-correction target ≤2. In sensitive regions (skin/highlights) reduce ΔE via masks.

---

## 4) Layout & Geometry

* Canvas is divided into:

  * Three TACFs at corners (iCP = 5×iPS suggested).
  * Horizontal/vertical **Timing Lines** (alternating high-contrast + known chroma) for precise row/col spacing estimation.
  * Payload grid occupying the remaining rectangle.
* Minimal viable grid for prototype: **R×C ≥ 16×16 tiles** (with N=8 → raw ~768 bits before overhead/FEC).

---

## 5) Framing & Error Correction

* **Header fields** (encoded first; RS-protected):

  * `ver` (u4), `iPS` (u8), `κ` (u4), `R` (u10), `C` (u10), `N` (u4), `FEC` (u8), `seed` (u32).
* **FEC**: Start with **RS(255, 191)** (~25% parity) for decent redundancy; tune later.
* **Interleaving**: Block-interleave symbols across rows to spread local damage.
* **Tile sampling**: For each tile, sample a 3×3 blue-noise subgrid; majority vote → tile mean.

---

## 6) Calibration & Classification

* **Geometry**: detect TACF rings; solve homography H via four corner ring centroids (three rings + one synthetic from timing lines).
* **Color**: use 6–12 Calibration Chips (W, 75% gray, 25% gray, R, G, B, plus two saturated hues).

  * Solve `RGB_cam → OKLab_ref` with least-squares: either (a) linear 3×3 + offset in linear-RGB, or (b) a compact 3D LUT (5³).
* **Classification**: for a tile’s corrected OKLab mean `x`, compute `argmin_j ΔE(x, v_j)` over CVP; map to CVI → symbol.

  * Reject if nearest–second ΔE gap < τ (ambiguous); mark as erasure for FEC.

---

## 7) Cryptographic Binding (Phase-1b, optional)

* **Manifest** (CBOR/JSON): version, device_id, capture_time, location (coarse), image_hash (SHA-256 of original cover or robust perceptual hash), CVP/ECC params, nonce.
* **Signature**: Ed25519 over the manifest; embed signature bytes via tiles.
* **Time-stamp MAC** and/or append-only log proof can be added later for auditability.

---

## 8) Prototype Parameters (recommended)

* **Image formats**: PNG / lossless WebP. Accept JPEG only at very high quality for lab.
* **iPS**: 14 px (range 12–16).
* **iCP**: κ = 5 → 70 px finders.
* **N (palette)**: 8 (3 bits/tile) to start.
* **FEC**: RS(255, 191), interleave by rows.
* **ΔE targets**: pre-corr ≤4 (cap 6), post-corr ≤2.
* **Dither**: blue-noise error-diffusion within tile toward target OKLab.
* **Seeding**: derive CVP orientation + dither seed from `seed` in header (or from the manifest nonce) for determinism.

---

## 9) Encoder Algorithm (prototype)

1. **Plan layout** from `(iPS, κ, R, C)`.
2. **Render TACFs** with ring + calibration chips (fixed reference OKLab).
3. **Frame payload**: header + (optional) manifest → RS encode → symbol stream.
4. For each payload tile:

   * Compute base OKLab from the underlying cover region.
   * Build CVP (N variants) around base; order as per §3.
   * Take next symbol `s`; choose CVI = `s` (CVI=⊘ when header says “null”).
   * Render toward the chosen variant using blue-noise dither (stay within gamut/ΔE budget).
5. Output PNG/lossless WebP.

---

## 10) Decoder Algorithm (prototype)

1. **Detect TACFs** (ring Hough/contour), compute homography **H**, **rectify**.
2. **Read timing lines**; refine grid spacing & skew.
3. **Calibrate color** from chips; apply correction to the rectified payload region.
4. **Tile sampling**: compute OKLab mean per tile (with 3×3 sub-samples).
5. **Classify** each tile → nearest CVP index; ambiguous → erasure.
6. **Deinterleave**, **RS decode**, parse header, recover payload.
7. If present, verify **signature** over the manifest.

---

## 11) Evaluation Plan

* **Datasets**: flat graphics, photos with skin/sky, high-texture scenes.
* **Perturbations**: brightness/contrast changes, white-balance shifts, slight blur, small rescale, print→scan loop (laser/inkjet).
* **Metrics**: tile classification accuracy, RS frame success rate, ΔE distributions pre/post correction, visual difference maps.
* **Ablations**: turn off calibration, reduce N, vary iPS, vary ΔE budget, with/without dither.

---

## 12) Threat Model (Phase-1)

* **Benign distortions**: exposure, WB, slight blur, mild rescale → handled by TACFs + calibration + FEC.
* **Malicious edits**: heavy recolor/filters, inpainting of finders, recompress to low-quality JPEG → out of scope for Phase-1; addressed in Phase-2.
* **Counterfeit**: Without the private key, an attacker can’t forge a valid signature/nonce bound to the image hash.

---

## 13) Roadmap

* **Phase-1 (this prototype)**: PNG/lossless path, N=8, iPS≈14, κ=5, RS(255,191), manifest optional.
* **Phase-1b**: Enable Ed25519 signing + manifest embedding + verification CLI.
* **Phase-2**: JPEG-robust variant:

  * Either keep macro layout but add **chroma-domain CVP** (work primarily in Cr) and larger ΔE;
  * Or add a **DCT-domain watermark** for redundancy under compression, cross-checking the macro code.

---

## 14) What we are (and aren’t) doing in the prototype

* **We are**: using **calibrated color** + **macro geometry** to embed data that survives recapture, starting with friendly formats (PNG/lossless).
* **We aren’t (yet)**: resisting harsh JPEG pipelines or aesthetic filters; that’s Phase-2 with a more aggressive chroma palette and/or DCT redundancy.

---