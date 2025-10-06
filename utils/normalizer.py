# utils/normalizer.py
# Normalize an image so every pixel is the canonical/default CVP (index 0),
# and provide a checker to verify if an image is fully normalized.
#
# USAGE
#   # Normalize (legacy form kept):
#   python utils/normalizer.py in.png out.png [--stepL 1.6 --stepa 2.2 --stepb 2.2]
#                                        [--dither ordered|random|none --strength 0.5]
#                                        [--preview_diff diff.png] [--strict_check]
#
#   # Check only (no output image written):
#   python utils/normalizer.py in.png --check_only [--stepL ... --stepa ... --stepb ...]
#                                     [--stride 1] [--max_report 50]
#
from __future__ import annotations
import argparse, json, os, sys
from typing import Dict, Tuple, List
import numpy as np

try:
    from PIL import Image
except Exception as e:
    raise SystemExit("Pillow is required: pip install pillow") from e

# Ensure we can import the sibling utils/cvp.py regardless of cwd
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from cvp import (
    rgb_to_lab01, lab_to_rgb01, Voxelizer, deltaE76,
    cvp_list_for_color, _rgb_to_hex, cvp_defaults
)

# load CVP spec defaults
SPEC = cvp_defaults()

# --------- ordered 8×8 blue-noise-ish threshold in [0,1)
_BN8 = np.array([
    [0.73,0.07,0.55,0.21,0.90,0.38,0.62,0.28],
    [0.45,0.97,0.31,0.83,0.14,0.66,0.52,0.04],
    [0.59,0.25,0.69,0.11,0.76,0.18,0.80,0.34],
    [0.03,0.50,0.36,0.88,0.26,0.78,0.42,0.94],
    [0.86,0.40,0.64,0.30,0.99,0.47,0.71,0.17],
    [0.22,0.74,0.08,0.60,0.33,0.85,0.19,0.67],
    [0.48,0.12,0.82,0.44,0.56,0.02,0.96,0.58],
    [0.16,0.68,0.24,0.72,0.06,0.54,0.32,0.84],
], dtype=np.float32)

def _make_bias_field(H: int, W: int, strength: float, mode: str, seed: int = 1234) -> np.ndarray | None:
    if mode == "none" or strength <= 0.0:
        return None
    amp = 0.25 * float(np.clip(strength, 0.0, 1.0))  # ±0.25 in center-grid units
    if mode == "ordered":
        T = np.tile(_BN8, ((H+7)//8, (W+7)//8))[:H,:W]
        bias = (T - 0.5) * (2.0*amp)
        return np.stack([bias, bias, bias], axis=-1).astype(np.float32)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(-amp, +amp, size=(H,W,3)).astype(np.float32)
    raise ValueError(f"Unknown dither mode: {mode}")

def _nearest_center_indices(Lab: np.ndarray, vox: Voxelizer, bias: np.ndarray | None):
    L, a, b = Lab[...,0], Lab[...,1], Lab[...,2]
    gL = L / vox.step_L
    ga = (a + 128.0) / vox.step_a
    gb = (b + 128.0) / vox.step_b
    if bias is not None:
        gL = gL + bias[...,0]; ga = ga + bias[...,1]; gb = gb + bias[...,2]
    iL = np.rint(gL - 0.5).astype(np.int32)
    ia = np.rint(ga - 0.5).astype(np.int32)
    ib = np.rint(gb - 0.5).astype(np.int32)
    return iL, ia, ib

def _center_lab_from_indices(iL, ia, ib, vox: Voxelizer) -> np.ndarray:
    cL = (iL + 0.5)*vox.step_L
    ca = (ia + 0.5)*vox.step_a - 128.0
    cb = (ib + 0.5)*vox.step_b - 128.0
    return np.stack([cL, ca, cb], axis=-1).astype(np.float32)

def _canonical_default_for_voxel(iL: int, ia: int, ib: int, vox: Voxelizer,
                                 stepL: float, stepa: float, stepb: float) -> Tuple[int,int,int]:
    """
    Compute the EXACT RGB default cvp_list_for_color(...) uses for this voxel.
    """
    cen_lab = _center_lab_from_indices(np.array(iL), np.array(ia), np.array(ib), vox)
    cen_rgb01 = np.clip(lab_to_rgb01(cen_lab[None,None,:])[0,0,:], 0.0, 1.0)
    seed_rgb = (cen_rgb01*255.0 + 0.5).astype(np.uint8)
    seed_str = f"{int(seed_rgb[0])},{int(seed_rgb[1])},{int(seed_rgb[2])}"
    cvps = cvp_list_for_color(seed_str, step_L=stepL, step_a=stepa, step_b=stepb,
                              directions=8, ring_count=2, deltaE_max=2.0, include_L_variants=False)
    return int(cvps[0][0]), int(cvps[0][1]), int(cvps[0][2])

def normalize_image(
    in_path: str,
    out_path: str,
    stepL: float = SPEC["stepL"],
    stepa: float = SPEC["stepa"],
    stepb: float = SPEC["stepb"],
    dither: str = "ordered",     # "none" | "ordered" | "random"
    strength: float = 0.5,
    preview_diff: str | None = None,
    strict_check: bool = False
) -> dict:
    img = Image.open(in_path).convert("RGB")
    np_in = np.asarray(img, dtype=np.uint8)
    H, W = np_in.shape[:2]
    rgb01 = np_in.astype(np.float32) / 255.0
    Lab = rgb_to_lab01(rgb01)

    vox = Voxelizer(step_L=stepL, step_a=stepa, step_b=stepb)
    bias = _make_bias_field(H, W, strength, dither)

    # Nearest voxel-center indices
    iL, ia, ib = _nearest_center_indices(Lab, vox, bias)

    # Cache canonical defaults per unique voxel
    uniq_keys = np.unique(np.stack([iL, ia, ib], axis=-1).reshape(-1,3), axis=0)
    cache: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}
    for k in uniq_keys:
        key = (int(k[0]), int(k[1]), int(k[2]))
        cache[key] = _canonical_default_for_voxel(key[0], key[1], key[2], vox, stepL, stepa, stepb)

    # Map each pixel to its canonical default
    out = np.zeros_like(np_in)
    for key, rgb in cache.items():
        mask = (iL == key[0]) & (ia == key[1]) & (ib == key[2])
        out[...,0][mask] = rgb[0]
        out[...,1][mask] = rgb[1]
        out[...,2][mask] = rgb[2]

    # ΔE stats
    out01 = out.astype(np.float32) / 255.0
    out_lab = rgb_to_lab01(out01)
    dE = np.sqrt(np.sum((Lab - out_lab)**2, axis=-1))
    mean_de = float(np.mean(dE))
    p95_de  = float(np.quantile(dE, 0.95))
    p99_de  = float(np.quantile(dE, 0.99))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(out, mode="RGB").save(out_path)

    if preview_diff:
        vis = (np.clip(dE / max(1e-6, p99_de), 0, 1) * 255.0).astype(np.uint8)
        vis_rgb = np.stack([vis*0, vis, vis*0], axis=-1)
        Image.fromarray(vis_rgb, mode="RGB").save(preview_diff)

    # Optional strict invariance check on unique colors
    invariance_ok = True
    violations = []
    if strict_check:
        unique_colors = np.unique(out.reshape(-1,3), axis=0)
        for rgb in unique_colors:
            s = f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}"
            lst = cvp_list_for_color(s, step_L=stepL, step_a=stepa, step_b=stepb,
                                     directions=8, ring_count=2, deltaE_max=2.0, include_L_variants=False)
            if tuple(lst[0]) != (int(rgb[0]), int(rgb[1]), int(rgb[2])):
                invariance_ok = False
                violations.append({"rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
                                   "default_of_cvp": [int(lst[0][0]), int(lst[0][1]), int(lst[0][2])],
                                   "hex": _rgb_to_hex(tuple(rgb))})

    return {
        "mode": "normalize",
        "verdict": "OK",
        "in": in_path,
        "out": out_path,
        "size": [H, W],
        "voxel_steps": {"L": stepL, "a": stepa, "b": stepb},
        "dither": dither, "strength": strength,
        "deltaE76": {"mean": mean_de, "p95": p95_de, "p99": p99_de},
        "invariance": {"checked": strict_check, "ok": invariance_ok, "violations": violations}
    }

def check_normalized_image(
    in_path: str,
    stepL: float = SPEC["stepL"],
    stepa: float = SPEC["stepa"],
    stepb: float = SPEC["stepb"],
    stride: int = 1,
    max_report: int = 50
) -> dict:
    """
    Verify that for every (sampled) pixel `p`, cvp_list_for_color(p)[0] == p.
    Returns a JSON-serializable dict with counts and sample violations.

    stride=1 checks every pixel; stride=2 checks every 2nd row/col, etc.
    """
    img = Image.open(in_path).convert("RGB")
    np_img = np.asarray(img, dtype=np.uint8)
    H, W = np_img.shape[:2]

    # Subsample if requested
    if stride < 1:
        stride = 1
    sub = np_img[::stride, ::stride, :]
    HH, WW = sub.shape[:2]
    total = HH * WW

    flat = sub.reshape(-1, 3)
    # Group by unique colors to minimize cvp calls
    colors, inv = np.unique(flat, axis=0, return_inverse=True)

    # Compute default for each unique color
    defaults = np.zeros_like(colors)
    same_mask = np.zeros((colors.shape[0],), dtype=bool)
    for i, rgb in enumerate(colors):
        s = f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}"
        lst = cvp_list_for_color(s, step_L=stepL, step_a=stepa, step_b=stepb,
                                 directions=8, ring_count=2, deltaE_max=2.0, include_L_variants=False)
        d = np.array([lst[0][0], lst[0][1], lst[0][2]], dtype=np.int32)
        defaults[i,:] = d
        same_mask[i] = (d[0] == rgb[0]) and (d[1] == rgb[1]) and (d[2] == rgb[2])

    # Any color groups that are not normalized?
    bad_groups = np.where(~same_mask)[0]
    bad_count = 0
    violations: List[dict] = []

    if bad_groups.size > 0:
        for gi in bad_groups:
            # pixels of this group
            idxs = np.where(inv == gi)[0]
            bad_count += idxs.size
            # record a few sample coordinates
            for k in idxs[:max(1, max_report // max(1, bad_groups.size))]:
                y = int(k // WW)
                x = int(k %  WW)
                rgb = colors[gi]
                d   = defaults[gi]
                violations.append({
                    "xy": [int(y*stride), int(x*stride)],
                    "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
                    "expected_default": [int(d[0]), int(d[1]), int(d[2])],
                    "hex": _rgb_to_hex((int(rgb[0]), int(rgb[1]), int(rgb[2])))
                })
                if len(violations) >= max_report:
                    break
            if len(violations) >= max_report:
                break

    frac_bad = float(bad_count) / float(total) if total > 0 else 0.0

    return {
        "mode": "check",
        "in": in_path,
        "size": [H, W],
        "stride": stride,
        "voxel_steps": {"L": stepL, "a": stepa, "b": stepb},
        "total_checked": int(total),
        "unique_colors_checked": int(colors.shape[0]),
        "not_normalized_count": int(bad_count),
        "not_normalized_fraction": frac_bad,
        "ok": bad_count == 0,
        "violations": violations
    }

def main():
    global SPEC
    ap = argparse.ArgumentParser(description="Normalize to canonical CVP defaults or check normalization.")
    ap.add_argument("input", help="Input image path")
    ap.add_argument("output", nargs="?", default=None, help="Output (for normalization mode)")

    # Shared / model parameters
    ap.add_argument("--stepL", type=float, default=None)
    ap.add_argument("--stepa", type=float, default=None)
    ap.add_argument("--stepb", type=float, default=None)

    # Normalize options
    ap.add_argument("--dither", type=str, default="ordered", choices=["none","ordered","random"])
    ap.add_argument("--strength", type=float, default=0.2)
    ap.add_argument("--preview_diff", type=str, default=None)
    ap.add_argument("--strict_check", action="store_true")

    # Check-only options
    ap.add_argument("--check_only", action="store_true", help="Only verify normalization; do not write an output image")
    ap.add_argument("--stride", type=int, default=1, help="Check every Nth pixel (default 1 = all)")
    ap.add_argument("--max_report", type=int, default=50, help="Max violations to include in report")

    args = ap.parse_args()
    SPEC = cvp_defaults()
    stepL = args.stepL if args.stepL is not None else SPEC["stepL"]
    stepa = args.stepa if args.stepa is not None else SPEC["stepa"]
    stepb = args.stepb if args.stepb is not None else SPEC["stepb"]

    if args.check_only:
        res = check_normalized_image(
            args.input,
            stepL=stepL, stepa=stepa, stepb=stepb,
            stride=args.stride, max_report=args.max_report
        )
        print(json.dumps(res, indent=2))
        return

    if not args.output:
        raise SystemExit("Please provide an output path for normalization, or use --check_only.")

    res = normalize_image(
        args.input, args.output,
        stepL=stepL, stepa=stepa, stepb=stepb,
        dither=args.dither, strength=args.strength,
        preview_diff=args.preview_diff, strict_check=args.strict_check
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
