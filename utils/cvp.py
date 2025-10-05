# utils/cvp.py
# Deterministic CVP generator + strip writer/reader with invariance checks.
# Index 0 is the canonical/default; index 1 -> iOI=0, index 2 -> iOI=1, etc.
#
# USAGE (generate CVPs from a color):
#   python utils/cvp.py "102,204,153"
#   python utils/cvp.py "#66cc99" --out ./example_img/cvp_strip.png --w 40
#
# USAGE (read a CVP strip back and recheck invariance):
#   python utils/cvp.py --from_strip ./example_img/cvp_strip.png [--w 40]
#
# If --w is omitted when reading a strip, we infer tile size from the image height.

from __future__ import annotations
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import math
import json
import argparse
import os

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

ColorIn = Union[str, Tuple[int,int,int], List[int], Tuple[float,float,float], List[float]]

# ---------------- sRGB <-> Lab utilities ----------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    m = c <= 0.04045
    out = np.empty_like(c)
    out[m]  = c[m] / 12.92
    out[~m] = ((c[~m] + 0.055) / 1.055) ** 2.4
    return out

def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    m = c <= 0.0031308
    out = np.empty_like(c)
    out[m]  = 12.92 * c[m]
    out[~m] = 1.055 * (np.power(c[~m], 1/2.4)) - 0.055
    return np.clip(out, 0.0, 1.0)

def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    r = _srgb_to_linear(rgb[...,0])
    g = _srgb_to_linear(rgb[...,1])
    b = _srgb_to_linear(rgb[...,2])
    X = 0.4124564*r + 0.3575761*g + 0.1804375*b
    Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    Z = 0.0193339*r + 0.1191920*g + 0.9503041*b
    return np.stack([X,Y,Z], axis=-1)

def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    X, Y, Z = xyz[...,0], xyz[...,1], xyz[...,2]
    r =  3.2404542*X - 1.5371385*Y - 0.4985314*Z
    g = -0.9692660*X + 1.8760108*Y + 0.0415560*Z
    b =  0.0556434*X - 0.2040259*Y + 1.0572252*Z
    return np.stack([_linear_to_srgb(r), _linear_to_srgb(g), _linear_to_srgb(b)], axis=-1)

def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = xyz[...,0] / Xn
    y = xyz[...,1] / Yn
    z = xyz[...,2] / Zn
    e = 216/24389
    k = 24389/27
    def f(t):
        t3 = np.cbrt(t)
        return np.where(t > e, t3, (k*t + 16)/116)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.stack([L,a,b], axis=-1)

def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    L, a, b = lab[...,0], lab[...,1], lab[...,2]
    fy = (L + 16)/116
    fx = fy + (a/500)
    fz = fy - (b/200)
    e = 216/24389
    k = 24389/27
    def finv(f):
        f3 = f**3
        return np.where(f3 > e, f3, (116*f - 16)/k)
    x = finv(fx)*Xn
    y = finv(fy)*Yn
    z = finv(fz)*Zn
    return np.stack([x,y,z], axis=-1)

def rgb_to_lab01(rgb01: np.ndarray) -> np.ndarray:
    return xyz_to_lab(rgb_to_xyz(rgb01))

def lab_to_rgb01(lab: np.ndarray) -> np.ndarray:
    return xyz_to_rgb(lab_to_xyz(lab))

def deltaE76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    d = lab1 - lab2
    return float(np.sqrt(np.sum(d*d)))

# --------------- Parsing & helpers ----------------

def parse_color(c: ColorIn) -> Tuple[float,float,float]:
    """
    Accept '#RRGGBB', 'RRGGBB', 'r,g,b' (0..255), or (r,g,b) as ints or floats.
    Returns floats in 0..1 range.
    """
    if isinstance(c, (tuple, list)) and len(c) == 3:
        r,g,b = c
        if max(r,g,b) <= 1.0: return float(r),float(g),float(b)
        return float(r)/255.0, float(g)/255.0, float(b)/255.0

    s = str(c).strip()
    if "," in s:
        r,g,b = [int(v.strip()) for v in s.split(",")]
        return r/255.0, g/255.0, b/255.0
    if s.startswith("#"): s = s[1:]
    if len(s) != 6: raise ValueError("Color must be #RRGGBB or 'r,g,b'")
    r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
    return r/255.0, g/255.0, b/255.0

def _ring_dirs(n: int) -> np.ndarray:
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([np.cos(ang), np.sin(ang)], axis=-1).astype(np.float32)

def _rgb01_to_uint8_tuple(rgb01: np.ndarray) -> Tuple[int,int,int]:
    v = np.clip(rgb01, 0.0, 1.0)
    arr = (v*255.0 + 0.5).astype(np.uint8)
    return (int(arr[0]), int(arr[1]), int(arr[2]))

def _rgb_to_hex(rgb: Tuple[int,int,int]) -> str:
    r,g,b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

# --------------- Voxelization (perceptual buckets) ----------------
# Any color in the same voxel -> SAME canonical center & SAME CVP list.

class Voxelizer:
    def __init__(self, step_L: float = 4.0, step_a: float = 6.0, step_b: float = 6.0):
        self.step_L = float(step_L)
        self.step_a = float(step_a)
        self.step_b = float(step_b)

    def voxel_id_and_center(self, lab: np.ndarray) -> Tuple[Tuple[int,int,int], np.ndarray]:
        L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
        iL = math.floor(L / self.step_L)
        ia = math.floor((a + 128.0) / self.step_a)
        ib = math.floor((b + 128.0) / self.step_b)
        cL = (iL + 0.5)*self.step_L
        ca = (ia + 0.5)*self.step_a - 128.0
        cb = (ib + 0.5)*self.step_b - 128.0
        center = np.array([cL, ca, cb], dtype=np.float32)
        return (iL, ia, ib), center

    def inside_voxel(self, lab: np.ndarray, vid: Tuple[int,int,int]) -> bool:
        iL, ia, ib = vid
        L,a,b = lab[0], lab[1], lab[2]
        L0 = iL*self.step_L; L1 = (iL+1)*self.step_L
        a0 = ia*self.step_a - 128.0; a1 = (ia+1)*self.step_a - 128.0
        b0 = ib*self.step_b - 128.0; b1 = (ib+1)*self.step_b - 128.0
        return (L0 <= L < L1) and (a0 <= a < a1) and (b0 <= b < b1)

# --------------- CVP generator (in-voxel) ----------------

def cvp_list_for_color(
    color: ColorIn,
    step_L: float = 4.0,
    step_a: float = 6.0,
    step_b: float = 6.0,
    directions: int = 8,
    ring_count: int = 2,
    deltaE_max: float = 2.0,
    include_L_variants: bool = False
) -> List[Tuple[int,int,int]]:
    """
    Returns an ordered list of CVPs (RGB ints 0..255) for the voxel that contains `color`.
    The list is INVARIANT for any input inside that voxel:
      - item 0 = canonical/default (voxel center mapped to sRGB)
      - item 1 = first alternate (iOI=0), item 2 = iOI=1, etc.

    All generated Lab points stay INSIDE the same voxel and under ΔE <= deltaE_max,
    and are clamped to sRGB gamut.
    """
    r,g,b = parse_color(color)
    base_lab = rgb_to_lab01(np.array([[[r,g,b]]], dtype=np.float32))[0,0,:]

    vox = Voxelizer(step_L, step_a, step_b)
    vid, center_lab = vox.voxel_id_and_center(base_lab)

    center_rgb01 = lab_to_rgb01(center_lab[None,None,:])[0,0,:]
    if np.any((center_rgb01 < 0.0) | (center_rgb01 > 1.0)):
        lo, hi = 0.0, 1.0
        for _ in range(24):
            t = 0.5*(lo+hi)
            cand = (1-t)*center_lab + t*base_lab
            rgb01 = lab_to_rgb01(cand[None,None,:])[0,0,:]
            if np.any((rgb01 < 0.0) | (rgb01 > 1.0)):
                lo = t
            else:
                hi = t
                center_rgb01 = rgb01
                center_lab = cand

    default_rgb = _rgb01_to_uint8_tuple(center_rgb01)
    out: List[Tuple[int,int,int]] = [default_rgb]

    safe_ra = 0.45 * (step_a/2.0)
    safe_rb = 0.45 * (step_b/2.0)
    safe_r  = float(min(safe_ra, safe_rb, deltaE_max*0.9))
    if safe_r <= 0:
        return out

    dirs = _ring_dirs(directions)

    for ring in range(1, ring_count+1):
        radius = ring * safe_r
        for k in range(directions):
            L, a, b = center_lab[0], center_lab[1], center_lab[2]
            a2 = a + dirs[k,0] * radius
            b2 = b + dirs[k,1] * radius
            candidate = np.array([L, a2, b2], dtype=np.float32)

            if not vox.inside_voxel(candidate, vid):
                continue
            if deltaE76(center_lab, candidate) > deltaE_max:
                continue

            rgb01 = lab_to_rgb01(candidate[None,None,:])[0,0,:]
            if np.any((rgb01 < 0.0) | (rgb01 > 1.0)):
                continue

            out.append(_rgb01_to_uint8_tuple(rgb01))

    if include_L_variants:
        Lwig = min(0.45*(step_L/2.0), deltaE_max*0.5)
        for sgn in (-1.0, +1.0):
            cand = center_lab.copy()
            cand[0] = cand[0] + sgn*Lwig
            if not vox.inside_voxel(cand, vid):
                continue
            rgb01 = lab_to_rgb01(cand[None,None,:])[0,0,:]
            if np.any((rgb01 < 0.0) | (rgb01 > 1.0)):
                continue
            if deltaE76(center_lab, cand) <= deltaE_max:
                out.append(_rgb01_to_uint8_tuple(rgb01))

    return out

# --------------- Image writer (CVP strip) ----------------

def write_cvp_strip(cvp_list: List[Tuple[int,int,int]], w: int, out_path: str) -> str:
    if not PIL_OK:
        raise RuntimeError("Pillow (PIL) is required for --out image writing. Install: pip install pillow")
    n = max(1, len(cvp_list))
    H = int(w)
    W = int(w) * n
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for i, (r, g, b) in enumerate(cvp_list):
        r = int(r); g = int(g); b = int(b)
        x0 = i * w
        x1 = x0 + w
        arr[:, x0:x1, 0] = r
        arr[:, x0:x1, 1] = g
        arr[:, x0:x1, 2] = b
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_path)
    return out_path

# --------------- Image reader (CVP strip -> CVP list) ----------------

def read_cvp_strip(
    path: str,
    w: Optional[int] = None,
    step_L: float = 4.0,
    step_a: float = 6.0,
    step_b: float = 6.0,
    directions: int = 8,
    ring_count: int = 2,
    deltaE_max: float = 2.0,
    include_L_variants: bool = False
) -> Dict[str, Any]:
    """
    Read a horizontal CVP strip: one row, tiles of size w×w, left->right.
    If w is None, infer w = image height (expects a single-row strip).
    Returns:
      {
        "tile_w": int,
        "tiles_read": [{"rgb":[r,g,b],"hex":"#rrggbb"}...],
        "cvps": [...canonical CVP list from first tile...],
        "invariant_ok": bool,
        "violations": [ {tile_index:int, "reason": str} ... ]
      }
    """
    if not PIL_OK:
        raise RuntimeError("Pillow (PIL) is required for reading strips. Install: pip install pillow")

    img = Image.open(path).convert("RGB")
    W, H = img.size
    np_img = np.asarray(img, dtype=np.uint8)

    if w is None or w <= 0:
        w = H  # infer tile size from height
    if H != w:
        raise ValueError(f"Strip height {H} must match tile size w={w}. You can pass --w to override.")
    if W % w != 0:
        raise ValueError(f"Strip width {W} is not a multiple of tile size w={w}.")

    tiles_n = W // w
    tiles_read: List[Dict[str, Any]] = []

    # Sample the center pixel of each tile (strip tiles are constant color if written by write_cvp_strip)
    for i in range(tiles_n):
        x0 = i*w
        x1 = x0 + w
        cx = x0 + w//2
        cy = H//2
        rgb = tuple(int(v) for v in np_img[cy, cx, :])
        tiles_read.append({"rgb": [rgb[0], rgb[1], rgb[2]], "hex": _rgb_to_hex(rgb)})

    # Reconstruct the canonical CVP list from the FIRST tile's color
    if tiles_n == 0:
        return {
            "tile_w": w,
            "tiles_read": [],
            "cvps": [],
            "invariant_ok": True,
            "violations": []
        }

    first_rgb = tiles_read[0]["rgb"]
    seed_color = "{},{},{}".format(first_rgb[0], first_rgb[1], first_rgb[2])
    canonical = cvp_list_for_color(
        seed_color,
        step_L=step_L, step_a=step_a, step_b=step_b,
        directions=directions, ring_count=ring_count,
        deltaE_max=deltaE_max, include_L_variants=include_L_variants
    )

    # Invariance recheck: every tile's color MUST return the exact same list
    violations: List[Dict[str, Any]] = []
    for idx, t in enumerate(tiles_read):
        c = "{},{},{}".format(t["rgb"][0], t["rgb"][1], t["rgb"][2])
        test_list = cvp_list_for_color(
            c,
            step_L=step_L, step_a=step_a, step_b=step_b,
            directions=directions, ring_count=ring_count,
            deltaE_max=deltaE_max, include_L_variants=include_L_variants
        )
        if len(test_list) != len(canonical):
            violations.append({"tile_index": idx, "reason": f"length differs: {len(test_list)} vs {len(canonical)}"})
            continue
        # strict tuple-by-tuple equality
        for k in range(len(canonical)):
            if test_list[k] != canonical[k]:
                violations.append({"tile_index": idx, "reason": f"tuple {k} differs: {test_list[k]} vs {canonical[k]}"} )
                break

    return {
        "tile_w": w,
        "tiles_read": tiles_read,
        "cvps": [ [int(r),int(g),int(b)] for (r,g,b) in canonical ],
        "invariant_ok": len(violations) == 0,
        "violations": violations
    }

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Deterministic CVP list for a color (voxel-based) + strip IO.")
    ap.add_argument("color", nargs="?", default=None, help="e.g. '#66cc99' or '102,204,153' (omit if using --from_strip)")
    ap.add_argument("--stepL", type=float, default=4.0, help="voxel step for L*")
    ap.add_argument("--stepa", type=float, default=6.0, help="voxel step for a*")
    ap.add_argument("--stepb", type=float, default=6.0, help="voxel step for b*")
    ap.add_argument("--directions", type=int, default=8, help="angles per ring")
    ap.add_argument("--rings", type=int, default=2, help="rings (inside voxel)")
    ap.add_argument("--deltaE", type=float, default=2.0, help="max ΔE to canonical")
    ap.add_argument("--Lvars", action="store_true", help="include tiny ±L* variants")

    # Output strip
    ap.add_argument("--out", type=str, default=None, help="Write a CVP strip image to this path (generation mode)")
    ap.add_argument("--w", type=int, default=50, help="Tile size (pixels) for each CVP square; default 50")

    # Read strip
    ap.add_argument("--from_strip", type=str, default=None, help="Read a CVP strip image and reconstruct CVPs; also recheck invariance")

    args = ap.parse_args()

    # Mode selection
    if args.from_strip:
        res = read_cvp_strip(
            args.from_strip,
            w=args.w if args.w > 0 else None,
            step_L=args.stepL, step_a=args.stepa, step_b=args.stepb,
            directions=args.directions, ring_count=args.rings,
            deltaE_max=args.deltaE, include_L_variants=args.Lvars
        )
        print(json.dumps({
            "mode": "parse_strip",
            "tile_w": res["tile_w"],
            "tiles": res["tiles_read"],
            "cvps": res["cvps"],
            "invariant_ok": res["invariant_ok"],
            "violations": res["violations"],
            "voxel_steps": {"L": args.stepL, "a": args.stepa, "b": args.stepb},
            "settings": {
                "directions": args.directions,
                "rings": args.rings,
                "deltaE_max": args.deltaE,
                "include_L_vars": args.Lvars
            }
        }, indent=2))
        return

    if not args.color:
        raise SystemExit("Provide a color (e.g. '#66cc99') or use --from_strip <path>.")

    cvps = cvp_list_for_color(
        args.color,
        step_L=args.stepL, step_a=args.stepa, step_b=args.stepb,
        directions=args.directions, ring_count=args.rings,
        deltaE_max=args.deltaE, include_L_variants=args.Lvars
    )

    payload = []
    for idx, rgb in enumerate(cvps):
        if idx == 0:
            payload.append({"index": "default", "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])], "hex": _rgb_to_hex(rgb)})
        else:
            payload.append({"index": idx-1, "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])], "hex": _rgb_to_hex(rgb)})

    out_path = None
    if args.out:
        out_path = write_cvp_strip(cvps, max(1, int(args.w)), args.out)

    print(json.dumps({
        "mode": "generate",
        "input": str(args.color),
        "voxel_steps": {"L": args.stepL, "a": args.stepa, "b": args.stepb},
        "settings": {
            "directions": args.directions,
            "rings": args.rings,
            "deltaE_max": args.deltaE,
            "include_L_vars": args.Lvars
        },
        "count": len(cvps),
        "cvps": payload,
        "image": out_path
    }, indent=2))

if __name__ == "__main__":
    main()
