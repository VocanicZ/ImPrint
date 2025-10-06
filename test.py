# test.py
# Compose TACF + centered tile grid on an image using utils/tacf.py and utils/tile_grid.py
#
# Examples:
#   python test.py ./example_img/demo_in.png --w 14 -iCP 5 \
#       --out_tacf ./example_img/out_tacf.png \
#       --out_grid ./example_img/out_grid.png \
#       --out_both ./example_img/out_both.png \
#       --json ./example_img/out_meta.json
#
# Notes:
# - --w is the iPS tile size in pixels.
# - -iCP is the TACF side length in tiles (each anchor is iCP x iCP tiles).
# - Outputs:
#     out_tacf  : original + TACF anchors
#     out_grid  : original + grid overlay (no anchors)
#     out_both  : TACF image + grid overlay (anchors + grid)
# - A JSON summary is printed and (optionally) written with --json.

from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Any, Dict

try:
    from PIL import Image
except Exception as e:
    raise SystemExit("Pillow is required. Install with: pip install pillow") from e

# Make sure we can import from ./utils
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(THIS_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

# Import our utilities
try:
    import tacf
except Exception as e:
    raise SystemExit("Unable to import utils/tacf.py — did you save it?") from e

try:
    import tile_grid
except Exception as e:
    raise SystemExit("Unable to import utils/tile_grid.py — did you save it?") from e


def _as_int(d: Dict[str, Any], key: str, default: int = 0) -> int:
    v = d.get(key, default)
    try:
        return int(v)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser(description="Generate TACF anchors + centered grid overlays.")
    ap.add_argument("image", type=str, help="Input image path")
    ap.add_argument("--w", type=int, required=True, help="Tile size (iPS) in pixels")
    ap.add_argument("-iCP", "--iCP", type=int, required=True, help="TACF side length in tiles (>=3 recommended)")
    ap.add_argument("--band_tiles", type=int, default=1, help="TACF ring thickness in tiles (default 1)")
    ap.add_argument("--out_tacf", type=str, default=None, help="Output path for image with TACF only")
    ap.add_argument("--out_grid", type=str, default=None, help="Output path for image with grid overlay only")
    ap.add_argument("--out_both", type=str, default=None, help="Output path for image with TACF + grid overlay")
    ap.add_argument("--json", type=str, default=None, help="Optional JSON metadata output path")

    # Styling for overlays
    ap.add_argument("--outer_color", type=str, default="255,255,255", help="TACF outer color R,G,B")
    ap.add_argument("--mid_color",   type=str, default="0,0,0",       help="TACF mid color R,G,B")
    ap.add_argument("--inner_color", type=str, default="255,255,255", help="TACF inner color R,G,B")

    ap.add_argument("--grid_color", type=str, default="0,255,0", help="Grid line color R,G,B")
    ap.add_argument("--outer_box_color", type=str, default="255,0,0", help="Outer box color R,G,B")
    ap.add_argument("--linew", type=int, default=2, help="Grid line width (px)")
    ap.add_argument("--alpha", type=int, default=160, help="Grid overlay alpha 0..255")

    args = ap.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f"Input not found: {args.image}")

    def parse_rgb(s: str):
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Color must be 'R,G,B', got: {s}")
        vals = tuple(int(p) for p in parts)
        for v in vals:
            if v < 0 or v > 255:
                raise ValueError(f"RGB components must be 0..255, got: {vals}")
        return vals

    # Open image
    img = Image.open(args.image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size

    # Use tile_grid to compute a centered grid
    grid = tile_grid.compute_centered_grid(W, H, args.w)
    # grid may be dict or a dataclass; normalize fields:
    rows = getattr(grid, "rows", grid.get("rows"))
    cols = getattr(grid, "cols", grid.get("cols"))
    used_w = getattr(grid, "used_w", grid.get("used_w"))
    used_h = getattr(grid, "used_h", grid.get("used_h"))
    off_x = getattr(grid, "off_x", grid.get("off_x"))
    off_y = getattr(grid, "off_y", grid.get("off_y"))
    tile  = getattr(grid, "tile",  args.w)

    # Build a tacf.GridMeta for overlay convenience
    grid_meta = tacf.GridMeta(rows=rows, cols=cols, used_w=used_w, used_h=used_h,
                              off_x=off_x, off_y=off_y, tile=tile)

    # Draw TACF using utils/tacf
    tacf_img, tacf_meta = tacf.draw_tacf_on_image(
        img, args.w, args.iCP, band_tiles=args.band_tiles,
        color_outer=parse_rgb(args.outer_color),
        color_mid=parse_rgb(args.mid_color),
        color_inner=parse_rgb(args.inner_color),
    )

    # Prepare outputs (defaults next to input)
    base, ext = os.path.splitext(os.path.basename(args.image))
    out_tacf = args.out_tacf or os.path.join(os.path.dirname(args.image), f"{base}_tacf{ext}")
    out_grid = args.out_grid or os.path.join(os.path.dirname(args.image), f"{base}_grid{ext}")
    out_both = args.out_both or os.path.join(os.path.dirname(args.image), f"{base}_tacf_grid{ext}")

    # Save TACF image
    os.makedirs(os.path.dirname(out_tacf) or ".", exist_ok=True)
    tacf_img.save(out_tacf)

    # Grid overlay on original (no anchors)
    grid_only = tacf.overlay_grid(img.copy(), grid_meta,
                                  grid_rgb=parse_rgb(args.grid_color),
                                  outer_rgb=parse_rgb(args.outer_box_color),
                                  linew=args.linew,
                                  alpha=max(0, min(255, int(args.alpha))))
    os.makedirs(os.path.dirname(out_grid) or ".", exist_ok=True)
    grid_only.save(out_grid)

    # Grid overlay on TACF image (anchors + grid)
    both = tacf.overlay_grid(tacf_img.copy(), grid_meta,
                             grid_rgb=parse_rgb(args.grid_color),
                             outer_rgb=parse_rgb(args.outer_box_color),
                             linew=args.linew,
                             alpha=max(0, min(255, int(args.alpha))))
    os.makedirs(os.path.dirname(out_both) or ".", exist_ok=True)
    both.save(out_both)

    payload = {
        "verdict": "OK",
        "input": args.image.replace("\\", "/"),
        "w_tile": args.w,
        "iCP_tiles": args.iCP,
        "band_tiles": args.band_tiles,
        "grid": {
            "rows": rows, "cols": cols,
            "off_x": off_x, "off_y": off_y,
            "used_w": used_w, "used_h": used_h,
            "tile": tile
        },
        "outputs": {
            "tacf": out_tacf.replace("\\", "/"),
            "grid": out_grid.replace("\\", "/"),
            "tacf_plus_grid": out_both.replace("\\", "/")
        },
        "anchors_px": tacf_meta["anchors_px"],
        "anchors_tiles": tacf_meta["anchors_tiles"]
    }

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
