# utils/tacf.py
# Draw 3 TACF (Triple Anchor Corner Fiducials) on a centered w×w grid.
# TACF size is iCP tiles per side (so side = iCP * w pixels).
#
# Example:
#   python utils/tacf.py ./example_img/demo_in.png \
#       --w 14 -iCP 5 --out ./example_img/demo_with_tacf.png \
#       --preview ./example_img/tacf_preview.png \
#       --mask ./example_img/tacf_mask.png
#
# Notes:
# - Grid is centered exactly like utils/tile_grid.py (no import needed).
# - TACF are drawn fully aligned to the grid; they won’t spill outside.
# - “QR-style” triple nested squares: white outer, black middle ring, white core.
# - iCP is in *tiles* (not pixels). iCP >= 3 recommended.

from __future__ import annotations
import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

try:
    from PIL import Image, ImageDraw
except Exception as e:
    raise SystemExit("Pillow is required. Install with: pip install pillow") from e


@dataclass
class GridMeta:
    rows: int
    cols: int
    used_w: int
    used_h: int
    off_x: int
    off_y: int
    tile: int  # w


def compute_centered_grid(W: int, H: int, w: int) -> GridMeta:
    if w <= 0:
        raise ValueError("w (tile size) must be > 0")
    cols = W // w
    rows = H // w
    used_w = cols * w
    used_h = rows * w
    off_x = (W - used_w) // 2
    off_y = (H - used_h) // 2
    return GridMeta(rows=rows, cols=cols, used_w=used_w, used_h=used_h, off_x=off_x, off_y=off_y, tile=w)


def _clamp_icp(icp: int, rows: int, cols: int) -> int:
    # Fit inside the grid; leave at least 1 tile breathing room if possible
    max_side = min(rows, cols)
    icp = max(3, min(icp, max_side))  # at least 3 tiles, at most grid min
    return icp


def _tacf_boxes(top_left_xy: Tuple[int, int], side: int, band: int) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    """
    Given top-left pixel (x0,y0), total side in px, and ring band thickness in px,
    return (outer_box, mid_box, inner_box) rectangles for the nested squares.
    """
    x0, y0 = top_left_xy
    s  = side
    b  = band
    outer = (x0, y0, x0 + s - 1, y0 + s - 1)
    mid   = (x0 + b, y0 + b, x0 + s - 1 - b, y0 + s - 1 - b)
    inner = (x0 + 2*b, y0 + 2*b, x0 + s - 1 - 2*b, y0 + s - 1 - 2*b)
    return outer, mid, inner


def _draw_tacf(draw: ImageDraw.ImageDraw, x0: int, y0: int, side: int, band: int,
               color_outer=(255,255,255), color_mid=(0,0,0), color_inner=(255,255,255)) -> Dict[str, Any]:
    """
    Draw the triple nested squares at (x0,y0), size=side, band=ring thickness.
    Returns boxes metadata.
    """
    outer, mid, inner = _tacf_boxes((x0, y0), side, band)
    # Fill mid with black, then "carve" inner white and outer white border by rectangles.
    # Simpler: draw solid outer (white), then solid mid (black), then inner (white).
    draw.rectangle(outer, fill=color_outer)
    draw.rectangle(mid,   fill=color_mid)
    draw.rectangle(inner, fill=color_inner)

    return {
        "outer": list(outer), "mid": list(mid), "inner": list(inner),
        "top_left": [x0, y0], "side": side, "band": band
    }


def _anchor_positions(meta: GridMeta, icp: int) -> Dict[str, Dict[str,int]]:
    w = meta.tile
    side_px = icp * w
    # Top-left anchor (TL)
    tl_x = meta.off_x
    tl_y = meta.off_y
    # Top-right anchor (TR)
    tr_x = meta.off_x + meta.cols * w - side_px
    tr_y = meta.off_y
    # Bottom-left anchor (BL)
    bl_x = meta.off_x
    bl_y = meta.off_y + meta.rows * w - side_px
    return {
        "tl": {"x": tl_x, "y": tl_y, "side": side_px},
        "tr": {"x": tr_x, "y": tr_y, "side": side_px},
        "bl": {"x": bl_x, "y": bl_y, "side": side_px},
    }


def _anchor_tile_ranges(meta: GridMeta, icp: int) -> Dict[str, Dict[str, Tuple[int,int]]]:
    # In tile units (grid indices), 0-based
    return {
        "tl": {"rows": (0, icp-1), "cols": (0, icp-1)},
        "tr": {"rows": (0, icp-1), "cols": (meta.cols-icp, meta.cols-1)},
        "bl": {"rows": (meta.rows-icp, meta.rows-1), "cols": (0, icp-1)},
    }


def draw_tacf_on_image(
    img: Image.Image,
    w: int,
    iCP: int,
    band_tiles: int = 1,  # ring thickness in tiles (1 tile thick by default)
    color_outer=(255,255,255),
    color_mid=(0,0,0),
    color_inner=(255,255,255),
) -> Tuple[Image.Image, Dict[str, Any]]:
    if img.mode != "RGB":
        img = img.convert("RGB")

    W, H = img.size
    meta = compute_centered_grid(W, H, w)
    icp = _clamp_icp(iCP, meta.rows, meta.cols)

    # Ensure the anchors fit fully inside the grid area
    if icp > meta.rows or icp > meta.cols:
        raise ValueError(f"iCP too large for this image/grid: icp={icp}, grid={meta.rows}x{meta.cols} tiles")

    side_px = icp * w
    band_px = max(1, band_tiles * w // 3)  # visually reasonable ring thickness

    out = img.copy()
    draw = ImageDraw.Draw(out)

    # Compute anchor positions (pixel)
    anchors_px = _anchor_positions(meta, icp)
    boxes = {}
    for name, pos in anchors_px.items():
        boxes[name] = _draw_tacf(
            draw, pos["x"], pos["y"], pos["side"], band_px,
            color_outer=color_outer, color_mid=color_mid, color_inner=color_inner
        )

    # Tile index ranges covered by anchors
    anchor_tiles = _anchor_tile_ranges(meta, icp)

    return out, {
        "grid": {
            "rows": meta.rows, "cols": meta.cols,
            "off_x": meta.off_x, "off_y": meta.off_y,
            "used_w": meta.used_w, "used_h": meta.used_h,
            "tile": meta.tile
        },
        "iCP_tiles": icp,
        "band_tiles": band_tiles,
        "anchors_px": anchors_px,
        "anchors_boxes": boxes,
        "anchors_tiles": anchor_tiles
    }


def overlay_grid(base: Image.Image, meta: GridMeta,
                 grid_rgb=(0,255,0), outer_rgb=(255,0,0), linew=2, alpha=160) -> Image.Image:
    if base.mode != "RGB":
        base = base.convert("RGB")
    W, H = base.size
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(overlay)

    w = meta.tile
    x0 = meta.off_x; y0 = meta.off_y
    x1 = x0 + meta.cols * w; y1 = y0 + meta.rows * w

    oc = (*outer_rgb, alpha)
    for lw in range(linew):
        d.rectangle([x0 - lw, y0 - lw, x1 + lw - 1, y1 + lw - 1], outline=oc)

    gc = (*grid_rgb, alpha)
    for c in range(1, meta.cols):
        x = x0 + c*w
        d.line([(x, y0), (x, y1-1)], fill=gc, width=linew)
    for r in range(1, meta.rows):
        y = y0 + r*w
        d.line([(x0, y), (x1-1, y)], fill=gc, width=linew)

    out = base.convert("RGBA")
    out.alpha_composite(overlay)
    return out.convert("RGB")


def make_mask(W: int, H: int, anchors_px: Dict[str, Dict[str, int]]) -> Image.Image:
    """
    Binary mask: 255 inside TACF rectangles, 0 elsewhere (outer squares).
    """
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    for pos in anchors_px.values():
        x0, y0 = pos["x"], pos["y"]
        s = pos["side"]
        d.rectangle([x0, y0, x0 + s - 1, y0 + s - 1], fill=255)
    return mask


def parse_rgb(s: str) -> Tuple[int,int,int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("Color must be 'R,G,B'")
    vals = tuple(int(p) for p in parts)
    for v in vals:
        if v < 0 or v > 255:
            raise ValueError("RGB components must be 0..255")
    return vals  # type: ignore[return-value]


def main():
    ap = argparse.ArgumentParser(description="Add TACF anchors aligned to a centered w×w grid.")
    ap.add_argument("image", type=str, help="Input image path")
    ap.add_argument("--out", type=str, required=True, help="Output image with TACF")
    ap.add_argument("--w", type=int, required=True, help="Tile size (iPS) in pixels")
    ap.add_argument("-iCP", "--iCP", type=int, required=True, help="Anchor side in tiles (iCP)")
    ap.add_argument("--band_tiles", type=int, default=1, help="Ring thickness in tiles (default 1)")
    ap.add_argument("--outer_color", type=str, default="255,255,255", help="Outer square color (R,G,B)")
    ap.add_argument("--mid_color",   type=str, default="0,0,0",       help="Middle ring color (R,G,B)")
    ap.add_argument("--inner_color", type=str, default="255,255,255", help="Inner square color (R,G,B)")
    ap.add_argument("--preview", type=str, default=None, help="Optional grid overlay preview path")
    ap.add_argument("--mask", type=str, default=None, help="Optional binary mask path (anchors region)")
    ap.add_argument("--json", type=str, default=None, help="Optional metadata JSON path")

    # Grid overlay styling (for --preview)
    ap.add_argument("--grid_color", type=str, default="0,255,0")
    ap.add_argument("--outer_box_color", type=str, default="255,0,0")
    ap.add_argument("--linew", type=int, default=2)
    ap.add_argument("--alpha", type=int, default=160)

    args = ap.parse_args()

    img = Image.open(args.image)
    W, H = img.size
    grid = compute_centered_grid(W, H, args.w)

    out_img, meta = draw_tacf_on_image(
        img, args.w, args.iCP, band_tiles=args.band_tiles,
        color_outer=parse_rgb(args.outer_color),
        color_mid=parse_rgb(args.mid_color),
        color_inner=parse_rgb(args.inner_color),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_img.save(args.out)

    overlays: Dict[str, str] = {}
    if args.preview:
        prev = overlay_grid(
            img.copy(), grid,
            grid_rgb=parse_rgb(args.grid_color),
            outer_rgb=parse_rgb(args.outer_box_color),
            linew=args.linew,
            alpha=max(0, min(255, int(args.alpha)))
        )
        prev.save(args.preview)
        overlays["preview"] = args.preview.replace("\\", "/")

    if args.mask:
        m = make_mask(W, H, meta["anchors_px"])
        os.makedirs(os.path.dirname(args.mask) or ".", exist_ok=True)
        m.save(args.mask)
        overlays["mask"] = args.mask.replace("\\", "/")

    payload = {
        "verdict": "OK",
        "input": args.image.replace("\\", "/"),
        "output": args.out.replace("\\", "/"),
        "grid": meta["grid"],
        "iCP_tiles": meta["iCP_tiles"],
        "band_tiles": meta["band_tiles"],
        "anchors_px": meta["anchors_px"],
        "anchors_tiles": meta["anchors_tiles"],
        "overlays": overlays
    }

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
