# utils/tile_grid.py
# Break an image into centered w×w tiles, save each tile, and export grid overlays.
#
# Usage examples:
#   python utils/tile_grid.py ./example_img/demo_in.png --w 14 --outdir ./tiles
#   python utils/tile_grid.py ./example_img/demo_in.png --w 14 --outdir ./tiles --preview ./tiles/preview.png
#   python utils/tile_grid.py ./example_img/demo_in.png --w 14 --outdir ./tiles --preview_labeled ./tiles/preview_labeled.png
#   python utils/tile_grid.py ./example_img/demo_in.png --w 14 --outdir ./tiles --preview ./tiles/preview.png \
#       --grid_color 0,255,0 --outer_color 255,0,0 --linew 2 --alpha 128 --label_every 1 --label_font_size 12
#
# Notes:
# - The grid is centered; you get as many w×w tiles as fit without resizing.
# - Tiles are saved as PNG (RGB). Filenames include zero-padded row/col.
# - --preview draws gridlines over the original.
# - --preview_labeled draws gridlines + tile indices (r,c) at tile centers (optionally every N tiles).

from __future__ import annotations
import os
import json
import argparse
from typing import Dict, Any, List, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    raise SystemExit("Pillow is required. Install with: pip install pillow") from e


def parse_rgb(s: str) -> Tuple[int, int, int]:
    """Parse 'R,G,B' into a 3-tuple of ints in 0..255."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"RGB must be 'R,G,B', got: {s}")
    vals = tuple(int(p) for p in parts)
    for v in vals:
        if not (0 <= v <= 255):
            raise ValueError(f"RGB component out of range: {vals}")
    return vals  # type: ignore[return-value]


def compute_centered_grid(W: int, H: int, w: int) -> Dict[str, int]:
    if w <= 0:
        raise ValueError("w must be > 0")
    cols = W // w
    rows = H // w
    used_w = cols * w
    used_h = rows * w
    off_x = (W - used_w) // 2
    off_y = (H - used_h) // 2
    return {
        "rows": rows, "cols": cols,
        "used_w": used_w, "used_h": used_h,
        "off_x": off_x, "off_y": off_y,
    }


def slice_tiles(
    img: Image.Image, w: int, outdir: str, prefix: str = "tile", ext: str = "png"
) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    W, H = img.size
    meta = compute_centered_grid(W, H, w)

    rows = meta["rows"]; cols = meta["cols"]
    off_x = meta["off_x"]; off_y = meta["off_y"]
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Image ({W}x{H}) too small for tiles of {w}x{w}.")

    if img.mode != "RGB":
        img = img.convert("RGB")

    rpad = max(2, len(str(rows - 1)))
    cpad = max(2, len(str(cols - 1)))

    tiles: List[Dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            x0 = off_x + c * w; y0 = off_y + r * w
            x1 = x0 + w;        y1 = y0 + w
            tile = img.crop((x0, y0, x1, y1))
            fname = f"{prefix}_r{str(r).zfill(rpad)}_c{str(c).zfill(cpad)}.{ext}"
            fpath = os.path.join(outdir, fname)
            tile.save(fpath)
            tiles.append({
                "r": int(r), "c": int(c),
                "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
                "file": fpath.replace("\\", "/"),
            })

    return {
        "input_size": {"W": int(W), "H": int(H)},
        "tile_size": int(w),
        "grid": {
            "rows": int(rows), "cols": int(cols),
            "used_w": int(meta["used_w"]), "used_h": int(meta["used_h"]),
            "off_x": int(off_x), "off_y": int(off_y),
        },
        "count": len(tiles),
        "outdir": outdir.replace("\\", "/"),
        "tiles": tiles,
    }


def _overlay_base(
    base_img: Image.Image,
    w: int,
    grid_meta: Dict[str, Any],
    grid_color: Tuple[int, int, int],
    outer_color: Tuple[int, int, int],
    line_width: int,
    alpha: int,
) -> Image.Image:
    """
    Returns a copy of base_img with a semi-transparent overlay containing the grid lines.
    """
    if base_img.mode != "RGB":
        base_img = base_img.convert("RGB")

    W, H = base_img.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    rows = grid_meta["grid"]["rows"]
    cols = grid_meta["grid"]["cols"]
    off_x = grid_meta["grid"]["off_x"]
    off_y = grid_meta["grid"]["off_y"]

    # Outer rectangle
    x0 = off_x; y0 = off_y
    x1 = off_x + cols * w; y1 = off_y + rows * w
    oc = (*outer_color, alpha)
    for lw in range(line_width):
        draw.rectangle([x0 - lw, y0 - lw, x1 + lw - 1, y1 + lw - 1], outline=oc)

    # Grid lines
    gc = (*grid_color, alpha)
    for c in range(1, cols):
        x = off_x + c * w
        draw.line([(x, y0), (x, y1 - 1)], fill=gc, width=line_width)
    for r in range(1, rows):
        y = off_y + r * w
        draw.line([(x0, y), (x1 - 1, y)], fill=gc, width=line_width)

    combined = base_img.convert("RGBA")
    combined.alpha_composite(overlay)
    return combined.convert("RGB")


def draw_preview_grid(
    img: Image.Image,
    w: int,
    meta: Dict[str, Any],
    out_path: str,
    grid_color: Tuple[int, int, int] = (0, 255, 0),
    outer_color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 2,
    alpha: int = 160,
) -> str:
    out = _overlay_base(img, w, meta, grid_color, outer_color, line_width, alpha)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.save(out_path)
    return out_path


def draw_preview_grid_labeled(
    img: Image.Image,
    w: int,
    meta: Dict[str, Any],
    out_path: str,
    grid_color: Tuple[int, int, int] = (0, 255, 0),
    outer_color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 2,
    alpha: int = 160,
    label_color: Tuple[int, int, int] = (255, 255, 255),
    label_font_size: int = 12,
    label_every: int = 1,
) -> str:
    """
    Grid overlay + draw '(r,c)' at each tile center (or every N tiles).
    """
    base = _overlay_base(img, w, meta, grid_color, outer_color, line_width, alpha)
    draw = ImageDraw.Draw(base)

    try:
        # Try to find a common TTF; fallback to default.
        font = ImageFont.truetype("arial.ttf", label_font_size)
    except Exception:
        font = ImageFont.load_default()

    rows = meta["grid"]["rows"]; cols = meta["grid"]["cols"]
    off_x = meta["grid"]["off_x"]; off_y = meta["grid"]["off_y"]

    for r in range(0, rows, max(1, label_every)):
        for c in range(0, cols, max(1, label_every)):
            cx = off_x + c * w + w // 2
            cy = off_y + r * w + w // 2
            text = f"({r},{c})"
            # Center the text
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            draw.rectangle([cx - tw//2 - 2, cy - th//2 - 1, cx + tw//2 + 2, cy + th//2 + 1],
                           fill=(0, 0, 0, 120))
            draw.text((cx - tw//2, cy - th//2), text, fill=label_color, font=font)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    base.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Center-cropped w×w tiler with grid previews.")
    ap.add_argument("image", type=str, help="Path to input image (any size).")
    ap.add_argument("--w", type=int, required=True, help="Tile size (iPS): w×w pixels.")
    ap.add_argument("--outdir", type=str, default="./tiles", help="Directory to write tiles.")
    ap.add_argument("--prefix", type=str, default="tile", help="Filename prefix for tiles.")
    ap.add_argument("--ext", type=str, default="png", help="Tile file extension (png/jpg...).")

    # Simple grid overlay (original + gridlines)
    ap.add_argument("--preview", type=str, default=None, help="Save grid overlay over the original image.")

    # Labeled grid overlay (original + gridlines + '(r,c)' per tile)
    ap.add_argument("--preview_labeled", type=str, default=None, help="Save labeled grid overlay.")
    ap.add_argument("--label_every", type=int, default=1, help="Label every N tiles (default 1 = every tile).")
    ap.add_argument("--label_font_size", type=int, default=12, help="Label font size.")
    ap.add_argument("--label_color", type=str, default="255,255,255", help="Label RGB, e.g. '255,255,255'.")

    # Style knobs for overlays
    ap.add_argument("--grid_color", type=str, default="0,255,0", help="Grid RGB for inner lines.")
    ap.add_argument("--outer_color", type=str, default="255,0,0", help="Outer box RGB.")
    ap.add_argument("--linew", type=int, default=2, help="Grid line width.")
    ap.add_argument("--alpha", type=int, default=160, help="Gridline alpha 0..255.")

    args = ap.parse_args()

    # Load + slice
    img = Image.open(args.image)
    meta = slice_tiles(img, args.w, args.outdir, prefix=args.prefix, ext=args.ext)

    # Overlays
    result_paths: Dict[str, str] = {}
    grid_rgb = parse_rgb(args.grid_color)
    outer_rgb = parse_rgb(args.outer_color)
    label_rgb = parse_rgb(args.label_color)
    alpha = max(0, min(255, int(args.alpha)))

    if args.preview:
        p = draw_preview_grid(
            img.copy(), args.w, meta, args.preview,
            grid_color=grid_rgb, outer_color=outer_rgb,
            line_width=args.linew, alpha=alpha
        )
        result_paths["preview"] = p.replace("\\", "/")

    if args.preview_labeled:
        p = draw_preview_grid_labeled(
            img.copy(), args.w, meta, args.preview_labeled,
            grid_color=grid_rgb, outer_color=outer_rgb,
            line_width=args.linew, alpha=alpha,
            label_color=label_rgb,
            label_font_size=args.label_font_size,
            label_every=max(1, args.label_every),
        )
        result_paths["preview_labeled"] = p.replace("\\", "/")

    print(json.dumps({
        "verdict": "OK",
        "input": args.image.replace("\\", "/"),
        "result": meta,
        "overlays": result_paths
    }, indent=2))


if __name__ == "__main__":
    main()
