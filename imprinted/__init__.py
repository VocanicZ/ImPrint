"""Imprinted: hide a short payload in a picture, read it back from the file or a photo of a print.

    import imprinted
    imprinted.encode("photo.png", "hello", "photo_imprinted.png")
    imprinted.encode("photo_imprinted.png", "world", "again.png", mode="append")   # reads "hello world"
    imprinted.decode("phone_shot.jpg")          # -> "hello", or None
    imprinted.detect("phone_shot.jpg")          # -> True if it holds an Imprinted pattern

`imprinted.core` is the full codec (encode stats, every Imprinted found in a photo, the strength ladder).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageOps

from . import core

__version__ = "1.0.2"
__all__ = ["encode", "decode", "detect", "core"]


def _load(image) -> np.ndarray:
    """Path, PIL image or uint8 array (RGB, RGBA or grayscale) -> uint8 array, EXIF rotation applied."""
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("L", "RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB")
    return np.asarray(image)


def encode(image, payload: str | bytes, out=None, *, mode: str | None = "override", print_width_cm: float = 15.0,
           strength: float | None = None) -> Image.Image:
    """Hide payload (1..32 bytes; a str is UTF-8 encoded) in image (a path, PIL image or RGB uint8
    array). Returns the encoded picture (same size and mode) and saves it to out when given; save
    it lossless (PNG). print_width_cm: how wide it will be printed or shown, which sets the pattern
    size. strength: None takes the strongest within the visibility budget (SSIM >= 0.955, mean
    dL* <= 0.99). ValueError when the picture is too small or too clipped to carry the payload.

    An image that is already Imprinted is encoded again on its own grid and at its own strength
    (print_width_cm doesn't apply), with its old text taken out first, so exactly one text reads:
    mode "override" writes payload, "append" writes "<old text> <payload>" (ValueError over 32
    bytes, or if the old text doesn't read). mode None skips that check (it takes a second, up to
    eight on a picture under 1 MP) for pictures known to be plain."""
    if mode not in ("override", "append", None):
        raise ValueError(f"mode must be 'override', 'append' or None, not {mode!r}")
    rgb = _load(image)
    if isinstance(payload, str):
        payload = payload.encode()
    old = core.existing(rgb, read=True) if mode else None
    if old:
        enc, _, _ = core.reencode(rgb, payload, old, mode, delta=strength)
    else:
        b = core.subblock_for_print(rgb.shape[1], print_width_cm * 10)
        enc, _ = core.encode(rgb, payload, delta=strength, subblock_px=b)
    img = Image.fromarray(enc)
    if out is not None:
        img.save(out)
    return img


def decode(image, raw: bool = False, check: bool = True) -> str | bytes | None:
    """First payload found in image (a file, screenshot or photo: path, PIL image or RGB uint8
    array), or None. A str (UTF-8) unless raw, then bytes. check: detect() first and stop there if
    it finds nothing (a picture made with 0.7.x has no sync cells: pass check=False for those)."""
    found = core.decode(_load(image), max_found=1, check=check)["found"]
    if not found:
        return None
    payload = found[0]["payload"]
    return payload if raw else payload.decode("utf-8", "replace")


def detect(image) -> bool:
    """Does image hold an Imprinted pattern? Most Imprinted pictures answer within a second, most
    others are ruled out at one; the rest are searched up to ten seconds (small pictures, under
    1 MP, are tracked closely for that long instead). decode() makes the same check first."""
    return core.detect(_load(image))
