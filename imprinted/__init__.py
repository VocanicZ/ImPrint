"""Imprinted: hide a short payload in a picture, read it back from the file or a photo of a print.

    import imprinted
    imprinted.encode("photo.png", "hello", "photo_imprinted.png")
    imprinted.decode("phone_shot.jpg")          # -> "hello", or None
    imprinted.detect("phone_shot.jpg")          # -> True if it holds an Imprinted pattern (under a second)

`imprinted.core` is the full codec (encode stats, every Imprinted found in a photo, the strength ladder).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageOps

from . import core

__version__ = "0.7.1"
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


def encode(image, payload: str | bytes, out=None, *, print_width_cm: float = 15.0,
           strength: float | None = None) -> Image.Image:
    """Hide payload (1..32 bytes; a str is UTF-8 encoded) in image (a path, PIL image or RGB uint8
    array). Returns the encoded picture (same size and mode) and saves it to out when given; save
    it lossless (PNG). print_width_cm: how wide it will be printed or shown, which sets the pattern
    size. strength: None takes the strongest within the visibility budget (SSIM >= 0.955, mean
    dL* <= 0.99). ValueError when the picture is too small or too clipped to carry the payload."""
    rgb = _load(image)
    if isinstance(payload, str):
        payload = payload.encode()
    b = core.subblock_for_print(rgb.shape[1], print_width_cm * 10)
    enc, _ = core.encode(rgb, payload, delta=strength, subblock_px=b)
    img = Image.fromarray(enc)
    if out is not None:
        img.save(out)
    return img


def decode(image, raw: bool = False) -> str | bytes | None:
    """First payload found in image (a file, screenshot or photo: path, PIL image or RGB uint8
    array), or None. A str (UTF-8) unless raw, then bytes."""
    found = core.decode(_load(image), max_found=1)["found"]
    if not found:
        return None
    payload = found[0]["payload"]
    return payload if raw else payload.decode("utf-8", "replace")


def detect(image) -> bool:
    """Does image hold an Imprinted pattern? Quick (under a second at any size), without reading the
    payload: decode() makes the same check first, so a picture without one costs no more than this."""
    return bool(core.quick_check(_load(image)))
