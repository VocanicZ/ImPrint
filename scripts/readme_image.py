"""Regenerate the README figure (versus.png) with the current imprinted encoder.

Run:  .venv/bin/python scripts/readme_image.py
Rows: original | encoded at default strength; the web UI's finished decode of each; a simulated
phone photo of a curled 15 cm print | its decode. Exits non-zero unless the encoded file and the
photo decode and the original reads nothing.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import imprinted  # noqa: E402
import imprinted.core as ic  # noqa: E402
import sim  # noqa: E402
import web  # noqa: E402

PAYLOAD = b"Imprinted"
PANEL_W, PANEL_H, HEAD, FOOT, PAD = 768, 512, 64, 40, 24
BG, FG = (32, 32, 32), (235, 235, 235)
GREEN, RED, BLUE, ORANGE = (43, 217, 107), (224, 85, 85), (63, 167, 255), (255, 176, 32)
CELL = [BLUE, GREEN, (255, 77, 77), (255, 255, 255, 166)]     # sync, bit right, bit wrong, unknown


def font(px: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", px)
    except OSError:
        return ImageFont.load_default(px)


def webui_decode(rgb: np.ndarray) -> tuple[np.ndarray, list]:
    """The web UI decode tab's finished picture and the payloads read: the events web.py streams,
    drawn as web/index.html's tracer draws them. On the photo: the quick check's probes, sync tests
    and grid. Over it: the last spot read, its probes, fitted grid, cells and the payload's quad."""
    evs = []
    found = [f["payload"] for f in ic.decode(rgb, max_found=1, trace=web.forward(evs.append))["found"]]
    k = min(1.0, 1200 / max(rgb.shape[:2]))                  # the page draws on web.b64jpg's preview
    small = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else rgb
    ctx = Image.fromarray((small * 0.75).astype(np.uint8)).convert("RGBA")     # under rgba(0,0,0,.25)
    P = lambda p: (p[0] * k, p[1] * k)  # noqa: E731

    def dot(d, p, r, **kw):
        x, y = P(p)
        d.ellipse((x - r, y - r, x + r, y + r), **kw)

    d = ImageDraw.Draw(sheet := Image.new("RGBA", ctx.size))
    for ev in evs:
        if ev["e"] == "probe":
            dot(d, ev["xy"], 6 if ev["locks"] >= 4 else 4,
                fill=BLUE if ev["locks"] >= 4 else ORANGE if ev["locks"] >= 3 else (170, 170, 170, 178))
        elif ev["e"] == "sync":
            col = GREEN if ev["z"] >= 6 else RED
            dot(d, ev["xy"], 13, outline=col, width=3)
            d.text((ev["xy"][0] * k + 16, ev["xy"][1] * k - 8), f"z {ev['z']}", col, font(13), anchor="ls")
        elif ev["e"] == "checked" and ev["ok"]:              # the grid the check found, 6 cells each way
            o, e1, e2 = (np.array(ev[n]) for n in ("xy", "e1", "e2"))
            Q = lambda a, b: P(o + a * e1 + b * e2)  # noqa: E731
            for i in range(-6, 7):
                d.line([Q(i, -6), Q(i, 6)], (*GREEN, 204), 1)
                d.line([Q(-6, i), Q(6, i)], (*GREEN, 204), 1)
        elif ev["e"] == "seed":
            dot(d, ev["xy"], 3, fill=(*GREEN, 153))
    ctx.alpha_composite(sheet)

    last = max((i for i, ev in enumerate(evs) if ev["e"] == "twin"), default=len(evs))  # each spot clears the layer
    read = evs[last:]
    d = ImageDraw.Draw(sheet := Image.new("RGBA", ctx.size))
    for ok, col, r in ((False, (*RED, 191), 1.6), (True, GREEN, 2.2)):
        for ev in read:
            if ev["e"] == "try" and ev["ok"] == ok:
                dot(d, ev["xy"], r, fill=col)
    fits = [ev for ev in read if ev["e"] in ("tracked", "refined") and ev["lines"]]
    for line in fits[-1]["lines"] if fits else []:
        d.line([P(p) for p in line], (63, 208, 255) if fits[-1]["e"] == "tracked" else (176, 124, 255), 1)
    for ev in read:
        if ev["e"] == "cells":
            r = max(1.5, ev["cell"] * k * 0.3)
            for (x, y), kind in zip(np.reshape(ev["xy"], (-1, 2)) * k, ev["kind"]):
                d.rectangle((x - r, y - r, x + r, y + r), CELL[kind])
        elif ev["e"] == "found":                         # its own sheet: PIL replaces pixels, canvas blends
            ctx.alpha_composite(sheet)
            d = ImageDraw.Draw(sheet := Image.new("RGBA", ctx.size))
            q = [P(p) for p in ev["quad"]]
            d.polygon(q, (*GREEN, 31), GREEN, 3)
            cx, cy = np.mean(q, axis=0)
            tw = d.textlength(ev["text"], font(28))
            d.rectangle((cx - tw / 2 - 10, cy - 30, cx + tw / 2 + 10, cy + 12), (0, 0, 0, 191))
            d.text((cx - tw / 2, cy), ev["text"], (255, 255, 255), font(28), anchor="ls")
    ctx.alpha_composite(sheet)
    if not found:                                             # the page tints a decode that read nothing red
        ctx.alpha_composite(Image.new("RGBA", ctx.size, (210, 59, 59, 71)))
    return np.asarray(ctx.convert("RGB")), found


def panel(img: np.ndarray, title: str, caption: str = "") -> np.ndarray:
    f = min(PANEL_W / img.shape[1], PANEL_H / img.shape[0])
    img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    out = np.full((HEAD + PANEL_H + FOOT, PANEL_W, 3), BG, np.uint8)
    y, x = HEAD + (PANEL_H - img.shape[0]) // 2, (PANEL_W - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.putText(out, title, (0, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, FG, 2, cv2.LINE_AA)
    cv2.putText(out, caption, (0, HEAD + PANEL_H + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, FG, 1, cv2.LINE_AA)
    return out


def main():
    rev = "v" + imprinted.__version__
    img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, "testdata", "kodim23.png")), cv2.COLOR_BGR2RGB)
    enc, st = ic.encode(img, PAYLOAD, subblock_px=ic.subblock_for_print(img.shape[1], 150))
    orig_view, orig = webui_decode(img)
    enc_view, clean = webui_decode(enc)
    photo = sim.channel(enc, np.random.default_rng(1), "curled")
    photo_view, got = webui_decode(photo)
    print(f"delta {st['delta']}  SSIM {st['ssim']}  mean dL {st['mean_dL']}  clean {clean}  photo {got}  original {orig}")
    said = lambda found: f'"{found[0].decode()}"' if found else "nothing"  # noqa: E731
    pairs = [(panel(img, "Original"),
              panel(enc, f"Imprinted {rev}", f'payload "{PAYLOAD.decode()}"   strength {st["delta"]}   '
                                             f'SSIM {st["ssim"]}   mean dL* {st["mean_dL"]}')),
             (panel(orig_view, f"Original decoded -> {said(orig)}"),
              panel(enc_view, f"Imprinted decoded -> {said(clean)}")),
             (panel(photo, "Curled print photo", "phone photo of a 15 cm print, simulated by sim.py"),
              panel(photo_view, f"Curled photo decoded -> {said(got)}"))]
    gap = np.full((HEAD + PANEL_H + FOOT, PAD, 3), BG, np.uint8)
    rows = [np.hstack([a, gap, b]) for a, b in pairs]
    space = np.full((PAD, rows[0].shape[1], 3), BG, np.uint8)
    fig = np.vstack([rows[0], space, rows[1], space, rows[2]])
    fig = cv2.copyMakeBorder(fig, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=BG)
    cv2.imwrite(os.path.join(ROOT, "versus.png"), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
    sys.exit(0 if PAYLOAD in clean and PAYLOAD in got and not orig else 1)


if __name__ == "__main__":
    main()
