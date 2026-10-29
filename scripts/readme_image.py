"""Regenerate the README figure (versus.png) with the current imprint_cam encoder.

Run:  .venv/bin/python scripts/readme_image.py
Panels: original | encoded at default strength | x10 difference | simulated phone photo of a
curled 15 cm print, with what decode() found there. Exits non-zero if either decode fails.
"""
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import imprint_cam as ic  # noqa: E402
import sim  # noqa: E402

PAYLOAD = b"ImPrint"
PANEL_W, PANEL_H, HEAD, PAD = 768, 512, 64, 24
BG, FG = (32, 32, 32), (235, 235, 235)


def panel(img: np.ndarray, title: str) -> np.ndarray:
    f = min(PANEL_W / img.shape[1], PANEL_H / img.shape[0])
    img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    out = np.full((HEAD + PANEL_H, PANEL_W, 3), BG, np.uint8)
    y, x = HEAD + (PANEL_H - img.shape[0]) // 2, (PANEL_W - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.putText(out, title, (0, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, FG, 2, cv2.LINE_AA)
    return out


def main():
    rev = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%h", "--", "imprint_cam.py"], capture_output=True, text=True).stdout.strip()
    img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, "testdata", "kodim23.png")), cv2.COLOR_BGR2RGB)
    enc, st = ic.encode(img, PAYLOAD, subblock_px=ic.subblock_for_print(img.shape[1], 150))
    clean = [f["payload"] for f in ic.decode(enc)["found"]]
    photo = sim.channel(enc, np.random.default_rng(1), "curled")
    r = ic.decode(photo)
    got = [f["payload"] for f in r["found"]]
    print(f"delta {st['delta']}  SSIM {st['ssim']}  mean dL {st['mean_dL']}  clean {clean}  photo {got}")
    diff = np.clip(np.abs(enc.astype(int) - img.astype(int)) * 10, 0, 255).astype(np.uint8)
    read = got[0].decode() if got else "nothing"
    tiles = [panel(img, "Original"), panel(enc, f"ImPrinted ({rev})"),
             panel(diff, "Difference x10"), panel(r["overlay"], f'Curled print photo -> "{read}"')]
    gap = np.full((HEAD + PANEL_H, PAD, 3), BG, np.uint8)
    rows = [np.hstack([tiles[0], gap, tiles[1]]), np.hstack([tiles[2], gap, tiles[3]])]
    fig = np.vstack([rows[0], np.full((PAD, rows[0].shape[1], 3), BG, np.uint8), rows[1]])
    foot = np.full((56, fig.shape[1], 3), BG, np.uint8)
    cv2.putText(foot, f'payload "{PAYLOAD.decode()}"   strength {st["delta"]}   SSIM {st["ssim"]}   '
                f'mean dL* {st["mean_dL"]}   (imprint_cam {rev}, simulated by sim.py)',
                (0, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, FG, 1, cv2.LINE_AA)
    fig = cv2.copyMakeBorder(np.vstack([fig, foot]), PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=BG)
    cv2.imwrite(os.path.join(ROOT, "versus.png"), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
    sys.exit(0 if PAYLOAD in clean and PAYLOAD in got else 1)


if __name__ == "__main__":
    main()
