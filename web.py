"""Local web UI for encode / decode / strength-ladder test page.

Run:  .venv/bin/uvicorn web:app --host 0.0.0.0 --port 8765
Then open http://<this-pc-ip>:8765 on the PC or on a phone in the same network.
"""
import base64
import csv
import io
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

import imprint_cam as ic

ROOT = Path(__file__).parent
LOG = ROOT / "results" / "decode_log.csv"
app = FastAPI(title="ImPrint")


def read_rgb(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # honours EXIF rotation
    if img is None:
        raise HTTPException(400, "not an image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def b64png(rgb: np.ndarray, max_side: int | None = None, dpi: float | None = None) -> str:
    if max_side and max(rgb.shape[:2]) > max_side:
        f = max_side / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", **({"dpi": (dpi, dpi)} if dpi else {}))
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text()


@app.post("/api/encode")
async def encode(file: UploadFile = File(...), text: str = Form(...), delta: float = Form(2.0),
                 width_cm: float = Form(15.0)):
    rgb = read_rgb(await file.read())
    b = ic.subblock_for_print(rgb.shape[1], width_cm * 10)
    try:
        enc, st = ic.encode(rgb, text.encode(), delta=delta, subblock_px=b)
    except ValueError as e:
        raise HTTPException(400, str(e))
    diff = np.clip(np.abs(enc.astype(int) - rgb.astype(int)) * 10, 0, 255).astype(np.uint8)
    dpi = rgb.shape[1] / (width_cm / 2.54)
    return dict(image=b64png(enc, dpi=dpi), original=b64png(rgb), diff=b64png(diff, 1600), stats=st,
                dpi=round(dpi), subblock_mm=round(b * 25.4 / dpi, 2),
                print_cm=[round(width_cm, 1), round(rgb.shape[0] / dpi * 2.54, 1)],
                min_subblock_mm=0.65)


@app.post("/api/decode")
async def decode(file: UploadFile = File(...), label: str = Form("")):
    rgb = read_rgb(await file.read())
    t = time.time()
    r = ic.decode(rgb)
    found = [dict(text=f["payload"].decode("utf-8", "replace"), hex=f["payload"].hex(),
                  **{k: f[k] for k in ("raw_ber", "cells", "tiles", "probes", "mirrored", "photo_px_per_subblock")})
             for f in r["found"]]
    secs = round(time.time() - t, 1)
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "label", "photo", "found", "payloads", "raw_ber", "seconds"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), label, f"{rgb.shape[1]}x{rgb.shape[0]}", len(found),
                    " ".join(f["text"] for f in found), " ".join(str(f["raw_ber"]) for f in found), secs])
    return dict(found=found, seeds=r["seeds"], tried=r["tried"], overlay=b64png(r["overlay"]),
                seconds=secs, photo=f"{rgb.shape[1]}x{rgb.shape[0]}")


@app.post("/api/ladder")
async def ladder(file: UploadFile = File(...), tag: str = Form("")):
    rgb = read_rgb(await file.read())
    try:
        page, stats = ic.make_ladder(rgb, tag)
    except ValueError as e:
        raise HTTPException(400, str(e))
    buf = io.BytesIO()
    Image.fromarray(page).save(buf, "PDF", resolution=ic.LADDER_DPI)
    return dict(pdf="data:application/pdf;base64," + base64.b64encode(buf.getvalue()).decode(),
                stats=[{k: s[k] for k in ("delta", "ssim", "psnr", "mean_dL", "p99_dL", "tiles")} for s in stats])
