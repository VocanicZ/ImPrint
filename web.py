"""Local web UI for encode / decode / strength-ladder test page.

Run:  .venv/bin/uvicorn web:app --host 0.0.0.0 --port 8765
Then open http://<this-pc-ip>:8765 on the PC or on a phone in the same network.
"""
import asyncio
import base64
import csv
import io
import json
import os
import queue
import re
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(50_000_000))   # the decoder itself refuses bombs
import cv2  # noqa: E402
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image

import imprinted.core as ic

ROOT = Path(__file__).parent
LOG = ROOT / "results" / "decode_log.csv"
app = FastAPI(title="Imprinted")
ic.quick_check(np.zeros((512, 512, 3), np.uint8))       # build the decoder's tables now, not on the first upload


MAX_UPLOAD = 60 * 1024 * 1024      # bytes
MAX_PIXELS = 50_000_000            # a 50 MP phone photo is fine; a decompression bomb is not


def read_rgb(data: bytes) -> np.ndarray:
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file too large")
    try:
        w, h = Image.open(io.BytesIO(data)).size       # header only: check before decoding pixels
    except Exception:
        raise HTTPException(400, "not an image")
    if w * h > MAX_PIXELS:
        raise HTTPException(413, f"image too large ({w}x{h}); max {MAX_PIXELS // 1_000_000} MP")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # honours EXIF rotation
    if img is None or img.shape[0] * img.shape[1] > MAX_PIXELS:          # check what was actually decoded
        raise HTTPException(400, "not an image (or too large)")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cell(v) -> str:
    """CSV field safe to open in a spreadsheet (no formula injection)."""
    v = re.sub(r"[\x00-\x1f\x7f]", " ", str(v))[:200]
    return "'" + v if v.lstrip()[:1] in ("=", "+", "-", "@", "|", "＝", "＋", "－", "＠") else v


def b64jpg(rgb: np.ndarray, max_side: int) -> str:
    f = min(1.0, max_side / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else rgb
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


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


JOBS = threading.BoundedSemaphore(2)    # encodes/decodes at once: each one already uses every core


class Cancelled(Exception):
    pass


def streamed(request: Request, rgb: np.ndarray, work) -> StreamingResponse:
    """Run work(put) in a thread and stream JSON lines: the photo's preview, every event work puts
    as it happens, then work's result ("done") or its failure ("error"). At most JOBS run at once;
    a job whose client goes away stops at its next event (put raises)."""
    events: queue.Queue = queue.Queue()
    gone = threading.Event()

    def put(ev):
        if gone.is_set():
            raise Cancelled
        events.put(ev)

    def run():
        try:
            events.put(dict(e="done", **work(put)))
        except Cancelled:
            pass
        except ValueError as e:                        # the codec's own "can't do this picture"
            events.put(dict(e="error", detail=str(e)))
            print("job refused:", e, flush=True)
        except Exception as e:                         # the stream must end either way
            events.put(dict(e="error", detail=f"{type(e).__name__}: {e}"))
            traceback.print_exc()
        finally:
            JOBS.release()

    async def stream():
        if not JOBS.acquire(blocking=False):
            print("busy: refused a job", flush=True)
            yield json.dumps(dict(e="error", detail="busy with other pictures: try again in a moment")) + "\n"
            return
        started = False
        try:
            img = await asyncio.to_thread(b64jpg, rgb, 1200)
            yield json.dumps(dict(e="preview", img=img, w=rgb.shape[1], h=rgb.shape[0])) + "\n"
            threading.Thread(target=run, daemon=True).start()
            started, seen = True, time.monotonic()
            while True:
                if time.monotonic() - seen > 1:            # also while events pour in: they never empty the queue
                    if await request.is_disconnected():    # (the server drops writes to a gone client silently)
                        return
                    seen = time.monotonic()
                try:
                    ev = events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                yield json.dumps(ev, default=lambda o: o.item()) + "\n"       # numpy scalars
                if ev["e"] in ("done", "error"):
                    return
        finally:                                       # finished, or the client went away
            gone.set()
            if not started:
                JOBS.release()
    return StreamingResponse(stream(), media_type="application/x-ndjson")


def gray(a: np.ndarray, signed: bool, top: np.ndarray) -> tuple[np.ndarray, float]:
    """A working array as an 8-bit picture, scaled by the spread of `top` (the whole array): signed
    ones around mid-gray. Returns it and the gain."""
    top = float(np.percentile(np.abs(top[::3, ::3]), 99.5)) or 1.0   # every 9th px: same spread, 9x faster
    g = (127 if signed else 255) / top
    return np.clip(a * g + (128 if signed else 0), 0, 255).astype(np.uint8), g


@app.post("/api/encode")
async def encode(request: Request, file: UploadFile = File(...), text: str = Form(...), delta: float = Form(0.0),
                 width_cm: float = Form(15.0), mode: str = Form("override")):
    """delta <= 0: the encoder's default, the strongest within its visibility budget. Streams the
    check for an older pattern as the decode endpoint streams it (probes, sync reads, the verdict and,
    if found, the read), then the encoder's stages as it computes them (their working data, once: later strengths only rescale
    it), each strength tried, then the result. A picture that is already Imprinted is encoded again
    (core.reencode: on its own grid, at its own strength, with its old text taken out first, else
    both read, or only the old one): mode "override" replaces the text, "append" encodes "old new"."""
    rgb = read_rgb(await file.read())
    b = ic.subblock_for_print(rgb.shape[1], width_cm * 10)
    t0, sent = time.time(), set()
    message = text.encode()

    def work(put):
        def trace(stage, v):
            ev = dict(e="stage", stage=stage, ms=round((time.time() - t0) * 1000))
            if stage == "tried":
                ev.update(stats={k: v[k] for k in ("delta", "ssim", "mean_dL", "p99_dL", "wrong_cells")},
                          ok=v["ssim"] >= ic.VIS_SSIM and v["mean_dL"] <= ic.VIS_DL)
            elif stage == "layout":
                ev.update(delta=round(float(v["delta"]), 3))
                if stage not in sent:                  # one px per cell: sync blue, payload bits light/dark
                    img = np.where(v["sync"][..., None], [60, 140, 255],
                                   np.where(v["bits"][..., None] > 0, [235, 235, 235], [45, 45, 45])).astype(np.uint8)
                    ev.update(cells=b64png(img), **{k: v[k] for k in ("rows", "cols", "tile", "cell_px")},
                              sync_share=round(float(v["sync"].mean()), 3))
            elif stage in ("mask", "pilot", "notch", "data") and stage not in sent:
                f = min(1.0, 900 / max(v.shape))
                small = cv2.resize(v, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else v
                y, x = v.shape[0] // 2 - 64, v.shape[1] // 2 - 64              # 1:1 crop of the middle
                pic, g = gray(small, signed=stage != "mask", top=v)
                crop, _ = gray(v[y:y + 128, x:x + 128], signed=stage != "mask", top=v)
                ev.update(img=b64jpg(np.repeat(pic[..., None], 3, 2), 900), gain=round(g, 1),
                          crop=b64png(np.repeat(crop[..., None], 3, 2)), crop_xy=[x, y])
            sent.add(stage)
            put(ev)

        nonlocal b, message
        put(dict(e="stage", stage="check", ms=round((time.time() - t0) * 1000)))
        old = ic.existing(rgb, read=True, trace=forward(put))      # its events drawn as on the decode page
        put(dict(e="existing", found=old is not None, ms=round((time.time() - t0) * 1000),
                 **(dict(subblock_px=old["subblock_px"], old=None if old["payload"] is None else
                         old["payload"].decode("utf-8", "replace")) if old else {})))
        if old:                                        # on its own grid, the old text taken out first
            enc, st, message = ic.reencode(rgb, message, old, mode, delta=delta if delta > 0 else None, trace=trace)
            b = old["subblock_px"]
        else:
            enc, st = ic.encode(rgb, message, delta=delta if delta > 0 else None, subblock_px=b, trace=trace)
        diff = np.clip(np.abs(enc.astype(int) - rgb.astype(int)) * 10, 0, 255).astype(np.uint8)
        dpi = rgb.shape[1] / (width_cm / 2.54)
        return dict(image=b64png(enc, dpi=dpi), original=b64png(rgb), diff=b64png(diff, 1600), stats=st,
                    ms=round((time.time() - t0) * 1000), dpi=round(dpi), subblock_mm=round(b * 25.4 / dpi, 2),
                    print_cm=[round(width_cm, 1), round(rgb.shape[0] / dpi * 2.54, 1)], min_subblock_mm=0.65,
                    wrong_max=ic.WRONG_WARN, payload=message.decode("utf-8", "replace"), reencoded=old is not None)
    return streamed(request, rgb, work)


def payload(f: dict) -> dict:
    return dict(text=f["payload"].decode("utf-8", "replace"), hex=f["payload"].hex(),
                quad=np.asarray(f["quad"]).round(1).tolist(),
                **{k: f[k] for k in ("raw_ber", "cells", "tiles", "probes", "mirrored", "photo_px_per_subblock")})


def xy(c) -> list:
    return [round(float(c[0]), 1), round(float(c[1]), 1)]


def grid_lines(model, most: int = 40) -> list:
    """A tracked lattice->photo map as polylines along its cell edges (at most `most` each way)."""
    fmap, (x0, x1, y0, y1), _ = model
    xs, ys = np.arange(np.ceil(x0), np.floor(x1) + 1), np.arange(np.ceil(y0), np.floor(y1) + 1)
    k = max(1, -(-max(len(xs), len(ys)) // most))
    tx, ty = np.linspace(x0, x1, 24), np.linspace(y0, y1, 24)
    return ([fmap(np.c_[np.full_like(ty, x), ty]).round(1).tolist() for x in xs[::k]] +
            [fmap(np.c_[tx, np.full_like(tx, y)]).round(1).tolist() for y in ys[::k]])


def read_cells(info: dict) -> dict:
    """What a read saw, per cell it used, in photo px: 0 sync cell, 1 payload bit read right,
    2 read wrong (fixed by error correction), 3 read (payload not known: the read failed)."""
    fmap, rows, cols, used = info["model"][0], info["rows"], info["cols"], info["used"]
    ii, jj = np.mgrid[0:rows, 0:cols].reshape(2, -1)
    lat = np.c_[info["xs"] + jj + 0.5, info["ys"] + ii + 0.5]
    keep = used | (info["sync"] if info["sync"] is not None else False)
    kind = np.full(rows * cols, 3)
    if info["right"] is not None:
        kind = np.where(info["right"], 1, 2)
    if info["sync"] is not None:
        kind = np.where(info["sync"], 0, kind)
    p = fmap(lat[keep])
    size = float(np.median(np.linalg.norm(fmap(lat[keep][:200] + [1, 0]) - p[:200], axis=1))) if len(p) else 0.0
    return dict(xy=p.round(1).ravel().tolist(), kind=kind[keep].tolist(), cell=round(size, 1),
                z=round(float(info["z"]), 1), sync_found=info["sync"] is not None)


def trace_event(ev: str, c, v) -> dict | None:
    """A decoder trace event (core.decode's docstring) as the JSON line the page draws, or None."""
    if ev == "probe":
        return dict(e=ev, xy=xy(c), locks=int(v))
    if ev == "sync":
        return dict(e=ev, xy=xy(c), z=round(float(v), 1))
    if ev == "checked":
        return dict(e=ev, ok=v is not None, **(dict(z=round(float(v[0]), 1), xy=xy(v[1]), e1=xy(v[2]),
                                                    e2=xy(v[3])) if v is not None else {}))
    if ev == "seed":
        return dict(e=ev, xy=xy(c))
    if ev == "twin":
        return dict(e=ev, xy=xy(c), sgn=int(v))
    if ev == "try":
        return dict(e=ev, xy=xy(c), ok=bool(v))
    if ev in ("tracked", "refined"):
        return dict(e=ev, lines=grid_lines(v) if v else [], probes=int(v[2]) if v else 0)
    if ev == "cells":
        return dict(e=ev, **read_cells(v))
    if ev == "found":
        return dict(e=ev, **payload(v))
    return None


def forward(put):
    """A core trace function that puts each decoder event as trace_event converts it."""
    return lambda ev, c, v: (j := trace_event(ev, c, v)) is not None and put(j)


@app.post("/api/decode")
async def decode(request: Request, file: UploadFile = File(...), label: str = Form(""), every: bool = Form(False)):
    """Streams JSON lines as the decoder works: the photo's preview, each quick-check probe and
    sync read, the check's verdict, each seed tried, each payload found, then the result. Done at
    the first payload unless `every` (a photo of the strength-ladder page: one per copy)."""
    rgb = read_rgb(await file.read())

    def work(put):
        t = time.time()
        r = ic.decode(rgb, max_found=None if every else 1, trace=forward(put))
        found = [payload(f) for f in r["found"]]
        secs = round(time.time() - t, 1)
        log(label, rgb, found, secs)
        return dict(found=found, seeds=r["seeds"], tried=r["tried"], seconds=secs, photo=f"{rgb.shape[1]}x{rgb.shape[0]}")
    return streamed(request, rgb, work)


def log(label: str, rgb: np.ndarray, found: list, secs: float):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "label", "photo", "found", "payloads", "raw_ber", "seconds"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), cell(label), f"{rgb.shape[1]}x{rgb.shape[0]}", len(found),
                    cell(" ".join(f["text"] for f in found)), " ".join(str(f["raw_ber"]) for f in found), secs])


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
                stats=[{k: s[k] for k in ("delta", "ssim", "psnr", "mean_dL", "p99_dL", "tiles", "reads_back")} for s in stats])
