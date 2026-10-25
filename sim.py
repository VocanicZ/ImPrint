"""Simulated print -> photo channel, and a BER benchmark over strength/axis.

Usage: python sim.py [images...] [--deltas 1.5,2] [--presets easy,hard,curled,gray] [--trials 2]

A predictor only: real printers/phones decide. Use it to pick starting values,
then confirm with the printed test sheet.
"""
import argparse
import glob

import cv2
import numpy as np

import imprint_cam as ic

PRESETS = {
    # tilt: 3D tilt, max ~tilt*200 deg; scale: camera px per printed px; frame: scene / page size
    "easy": dict(tilt=0.05, scale=1.0, blur=0.8, noise=2, light=0.1, wb=0.05, gamma=0.05, jpeg=90),
    "hard": dict(tilt=0.20, scale=0.6, blur=1.5, noise=4, light=0.4, wb=0.15, gamma=0.15, jpeg=75),
    # bend: total curl angle of the page (deg); mirror: selfie / through-paper shot
    "curled": dict(tilt=0.12, scale=0.8, blur=1.2, noise=3, light=0.3, wb=0.1, gamma=0.1, jpeg=80, bend=15, mirror=True),
    # mono: grayscale printer (single K screen)
    "gray": dict(tilt=0.08, scale=0.9, blur=1.0, noise=3, light=0.2, wb=0.08, gamma=0.1, jpeg=85, mono=True),
}


def halftone(rgb: np.ndarray, up: int = 2, period: float = 4.0, gain_sigma: float = 0.6,
             ink_noise: float = 0.08, rng=None, mono: bool = False) -> np.ndarray:
    """AM-screened CMY (or K-only if mono) print: rotated screens, dot gain, ragged dots, paper tint.
    period in up-sampled px."""
    rng = rng or np.random.default_rng()
    if mono:
        rgb = np.repeat(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
    out = np.empty((rgb.shape[0] * up, rgb.shape[1] * up, 3), np.float32)
    W = out.shape[1]
    paper, density = np.array([0.96, 0.95, 0.92], np.float32), 0.9
    for y0 in range(0, out.shape[0], 512):   # row strips keep memory bounded
        y1 = min(y0 + 512, out.shape[0])
        yy, xx = np.mgrid[y0:y1, 0:W].astype(np.float32)
        x = cv2.resize(rgb[y0 // up:-(-y1 // up)], None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
        x = x[:y1 - y0].astype(np.float32) / 255
        for ch, ang in enumerate(np.deg2rad([45, 45, 45] if mono else [15, 75, 0])):   # C<-R, M<-G, Y<-B
            u = xx * np.cos(ang) + yy * np.sin(ang)
            v = -xx * np.sin(ang) + yy * np.cos(ang)
            thr = 0.5 + 0.25 * (np.cos(2 * np.pi * u / period) + np.cos(2 * np.pi * v / period))
            thr += rng.normal(0, ink_noise, thr.shape).astype(np.float32)   # ragged dot edges
            ink = ((1 - x[..., ch]) > thr).astype(np.float32)
            out[y0:y1, :, ch] = paper[ch] * (1 - density * cv2.GaussianBlur(ink, (0, 0), gain_sigma))
    small = cv2.resize(out, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    return (small * 255).astype(np.uint8)


def curl(x: np.ndarray, bend_deg: float, rng, bg=0.0) -> np.ndarray:
    """Page lying on a cylinder (random axis), seen from above at ~1.5 page widths."""
    vertical = rng.random() < 0.5
    if vertical:
        x = x.transpose(1, 0, 2)
    H, W = x.shape[:2]
    pad = int(0.15 * max(H, W))                                  # lifted edges grow: leave room
    R = W / np.deg2rad(bend_deg)
    D, cx, cy = 1.5 * W, W / 2 + pad, H / 2 + pad
    s = np.linspace(-W / 2, W / 2, 4096)                       # arc position across the page
    k = D / (D - R * (1 - np.cos(s / R)))                        # nearer = bigger
    X = cx + R * np.sin(s / R) * k
    ox, oy = np.meshgrid(np.arange(W + 2 * pad, dtype=np.float64), np.arange(H + 2 * pad, dtype=np.float64))
    si = np.interp(ox, X, s, left=np.nan, right=np.nan)
    ki = D / (D - R * (1 - np.cos(si / R)))
    mx, my = (si + W / 2).astype(np.float32), (H / 2 + (oy - cy) / ki).astype(np.float32)
    out = cv2.remap(x, np.nan_to_num(mx, nan=-1), np.nan_to_num(my, nan=-1), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(bg,) * x.shape[2])
    return out.transpose(1, 0, 2) if vertical else out


def clutter(h: int, w: int, rng) -> np.ndarray:
    """A busy desk: colour blobs, grain, shapes, text and marker-like decoy squares."""
    bg = cv2.resize(rng.uniform(0.1, 0.9, (6, 6, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    bg += cv2.GaussianBlur(rng.normal(0, 0.1, (h, w, 3)).astype(np.float32), (0, 0), 3)
    s = max(h, w)
    for _ in range(40):
        c = tuple(float(v) for v in rng.uniform(0, 1, 3))
        x, y, r = int(rng.uniform(0, w)), int(rng.uniform(0, h)), int(rng.uniform(0.01, 0.08) * s)
        kind = rng.integers(3)
        if kind == 0:
            cv2.rectangle(bg, (x, y), (x + r, y + int(r * rng.uniform(0.3, 2))), c, -1)
        elif kind == 1:
            cv2.circle(bg, (x, y), r, c, -1)
        else:
            cv2.line(bg, (x, y), (int(rng.uniform(0, w)), int(rng.uniform(0, h))), c, max(1, r // 10))
    for _ in range(12):
        cv2.putText(bg, "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "), 12)),
                    (int(rng.uniform(0, w)), int(rng.uniform(0, h))), cv2.FONT_HERSHEY_SIMPLEX,
                    float(rng.uniform(0.5, 3)) * s / 2000, (0, 0, 0), max(1, s // 700))
    for _ in range(8):   # decoys: black-bordered random bit grids, like markers but not valid ones
        n = int(rng.uniform(0.03, 0.08) * s)
        g = np.zeros((6, 6), np.float32)
        g[1:5, 1:5] = rng.integers(0, 2, (4, 4))
        x, y = int(rng.uniform(0, w - n)), int(rng.uniform(0, h - n))
        bg[y:y + n, x:x + n] = cv2.resize(g, (n, n), interpolation=cv2.INTER_NEAREST)[..., None]
    return np.clip(bg, 0, 1)


def camera(rgb: np.ndarray, rng, tilt, scale, blur, noise, light, wb, gamma, jpeg,
           bend=0, mirror=False, frame=1.5) -> np.ndarray:
    """Photograph a printed page lying in a cluttered scene. frame = scene size / page size."""
    H, W = rgb.shape[:2]
    x = rgb.astype(np.float32) / 255
    ang = rng.uniform(0, 2 * np.pi)                  # uneven lighting across the page
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ramp = (np.cos(ang) * xx / W + np.sin(ang) * yy / H)
    ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-6)
    x *= (1 - light * ramp)[..., None]
    x = np.dstack([x, np.ones((H, W), np.float32)])  # alpha: where the paper is
    if bend:
        x = curl(x, bend, rng)
        H, W = x.shape[:2]
    x = cv2.GaussianBlur(x, (0, 0), blur / scale)     # lens blur, at paper resolution

    # Pinhole camera: page tilted in 3D (up to ~tilt*200 deg), any in-plane turn incl. upside down,
    # wide-ish phone lens; principal point at the photo centre like a real phone.
    L = max(W, H)
    size = (int(L * scale * frame), int(L * scale * frame * 0.8))
    fpx = 0.8 * size[0]
    rx, ry = np.deg2rad(rng.uniform(-200, 200, 2) * tilt)
    rz = np.deg2rad(rng.uniform(-12, 12) + 180 * rng.integers(2))
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    P = (Rz @ Ry @ Rx @ np.c_[src - [W / 2, H / 2], np.zeros(4)].T).T + [0, 0, fpx / scale]
    dst = (fpx * P[:, :2] / P[:, 2:] + [size[0] / 2, size[1] / 2]).astype(np.float32)
    x = cv2.warpPerspective(x, cv2.getPerspectiveTransform(src, dst), size, borderValue=(0, 0, 0, 0))
    a = x[..., 3:]
    x = x[..., :3] * a + clutter(size[1], size[0], rng) * (1 - a)

    x += rng.normal(0, noise / 255, x.shape).astype(np.float32)
    x *= 1 + rng.uniform(-wb, wb, 3).astype(np.float32)  # white balance
    x = np.clip(x, 0, 1) ** (1 + rng.uniform(-gamma, gamma))
    if mirror:
        x = x[:, ::-1]
    ok, buf = cv2.imencode(".jpg", (x * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def channel(rgb, rng, preset, print_w=1772, paper=0.08):
    """Print rgb print_w px wide at 300 dpi (1772 = 15 cm) on paper with a white margin, photograph it."""
    cam = dict(PRESETS[preset])
    mono = cam.pop("mono", False)
    f = print_w / rgb.shape[1]
    img = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR if f > 1 else cv2.INTER_AREA)
    m = int(paper * print_w)
    img = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return camera(halftone(img, rng=rng, mono=mono), rng, **cam)


def main():
    """Decode-rate matrix: images x strengths x presets, through the simulated print+photo."""
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", default=sorted(glob.glob("testdata/*.png")))
    ap.add_argument("--deltas", default="1.5,2")
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--print-cm", type=float, default=15.0)
    args = ap.parse_args()
    print("| image | delta | SSIM | mean dL | clean file | " + " | ".join(args.presets.split(",")) + " |")
    print("|---|---|---|---|---|" + "---|" * len(args.presets.split(",")))
    for path in args.images:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        b = ic.subblock_for_print(img.shape[1], args.print_cm * 10)
        for d in map(float, args.deltas.split(",")):
            enc, st = ic.encode(img, b"IMPRINT1", delta=d, subblock_px=b)
            clean = "OK" if ic.decode(enc)["found"] else "FAIL"
            cells = []
            for preset in args.presets.split(","):
                ok = sum(any(f["payload"] == b"IMPRINT1" for f in
                             ic.decode(channel(enc, np.random.default_rng(t + 1), preset,
                                               print_w=int(args.print_cm / 2.54 * 300)))["found"])
                         for t in range(args.trials))
                cells.append(f"{ok}/{args.trials}")
            print(f"| {path.rsplit('/', 1)[-1]} | {d:g} | {st['ssim']} | {st['mean_dL']} | {clean} | "
                  + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
