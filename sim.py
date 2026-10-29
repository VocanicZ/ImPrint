"""Simulated print -> photo channel, and a BER benchmark over strength/axis.

Usage: python sim.py [images...] [--deltas 1.5,2] [--presets easy,hard,curled,gray] [--trials 2]
       nice python sim.py --acceptance [--quick] [--trials 5] [--presets ...] [--out results/acceptance]

A predictor only: real printers/phones decide. Use it to pick starting values,
then confirm with the printed test sheet.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

import imprinted.core as ic

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


# ---------------------------------------------------------------- PRD #1 acceptance benchmark

SSIM_MIN, MEAN_DL_MAX = 0.95, 1.0
GEOMETRY = ("clean", "rot90", "rot180", "rot270", "mirror", "crop60")   # must decode 100%
PRINT_MIN = {   # image -> preset -> minimum simulated print+photo decode rate
    "kodim23": dict(easy=0.8, gray=0.8, hard=0.8, curled=0.8),
    "kodim05": dict(easy=0.6, gray=0.6),
    "kodim08": dict(easy=0.6, gray=0.6),
}
PAYLOAD = b"IMPRINT1"


TRIALS = 5   # print trials and random crops a rate needs before it can PASS
TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")


def _rate_row(image, name, m, need, trials=TRIALS):
    """A decode-rate row. Under `trials` attempts it is None (INCOMPLETE) unless even a perfect
    finish could not reach `need`, in which case it already FAILs."""
    if m is None:
        return dict(image=image, criterion=name, measured="not run", threshold=f">= {need:.2f}", passed=None)
    r = m["ok"] / m["n"]
    measured = f"{m['ok']}/{m['n']} ({r:.2f})"
    if m["n"] >= trials:
        passed = r >= need
    else:
        measured += f", < {trials} trials"
        passed = False if m["ok"] + trials - m["n"] < need * trials else None
    return dict(image=image, criterion=name, measured=measured, threshold=f">= {need:.2f}", passed=passed)


def criteria(results: dict) -> tuple[list[dict], str]:
    """Score measured results against every PRD #1 criterion.

    results: image -> dict(ssim, mean_dL, geometry={view: {ok, n}}, print={preset: {ok, n}}).
    Every PRD #1 image (PRINT_MIN) gets rows, "not run" when it is missing from results.
    Returns (rows, verdict). A row's passed is None when it was not (fully) measured; the
    verdict is FAIL if any row fails, INCOMPLETE if some were not measured, else PASS."""
    rows = []
    for image in [*results, *(i for i in PRINT_MIN if i not in results)]:
        r = results.get(image)
        if r is None:
            rows += [dict(image=image, criterion=c, measured="not run", threshold=t, passed=None)
                     for c, t in (("SSIM", f">= {SSIM_MIN}"), ("mean dL", f"<= {MEAN_DL_MAX}"))]
            r = dict(geometry={}, print={})
        else:
            rows.append(dict(image=image, criterion="SSIM", measured=f"{r['ssim']:.4f}",
                             threshold=f">= {SSIM_MIN}", passed=r["ssim"] >= SSIM_MIN))
            rows.append(dict(image=image, criterion="mean dL", measured=f"{r['mean_dL']:.2f}",
                             threshold=f"<= {MEAN_DL_MAX}", passed=r["mean_dL"] <= MEAN_DL_MAX))
        # decode is deterministic: one try settles a fixed view; only the random crop needs trials
        rows += [_rate_row(image, f"decode {v}", r["geometry"].get(v), 1.0, TRIALS if v == "crop60" else 1)
                 for v in GEOMETRY]
        rows += [_rate_row(image, f"print {p}", r["print"].get(p), need)
                 for p, need in PRINT_MIN.get(image, {}).items()]
    passed = [x["passed"] for x in rows]
    verdict = "FAIL" if False in passed else "INCOMPLETE" if None in passed else "PASS"
    return rows, verdict


def format_table(rows: list[dict], verdict: str) -> str:
    out = ["| image | criterion | measured | threshold | result |", "|---|---|---|---|---|"]
    out += [f"| {x['image']} | {x['criterion']} | {x['measured']} | {x['threshold']} | "
            f"{ {True: 'PASS', False: 'FAIL', None: 'SKIP'}[x['passed']] } |" for x in rows]
    return "\n".join(out) + f"\n\nverdict: {verdict}\n"


def write_report(out_dir, config: dict, results: dict) -> tuple[list[dict], str]:
    rows, verdict = criteria(results)
    with open(os.path.join(out_dir, "acceptance.json"), "w") as f:
        json.dump(dict(config=config, verdict=verdict, criteria=rows, results=results), f, indent=1)
    return rows, verdict


def views(enc: np.ndarray, rng) -> dict:
    """The geometric views that must decode: turned, mirrored, and a random 60%-area crop."""
    h, w = enc.shape[:2]
    ch, cw = int(h * 0.6 ** 0.5), int(w * 0.6 ** 0.5)
    y, x = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    return dict(clean=enc, rot90=np.rot90(enc), rot180=np.rot90(enc, 2), rot270=np.rot90(enc, 3),
                mirror=enc[:, ::-1], crop60=enc[y:y + ch, x:x + cw])


def _decodes(img: np.ndarray) -> dict | None:
    """The decode that found PAYLOAD (raw_ber, bit_ber, ...), or None."""
    return next((f for f in ic.decode(np.ascontiguousarray(img))["found"] if f["payload"] == PAYLOAD), None)


def measure(path: str, trials: int, presets: list[str], print_cm: float, out_dir: str) -> dict:
    """Encode one image at default strength; measure visibility, geometric and print+photo decode
    rates; write the encoded image, its x10 difference view and its strength ladder."""
    name = os.path.splitext(os.path.basename(path))[0]
    log = lambda *a: print(f"[{name}]", *a, file=sys.stderr, flush=True)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    enc, st = ic.encode(img, PAYLOAD, subblock_px=ic.subblock_for_print(img.shape[1], print_cm * 10))
    diff = np.clip(np.abs(enc.astype(int) - img.astype(int)) * 10, 0, 255).astype(np.uint8)
    for tag, im in (("enc", enc), ("diff_x10", diff), ("ladder", ic.make_ladder(img)[0])):
        cv2.imwrite(os.path.join(out_dir, f"{name}_{tag}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    log(f"delta {st['delta']}  SSIM {st['ssim']}  mean dL {st['mean_dL']}")
    geometry = {v: dict(ok=0, n=0) for v in GEOMETRY}
    ber = {}                  # bit error rate after combining repeats, before LDPC (decoded views)
    for t in range(trials):   # decode is deterministic: only the random crop changes between trials
        for v, im in views(enc, np.random.default_rng(t + 1)).items():
            if t == 0 or v == "crop60":
                f = _decodes(im)
                geometry[v]["ok"] += f is not None
                geometry[v]["n"] += 1
                if f and f.get("bit_ber") is not None:
                    ber.setdefault(v, []).append(f["bit_ber"])
    log("geometry", {v: f"{m['ok']}/{m['n']}" for v, m in geometry.items()})
    ber = {v: round(float(np.mean(b)), 4) for v, b in ber.items()}
    log("bit BER after combining", ber)
    prints = {}
    for p in [p for p in presets if p in PRINT_MIN.get(name, {})]:
        ok = sum(_decodes(channel(enc, np.random.default_rng(t + 1), p, print_w=int(print_cm / 2.54 * 300)))
                 is not None for t in range(trials))
        prints[p] = dict(ok=int(ok), n=trials)
        log(f"print {p} {ok}/{trials}")
    return dict(st, geometry=geometry, print=prints, bit_ber=ber)


def acceptance(images: list[str], trials: int, presets: list[str], print_cm: float, out_dir: str) -> str:
    """Run the PRD #1 benchmark; print the criteria table; write acceptance.json and images."""
    os.makedirs(out_dir, exist_ok=True)
    config = dict(images=images, trials=trials, presets=presets, print_cm=print_cm,
                  delta="auto", budget=dict(ssim=ic.VIS_SSIM, mean_dL=ic.VIS_DL))   # each image's delta: results
    results = {os.path.splitext(os.path.basename(p))[0]: measure(p, trials, presets, print_cm, out_dir)
               for p in images}
    rows, verdict = write_report(out_dir, config, results)
    print(format_table(rows, verdict), flush=True)
    return verdict


def main():
    """Decode-rate matrix: images x strengths x presets, through the simulated print+photo."""
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--deltas", default="1.5,2")
    ap.add_argument("--presets")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--print-cm", type=float, default=15.0)
    ap.add_argument("--acceptance", action="store_true", help="score every PRD #1 criterion (5 trials)")
    ap.add_argument("--quick", action="store_true", help="acceptance on kodim23 only, 1 trial (INCOMPLETE at best)")
    ap.add_argument("--out", default="results/acceptance", help="acceptance output dir")
    args = ap.parse_args()
    if args.acceptance or args.quick:
        images = args.images or ([os.path.join(TESTDATA, "kodim23.png")] if args.quick
                                 else sorted(glob.glob(os.path.join(TESTDATA, "*.png"))))
        if not images:
            ap.error(f"no test images found in {TESTDATA}")
        trials = args.trials or (1 if args.quick else TRIALS)
        presets = args.presets.split(",") if args.presets else list(PRESETS)
        verdict = acceptance(images, trials, presets, args.print_cm, args.out)
        # --quick can never PASS (one image, one trial): it fails only on a measured FAIL
        sys.exit(verdict == "FAIL" if args.quick else verdict != "PASS")
    args.images = args.images or sorted(glob.glob(os.path.join(TESTDATA, "*.png")))
    args.presets = args.presets or ",".join(PRESETS)
    args.trials = args.trials or 2
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
