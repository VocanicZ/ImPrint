"""Small-picture benchmark (issue #28): detect() / existing() / decode() time and success on pictures
under QUICK_SMALL px. Bar: detect() <= 10 s on every picture, and the encoded ones still detected
and decoded (not fewer than before).

    python scripts/small_bench.py [--cpu] [cases.pkl]     # builds the cases on first run

--cpu: on a shared, loaded host wall time says more about the neighbours than the code, so every
clock the codec reads (its patience and deadlines) and the bench's own timer run in this process's
CPU seconds instead, one thread each for OpenCV and BLAS: about the seconds an idle machine would
take (more, as quick_check's threads add up), and the same misses it would see.

Encoded: clean 768x512 files of the 3 testdata pictures at payloads of 2/5/8 bytes, each as is,
turned 90 degrees and cropped to its central 60%. Plain: the same pictures unencoded, at 768x512
and smaller, simulated prints of them, halftone screens and noise (no pattern: detect must say
no, and quickly)."""
import os, pickle, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import imprinted
import imprinted.core as ic
import sim

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata")
BOUND = 10.0


def build():
    cases = []
    for n in ("kodim23", "kodim05", "kodim08"):
        im = cv2.cvtColor(cv2.imread(os.path.join(T, n + ".png")), cv2.COLOR_BGR2RGB)
        for nb in (2, 5, 8):
            p = b"PAYLOAD!"[:nb]
            enc = np.asarray(imprinted.encode(im, p))
            h, w = enc.shape[:2]
            crop = enc[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)].copy()
            cases += [(p, f"{n} {nb}B file", enc), (p, f"{n} {nb}B rot90", np.rot90(enc).copy()),
                      (p, f"{n} {nb}B crop60", crop)]
        cases += [(None, f"{n} plain", im), (None, f"{n} plain rot90", np.rot90(im).copy()),
                  (None, f"{n} plain 500", cv2.resize(im, (500, 333), interpolation=cv2.INTER_AREA)),
                  (None, f"{n} plain crop60", im[102:410, 153:614].copy())]
        for preset in sim.PRESETS:
            sm = sim.channel(im, np.random.default_rng(3000 + len(preset)), preset)
            f = min(1.0, 1000 / max(sm.shape[:2]))
            cases.append((None, f"{n} plain {preset}", cv2.resize(sm, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)))
    return cases + extras()


def extras():
    """Lattices that aren't Imprinted (halftone screens lock the pilot probes, and the decoder's
    tracker follows them) and noise, all small."""
    rng, out = np.random.default_rng(7), []
    im = cv2.cvtColor(cv2.imread(os.path.join(T, "kodim23.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
    yy, xx = np.mgrid[0:512, 0:768].astype(np.float32)
    for pitch, turn in ((6, 45), (9, 45), (12, 15), (18, 0)):
        a = np.deg2rad(turn)
        u, v = (xx * np.cos(a) + yy * np.sin(a)) / pitch, (yy * np.cos(a) - xx * np.sin(a)) / pitch
        dots = np.cos(2 * np.pi * u) * np.cos(2 * np.pi * v) * 50
        out.append((None, f"halftone {pitch}px {turn}deg", np.clip(im * 0.7 + dots[..., None] + 30, 0, 255).astype(np.uint8)))
    out.append((None, "noise 768x512", rng.integers(0, 256, (512, 768, 3), np.uint8)))
    out.append((None, "gray 768x512", np.full((512, 768, 3), 128, np.uint8)))
    return out


def main():
    args = sys.argv[1:]
    clock = time.perf_counter
    if args[:1] == ["--cpu"]:
        import types
        from threadpoolctl import threadpool_limits
        args, clock = args[1:], time.process_time
        ic.time = types.SimpleNamespace(perf_counter=clock)
        cv2.setNumThreads(1)
        threadpool_limits(1)
    path = args[0] if args else os.path.join(T, "..", "results", "small_cases.pkl")
    which = args[1].split(",") if len(args) > 1 else ["detect", "decode", "existing"]
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pickle.dump(build(), open(path, "wb"))
    cases = pickle.load(open(path, "rb"))
    ic.quick_check(cases[0][2])                                   # tables built before timing
    worst, misses, alarms, stats = 0.0, [], [], {k: [0, 0] for k in which}
    for p, name, im in cases:
        row = []
        for k in which:
            t0 = clock()
            if k == "detect":
                r = ic.detect(im)
                ok = r == (p is not None)
            elif k == "decode":
                f = ic.decode(im, max_found=1)["found"]
                r = f[0]["payload"] if f else None
                ok = r == p
            else:
                e = ic.existing(im)
                r = e is not None
                ok = r == (p is not None)
            dt = clock() - t0
            if k == "detect":
                worst = max(worst, dt)
            stats[k][0] += ok
            stats[k][1] += 1
            (misses if p is not None else alarms).extend([] if ok else [f"{k}:{name}"])
            row.append(f"{k} {'ok ' if ok else 'BAD'} {dt:5.2f}s")
        print(f"{'+' if p else '-'} " + "  ".join(row) + f"  {name} {im.shape[1]}x{im.shape[0]}", flush=True)
    print("right: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in stats.items()))
    print(f"worst detect {worst:.2f}s (bound {BOUND} s); misses {misses}; false alarms {alarms}")
    ok = worst <= BOUND and not alarms
    print("verdict:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
