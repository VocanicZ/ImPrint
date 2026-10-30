"""Staged-detection benchmark: how soon quick_check confirms simulated Imprinted pictures, and that
pictures without a pattern never pass. Owner's bar (2026-10-29), on the in-spec cases (everything
but kodim05/kodim08 hard and curled, which the decoder can't read either):
  - no Imprinted picture ruled out in the first second (QUICK_REJECT),
  - >= 80% confirmed by 3 s, >= 95% by 10 s,
  - 0 false alarms (and report the clean pictures' best sync z: it must stay well under QUICK_Z).

    python scripts/detect_bench.py [--cpu] [cases.pkl]     # builds the cases on first run (a few minutes)

--cpu: see benchclock.py (CPU seconds on one thread instead of wall time).

The 200 real photos used on 2026-10-29 (0 false alarms, best z 5.11) are the owner's own and not
in the repo; this bench uses simulated prints of unencoded testdata pictures instead."""
import os, pickle, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import imprinted.core as ic
import sim

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata")


def build():
    cases = []
    for n in ("kodim23", "kodim05", "kodim08"):
        im = cv2.cvtColor(cv2.imread(os.path.join(T, n + ".png")), cv2.COLOR_BGR2RGB)
        enc, _ = ic.encode(im, b"VALID-08", subblock_px=ic.subblock_for_print(768, 150))
        cases += [(1, f"{n} file", enc), (1, f"{n} file rot270", np.rot90(enc, 3).copy()),
                  (1, f"{n} file mirror", enc[::-1].copy())]
        for preset in sim.PRESETS:
            for seed in range(4, 9):
                cases.append((1, f"{n} {preset} v{seed}", sim.channel(enc, np.random.default_rng(1000 * seed + len(preset)), preset)))
            for seed in range(4, 7):
                cases.append((0, f"{n} plain {preset} v{seed}", sim.channel(im, np.random.default_rng(2000 * seed + len(preset)), preset)))
    return cases


def main():
    args = sys.argv[1:]
    clock = time.perf_counter
    if args[:1] == ["--cpu"]:
        from benchclock import use_cpu_clock
        args, clock = args[1:], use_cpu_clock()
    path = args[0] if args else os.path.join(T, "..", "results", "detect_cases.pkl")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pickle.dump(build(), open(path, "wb"))
    cases = pickle.load(open(path, "rb"))
    ic.quick_check(cases[0][2])                                   # tables built before timing
    oos = lambda name: any(f"{k} {p}" in name for k in ("kodim05", "kodim08") for p in ("hard", "curled"))
    pos, neg = [], []
    for lab, name, im in cases:
        zs = []
        t0 = clock()
        hit = ic.quick_check(im, trace=lambda e, c, v: e == "sync" and zs.append(v))
        dt = clock() - t0
        (pos if lab else neg).append((name, bool(hit), dt, max(zs, default=0.0)))
        print(f"{'+' if lab else '-'} {'HIT ' if hit else 'miss'} {dt:5.2f}s zmax {max(zs, default=0):5.2f}  {name}", flush=True)
    spec = [p for p in pos if not oos(p[0])]
    for label, ps in (("in-spec", spec), ("all", pos)):
        by = {s: sum(h and t <= s for _, h, t, _ in ps) / len(ps) for s in (1, 3, 10)}
        print(f"{label} imprinted ({len(ps)}): " + ", ".join(f"{v:.0%} by {s} s" for s, v in by.items()))
    early = [n for n, h, t, _ in spec if not h and t <= ic.QUICK_REJECT + 0.1]
    fa = [n for n, h, _, _ in neg if h]
    print(f"in-spec ruled out in the first second: {len(early)} {early}")
    print(f"clean ({len(neg)}): false alarms {len(fa)} {fa}; best sync z {max(z for *_, z in neg):.2f} (QUICK_Z {ic.QUICK_Z})")
    ok = (not early and not fa and sum(h and t <= 3 for _, h, t, _ in spec) >= 0.8 * len(spec)
          and sum(h and t <= 10 for _, h, t, _ in spec) >= 0.95 * len(spec))
    print("verdict:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
