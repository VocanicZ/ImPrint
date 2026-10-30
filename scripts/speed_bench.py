"""Speed and re-encode benchmark for the README (issue #33): what the library calls take on the 3
testdata pictures, and how far a re-encode moves a picture from its original.

    python scripts/speed_bench.py [--cpu] [section,...]     # sections: encode, decode, detect, reencode

encode: imprinted.encode(mode=None) at 768x512 and 3000x2000 (the pictures resized), and the
  already-Imprinted check mode="override" adds on a plain picture (core.existing(read=True)).
decode: imprinted.decode of the clean files at both sizes, and of 2 simulated `easy` prints (15 cm)
  of each 768x512 file.
detect: imprinted.detect on each 768x512 file, an easy print of it, the plain picture and an easy
  print of that (the first two must say yes, the last two no).
reencode: mean dL* vs the original of a single encode ("hello") and of re-encodes of it (override
  "world", override "a longer new text", append "world"), at 768 and 1536 px wide; each must read
  exactly one text.
--cpu: see benchclock.py (CPU seconds on one thread instead of wall time)."""
import os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import imprinted
import imprinted.core as ic
import sim

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata")
NAMES = ("kodim23", "kodim05", "kodim08")
PAYLOAD = "IMPRINT1"
SIZES = ((768, 512), (3000, 2000))


def load(n, size=(768, 512)):
    im = cv2.cvtColor(cv2.imread(os.path.join(T, n + ".png")), cv2.COLOR_BGR2RGB)
    return im if im.shape[1::-1] == size else cv2.resize(im, size, interpolation=cv2.INTER_CUBIC)


def span(ts):
    return f"{min(ts):.1f}-{max(ts):.1f} s (median {np.median(ts):.1f}, n {len(ts)})"


def main():
    args = sys.argv[1:]
    clock = time.perf_counter
    if args[:1] == ["--cpu"]:
        from benchclock import use_cpu_clock
        args, clock = args[1:], use_cpu_clock()
    which = args[0].split(",") if args else ["encode", "decode", "detect", "reencode"]
    timed = lambda f, *a, **k: (lambda t0: (f(*a, **k), clock() - t0))(clock())
    ic.quick_check(load("kodim23"))                               # tables built before timing
    rows, bad = {}, []

    def note(key, name, dt, ok=True):
        rows.setdefault(key, []).append(dt)
        bad.extend([] if ok else [f"{key}: {name}"])
        print(f"{'ok ' if ok else 'BAD'} {dt:6.2f}s  {key:<22} {name}", flush=True)

    files = {}
    for n in NAMES:
        for size in SIZES:
            im = load(n, size)
            enc, dt = timed(imprinted.encode, im, PAYLOAD, mode=None)
            files[n, size] = im, np.asarray(enc)
            if "encode" in which:
                note(f"encode {size[0]}x{size[1]}", n, dt)
                e, dt = timed(ic.existing, im, read=True)
                note(f"check {size[0]}x{size[1]}", n, dt, e is None)
    rng = lambda n, k: np.random.default_rng(100 * NAMES.index(n) + k)
    if "decode" in which:
        for (n, size), (_, enc) in files.items():
            r, dt = timed(imprinted.decode, enc)
            note(f"decode file {size[0]}x{size[1]}", n, dt, r == PAYLOAD)
        for n in NAMES:
            for k in range(2):
                shot = sim.channel(files[n, SIZES[0]][1], rng(n, k), "easy")
                r, dt = timed(imprinted.decode, shot)
                note("decode easy print", f"{n} v{k} {shot.shape[1]}x{shot.shape[0]}", dt, r == PAYLOAD)
    if "detect" in which:
        for n in NAMES:
            im, enc = files[n, SIZES[0]]
            for want, name, pic in ((True, "file", enc), (True, "easy print", sim.channel(enc, rng(n, 5), "easy")),
                                    (False, "plain", im), (False, "plain easy print", sim.channel(im, rng(n, 6), "easy"))):
                r, dt = timed(imprinted.detect, pic)
                note(f"detect {name}", n, dt, r == want)
    if "reencode" in which:
        for w in (768, 1536):
            orig = load("kodim23", (w, w * 2 // 3))
            enc = np.asarray(imprinted.encode(orig, "hello"))
            print(f"    single encode {w}px: mean dL* {ic.visibility(orig, enc)['mean_dL']:.2f}")
            for mode, text, want in (("override", "world", "world"), ("override", "a longer new text", "a longer new text"),
                                     ("append", "world", "hello world")):
                out = np.asarray(imprinted.encode(enc, text, mode=mode))
                got = [f["payload"] for f in ic.decode(out)["found"]]
                ok = got == [want.encode()]
                bad += [] if ok else [f"reencode {w} {mode} {text}: {got}"]
                print(f"{'ok ' if ok else 'BAD'} re-encode {w}px {mode} {text!r}: mean dL* "
                      f"{ic.visibility(orig, out)['mean_dL']:.2f}", flush=True)
    for key, ts in rows.items():
        print(f"{key:<22} {span(ts)}")
    print(f"wrong: {len(bad)} {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
