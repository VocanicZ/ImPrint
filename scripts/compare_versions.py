"""Robustness of this checkout against another release, on the same pictures and simulated prints:
does encode refuse, does the clean file read, how many simulated phone photos read.

    git worktree add /tmp/imprinted-071 v0.7.1
    python scripts/compare_versions.py /tmp/imprinted-071 > old.log &
    python scripts/compare_versions.py . > new.log

2026-10-29 (owner's report "0.7.1 almost always works, 0.8.0 fails on photos"): photos read
0.7.1 20/24, 0.8.0 15/24, every loss on the busy pictures kodim05/kodim08; visibility identical.
The sync cells (SYNC_FRAC of the cells, known bits for detection) carry no payload, so each tile
copy loses that share of its cells."""
import os, sys, warnings
import cv2, numpy as np
root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [root, os.path.join(here, "..")]                 # imprinted from root, sim from this checkout
warnings.simplefilter("ignore")
import imprinted, imprinted.core as ic
sys.modules.pop("sim", None)
import importlib.util
spec = importlib.util.spec_from_file_location("simcopy", os.path.join(here, "..", "sim.py"))
sim = importlib.util.module_from_spec(spec)
sys.modules["imprinted.core"] = ic                               # sim must drive the version under test
spec.loader.exec_module(sim)

T = os.path.join(here, "..", "testdata")
print("version", imprinted.__version__, ic.__file__, flush=True)
n_ok = n = 0
for name in ("kodim23", "kodim05", "kodim08"):
    full = cv2.cvtColor(cv2.imread(os.path.join(T, name + ".png")), cv2.COLOR_BGR2RGB)
    for w in (500, 768):
        img = full if w == full.shape[1] else cv2.resize(full, (w, round(w * full.shape[0] / full.shape[1])), interpolation=cv2.INTER_AREA)
        for pl in (b"IMPRINT1", b"RainBowCreation"):
            try:
                enc, st = ic.encode(img, pl, subblock_px=ic.subblock_for_print(img.shape[1], 150))
            except ValueError as e:
                print(f"{name} {w}px {pl.decode():16s} ENCODE REFUSED: {str(e)[:70]}", flush=True)
                continue
            ok = [any(f["payload"] == pl for f in ic.decode(sim.channel(enc, np.random.default_rng(100 + t), "easy"), max_found=1)["found"])
                  for t in range(2)]
            fileok = any(f["payload"] == pl for f in ic.decode(enc, max_found=1)["found"])
            n_ok, n = n_ok + sum(ok), n + 2
            print(f"{name} {w}px {pl.decode():16s} tiles {st['tiles']:4} dL {st['mean_dL']:.2f} ssim {st['ssim']:.3f} "
                  f"file {'ok' if fileok else 'FAIL'} photos {sum(ok)}/2", flush=True)
print(f"photos read: {n_ok}/{n}")
