"""Run: .venv/bin/python test_imprint_cam.py  (also works under pytest)"""
import glob
import json
import sys

import cv2
import numpy as np
import pytest

import imprint_cam as ic
import sim


def photo_like(h=720, w=1080, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx / w * 200 + 30, yy / h * 180 + 40, (xx + yy) / (w + h) * 150 + 60], -1)
    img += cv2.GaussianBlur(rng.normal(0, 12, img.shape), (0, 0), 1.5)   # photo-like texture, not pixel noise
    return np.clip(img, 0, 255).astype(np.uint8)


def test_same_size_no_markers():
    img = photo_like()
    enc, st = ic.encode(img, b"hello", delta=2, subblock_px=6)
    assert enc.shape == img.shape
    assert st["ssim"] > 0.9 and st["mean_dL"] < 1.5, st


def test_found_without_border_any_orientation():
    enc, _ = ic.encode(photo_like(), b"hello", delta=2, subblock_px=6)
    h, w = enc.shape[:2]
    views = (enc, np.rot90(enc), np.rot90(enc, 2), enc[:, ::-1],          # turned / upside down / mirrored
             enc[h // 6:h // 6 + int(h * 0.7), w // 5:w // 5 + int(w * 0.7)])  # cropped: no picture edge at all
    for view in views:
        found = ic.decode(np.ascontiguousarray(view))["found"]
        assert found and found[0]["payload"] == b"hello", view.shape


def test_survives_simulated_print_and_photo():
    enc, _ = ic.encode(photo_like(), b"IMPRINT1", delta=2, subblock_px=6)
    found = ic.decode(sim.channel(enc, np.random.default_rng(3), "easy"))["found"]
    assert found and found[0]["payload"] == b"IMPRINT1"


def _passing(img="kodim23"):
    """Synthetic measurements that just meet every PRD #1 threshold for img."""
    return dict(ssim=0.95, mean_dL=1.0, geometry={v: dict(ok=5, n=5) for v in sim.GEOMETRY},
                print={p: dict(ok=round(r * 5), n=5) for p, r in sim.PRINT_MIN[img].items()})


def test_acceptance_criteria_pass_on_threshold():
    rows, verdict = sim.criteria({"kodim23": _passing(), "kodim05": _passing("kodim05")})
    assert verdict == "PASS" and all(r["passed"] for r in rows)
    k23 = [r["criterion"] for r in rows if r["image"] == "kodim23"]
    assert k23 == ["SSIM", "mean dL"] + [f"decode {v}" for v in sim.GEOMETRY] + \
        [f"print {p}" for p in ("easy", "gray", "hard", "curled")]


def test_acceptance_criteria_fail_just_past_threshold():
    for key, bad in (("ssim", 0.9499), ("mean_dL", 1.01)):
        r = _passing()
        r[key] = bad
        rows, verdict = sim.criteria({"kodim23": r})
        assert verdict == "FAIL" and [x["passed"] for x in rows].count(False) == 1, key
    r = _passing()
    r["geometry"]["crop60"] = dict(ok=4, n=5)            # geometry must be 100%
    r["print"]["hard"] = dict(ok=3, n=5)                 # kodim23 needs >= 4/5
    rows, verdict = sim.criteria({"kodim23": r})
    assert verdict == "FAIL"
    assert sorted(x["criterion"] for x in rows if not x["passed"]) == ["decode crop60", "print hard"]
    assert sim.criteria({"kodim05": dict(_passing("kodim05"), print=dict(easy=dict(ok=3, n=5),
                                                                          gray=dict(ok=2, n=5)))})[1] == "FAIL"


def test_acceptance_criteria_unmeasured_is_incomplete_not_pass():
    r = _passing()
    del r["print"]["curled"]                             # e.g. --presets easy,gray,hard
    rows, verdict = sim.criteria({"kodim23": r})
    assert verdict == "INCOMPLETE"
    assert [x["passed"] for x in rows if x["criterion"] == "print curled"] == [None]
    r["print"]["hard"] = dict(ok=0, n=5)
    assert sim.criteria({"kodim23": r})[1] == "FAIL"    # a measured failure beats a gap
    other = dict(_passing(), print={})                   # an image with no print requirement
    assert sim.criteria({"photo1": other})[1] == "PASS"


def test_acceptance_table_and_json(tmp_path):
    results = {"kodim23": _passing()}
    rows, verdict = sim.criteria(results)
    table = sim.format_table(rows, verdict)
    assert "| kodim23 | print curled | 4/5 (0.80) | >= 0.80 | PASS |" in table
    assert table.rstrip().endswith("PASS")
    sim.write_report(tmp_path, dict(trials=5), results)
    rep = json.loads((tmp_path / "acceptance.json").read_text())
    assert rep["verdict"] == "PASS" and rep["config"]["trials"] == 5 and len(rep["criteria"]) == len(rows)


def _fake_codec(monkeypatch):
    """Stub out decode (finds the payload unless the image is the 99x99 'failed photo') and the channel."""
    calls = []

    def decode(rgb, *a, **k):
        calls.append(rgb.shape)
        return dict(found=[] if rgb.shape[:2] == (99, 99) else [dict(payload=sim.PAYLOAD)])
    monkeypatch.setattr(ic, "decode", decode)
    monkeypatch.setattr(sim, "channel", lambda enc, rng, preset, print_w: np.zeros((99, 99, 3), np.uint8))
    return calls


def test_acceptance_run_writes_report_and_images(tmp_path, monkeypatch, capsys):
    calls = _fake_codec(monkeypatch)
    ladder = np.full((50, 40, 3), 7, np.uint8)
    monkeypatch.setattr(ic, "make_ladder", lambda img: (ladder, []))
    src = tmp_path / "kodim23.png"
    cv2.imwrite(str(src), photo_like(360, 540))
    out = tmp_path / "out"
    assert sim.acceptance([str(src)], 2, ["easy", "hard", "nonsense"], 15.0, str(out)) == "FAIL"
    assert len(calls) == 6 + 1 + 2 * 2              # all views once, a 2nd random crop, 2 presets x 2 trials
    rep = json.loads((out / "acceptance.json").read_text())
    r = rep["results"]["kodim23"]
    assert r["geometry"]["clean"] == dict(ok=1, n=1) and r["geometry"]["crop60"] == dict(ok=2, n=2)
    assert r["print"] == dict(easy=dict(ok=0, n=2), hard=dict(ok=0, n=2))
    assert rep["config"]["trials"] == 2 and rep["config"]["delta"] == 2.0 and rep["verdict"] == "FAIL"
    assert "| kodim23 | print gray | not run |" in capsys.readouterr().out
    img = cv2.imread(str(src)).astype(int)
    enc, diff = (cv2.imread(str(out / f"kodim23_{t}.png")).astype(int) for t in ("enc", "diff_x10"))
    assert enc.shape == img.shape and np.array_equal(diff, np.clip(np.abs(enc - img) * 10, 0, 255))
    assert np.array_equal(cv2.imread(str(out / "kodim23_ladder.png")), ladder)


def test_acceptance_views_crop_is_60_percent_area():
    enc = photo_like(300, 400)
    v = sim.views(enc, np.random.default_rng(0))
    assert list(v) == list(sim.GEOMETRY)
    assert np.array_equal(v["rot270"], np.rot90(enc, -1)) and np.array_equal(v["mirror"][:, 0], enc[:, -1])
    assert 0.59 <= v["crop60"].shape[0] * v["crop60"].shape[1] / (300 * 400) <= 0.6


def test_cli_acceptance_defaults_and_overrides(monkeypatch):
    got = []
    monkeypatch.setattr(sim, "acceptance", lambda *a: got.append(a) or ("PASS" if len(got) < 3 else "FAIL"))
    for argv, want, code in (
            (["--quick"], (["testdata/kodim23.png"], 1, list(sim.PRESETS)), 0),
            (["--acceptance"], (sorted(glob.glob("testdata/*.png")), 5, list(sim.PRESETS)), 0),
            (["--acceptance", "--trials", "2", "--presets", "easy,gray", "a.png"], (["a.png"], 2, ["easy", "gray"]), 1)):
        monkeypatch.setattr(sys, "argv", ["sim.py", *argv])
        with pytest.raises(SystemExit) as e:
            sim.main()
        assert got[-1][:3] == want and e.value.code == code, argv
    assert got[0][4] == "results/acceptance"


def test_cli_matrix_defaults(tmp_path, monkeypatch, capsys):
    _fake_codec(monkeypatch)
    monkeypatch.setattr(ic, "encode", lambda img, payload, **k: (img, dict(ssim=1.0, mean_dL=0.0)))
    src = tmp_path / "x.png"
    cv2.imwrite(str(src), photo_like(64, 64))
    monkeypatch.setattr(sys, "argv", ["sim.py", str(src)])
    sim.main()
    out = capsys.readouterr().out
    assert " | ".join(sim.PRESETS) in out and out.count("| 0/2 | 0/2 | 0/2 | 0/2 |") == 2   # 2 deltas x 2 trials


if __name__ == "__main__":
    test_same_size_no_markers()
    test_found_without_border_any_orientation()
    test_survives_simulated_print_and_photo()
    print("ok")
