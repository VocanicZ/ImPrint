"""Run: .venv/bin/python test_imprint_cam.py  (also works under pytest)"""
import cv2
import numpy as np

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


if __name__ == "__main__":
    test_same_size_no_markers()
    test_found_without_border_any_orientation()
    test_survives_simulated_print_and_photo()
    print("ok")
