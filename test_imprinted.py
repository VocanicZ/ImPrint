"""Run: .venv/bin/python test_imprinted.py  (also works under pytest)"""
import binascii
import glob
import json
import os
import sys

import cv2
import numpy as np
import pytest
from reedsolo import ReedSolomonError, RSCodec

import imprinted
import imprinted.core as ic
import sim


def photo_like(h=720, w=1080, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx / w * 200 + 30, yy / h * 180 + 40, (xx + yy) / (w + h) * 150 + 60], -1)
    img += cv2.GaussianBlur(rng.normal(0, 12, img.shape), (0, 0), 1.5)   # photo-like texture, not pixel noise
    return np.clip(img, 0, 255).astype(np.uint8)


def test_same_size_no_markers():
    """Same size as the original, at any strength; at the default strength (the visibility budget,
    which replaced a fixed delta = 2.2) inside the PRD's visibility bounds, not just near them."""
    img = photo_like()
    enc, st = ic.encode(img, b"hello", subblock_px=6)
    assert enc.shape == img.shape
    assert st["ssim"] >= ic.VIS_SSIM >= 0.95 and st["mean_dL"] <= ic.VIS_DL <= 1.0, st
    assert ic.encode(img, b"hello", delta=2, subblock_px=6)[0].shape == img.shape


def test_package_encode_decode_round_trip_through_files(tmp_path):
    """The pip package's front door: paths, PIL images or arrays in, a str payload out."""
    src, out = tmp_path / "in.png", tmp_path / "out.png"
    cv2.imwrite(str(src), cv2.cvtColor(photo_like(), cv2.COLOR_RGB2BGR))
    enc = imprinted.encode(src, "héllo", out)
    assert enc.size == (1080, 720) and enc.mode == "RGB" and out.exists()
    assert imprinted.decode(out) == "héllo"
    assert imprinted.decode(np.asarray(enc), raw=True) == "héllo".encode()
    assert imprinted.decode(photo_like(seed=1)) is None


def test_quick_check_finds_the_pilot_at_any_turn_size_or_tilt_and_nothing_elsewhere():
    """The gate decode() runs first: a clean file in any orientation, shrunk or blown up, and a page
    turned 50 degrees (as steep as the decoder reads) all read as Imprinted; plain pictures don't."""
    enc, _ = ic.encode(photo_like(), b"hi", delta=2, subblock_px=6)
    for view in (enc, np.rot90(enc), enc[:, ::-1], cv2.resize(enc, None, fx=0.6, fy=0.6),
                 cv2.resize(enc, None, fx=4, fy=4), tilted(enc, 50)[0]):
        assert ic.quick_check(np.ascontiguousarray(view)), view.shape
    rng = np.random.default_rng(0)
    for plain in (photo_like(), photo_like(seed=1), rng.integers(0, 256, (900, 1200, 3), np.uint8)):
        assert ic.quick_check(plain) == []
    assert imprinted.detect(enc) and not imprinted.detect(photo_like())


def test_patterns_and_sync_layout_are_pinned():
    """The data patterns and the sync layout are the file format: if either changed (a numpy
    whose generator draws differently, an edit), every picture made before would stop reading."""
    import hashlib
    sha = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
    assert sha(ic._all_patterns(ic.SEED).astype(np.int8)) == "f4b0b50997766d3b32261d77c14e837bbff0f85a75583f4818b4601f3df92a31"
    on, pat, bits = ic._sync_layout()
    assert sha(np.concatenate([on.ravel().astype(np.int8), pat.ravel().astype(np.int16).view(np.int8),
                               bits.ravel().astype(np.int8)])) == "6630d4cbd2e04920817236073822a4e64993ad9924339780319684817f670d5e"


def test_sync_layout_reads_on_the_right_twin_only():
    """A quarter of the cells carry known bits: on the decoder's lattice the matched filter stands
    far out, on its 180-degree twin (cells misaligned) and on an unencoded picture it doesn't."""
    img = photo_like()
    enc, _ = ic.encode(img, b"hi", delta=2, subblock_px=6)
    lum = ic.to_lab(enc)[..., 0]
    zs = []
    for sgn in (1, -1):
        model = ic.refine(lum, ic.track(lum, ic.find_seeds(lum)[0], sgn))
        for picture in (lum, ic.to_lab(img)[..., 0]):
            m, w, used, rows, cols, unit = ic.cell_means(picture, model)
            ii, jj = np.mgrid[0:rows, 0:cols].reshape(2, -1)
            zs.append(ic.sync_find(m[used], w[used], ii[used], jj[used], unit)[0])
    right, wrong = max(zs[0], zs[2]), min(zs[0], zs[2])
    assert right > 15 and wrong < 5 and max(zs[1], zs[3]) < 5, zs


def test_quick_check_rules_out_only_after_every_source_had_its_look(monkeypatch):
    """A busy picture's faint pilot often shows no spectral line at all, only local grids (some only
    at half scale), and on a slow machine those come well past QUICK_REJECT: ruled out on the
    spectrum alone, it was lost (0.8.0 lost busy print photos that 0.7.1 reads). Nothing promising
    once the grid search has had its look still rules a picture out."""
    enc, _ = ic.encode(photo_like(), b"hi", delta=2, subblock_px=6)
    monkeypatch.setattr(ic, "_level_sync", lambda *a, **k: [])          # no spectral line anywhere
    monkeypatch.setattr(ic, "QUICK_REJECT", 0.0)                         # a machine too slow for any of it
    assert ic.quick_check(enc)
    assert ic.quick_check(photo_like(seed=1)) == []


def test_quick_check_rejects_halftone_dots_fast():
    """A halftone screen is a lattice too, and locks the pilot probes; without the sync layout it
    isn't Imprinted. And the gate gives up in bounded time, at any picture size."""
    yy, xx = np.mgrid[0:1500, 0:2000].astype(np.float32)
    u, v = (xx + yy) / 9.0, (xx - yy) / 9.0                        # 45-degree screen, ~6 px pitch
    dots = (np.cos(2 * np.pi * u) * np.cos(2 * np.pi * v) * 60 + 128 + photo_like(1500, 2000)[..., 0] * 0.3 - 40)
    screen = np.clip(np.dstack([dots] * 3), 0, 255).astype(np.uint8)
    assert ic.quick_check(screen) == []
    import time
    big = np.random.default_rng(0).integers(0, 256, (3000, 4000, 3), np.uint8)
    t0 = time.perf_counter()
    assert ic.quick_check(big) == []
    assert time.perf_counter() - t0 < ic.QUICK_PATIENCE + ic.QUICK_GRACE + 1.0   # bounded, with slack for a busy machine


def test_found_without_border_any_orientation():
    enc, _ = ic.encode(photo_like(), b"hello", delta=2, subblock_px=6)
    h, w = enc.shape[:2]
    views = (enc, np.rot90(enc), np.rot90(enc, 2), enc[:, ::-1],          # turned / upside down / mirrored
             enc[h // 6:h // 6 + int(h * 0.7), w // 5:w // 5 + int(w * 0.7)])  # cropped: no picture edge at all
    for view in views:
        found = ic.decode(np.ascontiguousarray(view))["found"]
        assert found and found[0]["payload"] == b"hello", view.shape
        assert found[0]["bit_ber"] == 0, found[0]                   # no bit wrong before LDPC


def test_survives_simulated_print_and_photo():
    enc, _ = ic.encode(photo_like(), b"IMPRINT1", delta=2, subblock_px=6)
    found = ic.decode(sim.channel(enc, np.random.default_rng(3), "easy"))["found"]
    assert found and found[0]["payload"] == b"IMPRINT1"


def test_survives_simulated_grayscale_print_and_photo():
    """K-only print (one 45-degree screen), tilted photo: a fixed-seed regression for `gray`."""
    enc, _ = ic.encode(photo_like(), b"IMPRINT1", delta=2, subblock_px=6)
    found = ic.decode(sim.channel(enc, np.random.default_rng(3), "gray"))["found"]
    assert found and found[0]["payload"] == b"IMPRINT1"


def test_pilot_frequencies_align_uniquely_without_symmetry():
    """The pilot set must fix the cell grid exactly once per cell, show no rotation/mirror
    symmetry beyond the 180-degree turn no cosine can see, and leave no near-ambiguous offset:
    the matched filter's best wrong peak stays well under the true one."""
    import itertools
    import math
    F = ic.PILOT_FREQS.astype(int)
    assert len(F) == len(ic.PILOT_PHASES) and len({tuple(f) for f in F} | {tuple(-f) for f in F}) == 2 * len(F)
    g = 0
    for a, b in itertools.combinations(F, 2):
        g = math.gcd(g, int(abs(a[0] * b[1] - a[1] * b[0])))
    assert g == 1                                        # the lines repeat together once per cell
    lines = {tuple(f) for f in F} | {tuple(-f) for f in F}
    for m in ([[0, -1], [1, 0]], [[0, 1], [-1, 0]], [[1, 0], [0, -1]], [[-1, 0], [0, 1]],
              [[0, 1], [1, 0]], [[0, -1], [-1, 0]]):
        assert {tuple(np.array(m) @ f) for f in lines} != lines, m
    n = 240                                              # matched-filter score over every offset
    d = np.arange(n) / n
    off = np.stack(np.meshgrid(d, d), -1).reshape(-1, 2)
    s = np.cos(2 * np.pi * off @ F.T).mean(1).reshape(n, n)
    peak = np.ones_like(s, bool)
    for dy, dx in itertools.product(range(-2, 3), repeat=2):
        peak &= s >= np.roll(s, (dy, dx), (0, 1))
    peak[0, 0] = False                                   # the true alignment
    assert s[peak].max() <= ic.PILOT_ALIAS_MAX


def test_qim_target_reads_right_at_bounded_cost():
    rng = np.random.default_rng(0)
    u = rng.uniform(-20, 20, 5000)                       # host projections far past several bit flips
    bits = rng.choice([-1.0, 1.0], u.size)
    margin = rng.uniform(0.2, 2.5, u.size)               # past step/2 = 1.5: capped
    t = ic.qim_target(u, bits, margin)
    got = bits * ic.qim_soft(t)
    assert (got >= np.minimum(margin, ic.QIM_STEP / 2) - 1e-9).all()
    assert (np.abs(t - u) <= ic.QIM_STEP / 2 + np.minimum(margin, ic.QIM_STEP / 2) + 1e-9).all()
    fine = bits * ic.qim_soft(u) >= margin                # already reads right with margin: untouched
    assert fine.any() and np.array_equal(t[fine], u[fine])


def test_qim_soft_is_sign_near_zero_and_flips_every_step():
    B = ic.QIM_STEP
    u = np.array([0.1, B / 2, B - 0.1, B + 0.1, 1.5 * B, -0.1, -B / 2, 2 * B + 0.1])
    assert np.array_equal(np.sign(ic.qim_soft(u)), [1, 1, 1, -1, -1, -1, -1, 1])
    assert ic.qim_soft(np.array([B / 2]))[0] == B / 2 and abs(ic.qim_soft(np.array([B]))[0]) < 1e-12


def _rs_bits(payload):
    """The #6 code, for comparison: payload + CRC-32, rate-1/2 Reed-Solomon."""
    msg = payload + binascii.crc32(payload).to_bytes(4, "big")
    return np.unpackbits(np.frombuffer(bytes(RSCodec(len(msg)).encode(msg)), np.uint8)).astype(np.float32) * 2 - 1


def _rs_read(soft, nbytes):
    """The #6 decoder: hard bytes, least-confident bytes as erasures, CRC-32."""
    wire, nsym = np.packbits(soft > 0).tobytes(), nbytes + 4
    weak = np.argsort(np.abs(soft).reshape(-1, 8).min(1))
    for ne in range(0, nsym + 1, 2):
        try:
            msg = bytes(RSCodec(nsym).decode(wire, erase_pos=[int(i) for i in weak[:ne]] or None)[0])
        except ReedSolomonError:
            continue
        if binascii.crc32(msg[:nbytes]).to_bytes(4, "big") == msg[nbytes:]:
            return msg[:nbytes]
    return None


def test_soft_decision_code_beats_reed_solomon_on_awgn_and_bsc():
    """Same rate, same tile: at 2 dB Es/N0 (AWGN, soft) and 6% raw BER (BSC, hard +-1) the LDPC
    code decodes nearly always where Reed-Solomon nearly never does."""
    rng = np.random.default_rng(7)
    n = 20
    for nb in (8, 32):
        payload = bytes(rng.integers(0, 256, nb, dtype=np.uint8))
        rates = {}
        for name, bits, dec in (("ldpc", ic.to_bits, ic.read_soft), ("rs", _rs_bits, _rs_read)):
            x = bits(payload)
            assert x.size == 8 * ic.frame_len(nb)
            awgn = sum(dec(x + rng.normal(0, 10 ** (-2 / 20), x.size), nb) == payload for _ in range(n))
            bsc = sum(dec(x * np.where(rng.random(x.size) < 0.06, -1, 1), nb) == payload for _ in range(n))
            rates[name] = (awgn / n, bsc / n)
        assert rates["ldpc"][0] >= 0.85 and rates["rs"][0] <= 0.1, (nb, rates)
        assert rates["ldpc"][1] >= 0.85 and rates["rs"][1] <= 0.4, (nb, rates)


def test_ldpc_round_trips_every_payload_length():
    rng = np.random.default_rng(2)
    for nb in range(1, ic.MAX_BYTES + 1):
        payload = bytes(rng.integers(0, 256, nb, dtype=np.uint8))
        x = ic.to_bits(payload)
        cw = (x > 0).astype(np.uint8)
        chk, bit, slots = ic.ldpc(8 * (nb + 4), x.size)
        assert not (np.bincount(chk, cw[bit], x.size - 8 * (nb + 4)) % 2).any()      # every check holds
        H = np.zeros((x.size - 8 * (nb + 4), x.size), int)
        H[chk, bit] = 1
        assert (H.T @ H)[~np.eye(x.size, dtype=bool)].max() == 1                      # no 4-cycles
        assert np.packbits(cw[:8 * nb]).tobytes() == payload                       # systematic
        assert ic.read_soft(x * rng.uniform(0.1, 3, x.size), nb) == payload
        assert ic.read_soft(x, nb + 1 if nb < ic.MAX_BYTES else nb - 1) is None      # wrong length


def test_unencoded_images_yield_no_payload(monkeypatch):
    rng = np.random.default_rng(4)
    for _ in range(20):                                   # noise of any scale, every payload length
        for nb in range(1, ic.MAX_BYTES + 1):
            soft = rng.normal(0, 1, 8 * ic.frame_len(nb)) * rng.choice([1e-3, 1, 1e3])
            assert ic.read_soft(soft, nb) is None
    for nb in (1, 8, 32):
        assert ic.read_soft(np.zeros(8 * ic.frame_len(nb)), nb) is None
    img = photo_like(360, 540)
    assert ic.decode(img)["found"] == []
    assert ic.decode(rng.integers(0, 256, img.shape, np.uint8))["found"] == []
    lab = ic.to_lab(img)                                  # a pilot but no data: lattice found, cells read
    lab[..., 0] += 3 * ic.pilot_field(*lab.shape[:2], ic.K * 6)
    calls = []
    real = ic.read_soft
    monkeypatch.setattr(ic, "read_soft", lambda soft, nb: calls.append(nb) or real(soft, nb))
    assert ic.quick_check(ic.to_rgb(lab)) == []                       # a pilot without the sync layout isn't Imprinted
    assert ic.decode(ic.to_rgb(lab), max_seeds=2, check=False)["found"] == [] and len(set(calls)) >= 10   # many lengths


def test_adversarial_host_still_decodes():
    """A host whose texture IS the data patterns, opposing every payload bit at several times the
    embedding strength: sign margins would have to skip every cell; the lattice just steps over."""
    payload = b"hello"
    img = photo_like()
    h, w = img.shape[:2]
    b = 6
    c = ic.K * b
    rows, cols, T = h // c, w // c, ic.tile_size(len(payload))
    ii, jj = np.mgrid[0:rows, 0:cols]
    t = ((ii % T) * T + (jj % T)).ravel()
    s = ic.to_bits(payload)[ic.bit_map(T * T, 8 * ic.frame_len(len(payload)))[t]]
    lab = ic.to_lab(img)
    lab[:rows * c, :cols * c, 0] += ic.field((-12 * s).astype(np.float32), rows, cols, b, ic.pats(T * T)[t])
    host = ic.to_rgb(lab)
    enc, st = ic.encode(host, payload, delta=2, subblock_px=b)
    assert st["wrong_cells"] < 0.02, st
    found = ic.decode(enc)["found"]
    assert found and found[0]["payload"] == payload and found[0]["bit_ber"] == 0


def test_lattice_scale_recovers_decoder_gain():
    rng = np.random.default_rng(1)
    n = 3000
    bits = rng.choice([-1.0, 1.0], n)
    u = ic.qim_target(rng.normal(0, 2.5, n), bits, rng.uniform(0.6, 1.5, n))
    w, used = rng.uniform(0.3, 2, n), np.ones(n, bool)
    for gain in (0.8, 1.0, 1.25):                       # print blur / camera: data vs pilot gain drifts
        c = u * 2.0 * ic.DATA_GAIN * gain + rng.normal(0, 0.05, n)
        assert abs(ic.lattice_scale(c, w, used, 2.0) / (2.0 * ic.DATA_GAIN * gain) - 1) < 0.07, gain


def test_pilot_phase_locks_the_offset_under_heavy_noise():
    """The 4th line makes the matched filter robust: under noise 6x the pilot's amplitude, the
    3-line set misses the offset on ~1 in 5 patches (it drifts toward its near-repeat); 4 lines
    recover nearly every one."""
    rng = np.random.default_rng(1)
    nc, S, ok = 6, 24, 0
    for _ in range(100):
        o = rng.uniform(0, 1, 2)
        patch = ic.pilot_field(nc * S, nc * S, S, o[0] * S, o[1] * S) + rng.normal(0, 6, (nc * S, nc * S))
        off, coh, snr = ic.pilot_phase(patch.astype(np.float32), nc, S)
        ok += np.abs((off - o + 0.5) % 1 - 0.5).max() < 0.05
    assert ok >= 95


def test_whitened_spectrum_takes_regions_between_n_and_2n():
    """Regions shrink to fit small images: the 3x3 windows spread over whatever is there, and a
    line still stands far above the whitened texture (~1)."""
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:400, 0:320]
    region = rng.normal(0, 4, (400, 320)) + np.cos(2 * np.pi * (20 * xx + 9 * yy) / 256)
    R = ic._whitened_spectrum(region.astype(np.float32), 256)
    assert R.shape == (256, 256)
    assert R[128 + 9, 128 + 20] > 20 and np.median(R) < 1.5


def test_track_and_refine_lock_onto_a_faint_pilot():
    """At the default pilot (four lines share it, so each is fainter than three were), a seed tracks outward and a refine pass densifies the map.
    Of the two 180-degree twins, exactly one puts every lattice point on a cell corner of the
    encoded grid (the other mirrors it about a point off the corners), and that one reads."""
    enc, _ = ic.encode(photo_like(360, 540), b"hi", delta=2, subblock_px=6)
    lum = ic.to_lab(enc)[..., 0]
    c = ic.K * 6
    seed = ic.find_seeds(lum)[0]
    g = np.stack(np.meshgrid(np.arange(-4, 5, 2.0), np.arange(-3, 4, 2.0)), -1).reshape(-1, 2)
    on_grid = []
    for sgn in (1, -1):
        model = ic.track(lum, seed, sgn)
        assert model is not None and model[2] >= 30, sgn               # enough probes to fit curl
        dense = ic._refine_pass(lum, model)
        assert dense[2] >= 8, sgn                                       # the dense re-measure fits
        p = (dense[0](g) + 0.5) / c
        on_grid.append((np.abs((p + 0.5) % 1 - 0.5).max() < 0.02, dense))
    assert sum(ok for ok, _ in on_grid) == 1
    assert ic.read(lum, next(m for ok, m in on_grid if ok))["payload"] == b"hi"


def test_pilot_warp_finds_a_few_percent_scale_and_shear():
    """A probe sampled through a map whose local cell vectors are a few % off (a tilted page, one
    step from the last fitted point) smears the pilot: its phase no longer reads. pilot_warp finds
    the warp from where all four lines really are, even with the pilot far under the noise."""
    rng = np.random.default_rng(0)
    nc, S = 8, 24
    y, x = np.mgrid[0:nc * S, 0:nc * S]
    s = np.stack([(x + 0.5) / S - nc / 2, (y + 0.5) / S - nc / 2], -1)    # sampled lattice coords
    noise = rng.normal(0, 3, x.shape)
    seen = lambda u: (sum(np.cos(2 * np.pi * (u @ f) + ph) for f, ph in zip(ic.PILOT_FREQS, ic.PILOT_PHASES))
                      + noise).astype(np.float32)
    for M in (np.array([[1.06, 0.03], [-0.02, 0.95]]), np.array([[0.92, -0.05], [0.05, 1.06]])):
        patch = seen(s @ M)                                            # true lattice = M^T sampled
        assert ic.pilot_phase(patch, nc, S)[2] < ic.SNR_MIN            # smeared: the probe fails
        est = ic.pilot_warp(patch, nc, S)
        assert est is not None and np.abs(est - M).max() <= 0.021, (est, M)   # coarse grid: 0.02 steps
        assert ic.pilot_phase(seen(s @ np.linalg.inv(est) @ M), nc, S)[2] > 2 * ic.SNR_MIN   # re-aimed


def tilted(img, deg, f=1400.0):
    """img seen by a pinhole camera (focal length f px) with the page turned deg about its
    vertical axis: one side recedes, cells shrink across the page. Returns (photo, homography)."""
    h, w = img.shape[:2]
    r = np.deg2rad(deg)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    P = np.c_[src - [w / 2, h / 2], np.zeros(4)] @ np.array([[np.cos(r), 0, -np.sin(r)], [0, 1, 0],
                                                            [np.sin(r), 0, np.cos(r)]]).T
    P[:, 2] += f
    H = cv2.getPerspectiveTransform(src, (f * P[:, :2] / P[:, 2:] + [w / 2, h / 2]).astype(np.float32))
    return cv2.warpPerspective(img, H, (w, h), borderValue=(255, 255, 255)), H


def test_track_follows_a_steeply_tilted_page():
    """Page turned 40 degrees: cells shrink by a third across it, and a map that is right at the
    seed is a few % off 8 cells out. Growing probe by probe, each from its neighbours' local map
    and re-aimed when smeared, the tracker covers the page from a seed of the true local grid,
    and one twin reads. (Following the seed's affine map outward, it got no further than the seed.)"""
    enc, _ = ic.encode(photo_like(), b"hi", delta=2, subblock_px=6)
    img, H = tilted(enc, 40)
    lum = ic.to_lab(img)[..., 0]
    c = ic.K * 6
    p = cv2.perspectiveTransform(np.array([[[540, 360], [540 + c, 360], [540, 360 + c]]], np.float64), H)[0]
    seed = (0, p[0], p[1] - p[0], p[2] - p[0])
    reads = []
    for sgn in (1, -1):
        model = ic.track(lum, seed, sgn)
        assert model is not None and model[2] >= 100, sgn
        reads.append((ic.read(lum, ic.refine(lum, model)) or {}).get("payload"))
    assert b"hi" in reads


def test_homography_probe_reads_where_the_local_affine_smears():
    """Page turned 50 degrees: across an 8-cell probe the cell size changes enough that even the
    exact local affine map is off at the patch's edges, and the lines smear below the gate.
    Through the homography of six probes beside it, the same patch reads clearly and the probe
    lands on its lattice point."""
    enc, _ = ic.encode(photo_like(), b"hi", delta=2, subblock_px=6)
    img, H = tilted(enc, 50)
    img = np.clip(img + np.random.default_rng(0).normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    lum = ic.to_lab(img)[..., 0]
    T = lambda q: cv2.perspectiveTransform((np.atleast_2d(q) * 24)[None].astype(np.float64), H)[0] - 0.5
    for q in (np.array([10.0, 15.0]), np.array([30.0, 15.0])):
        P = T(q)[0]
        J = np.c_[T(q + [1, 0])[0] - P, T(q + [0, 1])[0] - P]
        affine = ic.pilot_phase(ic.resample(lum, lambda x: P + (x - q) @ J.T, q[0] - 4, q[1] - 4, 8, 8, 24), 8, 24)
        assert affine[2] < ic.SNR_MIN, affine
        near = [(q + d, T(q + d)[0]) for d in np.array([[-4, 0], [-2, 0], [-4, 2], [-4, -2], [-2, 2], [-2, -2]], float)]
        hit = ic.homography_probe(lum, q, near)
        assert hit is not None and hit[3] > 4 * ic.SNR_MIN and np.abs(hit[0] - q).max() < 0.05, hit


def test_map_probe_reaims_where_a_far_fitted_map_smears():
    """The tracker's last resort probes the frontier through the homography of every probe so far.
    Fitted on one side of a steeply tilted page, that map is off a few % in scale and turn on the
    far side, and an 8-cell patch through it smears below the gate. Re-aimed through the map's
    local affine at the point, the probe locks on the true lattice point and cell vectors."""
    enc, _ = ic.encode(photo_like(), b"hi", delta=2, subblock_px=6)
    img, H = tilted(enc, 30)
    img = np.clip(img + np.random.default_rng(0).normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    lum = ic.to_lab(img)[..., 0]
    T = lambda q: cv2.perspectiveTransform((np.atleast_2d(q) * 24)[None].astype(np.float64), H)[0] - 0.5
    for q in (np.array([10.0, 15.0]), np.array([30.0, 15.0])):
        J = np.c_[T(q + [1, 0])[0] - T(q)[0], T(q + [0, 1])[0] - T(q)[0]]
        for sc, deg in ((1.06, 4), (0.94, -5)):
            a = np.deg2rad(deg)
            M = sc * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            g = lambda x: T(q + (np.atleast_2d(x) - q) @ M.T)    # right at q, off around it
            assert ic.pilot_phase(ic.resample(lum, g, q[0] - 4, q[1] - 4, 8, 8, 24), 8, 24)[2] < ic.SNR_MIN
            assert ic.map_probe(lum, q, g) is None
            hit = ic.map_probe(lum, q, g, reaim=True)
            assert hit is not None and hit[3] >= ic.SNR_MIN, hit
            assert np.abs(hit[0] - q).max() < 0.1 and np.linalg.norm(hit[1] - T(q)[0]) < 1.0, hit
            assert np.linalg.norm(hit[2] - J) < 0.5 * np.linalg.norm(J @ M - J), (hit[2], J)


def test_grid_search_seeds_a_page_the_spectral_peaks_miss(monkeypatch):
    """With no spectral seed at all (a steep tilt smears every window's lines to the noise), decode
    falls back to the grid search: every plausible local grid scored on all four lines at once,
    verified by a probe and its neighbours. On a page turned 40 degrees it finds the true local
    grid, of the page and of its mirror image, and the page reads."""
    enc, _ = ic.encode(photo_like(), b"hi", delta=2, subblock_px=6)
    img, H = tilted(enc, 40)
    monkeypatch.setattr(ic, "find_seeds", lambda lum: [])
    found = ic.decode(img, check=False)["found"]
    assert found and found[0]["payload"] == b"hi"
    flip = np.diag([-1.0, 1, 1])
    flip[0, 2] = img.shape[1] - 1
    for view, Hv in ((img, H), (np.ascontiguousarray(img[:, ::-1]), flip @ H)):
        seeds = ic.grid_seeds(ic.to_lab(view)[..., 0])
        assert seeds, "no grid seed"
        _, P, e1, e2 = seeds[0]                      # the best: the true cell vectors there (either twin)
        p = cv2.perspectiveTransform(np.array([[P]], np.float64), np.linalg.inv(Hv))[0, 0]
        q = cv2.perspectiveTransform(np.array([[p, p + [24, 0], p + [0, 24]]], np.float64), Hv)[0]
        J = np.c_[q[1] - q[0], q[2] - q[0]]
        assert min(np.linalg.norm(np.c_[e1, e2] - s * J) for s in (1, -1)) < 0.05 * np.linalg.norm(J), (seeds[0], J)


def test_default_strength_fits_the_visibility_budget():
    """No delta given: the encoder takes the strongest delta inside the visibility budget. A busy
    picture gets less than a smooth one, both stay inside it, and a little more would not."""
    smooth = photo_like(360, 540)
    rng = np.random.default_rng(3)
    busy = np.clip(smooth + cv2.GaussianBlur(rng.normal(0, 60, smooth.shape), (0, 0), 1.0), 0, 255).astype(np.uint8)
    inside = lambda st: st["ssim"] >= ic.VIS_SSIM and st["mean_dL"] <= ic.VIS_DL
    deltas = []
    for img in (smooth, busy):
        enc, st = ic.encode(img, b"hi", subblock_px=6)
        assert inside(st), st
        assert st["delta"] == ic.DELTA_MAX or not inside(ic.encode(img, b"hi", st["delta"] * 1.1, 6)[1]), st
        deltas.append(st["delta"])
    assert ic.DELTA_MIN < deltas[1] < deltas[0], deltas
    assert ic.encode(smooth, b"hi", delta=1.3, subblock_px=6)[1]["delta"] == 1.3


def test_deridge_keeps_spikes_and_flattens_streaks():
    """A straight edge is a streak through the spectrum with strong local maxima all along it;
    a pilot line is an isolated spike. Only the spike may survive as a candidate line."""
    rng = np.random.default_rng(5)
    R = rng.gamma(9, 1 / 9, (128, 128)).astype(np.float32)   # Welch-averaged whitened texture: ~1
    R[40, 20:110] += 25 + 5 * np.cos(np.arange(90))          # streak with bumps along it
    R[90, 70] += 8                                           # faint isolated line
    D = ic._deridge(R)
    assert D[90, 70] == pytest.approx(R[90, 70], rel=0.3)    # the spike keeps its strength
    assert D[40, 30:100].max() < 4 < D[90, 70]                # the streak drops to texture level


def test_data_lattice_does_not_scale_with_the_pilot(monkeypatch):
    """The lattice unit is the pilot amplitude at PILOT = 1, so a fainter pilot leaves the data's
    lattice (its robustness) unchanged, and the decoder measures the same unit."""
    img = photo_like(360, 540)
    units = []
    for p in (1.0, 0.5):
        monkeypatch.setattr(ic, "PILOT", p)
        enc, st = ic.encode(img, b"hi", delta=2, subblock_px=6)
        assert st["wrong_cells"] < 0.02, (p, st)
        units.append(ic.cell_means(ic.to_lab(enc)[..., 0], (lambda q: q * 24 - 0.5, (2.0, 20.0, 2.0, 12.0), 4))[5])
    assert units[1] == pytest.approx(units[0], rel=0.15)


def test_small_image_scanned_with_large_spectral_windows(monkeypatch):
    """Images under 512 px a side still get 256-px spectral windows (the line's power over the
    texture grows with the window area); regions shrink to fit and the last one sits flush."""
    assert ic._starts(594, 512, 128) == [0, 82] and ic._starts(512, 512, 128) == [0]
    calls = []
    real = ic.sync_in_region
    monkeypatch.setattr(ic, "sync_in_region", lambda reg, n: calls.append((reg.shape, n)) or real(reg, n))
    enc, _ = ic.encode(photo_like(400, 560), b"hi", delta=2, subblock_px=6)
    seeds = ic.find_seeds(ic.to_lab(enc)[..., 0])
    assert calls[0] == ((400, 512), 256)
    cell = np.sqrt(abs(np.linalg.det(np.c_[seeds[0][2], seeds[0][3]])))
    assert cell == pytest.approx(24, rel=0.03)


def test_refine_snaps_map_onto_the_pilot():
    enc, _ = ic.encode(photo_like(), b"hello", delta=2, subblock_px=6)
    lum = ic.to_lab(enc)[..., 0]
    c = ic.K * 6
    truth = lambda q: q * c - 0.5
    off = lambda q: truth(q * 1.004 + [0.12, -0.08])     # 0.4% scale and ~3 px off
    model = ic.refine(lum, (off, (8, 14, 8, 14), 4))       # small tracked extent: refine reaches out
    g = np.stack(np.meshgrid(np.arange(2, 42, 4.0), np.arange(2, 28, 4.0)), -1).reshape(-1, 2)
    assert model[2] >= 30 and np.abs(model[0](g) - truth(g)).max() < 0.6
    assert model[1][1] - model[1][0] > 30                   # extent now covers most of the picture


def test_refine_and_read_fall_back_without_a_pilot(monkeypatch):
    flat = np.full((400, 600), 50.0, np.float32)          # no pilot anywhere: nothing to refit to
    model = (lambda q: q * 24 - 0.5, (2.0, 20.0, 2.0, 12.0), 4)
    assert ic.refine(flat, model) is model
    assert ic.read(flat, (model[0], (0.0, 2.0, 0.0, 2.0), 4), grow=0) is None   # too few cells
    monkeypatch.setattr(ic, "pilot_phase", lambda patch, nc, S: (np.zeros(2), 1.0, 9.0))
    monkeypatch.setattr(ic.cv2, "findHomography", lambda *a, **k: (None, None))
    assert ic.refine(flat, model) is model                # coherent windows, but the refit fails


def test_decode_skips_seeds_on_a_lattice_that_already_failed(monkeypatch):
    """Both twins locked onto a dense refit and neither read: other seeds in that picture refit to
    the same lattices, so they're skipped; a seed elsewhere (another picture) still gets its turn.
    Pictures: x < 200 both twins lock; x 200-400 neither; x > 400 one twin, and its turn-around."""
    seeds = [(9.0, np.array([x, 50.0]), np.array([10.0, 0]), np.array([0, 10.0])) for x in (40, 60, 80, 300, 500, 520)]
    monkeypatch.setattr(ic, "find_seeds", lambda lum: seeds)
    tracked = []
    monkeypatch.setattr(ic, "track", lambda lum, seed, sgn, **_: tracked.append(seed[1][0]) or
                        (lambda q, c=seed[1]: c + 10 * q, (-5.0, 5.0, -5.0, 5.0), 9 if sgn > 0 else 8))

    def refine(lum, m):
        x = m[0](np.zeros((1, 2)))[0, 0]
        return m[0], m[1], 40 if x < 200 or (x > 400 and m[2] != 8) else 5
    monkeypatch.setattr(ic, "refine", refine)
    monkeypatch.setattr(ic, "read", lambda lum, m, **_: None)
    r = ic.decode(np.zeros((100, 600, 3), np.uint8), check=False)
    assert r["tried"] == 3 and tracked == [40, 40, 300, 300, 500, 500] and r["found"] == []


def test_decode_turns_a_lone_locked_twin_around(monkeypatch):
    """Only the wrong 180-degree twin locks on; the right one tracks badly. decode refits the
    locked lattice turned around, and that reads."""
    c = np.array([100.0, 50.0])
    monkeypatch.setattr(ic, "find_seeds", lambda lum: [(9.0, c, np.array([10.0, 0]), np.array([0, 10.0]))])
    monkeypatch.setattr(ic, "track", lambda lum, seed, sgn, **_: (lambda q: c + 10 * q if sgn > 0 else c + 3 - 10 * q,
                                                             (-5.0, 5.0, -5.0, 5.0), 9))
    monkeypatch.setattr(ic, "refine", lambda lum, m: (m[0], m[1], 5 if m[0](np.zeros((1, 2)))[0, 0] != c[0] else 40))
    right = lambda m: np.allclose(m[0](np.array([[0.0, 0.0], [1.0, 0.0]])), [c, c - [10, 0]])
    monkeypatch.setattr(ic, "read", lambda lum, m, **_: dict(payload=b"twin") if right(m) else None)
    found = ic.decode(np.zeros((100, 400, 3), np.uint8), check=False)["found"]
    assert [f["payload"] for f in found] == [b"twin"]
    assert np.allclose(found[0]["quad"], [[150, 100], [50, 100], [50, 0], [150, 0]])


def test_small_picture_search_stops_at_its_patience(monkeypatch):
    """A small picture's quick look can't rule it out, so detect() and existing() search it, but no
    longer than SMALL_PATIENCE: 40 slow seeds (0.1 s each, 4 s in all) stop being tried once it's
    up, and the track under way is handed the same deadline, as is the grid search (whose seeds are
    tracked as it finds them: its probing takes seconds on a busy picture). A lattice whose
    full map holds the sync layout settles it (no payload read), and so does a quick look that
    confirms it."""
    import time
    seeds = [(9.0, np.array([20.0 + x, 50.0]), np.array([3.0 * ic.K, 0]), np.array([0, 3.0 * ic.K])) for x in range(40)]
    monkeypatch.setattr(ic, "find_seeds", lambda lum: seeds)
    grids = []
    monkeypatch.setattr(ic, "_grid_seeds", lambda lum, until=np.inf, **_: grids.append(until) or iter(()))
    monkeypatch.setattr(ic, "quick_check", lambda rgb, *a, **k: [])
    monkeypatch.setattr(ic, "read", lambda *a, **k: pytest.fail("the sync settles it: no payload read"))
    monkeypatch.setattr(ic, "SMALL_PATIENCE", 1.0)
    untils, lattice = [], (lambda q: q * 3.0 * ic.K, (0.0, 40.0, 0.0, 40.0), 50)
    monkeypatch.setattr(ic, "track", lambda lum, seed, sgn, until=np.inf, **_: untils.append(until) or
                        time.sleep(0.1) or lattice)
    monkeypatch.setattr(ic, "_sync_read", lambda img, model, *a: 4.0)  # tracks, but holds no sync layout
    small = np.zeros((400, 600, 3), np.uint8)
    ic.to_lab(small)                                              # OpenCV's first colour conversion: ~1 s
    for look in (ic.detect, ic.existing):
        untils.clear()
        t0 = time.perf_counter()
        assert not look(small)
        assert time.perf_counter() - t0 < 2.5 and 0 < len(untils) < 30, (look, len(untils))
        assert 1.0 < untils[0] - t0 < 1.1                     # its call's start + SMALL_PATIENCE
    monkeypatch.setattr(ic, "find_seeds", lambda lum: [])           # nothing spectral: the grid search's turn
    t0 = time.perf_counter()
    assert not ic.detect(small) and len(grids) == 1 and 1.0 < grids[0] - t0 < 1.1
    monkeypatch.setattr(ic, "find_seeds", lambda lum: seeds)
    monkeypatch.setattr(ic, "_sync_read", lambda img, model, *a: 30.0 if len(untils) >= 3 else 4.0)
    for look in (ic.detect, ic.existing):
        untils.clear()
        assert look(small) and len(untils) == 3
    assert ic.existing(small) == dict(subblock_px=3, payload=None)
    monkeypatch.setattr(ic, "quick_check", lambda rgb, *a, **k: [seeds[0]])
    monkeypatch.setattr(ic, "_sync_search", lambda *a, **k: pytest.fail("confirmed: nothing to search"))
    assert ic.detect(small)
    monkeypatch.setattr(ic, "pilot_probe", lambda *a, **k: pytest.fail("past its deadline: nothing probed"))
    noise = np.random.default_rng(0).normal(50, 10, (600, 800)).astype(np.float32)
    assert ic.grid_candidates(noise[:256, :256])                   # candidates to probe, but no time to
    assert list(ic._grid_seeds(noise, until=time.perf_counter())) == []


def test_cell_means_grows_but_drops_cells_outside_the_photo():
    enc, _ = ic.encode(photo_like(), b"hello", delta=2, subblock_px=6)
    lum = ic.to_lab(enc)[..., 0]
    c = ic.K * 6
    model = (lambda q: q * c - 0.5, (0.0, 10.0, 0.0, 10.0), 4)     # tracked: one corner of the picture
    m, w, used, rows, cols, unit = ic.cell_means(lum, model, grow=8)
    assert (rows, cols) == (26, 26) and unit > 0
    inside = np.zeros((rows, cols), bool)
    inside[8:, 8:] = True                                 # lattice -8..18: cells < 0 lie off the photo
    assert not used.reshape(rows, cols)[~inside].any() and used.reshape(rows, cols)[inside].mean() > 0.5


def test_decoder_sees_the_cells_the_encoder_set_on_a_busy_host():
    """Busy pictures at a busy picture's low strength, 4-px sub-blocks: the decoder resamples at
    its own 6 px per sub-block. It averages the same span of each sub-block as the encoder, and the
    encoder's target leaves out the mean cell it subtracts, so the clean file reads at the true
    grid with few wrong cells. (A wider decoder window and a target that kept the pilot in left
    7% / 4.6% of the cells wrong: busy texture seen through a different window is noise.)"""
    for name, most in (("kodim08", 0.05), ("kodim05", 0.025)):
        img = cv2.cvtColor(cv2.imread(f"testdata/{name}.png"), cv2.COLOR_BGR2RGB)
        enc, st = ic.encode(img, b"IMPRINT1", delta=0.8, subblock_px=4)
        lum = ic.to_lab(enc)[..., 0]
        r = ic.read(lum, (lambda q: q * 16 - 0.5, (0.0, st["cols"], 0.0, st["rows"]), 100), grow=0)
        assert r and r["payload"] == b"IMPRINT1" and r["raw_ber"] < most, (name, r)


def test_inner_window_spans_the_same_half_of_a_sub_block_at_any_size():
    """The averaged span is [b/4, 3b/4) at every sub-block size, each px weighted by its overlap,
    so a 4-px encoder sub-block and a 6-px decoder resample see the same thing: the mean of a
    linear ramp across the sub-block (the span's centre) and the window's mean over that span."""
    for b in (4, 5, 6, 8):
        w = ic.inner(b)
        assert w.sum() == pytest.approx(1) and w.min() >= 0
        x = (np.arange(b) + 0.5) / b                                  # px centres, as a fraction of the sub-block
        assert w @ x == pytest.approx(0.5)
        assert np.flatnonzero(w)[[0, -1]].tolist() == [int(b / 4), int(np.ceil(3 * b / 4)) - 1]
    sin = lambda u: np.sin(np.pi * u)                                 # the window's profile
    span = np.linspace(0.25, 0.75, 2001)
    for b in (8, 16, 32):                                             # gain -> the window's mean over the span
        assert np.sqrt(ic.gain(b)) == pytest.approx(sin(span).mean(), abs=0.01), b   # (small b: px sampling)
    assert ic.gain(6) == pytest.approx(ic.inner(6) @ ic.window(6) @ ic.inner(6))


def test_grid_bases_cover_every_turn_mirror_and_foreshortening():
    """The grid search's candidate unit cells: area 1, both mirror images, every turn up to 180
    degrees (a cosine can't see a 180-degree turn), foreshortening up to GRID_ANIS[-1]; G puts
    each pilot line where that grid would."""
    B, G = ic._grid_bases()
    det = np.linalg.det(B)
    assert np.allclose(np.abs(det), 1) and (det > 0).any() and (det < 0).any()
    assert np.allclose(G, ic.PILOT_FREQS @ np.linalg.inv(B))
    ratio = np.array([np.sqrt(np.linalg.eigvalsh(b.T @ b)[::-1][0] / np.linalg.eigvalsh(b.T @ b)[0]) for b in B])
    assert ratio.max() == pytest.approx(ic.GRID_ANIS[-1]) and ratio.min() == pytest.approx(1)
    rng = np.random.default_rng(0)
    for _ in range(20):                                               # any similarity grid has a close base
        th, mir = rng.uniform(0, 2 * np.pi), rng.choice([1, -1])
        A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([1, mir])
        assert min(np.abs(b - s * A).max() for b in B for s in (1, -1)) < np.deg2rad(ic.GRID_ROT)


def test_pilot_rect_finds_the_picture():
    """Cells on the picture carry the pilot (~1 x the median), cells off it don't (paper, clutter);
    a few noisy cells either way don't move the rectangle."""
    rng = np.random.default_rng(0)
    level = rng.uniform(0, 0.3, (40, 60))
    level[6:30, 9:51] = rng.uniform(0.6, 1.6, (24, 42))
    level[10, 20] = level[20, 30] = 0.0                       # weak cells inside
    level[2, 2] = level[35, 55] = 2.0                         # strays outside
    assert ic.pilot_rect(level) == (6, 30, 9, 51)


def test_decode_reads_the_tracked_map_when_the_refit_misses(monkeypatch):
    """The dense tracker can fit busy texture better than a refine pass of small windows: when the
    refined map doesn't read, the tracked one gets its turn."""
    c = np.array([100.0, 50.0])
    tracked = (lambda q: c + 10 * q, (-5.0, 5.0, -5.0, 5.0), 200)
    monkeypatch.setattr(ic, "find_seeds", lambda lum: [(9.0, c, np.array([10.0, 0]), np.array([0, 10.0]))])
    monkeypatch.setattr(ic, "grid_seeds", lambda lum: [])
    monkeypatch.setattr(ic, "track", lambda lum, seed, sgn, **_: tracked)
    monkeypatch.setattr(ic, "refine", lambda lum, m: (lambda q: c + 3 + 10 * q, m[1], 40))
    monkeypatch.setattr(ic, "read", lambda lum, m, **_: dict(payload=b"trk") if m is tracked else None)
    found = ic.decode(np.zeros((100, 400, 3), np.uint8), check=False)["found"]
    assert [f["payload"] for f in found] == [b"trk"]


def zones(h=288, w=576, seed=0):
    """Three side-by-side zones: flat mid-gray | texture at the pattern's scale | clipped (pure white
    over pure black)."""
    rng = np.random.default_rng(seed)
    z = w // 3
    img = np.full((h, w, 3), 128.0)
    img[:, z:2 * z] += cv2.GaussianBlur(rng.normal(0, 50, (h, z)), (0, 0), 1.5)[..., None]
    img[:h // 2, 2 * z:], img[h // 2:, 2 * z:] = 255, 0
    return np.clip(img, 0, 255).astype(np.uint8)


def test_energy_goes_to_texture_not_flat_or_clipped():
    img = zones()
    enc, _ = ic.encode(img, b"hi", delta=2, subblock_px=6)
    d = np.abs(ic.to_lab(enc)[..., 0] - ic.to_lab(img)[..., 0])
    z, m = img.shape[1] // 3, 24                                     # zone width, one-cell margin at seams
    flat, tex, clipped = d[:, m:z - m], d[:, z + m:2 * z - m], d[:, 2 * z + m:]
    assert clipped.max() == 0
    # flat keeps only the floor (it carries the cheapest margin: a flat host needs no QIM nudge);
    # texture gets far more. The Laplacian mask gave flat 0.64x texture here.
    assert flat.mean() < 0.55 * tex.mean(), (flat.mean(), tex.mean())


def test_jnd_follows_luminance_and_texture():
    b, flat = 6, np.full((96, 96), 50, np.float32)
    mid, dark = ic.jnd(flat, b).mean(), ic.jnd(flat - 40, b).mean()
    assert dark > mid                                                # luminance adaptation: dark hides more
    rng = np.random.default_rng(0)
    tex = flat + cv2.GaussianBlur(rng.normal(0, 30, flat.shape).astype(np.float32), (0, 0), 1.5)
    assert ic.jnd(tex, b)[24:-24, 24:-24].mean() > 3 * mid           # contrast masking


def kodim(name, h=None, w=None):
    img = cv2.cvtColor(cv2.imread(f"testdata/{name}.png"), cv2.COLOR_BGR2RGB)
    return img if h is None else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def test_no_pixel_moves_more_than_a_bounded_multiple_of_delta():
    """kodim08's clipped highlights leave some cells room in only a corner: the cell's whole QIM
    nudge went onto those pixels (41 L*, 37x delta, at the default strength). Now the data swing
    is capped at SWING_MAX x delta and the pilot is at most PILOT x PILOT_HI x delta; the notch
    only takes out the host's own content at the pilot's lines (a few L*)."""
    img = kodim("kodim08")
    for delta in (None, 2.0):
        enc, st = ic.encode(img, b"hello", delta=delta)
        d = np.abs(ic.to_lab(enc)[..., 0] - ic.to_lab(img)[..., 0])
        assert d.max() <= (ic.SWING_MAX + ic.PILOT * ic.PILOT_HI) * st["delta"], (d.max(), st)
        assert st["wrong_cells"] < 0.02, st


def test_encode_rejects_pictures_too_small_to_read_back():
    """Every picture encode accepts reads back from its clean file; smaller ones are a ValueError.
    These all encoded and then read nothing: 240x300 px (under the decoder's 256-px windows), and
    busy kodim05 at 14x14 cells (1.4 tiles) and at 15x22 cells (3.3 tiles: how few cells the pilot
    locks on depends on the picture, so a small one is decoded before encode returns it)."""
    for (h, w), b in (((240, 300), 4), ((255, 320), 4), ((448, 448), 8), ((360, 540), 6)):
        with pytest.raises(ValueError, match="too small"):
            ic.encode(kodim("kodim05", h, w), b"hello", subblock_px=b)
    enc, st = ic.encode(kodim("kodim08", 256, 320), b"hello", subblock_px=4)     # at the px floor
    found = ic.decode(enc)["found"]
    assert found and found[0]["payload"] == b"hello", st


def test_grayscale_and_rgba_pictures_round_trip():
    """Grayscale in, grayscale out; RGBA keeps its alpha; both decode. Other shapes: ValueError."""
    rgb = photo_like(360, 540)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgba = np.dstack([rgb, np.tile(np.linspace(0, 255, rgb.shape[1]).astype(np.uint8), (rgb.shape[0], 1))])
    for img in (gray, gray[..., None], rgba):
        enc, st = ic.encode(img, b"hi", delta=2, subblock_px=6)
        assert enc.shape == img.shape and enc.dtype == np.uint8 and st["wrong_cells"] < 0.02, st
        if img.ndim == 3 and img.shape[2] == 4:
            assert np.array_equal(enc[..., 3], img[..., 3])
        found = ic.decode(enc)["found"]
        assert found and found[0]["payload"] == b"hi", img.shape
    for bad in (rgb[..., :2], rgb[..., 0, 0]):
        with pytest.raises(ValueError, match="grayscale"):
            ic.encode(bad, b"hi", delta=2, subblock_px=6)
        with pytest.raises(ValueError, match="grayscale"):
            ic.decode(bad)


def test_host_without_headroom_raises_or_warns():
    """Pure black or white can't swing either way: nothing would be embedded, so encode raises
    instead of returning wrong_cells 1.0. Half of it clipped still carries the payload, but warns."""
    img = photo_like(360, 540)
    for v in (0, 255):
        with pytest.raises(ValueError, match="headroom"):
            ic.encode(np.full_like(img, v), b"hi", subblock_px=6)
    half = photo_like()
    half[:, : half.shape[1] // 2] = 0
    with pytest.warns(RuntimeWarning, match="read wrong"):
        ic.encode(half, b"hi", delta=2, subblock_px=6)


def _passing(img="kodim23"):
    """Synthetic measurements that just meet every PRD #1 threshold for img."""
    return dict(ssim=0.95, mean_dL=1.0, geometry={v: dict(ok=5, n=5) for v in sim.GEOMETRY},
                print={p: dict(ok=round(r * 5), n=5) for p, r in sim.PRINT_MIN[img].items()})


def _all_passing():
    """Every PRD #1 image at 5 trials, each just meeting its thresholds."""
    return {img: _passing(img) for img in sim.PRINT_MIN}


def test_acceptance_criteria_pass_on_threshold():
    rows, verdict = sim.criteria(_all_passing())
    assert verdict == "PASS" and all(r["passed"] for r in rows)
    assert sorted({r["image"] for r in rows}) == sorted(sim.PRINT_MIN)
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
    rows, verdict = sim.criteria(dict(_all_passing(), kodim23=r))
    assert verdict == "FAIL"
    assert sorted(x["criterion"] for x in rows if x["passed"] is False) == ["decode crop60", "print hard"]
    assert sim.criteria(dict(_all_passing(), kodim05=dict(_passing("kodim05"), print=dict(
        easy=dict(ok=3, n=5), gray=dict(ok=2, n=5)))))[1] == "FAIL"


def test_acceptance_criteria_unmeasured_is_incomplete_not_pass():
    r = _passing()
    del r["print"]["curled"]                             # e.g. --presets easy,gray,hard
    rows, verdict = sim.criteria(dict(_all_passing(), kodim23=r))
    assert verdict == "INCOMPLETE"
    assert [x["passed"] for x in rows if x["criterion"] == "print curled"] == [None]
    r["print"]["hard"] = dict(ok=0, n=5)
    assert sim.criteria({"kodim23": r})[1] == "FAIL"    # a measured failure beats a gap
    other = dict(_passing(), print={})                   # an image with no print requirement
    assert sim.criteria(dict(_all_passing(), photo1=other))[1] == "PASS"
    assert sim.criteria({"photo1": other})[1] == "INCOMPLETE"


def test_acceptance_criteria_missing_images_are_incomplete():
    rows, verdict = sim.criteria({})
    assert verdict == "INCOMPLETE" and rows and all(x["passed"] is None for x in rows)
    assert [x["image"] for x in rows if x["criterion"] == "SSIM"] == list(sim.PRINT_MIN)
    rows, verdict = sim.criteria({"kodim23": _passing()})      # e.g. --acceptance testdata/kodim23.png
    assert verdict == "INCOMPLETE"
    assert all(x["passed"] for x in rows if x["image"] == "kodim23")
    for img in ("kodim05", "kodim08"):
        miss = [x for x in rows if x["image"] == img]
        assert [x["criterion"] for x in miss] == ["SSIM", "mean dL"] + [f"decode {v}" for v in sim.GEOMETRY] + \
            [f"print {p}" for p in sim.PRINT_MIN[img]]
        assert all(x["passed"] is None and x["measured"] == "not run" for x in miss)
    assert "| kodim05 | print easy | not run | >= 0.60 | SKIP |" in sim.format_table(rows, verdict)


def test_acceptance_criteria_too_few_trials_are_incomplete():
    one = lambda img: dict(_passing(img), geometry={v: dict(ok=1, n=1) for v in sim.GEOMETRY},
                           print={p: dict(ok=1, n=1) for p in sim.PRINT_MIN[img]})
    rows, verdict = sim.criteria({img: one(img) for img in sim.PRINT_MIN})   # --trials 1, all decoded
    assert verdict == "INCOMPLETE"
    assert {x["criterion"] for x in rows if x["passed"] is None} == \
        {"decode crop60", "print easy", "print gray", "print hard", "print curled"}
    assert "| kodim23 | print hard | 1/1 (1.00), < 5 trials | >= 0.80 | SKIP |" in sim.format_table(rows, verdict)
    r = _passing()
    r["print"]["hard"] = dict(ok=3, n=4)                 # 4/5 still reachable
    assert sim.criteria(dict(_all_passing(), kodim23=r))[1] == "INCOMPLETE"
    r["print"]["hard"] = dict(ok=2, n=4)                 # 3/5 at best: already failed
    assert sim.criteria(dict(_all_passing(), kodim23=r))[1] == "FAIL"
    r = _passing()
    r["geometry"]["crop60"] = dict(ok=0, n=1)            # one missed crop already fails 100%
    assert sim.criteria(dict(_all_passing(), kodim23=r))[1] == "FAIL"
    r = _passing()
    r["print"]["easy"] = dict(ok=8, n=10)                # more trials than needed is fine
    assert sim.criteria(dict(_all_passing(), kodim23=r))[1] == "PASS"


def test_acceptance_table_and_json(tmp_path):
    results = _all_passing()
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
        return dict(found=[] if rgb.shape[:2] == (99, 99) else [dict(payload=sim.PAYLOAD, bit_ber=0.0125)])
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
    assert r["bit_ber"] == {v: 0.0125 for v in sim.GEOMETRY}      # reported for every decoded view
    assert rep["config"]["trials"] == 2 and rep["config"]["delta"] == "auto" and rep["verdict"] == "FAIL"
    assert ic.DELTA_MIN <= r["delta"] <= ic.DELTA_MAX                # the strength the budget picked
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


def test_cli_acceptance_defaults_and_overrides(monkeypatch, tmp_path):
    got, verdict = [], []
    monkeypatch.setattr(sim, "acceptance", lambda *a: got.append(a) or verdict[0])
    monkeypatch.chdir(tmp_path)                          # testdata/ resolves next to sim.py, not the cwd
    k23, every = [os.path.join(sim.TESTDATA, "kodim23.png")], sorted(glob.glob(os.path.join(sim.TESTDATA, "*.png")))
    assert [os.path.basename(p) for p in every] == ["kodim05.png", "kodim08.png", "kodim23.png"]
    for argv, v, want, code in (
            (["--quick"], "INCOMPLETE", (k23, 1, list(sim.PRESETS)), 0),   # --quick cannot PASS
            (["--quick"], "FAIL", (k23, 1, list(sim.PRESETS)), 1),
            (["--acceptance"], "PASS", (every, 5, list(sim.PRESETS)), 0),
            (["--acceptance"], "INCOMPLETE", (every, 5, list(sim.PRESETS)), 1),
            (["--acceptance", "--trials", "2", "--presets", "easy,gray", "a.png"], "FAIL",
             (["a.png"], 2, ["easy", "gray"]), 1)):
        verdict[:] = [v]
        monkeypatch.setattr(sys, "argv", ["sim.py", *argv])
        with pytest.raises(SystemExit) as e:
            sim.main()
        assert got[-1][:3] == want and e.value.code == code, (argv, v)
    assert got[0][4] == "results/acceptance"
    monkeypatch.setattr(sim, "TESTDATA", str(tmp_path / "empty"))
    monkeypatch.setattr(sys, "argv", ["sim.py", "--acceptance"])
    with pytest.raises(SystemExit) as e:
        sim.main()
    assert e.value.code == 2 and len(got) == 5          # argparse error, nothing run


def test_cli_matrix_defaults(tmp_path, monkeypatch, capsys):
    _fake_codec(monkeypatch)
    monkeypatch.setattr(ic, "encode", lambda img, payload, **k: (img, dict(ssim=1.0, mean_dL=0.0)))
    src = tmp_path / "x.png"
    cv2.imwrite(str(src), photo_like(64, 64))
    monkeypatch.setattr(sys, "argv", ["sim.py", str(src)])
    sim.main()
    out = capsys.readouterr().out
    assert " | ".join(sim.PRESETS) in out and out.count("| 0/2 | 0/2 | 0/2 | 0/2 |") == 2   # 2 deltas x 2 trials


def test_small_bench_fails_on_a_false_alarm_or_a_slow_detect(tmp_path, monkeypatch, capsys):
    """scripts/small_bench.py: its no-pattern extras are small pictures; a detect() that says yes
    to one of them, or takes past BOUND, is a FAIL, a miss alone is reported but not."""
    import pickle, time
    monkeypatch.syspath_prepend(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
    import small_bench
    extras = small_bench.extras()
    assert len(extras) == 6 and all(p is None and im.shape == (512, 768, 3) and im.dtype == np.uint8
                                    for p, _, im in extras)
    assert all(im.std() > 10 for _, name, im in extras if name.startswith("halftone"))
    cases = [(b"hi", "enc", photo_like(64, 64)), (None, "plain", photo_like(64, 64, 1))]
    path = tmp_path / "cases.pkl"
    pickle.dump(cases, open(path, "wb"))
    monkeypatch.setattr(ic, "quick_check", lambda rgb, *a, **k: [])
    monkeypatch.setattr(ic, "decode", lambda im, **k: dict(found=[]))
    monkeypatch.setattr(ic, "existing", lambda im: None)

    def run(detect):
        monkeypatch.setattr(ic, "detect", detect)
        monkeypatch.setattr(sys, "argv", ["small_bench.py", str(path)])
        with pytest.raises(SystemExit) as e:
            small_bench.main()
        return e.value.code, capsys.readouterr().out

    code, out = run(lambda im: False)                             # misses the encoded one, no alarm
    assert code == 0 and "verdict: PASS" in out and "detect:enc" in out and "detect 1/2" in out
    code, out = run(lambda im: True)                              # a yes on the plain picture
    assert code == 1 and "false alarms ['detect:plain'" in out
    monkeypatch.setattr(small_bench, "BOUND", 0.05)
    code, out = run(lambda im: time.sleep(0.1) or im is cases[0][2])
    assert code == 1 and "verdict: FAIL" in out and "false alarms []" in out


if __name__ == "__main__":
    test_same_size_no_markers()
    test_found_without_border_any_orientation()
    test_survives_simulated_print_and_photo()
    print("ok")
