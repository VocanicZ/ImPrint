"""Run: .venv/bin/python test_imprint_cam.py  (also works under pytest)"""
import binascii
import glob
import json
import sys

import cv2
import numpy as np
import pytest
from reedsolo import ReedSolomonError, RSCodec

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
    assert ic.decode(ic.to_rgb(lab), max_seeds=2)["found"] == [] and len(set(calls)) >= 10   # many lengths tried


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
    monkeypatch.setattr(ic, "track", lambda lum, seed, sgn: tracked.append(seed[1][0]) or
                        (lambda q, c=seed[1]: c + 10 * q, (-5.0, 5.0, -5.0, 5.0), 9 if sgn > 0 else 8))

    def refine(lum, m):
        x = m[0](np.zeros((1, 2)))[0, 0]
        return m[0], m[1], 40 if x < 200 or (x > 400 and m[2] != 8) else 5
    monkeypatch.setattr(ic, "refine", refine)
    monkeypatch.setattr(ic, "read", lambda lum, m: None)
    r = ic.decode(np.zeros((100, 600, 3), np.uint8))
    assert r["tried"] == 3 and tracked == [40, 40, 300, 300, 500, 500] and r["found"] == []


def test_decode_turns_a_lone_locked_twin_around(monkeypatch):
    """Only the wrong 180-degree twin locks on; the right one tracks badly. decode refits the
    locked lattice turned around, and that reads."""
    c = np.array([100.0, 50.0])
    monkeypatch.setattr(ic, "find_seeds", lambda lum: [(9.0, c, np.array([10.0, 0]), np.array([0, 10.0]))])
    monkeypatch.setattr(ic, "track", lambda lum, seed, sgn: (lambda q: c + 10 * q if sgn > 0 else c + 3 - 10 * q,
                                                             (-5.0, 5.0, -5.0, 5.0), 9))
    monkeypatch.setattr(ic, "refine", lambda lum, m: (m[0], m[1], 5 if m[0](np.zeros((1, 2)))[0, 0] != c[0] else 40))
    right = lambda m: np.allclose(m[0](np.array([[0.0, 0.0], [1.0, 0.0]])), [c, c - [10, 0]])
    monkeypatch.setattr(ic, "read", lambda lum, m: dict(payload=b"twin") if right(m) else None)
    found = ic.decode(np.zeros((100, 400, 3), np.uint8))["found"]
    assert [f["payload"] for f in found] == [b"twin"]
    assert np.allclose(found[0]["quad"], [[150, 100], [50, 100], [50, 0], [150, 0]])


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
    assert rep["config"]["trials"] == 2 and rep["config"]["delta"] == 2.2 and rep["verdict"] == "FAIL"
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
