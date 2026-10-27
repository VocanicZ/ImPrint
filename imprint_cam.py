"""Camera-robust ImPrint: invisible and borderless.

The encoded image IS the original image (same size, no frame, no markers) plus a faint lightness
texture made of cells, each a KxK grid of sub-blocks. On top of the cells:

* the PILOT (hidden sync, the invisible stand-in for a QR code's finder patterns): four faint
  cosines locked to the cell grid (PILOT_FREQS, cycles per cell). Each is one sharp spectral line,
  so the decoder finds the grid - spacing, angle, mirror, perspective - anywhere in a photo from
  the spectrum, with no border or picture outline. The encoder first notches the image's own
  content at those frequencies so busy texture can't drown them. Matched-filter phase gives
  the exact cell offset; the decoder then tracks the pilot outward probe by probe (each from its
  neighbours' local grid, re-aimed when a tilt smears it) and fits perspective + curl.
* one DATA bit per cell, carried by the cell's correlation with its own +-1 sub-block pattern,
  quantized onto a lattice (QIM): the bit is the sign near zero and flips every QIM_STEP pilot
  amplitudes beyond, so whatever the host texture already holds there, the encoder only nudges it
  to the nearest point that reads right - no cell is ever too busy to carry its bit. Bits come in
  tiles of T x T cells repeated over the whole picture, so any big-enough piece decodes
  (crop-robust); the decoder tries every tile phase and both 180-degree twins, then a rate-1/2
  LDPC code, decoded by belief propagation on the soft cell values of every repeat, and CRC-32
  confirm.

Patterns are orthogonal to flat color and linear ramps (lighting, white balance, printer drift and
smooth gradients cancel) and only touch L*, so grayscale prints work. Strength follows a perceptual
visibility threshold (luminance adaptation, the CSF at the printed pattern's frequency, contrast
masking) and gamut headroom: flat areas get little, clipped areas (pure black/white, saturated) get
nothing.
"""
from __future__ import annotations

import binascii
import functools
import heapq

import cv2
import numpy as np

K = 4                       # sub-blocks per cell side
SEED = 0x1A9
MAX_BYTES = 32
PILOT = 1.0                 # pilot (sync) amplitude relative to the data strength, shared by the
                            # PILOT_FREQS lines: four lines make each one 3/4 as strong as three did
                            # (0.85 or less loses print sync: kodim05 `easy` drops below the base)
SUBBLOCK_MM = 0.8           # target printed sub-block size
VIEW_MM = 350.0             # viewing distance the perceptual mask assumes (a print at arm's length)
JND_FLAT = 0.2              # printed grating's flat-area threshold, x Chou-Li's display JND
JND_REF = 1.0               # visibility threshold (L*) that gets 1x strength
MASK_LO, MASK_HI = 0.6, 3.2 # flat floor: a QIM cell on a flat host costs only its margin
PILOT_HI = 1.5              # pilot's mask ceiling: it sets the data lattice unit, so more costs energy
QIM_STEP = 2.8              # data lattice: a cell's bit flips every QIM_STEP pilot amplitudes
DATA_GAIN = 0.85            # decoder's data correlation per lattice unit (its sub-block means see a
                            # little less of the data window than the encoder's)
RAMP = np.arange(K) - (K - 1) / 2


# ---------------------------------------------------------------- color / metrics

def to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)


def to_rgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB) * 255 + 0.5, 0, 255).astype(np.uint8)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    x, y = (cv2.cvtColor(i, cv2.COLOR_RGB2GRAY).astype(np.float64) for i in (a, b))
    mu = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mx, my = mu(x), mu(y)
    sx, sy, sxy = mu(x * x) - mx ** 2, mu(y * y) - my ** 2, mu(x * y) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float((((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sx + sy + c2))).mean())


def visibility(orig: np.ndarray, enc: np.ndarray) -> dict:
    d = np.abs(to_lab(orig)[..., 0] - to_lab(enc)[..., 0])
    return dict(ssim=round(ssim(orig, enc), 4), psnr=round(float(cv2.PSNR(orig, enc)), 1),
                mean_dL=round(float(d.mean()), 2), p99_dL=round(float(np.percentile(d, 99)), 2))


# ---------------------------------------------------------------- patterns

# Pilot: four faint cosines locked to the cell grid, frequencies in cycles per cell (x, y).
# Their 2x2 determinants (10, -11, 7, -3, ...) share no factor, so together they repeat exactly
# once per cell (unique cell alignment), and the set has no rotation/mirror symmetry beyond the
# 180-degree turn no cosine can see (unique orientation). Each is one sharp spectral line: easy to
# find under any image texture. The 4th line, (2, 3), makes the alignment robust: the first three
# alone nearly repeat at a wrong offset (all three phases within 0.1 cycle of the truth there:
# the matched filter scores it 0.97 of the true peak), so a faint pilot under busy texture
# locked onto it; with (2, 3) the best wrong offset scores <= PILOT_ALIAS_MAX. It stays off the
# spectrum's axes, where natural images keep most of their texture.
PILOT_FREQS = np.array([[3.0, 1.0], [-1.0, 3.0], [2.0, -3.0], [2.0, 3.0]])
PILOT_PHASES = np.array([0.3, 1.7, 4.1, 2.8])       # 4th phase: lowest peak of the summed field
PILOT_ALIAS_MAX = 0.85
SYNC_MIN_SIDE = 320         # smallest image/level side scanned with 256-px spectral windows
SNR_MIN = 2.0               # a pilot probe counts when its lines stand this far above their
                            # neighbours (a 4-line fit rejects wrong offsets, so a lower bar is safe)


def pilot_field(h: int, w: int, c: float, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """Pilot over an h x w px area whose top-left pixel sits at (x0, y0) px of the cell grid (cell = c px)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx, yy = (xx + 0.5 + x0) / c, (yy + 0.5 + y0) / c
    out = sum(np.cos(2 * np.pi * (fx * xx + fy * yy) + ph) for (fx, fy), ph in zip(PILOT_FREQS, PILOT_PHASES))
    return (out / len(PILOT_FREQS)).astype(np.float32)


def notch_pilot(lum: np.ndarray, c: float, sigma_cells: float = 1.0) -> np.ndarray:
    """Remove the image's own content at the pilot frequencies (a notch ~1 cell wide in
    frequency), so after embedding those lines carry only the pilot: the decoder then finds it
    even under busy texture. Taking out a sliver of texture at exact frequencies is imperceptible."""
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = (xx + 0.5) / c, (yy + 0.5) / c
    out = lum.astype(np.float64).copy()
    for fx, fy in PILOT_FREQS:
        e = np.exp(-2j * np.pi * (fx * u + fy * v))
        z = lum * e
        blur = lambda a: cv2.GaussianBlur(a, (0, 0), sigma_cells * c)
        a = blur(z.real) + 1j * blur(z.imag)                 # local amplitude of the host at this line
        out -= 2 * np.real(a * np.conj(e))
    return out.astype(np.float32)


def _ramp_free(p: np.ndarray) -> bool:
    return (p * RAMP[:, None]).sum() == 0 and (p * RAMP[None, :]).sum() == 0


@functools.cache
def _all_patterns(seed: int) -> np.ndarray:
    """Data patterns: balanced, ramp-free +-1 KxK. Pattern i depends only on (seed, i)."""
    rng = np.random.default_rng(seed)
    base = np.array([1] * (K * K // 2) + [-1] * (K * K // 2), np.float32)
    out = []
    while len(out) < tile_size(MAX_BYTES) ** 2:
        p = rng.permutation(base).reshape(K, K)
        if _ramp_free(p):
            out.append(p)
    return np.stack(out)


def pats(n: int, seed: int = SEED) -> np.ndarray:
    return _all_patterns(seed)[:n]


def window(b: int) -> np.ndarray:
    w = np.sin(np.pi * (np.arange(b) + 0.5) / b)   # soft sub-block edges
    return np.outer(w, w).astype(np.float32)


def inner(b: int) -> slice:
    """Central half of a sub-block: what the decoder averages (tolerates blur/misalignment)."""
    return slice(b // 4, b - b // 4)


def gain(b: int) -> float:
    return float(window(b)[inner(b), inner(b)].mean())


def qim_soft(u: np.ndarray, step: float = QIM_STEP) -> np.ndarray:
    """Soft bit of a lattice coordinate: positive on (0, step), negative on (step, 2 step), period
    2 step; a triangle peaking at +-step/2, so |value| is the distance to the nearest flip."""
    return step / 2 - np.abs(np.mod(u - step / 2 + step, 2 * step) - step)


def qim_target(u: np.ndarray, bits: np.ndarray, margin: np.ndarray, step: float = QIM_STEP) -> np.ndarray:
    """Nearest coordinate to u whose soft bit has the sign of bits and is at least margin (capped
    at step/2, the most a lattice cell holds): the least change that makes the cell read right."""
    half = np.maximum(step / 2 - margin, 0.0)                    # half-width of each right zone
    centre = bits * step / 2
    k = np.round((u - centre) / (2 * step))
    best = None
    for dk in (-1, 0, 1):
        c = centre + (k + dk) * 2 * step
        t = np.clip(u, c - half, c + half)
        best = t if best is None else np.where(np.abs(t - u) < np.abs(best - u), t, best)
    return best


def pilot_amp(chan: np.ndarray, rows: int, cols: int, c: int) -> np.ndarray:
    """Pilot amplitude per cell (matched to the pilot's own cell shape). chan: (rows*c, cols*c)."""
    tpl = _pilot_cell(c)
    return (_cells(chan, c).reshape(rows * cols, c, c) * tpl).sum(axis=(1, 2)) / (np.linalg.norm(tpl) ** 2)


def subblock_means(chan: np.ndarray, rows: int, cols: int, b: int) -> np.ndarray:
    """(rows*cols, K, K) inner means of every sub-block. chan: (rows*K*b, cols*K*b)."""
    s = inner(b)
    m = chan.reshape(rows, K, b, cols, K, b)[:, :, s, :, :, s].mean(axis=(2, 5))
    return m.transpose(0, 2, 1, 3).reshape(rows * cols, K, K)


def measure(chan: np.ndarray, rows: int, cols: int, b: int, p: np.ndarray) -> np.ndarray:
    """Per-cell correlation with its pattern."""
    return (subblock_means(chan, rows, cols, b) * p).mean(axis=(1, 2))


def field(amps: np.ndarray, rows: int, cols: int, b: int, p: np.ndarray) -> np.ndarray:
    """Per-pixel offset map: amps[i] * pattern_i, shaped by the sub-block window."""
    f = (amps[:, None, None] * p).reshape(rows, cols, K, K).transpose(0, 2, 1, 3)
    f = f[:, :, None, :, :, None] * window(b)[None, None, :, None, None, :]
    return f.reshape(rows * K * b, cols * K * b)


def _lstar(gray: np.ndarray) -> np.ndarray:
    """sRGB gray level (0..255) -> L*."""
    v = np.clip(gray, 0, 255) / 255
    y = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return np.where(y > 0.008856, 116 * np.cbrt(y) - 16, 903.3 * y)


def _gray(lum: np.ndarray) -> np.ndarray:
    """L* -> sRGB gray level (0..255)."""
    y = np.where(lum > 8, ((lum + 16) / 116) ** 3, lum / 903.3)
    return 255 * np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.cbrt(y) ** (3 / 2.4) - 0.055)


def csf(f: float) -> float:
    """Mannos-Sakrison contrast sensitivity at f cycles/degree (peak ~1 near 8 cpd)."""
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f) ** 1.1)


def jnd(lum: np.ndarray, b: int) -> np.ndarray:
    """Per-pixel visibility threshold (L*) of the b-px sub-block pattern, printed at SUBBLOCK_MM and
    seen from VIEW_MM:
    * luminance adaptation: Chou & Li's (1995) threshold on the local background gray (dark and
      bright backgrounds hide more), converted to L*;
    * the CSF at the pattern's own frequency (period two sub-blocks) scales that base threshold:
      what the eye needs to see THIS grating, not a flat step;
    * contrast masking: the image's own band at the pattern's scale (sub-block to two sub-blocks)
      raises the threshold in proportion to its local RMS once it exceeds the base (Legge & Foley
      1980, exponent 1).
    Pooled over a quarter sub-block, so smooth pixels next to texture don't borrow its masking."""
    lum = lum.astype(np.float32)
    g = _gray(cv2.GaussianBlur(lum, (0, 0), b))
    t_gray = np.where(g <= 127, 17 * (1 - np.sqrt(np.maximum(g, 0) / 127)) + 3, 3 / 128 * (g - 127) + 3)
    f = VIEW_MM * np.pi / 180 / (2 * SUBBLOCK_MM)              # cycles/degree of the data pattern
    t0 = JND_FLAT * float(csf(8.0) / csf(f)) * (_lstar(g + t_gray / 2) - _lstar(g - t_gray / 2))
    band = cv2.GaussianBlur(lum, (0, 0), b / 4) - cv2.GaussianBlur(lum, (0, 0), b / 2)
    contrast = np.sqrt(cv2.GaussianBlur(band * band, (0, 0), b / 4))
    return np.maximum(t0, contrast).astype(np.float32)


def mask(lum: np.ndarray, b: int) -> np.ndarray:
    """Per-pixel strength multiplier: the visibility threshold in units of JND_REF - high in real
    texture, low in flat areas. Pooled over a sub-block: print blur and the decoder's sub-block
    means wash out finer modulation, so it would cost energy and carry nothing."""
    return cv2.GaussianBlur(np.clip(jnd(lum, b) / JND_REF, MASK_LO, MASK_HI), (0, 0), b)


def headroom(lab: np.ndarray, amp: float | np.ndarray) -> np.ndarray:
    """Fraction (0..1) of a +-amp L* swing each pixel can take symmetrically before RGB clips.
    A one-sided swing (pure black can't go down) would just shift brightness: visible, no signal."""
    l0 = to_lab(to_rgb(lab))[..., 0]
    out = np.ones(lab.shape[:2], np.float32)
    for sign in (1, -1):
        moved = lab.copy()
        moved[..., 0] += sign * amp
        out = np.minimum(out, np.clip(sign * (to_lab(to_rgb(moved))[..., 0] - l0) / amp, 0, 1))
    return out


# ---------------------------------------------------------------- payload

LDPC_COL_W = 4              # parity checks per message bit
LDPC_ITERS = 60             # belief-propagation iterations before giving up
LDPC_ALPHA = 0.85           # normalized min-sum: scale on check messages (tames min-sum's overconfidence)


def frame_len(nbytes: int) -> int:
    """Bytes on the wire for a payload: payload + CRC-32, then a rate-1/2 LDPC code."""
    return 2 * (nbytes + 4)


def tile_size(nbytes: int) -> int:
    """Tile side T (cells): a T x T tile holds the whole codeword once."""
    return int(np.ceil(np.sqrt(8 * frame_len(nbytes))))


@functools.cache
def ldpc(k: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rate k/n LDPC code (irregular repeat-accumulate): message bits first, each in LDPC_COL_W of
    the n - k parity checks (rows balanced, no two bits sharing two checks: no 4-cycles),
    then n - k parity bits on a staircase (check i holds parity bits i-1 and i), so encoding is a
    running XOR. Depends only on (k, n). Returns the Tanner graph's edges as (check, bit) arrays,
    and the bits of each check as an (n - k, max degree) array padded with n."""
    m = n - k
    rng = np.random.default_rng(SEED + k * 1000 + n)
    deg = np.zeros(m, int)
    rows: list[set] = []
    for _ in range(k):
        mine: set = set()
        for _ in range(LDPC_COL_W):
            near = set().union(*(r for r in rows if r & mine))      # checks that would close a 4-cycle,
            near |= {i + s for i in mine for s in (-1, 1)}         # also with a staircase parity bit
            pool = [i for i in range(m) if i not in mine and i not in near]
            low = min(deg[pool])
            i = int(rng.choice([i for i in pool if deg[i] == low]))
            mine.add(i)
            deg[i] += 1
        rows.append(mine)
    chk = np.array([i for r in rows for i in sorted(r)] + list(range(m)) + list(range(1, m)))
    bit = np.array([j for j, r in enumerate(rows) for _ in r] + list(range(k, n)) + list(range(k, n - 1)))
    d = np.bincount(chk, minlength=m)
    slots = np.full((m, d.max()), n)
    slots[np.repeat(np.arange(m), d), np.concatenate([np.arange(x) for x in d])] = bit[np.argsort(chk, kind="stable")]
    return chk, bit, slots


def ldpc_encode(msg: np.ndarray, n: int) -> np.ndarray:
    """0/1 message bits -> 0/1 codeword (message, then parity)."""
    chk, bit, _ = ldpc(len(msg), n)
    s = np.bincount(chk[bit < len(msg)], msg[bit[bit < len(msg)]], n - len(msg)).astype(int) % 2
    return np.concatenate([msg, np.cumsum(s) % 2]).astype(np.uint8)


def ldpc_decode(soft: np.ndarray, k: int) -> np.ndarray | None:
    """Soft bits (positive = 1, any scale) -> 0/1 codeword by normalized min-sum belief propagation,
    or None if it doesn't converge to a codeword. Min-sum needs no noise estimate: the scale cancels."""
    n = len(soft)
    slots = ldpc(k, n)[2]
    rows, first = np.arange(len(slots)), np.arange(slots.shape[1])
    ch = np.append(-np.asarray(soft, np.float64), np.inf)       # LLR (positive = 0); padding: sure 0
    c2v = np.zeros(slots.shape)
    tot = ch
    for _ in range(LDPC_ITERS):
        hard = tot < 0
        if not (hard[slots].sum(1) % 2).any():                    # every parity check holds
            return hard[:n].astype(np.uint8)
        v2c = tot[slots] - c2v
        sg = np.where(v2c < 0, -1.0, 1.0)
        mag = np.abs(v2c)
        i1 = mag.argmin(1)
        m1 = mag[rows, i1]
        mag[rows, i1] = np.inf
        m2 = mag.min(1)                                           # each check: own min excluded
        c2v = np.where(first == i1[:, None], m2[:, None], m1[:, None]) * sg * sg.prod(1, keepdims=True)
        c2v *= LDPC_ALPHA
        tot = ch + np.bincount(slots.ravel(), c2v.ravel(), n + 1) * (np.arange(n + 1) < n)
    return None


def to_bits(payload: bytes) -> np.ndarray:
    msg = np.unpackbits(np.frombuffer(payload + binascii.crc32(payload).to_bytes(4, "big"), np.uint8))
    return ldpc_encode(msg, 8 * frame_len(len(payload))).astype(np.float32) * 2 - 1


def bit_map(n_cells: int, n_bits: int) -> np.ndarray:
    """Which bit each tile position carries (scattered, so local damage hits many bits lightly)."""
    perm = np.random.default_rng(SEED + 1).permutation(n_cells)
    idx = np.empty(n_cells, int)
    idx[perm] = np.arange(n_cells) % n_bits
    return idx


def read_soft(soft: np.ndarray, nbytes: int) -> bytes | None:
    """Soft bits -> LDPC belief propagation -> CRC-32."""
    cw = ldpc_decode(soft, 8 * (nbytes + 4))
    if cw is None:
        return None
    msg = np.packbits(cw[:8 * (nbytes + 4)]).tobytes()
    if binascii.crc32(msg[:nbytes]).to_bytes(4, "big") == msg[nbytes:]:
        return msg[:nbytes]
    return None


# ---------------------------------------------------------------- encode

def subblock_for_print(width_px: int, print_mm: float) -> int:
    """Sub-block size (image px) that prints at about SUBBLOCK_MM for this print width."""
    return max(4, int(round(width_px * SUBBLOCK_MM / print_mm)))


def encode(rgb: np.ndarray, payload: bytes, delta: float = 2.2, subblock_px: int = 8):
    """Embed payload into an RGB uint8 image. Returns (image of the same size, stats)."""
    nb = len(payload)
    if not 1 <= nb <= MAX_BYTES:
        raise ValueError(f"payload must be 1..{MAX_BYTES} bytes")
    H, W = rgb.shape[:2]
    b = int(subblock_px)
    c = K * b
    rows, cols, T = H // c, W // c, tile_size(nb)
    if rows < T or cols < T:
        raise ValueError(f"image too small for a {nb}-byte payload: needs at least {T * c} px per side "
                         f"(sub-block {b} px); use a shorter payload or a smaller sub-block")
    n, nbits = rows * cols, 8 * frame_len(nb)
    ii, jj = np.mgrid[0:rows, 0:cols]
    t = ((ii % T) * T + (jj % T)).ravel()
    D = pats(T * T)[t]
    s = to_bits(payload)[bit_map(T * T, nbits)[t]]

    lab = to_lab(rgb)
    region = lab[:rows * c, :cols * c]                         # whole cells only; view into lab
    lum = region[..., 0].copy()
    m = mask(lum, b)
    pm = np.minimum(m, PILOT_HI)
    room = headroom(region, (2.5 * m + PILOT * pm) * delta)   # the swing this pixel will get
    room = -cv2.dilate(-room, np.ones((b, b), np.uint8))      # erode: whole sub-block must have room
    pm = pm * room
    m = m * room
    # the pilot follows the same mask, but only up to PILOT_HI: it sets the lattice unit, and a
    # bigger unit makes every cell's QIM nudge bigger
    pil = PILOT * delta * pilot_field(rows * c, cols * c, c) * pm
    lum_n = notch_pilot(lum, c)
    notch = (lum_n - lum) * room                              # only where there's room to change
    h = measure(lum + notch + pil, rows, cols, b, D)          # pilot + notch are part of what data must beat
    g = measure(field(np.ones(n, np.float32), rows, cols, b, D) * m, rows, cols, b, D)
    dc = delta * g / gain(b)                                  # margin: delta where mask = 1
    # The lattice unit is the median pilot amplitude at PILOT = 1: the decoder measures the same
    # median, so a fainter pilot leaves the data lattice (and the data's robustness) unchanged.
    pa = pilot_amp(pil, rows, cols, c)
    unit = float(np.median(pa[pa > 0])) / PILOT if (pa > 0).any() else delta
    # Host-interference rejection (QIM): move each cell's correlation to the nearest lattice point
    # that reads its bit, so a busy host costs at most step/2 + margin, never a skipped cell.
    ok = g > 1e-6                                             # no room (clipped) -> leave the cell
    a = np.where(ok, (qim_target(h / unit, s, dc / unit) * unit - h) / np.where(ok, g, 1.0), 0.0)
    region[..., 0] += field(a.astype(np.float32), rows, cols, b, D) * m + pil + notch
    out = to_rgb(lab)

    got = s * qim_soft(measure(to_lab(out)[:rows * c, :cols * c, 0], rows, cols, b, D) / unit)
    stats = dict(rows=rows, cols=cols, tile=T, tiles=round(n / T ** 2, 1), subblock_px=b, bits=nbits,
                 wrong_cells=round(float((got <= 0).mean()), 4), **visibility(rgb, out))
    return out, stats


# ---------------------------------------------------------------- geometry helpers

def fit_map(lat: np.ndarray, photo: np.ndarray, cubic: bool):
    """Lattice coords (cells) -> photo px: homography, plus a cubic residual for curl."""
    lat, photo = np.asarray(lat, np.float64), np.asarray(photo, np.float64)
    H, _ = cv2.findHomography(lat, photo)
    if H is None:
        return None

    def proj(c):
        p = np.c_[c, np.ones(len(c))] @ H.T
        return p[:, :2] / p[:, 2:]

    if not cubic:
        return proj
    lo, span = lat.min(0), np.ptp(lat, 0) + 1e-9

    def basis(c):
        u, v = (2 * (c - lo) / span - 1).T
        return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3], 1)

    A = basis(lat)
    coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(10), A.T @ (photo - proj(lat)))
    return lambda c: proj(c) + basis(c) @ coef


def resample(img: np.ndarray, fmap, x0: float, y0: float, ncols: int, nrows: int, S: int, step: int = 8):
    """Lattice rect [x0, x0+ncols) x [y0, y0+nrows) cells -> image at S px per cell."""
    W, H = ncols * S, nrows * S
    gx, gy = np.arange(0, W + step, step), np.arange(0, H + step, step)
    cx, cy = np.meshgrid(x0 + (gx + 0.5) / S, y0 + (gy + 0.5) / S)
    p = fmap(np.c_[cx.ravel(), cy.ravel()]).reshape(len(gy), len(gx), 2).astype(np.float32)
    ix, iy = np.meshgrid(np.arange(W, dtype=np.float32) / step, np.arange(H, dtype=np.float32) / step)
    m = cv2.remap(p, ix, iy, cv2.INTER_LINEAR)
    return cv2.remap(img, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


@functools.cache
def _hann(n: int) -> np.ndarray:
    w = np.hanning(n).astype(np.float32)
    return np.outer(w, w)


def _subpixel(R: np.ndarray, y: int, x: int) -> tuple[float, float]:
    def off(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d >= 0 else 0.5 * (a - c) / d
    l = np.log(R[y - 1:y + 2, x - 1:x + 2] + 1e-9)
    return y + off(l[0, 1], l[1, 1], l[2, 1]), x + off(l[1, 0], l[1, 1], l[1, 2])


def _whitened_spectrum(region: np.ndarray, n: int = 256) -> np.ndarray:
    """Welch-averaged power spectrum of a region of n..2n px per side (3x3 overlapping n-windows,
    spread over it), divided by its local geometric mean: image texture -> ~1, random spikes
    average down, pilot lines stay."""
    P = 0
    for y in np.linspace(0, region.shape[0] - n, 3).astype(int):
        for x in np.linspace(0, region.shape[1] - n, 3).astype(int):
            w = region[y:y + n, x:x + n]
            P = P + np.abs(np.fft.fft2((w - w.mean()) * _hann(n))) ** 2
    P = np.fft.fftshift(P).astype(np.float32)
    return P / np.exp(cv2.blur(np.log(P + 1e-9), (9, 9)))


@functools.cache
def _ring_pairs(r: int = 3) -> tuple:
    """Half of the ring of bins at Chebyshev distance r: each with its mirror makes one pair."""
    return tuple((dx, dy) for dx in range(-r, r + 1) for dy in range(0, r + 1)
                 if max(abs(dx), dy) == r and (dy > 0 or dx > 0))


def _deridge(R: np.ndarray) -> np.ndarray:
    """Divide out ridges. A straight edge in the picture (page border, clutter, a horizon) is a
    streak through the spectrum, and a streak has strong local maxima all along it that can pass
    for pilot lines. A pilot line is an isolated spike: across it, both sides drop to the
    texture level. The ridge level at a bin is the best min(R(p + d), R(p - d)) over opposite
    ring bins 3 away: ~1 beside a spike, the streak's own level on a streak."""
    ridge = np.zeros_like(R)
    for dx, dy in _ring_pairs():
        ridge = np.maximum(ridge, np.minimum(np.roll(R, (dy, dx), (0, 1)), np.roll(R, (-dy, -dx), (0, 1))))
    return R / np.maximum(ridge, 1.0)


def _plausible(A: np.ndarray) -> bool:
    """A photographed cell grid: perspective skews it, but cells stay roughly square."""
    l1, l2 = np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1])
    cos = abs(A[:, 0] @ A[:, 1]) / (l1 * l2 + 1e-9)
    return 0.5 < l1 / l2 < 2 and cos < 0.65


def sync_in_region(region: np.ndarray, n: int = 256, top: int = 10):
    """Find the pilot lines in a region's spectrum and solve the cell grid: any two strong peaks
    fix a candidate grid, the other lines must show up where it predicts them.
    Returns (score, A) with A: cell coords -> px offsets (columns = cell vectors), or None."""
    R = _deridge(_whitened_spectrum(region, n))
    c = n // 2
    loc = (R == cv2.dilate(R, np.ones((3, 3), np.uint8))) & (R > 3)
    yy, xx = np.nonzero(loc)
    fy, fx = yy - c, xx - c
    rad = np.hypot(fx, fy)
    sel = (rad >= 5) & (rad <= n / 2.3) & ((fy > 0) | ((fy == 0) & (fx > 0)))
    if sel.sum() < 2:
        return None
    order = np.argsort(-R[yy[sel], xx[sel]])[:top]
    pk = [_subpixel(R, yy[sel][i], xx[sel][i]) for i in order]
    obs = np.array([[(x - c) / n, (y - c) / n] for y, x in pk])
    obs = np.concatenate([obs, -obs])                              # both signs of every line
    logR = np.log(np.concatenate([R[yy[sel][order], xx[sel][order]]] * 2))

    def r_at(f):                                                   # whitened power at a frequency
        x, y = c + f[0] * n, c + f[1] * n
        xi, yi = int(round(x)), int(round(y))
        if not (1 <= xi < n - 1 and 1 <= yi < n - 1):
            return 1e-3
        return float(R[yi - 1:yi + 2, xi - 1:xi + 2].max())

    best, nf = None, len(PILOT_FREQS)
    for a in range(len(obs)):
        for b in range(len(obs)):
            if a == b or a % (len(obs) // 2) == b % (len(obs) // 2):
                continue
            for i in range(nf):
                for j in range(nf):
                    if i == j:
                        continue
                    F2 = PILOT_FREQS[[i, j]]
                    Mt = np.linalg.solve(F2, obs[[a, b]])            # obs = F @ Mt
                    A = np.linalg.inv(Mt)
                    cell = np.sqrt(abs(np.linalg.det(A)))
                    if not 12 <= cell <= n / 3 or not _plausible(A):
                        continue
                    rest = [k for k in range(nf) if k not in (i, j)]
                    pred = PILOT_FREQS[rest] @ Mt
                    lr = np.log([r_at(f) for f in pred])
                    if lr.mean() < np.log(4):                         # the other lines, on average
                        continue
                    score = float(logR[a] + logR[b] + lr.sum())
                    if best is None or score > best[0]:
                        best = (score, Mt, (a, b, i, j, rest, pred, lr))
    if best is None:
        return None
    # refine with every line that shows (sub-pixel positions of the predicted ones)
    score, Mt, (a, b, i, j, rest, pred, lr) = best
    fo, used = [obs[a], obs[b]], [i, j]
    for k, f, l in zip(rest, pred, lr):
        if l >= np.log(2):
            yk, xk = _subpixel(R, int(round(c + f[1] * n)), int(round(c + f[0] * n)))
            fo.append([(xk - c) / n, (yk - c) / n])
            used.append(k)
    Mt, *_ = np.linalg.lstsq(PILOT_FREQS[used], np.array(fo), rcond=None)
    return score, np.linalg.inv(Mt)


def _starts(size: int, span: int, step: int) -> list[int]:
    """Offsets of span-long windows every step px over size px, the last one flush with the end."""
    out = list(range(0, size - span + 1, step))
    return out if out[-1] == size - span else out + [size - span]


def find_seeds(lum: np.ndarray) -> list:
    """Scan the photo (all pyramid levels) for the pilot in regions of up to 2n x 2n px, n-px
    spectral windows. A line's power over the texture grows with the window's area, so n = 256
    wherever it fits (n = 128 only on small images/levels). Returns [(score, centre_xy, e1, e2)]
    in full-res px, best first."""
    seeds, img, f = [], lum.astype(np.float32), 1
    while min(img.shape) >= 256:
        n = 256 if min(img.shape) >= SYNC_MIN_SIDE else 128
        h, w = min(2 * n, img.shape[0]), min(2 * n, img.shape[1])
        for y in _starts(img.shape[0], h, n // 2):
            for x in _starts(img.shape[1], w, n // 2):
                r = sync_in_region(img[y:y + h, x:x + w], n)
                if r:
                    seeds.append((r[0], np.array([x + w / 2, y + h / 2]) * f, r[1][:, 0] * f, r[1][:, 1] * f))
        img, f = cv2.pyrDown(img), f * 2
    # the real code yields many nearby seeds with the same grid; clutter spikes are loners
    out = []
    for sc, c, e1, e2 in seeds:
        A = np.c_[e1, e2]
        agree = sum(1 for _, c2, f1, f2 in seeds
                    if np.linalg.norm(c2 - c) < 8 * np.linalg.norm(e1)
                    and np.linalg.norm(np.c_[f1, f2] - A) < 0.05 * np.linalg.norm(A)) - 1
        out.append((sc + 2 * np.log1p(agree), c, e1, e2))
    return sorted(out, key=lambda s: -s[0])


@functools.cache
def _pilot_cell(S: int) -> np.ndarray:
    t = pilot_field(S, S, S)
    return t - t.mean()


def _cells(arr: np.ndarray, S: int) -> np.ndarray:
    nr, nc = arr.shape[0] // S, arr.shape[1] // S
    return arr[:nr * S, :nc * S].reshape(nr, S, nc, S).transpose(0, 2, 1, 3)


@functools.cache
def _pilot_basis(nc: int, S: int) -> np.ndarray:
    """exp(-2 pi i f.u) over an nc x nc-cell patch at S px/cell, one row per pilot frequency."""
    u = (np.arange(nc * S) + 0.5) / S
    U, V = np.meshgrid(u, u)
    return np.stack([np.exp(-2j * np.pi * (fx * U + fy * V)).ravel() for fx, fy in PILOT_FREQS])


@functools.cache
def _offset_grid(n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    d = np.arange(n) / n
    DX, DY = np.meshgrid(d, d)
    off = np.stack([DX.ravel(), DY.ravel()], 1)
    return off, 2 * np.pi * off @ PILOT_FREQS.T                 # (n*n, 2), phase per candidate & freq


def pilot_phase(patch: np.ndarray, nc: int, S: int):
    """Matched filter at the pilot frequencies only (data and image live elsewhere).
    Returns (offset in cells (dx, dy): true lattice = sampled + offset, coherence 0..1, snr)."""
    x = (patch - patch.mean()).ravel()
    cf = _pilot_basis(nc, S) @ x                                 # complex coefficient per pilot line
    off, ph = _offset_grid()
    score = (np.abs(cf) * np.cos(np.angle(cf)[None, :] - PILOT_PHASES[None, :] - ph)).sum(1)
    i = int(np.argmax(score))
    # noise reference: coefficients one patch-resolution step beside each line (orthogonal to
    # the pilot over the patch, and inside the encoder's notch, so image texture is gone there too)
    ref = np.abs(_ref_basis(nc, S) @ x)
    return off[i], float(score[i] / (np.abs(cf).sum() + 1e-12)), float(np.abs(cf).mean() / (ref.mean() + 1e-12))


@functools.cache
def _ref_basis(nc: int, S: int) -> np.ndarray:
    u = (np.arange(nc * S) + 0.5) / S
    U, V = np.meshgrid(u, u)
    step = 1 / nc
    rows = [np.exp(-2j * np.pi * ((fx + dx) * U + (fy + dy) * V)).ravel()
            for fx, fy in PILOT_FREQS for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step))]
    return np.stack(rows)


@functools.cache
def _zoom_basis(nc: int, S: int, span: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """1-D DFT rows at each pilot frequency +- span cycles/cell (n steps), for x and for y."""
    u = (np.arange(nc * S) + 0.5) / S
    d = np.linspace(-span, span, n)
    ex = np.exp(-2j * np.pi * (PILOT_FREQS[:, 0, None, None] + d[None, :, None]) * u)   # (lines, n, px)
    ey = np.exp(-2j * np.pi * (PILOT_FREQS[:, 1, None, None] + d[None, :, None]) * u)
    return ex.astype(np.complex64), ey.astype(np.complex64)


@functools.cache
def _warp_grid(span: float, n: int, lim: float = 0.12, steps: int = 13) -> tuple[np.ndarray, np.ndarray]:
    """Candidate local warps M = I + E (all four entries of E on a grid), and where each puts
    every pilot line on the zoomed DFT grid: (cand, 2, 2), (cand, line, 2) as (y, x) indices."""
    e = np.linspace(-lim, lim, steps)
    M = np.eye(2) + np.stack(np.meshgrid(e, e, e, e, indexing="ij"), -1).reshape(-1, 2, 2)
    d = (M @ PILOT_FREQS.T).transpose(0, 2, 1) - PILOT_FREQS     # frequency shift of every line
    idx = np.rint((d + span) / (2 * span) * (n - 1)).astype(int)[..., ::-1]
    assert idx.min() >= 0 and idx.max() < n, "warp grid exceeds the zoomed DFT span"
    return M, idx


WARP_COARSE = (0.12, 13, 0.75, 49)   # pilot_warp search: +-lim, steps per entry, DFT +-span, DFT steps
WARP_FINE = (0.025, 11, 0.15, 61)


def pilot_warp(patch: np.ndarray, nc: int, S: int, grid: tuple = WARP_COARSE) -> np.ndarray | None:
    """Where the pilot lines really are in a resampled patch: a map whose local scale/turn/shear is
    off by a few % moves every line off its nominal frequency f, to M f, and smears the patch's
    phase. Scores each candidate M by the summed amplitude it finds on all lines at once (a zoomed
    DFT around each line, looked up), so one consistent warp must explain every line: noise can
    fake one line's peak, not four. The true lattice around the patch centre is M^T times the
    sampled one. Returns M, or None if the best is on the search's rim."""
    lim, steps, span, n = grid
    x = (patch - patch.mean()).astype(np.complex64)
    ex, ey = _zoom_basis(nc, S, span, n)
    R = np.abs(np.stack([ey[k] @ x @ ex[k].T for k in range(len(PILOT_FREQS))]))   # (line, dy, dx)
    M, idx = _warp_grid(span, n, lim, steps)
    score = R[np.arange(len(PILOT_FREQS)), idx[..., 0], idx[..., 1]].sum(1)
    best = int(np.argmax(score))
    if np.abs(M[best] - np.eye(2)).max() >= lim - 1e-9:
        return None
    return M[best]


def _pilot_shift(mc: np.ndarray, tpl: np.ndarray):
    """Circular cross-correlation of a mean cell with the pilot cell: (shift_yx, ncc)."""
    mc = mc - mc.mean()
    C = np.fft.ifft2(np.fft.fft2(mc) * np.conj(np.fft.fft2(tpl))).real
    dy, dx = np.unravel_index(np.argmax(C), C.shape)
    return np.array([dy, dx]), float(C[dy, dx] / (np.linalg.norm(mc) * np.linalg.norm(tpl) + 1e-12))


# ---------------------------------------------------------------- decode

def track(lum: np.ndarray, seed, sgn: int = 1, r: int = 6, nc: int = 8, step: int = 2,
          coh_min: float = 0.7, snr_min: float = SNR_MIN):
    """From one seed window, lock onto the pilot phase (sub-cell offset) and grow outward over the
    picture probe by probe, `step` cells apart, strongest first; then fit a lattice->photo map
    (perspective + curl) to every probe. Lattice coords = true cell coords up to a whole-cell
    shift. sgn picks one of the two 180-degree twins that cosine lines can't tell apart (the caller
    tries both; the CRC decides).
    No single affine map reaches far under a strong tilt (its scale is off a few % some 8 cells
    out, enough to smear an 8-cell probe's phase), so every probe carries its own local map: its
    lattice point, its photo px and its local cell vectors, refit from the probes around it. A
    neighbour is predicted one step from there; a probe still smeared is re-aimed (pilot_warp).
    Returns (fmap, extent, n_probes) or None."""
    _, c0, e1, e2 = seed
    S = K * r
    cell = float(np.sqrt(abs(e1[0] * e2[1] - e1[1] * e2[0])))

    def probe(q, Q, P, J):
        """Probe the patch centred on lattice point q of the local map x -> P + (x - Q) J^T.
        Returns (true lattice point, its photo px, local J, snr) or None."""
        g = lambda x: P + (x - Q) @ J.T
        patch = resample(lum, g, q[0] - nc / 2, q[1] - nc / 2, nc, nc, S)
        off, coh, snr = pilot_phase(patch, nc, S)
        for grid in (WARP_COARSE, WARP_FINE):                     # smeared: re-aim at where the lines are
            if coh >= coh_min and snr >= snr_min:
                break
            M = pilot_warp(patch, nc, S, grid)
            if M is None:
                return None
            J = J @ np.linalg.inv(M).T
            g = lambda x: P + (x - Q) @ J.T
            patch = resample(lum, g, q[0] - nc / 2, q[1] - nc / 2, nc, nc, S)
            off, coh, snr = pilot_phase(patch, nc, S)
        if coh < coh_min or snr < snr_min:
            return None
        return q + (off + 0.5) % 1 - 0.5, g(q[None])[0], J, snr

    first = probe(np.zeros(2), np.zeros(2), np.asarray(c0, np.float64), sgn * np.c_[e1, e2])
    if first is None:
        return None
    nodes = {(0, 0): first}
    heap, done = [], {(0, 0)}

    def push(key):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = (key[0] + di, key[1] + dj)
            if k not in done:
                heapq.heappush(heap, (-nodes[key][3], k, key))

    push((0, 0))
    while heap and len(done) < 4000:
        _, key, parent = heapq.heappop(heap)
        if key in done:
            continue
        Q, P, J, _ = nodes[parent]
        q = np.array(key, np.float64) * step
        p = P + (q - Q) @ J.T
        if not (0 <= p[0] < lum.shape[1] and 0 <= p[1] < lum.shape[0]):
            done.add(key)
            continue
        res = probe(q, Q, P, J)
        done.add(key)
        if res is None:
            continue
        Qn, Pn, Jn, snr = res
        near = [nodes[k] for k in ((key[0] + di, key[1] + dj) for di in range(-2, 3) for dj in range(-2, 3))
                if k in nodes]
        if len(near) >= 3:                                        # local cell vectors from the neighbours
            dQ = np.array([n[0] for n in near]) - Qn
            dP = np.array([n[1] for n in near]) - Pn
            if abs(np.linalg.det(dQ.T @ dQ)) > 1e-3 * step ** 4:
                Jn = np.linalg.lstsq(dQ, dP, rcond=None)[0].T
        nodes[key] = (Qn, Pn, Jn, snr)
        push(key)
    L_ = np.array([n[0] for n in nodes.values()])
    P_ = np.array([n[1] for n in nodes.values()])
    fmap = None
    if len(L_) >= 6:
        H, inl = cv2.findHomography(L_, P_, cv2.RANSAC, 0.5 * cell)
        keep = inl.ravel().astype(bool) if H is not None else np.ones(len(L_), bool)
        for _ in range(2):                                        # perspective inliers, then whatever curl fits
            fm = fit_map(L_[keep], P_[keep], cubic=keep.sum() >= 30) if keep.sum() >= 4 else None
            if fm is None:
                break
            fmap, keep = fm, np.linalg.norm(fm(L_) - P_, axis=1) < 0.3 * cell
        if fmap is not None:
            L_ = L_[keep] if keep.sum() >= 4 else L_
    elif len(L_) >= 4:
        fmap = fit_map(L_, P_, cubic=False)
    if fmap is None:                                              # seed only: its local map
        Q, P, J, _ = first
        fmap = lambda x: P + (x - Q) @ J.T
        L_ = L_[:1]
    ext = (L_[:, 0].min() - nc / 2, L_[:, 0].max() + nc / 2, L_[:, 1].min() - nc / 2, L_[:, 1].max() + nc / 2)
    return fmap, ext, len(L_)


def refine(lum: np.ndarray, model, passes: int = 3, **kw):
    """Tracking only needs enough probes to lock on; the data lattice needs every px right. Each
    pass re-measures the pilot on a dense grid of small windows (reaching `grow` cells past the
    area fitted so far) and refits the map to all of them; stop once a pass adds nothing."""
    for _ in range(passes):
        n0, model = model[2], _refine_pass(lum, model, **kw)
        if model[2] <= n0:
            break
    return model


def _refine_pass(lum: np.ndarray, model, r: int = 6, nc: int = 6, grow: float = 8,
                 coh_min: float = 0.7, snr_min: float = SNR_MIN):
    fmap, (x0, x1, y0, y1), nprobe = model
    S = K * r
    lat, pho = [], []
    h = nc // 2                       # windows start on whole cells: pilot phase is measured from there
    for cy in range(int(np.ceil(y0 - grow)) + h, int(y1 + grow) - h + 1, h):
        for cx in range(int(np.ceil(x0 - grow)) + h, int(x1 + grow) - h + 1, h):
            q = fmap(np.array([[cx, cy], [cx - h, cy - h], [cx + h, cy - h], [cx - h, cy + h], [cx + h, cy + h]]))
            if not ((q >= 0) & (q < [lum.shape[1], lum.shape[0]])).all():
                continue                                          # whole window in the photo: no edge bias
            p = q[0]
            off, coh, snr = pilot_phase(resample(lum, fmap, cx - h, cy - h, nc, nc, S), nc, S)
            if coh >= coh_min and snr >= snr_min:
                lat.append(np.array([cx, cy]) + (off + 0.5) % 1 - 0.5)   # the map is already close
                pho.append(p)
    if len(lat) < 8:
        return model
    L_, P_ = np.array(lat), np.array(pho)
    o, e1, e2 = fmap(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    (ax, ay), (bx, by) = e1 - o, e2 - o
    cell = np.sqrt(abs(ax * by - ay * bx))                        # photo px per cell
    H, inl = cv2.findHomography(L_, P_, cv2.RANSAC, 0.1 * cell)
    keep = inl.ravel().astype(bool) if H is not None else np.zeros(len(L_), bool)
    fm = fit_map(L_[keep], P_[keep], cubic=keep.sum() >= 30) if keep.sum() >= 8 else None
    if fm is None:                                                # refit failed: keep the tracked map
        return model
    L_ = L_[keep]
    return fm, (L_[:, 0].min() - nc / 2, L_[:, 0].max() + nc / 2, L_[:, 1].min() - nc / 2, L_[:, 1].max() + nc / 2), \
        int(keep.sum())


def cell_means(lum: np.ndarray, model, r: int = 6, grow: float = 8):
    """Rectify the tracked lattice, align cells on the pilot and remove it. The tracked extent grows
    by `grow` cells per side (probes near the photo edge or in busy texture fail, but the fitted map
    still holds there); cells that don't map wholly inside the photo are dropped.
    Returns (m (cells, K*K) pilot-free sub-block means, w per-cell trust, used, rows, cols, unit) or
    None; unit is the median pilot amplitude at PILOT = 1, the scale of the data lattice."""
    fmap, (x0, x1, y0, y1), nprobe = model
    x0, x1, y0, y1 = x0 - grow, x1 + grow, y0 - grow, y1 + grow
    S = K * r
    xs, ys = np.floor(x0), np.floor(y0)                           # lattice coords are true cell coords
    ncols, nrows = int(np.ceil(x1 - xs)), int(np.ceil(y1 - ys))
    if ncols < 4 or nrows < 4:
        return None
    arr = resample(lum, fmap, xs, ys, ncols, nrows, S)
    rows, cols = arr.shape[0] // S, arr.shape[1] // S
    tpl = _pilot_cell(S)

    # per-block fine alignment on the pilot (data-independent)
    block = 6
    nbc = -(-cols // block)
    bid = ((np.arange(rows) // block)[:, None] * nbc + (np.arange(cols) // block)[None, :]).ravel()
    best_m = best_e = best_p = None
    for dy in range(-(r // 2), r // 2 + 1):
        for dx in range(-(r // 2), r // 2 + 1):
            sh = cv2.warpAffine(arr, np.float32([[1, 0, dx], [0, 1, dy]]), (arr.shape[1], arr.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cells = _cells(sh, S).reshape(rows * cols, S, S)
            pc = (cells * tpl).sum(axis=(1, 2)) / (np.linalg.norm(tpl) ** 2)   # pilot amplitude per cell
            m = subblock_means(sh, rows, cols, r)
            e = np.bincount(bid, pc, bid.max() + 1)
            if best_m is None:
                best_m, best_e, best_p = m.copy(), e, pc.copy()
            else:
                better = (e > best_e)[bid]
                best_m[better], best_p[better] = m[better], pc[better]
                best_e = np.maximum(best_e, e)
    q = 0.25                                                      # margin: covers the fine shifts above
    cy, cx = np.mgrid[0:rows, 0:cols].reshape(2, -1) + np.array([ys, xs])[:, None]
    inside = np.ones(rows * cols, bool)
    for ox, oy in ((-q, -q), (1 + q, -q), (-q, 1 + q), (1 + q, 1 + q)):
        p = fmap(np.c_[cx + ox, cy + oy].astype(np.float64))
        inside &= (p[:, 0] >= 0) & (p[:, 0] < lum.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < lum.shape[0])
    best_p = np.where(inside, best_p, 0.0)
    pos = best_p[best_p > 0]
    med = np.median(pos) if len(pos) else 1.0
    # pilot strength = cell trust, averaged over neighbours: one cell's pilot reading is swamped by
    # busy texture, the pilot's local level is not
    blur = lambda a: cv2.GaussianBlur(a.reshape(rows, cols).astype(np.float32), (0, 0), 1.0).ravel()
    local = blur(best_p) / np.maximum(blur(inside.astype(np.float32)), 1e-3)
    w = np.where(inside, np.clip(local / (med + 1e-9), 0, 2), 0.0)
    m = best_m.reshape(rows * cols, K * K)
    used = w > 0.2
    if used.sum() < 16:
        return None
    m = m - np.average(m[used], axis=0, weights=w[used])          # remove the pilot (same in every cell)
    return m, w, used, rows, cols, float(med) / PILOT


def lattice_scale(c: np.ndarray, w: np.ndarray, used: np.ndarray, unit: float) -> float:
    """Decoder correlation per lattice unit. Starts from the pilot (unit * DATA_GAIN), then fits the lattice
    itself within +-40% (print blur and the camera treat data and pilot a bit differently): the
    scale at which the most cells sit clear of a bit flip."""
    s0 = unit * DATA_GAIN
    ks = np.arange(-6, 7)
    fit = [float((w[used] * np.clip(np.abs(qim_soft(c[used] / (s0 * 2 ** (k / 12)))) / 0.5, 0, 1)).sum())
           - 1e-6 * abs(k) for k in ks]                       # ties go to the pilot's estimate
    return s0 * 2 ** (ks[int(np.argmax(fit))] / 12)


def read(lum: np.ndarray, model, r: int = 6, grow: float = 8) -> dict | None:
    """Try every payload length and tile phase on the pilot-free cells: soft-combine repeats,
    LDPC belief propagation, CRC."""
    cm = cell_means(lum, model, r, grow)
    if cm is None:
        return None
    m, w, used, rows, cols, unit = cm
    nprobe = model[2]
    ii, jj = np.mgrid[0:rows, 0:cols]
    ii, jj = ii.ravel(), jj.ravel()
    for nb in range(1, MAX_BYTES + 1):
        T, nbits = tile_size(nb), 8 * frame_len(nb)
        if used.sum() < T * T:
            break
        D = pats(T * T).reshape(T * T, K * K)
        bm = bit_map(T * T, nbits)
        C = (m @ D.T) / (K * K)                                  # cell x tile-position correlation
        V = C * w[:, None]
        lim = np.median(np.abs(V[used])) + 1e-9               # soft limiter: no single cell outvotes its repeats
        V = np.clip(V, -lim, lim)
        cand = []
        for pi in range(T):
            for pj in range(T):
                t = ((ii + pi) % T) * T + ((jj + pj) % T)
                soft = np.bincount(bm[t], V[np.arange(len(t)), t], nbits)
                cand.append((float(np.abs(soft).sum()), soft, t))
        cand.sort(key=lambda c: -c[0])
        for _, soft, t in cand[:3]:
            c = C[np.arange(len(t)), t]
            q = qim_soft(c / lattice_scale(c, w, used, unit)) * w   # lattice soft bits (primary)
            for v in (q, V[np.arange(len(t)), t]):               # then plain sign + limiter
                soft = np.bincount(bm[t], v, nbits)
                payload = read_soft(soft, nb)
                if payload:
                    bits = to_bits(payload)
                    s = bits[bm[t]]
                    e = bits * soft / np.maximum(np.bincount(bm[t], minlength=nbits), 1)  # per-bit evidence
                    return dict(payload=payload, raw_ber=round(float((s[used] * v[used] <= 0).mean()), 4),
                                bit_ber=round(float((np.sign(soft) != bits).mean()), 4),
                                bit_snr=round(float(e.mean() / (e.std() + 1e-9)), 2),
                                cells=int(used.sum()), tiles=round(used.sum() / T ** 2, 1), probes=nprobe)
    return None


def _twin(model):
    """The same lattice turned 180 degrees (refit from scratch: n = 0)."""
    fmap, (x0, x1, y0, y1), _ = model
    return (lambda q: fmap(-q)), (-x1, -x0, -y1, -y0), 0


def _quad(model) -> np.ndarray:
    fmap, (x0, x1, y0, y1), _ = model
    return fmap(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)).astype(np.float32)


def decode(rgb: np.ndarray, max_seeds: int = 40) -> dict:
    """Find and decode every ImPrint in a photo or file: no border or outline needed."""
    lum = to_lab(rgb)[..., 0]
    seeds = find_seeds(lum)
    found, regions, tried = [], [], 0
    for seed in seeds:
        if tried >= max_seeds:
            break
        if any(cv2.pointPolygonTest(poly, tuple(map(float, seed[1])), False) >= 0 for poly in regions):
            continue                                                # inside a picture already read
        tried += 1
        res, locked = None, []
        for sgn in (1, -1):
            model = track(lum, seed, sgn)
            model = refine(lum, model) if model else None
            res = read(lum, model) if model else None
            if res:
                break
            if model and model[2] >= 30:
                locked.append(model)
        if not res and len(locked) == 1:
            # one twin locked, the other tracked badly: the pilot alone can't tell twins apart, so
            # the locked one may be the wrong one - turn its lattice 180 degrees and refit that
            model = refine(lum, _twin(locked[0]))
            res = read(lum, model)
            if not res and model[2] >= 30:
                locked.append(model)
        if not res and len(locked) == 2:
            # both twins locked onto this picture's lattice and neither read: other seeds inside
            # it would refit to the same lattices, so don't spend them there
            regions.append(_quad(locked[0]))
        if res:
            poly = _quad(model)
            e1, e2 = seed[2], seed[3]
            res.update(quad=poly, mirrored=bool(e1[0] * e2[1] - e1[1] * e2[0] < 0),
                       photo_px_per_subblock=round(float(np.sqrt(abs(e1[0] * e2[1] - e1[1] * e2[0]))) / K, 1))
            found.append(res)
            regions.append(poly)
    return dict(found=found, tried=tried, seeds=len(seeds), overlay=draw(rgb, found))


def draw(rgb: np.ndarray, found: list[dict]) -> np.ndarray:
    f = min(1.0, 1200 / max(rgb.shape[:2]))
    img = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    for i, r in enumerate(found):
        q = (r["quad"] * f).astype(np.int32)
        cv2.polylines(img, [q], True, (40, 220, 60), 3)
        cv2.putText(img, f"#{i + 1}", tuple(int(v) for v in q.mean(0)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 220, 60), 3)
    return img


# ---------------------------------------------------------------- strength ladder (test page)

LADDER_DPI, PAGE_W, PAGE_H = 300, 2480, 3508
LADDER_DELTAS = (1.0, 1.5, 2.0, 3.0)


def make_ladder(rgb: np.ndarray, tag: str = "") -> tuple[np.ndarray, list[dict]]:
    """A4 test page: the same picture encoded at 4 strengths (payload 'd<delta>' + tag), text labels only."""
    page = np.full((PAGE_H, PAGE_W, 3), 255, np.uint8)
    cw, ch = 1040, 1300                    # slot for each copy (8.8 x 11 cm)
    h, w = rgb.shape[:2]
    f = min(cw / w, ch / h)
    img = cv2.resize(rgb, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    b = subblock_for_print(cw, cw / LADDER_DPI * 25.4)
    put = lambda t, x, y, s=1.0: cv2.putText(page, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 2, cv2.LINE_AA)
    put("ImPrint strength ladder - print A4 at 100%, photograph, decode", 150, 170, 1.2)
    stats = []
    for i, d in enumerate(LADDER_DELTAS):
        enc, st = encode(img, f"d{d:g}{tag}".encode()[:MAX_BYTES], delta=d, subblock_px=b)
        x, y = 150 + (i % 2) * (cw + 100), 260 + (i // 2) * (ch + 260)
        page[y:y + enc.shape[0], x:x + enc.shape[1]] = enc
        put(f"strength {d:g}   SSIM {st['ssim']}", x, y + enc.shape[0] + 60, 0.8)
        stats.append(dict(delta=d, **st))
    return page, stats
