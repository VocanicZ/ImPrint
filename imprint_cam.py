"""Camera-robust ImPrint: invisible and borderless.

The encoded image IS the original image (same size, no frame, no markers) plus a faint lightness
texture made of cells, each a KxK grid of sub-blocks. On top of the cells:

* the PILOT (hidden sync, the invisible stand-in for a QR code's finder patterns): three faint
  cosines locked to the cell grid (PILOT_FREQS, cycles per cell). Each is one sharp spectral line,
  so the decoder finds the grid - spacing, angle, mirror, perspective - anywhere in a photo from
  the spectrum, with no border or picture outline. The encoder first notches the image's own
  content at those three frequencies so busy texture can't drown them. Matched-filter phase gives
  the exact cell offset; the decoder then tracks the pilot outward and fits perspective + curl.
* one DATA bit per cell - the sign of a per-cell +-1 sub-block pattern. Bits come in tiles of
  T x T cells repeated over the whole picture, so any big-enough piece decodes (crop-robust); the
  decoder tries every tile phase and both 180-degree twins, Reed-Solomon + CRC-32 confirm.

Patterns are orthogonal to flat color and linear ramps (lighting, white balance, printer drift and
smooth gradients cancel) and only touch L*, so grayscale prints work. Strength follows local texture
and gamut headroom: flat areas get little, clipped areas (pure black/white, saturated) get nothing.
"""
from __future__ import annotations

import binascii
import functools

import cv2
import numpy as np
from reedsolo import ReedSolomonError, RSCodec

K = 4                       # sub-blocks per cell side
SEED = 0x1A9
MAX_BYTES = 32
PILOT = 1.0                 # pilot (sync) amplitude relative to the data strength
SUBBLOCK_MM = 0.8           # target printed sub-block size
ACT_REF = 2.0               # local |Laplacian| of L* that gets 1x strength
MASK_LO, MASK_HI = 0.6, 2.0
PILOT_HI = 1.5              # pilot's mask ceiling (busy texture hides it)
SKIP = 3.0                  # skip cells whose host needs > SKIP x target to overcome
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

# Pilot: three faint cosines locked to the cell grid, frequencies in cycles per cell (x, y).
# Their 2x2 determinants (10, -11, -3) share no factor, so together they repeat exactly once
# per cell (unique cell alignment), and the set has no rotation/mirror symmetry (unique
# orientation). Each is one sharp spectral line: easy to find under any image texture.
PILOT_FREQS = np.array([[3.0, 1.0], [-1.0, 3.0], [2.0, -3.0]])
PILOT_PHASES = np.array([0.3, 1.7, 4.1])


def pilot_field(h: int, w: int, c: float, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """Pilot over an h x w px area whose top-left pixel sits at (x0, y0) px of the cell grid (cell = c px)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx, yy = (xx + 0.5 + x0) / c, (yy + 0.5 + y0) / c
    out = sum(np.cos(2 * np.pi * (fx * xx + fy * yy) + ph) for (fx, fy), ph in zip(PILOT_FREQS, PILOT_PHASES))
    return (out / len(PILOT_FREQS)).astype(np.float32)


def notch_pilot(lum: np.ndarray, c: float, sigma_cells: float = 1.0) -> np.ndarray:
    """Remove the image's own content at the three pilot frequencies (a notch ~1 cell wide in
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


def mask(lum: np.ndarray, b: int, hi: float = MASK_HI) -> np.ndarray:
    """Per-pixel strength multiplier: high in real texture, low in flat areas."""
    act = cv2.GaussianBlur(np.abs(cv2.Laplacian(lum, cv2.CV_32F)), (0, 0), b / 2)
    return cv2.GaussianBlur(np.clip(act / ACT_REF, MASK_LO, hi), (0, 0), b / 2)


def headroom(lab: np.ndarray, amp: float) -> np.ndarray:
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

def frame_len(nbytes: int) -> int:
    """Bytes on the wire for a payload: payload + CRC-32, then rate-1/2 Reed-Solomon."""
    return 2 * (nbytes + 4)


def tile_size(nbytes: int) -> int:
    """Tile side T (cells): a T x T tile holds the whole codeword once."""
    return int(np.ceil(np.sqrt(8 * frame_len(nbytes))))


def to_bits(payload: bytes) -> np.ndarray:
    msg = payload + binascii.crc32(payload).to_bytes(4, "big")
    wire = bytes(RSCodec(len(msg)).encode(msg))
    return np.unpackbits(np.frombuffer(wire, np.uint8)).astype(np.float32) * 2 - 1


def bit_map(n_cells: int, n_bits: int) -> np.ndarray:
    """Which bit each tile position carries (scattered, so local damage hits many bits lightly)."""
    perm = np.random.default_rng(SEED + 1).permutation(n_cells)
    idx = np.empty(n_cells, int)
    idx[perm] = np.arange(n_cells) % n_bits
    return idx


def read_soft(soft: np.ndarray, nbytes: int) -> bytes | None:
    """Soft bits -> Reed-Solomon (least-confident bytes as erasures) -> CRC-32."""
    wire = np.packbits(soft > 0).tobytes()
    nsym = nbytes + 4
    weak = np.argsort(np.abs(soft).reshape(-1, 8).min(1))
    for ne in range(0, nsym + 1, 2):
        try:
            msg = bytes(RSCodec(nsym).decode(wire, erase_pos=[int(i) for i in weak[:ne]] or None)[0])
        except ReedSolomonError:
            continue
        if binascii.crc32(msg[:nbytes]).to_bytes(4, "big") == msg[nbytes:]:
            return msg[:nbytes]
    return None


# ---------------------------------------------------------------- encode

def subblock_for_print(width_px: int, print_mm: float) -> int:
    """Sub-block size (image px) that prints at about SUBBLOCK_MM for this print width."""
    return max(4, int(round(width_px * SUBBLOCK_MM / print_mm)))


def encode(rgb: np.ndarray, payload: bytes, delta: float = 2.0, subblock_px: int = 8):
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
    room = headroom(region, (2.5 * MASK_HI + PILOT * PILOT_HI) * delta)
    room = -cv2.dilate(-room, np.ones((b, b), np.uint8))      # erode: whole sub-block must have room
    m = mask(lum, b) * room
    # the smooth pilot may go stronger where heavy texture hides it; flat areas stay as faint
    pil = PILOT * delta * pilot_field(rows * c, cols * c, c) * mask(lum, b, PILOT_HI) * room
    lum_n = notch_pilot(lum, c)
    notch = (lum_n - lum) * room                              # only where there's room to change
    h = measure(lum + notch + pil, rows, cols, b, D)          # pilot + notch are part of what data must beat
    g = measure(field(np.ones(n, np.float32), rows, cols, b, D) * m, rows, cols, b, D)
    dc = delta * g / gain(b)                                  # target: delta where mask = 1
    # Informed embedding: add only what the host lacks; where it fights hard (or there is no
    # room) skip the cell - repetition and Reed-Solomon cover it - instead of a visible push.
    need = np.maximum(0.0, dc - s * h)
    ok = (g > 1e-6) & (need <= SKIP * dc)
    a = np.where(ok, s * need / np.where(ok, g, 1.0), 0.0).astype(np.float32)
    region[..., 0] += field(a, rows, cols, b, D) * m + pil + notch
    out = to_rgb(lab)

    got = s * measure(to_lab(out)[:rows * c, :cols * c, 0], rows, cols, b, D)
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
    """Welch-averaged power spectrum of a 2n x 2n region (3x3 overlapping n-windows), divided by
    its local geometric mean: image texture -> ~1, random spikes average down, pilot lines stay."""
    P = 0
    for y in (0, n // 2, n):
        for x in (0, n // 2, n):
            w = region[y:y + n, x:x + n]
            P = P + np.abs(np.fft.fft2((w - w.mean()) * _hann(n))) ** 2
    P = np.fft.fftshift(P).astype(np.float32)
    return P / np.exp(cv2.blur(np.log(P + 1e-9), (9, 9)))


def _plausible(A: np.ndarray) -> bool:
    """A photographed cell grid: perspective skews it, but cells stay roughly square."""
    l1, l2 = np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1])
    cos = abs(A[:, 0] @ A[:, 1]) / (l1 * l2 + 1e-9)
    return 0.5 < l1 / l2 < 2 and cos < 0.65


def sync_in_region(region: np.ndarray, n: int = 256, top: int = 10):
    """Find the three pilot lines in a region's spectrum and solve the cell grid.
    Returns (score, A) with A: cell coords -> px offsets (columns = cell vectors), or None."""
    R = _whitened_spectrum(region, n)
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

    best = None
    for a in range(len(obs)):
        for b in range(len(obs)):
            if a == b or a % (len(obs) // 2) == b % (len(obs) // 2):
                continue
            for i in range(3):
                for j in range(3):
                    if i == j:
                        continue
                    F2 = PILOT_FREQS[[i, j]]
                    Mt = np.linalg.solve(F2, obs[[a, b]])            # obs = F @ Mt
                    k = 3 - i - j
                    third = PILOT_FREQS[k] @ Mt
                    A = np.linalg.inv(Mt)
                    cell = np.sqrt(abs(np.linalg.det(A)))
                    if not 12 <= cell <= n / 3 or not _plausible(A):
                        continue
                    rk = r_at(third)
                    if rk < 4:
                        continue
                    score = float(logR[a] + logR[b] + np.log(rk))
                    if best is None or score > best[0]:
                        best = (score, Mt, (a, b, i, j, third))
    if best is None:
        return None
    # refine with all three lines (sub-pixel position of the predicted third)
    score, Mt, (a, b, i, j, third) = best
    xi, yi = int(round(c + third[0] * n)), int(round(c + third[1] * n))
    y3, x3 = _subpixel(R, yi, xi)
    fo = np.array([obs[a], obs[b], [(x3 - c) / n, (y3 - c) / n]])
    Mt, *_ = np.linalg.lstsq(PILOT_FREQS[[i, j, 3 - i - j]], fo, rcond=None)
    return score, np.linalg.inv(Mt)


def find_seeds(lum: np.ndarray) -> list:
    """Scan the photo (all pyramid levels) in 2n x 2n regions for the pilot (n = 256, or 128 when
    the image/level is small). Returns [(score, centre_xy, e1, e2)] in full-res px, best first."""
    seeds, img, f = [], lum.astype(np.float32), 1
    while min(img.shape) >= 256:
        n = 256 if min(img.shape) >= 512 else 128
        for y in range(0, img.shape[0] - 2 * n + 1, n // 2):
            for x in range(0, img.shape[1] - 2 * n + 1, n // 2):
                r = sync_in_region(img[y:y + 2 * n, x:x + 2 * n], n)
                if r:
                    seeds.append((r[0], np.array([x + n, y + n]) * f, r[1][:, 0] * f, r[1][:, 1] * f))
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
    """Matched filter at the three pilot frequencies only (data and image live elsewhere).
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


def _pilot_shift(mc: np.ndarray, tpl: np.ndarray):
    """Circular cross-correlation of a mean cell with the pilot cell: (shift_yx, ncc)."""
    mc = mc - mc.mean()
    C = np.fft.ifft2(np.fft.fft2(mc) * np.conj(np.fft.fft2(tpl))).real
    dy, dx = np.unravel_index(np.argmax(C), C.shape)
    return np.array([dy, dx]), float(C[dy, dx] / (np.linalg.norm(mc) * np.linalg.norm(tpl) + 1e-12))


# ---------------------------------------------------------------- decode

def track(lum: np.ndarray, seed, sgn: int = 1, r: int = 6, nc: int = 8, coh_min: float = 0.7,
          snr_min: float = 2.5):
    """From one seed window, lock onto the pilot phase (sub-cell offset) and follow it outward over
    the picture, refitting a lattice->photo map (perspective + curl) as it grows. Lattice coords =
    true cell coords up to a whole-cell shift. sgn picks one of the two 180-degree twins that
    cosine lines can't tell apart (the caller tries both; the CRC decides).
    Returns (fmap, extent, n_probes) or None."""
    _, c0, e1, e2 = seed
    S = K * r
    e1, e2 = sgn * e1, sgn * e2

    def probe(fm, cx, cy):
        return pilot_phase(resample(lum, fm, cx - nc / 2, cy - nc / 2, nc, nc, S), nc, S)

    fmap = lambda q: c0 + q[:, :1] * e1 + q[:, 1:2] * e2
    off0, coh0, snr0 = probe(fmap, 0.0, 0.0)
    if coh0 < coh_min or snr0 < snr_min:
        return None
    lat, pho = [off0.copy()], [c0.copy()]
    expect = off0.copy()                # offset the current map should show (0 once it's in true coords)
    done, ring = {(0, 0)}, 1
    while ring < 60:
        added = inside = 0
        for i in range(-ring, ring + 1):
            for j in range(-ring, ring + 1):
                if max(abs(i), abs(j)) != ring or (i, j) in done:
                    continue
                done.add((i, j))
                cx, cy = i * nc, j * nc
                p = fmap(np.array([[cx, cy]], np.float64))[0]
                if not (0 <= p[0] < lum.shape[1] and 0 <= p[1] < lum.shape[0]):
                    continue
                inside += 1
                off, coh, snr = probe(fmap, cx, cy)
                if coh < coh_min or snr < snr_min:
                    continue
                off = expect + (off - expect + 0.5) % 1 - 0.5          # unwrap next to the expected offset
                lat.append(np.array([cx, cy]) + off)
                pho.append(p)
                added += 1
        if inside == 0:                                         # whole ring outside the photo
            break
        if added == 0:                                          # faint area (flat image region): keep going
            ring += 1
            continue
        L_, P_ = np.array(lat), np.array(pho)
        if len(L_) >= 6:
            H, inl = cv2.findHomography(L_, P_, cv2.RANSAC, 0.3 * np.linalg.norm(e1))
            if H is not None:
                keep = inl.ravel().astype(bool)
                lat, pho = list(L_[keep]), list(P_[keep])
                fm = fit_map(np.array(lat), np.array(pho), cubic=len(lat) >= 30)
                if fm is not None:
                    fmap, expect = fm, np.zeros(2)          # map now speaks true lattice coords
        ring += 1
    L_ = np.array(lat)
    if len(L_) >= 4:                                            # final fit in true lattice coords
        fm = fit_map(L_, np.array(pho), cubic=len(L_) >= 30)
        if fm is not None:
            fmap = fm
    else:                                                       # seed only: shift the affine
        fmap = lambda q, f0=fmap, o=off0: f0(q - o)
    ext = (L_[:, 0].min() - nc / 2, L_[:, 0].max() + nc / 2, L_[:, 1].min() - nc / 2, L_[:, 1].max() + nc / 2)
    return fmap, ext, len(L_)


def read(lum: np.ndarray, model, r: int = 6) -> dict | None:
    """Rectify the tracked lattice, align cells on the pilot, remove it, then try every payload
    length and tile phase: soft-combine repeats, Reed-Solomon, CRC."""
    fmap, (x0, x1, y0, y1), nprobe = model
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
    pos = best_p[best_p > 0]
    med = np.median(pos) if len(pos) else 1.0
    w = np.clip(best_p / (med + 1e-9), 0, 2)                      # pilot strength = cell trust
    m = best_m.reshape(rows * cols, K * K)
    used = w > 0.2
    if used.sum() < 16:
        return None
    m = m - np.average(m[used], axis=0, weights=w[used])          # remove the pilot (same in every cell)

    ii, jj = np.mgrid[0:rows, 0:cols]
    ii, jj = ii.ravel(), jj.ravel()
    for nb in range(1, MAX_BYTES + 1):
        T, nbits = tile_size(nb), 8 * frame_len(nb)
        if used.sum() < T * T:
            break
        D = pats(T * T).reshape(T * T, K * K)
        bm = bit_map(T * T, nbits)
        V = (m @ D.T) / (K * K) * w[:, None]                     # cell x tile-position correlation
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
            payload = read_soft(soft, nb)
            if payload:
                s = to_bits(payload)[bm[t]]
                v = V[np.arange(len(t)), t]
                return dict(payload=payload, raw_ber=round(float((s[used] * v[used] <= 0).mean()), 4),
                            cells=int(used.sum()), tiles=round(used.sum() / T ** 2, 1), probes=nprobe)
    return None


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
        res = None
        for sgn in (1, -1):
            model = track(lum, seed, sgn)
            res = read(lum, model) if model else None
            if res:
                break
        if res:
            fmap, (x0, x1, y0, y1), _ = model
            poly = fmap(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)).astype(np.float32)
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
