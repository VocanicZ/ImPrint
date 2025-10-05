#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Imprint prototype v0.3 — Mixed-Radix per tile (8 or 16), auto rows, auto decode.

Key ideas:
- Corner markers (3) define geometry; decode scans iPS∈[10..20], κ∈[4..7].
- Each tile's CVP capacity is estimated in Lab by checking whether 1 or 2
  small radii (ΔE step) survive gamut/clipping and remain below ΔE_max.
- Header is stored first in base-8 over the earliest tiles (robust).
- Payload is stored in mixed-radix across the remaining tiles.
- Decoder recomputes the same bases from the image and reads digits.
"""

from __future__ import annotations
import argparse, json, struct, zlib, math
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw

# ----------------------- sRGB <-> Lab -----------------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    m = c <= 0.04045
    out = np.empty_like(c)
    out[m] = c[m] / 12.92
    out[~m] = ((c[~m] + 0.055) / 1.055) ** 2.4
    return out

def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    m = c <= 0.0031308
    out = np.empty_like(c)
    out[m] = 12.92 * c[m]
    out[~m] = 1.055 * (c[~m] ** (1/2.4)) - 0.055
    return np.clip(out, 0.0, 1.0)

def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    r, g, b = _srgb_to_linear(rgb[...,0]), _srgb_to_linear(rgb[...,1]), _srgb_to_linear(rgb[...,2])
    X = 0.4124564*r + 0.3575761*g + 0.1804375*b
    Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    Z = 0.0193339*r + 0.1191920*g + 0.9503041*b
    return np.stack([X,Y,Z], axis=-1)

def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    X, Y, Z = xyz[...,0], xyz[...,1], xyz[...,2]
    r =  3.2404542*X - 1.5371385*Y - 0.4985314*Z
    g = -0.9692660*X + 1.8760108*Y + 0.0415560*Z
    b =  0.0556434*X - 0.2040259*Y + 1.0572252*Z
    return np.stack([_linear_to_srgb(r), _linear_to_srgb(g), _linear_to_srgb(b)], axis=-1)

def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[...,0]/Xn, xyz[...,1]/Yn, xyz[...,2]/Zn
    def f(t):
        e = 216/24389
        k = 24389/27
        t3 = np.cbrt(t)
        return np.where(t > e, t3, (k*t + 16)/116)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.stack([L,a,b], axis=-1)

def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    L, a, b = lab[...,0], lab[...,1], lab[...,2]
    fy = (L + 16)/116
    fx = fy + a/500
    fz = fy - b/200
    def finv(f):
        e = 216/24389
        k = 24389/27
        f3 = f**3
        return np.where(f3 > e, f3, (116*f - 16)/k)
    x = finv(fx) * Xn
    y = finv(fy) * Yn
    z = finv(fz) * Zn
    return np.stack([x,y,z], axis=-1)

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return xyz_to_lab(rgb_to_xyz(rgb))

def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return xyz_to_rgb(lab_to_xyz(lab))

# ---------------------- geometry & markers ----------------------

PAD = 12
MARK_BORDER = 6
MICRO_BAR_H = 4
MICRO_BAR_W = 4

def draw_marker(img: Image.Image, x: int, y: int, size: int):
    d = ImageDraw.Draw(img)
    d.rectangle([x, y, x+size, y+size], fill=(0,0,0))
    inner = [x+MARK_BORDER, y+MARK_BORDER, x+size-MARK_BORDER, y+size-MARK_BORDER]
    d.rectangle(inner, fill=(255,255,255))
    ix0, iy0, ix1, iy1 = inner
    bar_y = iy0 + 2
    colors = [(255,255,255),(255,0,0),(0,255,0),(0,0,255),(0,0,0)]
    for i, col in enumerate(colors):
        bx0 = ix0 + 2 + i*MICRO_BAR_W
        bx1 = bx0 + MICRO_BAR_W - 1
        by0 = bar_y
        by1 = by0 + MICRO_BAR_H - 1
        ImageDraw.Draw(img).rectangle([bx0,by0,bx1,by1], fill=col)

def place_markers(pil: Image.Image, ips: int, kappa: int):
    size = kappa*ips + 2*MARK_BORDER
    W, H = pil.size
    draw_marker(pil, PAD, PAD, size)                    # TL
    draw_marker(pil, W - PAD - size, PAD, size)         # TR
    draw_marker(pil, PAD, H - PAD - size, size)         # BL
    def inner_rect(x,y): return (x+MARK_BORDER,y+MARK_BORDER,x+size-MARK_BORDER,y+size-MARK_BORDER)
    tl = inner_rect(PAD, PAD)
    tr = inner_rect(W-PAD-size, PAD)
    bl = inner_rect(PAD, H-PAD-size)
    return tl,tr,bl,size

# ---------------------- symbols, masks, utils ----------------------

def hann2(h:int, w:int, eps:float=1e-6)->np.ndarray:
    wx = 0.5*(1-np.cos(2*np.pi*(np.arange(w)+0.5)/w))
    wy = 0.5*(1-np.cos(2*np.pi*(np.arange(h)+0.5)/h))
    m = np.outer(wy, wx)
    m = (m - m.min())/(m.max()-m.min()+eps)
    return m.astype(np.float32)

def ring_dirs(N:int=8)->np.ndarray:
    ang = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.stack([np.cos(ang), np.sin(ang)], axis=-1).astype(np.float32)

def moving_avg(v: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win))
    k = np.ones(win, dtype=np.float32) / win
    pad = win//2
    vp = np.pad(v, (pad,pad), mode="edge")
    s = np.convolve(vp, k, mode="valid")
    return s[:len(v)]

# --------------- mixed-radix helpers ----------------

def digits_to_bits(digs: List[int], bases: List[int]) -> List[int]:
    """Inverse of bits_to_digits when bases are all 8 or 16.
       Used ONLY for header (bases constant=8) in this file."""
    out = []
    for d, b in zip(digs, bases):
        if b == 8:
            out += [ (d>>2)&1, (d>>1)&1, d&1 ]
        elif b == 16:
            out += [ (d>>3)&1, (d>>2)&1, (d>>1)&1, d&1 ]
        else:
            raise ValueError("invalid base for digits_to_bits")
    return out

def bits_to_bytes(bits: List[int]) -> bytes:
    if len(bits)%8: bits += [0]*(8 - (len(bits)%8))
    out = bytearray()
    for i in range(0,len(bits),8):
        b=0
        for j in range(8):
            b=(b<<1)|(bits[i+j]&1)
        out.append(b)
    return bytes(out)

def bytes_to_bits(bb: bytes) -> List[int]:
    return [ (b>>(7-i)) & 1 for b in bb for i in range(8) ]

def pack_mixed_radix(value_bytes: bytes, bases: List[int]) -> Tuple[List[int], int]:
    """Pack bytes into digits with bases[i] ∈ {8,16}.
       We pick the minimal K such that Π_{i< K} base[i] >= 2^(8*len(bytes)).
       Return (digits, K)."""
    M_bits = 8*len(value_bytes)
    if M_bits == 0:
        return [], 0
    # minimal K
    acc = 1
    K = 0
    for b in bases:
        acc *= b
        K += 1
        if acc >= (1 << M_bits):
            break
    if acc < (1 << M_bits):
        raise ValueError("Not enough capacity for payload")
    # interpret bytes as big integer
    M = int.from_bytes(value_bytes, "big")
    digs = []
    for i in range(K):
        b = bases[i]
        d = M % b
        digs.append(int(d))
        M //= b
    return digs, K

def unpack_mixed_radix(digs: List[int], bases: List[int], K:int, out_bytes:int) -> bytes:
    if K == 0: return b""
    M = 0
    for i in reversed(range(K)):
        M = M * bases[i] + int(digs[i])
    return M.to_bytes(out_bytes, "big")

# ------------------ header ------------------

MAGIC = b"IMRX"   # new magic for mixed-radix
VER   = 1
# Header (fixed size, encoded in base-8 over earliest tiles):
# IMRX (4) | ver:1 | ips:2 | kappa:1 | rows:1 | cols:2 | tiles:2 |
# hdr_tiles:2 | payload_bytes:4 | payload_K:4 | CRC32:4
HDR_FMT = ">B H B B H H H I I I"
HDR_LEN = 1+2+1+1+2+2+2+4+4+4

def build_header(ips:int, kappa:int, rows:int, cols:int, tiles:int,
                 hdr_tiles:int, payload_bytes:int, payload_K:int,
                 payload: bytes) -> bytes:
    core = struct.pack(HDR_FMT, VER, ips, kappa, rows, cols, tiles, hdr_tiles,
                       payload_bytes, payload_K, 0)
    crc = zlib.crc32(core[:-4] + payload) & 0xffffffff
    return MAGIC + core[:-4] + struct.pack(">I", crc)

def parse_header(bb: bytes):
    if len(bb) < 4 + HDR_LEN or bb[:4] != MAGIC:
        raise ValueError("bad magic/length")
    off = 4
    ver, ips, kappa, rows, cols, tiles, hdr_tiles, pbytes, pK, crc = struct.unpack(
        HDR_FMT, bb[off: off+HDR_LEN])
    return ver, ips, kappa, rows, cols, tiles, hdr_tiles, pbytes, pK, crc

# ------------------ capacity estimator (CVP) ------------------

def estimate_tile_base(mean_lab: np.ndarray, delta_step: float, delta_max: float) -> int:
    """Return 0, 8, or 16. Conservative check on 8 directions, 1 or 2 radii."""
    dirs = ring_dirs(8)
    base = 0
    # radius 1
    ok1 = True
    for d in dirs:
        test = mean_lab.copy()
        test[1] += d[0]*delta_step
        test[2] += d[1]*delta_step
        rgb = lab_to_rgb(test[None,None,:])[0,0,:]
        if np.any((rgb <= 0.0) | (rgb >= 1.0)):
            ok1 = False
            break
    if ok1:
        base = 8
    else:
        return 0
    # radius 2 (if allowed and below delta_max)
    if delta_step*2.0 <= delta_max:
        ok2 = True
        for d in dirs:
            test = mean_lab.copy()
            test[1] += d[0]*delta_step*2.0
            test[2] += d[1]*delta_step*2.0
            rgb = lab_to_rgb(test[None,None,:])[0,0,:]
            if np.any((rgb <= 0.0) | (rgb >= 1.0)):
                ok2 = False
                break
        if ok2:
            base = 16
    return int(base)

# ------------------ ENCODE ------------------

def encode_image(p_in:str, p_out:str, text:str,
                 ips:int, kappa:int,
                 delta_step: float, delta_max: float) -> dict:
    pil = Image.open(p_in).convert("RGB")
    W,H = pil.size

    # place markers
    tl,tr,bl, msize = place_markers(pil, ips, kappa)

    # band geometry
    band_h = ips
    x0, x1 = PAD, W-PAD
    cols = (x1 - x0) // ips
    base_top = tl[3] + ips//2

    # compute as many rows as needed automatically
    # first, scan row by row and compute bases
    np_img = np.asarray(pil, dtype=np.float32)/255.0
    lab = rgb_to_lab(np_img)
    rows_bases : List[List[int]] = []
    rows_means : List[List[np.ndarray]] = []
    r = 0
    while True:
        yy0 = base_top + r * (ips + ips//2)
        if yy0 + band_h > H - PAD:
            break
        row_bases = []
        row_means = []
        for c in range(cols):
            cx0 = x0 + c*ips
            cx1 = cx0 + ips
            tile = lab[yy0:yy0+band_h, cx0:cx1, :]
            mean_lab = tile.reshape(-1,3).mean(axis=0)
            base = estimate_tile_base(mean_lab, delta_step, delta_max)
            row_bases.append(base)
            row_means.append(mean_lab)
        rows_bases.append(row_bases)
        rows_means.append(row_means)
        r += 1
    rows = len(rows_bases)
    tiles = rows * cols
    if rows == 0 or cols <= 0:
        raise ValueError("Image too small for these iPS/κ settings")

    # header bytes
    payload = text.encode("utf-8")

    # header encoding plan: use first H tiles that have base >= 8 in scan order
    header = build_header(ips, kappa, rows, cols, tiles, 0, len(payload), 0, payload)  # hdr_tiles & pK filled later
    header_bits = bytes_to_bits(header)
    header_digits = []
    for i in range(0, len(header_bits), 3):
        tri = header_bits[i:i+3]
        if len(tri)<3: tri += [0]*(3-len(tri))
        d = (tri[0]<<2)|(tri[1]<<1)|tri[2]
        header_digits.append(d)
    # count how many header tiles we need (base 8 only)
    H_need = len(header_digits)

    # make a linear list of bases (skip base==0)
    bases_linear : List[int] = []
    coords_linear: List[Tuple[int,int]] = []
    for rr in range(rows):
        for cc in range(cols):
            b = rows_bases[rr][cc]
            if b >= 8:
                bases_linear.append(b)
                coords_linear.append((rr,cc))

    if len(bases_linear) < H_need:
        raise ValueError("Not enough robust tiles to write header")

    # payload packing with remaining tiles
    payload_bases = bases_linear[H_need:]
    payload_digits, K = pack_mixed_radix(payload, payload_bases)

    # Rebuild header with final hdr_tiles and payload_K
    header = build_header(ips, kappa, rows, cols, tiles, H_need, len(payload), K, payload)
    header_bits = bytes_to_bits(header)
    header_digits = []
    for i in range(0, len(header_bits), 3):
        tri = header_bits[i:i+3]
        if len(tri)<3: tri += [0]*(3-len(tri))
        d = (tri[0]<<2)|(tri[1]<<1)|tri[2]
        header_digits.append(d)
    H_need = len(header_digits)  # unchanged, but recomputed for sanity

    # Compose final digits per used tile
    used_digits = header_digits + payload_digits
    used_coords = coords_linear[:H_need+K]
    used_bases  = bases_linear[:H_need] + payload_bases[:K]

    # embed: Lab ring with 1 or 2 radii, blended
    dirs = ring_dirs(8)
    taper = hann2(ips, ips)

    for idx, (rr,cc) in enumerate(used_coords):
        yy0 = base_top + rr*(ips + ips//2)
        cx0 = x0 + cc*ips
        tile = lab[yy0:yy0+ips, cx0:cx0+ips, :]
        b = used_bases[idx]
        d = used_digits[idx]
        if b == 8:
            ang_idx = d % 8
            rad_idx = 0
        else: # 16
            ang_idx = d % 8
            rad_idx = d // 8
        va, vb = dirs[ang_idx]
        radius = delta_step * (1 if rad_idx==0 else 2.0)
        tile[...,1] = tile[...,1] + va * radius * taper
        tile[...,2] = tile[...,2] + vb * radius * taper
        lab[yy0:yy0+ips, cx0:cx0+ips, :] = tile

    out_rgb = (lab_to_rgb(lab)*255.0).clip(0,255).astype(np.uint8)
    out = Image.fromarray(out_rgb, mode="RGB")
    out.save(p_out, format="PNG", optimize=True)

    return {
        "verdict":"IMPRINT_OK",
        "output": p_out,
        "layout":{"iPS":ips,"kappa":kappa,"rows":rows,"cols":cols,"tiles":tiles},
        "header_tiles": H_need,
        "payload_K": K,
        "debug":{"delta_step":delta_step,"delta_max":delta_max}
    }

# ------------------ DECODE ------------------

def decode_image(p_img:str) -> dict:
    pil = Image.open(p_img).convert("RGB")
    W,H = pil.size
    lab = rgb_to_lab(np.asarray(pil, dtype=np.float32)/255.0)
    dirs = ring_dirs(8)

    # scan small grids for (ips,kappa); choose best by directional coherence
    best = None
    for ips in range(10,21):         # 10..20
        for kappa in range(4,8):     # 4..7
            size = kappa*ips + 2*MARK_BORDER
            base_top = PAD+MARK_BORDER + (size-2*MARK_BORDER) + ips//2
            band_h = ips
            x0, x1 = PAD, W-PAD
            cols = (x1 - x0)//ips
            if cols < 24: continue
            # try 1 row first to score
            rr = 0
            yy0 = base_top + rr*(ips + ips//2)
            if yy0 + band_h > H-PAD: continue
            a_row = []; b_row=[]
            for c in range(cols):
                cx0 = x0 + c*ips
                tile = lab[yy0:yy0+band_h, cx0:cx0+ips, :]
                a_row.append(float(tile[...,1].mean()))
                b_row.append(float(tile[...,2].mean()))
            a_row = np.array(a_row, dtype=np.float32)
            b_row = np.array(b_row, dtype=np.float32)
            a_res = a_row - moving_avg(a_row, max(5, cols//32))
            b_res = b_row - moving_avg(b_row, max(5, cols//32))
            V = np.stack([a_res,b_res], axis=-1)
            norms = np.linalg.norm(V, axis=1) + 1e-9
            U = (V.T / norms).T
            sims = U @ dirs.T
            ang = np.max(sims, axis=1)
            score = float(np.clip(ang,0,1).mean() * np.median(norms))
            cand = {"score":score,"ips":ips,"kappa":kappa,"cols":int(cols),"base_top":int(base_top)}
            if best is None or score > best["score"]:
                best = cand

    if best is None:
        return {"verdict":"NO_IMPRINT","reason":"no geometry"}

    ips   = best["ips"]
    kappa = best["kappa"]
    cols  = best["cols"]
    base_top = best["base_top"]
    band_h = ips
    x0 = PAD

    # determine rows that still fit
    rows=0
    while True:
        yy0 = base_top + rows*(ips + ips//2)
        if yy0 + band_h > H-PAD: break
        rows += 1
    if rows == 0:
        return {"verdict":"NO_IMPRINT","reason":"no rows"}

    # recompute per-tile bases (same as encoder)
    bases_linear: List[int] = []
    coords_linear: List[Tuple[int,int]] = []
    means_linear: List[np.ndarray] = []
    for rr in range(rows):
        yy0 = base_top + rr*(ips + ips//2)
        for cc in range(cols):
            cx0 = x0 + cc*ips
            tile = lab[yy0:yy0+ips, cx0:cx0+ips, :]
            mean_lab = tile.reshape(-1,3).mean(axis=0)
            # decoder uses fixed step/max; they must match encoder defaults
            base = estimate_tile_base(mean_lab, delta_step=0.6, delta_max=1.6)
            # be tolerant: if 0, still push to 8 so header can survive in easy images
            if base == 0: base = 8
            bases_linear.append(base)
            coords_linear.append((rr,cc))
            means_linear.append(mean_lab)

    # classify first H header tiles as base-8 to read header
    # compute residual vector per tile
    a_cols = []
    b_cols = []
    for rr in range(rows):
        yy0 = base_top + rr*(ips + ips//2)
        a_row=[]; b_row=[]
        for cc in range(cols):
            cx0 = x0 + cc*ips
            tile = lab[yy0:yy0+ips, cx0:cx0+ips, :]
            a_row.append(float(tile[...,1].mean()))
            b_row.append(float(tile[...,2].mean()))
        a_cols.append(np.array(a_row, dtype=np.float32))
        b_cols.append(np.array(b_row, dtype=np.float32))
    # lazy baseline per row
    a_res_all=[]; b_res_all=[]
    for rr in range(rows):
        a_res_all.append(a_cols[rr] - moving_avg(a_cols[rr], max(5, cols//32)))
        b_res_all.append(b_cols[rr] - moving_avg(b_cols[rr], max(5, cols//32)))

    def tile_vec(idx:int)->np.ndarray:
        rr,cc = coords_linear[idx]
        return np.array([a_res_all[rr][cc], b_res_all[rr][cc]], dtype=np.float32)

    # helper to read base-8 digit by angle
    def read_digit8(idx:int)->int:
        v = tile_vec(idx)
        n = np.linalg.norm(v) + 1e-9
        u = v / n
        sims = u @ dirs.T
        return int(np.argmax(sims))

    # try increasing H until header parses cleanly (cap at 256 tiles)
    H_try = 16
    header_bytes = None
    hdr = None
    while H_try <= min(256, len(bases_linear)):
        digs8 = []
        for i in range(H_try):
            digs8.append(read_digit8(i))
        bits = digits_to_bits(digs8, [8]*len(digs8))
        bb = bits_to_bytes(bits)
        try:
            ver, ips_h, kappa_h, rows_h, cols_h, tiles_h, hdr_tiles_h, pbytes_h, pK_h, crc = parse_header(bb)
            core = bb[4:4+HDR_LEN]
            if len(bb) < 4+HDR_LEN: raise ValueError("short header core")
            # Verify CRC with payload later; for now just sanity on fields
            if not(8 <= ips_h <= 64): raise ValueError("ips range")
            if not(1 <= rows_h <= 128): raise ValueError("rows range")
            header_bytes = bb[:4+HDR_LEN]
            hdr = (ver, ips_h, kappa_h, rows_h, cols_h, tiles_h, hdr_tiles_h, pbytes_h, pK_h, crc)
            break
        except Exception:
            H_try += 4
            continue

    if hdr is None:
        return {"verdict":"IMPRINT_FOUND_BUT_HEADER_FAIL","score":best["score"],"debug":{"best":best}}

    ver, ips_h, kappa_h, rows_h, cols_h, tiles_h, hdr_tiles_h, pbytes_h, pK_h, crc_h = hdr
    H_need = int(hdr_tiles_h)
    if H_need > len(bases_linear):
        return {"verdict":"IMPRINT_TRUNCATED","debug":{"need":H_need,"have":len(bases_linear)}}

    # read payload digits next
    start = H_need
    K = int(pK_h)
    if start+K > len(bases_linear):
        return {"verdict":"IMPRINT_TRUNCATED","debug":{"payload_need":start+K,"have":len(bases_linear)}}

    payload_digs = []
    for i in range(K):
        idx = start + i
        v = tile_vec(idx)
        n = np.linalg.norm(v) + 1e-9
        u = v / n
        ang = int(np.argmax(u @ dirs.T))
        if bases_linear[idx] == 16:
            # data-driven split by relative radius (row median)
            rr,_ = coords_linear[idx]
            # robust scale
            norms_row = np.linalg.norm(np.stack([a_res_all[rr], b_res_all[rr]], axis=-1), axis=1) + 1e-9
            med = np.median(norms_row)
            radius_idx = 1 if (n >= 1.5*med) else 0
            d = radius_idx*8 + ang
        else:
            d = ang
        payload_digs.append(int(d))

    payload = unpack_mixed_radix(payload_digs, bases_linear[start:start+K], K, int(pbytes_h))

    # CRC check (header core + payload)
    core = header_bytes[4:4+HDR_LEN]
    crc_ok = (zlib.crc32(core[:-4] + payload) & 0xffffffff) == struct.unpack(">I", core[-4:])[0]

    return {
        "verdict": "OK" if crc_ok else "IMPRINT_FOUND_BUT_BAD_CRC",
        "score": best["score"],
        "message": payload.decode("utf-8","replace") if crc_ok else None,
        "header": {
            "version": ver, "iPS": ips_h, "kappa": kappa_h,
            "rows": rows_h, "cols": cols_h, "tiles": tiles_h,
            "header_tiles": hdr_tiles_h, "payload_bytes": pbytes_h, "payload_K": pK_h
        },
        "debug": {"best":best, "crc_ok": bool(crc_ok)}
    }

# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser(description="Imprint mixed-radix prototype")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("encode", help="encode message")
    ep.add_argument("input")
    ep.add_argument("output")
    ep.add_argument("--text", required=True)
    ep.add_argument("--iPS", type=int, default=14)
    ep.add_argument("--kappa", type=int, default=5)
    ep.add_argument("--delta_step", type=float, default=0.6, help="ΔE per radius")
    ep.add_argument("--delta_max",  type=float, default=1.6, help="max total ΔE")

    dp = sub.add_parser("decode", help="decode message")
    dp.add_argument("input")

    args = ap.parse_args()

    if args.cmd == "encode":
        res = encode_image(args.input, args.output, args.text,
                           args.iPS, args.kappa, args.delta_step, args.delta_max)
        print(json.dumps(res, indent=2))
    else:
        res = decode_image(args.input)
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()