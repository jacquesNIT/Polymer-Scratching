# -*- coding: utf-8 -*-
"""lab_values2.py -- residual depth and pile-up from a TriboSoft .bcrf scan.

Reads a .bcrf height map of a scratch track and extracts:

    * the residual groove depth  h_r  along the track
    * the lateral pile-up heights  h_p  on both sides, along the track
    * the terminal frontal mound left at the lift-off point
    * transverse sections at N positions, one of which is the deepest point
    * the groove / pile-up area balance, section by section

Reference surface
-----------------
The form is fitted on lateral reference bands only, well outside the pile-up
ridges, then extrapolated underneath the track. Fitting on the whole field
would absorb part of the pile-up into the form and bias h_p low. The fit is
iterated with sigma clipping so that debris and fringe-order artefacts do not
drag the reference.

Usage::

    python lab_values2.py Test3_PMMAXT_10N.bcrf
    python lab_values2.py scan.bcrf --smooth-x 40 --smooth-y 6 --sections 5
    python lab_values2.py scan.bcrf --outdir figures/ --export profiles/

Dependencies: numpy and matplotlib only.

Units: lengths in um throughout.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================================
# 1. bcrf reader
# ==========================================================================

_LENGTH_SCALE = {"nm": 1e-3, "um": 1.0, "mm": 1e3}   # -> micrometres


def read_bcrf(path):
    """Return (Z, header) with Z a 2-D float array in the header's zunit.

    Layout: a UTF-16LE text header of `headersize` CHARACTERS, followed by
    xpixels*ypixels float32 little-endian values, row-major (x fastest).
    Non-measured pixels carry the `voidpixels` sentinel and are set to NaN.
    """
    with open(path, "rb") as fh:
        raw = fh.read(4096)
    txt = raw.decode("utf-16-le", errors="replace")

    hdr = {}
    for line in txt.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            hdr[k.strip()] = v.strip()

    if "xpixels" not in hdr or "ypixels" not in hdr:
        raise ValueError("%s: not a bcrf file (no pixel count in header)" % path)

    offset = 2 * int(hdr.get("headersize", 2048))
    nx, ny = int(hdr["xpixels"]), int(hdr["ypixels"])

    data = np.fromfile(path, dtype="<f4", offset=offset, count=nx * ny)
    if data.size != nx * ny:
        raise ValueError("%s: truncated, got %d values, expected %d"
                         % (path, data.size, nx * ny))

    z = data.astype(np.float64).reshape(ny, nx)
    void = float(hdr.get("voidpixels", 3.402823e38))
    z[~np.isfinite(z) | (np.abs(z) >= 0.99 * void)] = np.nan

    return z, hdr


def pixel_size(hdr):
    """Return (dx, dy) in micrometres."""
    sx = _LENGTH_SCALE[hdr.get("xunit", "um").strip()]
    sy = _LENGTH_SCALE[hdr.get("yunit", "um").strip()]
    dx = float(hdr["xlength"]) * sx / int(hdr["xpixels"])
    dy = float(hdr["ylength"]) * sy / int(hdr["ypixels"])
    return dx, dy


# ==========================================================================
# 2. numerics
# ==========================================================================

def moving_average(v, win):
    """Centred moving average with edge replication, NaN-tolerant."""
    v = np.asarray(v, dtype=float)
    win = int(max(1, win))
    if win <= 1 or v.size == 0:
        return v.astype(float).copy()
    win = min(win, v.size)
    half = win // 2

    finite = np.isfinite(v)
    filled = np.where(finite, v, 0.0)

    pad_l, pad_r = half, win - 1 - half
    fp = np.concatenate([np.full(pad_l, filled[0]), filled,
                         np.full(pad_r, filled[-1])])
    wp = np.concatenate([np.full(pad_l, float(finite[0])),
                         finite.astype(float),
                         np.full(pad_r, float(finite[-1]))])

    cs = np.concatenate([[0.0], np.cumsum(fp)])
    cw = np.concatenate([[0.0], np.cumsum(wp)])
    num, den = cs[win:] - cs[:-win], cw[win:] - cw[:-win]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def smooth2d(Z, win_x=1, win_y=1):
    """Separable moving average along x (axis 1) then y (axis 0)."""
    out = Z
    if win_x > 1:
        out = np.apply_along_axis(moving_average, 1, out, win_x)
    if win_y > 1:
        out = np.apply_along_axis(moving_average, 0, out, win_y)
    return out


def robust_sigma(v):
    """MAD-based standard deviation estimate, immune to outliers."""
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(v - np.median(v)))


# ==========================================================================
# 2b. outlier rejection                                     [despike-patch]
#
# Dust, debris and fringe-order errors on a transparent polymer show up as
# compact spikes a few pixels wide and tens of times the noise floor. They
# are fatal here because h_r and h_p are *extrema* over a band: one bad
# pixel sets the value of a whole column, and one bad column can be picked
# as "the deepest section". A moving average cannot fix that -- it spreads
# the spike instead of removing it. What is needed is rejection.
#
# Three levels, from the map to the profiles:
#     despike_map()   compact outliers in Z  ->  local robust baseline
#     median_axis()   running median across the track, before the extremum
#     hampel()        residual outliers on the along-track profiles
#
# The baseline used for detection is a running median taken ALONG THE TRACK.
# That direction is chosen on purpose: groove, ridges and mound vary slowly
# with x, so a legitimate feature leaves almost no residual, while a blob of
# limited extent in x stands out whatever its width in y.
# ==========================================================================

def median_axis(Z, win, axis=0, max_samples=41, block_elems=8_000_000):
    """Centred running median along one axis, edge replication, NaN-tolerant.

    For a long window the median is evaluated on a regularly decimated subset
    of the window (at most `max_samples` points). A median of ~40 samples is
    statistically as good a baseline as a median of 160 and costs four times
    less; the residual jitter is ~0.2 sigma, far below the rejection
    threshold. The gather is blocked so that peak memory stays bounded
    whatever the size of the map.
    """
    import warnings

    Z = np.asarray(Z, dtype=float)
    win = int(max(1, win))
    n = Z.shape[axis]
    if win <= 1 or n < 3:
        return Z.copy()
    win = min(win, n)
    half = win // 2
    if half < 1:
        return Z.copy()

    step = max(1, int(np.ceil((2 * half + 1) / float(max_samples))))
    offs = np.arange(-half, half + 1, step)
    if offs.size == 0:
        offs = np.array([0])
    idx = np.clip(np.arange(n)[:, None] + offs[None, :], 0, n - 1)

    M = np.moveaxis(Z, axis, -1)
    shape = M.shape
    flat = np.ascontiguousarray(M).reshape(-1, n)
    out = np.empty_like(flat)

    block = max(1, int(block_elems // max(1, n * offs.size)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a in range(0, flat.shape[0], block):
            b = min(flat.shape[0], a + block)
            out[a:b] = np.nanmedian(flat[a:b][:, idx], axis=-1)

    return np.moveaxis(out.reshape(shape), -1, axis)


def despike_map(Z, win_x_px, win_y_px, k=5.0, band=None, max_col_frac=0.50):
    """Replace compact outliers of the height map by a robust local baseline.

    Parameters
    ----------
    win_x_px, win_y_px : median window along the track / across it, in pixels.
        win_x_px must be clearly larger than the longest defect and clearly
        smaller than the length over which the real topography changes.
    k : rejection threshold, in robust sigma of the residual field.
    band : row slice of the track CORE, |y - y_c| < half, used by the guard.
    max_col_frac : a column in which more than this fraction of the core is
        flagged is left untouched. The discriminator is transverse coherence
        with respect to the track axis, not size: the frontal mound, the
        groove start and the lift-off step cover the axis over its full
        width, a debris patch covers a fraction of it and is usually off
        axis. Without this guard the despiker flattens the terminal mound,
        which is the one genuine feature that is compact along x.

    Returns (Z_clean, bad_mask, sigma_resid).
    """
    Z = np.asarray(Z, dtype=float)
    if k <= 0:
        return Z.copy(), np.zeros(Z.shape, dtype=bool), np.nan

    base = median_axis(Z, win_x_px, axis=1)
    if win_y_px > 1:
        base = median_axis(base, win_y_px, axis=0)

    resid = Z - base
    sig = robust_sigma(resid)
    if not np.isfinite(sig) or sig <= 0:
        return Z.copy(), np.zeros(Z.shape, dtype=bool), sig

    bad = np.isfinite(resid) & (np.abs(resid) > k * sig)

    if bad.any() and max_col_frac > 0:
        ny = Z.shape[0]
        lo, hi = (0, ny) if band is None else band
        lo, hi = max(0, int(lo)), min(ny, int(hi))
        sub = bad[lo:hi, :]
        den = np.maximum(1, np.isfinite(Z[lo:hi, :]).sum(axis=0))
        keep = (sub.sum(axis=0) / den) <= max_col_frac
        bad &= keep[None, :]

    out = np.where(bad, base, Z)
    return out, bad, sig


def hampel(v, win, k=4.0):
    """Hampel filter: replace |v - running median| > k * local MAD sigma.

    Self-protecting at a genuine step: the window then straddles both levels,
    the local MAD is of the order of the step itself, and nothing is rejected.
    """
    v = np.asarray(v, dtype=float)
    win = int(max(3, win))
    if v.size < 5:
        return v.copy()

    med = median_axis(v[None, :], win, axis=1)[0]
    resid = v - med
    loc = 1.4826 * median_axis(np.abs(resid)[None, :], win, axis=1)[0]
    floor = robust_sigma(resid)
    if not np.isfinite(floor) or floor <= 0:
        floor = np.nanstd(resid)
    if not np.isfinite(floor) or floor <= 0:
        return v.copy()
    sig = np.maximum(loc, 0.5 * floor)

    bad = np.isfinite(resid) & (np.abs(resid) > k * sig)
    return np.where(bad, med, v)


def polyfit2d(Z, mask, degree=2):
    """Least-squares 2-D polynomial fitted on `mask`, evaluated everywhere."""
    ny, nx = Z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    # normalise coordinates so the design matrix stays well conditioned
    u = (xx - nx / 2.0) / (nx / 2.0)
    v = (yy - ny / 2.0) / (ny / 2.0)

    terms = [(i, j) for i in range(degree + 1)
             for j in range(degree + 1 - i)]
    basis = [u ** i * v ** j for i, j in terms]

    use = mask & np.isfinite(Z)
    if use.sum() < len(terms) * 10:
        raise ValueError("not enough reference pixels for the form fit "
                         "(%d valid, %d needed)" % (use.sum(), len(terms) * 10))

    A = np.column_stack([b[use].ravel() for b in basis])
    coef, *_ = np.linalg.lstsq(A, Z[use].ravel(), rcond=None)

    form = np.zeros_like(Z)
    for c, b in zip(coef, basis):
        form += c * b
    return form


# ==========================================================================
# 3. geometry of the track
# ==========================================================================

def locate_track(Z, dy, smooth_y_px):
    """Return (row_centre, half_width_px, ridge_outer_px, profile) in pixels.

    Detection only -- these numbers select the reference bands, they are not
    used as measurements.

    The field is high-pass filtered along y by subtracting a long moving
    average taken column by column. This removes plate curvature and tilt,
    which on a transparent polymer can be several times larger than the
    groove and would otherwise capture the minimum at the edge of the field.
    The centre is then the minimum of the transverse median, searched in the
    central part of the field. The groove is delimited by the zero crossings
    on either side, and the pile-up crests just outside them set the limit of
    the region excluded from the form fit.
    """
    ny, nx = Z.shape

    long_win = max(9, ny // 3)
    base = np.apply_along_axis(moving_average, 0, Z, long_win)
    hp = Z - base

    prof = moving_average(np.nanmedian(hp, axis=1), max(3, smooth_y_px))

    # search the centre away from the borders, where the high-pass is biased
    edge_guard = max(3, ny // 10)
    core = slice(edge_guard, ny - edge_guard)
    c = edge_guard + int(np.nanargmin(prof[core]))

    # refine on the columns where the groove is actually present: averaging
    # over the whole field dilutes it when the load ramps up progressively
    w0 = max(3, ny // 10)
    with np.errstate(invalid="ignore"):
        depth_x = np.nanmin(hp[max(0, c - w0):c + w0 + 1, :], axis=0)
    sig = robust_sigma(depth_x)
    deep = np.isfinite(depth_x) & (depth_x < -2.0 * sig)
    if deep.sum() > 0.05 * nx:
        prof = moving_average(np.nanmean(hp[:, deep], axis=1),
                              max(3, smooth_y_px))
        c = edge_guard + int(np.nanargmin(prof[core]))

    prof = prof - np.nanmedian(prof)

    def edge(direction):
        i = c
        while 0 < i < ny - 1 and prof[i] < 0:
            i += direction
        return i

    i_lo, i_hi = edge(-1), edge(+1)
    half = max(3, int(0.5 * (i_hi - i_lo)))

    # ridge crests: highest point within two groove widths outside each edge
    span = max(half, 5)
    lo_a, lo_b = max(0, i_lo - 2 * span), max(1, i_lo)
    hi_a, hi_b = min(ny - 1, i_hi), min(ny, i_hi + 2 * span)
    r_lo = lo_a + int(np.nanargmax(prof[lo_a:lo_b])) if lo_b > lo_a else i_lo
    r_hi = hi_a + int(np.nanargmax(prof[hi_a:hi_b])) if hi_b > hi_a else i_hi

    # exclude out to 1.5 times the ridge offset: far enough to clear the
    # pile-up, close enough to leave usable reference bands on both sides
    outer = int(max(abs(c - r_lo), abs(r_hi - c)) * 1.5) + half
    outer = int(np.clip(outer, 2 * half + 2, 0.40 * ny))

    return c, half, outer, prof


def flatten(Z, row_c, outer, degree=2, n_iter=3, clip=3.0):
    """Remove the form, fitted on the lateral reference bands only."""
    ny, nx = Z.shape
    yy = np.arange(ny)[:, None] * np.ones((1, nx))
    ref = np.abs(yy - row_c) > outer

    mask = ref & np.isfinite(Z)
    form = None
    for _ in range(n_iter):
        form = polyfit2d(Z, mask, degree=degree)
        resid = Z - form
        sig = robust_sigma(resid[mask])
        if not np.isfinite(sig) or sig == 0:
            break
        mask = ref & np.isfinite(Z) & (np.abs(resid) < clip * sig)

    resid = Z - form
    sig = robust_sigma(resid[ref & np.isfinite(Z)])
    return resid, ref, mask, sig


# ==========================================================================
# 4. profiles and section values
# ==========================================================================

def section_values(prof_y, y_um, row_c, half, outer):
    """Depth, pile-up heights and areas of one transverse section.

    prof_y is a transverse profile already referenced to zero far from the
    track. Indices are pixels; areas are returned in um^2.
    """
    ny = prof_y.size
    dy = y_um[1] - y_um[0] if ny > 1 else 1.0

    lo = max(0, row_c - outer)
    hi = min(ny, row_c + outer + 1)
    win = slice(lo, hi)

    seg = prof_y[win]
    if not np.isfinite(seg).any():
        return None

    i_min = lo + int(np.nanargmin(seg))
    h_r = float(prof_y[i_min])

    g_lo = max(0, row_c - half)
    g_hi = min(ny, row_c + half + 1)

    left = prof_y[lo:g_lo]
    right = prof_y[g_hi:hi]
    h_l = float(np.nanmax(left)) if np.isfinite(left).any() else np.nan
    h_rt = float(np.nanmax(right)) if np.isfinite(right).any() else np.nan
    y_l = (lo + int(np.nanargmax(left))) * dy if np.isfinite(left).any() else np.nan
    y_r = (g_hi + int(np.nanargmax(right))) * dy if np.isfinite(right).any() else np.nan

    below = np.where(np.isfinite(seg) & (seg < 0), seg, 0.0)
    above = np.where(np.isfinite(seg) & (seg > 0), seg, 0.0)
    a_groove = float(-below.sum() * dy)
    a_pileup = float(above.sum() * dy)

    return {
        "y_min": y_um[i_min], "h_r": h_r,
        "h_p_left": h_l, "h_p_right": h_rt,
        "y_p_left": y_l, "y_p_right": y_r,
        "area_groove": a_groove, "area_pileup": a_pileup,
        "area_ratio": a_pileup / a_groove if a_groove > 0 else np.nan,
    }


def build_profiles(Zf, x_um, y_um, row_c, half, outer, sigma,
                   depth_threshold=3.0, med_y_px=1, hampel_win_px=0,
                   hampel_k=4.0, detect_win_px=1):           # [despike-patch]
    """Along-track profiles of h_r and of the two lateral pile-up heights.

    [despike-patch] The extrema are taken on a transversally median-filtered
    copy of the field, and the resulting profiles pass through a Hampel
    filter. A min over ~1000 rows has no breakdown point at all: the two
    filters give it one, without touching the definition of h_r and h_p.
    """
    ny, nx = Zf.shape
    lo, hi = max(0, row_c - outer), min(ny, row_c + outer + 1)
    g_lo, g_hi = max(0, row_c - half), min(ny, row_c + half + 1)

    Zq = median_axis(Zf, med_y_px, axis=0) if med_y_px > 1 else Zf

    with np.errstate(invalid="ignore"):
        # [despike-patch] original: extrema taken directly on Zf
        # h_r = np.nanmin(Zf[lo:hi, :], axis=0)
        # h_l = np.nanmax(Zf[lo:g_lo, :], axis=0) if g_lo > lo else np.full(nx, np.nan)
        # h_rt = np.nanmax(Zf[g_hi:hi, :], axis=0) if hi > g_hi else np.full(nx, np.nan)
        h_r = np.nanmin(Zq[lo:hi, :], axis=0)
        h_l = np.nanmax(Zq[lo:g_lo, :], axis=0) if g_lo > lo else np.full(nx, np.nan)
        h_rt = np.nanmax(Zq[g_hi:hi, :], axis=0) if hi > g_hi else np.full(nx, np.nan)

    if hampel_win_px and hampel_win_px > 2:                  # [despike-patch]
        h_r = hampel(h_r, hampel_win_px, hampel_k)
        h_l = hampel(h_l, hampel_win_px, hampel_k)
        h_rt = hampel(h_rt, hampel_win_px, hampel_k)

    # the groove exists where it is deeper than a few times the noise floor,
    # and over a contiguous stretch rather than a scattering of columns
    thr = -depth_threshold * sigma if np.isfinite(sigma) else -1.0
    # [measure-field-patch] detection on a smoothed copy, values on the raw
    # one: h_r is no longer low-passed along x, so a bare threshold test
    # would latch onto isolated negative excursions far ahead of the groove.
    h_det = moving_average(h_r, detect_win_px) if detect_win_px > 1 else h_r
    present = np.isfinite(h_det) & (h_det < thr)
    present = longest_run(present, close=max(5, nx // 100),
                          min_len=max(5, nx // 50))

    return {"h_r": h_r, "h_p_left": h_l, "h_p_right": h_rt,
            "present": present, "threshold": thr}


def longest_run(flag, close=0, min_len=0):
    """Keep only the longest contiguous True run, after closing small gaps.

    Isolated columns can dip below the detection threshold because of debris
    or a fringe-order artefact. Requiring a contiguous run avoids reporting a
    groove that starts hundreds of microns before the real one.
    """
    f = np.asarray(flag, dtype=bool).copy()
    if not f.any():
        return f

    if close > 0:                                  # bridge gaps <= close
        idx = np.nonzero(f)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= close + 1:
                f[a:b] = True

    edges = np.diff(np.r_[0, f.view(np.int8), 0])
    starts = np.nonzero(edges == 1)[0]
    stops = np.nonzero(edges == -1)[0]
    lengths = stops - starts
    k = int(np.argmax(lengths))

    out = np.zeros_like(f)
    if lengths[k] >= min_len:
        out[starts[k]:stops[k]] = True
    return out


def terminal_mound(Zf, x_um, row_c, half, present, tail_frac=0.25):
    """Height and position of the mound left at the lift-off point.

    Searched inside the track band, downstream of the last point where the
    groove is still present. If the groove runs to the edge of the field the
    mound was not scanned and None is returned.
    """
    ny, nx = Zf.shape
    if not present.any():
        return None
    i_end = int(np.max(np.nonzero(present)[0]))
    if i_end >= nx - 5:
        return None                      # groove leaves the field of view

    # [measure-field-patch] original: band was |y - y_c| < 2 * half, which
    # reaches the lateral ridges; their crest is comparable to the frontal
    # mound and can be returned instead of it.
    # band = Zf[max(0, row_c - 2 * half):min(ny, row_c + 2 * half + 1), :]
    band = Zf[max(0, row_c - half):min(ny, row_c + half + 1), :]
    with np.errstate(invalid="ignore"):
        crest = np.nanmax(band, axis=0)

    i_hi = min(nx, i_end + int(tail_frac * nx))
    seg = crest[i_end:i_hi]
    if not np.isfinite(seg).any():
        return None
    k = int(np.nanargmax(seg))
    return {"height": float(seg[k]), "x": float(x_um[i_end + k]),
            "x_groove_end": float(x_um[i_end]), "crest": crest}


def pick_sections(profiles, x_um, n):
    """Return n column indices: the deepest one plus a spread over the track."""
    present = profiles["present"]
    h_r = profiles["h_r"]
    if not present.any():
        idx = np.linspace(0, x_um.size - 1, n)
        return sorted(int(round(i)) for i in idx)

    i_deep = int(np.nanargmin(np.where(present, h_r, np.nan)))
    lo = int(np.min(np.nonzero(present)[0]))
    hi = int(np.max(np.nonzero(present)[0]))

    others = np.linspace(lo, hi, n)
    idx = {i_deep}
    for cand in others:
        cand = int(round(cand))
        # do not stack a section on top of the deepest one
        if all(abs(cand - k) > 0.03 * x_um.size for k in idx):
            idx.add(cand)
        if len(idx) == n:
            break
    while len(idx) < n:
        cand = int(np.random.randint(lo, hi + 1))
        if all(abs(cand - k) > 0.03 * x_um.size for k in idx):
            idx.add(cand)
    return sorted(idx)


# ==========================================================================
# 5. plotting
# ==========================================================================

def plot_all(res, title, out):
    Zf = res["Zf"]
    x_um, y_um = res["x_um"], res["y_um"]
    prof, sec_idx, sections = res["profiles"], res["sec_idx"], res["sections"]
    row_c, half, outer = res["row_c"], res["half"], res["outer"]
    dy = res["dy"]

    v = np.nanpercentile(np.abs(Zf), 99)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0],
                          hspace=0.42, wspace=0.24)

    # --- map --------------------------------------------------------------
    a = fig.add_subplot(gs[0, :])
    im = a.imshow(Zf, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                  extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
                  aspect="equal", interpolation="nearest")
    a.axhline((row_c - outer) * dy, color="0.3", lw=0.7, ls="--")
    a.axhline((row_c + outer) * dy, color="0.3", lw=0.7, ls="--")
    for i in sec_idx:
        a.axvline(x_um[i], color="k", lw=0.8, alpha=0.7)
    a.set_xlabel("x  [um]")
    a.set_ylabel("y  [um]")
    a.set_title("Height map, form removed on the lateral bands "
                "(dashed = reference limit, vertical = sections)")
    fig.colorbar(im, ax=a, label="Z  [um]", fraction=0.022, pad=0.01)

    # --- along-track profiles --------------------------------------------
    a = fig.add_subplot(gs[1, :])
    a.plot(x_um, prof["h_r_s"], lw=1.6, color="C0", label=r"$h_r$ groove floor")
    a.plot(x_um, prof["h_p_left_s"], lw=1.4, color="C3",
           label=r"$h_p$ pile-up, $y<y_c$")
    a.plot(x_um, prof["h_p_right_s"], lw=1.4, color="C1",
           label=r"$h_p$ pile-up, $y>y_c$")
    a.axhline(0, color="k", lw=0.6)
    a.axhline(prof["threshold"], color="0.5", lw=0.8, ls=":",
              label="detection threshold")
    # [frontal-curve-patch] the groove floor, continued past the lift-off
    # point on the track axis and recoloured where it becomes the frontal
    # pile-up. Drawn from the deepest point so that the blue part overlays
    # h_r and the eye reads one single curve.
    ha = prof.get("h_axis")   # unsmoothed: the mound apex must not be flattened
    if ha is not None and prof["present"].any():
        i_d = int(np.nanargmin(np.where(prof["present"], prof["h_r"], np.nan)))
        up = np.nonzero(ha[i_d:] >= 0.0)[0]
        if up.size:
            j = i_d + int(up[0])
            down = np.nonzero(ha[j:] < 0.0)[0]
            k = j + int(down[0]) if down.size else ha.size - 1
            a.plot(x_um[i_d:j + 1], ha[i_d:j + 1], lw=1.6, color="C0")
            a.plot(x_um[j:k + 1], ha[j:k + 1], lw=2.0, color="#8c2d04",
                   label=r"frontal pile-up (axial)")
            a.fill_between(x_um[j:k + 1], 0.0, ha[j:k + 1],
                           color="#8c2d04", alpha=0.15, lw=0)

    if res["mound"] is not None:
        m = res["mound"]
        a.plot([m["x"]], [m["height"]], "kv", ms=8,
               label="terminal mound  %.2f um" % m["height"])
    for i in sec_idx:
        a.axvline(x_um[i], color="k", lw=0.7, alpha=0.35)
    a.set_xlabel("x  [um]   (scratch direction)")
    a.set_ylabel("Z  [um]")
    a.set_title("Residual depth and lateral pile-up along the track")
    a.legend(fontsize=8.5, ncol=3)
    a.grid(alpha=0.3)

    # --- transverse sections ---------------------------------------------
    a = fig.add_subplot(gs[2, 0])
    cmap = plt.get_cmap("viridis")
    n = max(1, len(sec_idx) - 1)
    for k, (i, s) in enumerate(zip(sec_idx, sections)):
        lbl = "x = %.0f um" % x_um[i]
        if i == res["i_deep"]:
            lbl += "  (deepest)"
        a.plot(y_um, res["sec_profiles"][k], lw=1.5, color=cmap(k / n),
               label=lbl)
    a.axhline(0, color="k", lw=0.6)
    a.axvline(row_c * dy, color="k", lw=0.7, ls="--")
    a.set_xlabel("y  [um]")
    a.set_ylabel("Z  [um]")
    a.set_title("Transverse sections")
    a.legend(fontsize=7.5)
    a.grid(alpha=0.3)

    # --- area balance ------------------------------------------------------
    a = fig.add_subplot(gs[2, 1])
    xs = [x_um[i] for i in sec_idx]
    ag = [s["area_groove"] if s else np.nan for s in sections]
    ap = [s["area_pileup"] if s else np.nan for s in sections]
    w = 0.35 * (max(xs) - min(xs)) / max(1, len(xs))
    a.bar([u - w / 2 for u in xs], ag, width=w, color="C0", label="groove area")
    a.bar([u + w / 2 for u in xs], ap, width=w, color="C3", label="pile-up area")
    for u, g, p in zip(xs, ag, ap):
        if g and np.isfinite(g) and g > 0:
            a.text(u, max(g, p) * 1.03, "%.2f" % (p / g), ha="center",
                   fontsize=8)
    a.set_xlabel("x  [um]")
    a.set_ylabel("cross-section area  [um$^2$]")
    a.set_title("Displaced-material balance (label = pile-up / groove)")
    a.legend(fontsize=8.5)
    a.grid(alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=12.5)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ==========================================================================
# 6. driver
# ==========================================================================

def analyse(path, smooth_x=25.0, smooth_y=4.0, degree=2, n_sections=5,
            depth_threshold=3.0, outer_px=None,
            despike_k=5.0, despike_x=200.0, despike_y=9.0,   # [despike-patch]
            despike_col_frac=0.50, median_y=8.0,             # [despike-patch]
            hampel_len=120.0, hampel_k=4.0,                  # [despike-patch]
            axial_band=0.0):                        # [frontal-curve-patch]
    Z, hdr = read_bcrf(path)
    dx, dy = pixel_size(hdr)
    ny, nx = Z.shape

    win_x = max(1, int(round(smooth_x / dx)))
    win_y = max(1, int(round(smooth_y / dy)))

    # [despike-patch] reject the compact defects on the RAW map, before the
    # form fit: a debris patch sitting in a reference band also drags the
    # polynomial and inflates the noise floor.
    Z_raw = Z
    Z, bad, sig_spike = despike_map(
        Z,
        max(3, int(round(despike_x / dx))),
        max(1, int(round(despike_y / dy))),
        k=despike_k, max_col_frac=despike_col_frac)

    row_c, half, outer, _ = locate_track(Z, dy, win_y)
    if outer_px is not None:
        outer = int(outer_px)

    # [despike-patch] second pass, now that the track band is known, so the
    # column guard can distinguish a blob from a real transverse event
    if despike_k > 0:
        Z, bad2, sig_spike = despike_map(
            Z_raw,
            max(3, int(round(despike_x / dx))),
            max(1, int(round(despike_y / dy))),
            k=despike_k, max_col_frac=despike_col_frac,
            band=(max(0, row_c - half), min(ny, row_c + half + 1)))
        bad = bad2

    Zf, ref, mask, sigma = flatten(Z, row_c, outer, degree=degree)
    Zs = smooth2d(Zf, win_x, win_y)

    # [measure-field-patch] two fields from here on:
    #   Zs  smoothed along x -- map and display curves ONLY
    #   Zm  transverse median only -- everything that is measured
    # The moving average along x is a low-pass on the signal, and h_r, h_p
    # and the frontal mound are extrema of features whose curvature radius
    # is of the order of the window. It used to cost 1.2 um on the mound.
    Zm = median_axis(Zf, max(1, int(round(median_y / dy))), axis=0)

    x_um = np.arange(nx) * dx
    y_um = np.arange(ny) * dy

    # [measure-field-patch] original: prof = build_profiles(Zs, ...)
    prof = build_profiles(Zm, x_um, y_um, row_c, half, outer, sigma,
                          depth_threshold=depth_threshold,
                          med_y_px=1,        # already applied on Zm
                          hampel_win_px=int(round(hampel_len / dx)),
                          hampel_k=hampel_k,
                          detect_win_px=win_x)               # [despike-patch]
    # [frontal-curve-patch] axial profile on the track axis: the only
    # quantity that stays single-valued through the lift-off transition, and
    # therefore the only one that can carry the frontal mound continuously.
    # Zm already holds a transverse median, so axial_band = 0 (one row) is
    # already an average over median_y.
    w_ax = max(0, int(round(axial_band / dy)))
    with np.errstate(invalid="ignore"):
        prof["h_axis"] = np.nanmean(Zm[row_c - w_ax:row_c + w_ax + 1, :], axis=0)

    for key in ("h_r", "h_p_left", "h_p_right", "h_axis"):
        prof[key + "_s"] = moving_average(prof[key], win_x)

    # [measure-field-patch] original: terminal_mound(Zs, ...)
    mound = terminal_mound(Zm, x_um, row_c, half, prof["present"])

    sec_idx = pick_sections(prof, x_um, n_sections)
    i_deep = (int(np.nanargmin(np.where(prof["present"], prof["h_r"], np.nan)))
              if prof["present"].any() else sec_idx[0])

    sec_profiles, sections = [], []
    for i in sec_idx:
        a, b = max(0, i - win_x // 2), min(nx, i + win_x // 2 + 1)
        # [measure-field-patch] original: p = np.nanmean(Zs[:, a:b], axis=1)
        p = np.nanmean(Zm[:, a:b], axis=1)
        # re-zero on the CLIPPED reference rows: using the raw bands would
        # let a debris patch or an artefact drag the whole section
        sel = mask[:, a:b].any(axis=1)
        if sel.sum() < 10:
            sel = ref[:, i]
        p = p - np.nanmedian(p[sel])
        sec_profiles.append(p)
        sections.append(section_values(p, y_um, row_c, half, outer))

    return {
        "path": path, "hdr": hdr, "Z": Z, "Zf": Zs, "ref": ref,
        "Z_raw": Z_raw, "bad": bad, "sigma_spike": sig_spike,  # [despike-patch]
        "Zm": Zm,                                    # [measure-field-patch]
        "dx": dx, "dy": dy, "x_um": x_um, "y_um": y_um,
        "row_c": row_c, "half": half, "outer": outer, "sigma": sigma,
        "win_x": win_x, "win_y": win_y,
        "profiles": prof, "mound": mound,
        "sec_idx": sec_idx, "sections": sections,
        "sec_profiles": sec_profiles, "i_deep": i_deep,
    }


def describe(res):
    p, x = res["profiles"], res["x_um"]
    out = ["file            : %s" % os.path.basename(res["path"])]
    out.append("field           : %.0f x %.0f um  (%.4f um/px)"
               % (x[-1], res["y_um"][-1], res["dx"]))
    out.append("smoothing       : %.0f um along x (%d px), %.0f um along y (%d px)"
               "  [display only]"                     # [measure-field-patch]
               % (res["win_x"] * res["dx"], res["win_x"],
                  res["win_y"] * res["dy"], res["win_y"]))
    out.append("track centre    : y = %.1f um, half-width %.1f um"
               % (res["row_c"] * res["dy"], res["half"] * res["dy"]))
    out.append("reference bands : |y - yc| > %.1f um" % (res["outer"] * res["dy"]))
    out.append("noise floor     : %.2f um (robust sigma on the bands)" % res["sigma"])
    if "bad" in res:                                          # [despike-patch]
        nb = int(res["bad"].sum())
        out.append("defects removed : %d pixels (%.3f %% of the field), "
                   "residual sigma %.3f um"
                   % (nb, 100.0 * nb / res["bad"].size,
                      res.get("sigma_spike", float("nan"))))

    if p["present"].any():
        i0 = int(np.min(np.nonzero(p["present"])[0]))
        i1 = int(np.max(np.nonzero(p["present"])[0]))
        out.append("groove detected : x = %.0f -> %.0f um" % (x[i0], x[i1]))
    else:
        out.append("groove detected : NONE above %.2f um" % abs(p["threshold"]))

    if res["mound"] is not None:
        m = res["mound"]
        out.append("terminal mound  : %.2f um at x = %.0f um "
                   "(groove ends at %.0f um)"
                   % (m["height"], m["x"], m["x_groove_end"]))
        if "h_axis" in p:                        # [frontal-curve-patch]
            i_e = int(np.max(np.nonzero(p["present"])[0]))
            out.append("  on the axis   : %.2f um  (crest over the core band "
                       "is the value above)" % np.nanmax(p["h_axis"][i_e:]))
    else:
        out.append("terminal mound  : not in the field of view")

    out.append("")
    if not p["present"].any():
        out.append("  WARNING: no groove found above the noise floor. The "
                   "sections below are")
        out.append("  evenly spaced fallbacks and their values are noise, "
                   "not measurements.")
        out.append("")
    out.append("  %-10s %10s %10s %10s %10s %10s"
               % ("x [um]", "h_r", "h_p left", "h_p right", "A_groove", "A_pile/A_gr"))
    for i, s in zip(res["sec_idx"], res["sections"]):
        if s is None:
            continue
        tag = " *" if i == res["i_deep"] else "  "
        out.append("%s%-10.0f %10.2f %10.2f %10.2f %10.1f %10.2f"
                   % (tag, x[i], s["h_r"], s["h_p_left"], s["h_p_right"],
                      s["area_groove"], s["area_ratio"]))
    out.append("  (* = deepest section; h in um, areas in um^2)")
    return "\n".join(out)


def export(res, stem_dir):
    """Write the along-track profiles and the section table as CSV."""
    x = res["x_um"]
    p = res["profiles"]
    stem = os.path.splitext(os.path.basename(res["path"]))[0]

    f1 = os.path.join(stem_dir, stem + "_track.csv")
    with open(f1, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_um", "h_r_um", "h_p_left_um", "h_p_right_um",
                    "h_axis_um",                     # [frontal-curve-patch]
                    "groove_present"])
        for i in range(x.size):
            w.writerow(["%.4f" % x[i], "%.5g" % p["h_r_s"][i],
                        "%.5g" % p["h_p_left_s"][i], "%.5g" % p["h_p_right_s"][i],
                        "%.5g" % p["h_axis_s"][i],   # [frontal-curve-patch]
                        int(p["present"][i])])

    f2 = os.path.join(stem_dir, stem + "_sections.csv")
    with open(f2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_um", "is_deepest", "h_r_um", "h_p_left_um",
                    "h_p_right_um", "y_p_left_um", "y_p_right_um",
                    "area_groove_um2", "area_pileup_um2", "area_ratio"])
        for i, s in zip(res["sec_idx"], res["sections"]):
            if s is None:
                continue
            w.writerow(["%.2f" % x[i], int(i == res["i_deep"]),
                        "%.4f" % s["h_r"], "%.4f" % s["h_p_left"],
                        "%.4f" % s["h_p_right"], "%.2f" % s["y_p_left"],
                        "%.2f" % s["y_p_right"], "%.3f" % s["area_groove"],
                        "%.3f" % s["area_pileup"], "%.4f" % s["area_ratio"]])
    return f1, f2


def process(path, outdir=None, out=None, exportdir=None, **kw):
    res = analyse(path, **kw)
    stem = os.path.splitext(os.path.basename(path))[0]
    if out is None:
        out = os.path.join(outdir or os.path.dirname(path) or ".",
                           stem + "_topography.png")
    plot_all(res, stem, out)
    print(describe(res))
    print("figure          : %s" % out)
    if exportdir is not None:
        os.makedirs(exportdir, exist_ok=True)
        f1, f2 = export(res, exportdir)
        print("track csv       : %s" % f1)
        print("sections csv    : %s" % f2)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Residual depth and pile-up from a .bcrf scratch scan.")
    p.add_argument("bcrf", nargs="+", help="bcrf scan file(s)")
    p.add_argument("--out", default=None, help="output PNG (single input only)")
    p.add_argument("--outdir", default=None, help="directory for the figures")
    p.add_argument("--export", default=None,
                   help="directory for the CSV exports")
    p.add_argument("--smooth-x", type=float, default=25.0,
                   help="moving average along the track, in um (default 25)")
    p.add_argument("--smooth-y", type=float, default=4.0,
                   help="moving average across the track, in um (default 4)")
    p.add_argument("--sections", type=int, default=5,
                   help="number of transverse sections (default 5)")
    p.add_argument("--degree", type=int, default=2,
                   help="degree of the form polynomial (default 2)")
    p.add_argument("--threshold", type=float, default=3.0,
                   help="groove detection threshold, in noise sigma (default 3)")
    p.add_argument("--outer", type=int, default=None,
                   help="override the reference-band limit, in pixels")
    # ---------------- outlier rejection ----------------  [despike-patch]
    p.add_argument("--despike-k", type=float, default=5.0,
                   help="defect rejection threshold in robust sigma "
                        "(default 5; 0 disables the despiker)")
    p.add_argument("--despike-x", type=float, default=200.0,
                   help="baseline median window along the track, in um "
                        "(default 200; must exceed the longest defect)")
    p.add_argument("--despike-y", type=float, default=9.0,
                   help="baseline median window across the track, in um "
                        "(default 9)")
    p.add_argument("--despike-col-frac", type=float, default=0.50,
                   help="a column with more than this fraction of its band "
                        "flagged is left untouched (default 0.50)")
    p.add_argument("--median-y", type=float, default=8.0,
                   help="running median across the track before the h_r / "
                        "h_p extrema, in um (default 8)")
    p.add_argument("--hampel-len", type=float, default=120.0,
                   help="Hampel window on the along-track profiles, in um "
                        "(default 120; 0 disables it)")
    p.add_argument("--hampel-k", type=float, default=4.0,
                   help="Hampel threshold in local sigma (default 4)")
    p.add_argument("--axial-band", type=float, default=0.0,
                   help="half-width of the axial profile, in um either side "
                        "of the track axis (default 0: one row of the "
                        "transversally median-filtered field)")
    args = p.parse_args(argv)

    paths = []
    for pattern in args.bcrf:
        hits = sorted(glob.glob(pattern))
        paths.extend(hits if hits else [pattern])

    if args.out is not None and len(paths) > 1:
        p.error("--out cannot be used with several input files; use --outdir")
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    failed = 0
    for path in paths:
        if len(paths) > 1:
            print("=" * 70)
        try:
            process(path, outdir=args.outdir, out=args.out,
                    exportdir=args.export, smooth_x=args.smooth_x,
                    smooth_y=args.smooth_y, n_sections=args.sections,
                    degree=args.degree, depth_threshold=args.threshold,
                    outer_px=args.outer,
                    despike_k=args.despike_k,                # [despike-patch]
                    despike_x=args.despike_x,
                    despike_y=args.despike_y,
                    despike_col_frac=args.despike_col_frac,
                    median_y=args.median_y,
                    hampel_len=args.hampel_len,
                    hampel_k=args.hampel_k,
                    axial_band=args.axial_band)  # [frontal-curve-patch]
        except Exception as exc:
            failed += 1
            print("FAILED %s: %s: %s"
                  % (os.path.basename(path), type(exc).__name__, exc))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())