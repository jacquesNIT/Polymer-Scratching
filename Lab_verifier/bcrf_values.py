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

def locate_track(Z, dy, smooth_y_px, centre_guard=0.25):
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
    # [anchor-patch] original: edge_guard = max(3, ny // 10), then the centre
    # was simply the deepest row. On a material whose pile-up reaches the
    # reference bands the form fit tilts and leaves a dark band against the
    # edge of the field, deeper than the groove; the centre locked onto it.
    # Search the central part of the field only: on these scans the
    # artefact sits against the edge. The operator centres the scratch in
    # the field, so restricting the search to the central part is enough and
    # stays predictable; --centre-guard widens or narrows it.
    edge_guard = max(3, int(round(centre_guard * ny)))
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
        c = edge_guard + int(np.nanargmin(prof[core]))    # [anchor-patch]

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
    # [rough-patch] original: outer = int(np.clip(outer, 2*half+2, 0.40*ny))
    # 0.40 * ny is a fraction of the FIELD and ignores where the centre sits.
    # When the track is off-centre it leaves one reference band a few rows
    # tall, and the form fit then rests on one side only. Clip by the
    # distance to each edge as well, keeping ny/10 rows on both sides.
    room = min(c, ny - 1 - c) - max(5, ny // 10)
    outer = int(np.clip(outer, 2 * half + 2, max(2 * half + 2,
                                                 min(0.40 * ny, room))))

    return c, half, outer, prof


def upstream_limit(Z, row_c, half, k=5.0, margin_frac=0.03):
    """[upstream-patch] Last column before the track disturbs the surface.

    Form-independent: the field is high-passed along y, and each column is
    given a disturbance level -- the 95th percentile of |Z| across three
    groove widths. Upstream of the scratch that level is the roughness of
    the material; it rises as soon as the groove or the pile-up appears.
    Returns 0 when no quiet zone exists, which is the honest answer on a
    rough surface.
    """
    ny, nx = Z.shape
    base = np.apply_along_axis(moving_average, 0, Z, max(9, ny // 3))
    hp = Z - base
    lo, hi = max(0, row_c - 3 * half), min(ny, row_c + 3 * half + 1)
    with np.errstate(invalid="ignore"):
        d = np.nanpercentile(np.abs(hp[lo:hi, :]), 95, axis=0)
    d = moving_average(d, max(3, nx // 100))

    quiet = np.nanpercentile(d, 10)
    scat = robust_sigma(d[d <= np.nanpercentile(d, 30)])
    if not np.isfinite(scat) or scat <= 0:
        return 0
    bad = np.isfinite(d) & (d > quiet + k * scat)
    if not bad.any():
        return 0
    i0 = int(np.nonzero(bad)[0].min() - margin_frac * nx)
    return i0 if i0 > 0.03 * nx else 0


def flatten(Z, row_c, outer, degree=2, n_iter=3, clip=3.0, up_cols=0):
    """Remove the form, fitted on the lateral reference bands.

    [upstream-patch] When an undisturbed upstream zone exists, its columns
    are added to the reference over their full height. On a narrow field
    that is the only support the polynomial has across the middle.
    """
    ny, nx = Z.shape
    yy = np.arange(ny)[:, None] * np.ones((1, nx))
    ref = np.abs(yy - row_c) > outer
    if up_cols > 0:                                       # [upstream-patch]
        ref = ref | (np.arange(nx)[None, :] < int(up_cols))

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

def ridge_band_px(half, outer):                     # [consistency-patch]
    """Outer limit of the lateral pile-up search, in pixels.

    Wide enough for the crest to migrate outward as the load ramps up,
    narrow enough not to pick up the waviness of a rough surface. Used by
    build_profiles() and section_values() alike -- they used to disagree.
    """
    return int(max(2 * half, min(outer, 3 * half)))


def section_values(prof_y, y_um, row_c, half, outer):
    """Depth, pile-up heights and areas of one transverse section.

    prof_y is a transverse profile already referenced to zero far from the
    track. Indices are pixels; areas are returned in um^2.
    """
    ny = prof_y.size
    dy = y_um[1] - y_um[0] if ny > 1 else 1.0

    # [consistency-patch] the areas keep the full band, but the depth and
    # the crests are now searched exactly where build_profiles() searches
    # them: the floor in the track core, the ridges out to ridge_band_px.
    lo = max(0, row_c - outer)
    hi = min(ny, row_c + outer + 1)
    win = slice(lo, hi)

    seg = prof_y[win]
    if not np.isfinite(seg).any():
        return None

    g_lo = max(0, row_c - half)
    g_hi = min(ny, row_c + half + 1)

    core = prof_y[g_lo:g_hi]
    i_min = g_lo + int(np.nanargmin(core))
    h_r = float(prof_y[i_min])

    r_px = ridge_band_px(half, outer)
    r_lo = max(0, row_c - r_px)
    r_hi = min(ny, row_c + r_px + 1)

    left = prof_y[r_lo:g_lo]
    right = prof_y[g_hi:r_hi]
    h_l = float(np.nanmax(left)) if np.isfinite(left).any() else np.nan
    h_rt = float(np.nanmax(right)) if np.isfinite(right).any() else np.nan
    y_l = (r_lo + int(np.nanargmax(left))) * dy if np.isfinite(left).any() else np.nan
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
                   hampel_k=4.0, detect_win_px=1,            # [despike-patch]
                   up_cols=0):                               # [upstream-patch]
    """Along-track profiles of h_r and of the two lateral pile-up heights.

    [despike-patch] The extrema are taken on a transversally median-filtered
    copy of the field, and the resulting profiles pass through a Hampel
    filter. A min over ~1000 rows has no breakdown point at all: the two
    filters give it one, without touching the definition of h_r and h_p.
    """
    ny, nx = Zf.shape
    # [rough-patch] original:
    #   lo, hi = max(0, row_c - outer), min(ny, row_c + outer + 1)
    # h_r was the minimum over the WHOLE band, up to 240 um off axis. The
    # expected minimum of N samples of a field of scatter s is about -3s, so
    # on a wavy surface that alone reads -15 um where there is no groove.
    # The floor can only be inside the track core; the ridges can only be
    # just outside it.
    lo, hi = max(0, row_c - half), min(ny, row_c + half + 1)
    g_lo, g_hi = lo, hi
    # [consistency-patch] original band was min(2*half, outer). Too narrow:
    # under progressive loading the ridge crest migrates outward, and past
    # mid-track it leaves the window, so h_p reads a flank instead of the
    # crest (-5.4 um at x = 2195 um on PC_20N). min(outer, 3*half) keeps the
    # crest and costs 0.1 um of extra baseline on a rough surface.
    r_lo = max(0, row_c - ridge_band_px(half, outer))
    r_hi = min(ny, row_c + ridge_band_px(half, outer) + 1)

    Zq = median_axis(Zf, med_y_px, axis=0) if med_y_px > 1 else Zf

    with np.errstate(invalid="ignore"):
        # [despike-patch] original: extrema taken directly on Zf
        # h_r = np.nanmin(Zf[lo:hi, :], axis=0)
        # h_l = np.nanmax(Zf[lo:g_lo, :], axis=0) if g_lo > lo else np.full(nx, np.nan)
        # h_rt = np.nanmax(Zf[g_hi:hi, :], axis=0) if hi > g_hi else np.full(nx, np.nan)
        h_r = np.nanmin(Zq[lo:hi, :], axis=0)
        # [rough-patch] ridges searched in half < |y - y_c| < 2*half
        h_l = (np.nanmax(Zq[r_lo:g_lo, :], axis=0) if g_lo > r_lo
               else np.full(nx, np.nan))
        h_rt = (np.nanmax(Zq[g_hi:r_hi, :], axis=0) if r_hi > g_hi
                else np.full(nx, np.nan))

    if hampel_win_px and hampel_win_px > 2:                  # [despike-patch]
        h_r = hampel(h_r, hampel_win_px, hampel_k)
        h_l = hampel(h_l, hampel_win_px, hampel_k)
        h_rt = hampel(h_rt, hampel_win_px, hampel_k)

    # the groove exists where it is deeper than a few times the noise floor,
    # and over a contiguous stretch rather than a scattering of columns
    # [rough-patch] The threshold used to be depth_threshold * sigma,
    # sigma being the scatter of the RAW reference bands. But h_r is not a
    # sample of the field, it is the minimum of one over ~2*half rows, and
    # the expected minimum of N samples of scatter s sits near -3s. On a
    # rough surface that bias is -15 um and the test becomes meaningless.
    # So run the SAME estimator on unscratched material -- a core-sized band
    # placed in the reference region on both sides -- and read the bias and
    # the scatter of h_r directly off it.
    # [upstream-patch] Prefer the upstream columns: same estimator, same y,
    # on material the indenter never touched. The lateral placement below is
    # only a fallback -- on a wide pile-up it lands on the ridge itself and
    # returns the ridge height as a "bias".
    ref_rows = []
    if up_cols > 20:
        ref_rows.append(h_r[:int(up_cols)])
    else:
        c_off = int(0.5 * (half + outer))
        for c_ref in (row_c - c_off, row_c + c_off):
            a0, a1 = c_ref - half, c_ref + half + 1
            if a0 >= 0 and a1 <= ny:
                ref_rows.append(np.nanmin(Zq[a0:a1, :], axis=0))
    if ref_rows:
        h_r_ref = np.concatenate(ref_rows)
        bias_r = float(np.nanmedian(h_r_ref))
        scat_r = float(robust_sigma(h_r_ref))
        if not np.isfinite(scat_r) or scat_r <= 0:
            scat_r = sigma
        thr = bias_r - depth_threshold * scat_r
    else:
        bias_r, scat_r = 0.0, sigma
        thr = -depth_threshold * sigma if np.isfinite(sigma) else -1.0
    # [measure-field-patch] detection on a smoothed copy, values on the raw
    # one: h_r is no longer low-passed along x, so a bare threshold test
    # would latch onto isolated negative excursions far ahead of the groove.
    h_det = moving_average(h_r, detect_win_px) if detect_win_px > 1 else h_r
    present = np.isfinite(h_det) & (h_det < thr)
    present = longest_run(present, close=max(5, nx // 100),
                          min_len=max(5, nx // 50))

    return {"bias_r": bias_r, "scatter_r": scat_r,      # [rough-patch]
            "h_r": h_r, "h_p_left": h_l, "h_p_right": h_rt,
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


def pick_sections(profiles, x_um, n, deep_margin=0.05, deep_win_px=1):
    """Return n column indices, anchored on the deepest point of the track.

    [anchor-patch] The first section is the deepest point of the scratch,
    which under progressive loading is reached near its end. It is searched
    on a smoothed h_r and away from the edges of the image, where imaging
    artefacts and the lift-off ramp both live. The other n-1 are stepped
    uniformly to the LEFT of it, down the loading ramp.
    """
    present = profiles["present"]
    h_r = profiles["h_r"]
    nx = x_um.size
    if not present.any():
        idx = np.linspace(0, nx - 1, n)
        return sorted(int(round(i)) for i in idx)

    lo = int(np.min(np.nonzero(present)[0]))
    hi = int(np.max(np.nonzero(present)[0]))

    h_s = moving_average(h_r, deep_win_px) if deep_win_px > 1 else h_r
    m = int(round(max(0.0, deep_margin) * nx))
    a, b = max(lo, m), min(hi, nx - 1 - m)
    if b <= a:
        a, b = lo, hi
    win = np.where(present[a:b + 1], h_s[a:b + 1], np.nan)
    if not np.isfinite(win).any():
        win = h_s[a:b + 1]
    i_deep = a + int(np.nanargmin(win))

    # step to the left of the deepest point, over the groove; if the groove
    # is too short for n distinct sections, reach further left in the image
    min_step = 0.02 * nx
    left = min(lo, i_deep - (n - 1) * min_step)
    left = int(max(0, round(left)))
    idx = sorted({int(round(v)) for v in np.linspace(left, i_deep, n)})
    return idx

    # ------------------------------------------------------------------
    # [anchor-patch] original body kept for reference
    # i_deep = int(np.nanargmin(np.where(present, h_r, np.nan)))
    # lo = int(np.min(np.nonzero(present)[0]))
    # hi = int(np.max(np.nonzero(present)[0]))

    # [picksections-patch] The separation is set by the span actually
    # detected, not by the field width. Asking for n sections spread over the
    # groove and then forbidding them to sit closer than 3 % of a field ten
    # times longer is a contradiction, and the old random retry loop below
    # spun forever whenever it happened.
    #
    #   others = np.linspace(lo, hi, n)
    #   idx = {i_deep}
    #   for cand in others:
    #       cand = int(round(cand))
    #       if all(abs(cand - k) > 0.03 * x_um.size for k in idx):
    #           idx.add(cand)
    #       if len(idx) == n:
    #           break
    #   while len(idx) < n:                      # <-- unbounded, could not exit
    #       cand = int(np.random.randint(lo, hi + 1))
    #       if all(abs(cand - k) > 0.03 * x_um.size for k in idx):
    #           idx.add(cand)
    #   return sorted(idx)

    span = max(1, hi - lo)
    sep = max(1.0, min(0.03 * x_um.size, 0.5 * span / float(max(1, n - 1))))

    idx = {i_deep}
    for cand in np.linspace(lo, hi, n):
        cand = int(round(cand))
        # do not stack a section on top of the deepest one
        if all(abs(cand - k) > sep for k in idx):
            idx.add(cand)
        if len(idx) >= n:
            break

    # bounded deterministic fill, relaxing the separation once. If n
    # well-separated positions do not exist, return fewer: analyse() and
    # plot_all() both work off len(sec_idx).
    if len(idx) < n:
        for cand in np.linspace(lo, hi, 8 * n):
            cand = int(round(cand))
            if cand in idx:
                continue
            if all(abs(cand - k) > max(1.0, 0.5 * sep) for k in idx):
                idx.add(cand)
            if len(idx) >= n:
                break

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
            axial_band=0.0,                         # [frontal-curve-patch]
            detect_win=25.0, section_win=25.0,             # [no-smooth-patch]
            locate_win=4.0,                               # [no-smooth-patch]
            measure_win=0.0,                                   # [rough-patch]
            centre_guard=0.25, track_y=None,                   # [anchor-patch]
            deep_margin=0.05, deep_win=100.0,                  # [anchor-patch]
            upstream=None,                                  # [upstream-patch]
            section_rezero=True):                       # [consistency-patch]
    Z, hdr = read_bcrf(path)
    dx, dy = pixel_size(hdr)
    ny, nx = Z.shape

    # [no-smooth-patch] win_x / win_y are now COSMETIC ONLY. The detection
    # and the section width have their own windows so that --no-smooth
    # cannot silently change what is measured.
    win_x = max(1, int(round(smooth_x / dx)))
    win_y = max(1, int(round(smooth_y / dy)))
    win_det = max(1, int(round(detect_win / dx)))
    win_sec = max(1, int(round(section_win / dx)))
    win_loc = max(1, int(round(locate_win / dy)))

    # [despike-patch] reject the compact defects on the RAW map, before the
    # form fit: a debris patch sitting in a reference band also drags the
    # polynomial and inflates the noise floor.
    Z_raw = Z
    Z, bad, sig_spike = despike_map(
        Z,
        max(3, int(round(despike_x / dx))),
        max(1, int(round(despike_y / dy))),
        k=despike_k, max_col_frac=despike_col_frac)

    # [no-smooth-patch] original: locate_track(Z, dy, win_y)
    row_c, half, outer, _ = locate_track(Z, dy, win_loc,      # [anchor-patch]
                                        centre_guard=centre_guard)
    if track_y is not None:                                  # [anchor-patch]
        row_c = int(np.clip(round(track_y / dy), 0, ny - 1))
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

    # [upstream-patch] undisturbed columns, before the indenter came down
    if upstream is None:
        up_cols = upstream_limit(Z, row_c, half)
    elif upstream <= 0:
        up_cols = 0
    else:
        up_cols = int(round(upstream / dx))

    Zf, ref, mask, sigma = flatten(Z, row_c, outer, degree=degree,
                                   up_cols=up_cols)         # [upstream-patch]
    Zs = smooth2d(Zf, win_x, win_y)

    # [measure-field-patch] two fields from here on:
    #   Zs  smoothed along x -- map and display curves ONLY
    #   Zm  transverse median only -- everything that is measured
    # The moving average along x is a low-pass on the signal, and h_r, h_p
    # and the frontal mound are extrema of features whose curvature radius
    # is of the order of the window. It used to cost 1.2 um on the mound.
    Zm = median_axis(Zf, max(1, int(round(median_y / dy))), axis=0)

    # [rough-patch] optional along-track average of the MEASUREMENT field.
    # The waviness of a rough polymer decorrelates over a few tens of um of
    # x while the groove is invariant over hundreds, so averaging along the
    # track is the only filter that separates them. Off by default: it
    # flattens the groove end and the frontal mound, which is why the
    # measure-field patch took it out of the default path.
    Zm_raw = Zm
    if measure_win > 0:
        Zm = smooth2d(Zm, max(1, int(round(measure_win / dx))), 1)
        # The frontal mound is only ~150 um long: the full averaging window
        # would erase it (22.4 -> 5.1 um, checked on PMMAGS_20N). It keeps a
        # short window of its own, enough to tame the waviness.
        Zm_raw = smooth2d(Zm_raw, max(1, int(round(min(measure_win, 60.0) / dx))), 1)

    x_um = np.arange(nx) * dx
    y_um = np.arange(ny) * dy

    # [measure-field-patch] original: prof = build_profiles(Zs, ...)
    prof = build_profiles(Zm, x_um, y_um, row_c, half, outer, sigma,
                          depth_threshold=depth_threshold,
                          med_y_px=1,        # already applied on Zm
                          hampel_win_px=int(round(hampel_len / dx)),
                          hampel_k=hampel_k,
                          detect_win_px=win_det,             # [no-smooth-patch]
                          up_cols=up_cols)                   # [upstream-patch]
    # [frontal-curve-patch] axial profile on the track axis: the only
    # quantity that stays single-valued through the lift-off transition, and
    # therefore the only one that can carry the frontal mound continuously.
    # Zm already holds a transverse median, so axial_band = 0 (one row) is
    # already an average over median_y.
    w_ax = max(0, int(round(axial_band / dy)))
    with np.errstate(invalid="ignore"):
        prof["h_axis"] = np.nanmean(               # [rough-patch] Zm_raw
            Zm_raw[row_c - w_ax:row_c + w_ax + 1, :], axis=0)

    for key in ("h_r", "h_p_left", "h_p_right", "h_axis"):
        prof[key + "_s"] = moving_average(prof[key], win_x)

    # [measure-field-patch] original: terminal_mound(Zs, ...)
    mound = terminal_mound(Zm_raw, x_um, row_c, half,   # [rough-patch]
                           prof["present"])

    sec_idx = pick_sections(prof, x_um, n_sections,          # [anchor-patch]
                            deep_margin=deep_margin,
                            deep_win_px=max(1, int(round(deep_win / dx))))
    i_deep = (int(np.nanargmin(np.where(prof["present"], prof["h_r"], np.nan)))
              if prof["present"].any() else sec_idx[0])

    sec_profiles, sections = [], []
    for i in sec_idx:
        # [no-smooth-patch] original: half-width was win_x // 2
        a, b = max(0, i - win_sec // 2), min(nx, i + win_sec // 2 + 1)
        # [measure-field-patch] original: p = np.nanmean(Zs[:, a:b], axis=1)
        p = np.nanmean(Zm[:, a:b], axis=1)
        # re-zero on the CLIPPED reference rows: using the raw bands would
        # let a debris patch or an artefact drag the whole section
        if section_rezero:                       # [consistency-patch]
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
        "up_cols": up_cols,                          # [upstream-patch]
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
    if res["win_x"] <= 1 and res["win_y"] <= 1:       # [no-smooth-patch]
        out.append("smoothing       : off (--no-smooth); nothing measured "
                   "went through it either way")
    else:
        out.append("smoothing       : %.0f um along x (%d px), "
                   "%.0f um along y (%d px)  [display only]"
                   % (res["win_x"] * res["dx"], res["win_x"],
                      res["win_y"] * res["dy"], res["win_y"]))
    out.append("track centre    : y = %.1f um, half-width %.1f um"
               % (res["row_c"] * res["dy"], res["half"] * res["dy"]))
    out.append("reference bands : |y - yc| > %.1f um" % (res["outer"] * res["dy"]))
    if res["outer"] < 3 * res["half"]:                        # [anchor-patch]
        out.append("  WARNING       : that is only %.1f groove half-widths "
                   "from the axis. The pile-up probably reaches into the "
                   "reference bands, so the form fit is biased. Widen the "
                   "scan across the track, or set --outer / --track-y."
                   % (res["outer"] / float(max(1, res["half"]))))
    if res.get("up_cols", 0) > 0:                          # [upstream-patch]
        out.append("upstream zone   : x < %.0f um, used as reference over "
                   "the full height" % (res["up_cols"] * res["dx"]))
    else:
        out.append("upstream zone   : none found; reference is the lateral "
                   "bands only")
    out.append("noise floor     : %.2f um (robust sigma on the bands)" % res["sigma"])
    if "bias_r" in res["profiles"]:                          # [rough-patch]
        out.append("h_r estimator    : bias %+.2f um, scatter %.2f um, "
                   "measured on unscratched material with the same estimator"
                   % (res["profiles"]["bias_r"], res["profiles"]["scatter_r"]))
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
        if (i1 - i0) < 0.2 * x.size:             # [picksections-patch]
            out.append("  WARNING       : only %.0f %% of the field. The "
                       "presence test sits at %.2f um, %.1f x the %.2f um "
                       "noise floor -- on a rough surface, lower --threshold."
                       % (100.0 * (i1 - i0) / x.size, p["threshold"],
                          abs(p["threshold"]) / max(1e-9, res["sigma"]),
                          res["sigma"]))
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
        if res["profiles"].get("scatter_r", 0.0) > 1.5:      # [rough-patch]
            out.append("  The h_r estimator scatters by %.1f um on this "
                       "surface: the waviness is comparable to the scratch. "
                       "Try --rough."
                       % res["profiles"]["scatter_r"])
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
    # ------------------ smoothing ------------------  [no-smooth-patch]
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smooth", dest="no_smooth", action="store_false",
                   help="smooth the map and the display curves (default)")
    g.add_argument("--no-smooth", dest="no_smooth", action="store_true",
                   help="draw the map and the profile curves raw. Cosmetic "
                        "only: no measured quantity passes through the "
                        "smoothing any more")
    p.set_defaults(no_smooth=False)
    p.add_argument("--detect-win", type=float, default=25.0,
                   help="window used to test h_r against the depth threshold, "
                        "in um (default 25; 0 = raw h_r, fires early)")
    p.add_argument("--section-win", type=float, default=25.0,
                   help="columns averaged into one transverse section, in um "
                        "(default 25)")
    p.add_argument("--measure-win", type=float, default=None,
                   help="along-track average applied to the measurement "
                        "field before the h_r / h_p extrema, in um "
                        "(default 0). Use 200-400 on a rough surface")
    p.add_argument("--rough", action="store_true",
                   help="preset for materials whose waviness is comparable "
                        "to the scratch (cast PMMA GS): --median-y 40 "
                        "--measure-win 300 --threshold 2")
    p.add_argument("--no-section-rezero", dest="section_rezero",
                   action="store_false",
                   help="do not re-zero each transverse section on its own "
                        "reference rows; the table then matches the plotted "
                        "curves exactly")
    p.set_defaults(section_rezero=True)
    p.add_argument("--upstream", type=float, default=None,
                   help="length of the undisturbed upstream zone, in um, "
                        "used as form reference and to calibrate the "
                        "estimators (default: detected; 0 disables it)")
    p.add_argument("--centre-guard", type=float, default=0.25,
                   help="fraction of the field excluded on each side when "
                        "searching the track centre (default 0.25)")
    p.add_argument("--track-y", type=float, default=None,
                   help="force the track centre, in um; overrides detection")
    p.add_argument("--deep-margin", type=float, default=0.05,
                   help="fraction of the image excluded at each end when "
                        "searching the deepest point (default 0.05)")
    p.add_argument("--deep-win", type=float, default=100.0,
                   help="smoothing applied to h_r before the deepest point "
                        "is located, in um (default 100)")
    p.add_argument("--locate-win", type=float, default=4.0,
                   help="transverse smoothing used to locate the track, in um "
                        "(default 4)")
    p.add_argument("--axial-band", type=float, default=0.0,
                   help="half-width of the axial profile, in um either side "
                        "of the track axis (default 0: one row of the "
                        "transversally median-filtered field)")
    args = p.parse_args(argv)

    if args.measure_win is None:                             # [rough-patch]
        args.measure_win = 300.0 if args.rough else 0.0
    if args.rough:
        if args.median_y == 8.0:
            args.median_y = 40.0
        if args.threshold == 3.0:
            args.threshold = 2.0

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
                    exportdir=args.export,            # [no-smooth-patch]
                    smooth_x=0.0 if args.no_smooth else args.smooth_x,
                    smooth_y=0.0 if args.no_smooth else args.smooth_y,
                    detect_win=args.detect_win,
                    section_win=args.section_win,
                    locate_win=args.locate_win,
                    upstream=args.upstream,               # [upstream-patch]
                    section_rezero=args.section_rezero,  # [consistency-patch]
                    centre_guard=args.centre_guard,           # [anchor-patch]
                    track_y=args.track_y,
                    deep_margin=args.deep_margin,
                    deep_win=args.deep_win,
                    measure_win=args.measure_win,                # [rough-patch]
                    n_sections=args.sections,
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