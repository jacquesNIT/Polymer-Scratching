"""Along-track curves for a simulated scratch: SCOF, w0, A_pile, A_groove.

    python sim_values.py <folder_or_csv> [options]

Options
    --csv <file>        write the sampled curves (one row per z sample)
    --png <file>        save the main figure
    --sections <file>   save the transverse-section figure
    --smooth <mm>       gaussian smoothing length of the plotted SCOF, in mm
    --smooth-x <um>     along-track moving average of the topography curves
    --rows / --cols     resampling grid of the topography field
    --no-show           do not open a window (useful on the cluster)
    --no-plot           compute and export only

WHAT THIS MODULE IS
    `scof.py` kept the point-by-point SCOF that `results_values.py` collapses
    into a band average. This module does the same for the topography: it
    keeps the whole along-track curve of every quantity the laboratory
    measures on a .bcrf scan, so that a simulation and a scan can be compared
    curve against curve rather than scalar against scalar.

SINGLE SOURCE OF TRUTH
    Nothing is reimplemented here. Two modules are imported and used as they
    are:

      results_values.py  parsing, time masks, RF2/CFN2 choice, node cloud to
                         regular grid, SCOF band bounds
      bcrf_values.py     track location, form removal, along-track profiles,
                         zero-crossing groove width, section areas

    Every topographic estimator is the LABORATORY estimator, called on the
    simulated field. A correction made in either module propagates here with
    no intervention. The only thing this file adds is the adaptation of the
    simulated node cloud to the array layout bcrf_values expects, plus the
    commanded-depth axis, which has no laboratory equivalent.

AXES AND UNITS
    The simulation works in mm; bcrf_values works in um. The field handed to
    the laboratory estimators is therefore in um, laid out exactly like a
    scan:

        field[i, j]   i = across the track (model x), j = along it (model z)

    Curves are reported against z in mm, the abscissa convention of scof.py
    and force_values: on the ACTIVE window z advances linearly from 0 to
    scratch_length. The half-model factor 2 cancels in the RF3/RF2 ratio; it
    is applied to the plotted forces only.

WHAT IS DELIBERATELY NOT COPIED FROM THE LABORATORY PIPELINE
    The despiking, Hampel and transverse-median stages of bcrf_values exist
    to survive dust, debris and surface roughness. A simulated field has
    none of those, and each of these filters costs a fraction of a um on the
    extrema of a feature whose curvature radius is of the order of the
    window. They are therefore OFF by default and exposed as options, so the
    same filters can be switched on when the point is to measure the bias
    the filters themselves introduce on the laboratory side.
"""

import argparse
import csv as _csv
import importlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


# =====================================================================
#  Imports of the two source modules
# =====================================================================
def _load_module(name, filenames):
    """Import `name`, first through sys.path then by explicit file search."""
    try:
        return importlib.import_module(name)
    except ImportError:
        pass

    here = Path(__file__).resolve().parent
    roots = [here, Path.cwd(), here.parent, here.parent / "AbaqusModel"]
    for root in roots:
        for fn in filenames:
            cand = root / fn
            if cand.is_file():
                spec = importlib.util.spec_from_file_location(name, str(cand))
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
                return mod
    raise ImportError(
        "%s not found. Put sim_values.py next to %s, or add its folder to "
        "PYTHONPATH." % (name, " / ".join(filenames))
    )


rv = _load_module("results_values", ["results_values.py"])
bv = _load_module("bcrf_values", ["bcrf_values.py"])


# Band bounds re-read from results_values so the two stay synchronised.
SCOF_LO_FRAC = rv.SCOF_LO_FRAC
SCOF_HI_FRAC = rv.SCOF_HI_FRAC

# Resampling grid of the node cloud. results_values uses TARGET_SHAPE =
# (80, 420), i.e. 7.6 um across the track: a 120 um groove is then 16 pixels
# wide and its two zero crossings are resolved to 6 % of w0 each. That is
# enough for a scalar h_r and far too coarse for an area. The default here is
# five times finer across the track; the cost is one griddata call.
DEFAULT_ROWS = 401           # across the track (model x)
DEFAULT_COLS = 420           # along the track (model z)

# Minimum |h_r| for a column to count as grooved, in um. bcrf_values derives
# its threshold from the scatter of the reference bands, which is a few nm on
# a simulated field: without a floor, the elastic sink-in ahead of the
# indenter would register as a groove.
DEFAULT_MIN_DEPTH_UM = 0.20

# Cosmetic along-track window, same default as bcrf_values.analyse().
DEFAULT_SMOOTH_X_UM = 25.0


# =====================================================================
#  Indenter geometry
# =====================================================================
def sphere_cone_transition_depth(metadata):
    """Depth h_t (mm) at which contact leaves the spherical cap.

    For a cone of half-angle theta measured from the axis, tangentially
    blended into a sphere of radius R, tangency is reached at
    h_t = R (1 - sin theta). Rockwell C (R = 0.2 mm, theta = 60 deg) gives
    h_t = 26.8 um. Returns None when the geometry cannot be read from the
    header.
    """
    R = metadata.get("tip_radius")
    theta = metadata.get("cone_angle")
    if R is None or theta is None:
        return None
    try:
        R = float(R)
        theta = float(theta)
    except (TypeError, ValueError):
        return None
    if R <= 0.0:
        return None  # equivalent pyramid: no cap, no transition
    return R * (1.0 - np.sin(np.radians(theta)))


def sphere_cone_transition_z(metadata, scratch_length):
    """Abscissa z (mm) of the sphere -> cone transition, progressive mode only.

    In depth_mode=progressive the depth command rises linearly with travel,
    so z_t = L * h_t / depth_max. In depth_mode=constant the depth is reached
    before the scratch starts, the transition has no abscissa along the
    groove, and the function returns None.
    """
    if str(metadata.get("depth_mode", "")).lower().startswith("prog") is False:
        return None
    h_t = sphere_cone_transition_depth(metadata)
    depth = metadata.get("scratch_depth")
    if h_t is None or depth is None:
        return None
    depth = abs(float(depth))
    if depth <= 0.0 or h_t >= depth:
        return None
    return scratch_length * h_t / depth


def commanded_depth(metadata, z_mm, scratch_length):
    """Commanded penetration (um, positive) at abscissa z.

    This is the one quantity the laboratory does not have: on a scan the
    depth has to be inverted from w0 through the recovery factor k(d),
    whereas here it is a boundary condition. Progressive mode gives
    d(z) = depth_max * z / L, clipped to the travel; constant mode gives the
    same depth everywhere the groove exists.

    Returned so that the simulated curves can be read against depth directly,
    and so that k(d) = a_res / a_geom can be recomputed -- which is exactly
    how the laboratory preset in bcrf_values was calibrated in the first
    place.
    """
    depth = metadata.get("scratch_depth")
    if depth is None:
        return None
    d_max = abs(float(depth)) * 1e3          # mm -> um
    z = np.asarray(z_mm, dtype=float)
    if str(metadata.get("depth_mode", "")).lower().startswith("prog"):
        L = float(scratch_length)
        if not np.isfinite(L) or L <= 0.0:
            return None
        return d_max * np.clip(z / L, 0.0, 1.0)
    return np.full(z.shape, d_max)


# =====================================================================
#  Forces and SCOF -- unchanged transcription of scof.py
# =====================================================================
def _interactive_backend():
    """Try to restore an interactive backend before showing a figure.

    bcrf_values sets matplotlib to "Agg" at import time, which is right for a
    batch tool but silently turns plt.show() into a no-op here. Returns True
    when a window can actually be opened.
    """
    import matplotlib
    if not matplotlib.get_backend().lower().startswith("agg"):
        return True
    for backend in ("QtAgg", "TkAgg", "MacOSX"):
        try:
            matplotlib.use(backend, force=True)
            return True
        except Exception:
            continue
    return False


def _nan_gaussian(y, sigma):
    """NaN-tolerant gaussian smoothing (normalised by the valid weight)."""
    from scipy.ndimage import gaussian_filter1d

    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if sigma <= 0.0 or not ok.any():
        return y.copy()
    filled = np.where(ok, y, 0.0)
    num = gaussian_filter1d(filled, sigma=sigma, mode="nearest")
    den = gaussian_filter1d(ok.astype(float), sigma=sigma, mode="nearest")
    out = np.full_like(y, np.nan)
    good = den > 1e-12
    out[good] = num[good] / den[good]
    out[~ok] = np.nan
    return out


def scof_curve(timeseries, metadata, smooth_mm=0.0):
    """Point-by-point SCOF along the groove.

    Returns a dict:
        z         (n,) abscissa along the scratch [mm]
        scof      (n,) |RF3| / |RF2|, NaN where RF2 vanishes
        scof_smooth (n,) same series smoothed when smooth_mm > 0
        F_n, F_t  (n,) FULL-model forces (factor 2) [N]
        in_load   (n,) bool, sample before the peak of the command
        in_band   (n,) bool, sample kept by the results_values average
        z_t       sphere -> cone transition [mm] or None
        SCOF_mean, SCOF_std, SCOF_n   band average, identical to scof_values
    """
    rf3 = timeseries.get("RF3")
    rf2, rf2_src = rv._normal_force_series(timeseries, metadata)
    if rf2 is None or rf3 is None:
        return None

    active = rv._active_mask(timeseries, metadata)
    if len(active) != len(rf2) or not active.any():
        return None

    idx = np.flatnonzero(active)

    # scratch_length is read from the header when present: a 3 mm run is
    # plotted over 3 mm without touching the results_values constant.
    L = metadata.get("scratch_length")
    try:
        L = float(L)
    except (TypeError, ValueError):
        L = float(rv.SCRATCH_LENGTH)
    if not np.isfinite(L) or L <= 0.0:
        L = float(rv.SCRATCH_LENGTH)

    z = np.linspace(0.0, L, len(idx))

    load = rv._loading_mask(timeseries, metadata)
    in_load = load[idx] if len(load) == len(active) else np.ones(len(idx), dtype=bool)
    if not in_load.any():
        in_load = np.ones(len(idx), dtype=bool)

    rf2_abs = np.abs(np.asarray(rf2, dtype=float))[idx]
    rf3_abs = np.abs(np.asarray(rf3, dtype=float))[idx]

    peak = float(np.nanmax(rf2_abs[in_load]))
    if not np.isfinite(peak) or peak <= 0.0:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        scof = np.where(rf2_abs > 0.0, rf3_abs / rf2_abs, np.nan)
    scof[~np.isfinite(scof)] = np.nan

    in_band = (
        in_load
        & (rf2_abs >= SCOF_LO_FRAC * peak)
        & (rf2_abs <= SCOF_HI_FRAC * peak)
        & np.isfinite(scof)
    )

    sigma = 0.0
    if smooth_mm and len(z) > 2:
        dz = float(np.median(np.diff(z)))
        if np.isfinite(dz) and dz > 0.0:
            sigma = max(float(smooth_mm) / dz, 1e-6)
    scof_s = _nan_gaussian(scof, sigma) if sigma > 0.0 else scof.copy()

    out = {
        "z": z,
        "scof": scof,
        "scof_smooth": scof_s,
        "F_n": 2.0 * rf2_abs,
        "F_t": 2.0 * rf3_abs,
        "in_load": in_load,
        "in_band": in_band,
        "scratch_length": L,
        "rf2_source": rf2_src,
        "z_t": sphere_cone_transition_z(metadata, L),
        "smooth_mm": float(smooth_mm or 0.0),
    }
    if in_band.any():
        band = scof[in_band]
        out["SCOF_mean"] = float(np.mean(band))
        out["SCOF_std"] = float(np.std(band))
        out["SCOF_n"] = int(band.size)
    return out


# =====================================================================
#  Topography field
# =====================================================================
def build_field(nodes, rows=DEFAULT_ROWS, cols=DEFAULT_COLS, method="linear"):
    """Node cloud -> levelling-ready field, laid out like a .bcrf map.

    Returns (field_um, along_um, across_um, z_mm) where

        field_um[i, j]   height in um, i across the track, j along it
        along_um         along-track abscissa in um, starting at 0
        across_um        across-track abscissa in um, starting at 0
        z_mm             along-track abscissa in mm, in model coordinates

    The heavy lifting is rv.map_coords_to_new_grid(): mirroring of the half
    model about x = 0, and interpolation of the DEFORMED coordinates onto a
    regular grid, i.e. a eulerian topographic map. Only the axis naming and
    the mm -> um conversion happen here, so that the array handed to
    bcrf_values has the same meaning as a scan: rows across the track,
    columns along it, heights referenced to the far field.
    """
    undef, deform = nodes["undeformed"], nodes["deformed"]
    if len(deform) == 0:
        return None
    coords = np.hstack([undef, deform])
    X, Y, Z = rv.map_coords_to_new_grid(coords, method=method,
                                        target_shape=(rows, cols))

    field_um = np.asarray(Y, dtype=float) * 1e3        # mm -> um

    # bcrf_values indexes with y_um = arange(ny)*dy, i.e. an axis starting at
    # zero. The model axis is centred on the symmetry plane, so it is shifted
    # here and the offset kept for display only.
    across_model = np.asarray(X[:, 0], dtype=float) * 1e3
    across_um = across_model - across_model[0]
    z_mm = np.asarray(Z[0, :], dtype=float)
    along_um = (z_mm - z_mm[0]) * 1e3

    return field_um, along_um, across_um, z_mm


def _section_areas(profile, y_um, row_c, half, outer, ref_rows, rezero=True):
    """Areas of one transverse column, through the laboratory estimator.

    bcrf_values computes areas on the few sections it picks for the figure;
    the calibration needs them on every column. The estimator is unchanged --
    bv.section_values is called as is -- and the re-zeroing on the clipped
    reference rows is the one bcrf_values.analyse() applies to its own
    sections (section_rezero=True).
    """
    p = np.asarray(profile, dtype=float)
    if rezero and ref_rows is not None and np.any(ref_rows):
        base = np.nanmedian(p[ref_rows])
        if np.isfinite(base):
            p = p - base
    if not np.isfinite(p).any():
        return None
    return bv.section_values(p, y_um, row_c, half, outer, penetration=False)


def topography_curves(field_um, along_um, across_um, metadata, z_mm,
                      smooth_x=DEFAULT_SMOOTH_X_UM, degree=2,
                      outer_px=None, track_offset=None, centre_guard=0.25,
                      locate_win=4.0, detect_win=25.0,
                      min_depth_um=DEFAULT_MIN_DEPTH_UM,
                      median_y=0.0, hampel_len=0.0, hampel_k=4.0,
                      upstream=None, n_sections=5, section_win=25.0,
                      section_rezero=True):
    """Along-track curves of w0, h_r, h_p, A_groove, A_pile and their ratio.

    The chain is bcrf_values.analyse() with the instrument-artefact stages
    made optional:

        locate_track  -> flatten  -> [median / hampel] -> build_profiles
                      -> section_values on every column

    Returns a dict of arrays indexed by grid column, plus the detection
    geometry (row_c, half, outer) and the picked sections.
    """
    ny, nx = field_um.shape
    dy = float(across_um[1] - across_um[0]) if ny > 1 else 1.0
    dx = float(along_um[1] - along_um[0]) if nx > 1 else 1.0

    win_x = max(1, int(round(smooth_x / dx))) if smooth_x > 0 else 1
    win_det = max(1, int(round(detect_win / dx)))
    win_loc = max(1, int(round(locate_win / dy)))
    win_sec = max(1, int(round(section_win / dx)))

    # --- track geometry. On a half model the centre is known exactly, but it
    # is detected the same way as on a scan so that any asymmetry introduced
    # by the interpolation shows up as a measurable offset rather than being
    # assumed away. The offset is reported; --track-x forces the plane.
    row_sym = int(round(0.5 * (ny - 1)))     # mirror plane, model x = 0
    row_c, half, outer, _ = bv.locate_track(field_um, dy, win_loc,
                                            centre_guard=centre_guard)
    if track_offset is not None:
        row_c = int(np.clip(row_sym + round(float(track_offset) / dy),
                            0, ny - 1))
    if outer_px is not None:
        outer = int(outer_px)
    centre_offset_um = (row_c - row_sym) * dy

    # --- form removal. On a simulated field the far surface is flat by
    # construction, so the polynomial comes out near zero; the step is kept
    # because it is what DEFINES the zero from which the crossings of w0 and
    # the sign split of the areas are taken. Skipping it would silently give
    # the two sides different references.
    if upstream is None:
        up_cols = bv.upstream_limit(field_um, row_c, half)
    elif upstream <= 0:
        up_cols = 0
    else:
        up_cols = int(round(upstream / dx))

    Zf, ref, mask, sigma = bv.flatten(field_um, row_c, outer, degree=degree,
                                      up_cols=up_cols)

    # --- optional laboratory filters, off by default (see the module header)
    Zm = Zf
    if median_y > 0:
        Zm = bv.median_axis(Zf, max(1, int(round(median_y / dy))), axis=0)

    prof = bv.build_profiles(
        Zm, along_um, across_um, row_c, half, outer, sigma,
        depth_threshold=3.0, med_y_px=1,
        hampel_win_px=int(round(hampel_len / dx)) if hampel_len > 0 else 0,
        hampel_k=hampel_k, detect_win_px=win_det, up_cols=up_cols)

    # --- absolute depth floor on the presence flag. build_profiles scales its
    # threshold on the scatter of the reference bands; that scatter is a few
    # nm here, so the flag would follow the elastic sink-in ahead of the tip.
    if min_depth_um > 0:
        keep = prof["present"] & (prof["h_r"] < -abs(min_depth_um))
        prof["present"] = bv.longest_run(keep, close=max(5, nx // 100),
                                         min_len=max(5, nx // 50))
        prof["w0"] = np.where(prof["present"], prof["w0"], np.nan)
        prof["threshold"] = min(prof["threshold"], -abs(min_depth_um))

    # --- areas, column by column, through the laboratory section estimator
    a_groove = np.full(nx, np.nan)
    a_pileup = np.full(nx, np.nan)
    for j in np.flatnonzero(prof["present"]):
        sec = _section_areas(Zm[:, j], across_um, row_c, half, outer,
                             ref[:, j], rezero=section_rezero)
        if sec is None:
            continue
        a_groove[j] = sec["area_groove"]
        a_pileup[j] = sec["area_pileup"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a_groove > 0, a_pileup / a_groove, np.nan)

    # --- commanded depth and the lateral recovery factor it implies.
    # k = a_res / a_geom is the quantity the bcrf_values presets encode; on a
    # simulation both terms are known, so the preset can be regenerated from
    # here instead of being trusted.
    L = metadata.get("scratch_length")
    try:
        L = float(L)
    except (TypeError, ValueError):
        L = float(rv.SCRATCH_LENGTH)
    if not np.isfinite(L) or L <= 0.0:
        L = float(rv.SCRATCH_LENGTH)

    d_cmd = commanded_depth(metadata, z_mm, L)
    k_lat = np.full(nx, np.nan)
    if d_cmd is not None:
        R = float(metadata.get("tip_radius", bv.TIP_RADIUS_UM * 1e-3)) * 1e3
        ang = float(metadata.get("cone_angle", bv.TIP_HALF_ANGLE_DEG))
        ok = np.isfinite(prof["w0"]) & (d_cmd > 0)
        if ok.any():
            a_geom = bv.contact_radius(d_cmd[ok], R, ang)
            with np.errstate(divide="ignore", invalid="ignore"):
                k_lat[ok] = np.where(a_geom > 0,
                                     0.5 * prof["w0"][ok] / a_geom, np.nan)

    # --- cosmetic along-track smoothing, same convention as bcrf_values
    out = {
        "z_mm": z_mm,
        "d_cmd_um": d_cmd if d_cmd is not None else np.full(nx, np.nan),
        "h_r": prof["h_r"], "h_p_left": prof["h_p_left"],
        "h_p_right": prof["h_p_right"], "w0": prof["w0"],
        "area_groove": a_groove, "area_pileup": a_pileup, "area_ratio": ratio,
        "k_lateral": k_lat,
        "present": prof["present"], "threshold": prof["threshold"],
    }
    with np.errstate(invalid="ignore"):
        out["h_p"] = np.nanmean(np.vstack([prof["h_p_left"],
                                           prof["h_p_right"]]), axis=0)
    for key in ("h_r", "h_p", "h_p_left", "h_p_right", "w0",
                "area_groove", "area_pileup", "area_ratio", "k_lateral"):
        out[key + "_s"] = (bv.moving_average(out[key], win_x) if win_x > 1
                           else out[key].copy())

    # --- transverse sections, same picker as the laboratory figure
    sec_idx = bv.pick_sections(prof, along_um, n_sections,
                               deep_margin=0.05,
                               deep_win_px=max(1, int(round(100.0 / dx))))
    sec_profiles, sections = [], []
    for i in sec_idx:
        a, b = max(0, i - win_sec // 2), min(nx, i + win_sec // 2 + 1)
        p = np.nanmean(Zm[:, a:b], axis=1)
        if section_rezero:
            sel = mask[:, a:b].any(axis=1)
            if sel.sum() < 10:
                sel = ref[:, i]
            base = np.nanmedian(p[sel])
            if np.isfinite(base):
                p = p - base
        sec_profiles.append(p)
        sections.append(bv.section_values(p, across_um, row_c, half, outer,
                                          penetration=False))

    out.update({
        "field": Zm, "ref": ref, "sigma": sigma,
        "along_um": along_um, "across_um": across_um,
        "dx": dx, "dy": dy, "row_c": row_c, "half": half, "outer": outer,
        "row_sym": row_sym, "centre_offset_um": centre_offset_um,
        "up_cols": up_cols, "win_x": win_x,
        "sec_idx": sec_idx, "sections": sections, "sec_profiles": sec_profiles,
        "scratch_length": L,
    })
    if prof["present"].any():
        out["i_deep"] = int(np.nanargmin(np.where(prof["present"],
                                                  prof["h_r"], np.nan)))
    else:
        out["i_deep"] = None
    return out


# =====================================================================
#  One file in, one result dict out
# =====================================================================
def analyse(filepath, smooth_mm=0.0, rows=DEFAULT_ROWS, cols=DEFAULT_COLS,
            method="linear", **topo_kw):
    """Read a *_Results.csv and return forces, SCOF and topography curves.

    The force series and the topography live on different samplings -- one
    per written frame, one per grid column -- so the forces are projected
    onto the topography abscissa. That projection is what makes SCOF and w0
    readable at the SAME depth, which is the whole point of the exercise.
    """
    metadata, timeseries, nodes = rv.parse_results_csv(str(filepath))
    curve = scof_curve(timeseries, metadata, smooth_mm=smooth_mm)

    built = build_field(nodes, rows=rows, cols=cols, method=method)
    topo = None
    if built is not None:
        field_um, along_um, across_um, z_mm = built
        topo = topography_curves(field_um, along_um, across_um, metadata,
                                 z_mm, **topo_kw)

    res = {"file": Path(filepath).name, "path": str(filepath),
           "metadata": metadata, "scof": curve, "topo": topo}

    # Forces interpolated onto the topography grid, NaN outside the active
    # window: no extrapolation, a column with no force is left empty.
    if curve is not None and topo is not None:
        zt = topo["z_mm"]
        for key in ("F_n", "F_t", "scof", "scof_smooth"):
            res.setdefault("on_grid", {})[key] = np.interp(
                zt, curve["z"], curve[key], left=np.nan, right=np.nan)
    return res


# =====================================================================
#  Outputs
# =====================================================================
_CURVE_COLUMNS = [
    ("z [mm]", "z_mm"),
    ("d_cmd [um]", "d_cmd_um"),
    ("F_n [N]", None),
    ("F_t [N]", None),
    ("SCOF [-]", None),
    ("w0 [um]", "w0"),
    ("w0_s [um]", "w0_s"),
    ("h_r [um]", "h_r"),
    ("h_p [um]", "h_p"),
    ("h_p_left [um]", "h_p_left"),
    ("h_p_right [um]", "h_p_right"),
    ("A_groove [um^2]", "area_groove"),
    ("A_pile [um^2]", "area_pileup"),
    ("A_ratio [-]", "area_ratio"),
    ("k_lateral [-]", "k_lateral"),
    ("present", "present"),
]


def write_curves_csv(results, out_csv):
    """Long format: one row per (file, grid column)."""
    parent = Path(out_csv).parent
    if str(parent):
        os.makedirs(str(parent), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["file"] + [h for h, _ in _CURVE_COLUMNS])
        for res in results:
            topo = res.get("topo")
            if topo is None:
                continue
            grid = res.get("on_grid", {})
            n = topo["z_mm"].size
            for i in range(n):
                row = [res["file"]]
                for head, key in _CURVE_COLUMNS:
                    if key is None:
                        src = {"F_n [N]": "F_n", "F_t [N]": "F_t",
                               "SCOF [-]": "scof"}[head]
                        v = grid.get(src, np.full(n, np.nan))[i]
                    else:
                        v = topo[key][i]
                    if isinstance(v, (bool, np.bool_)):
                        row.append(int(v))
                    elif v is None or not np.isfinite(v):
                        row.append("")
                    else:
                        row.append("%.6g" % v)
                w.writerow(row)
    print("Wrote %d curve set(s) to %s" % (len(results), out_csv))


def plot_curves(results, png=None, show=True):
    """Four panels sharing the z abscissa: forces, SCOF, widths, areas."""
    import matplotlib
    if show:
        show = _interactive_backend()
    else:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 11.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.2, 1.2, 1.2]})
    ax_f, ax_s, ax_w, ax_a = axes
    single = len(results) == 1

    for res in results:
        label = res["file"]
        curve, topo = res.get("scof"), res.get("topo")

        if curve is not None:
            z = curve["z"]
            (ln,) = ax_f.plot(z, curve["F_n"], lw=1.2,
                              label="F_n" if single else "F_n -- " + label)
            colour = ln.get_color()
            ax_f.plot(z, curve["F_t"], lw=1.2, ls="--", color=colour,
                      label="F_t" if single else "F_t -- " + label)

            if curve["smooth_mm"] > 0.0:
                ax_s.plot(z, curve["scof"], lw=0.8, alpha=0.30, color=colour)
                ax_s.plot(z, curve["scof_smooth"], lw=1.6, color=colour,
                          label=label if not single
                          else "SCOF (smoothed %.3g mm)" % curve["smooth_mm"])
            else:
                ax_s.plot(z, curve["scof"], lw=1.2, color=colour,
                          label=label if not single else "SCOF")

            if single and curve["in_band"].any():
                zb = z[curve["in_band"]]
                ax_s.axvspan(float(zb.min()), float(zb.max()), color="0.88",
                             zorder=0,
                             label="band [%.0f %%, %.0f %%] of the F_n peak"
                                   % (100 * SCOF_LO_FRAC, 100 * SCOF_HI_FRAC))
                if "SCOF_mean" in curve:
                    ax_s.axhline(curve["SCOF_mean"], color="k", lw=1.0, ls=":",
                                 label="SCOF_mean = %.4f" % curve["SCOF_mean"])
            if curve.get("z_t") is not None:
                for ax in axes:
                    ax.axvline(curve["z_t"], color="0.4", lw=1.0, ls="-.")
        else:
            colour = None

        if topo is None:
            continue
        zt = topo["z_mm"]
        m = topo["present"]
        ax_w.plot(zt[m], topo["w0_s"][m], lw=1.5, color=colour,
                  label="w0" if single else "w0 -- " + label)
        ax_w.plot(zt[m], -topo["h_r_s"][m] * 4.0, lw=1.0, ls="--", color=colour,
                  label="4 x |h_r|" if single else None)
        ax_w.plot(zt[m], topo["h_p_s"][m] * 4.0, lw=1.0, ls=":", color=colour,
                  label="4 x h_p" if single else None)

        ax_a.plot(zt[m], topo["area_groove_s"][m], lw=1.4, color=colour,
                  label="A_groove" if single else "A_groove -- " + label)
        ax_a.plot(zt[m], topo["area_pileup_s"][m], lw=1.4, ls="--",
                  color=colour, label="A_pile" if single else None)

    ax_f.set_ylabel("Force [N]")
    ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=8, loc="upper left")

    ax_s.set_ylabel("SCOF = F_t / F_n [-]")
    ax_s.grid(alpha=0.3)
    ax_s.legend(fontsize=8, loc="lower right")

    # The ratio diverges at run-in, where F_n is still near zero; bound the
    # scale on the retained band so the rest of the curve stays readable.
    bands = [r["scof"]["scof"][r["scof"]["in_band"]] for r in results
             if r.get("scof") is not None and r["scof"]["in_band"].any()]
    if bands:
        vals = np.concatenate(bands)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        pad = 0.35 * max(hi - lo, 1e-3)
        ax_s.set_ylim(max(0.0, lo - pad), hi + pad)

    ax_w.set_ylabel("width / height [um]")
    ax_w.grid(alpha=0.3)
    ax_w.legend(fontsize=8, loc="upper left")

    ax_a.set_ylabel("area [um^2]")
    ax_a.set_xlabel("z along the scratch [mm]")
    ax_a.grid(alpha=0.3)
    ax_a.legend(fontsize=8, loc="upper left")

    # The ratio carries the dilatancy and deserves its own axis.
    ax_r = ax_a.twinx()
    for res in results:
        topo = res.get("topo")
        if topo is None:
            continue
        m = topo["present"]
        ax_r.plot(topo["z_mm"][m], topo["area_ratio_s"][m], lw=1.0,
                  color="0.35", alpha=0.8)
    ax_r.set_ylabel("A_pile / A_groove [-]", color="0.35")
    ax_r.tick_params(axis="y", labelcolor="0.35")

    lengths = [r["scof"]["scratch_length"] for r in results
               if r.get("scof") is not None]
    lengths += [r["topo"]["scratch_length"] for r in results
                if r.get("topo") is not None]
    if lengths:
        ax_a.set_xlim(0.0, max(lengths))

    if single:
        fig.suptitle(results[0]["file"], fontsize=10)
    fig.tight_layout()

    if png:
        parent = Path(png).parent
        if str(parent):
            os.makedirs(str(parent), exist_ok=True)
        fig.savefig(png, dpi=150)
        print("Wrote figure to %s" % png)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_sections(results, png=None, show=True):
    """Transverse profiles at the picked sections, centred on the track.

    Same layout as the laboratory section figure, so a scan and a simulation
    can be laid side by side without re-centring anything by hand.
    """
    import matplotlib
    if show:
        show = _interactive_backend()
    else:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for res in results:
        topo = res.get("topo")
        if topo is None:
            continue
        y_c = topo["across_um"][topo["row_c"]]
        yy = topo["across_um"] - y_c
        for p, i in zip(topo["sec_profiles"], topo["sec_idx"]):
            d = topo["d_cmd_um"][i]
            lab = "%s  z=%.2f mm" % (res["file"], topo["z_mm"][i])
            if np.isfinite(d):
                lab += "  d_cmd=%.1f um" % d
            ax.plot(yy, p, lw=1.1, label=lab)

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("y - y_c [um]")
    ax.set_ylabel("height [um]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()

    if png:
        parent = Path(png).parent
        if str(parent):
            os.makedirs(str(parent), exist_ok=True)
        fig.savefig(png, dpi=150)
        print("Wrote section figure to %s" % png)
    if show:
        plt.show()
    else:
        plt.close(fig)


def describe(res):
    """One console line per file, read at the deepest section."""
    name = res["file"]
    curve, topo = res.get("scof"), res.get("topo")
    scof_mean = curve.get("SCOF_mean", np.nan) if curve else np.nan

    if topo is None or topo["i_deep"] is None:
        print("%-38s SCOF_mean=%-8s topography unavailable"
              % (name, "%.4f" % scof_mean if np.isfinite(scof_mean) else "n/a"))
        return

    def _f(v, fmt="%.2f"):
        return fmt % v if v is not None and np.isfinite(v) else "n/a"

    i = topo["i_deep"]
    print("%-38s SCOF_mean=%s" % (name, _f(scof_mean, "%.4f")))
    print("    deepest section at z = %s mm   commanded depth = %s um"
          % (_f(topo["z_mm"][i], "%.3f"), _f(topo["d_cmd_um"][i])))
    print("    w0 = %s um   h_r = %s um   h_p = %s um   k_lat = %s"
          % (_f(topo["w0"][i]), _f(topo["h_r"][i]), _f(topo["h_p"][i]),
             _f(topo["k_lateral"][i], "%.3f")))
    print("    A_groove = %s um^2   A_pile = %s um^2   ratio = %s"
          % (_f(topo["area_groove"][i], "%.1f"),
             _f(topo["area_pileup"][i], "%.1f"),
             _f(topo["area_ratio"][i], "%.3f")))
    print("    track centre offset from the symmetry plane: %+.2f um "
          "(bands: half=%d px, outer=%d px)"
          % (topo["centre_offset_um"], topo["half"], topo["outer"]))


# =====================================================================
#  CLI
# =====================================================================
def main(argv=None):
    p = argparse.ArgumentParser(
        description="SCOF, w0, A_pile and A_groove curves along a simulated "
                    "scratch, using the bcrf_values estimators.")
    p.add_argument("target", nargs="?", default=None,
                   help="a *_Results.csv file or a folder containing some")
    p.add_argument("--csv", dest="out_csv", default=None,
                   help="export of the sampled curves")
    p.add_argument("--png", dest="png", default=None, help="main figure")
    p.add_argument("--sections", dest="sections_png", default=None,
                   help="transverse-section figure")
    p.add_argument("--smooth", dest="smooth", type=float, default=0.0,
                   help="gaussian smoothing of the plotted SCOF, in mm")
    p.add_argument("--smooth-x", dest="smooth_x", type=float,
                   default=DEFAULT_SMOOTH_X_UM,
                   help="along-track moving average of the topography curves, "
                        "in um (cosmetic; 0 disables)")
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                   help="grid rows across the track (default %d)" % DEFAULT_ROWS)
    p.add_argument("--cols", type=int, default=DEFAULT_COLS,
                   help="grid columns along the track (default %d)" % DEFAULT_COLS)
    p.add_argument("--method", default="linear",
                   choices=("linear", "nearest", "cubic"),
                   help="griddata interpolation of the node cloud")
    p.add_argument("--min-depth", dest="min_depth", type=float,
                   default=DEFAULT_MIN_DEPTH_UM,
                   help="absolute |h_r| floor for a column to count as "
                        "grooved, in um (default %.2f)" % DEFAULT_MIN_DEPTH_UM)
    p.add_argument("--track-offset", dest="track_offset", type=float,
                   default=None,
                   help="force the track centre this many um away from the "
                        "symmetry plane (default: detected like on a scan)")
    p.add_argument("--outer-px", dest="outer_px", type=int, default=None,
                   help="force the outer limit of the excluded band, in pixels")
    p.add_argument("--degree", type=int, default=2,
                   help="degree of the form polynomial (default 2)")
    p.add_argument("--sections-n", dest="n_sections", type=int, default=5,
                   help="number of transverse sections (default 5)")
    p.add_argument("--median-y", dest="median_y", type=float, default=0.0,
                   help="transverse median window in um; OFF by default, "
                        "it is a spike filter and simulated fields have no "
                        "spikes")
    p.add_argument("--hampel", dest="hampel_len", type=float, default=0.0,
                   help="along-track Hampel window in um; OFF by default, "
                        "same reason")
    p.add_argument("--no-rezero", dest="section_rezero", action="store_false",
                   help="do not re-zero each section on its reference rows")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="do not open a window")
    p.add_argument("--no-plot", dest="plot", action="store_false",
                   help="compute and export only")
    args = p.parse_args(argv)

    target = Path(args.target) if args.target else Path.cwd()
    files = ([target] if target.is_file()
             else sorted(target.rglob("*_Results.csv")))
    if not files:
        print("No *_Results.csv found in %s" % target)
        return 1

    results = []
    for f in files:
        try:
            res = analyse(
                f, smooth_mm=args.smooth, rows=args.rows, cols=args.cols,
                method=args.method, smooth_x=args.smooth_x,
                degree=args.degree, outer_px=args.outer_px,
                track_offset=args.track_offset, min_depth_um=args.min_depth,
                median_y=args.median_y, hampel_len=args.hampel_len,
                n_sections=args.n_sections,
                section_rezero=args.section_rezero)
        except Exception as exc:                      # keep the batch alive
            print("%-38s FAILED: %s" % (Path(f).name, exc))
            continue
        if res.get("scof") is None and res.get("topo") is None:
            print("%-38s nothing usable (no forces, no nodes)" % Path(f).name)
            continue
        results.append(res)
        describe(res)

    if not results:
        return 1
    if args.out_csv:
        write_curves_csv(results, args.out_csv)
    if args.plot:
        plot_curves(results, png=args.png, show=args.show)
        if args.sections_png or args.show:
            plot_sections(results, png=args.sections_png, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())