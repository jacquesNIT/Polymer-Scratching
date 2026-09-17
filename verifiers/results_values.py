"""Scalar results (energies, forces, SCOF, residual topography) from Abaqus *_Results.csv files.

Usage: python results_values.py <folder_or_csv> [<output_csv>] [<z_mm>]
       python results_values.py run_001_Results.csv values.csv
"""

import csv as _csv
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# Grid constants (copied from ScratchFeatures/constants.py)
# TARGET_SHAPE          : (rows, columns) of the regular resampling grid.
# SCRATCH_LENGTH        : commanded scratch length (mm).
# SCRATCH_DOMAIN_WIDTH  : full domain width after mirroring (mm).
# SCRATCH_DOMAIN_LENGTH : total meshed domain length (mm).
TARGET_SHAPE = (80, 420)
SCRATCH_LENGTH = 2.0
SCRATCH_DOMAIN_WIDTH = 0.6
SCRATCH_DOMAIN_LENGTH = 2.5

# Relative tolerance of the time windows (Abaqus writes float32 time stamps).
TIME_TOL = 1e-6

# Location of the topography measurement section
# Z_SEARCH_LO_FRAC   : lower search bound, as a fraction of SCRATCH_LENGTH.
# Z_LOCATE_SMOOTH_MM : gaussian smoothing length (mm) used to locate the minimum.
# Z_REFINE_HALF_MM   : half-width (mm) of the window where the minimum is re-read on the raw profile.
Z_SEARCH_LO_FRAC = 0.25
Z_LOCATE_SMOOTH_MM = 0.06
Z_REFINE_HALF_MM = 0.06

# Normal-force band (fraction of the peak) used for the SCOF average
SCOF_LO_FRAC = 0.10
SCOF_HI_FRAC = 0.90

# End-of-travel force estimate
# FORCE_FIT_Z_FRAC  : fit window length, as a fraction of SCRATCH_LENGTH, from the end of travel.
# FORCE_FIT_MIN_PTS : below this number of points, the window median is used instead of the fit.
FORCE_FIT_Z_FRAC = 0.20
FORCE_FIT_MIN_PTS = 5

# Relative tolerance used to detect frames at full command.
LOAD_PEAK_TOL = 5e-3


# CSV parsing (copied from results_verifier.parse_results_csv)
def parse_results_csv(filepath):
    """Parse the post-processor CSV.
    Returns (metadata, timeseries, nodes).
    """
    metadata = {}
    header_cols = []
    ts_rows = []
    node_labels, node_undef, node_def = [], [], []

    with open(filepath, "r", encoding="latin-1") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().replace("\r", "")
        if not line:
            continue

        if line.startswith("#"):
            if "WallclockTime=" in line:
                m = re.search(r"WallclockTime=([\d\.eE+-]+)", line)
                if m:
                    metadata["wallclock"] = float(m.group(1))
            if "Material parameters:" in line or "Material:" in line:
                for m in re.finditer(r"(\w+)=([\d\.eE+-]+)", line):
                    try:
                        metadata[m.group(1)] = float(m.group(2))
                    except ValueError:
                        pass
            if "tip radius" in line.lower():
                m = re.search(r"tip radius\s*([\d\.eE+-]+)\s*mm", line, re.IGNORECASE)
                if m:
                    metadata["tip_radius"] = float(m.group(1))
                m = re.search(r"cone angle\s*([\d\.eE+-]+)", line, re.IGNORECASE)
                if m:
                    metadata["cone_angle"] = float(m.group(1))
            if "Simulation Parameters" in line:
                body = line.split("Parameters:", 1)[1]
                for k, v in re.findall(r"(\w+)=([A-Za-z0-9\.eE+-]+)", body):
                    try:
                        metadata[k] = float(v)
                    except ValueError:
                        metadata[k] = v

            if (line.count("=") == 1 and "parameters:" not in line.lower()
                    and "WallclockTime" not in line):
                m = re.match(r"#\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", line)
                if m:
                    key, val = m.group(1), m.group(2)
                    try:
                        metadata[key] = float(val)
                    except ValueError:
                        metadata[key] = val
            continue

        if "Time" in line and "RF1" in line:
            header_cols = [c.strip() for c in line.split(",")]
            continue

        parts = line.split(",")
        if len(parts) < 2 or not header_cols:
            continue

        if parts[0].strip():
            try:
                row = {}
                for ci, col in enumerate(header_cols):
                    if ci < len(parts) and parts[ci].strip():
                        row[col] = float(parts[ci])
                ts_rows.append(row)
            except ValueError:
                pass

        label_idx = header_cols.index("NodeLabel") if "NodeLabel" in header_cols else 7
        if label_idx < len(parts) and parts[label_idx].strip():
            try:
                node_labels.append(int(float(parts[label_idx])))
                node_undef.append([float(parts[label_idx + i]) for i in (1, 2, 3)])
                node_def.append([float(parts[label_idx + i]) for i in (4, 5, 6)])
            except (ValueError, IndexError):
                pass

    timeseries = {}
    if ts_rows:
        all_cols = set()
        for row in ts_rows:
            all_cols.update(row.keys())
        for col in all_cols:
            timeseries[col] = np.array([row.get(col, 0.0) for row in ts_rows])

    nodes = {
        "labels": np.array(node_labels),
        "undeformed": np.array(node_undef) if node_undef else np.empty((0, 3)),
        "deformed": np.array(node_def) if node_def else np.empty((0, 3)),
    }

    return metadata, timeseries, nodes


# Helpers (copied from results_verifier)
WM_BALANCE_TERMS = ("WM_ALLIE", "WM_ALLVD", "WM_ALLFD", "WM_ALLKE",
                    "WM_ALLWK", "WM_ALLPW", "WM_ALLCW", "WM_ALLMW")


def _peak(x):
    """Maximum absolute value of a series, 0.0 if missing or empty."""
    return float(np.max(np.abs(x))) if x is not None and len(x) else 0.0


def _active_end(metadata, t_last):
    """End of the active phase (indentation + scratch), capped at t_last."""
    st = float(metadata.get("scratch_time", 0.0) or 0.0)
    it = float(metadata.get("indentation_time", 0.0) or 0.0)
    constant = not str(metadata.get("depth_mode", "")).lower().startswith("prog")
    t_act = st + (it if constant else 0.0)
    if t_act <= 0.0:
        return t_last
    return min(t_last, t_act)


def _normal_force_series(timeseries, metadata):
    """Normal-force series and its source (RF2 or CFN2, depending on the control mode)."""
    rf2 = timeseries.get("RF2")
    cfn2 = timeseries.get("CFN2")
    control_mode = str(metadata.get("control_mode", "displacement"))

    def _nonzero(arr):
        return arr is not None and len(arr) and float(np.max(np.abs(arr))) > 1e-20

    if control_mode == "force":
        if _nonzero(cfn2):
            return cfn2, "CFN2"
        if _nonzero(rf2):
            return rf2, "RF2 (fallback, CFN2 unavailable/zero)"
        return None, "unavailable"

    if _nonzero(rf2):
        return rf2, "RF2"
    if _nonzero(cfn2):
        return cfn2, "CFN2 (fallback, RF2 unavailable/zero)"
    return None, "unavailable"


def _active_mask(timeseries, metadata):
    """Mask of the active phase, shared by forces and SCOF."""
    time = timeseries.get("Time")
    ref, _ = _normal_force_series(timeseries, metadata)
    if time is None or ref is None or len(time) != len(ref):
        n = 0 if ref is None else len(ref)
        return np.ones(n, dtype=bool)

    t_act = _active_end(metadata, float(time[-1]))
    keep = time <= t_act * (1.0 + TIME_TOL)

    st = metadata.get("scratch_time")
    constant = not str(metadata.get("depth_mode", "")).lower().startswith("prog")
    if constant and st and float(st) > 0.0:
        keep = keep & (time >= t_act - float(st) - abs(t_act) * TIME_TOL)
    return keep


def _loading_mask(timeseries, metadata):
    """Active window truncated after the last frame at full command."""
    active = _active_mask(timeseries, metadata)
    if not len(active) or not active.any():
        return active

    u2 = timeseries.get("IndenterU2")
    control_mode = str(metadata.get("control_mode", "displacement"))
    anchor = None
    if control_mode != "force" and u2 is not None and len(u2) == len(active):
        if float(np.max(np.abs(u2))) > 1e-20:
            anchor = np.abs(np.asarray(u2, dtype=float))
    if anchor is None:
        fn, _ = _normal_force_series(timeseries, metadata)
        if fn is None or len(fn) != len(active):
            return active
        anchor = np.abs(np.asarray(fn, dtype=float))

    idx = np.flatnonzero(active)
    a = anchor[idx]
    peak = float(np.nanmax(a))
    if not np.isfinite(peak) or peak <= 0.0:
        return active
    # Last frame at full command, so that a constant-depth plateau is kept whole.
    at_peak = np.flatnonzero(a >= (1.0 - LOAD_PEAK_TOL) * peak)
    if not at_peak.size:
        return active
    k = int(at_peak[-1])
    mask = np.zeros(len(active), dtype=bool)
    mask[idx[:k + 1]] = True
    return mask


def locate_max_depth_z(yz_profile, scratch_length=None):
    """z position (mm) of the deepest section of the groove, or None."""
    y, z = yz_profile
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if y.size != z.size or y.size < 3:
        return None

    L = float(scratch_length) if scratch_length else SCRATCH_LENGTH
    window = (z >= Z_SEARCH_LO_FRAC * L) & (z <= L)
    if not window.any() or not np.isfinite(y[window]).any():
        return None

    dz = float(np.median(np.diff(z)))
    if not np.isfinite(dz) or dz <= 0.0:
        return None

    sigma = max(Z_LOCATE_SMOOTH_MM / dz, 1e-6)
    y_macro = gaussian_filter1d(np.nan_to_num(y, nan=0.0), sigma=sigma)
    z_coarse = float(z[window][int(np.nanargmin(y_macro[window]))])

    near = window & (np.abs(z - z_coarse) <= Z_REFINE_HALF_MM)
    if not near.any() or not np.isfinite(y[near]).any():
        return z_coarse
    return float(z[near][int(np.nanargmin(y[near]))])


# Grid and profiles (copied from ScratchFeatures)
def extract_forces(rfs, z_values):
    """Full-model reaction forces projected onto z_values."""
    # Half model: x2 for the full-model force.
    normal_force_clean = 2 * rfs[~np.isnan(rfs[:, 1]), 1]
    tangential_force_clean = 2 * rfs[~np.isnan(rfs[:, 2]), 2]
    rfs_len = len(normal_force_clean)

    z_force_domain = np.linspace(0, SCRATCH_LENGTH, rfs_len)

    normal_forces_mapped = np.interp(
        z_values, z_force_domain, normal_force_clean, right=np.nan
    )
    tangential_forces_mapped = np.interp(
        z_values, z_force_domain, tangential_force_clean, right=np.nan
    )

    return normal_forces_mapped, tangential_forces_mapped


def map_coords_to_new_grid(coords, method="linear", target_shape=None):
    """Project the deformed node cloud onto a regular z-x grid, after mirroring about x = 0."""
    x, y, z = coords[:, 3], coords[:, 4], coords[:, 5]

    # Mirror about x = 0
    x = np.append(x, -x)
    y = np.append(y, y)
    z = np.append(z, z)

    points = np.column_stack((z.ravel(), x.ravel()))
    values = y.ravel()

    new_rows, new_cols = target_shape or TARGET_SHAPE

    x_new = np.linspace(-SCRATCH_DOMAIN_WIDTH / 2, SCRATCH_DOMAIN_WIDTH / 2, new_rows)
    z_new = np.linspace(0, SCRATCH_DOMAIN_LENGTH, new_cols)
    Z_new, X_new = np.meshgrid(z_new, x_new, indexing="xy")
    Y_new = griddata(points, values, (Z_new, X_new), method=method)

    return X_new, Y_new, Z_new


def get_profiles_from_coords(X, Y, Z, x_value=0.0, z_value=2.0):
    """Transverse (xy) profile at z_value and longitudinal (yz) profile at x_value."""
    idx = np.argmin(np.abs(Z[0, :] - z_value))
    xy_profile = (X[:, idx].squeeze(), Y[:, idx].squeeze())

    idx = np.argmin(np.abs(X[:, 0] - x_value))
    y_coords = Y[idx, :]
    yz_profile = (y_coords, Z[0, :])

    return xy_profile, yz_profile


def _safe_nanmean(*args):
    """Mean of the given scalars, ignoring NaN."""
    valid = [v for v in args if not np.isnan(v)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def calc_xy_peak_indexes(x, y, noise_floor_fraction=0.03):
    """Indices of the two most prominent pile-up ridges on either side of the groove."""
    y_macro = gaussian_filter1d(y, sigma=3.0)
    idx_groove = int(np.argmin(y_macro))

    profile_relief = float(np.ptp(y))
    min_prom = noise_floor_fraction * profile_relief

    peaks, props = find_peaks(y, prominence=min_prom)
    prominences = props["prominences"]

    if len(peaks) < 2:
        raise ValueError(
            f"Profile topography resolved {len(peaks)} peaks; a minimum of 2 "
            "are required to bracket a scratch groove."
        )

    left_mask = peaks < idx_groove
    right_mask = peaks > idx_groove

    if not np.any(left_mask) or not np.any(right_mask):
        raise ValueError(
            f"Groove anchor at index {idx_groove} is not bracketed by peaks "
            "on both sides."
        )

    left_candidates = peaks[left_mask]
    left_prominences = prominences[left_mask]
    right_candidates = peaks[right_mask]
    right_prominences = prominences[right_mask]

    peak_idx_1 = int(left_candidates[np.argmax(left_prominences)])
    peak_idx_2 = int(right_candidates[np.argmax(right_prominences)])

    if peak_idx_1 > peak_idx_2:
        peak_idx_1, peak_idx_2 = peak_idx_2, peak_idx_1

    return peak_idx_1, peak_idx_2


def get_pile_up_height(x, y, peak_idx=None):
    """Pile-up height: profile value at peak_idx, or profile maximum if None."""
    return y[peak_idx] if peak_idx is not None else np.nanmax(y)


def get_residual_depth(x, y):
    """Residual depth: absolute value of the profile minimum."""
    h_r = np.nanmin(y)
    return np.abs(h_r)


def get_frontal_pile_up_height(yz_profile):
    """Frontal pile-up height: maximum of the longitudinal profile for z >= 2.0."""
    y, z = yz_profile
    mask = z >= 2.0
    y = y[mask]
    z = z[mask]
    return get_pile_up_height(z, y)


# Value assembly
def energy_values(timeseries, metadata):
    """Energy-based quality values (KE/IE, AE/IE, energy drift, ALLPW, settling)."""
    out = {}

    ke = timeseries.get("ALLKE")
    ie = timeseries.get("ALLIE")
    ae = timeseries.get("ALLAE")
    time = timeseries.get("Time")

    # KE/IE: kinetic / internal energy of the substrate, in %.
    # steady_max excludes the first 10 % of the active phase.
    if ke is not None and ie is not None:
        mask = ie > 1e-20
        if mask.any():
            ratio = ke[mask] / ie[mask] * 100.0
            time_m = time[mask] if time is not None else np.arange(len(ratio))
            t_max = time_m[-1] if len(time_m) else 1.0
            t_act = _active_end(metadata, t_max) if metadata is not None else t_max
            steady = (time_m > 0.1 * t_act) & (time_m <= t_act * (1.0 + 1e-9))
            out["KE_IE_steady_max"] = float(
                np.max(ratio[steady]) if steady.any() else np.max(ratio)
            )
            out["KE_IE_overall_max"] = float(np.max(ratio))

    # AE/IE: hourglass / internal energy, in %, at the last sample of the active phase.
    if ae is not None and ie is not None:
        mask = ie > 1e-20
        if mask.any():
            ratio = ae[mask] / ie[mask] * 100.0
            final = ratio[-1]
            if metadata is not None and time is not None and len(time) == len(ae):
                t_act = _active_end(metadata, float(time[-1]))
                tm = time[mask]
                sel = np.nonzero(tm <= t_act * (1.0 + 1e-9))[0]
                if sel.size:
                    final = ratio[sel[-1]]
            out["AE_IE_final"] = float(final)

    et = timeseries.get("ETOTAL")
    wk = timeseries.get("WM_ALLWK")

    # E_ref = max(|ALLIE|, |WM_ALLWK|): energy scale of the normalised quantities.
    e_ref = max(_peak(ie), _peak(wk))
    if e_ref >= 1e-20:
        out["E_ref"] = float(e_ref)

        # Energy balance from whole-model terms: IE + VD + FD + KE - WK - PW - CW - MW.
        have_wm = all(timeseries.get(k) is not None for k in WM_BALANCE_TERMS)
        recon = None
        if have_wm:
            recon = (
                timeseries["WM_ALLIE"] + timeseries["WM_ALLVD"]
                + timeseries["WM_ALLFD"] + timeseries["WM_ALLKE"]
                - timeseries["WM_ALLWK"] - timeseries["WM_ALLPW"]
                - timeseries["WM_ALLCW"] - timeseries["WM_ALLMW"]
            )

        bal = et if et is not None else recon
        if bal is not None:
            # Balance value at t = 0, reference for the drift.
            baseline = float(bal[0])
            # ETOTAL_drift: maximum deviation of the balance from its initial value, relative to E_ref.
            out["ETOTAL_drift"] = (
                float(np.max(np.abs(bal - baseline))) / e_ref * 100.0
            )
        if ie is not None:
            # Peak internal energy of the substrate
            out["ALLIE_max"] = float(np.max(ie))

        # ALLPW: peak contact penalty work relative to E_ref, in %.
        pw = timeseries.get("WM_ALLPW")
        if pw is not None:
            out["ALLPW"] = float(_peak(pw) / e_ref * 100.0)

    # Settling: substrate kinetic energy at the last sample / peak internal energy, in %.
    if time is not None and ke is not None and ie is not None and len(time) >= 2:
        ie_peak = _peak(ie)
        if ie_peak >= 1e-20:
            out["KE_final_over_IE_peak"] = abs(float(ke[-1])) / ie_peak * 100.0

    return out


def scof_values(timeseries, metadata):
    """Mean SCOF over the SCOF_LO_FRAC-SCOF_HI_FRAC band of the normal-force ramp."""
    out = {}
    rf3 = timeseries.get("RF3")
    rf2, _ = _normal_force_series(timeseries, metadata)
    if rf2 is None or rf3 is None:
        return out

    active = _loading_mask(timeseries, metadata)
    if len(active) != len(rf2) or not active.any():
        active = np.ones(len(rf2), dtype=bool)

    rf2_abs = np.abs(np.asarray(rf2, dtype=float))
    peak = float(np.nanmax(rf2_abs[active]))
    if not np.isfinite(peak) or peak <= 0.0:
        return out

    mask = active & (rf2_abs >= SCOF_LO_FRAC * peak) & (rf2_abs <= SCOF_HI_FRAC * peak)
    if not mask.any():
        return out

    scof = np.abs(np.asarray(rf3, dtype=float)[mask]) / rf2_abs[mask]
    scof = scof[np.isfinite(scof)]
    if not scof.size:
        return out

    out["SCOF_mean"] = float(np.mean(scof))
    out["SCOF_std"] = float(np.std(scof))
    out["SCOF_n"] = int(scof.size)
    return out


def force_values(timeseries, metadata, z_value):
    """Full-model normal and tangential forces at the end of the loading ramp,
    from a linear fit over the last FORCE_FIT_Z_FRAC of the travel.
    """
    out = {}
    rf3 = timeseries.get("RF3")
    rf2, _ = _normal_force_series(timeseries, metadata)
    if rf2 is None or rf3 is None:
        return out

    active = _active_mask(timeseries, metadata)
    if len(active) != len(rf2) or not active.any():
        return out

    # z advances linearly from 0 to SCRATCH_LENGTH over the active window.
    idx = np.flatnonzero(active)
    z = np.linspace(0.0, SCRATCH_LENGTH, len(idx))

    # Frames after the command peak belong to the unloading.
    load = _loading_mask(timeseries, metadata)
    on = load[idx] if len(load) == len(active) else np.ones(len(idx), dtype=bool)
    if not on.any():
        on = np.ones(len(idx), dtype=bool)

    z_lo = SCRATCH_LENGTH * (1.0 - FORCE_FIT_Z_FRAC)
    sel = on & (z >= z_lo)
    if not sel.any():
        sel = on

    zf = z[sel]
    fn = 2.0 * np.abs(np.asarray(rf2, dtype=float)[idx][sel])
    ft = 2.0 * np.abs(np.asarray(rf3, dtype=float)[idx][sel])
    ok = np.isfinite(zf) & np.isfinite(fn) & np.isfinite(ft)
    zf, fn, ft = zf[ok], fn[ok], ft[ok]
    if not zf.size:
        return out

    # The fit is evaluated at the last loading abscissa, never beyond.
    z_eval = float(np.max(zf))
    if zf.size >= FORCE_FIT_MIN_PTS and float(np.ptp(zf)) > 0.0:
        out["F_n"] = float(np.polyval(np.polyfit(zf, fn, 1), z_eval))
        out["F_t"] = float(np.polyval(np.polyfit(zf, ft, 1), z_eval))
        out["F_fit_mode"] = "ramp_fit"
    else:
        out["F_n"] = float(np.median(fn))
        out["F_t"] = float(np.median(ft))
        out["F_fit_mode"] = "median"
    out["F_fit_n"] = int(zf.size)
    out["F_fit_z"] = z_eval
    return out


def profile_values(nodes, z_value):
    """Residual topography values (h_r, h_p, h_fp) at the deepest section."""
    out = {}
    undef = nodes["undeformed"]
    deform = nodes["deformed"]
    if len(deform) == 0:
        return out

    # map_coords_to_new_grid expects the deformed coordinates in columns 3-5.
    coords = np.hstack([undef, deform])

    X, Y, Z = map_coords_to_new_grid(coords)
    # Locate the deepest section on the longitudinal profile, then read the transverse cut there.
    _, yz_profile = get_profiles_from_coords(X, Y, Z, z_value=z_value)
    z_star = locate_max_depth_z(yz_profile)
    if z_star is not None:
        z_value = z_star
    xy_profile, yz_profile = get_profiles_from_coords(X, Y, Z, z_value=z_value)
    x, y = xy_profile
    out["z_extraction"] = float(z_value)

    # h_r: |min| of the transverse cut
    out["h_r"] = float(get_residual_depth(x, y))

    # h_p: mean height of the two prominent ridges around the groove
    try:
        peak_idx_left, peak_idx_right = calc_xy_peak_indexes(x, y)
        idx_groove = peak_idx_left + int(
            np.argmin(y[peak_idx_left : peak_idx_right + 1])
        )
        h_p_left = get_pile_up_height(x[:idx_groove], y[:idx_groove], peak_idx_left)
        h_p_right = get_pile_up_height(
            x[idx_groove:], y[idx_groove:], peak_idx_right - idx_groove
        )
        out["h_p"] = float(_safe_nanmean(h_p_left, h_p_right))
    except ValueError:
        out["h_p"] = np.nan

    # h_fp: maximum of the longitudinal profile at x ~ 0, for z >= 2.0
    out["h_fp"] = float(get_frontal_pile_up_height(yz_profile))
    return out


# Output units (mm-tonne-s-MPa-N system: energy in N.mm = mJ)
UNITS = {
    "z_extraction": "mm",
    "wallclock": "s",
    "KE_IE_steady_max": "%",
    "KE_IE_overall_max": "%",
    "AE_IE_final": "%",
    "E_ref": "mJ",
    "ETOTAL_drift": "%",
    "ALLIE_max": "mJ",
    "ALLPW": "%",
    "KE_final_over_IE_peak": "%",
    "F_n": "N",
    "F_t": "N",
    "SCOF_mean": "-",
    "SCOF_std": "-",
    "SCOF_n": "samples",
    "F_fit_n": "samples",
    "F_fit_z": "mm",
    "F_fit_mode": "-",
    "h_r": "mm",
    "h_p": "mm",
    "h_fp": "mm",
}


def extract_values(filepath, z_value=None):
    z_value = float(z_value) if z_value is not None else SCRATCH_LENGTH

    metadata, timeseries, nodes = parse_results_csv(filepath)

    # z_extraction is overwritten by profile_values (deepest section).
    values = {"file": Path(filepath).name, "z_extraction": z_value}
    # Run wallclock time, from the '# WallclockTime=' header line
    if metadata.get("wallclock") is not None:
        values["wallclock"] = float(metadata["wallclock"])
    values.update(energy_values(timeseries, metadata))
    values.update(force_values(timeseries, metadata, z_value))
    values.update(scof_values(timeseries, metadata))
    values.update(profile_values(nodes, z_value))
    return values


def _print_values(values):
    for k, v in values.items():
        unit = UNITS.get(k, "")
        if isinstance(v, float):
            print("%-28s %14.6g  %s" % (k, v, unit))
        else:
            print("%-28s %14s  %s" % (k, v, unit))
    print()


def main(target=None, out_csv=None, z_value=None):
    target = Path(target) if target else Path.cwd()
    files = [target] if target.is_file() else sorted(target.glob("*_Results.csv"))

    if not files:
        print(f"No *_Results.csv found in {target}")
        return

    rows = [extract_values(f, z_value) for f in files]
    for r in rows:
        _print_values(r)

    if out_csv:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        parent = Path(out_csv).parent
        if str(parent):
            os.makedirs(str(parent), exist_ok=True)
        # The unit is carried by the column name
        headers = [k + (" [%s]" % UNITS[k] if k in UNITS else "") for k in keys]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(headers)
            for r in rows:
                w.writerow([r.get(k, "") for k in keys])
        print(f"Wrote {len(rows)} row(s) to {out_csv}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else None,
        sys.argv[2] if len(sys.argv) > 2 else None,
        sys.argv[3] if len(sys.argv) > 3 else None,
    )