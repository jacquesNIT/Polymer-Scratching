"""
    SIMULATION CONSISTENCY
    1. ALLKE/ALLIE < 5%
    2. ALLAE/ALLIE < 10%
    3. ETOTAL, drift < 1%

    MATERIAL CONSISTENCY (Mooney-Rivlin specific)
    4. D1 validity         — K/mu ratio in the numerically safe window
    5. Force magnitude     — RF2 peak vs Hertz analytical estimate
    6. Strain level & rate — Tabor characteristic strain, MR validity range

    PHYSICAL CONSISTENCY
    7. Friction physics    — SCOF >= mu_input, bounded, low scatter
    8. Full recovery       — residual depth ~ 0 (pure hyperelastic => no groove)
"""

import numpy as np
import os
import re
import sys

#  Thresholds
KE_IE_THRESHOLD = 5.0            # [%]  ALLKE/ALLIE quasi-static limit
AE_IE_THRESHOLD = 10.0           # [%]  ALLAE/ALLIE hourglass limit
ETOTAL_DRIFT_THRESHOLD  = 1.0    # [%]  Etotal drift limit
K_MU_MIN = 10.0                  # min K/mu (below: too compressible)
K_MU_MAX = 100.0                 # max K/mu (above: noise risk)
HERTZ_TOLERANCE_FACTOR = 10.0    # RF2 must be within x10 of Hertz estimate
MR_STRAIN_VALIDITY = 1.0         # MR validity limit (~100-150%)
RESIDUAL_DEPTH_TOLERANCE = 0.05  # residual depth < 5% of scratch depth

#  CSV Parser
def parse_results_csv(filepath):
    """
    Parse the post-processor CSV.

    metadata   : dict — C10, C01, D1, rho, mu_friction, tip_radius, wallclock...
    timeseries : dict — {column: np.ndarray} for Time, RF1-3, energies
    nodes      : dict — {"labels", "undeformed" (Nx3), "deformed" (Nx3)}
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

        # Metadata lines 
        if line.startswith("#"):
            if "WallclockTime=" in line:
                m = re.search(r"WallclockTime=([\d\.eE+-]+)", line)
                if m:
                    metadata["wallclock"] = float(m.group(1))
            if "Material parameters:" in line or "Material:" in line:
                # Parse key=value pairs:  rho=1.2e-09, C10=0.3, ...
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
                # (was previously unreachable in the data-row section: this
                #  "#"-prefixed line is consumed by the metadata block first)
                body = line.split("Parameters:", 1)[1]
                for k, v in re.findall(r"(\w+)=([A-Za-z0-9\.eE+-]+)", body):
                    try:
                        metadata[k] = float(v)
                    except ValueError:
                        metadata[k] = v       # garde depth_mode='progressive' comme str

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

        # Column header 
        if "Time" in line and "RF1" in line:
            header_cols = [c.strip() for c in line.split(",")]
            continue

        # Data rows 
        parts = line.split(",")
        if len(parts) < 2 or not header_cols:
            continue

        # Time-series part (non-empty Time column)
        if parts[0].strip():
            try:
                row = {}
                for ci, col in enumerate(header_cols):
                    if ci < len(parts) and parts[ci].strip():
                        row[col] = float(parts[ci])
                ts_rows.append(row)
            except ValueError:
                pass

        # Node part (non-empty NodeLabel column)
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


#  Mooney-Rivlin derived properties 
def mr_properties(metadata):
    """
    Small-strain elastic properties from the Mooney-Rivlin parameters
    found in the CSV metadata.  None if C10/C01/D1 are missing.
    """
    if not all(k in metadata for k in ("C10", "C01", "D1")):
        return None

    C10, C01, D1 = metadata["C10"], metadata["C01"], metadata["D1"]

    mu0 = 2.0 * (C10 + C01)
    K0 = 2.0 / D1 if D1 > 0 else float("inf")

    if K0 == float("inf"):
        E0, nu0 = 3.0 * mu0, 0.5
    else:
        E0 = 9.0 * K0 * mu0 / (3.0 * K0 + mu0)
        nu0 = (3.0 * K0 - 2.0 * mu0) / (2.0 * (3.0 * K0 + mu0))

    return {
        "mu_0": mu0, "K_0": K0, "E_0": E0, "nu_0": nu0,
        "K_mu_ratio": K0 / mu0 if mu0 > 0 else float("inf"),
    }

def material_properties(metadata):
    """
    Small-strain isotropic elastic properties, from EITHER the linear-elastic
    parameters (E, nu) when present, or the Mooney-Rivlin parameters
    (C10, C01, D1). Returns None if neither set is available.
    """
    if "E" in metadata and "nu" in metadata:
        E0 = float(metadata["E"])
        nu0 = float(metadata["nu"])
        mu0 = E0 / (2.0 * (1.0 + nu0))
        K0 = E0 / (3.0 * (1.0 - 2.0 * nu0)) if nu0 < 0.5 else float("inf")
        return {
            "mu_0": mu0, "K_0": K0, "E_0": E0, "nu_0": nu0,
            "K_mu_ratio": K0 / mu0 if mu0 > 0 else float("inf"),
        }
    return mr_properties(metadata)


#  Checks — numerical quality

def check_quasi_static(timeseries):
    ke, ie = timeseries.get("ALLKE"), timeseries.get("ALLIE")
    time = timeseries.get("Time")

    if ke is None or ie is None:
        return {"status": "SKIP", "message": "ALLKE or ALLIE not in outputs"}

    mask = ie > 1e-20
    if not mask.any():
        return {"status": "SKIP", "message": "ALLIE is zero everywhere"}

    ratio = ke[mask] / ie[mask] * 100.0
    time_m = time[mask] if time is not None else np.arange(len(ratio))

    # Exclude the first 10% of time (contact-onset transient)
    t_max = time_m[-1] if len(time_m) else 1.0
    steady = time_m > 0.1 * t_max
    steady_max = np.max(ratio[steady]) if steady.any() else np.max(ratio)

    passed = steady_max < KE_IE_THRESHOLD
    return {
        "status": "PASS" if passed else "FAIL",
        "steady_max_percent": steady_max,
        "overall_max_percent": np.max(ratio),
        "message": (
            "KE/IE = %.2f%% (steady-state max, threshold %.0f%%). %s"
            % (steady_max, KE_IE_THRESHOLD,
               "OK" if passed else
               "NOT quasi-static")
        ),
    }


def check_hourglass(timeseries):
    ae, ie = timeseries.get("ALLAE"), timeseries.get("ALLIE")

    if ae is None or ie is None:
        return {"status": "SKIP", "message": "ALLAE or ALLIE not in outputs"}

    mask = ie > 1e-20
    if not mask.any():
        return {"status": "SKIP", "message": "ALLIE is zero everywhere"}

    ratio = ae[mask] / ie[mask] * 100.0
    final = ratio[-1]

    passed = final < AE_IE_THRESHOLD
    return {
        "status": "PASS" if passed else "FAIL",
        "final_percent": final,
        "message": (
            "AE/IE = %.2f%% (final, threshold %.0f%%). %s"
            % (final, AE_IE_THRESHOLD,
               "OK" if passed else
               "Hourglass energy too high")
        ),
    }

WM_BALANCE_TERMS = ("WM_ALLIE", "WM_ALLVD", "WM_ALLFD", "WM_ALLKE",
                    "WM_ALLWK", "WM_ALLPW", "WM_ALLCW", "WM_ALLMW")

def _peak(x):
    return float(np.max(np.abs(x))) if x is not None and len(x) else 0.0


def check_energy_total(timeseries):
    """
    Energy-balance verification.

    ETOTAL is supposed to stay constant.
    The whole-model ETOTAL should be equal to the driver's kinetic energy (at t=0). 
    The conservation metric is the drift of ETOTAL away from its initial value.

    Scopes:
      * substrate ALLIE / ALLKE  -> physical deformation energy (quasi-static
        check uses these; here ALLIE sets the normalisation).
      * WM_*  / ETOTAL           -> whole-model balance (driver KE included).
    """

    # gather values timeseries
    et = timeseries.get("ETOTAL")
    ie_sub = timeseries.get("ALLIE")          # substrate internal energy (physical)
    wk = timeseries.get("WM_ALLWK")           # external work input (whole model)

    if et is None and ie_sub is None:
        return {"status": "SKIP", "message": "Neither ETOTAL nor ALLIE present in outputs."}

    # Physical energy scale, driver KE excluded.
    e_ref = max(_peak(ie_sub), _peak(wk))
    if e_ref < 1e-20:
        return {"status": "SKIP", "message": "No physical energy yet."}

    # Legacy failure mode: ETOTAL identically zero (requested on a set).
    if et is not None and _peak(et) < 1e-30:
        return {"status": "FAIL",
                "message": ("ETOTAL identically zero while physical energy = %.3e" % e_ref)}

    # Reconstruct the balance from whole-model components when available.
    have_wm = all(timeseries.get(k) is not None for k in WM_BALANCE_TERMS)
    recon = None
    if have_wm:
        recon = (timeseries["WM_ALLIE"] + timeseries["WM_ALLVD"]
                 + timeseries["WM_ALLFD"] + timeseries["WM_ALLKE"]
                 - timeseries["WM_ALLWK"] - timeseries["WM_ALLPW"]
                 - timeseries["WM_ALLCW"] - timeseries["WM_ALLMW"])

    # Creation of baseline according to available values.
    bal = et if et is not None else recon
    if bal is None:
        return {"status": "SKIP", "message": "ETOTAL absent and WM_* components missing."}

    baseline = float(bal[0])                                   # driver-KE baseline
    drift_pct = float(np.max(np.abs(bal - baseline))) / e_ref * 100.0

    consistency_pct = None
    if et is not None and recon is not None:
        consistency_pct = float(np.max(np.abs(et - recon))) / e_ref * 100.0

    # Physical, time-varying energy curve (non-constant).
    e_phys_span = (float(np.min(ie_sub)), float(np.max(ie_sub))) if ie_sub is not None else (0.0, 0.0)

    status, issues = "PASS", []

    if drift_pct > ETOTAL_DRIFT_THRESHOLD:
        status = "FAIL"
        issues.append(
            "energy NOT conserved: drift = %.3f%% > %.0f%% of the physical scale (%.3e)." % (drift_pct, ETOTAL_DRIFT_THRESHOLD, e_ref))

    if consistency_pct is not None and consistency_pct > 5.0:
        status = "FAIL" if status == "FAIL" else "WARN"
        issues.append(
            "Abaqus ETOTAL and the reconstructed balance differ by %.1f%%" % consistency_pct)

    if not have_wm:
        if status == "PASS":
            status = "WARN"
        issues.append(
            "WM_* balance components absent")

    recon_msg = "" if consistency_pct is None else " ( ETOTAL vs reconstruction = %.2f%% )" % consistency_pct
    verdict = "OK" if status == "PASS" else " ; ".join(issues)

    return {
        "status": status,
        "drift_percent": drift_pct,
        "baseline": baseline,
        "consistency_percent": consistency_pct,
        "e_phys_min_max": e_phys_span,
        "message": (
            "Conservation drift = %.3f%% of physical energy (scale %.3e). Driver-KE baseline = %.3e. Substrate ALLIE varies %.3e -> %.3e%s. %s"
            % (drift_pct, e_ref, baseline,
               e_phys_span[0], e_phys_span[1], recon_msg, verdict)
        ),
    }


#  Checks — Mooney-Rivlin material consistency

def check_d1_validity(metadata):
    """
    Verify D1 puts K/mu in the numerically safe window [10, 100].

    K/mu < 10   -> artificially compressible, not polymer-like
    K/mu > 100  -> single-precision round-off noise (Abaqus 'D1 too small')
    Sweet spot: K/mu = 20-50  ->  nu_0 = 0.45-0.49
    """
    props = mr_properties(metadata)
    if props is None:
        return {"status": "SKIP", "message": "C10/C01/D1 not found in CSV metadata"}

    ratio = props["K_mu_ratio"]
    nu0 = props["nu_0"]

    if ratio < K_MU_MIN:
        status, verdict = "FAIL", (
            "K/mu too LOW — material artificially compressible "
            "(nu_0=%.3f < ~0.42)" % nu0)
    elif ratio > K_MU_MAX:
        status, verdict = "WARN", (
            "K/mu too HIGH — single-precision noise risk (Abaqus 'D1 too small' ")
    else:
        status, verdict = "PASS", "OK"

    return {
        "status": status,
        "mu_0_MPa": props["mu_0"],
        "K_0_MPa": props["K_0"],
        "E_0_MPa": props["E_0"],
        "nu_0": nu0,
        "K_mu_ratio": ratio,
        "message": (
            "D1=%.3g -> mu_0=%.3g MPa, K_0=%.3g MPa, E_0=%.3g MPa, nu_0=%.4f, "
            "K/mu=%.1f (window [%.0f, %.0f]). %s"
            % (metadata["D1"], props["mu_0"], props["K_0"], props["E_0"], nu0,
               ratio, K_MU_MIN, K_MU_MAX, verdict)
        ),
    }


def check_force_magnitude(timeseries, metadata, nodes):
    """
    Order-of-magnitude check of the peak normal force against Hertz:

        F_hertz = (4/3) * E_star * sqrt(R) * depth^1.5,   E_star = E_0/(1-nu_0^2)

    The depth must be the penetration at the instant of peak RF2 (force and
    depth synchronised), taken from the IndenterU2 trace. Remains an
    order-of-magnitude check for a large-strain polymer / conical tip.
    """

    rf2 = timeseries.get("RF2")
    props = material_properties(metadata)

    if rf2 is None:
        return {"status": "SKIP", "message": "RF2 not in outputs"}
    if props is None:
        return {"status": "SKIP", "message": "Material params not in metadata"}
    if "tip_radius" not in metadata:
        return {"status": "SKIP", "message": "Tip radius not in metadata"}

    R = metadata["tip_radius"]
    E_star = props["E_0"] / (1.0 - props["nu_0"] ** 2)

    rf2_peak = float(np.max(np.abs(rf2)))
    if rf2_peak < 1e-20:
        return {"status": "SKIP", "message": "RF2 is zero"}
    
    depth, dsrc = _penetration_depth(timeseries, metadata, nodes, at_peak_force=True)
    if depth < 1e-9:
        return {"status": "SKIP",
                "message": "No penetration depth available"}

    f_hertz = (4.0 / 3.0) * E_star * np.sqrt(R) * depth ** 1.5 / 2.0   # /2 for  half model
    ratio = rf2_peak / f_hertz if f_hertz > 0 else float("inf")
    ok = (1.0 / HERTZ_TOLERANCE_FACTOR) < ratio < HERTZ_TOLERANCE_FACTOR

    note = ""
    if "residual" in dsrc:
        note = " [depth is residual, not peak."

    return {
        "status": "PASS" if ok else "WARN",
        "rf2_peak_N": rf2_peak,
        "f_hertz_N": f_hertz,
        "ratio": ratio,
        "depth_mm": depth,
        "depth_source": dsrc,
        "message": (
            "RF2 peak = %.3e N | Hertz = %.3e N (depth %.4f mm at %s) | ratio %.2f. %s%s"
            % (rf2_peak, f_hertz, depth, dsrc, ratio,
               "Order of magnitude OK" if ok else
               "Force inconsistent with stiffness",
               note)
        ),
    }



def _penetration_depth(timeseries, metadata, nodes, at_peak_force=False):
    """
    Return (depth_mm, source) for the peak penetration.
    """

    u2 = timeseries.get("IndenterU2")
    rf2 = timeseries.get("RF2")
    if u2 is not None and len(u2) and float(np.max(np.abs(u2))) > 1e-12:
        if at_peak_force and rf2 is not None and float(np.max(np.abs(rf2))) > 1e-20:
            idx = int(np.argmax(np.abs(rf2)))
            return abs(float(u2[idx])), "indenter U2 at peak RF2"
        return abs(float(np.min(u2))), "indenter U2 (max penetration)"
    d = abs(float(metadata.get("scratch_depth", 0.0)))
    if d > 1e-12:
        return d, "commanded scratch_depth"
    if nodes["deformed"].shape[0] > 0:
        d = abs(min(float(np.min(nodes["deformed"][:, 1])), 0.0))
        if d > 1e-12:
            return d, "final frame (residual)"
    return 0.0, "unavailable"


def check_strain_level(timeseries, metadata, nodes):
    """
    Characteristic strain and mean strain rate of the scratch, evaluated at peak penetration.

        depth    = max penetration (IndenterU2 / commanded scratch_depth)
        a        = sqrt(depth * R)                contact length scale
        eps_char = 0.2 * a / R     (spherical regime, depth < delta*)
                 = 0.2 * tan(beta) (conical regime,  depth > delta*)
        delta*   = R*(1 - sin(alpha))             sphere->cone transition depth
        v        = scratch_length / scratch_time  commanded indenter velocity
        eps_rate = eps_char / (2a / v)            strain rate over a transit

    eps_char is checked against the Mooney-Rivlin validity range (~100-150%).
    """

    time = timeseries.get("Time")
    if time is None or len(time) < 2:
        return {"status": "SKIP", "message": "No time data"}

    if "tip_radius" not in metadata:
        return {"status": "SKIP", "message": "Tip radius not in metadata"}
    
    R = metadata["tip_radius"]
    t_total = time[-1] - time[0]

    # Peak penetration depth (NOT the residual final frame).
    depth, source = _penetration_depth(timeseries, metadata, nodes, at_peak_force=False)
    if depth < 1e-9:
        return {"status": "SKIP", "message": "Cannot estimate peak penetration depth"}

    a = np.sqrt(depth * R)   # characteristic contact length (transit-time scale)

    # Regime-aware Tabor representative strain (sphere vs cone).
    cone_angle = metadata.get("cone_angle", None)
    if cone_angle:
        alpha = np.radians(float(cone_angle) / 2.0)        # half-apex from axis
        delta_star = R * (1.0 - np.sin(alpha))
    else:
        alpha = None
        delta_star = float("inf")

    if depth <= delta_star:
        eps_char = 0.2 * a / R
        regime = "spherical"
    else:
        beta = (np.pi / 2.0) - alpha                       # attack angle (face-to-surface)
        eps_char = 0.2 * np.tan(beta)
        regime = "conical"

    # Commanded scratch velocity (fallback to node z-extent / total time).
    sl = metadata.get("scratch_length")
    st = metadata.get("scratch_time")
    if sl and st and float(st) > 0:
        v = abs(float(sl)) / float(st)
        v_src = "commanded"
    elif nodes["undeformed"].shape[0] > 0 and t_total > 0:
        v = (np.max(nodes["undeformed"][:, 2]) - np.min(nodes["undeformed"][:, 2])) / t_total
        v_src = "node z-extent / t (approx.)"
    else:
        v = 0.0
        v_src = "n/a"

    t_contact = 2.0 * a / v if v > 0 else float("inf")
    eps_rate = eps_char / t_contact if t_contact < float("inf") else 0.0

    within = eps_char < MR_STRAIN_VALIDITY
    return {
        "status": "PASS" if within else "WARN",
        "eps_characteristic": eps_char,
        "regime": regime,
        "delta_star_mm": delta_star,
        "mean_strain_rate_per_s": eps_rate,
        "contact_radius_mm": a,
        "scratch_velocity_mm_s": v,
        "depth_mm": depth,
        "depth_source": source,
        "message": (
            "eps_char = %.3f (%s, depth %.4fmm), "
            "strain rate ~ %.2e /s | v = %.1f mm/s. %s"
            % (eps_char, regime, depth, eps_rate, v, 
               "Within MR validity (<%.0f%% strain)" % (MR_STRAIN_VALIDITY * 100)
               if within else
               "Beyond MR validity (~100-150%%)")
        ),
    }


#  Checks — physical consistency

def check_friction_physics(timeseries, metadata):
    """
    The apparent friction SCOF = |RF3|/|RF2| must make physical sense:

      (a) SCOF >= mu_input        — ploughing only ADDS friction.
      (b) SCOF <= mu_input + 0.5  — mu_plough ~ (2/pi)*(a/R) << 1 for a << R
    """

    rf2, rf3 = timeseries.get("RF2"), timeseries.get("RF3")
    mu_input = metadata.get("mu_friction", metadata.get("mu", None))

    if rf2 is None or rf3 is None:
        return {"status": "SKIP", "message": "RF2 or RF3 not in outputs"}
    if mu_input is None:
        return {"status": "SKIP", "message": "mu_friction not found in CSV metadata"}

    rf2_abs = np.abs(rf2)
    mask = rf2_abs > np.max(rf2_abs) * 0.10

    if not mask.any():
        return {"status": "SKIP", "message": "RF2 never exceeds 10% of peak"}

    scof = np.abs(rf3[mask]) / rf2_abs[mask]
    scof_mean, scof_std = np.mean(scof), np.std(scof)

    issues = []
    if scof_mean < mu_input * 0.95:
        issues.append("SCOF < mu_input — NON-PHYSICAL (ploughing cannot reduce friction)")
    if scof_mean > mu_input + 0.5:
        issues.append("Ploughing term too large")
    if scof_mean > 0 and scof_std / scof_mean > 0.30:
        issues.append("High SCOF scatter (std/mean=%.0f%%)"% (scof_std / scof_mean * 100))

    plough_pct = (scof_mean / mu_input - 1.0) * 100.0 if mu_input > 0 else 0.0

    return {
        "status": "PASS" if not issues else "WARN",
        "mu_input": mu_input,
        "scof_mean": scof_mean,
        "scof_std": scof_std,
        "ploughing_contribution_percent": plough_pct,
        "message": (
            "mu_input=%.2f | SCOF=%.3f +/- %.3f | ploughing adds %.0f%%. %s"
            % (mu_input, scof_mean, scof_std, plough_pct,
               "Physically consistent" if not issues else " ; ".join(issues))
        ),
    }


def check_full_recovery(nodes, metadata, is_dissipative=None):
    """
    Pure Mooney-Rivlin has no dissipation mechanism, so the groove must fully
    recover — residual surface depth ~ 0 once the material has relaxed.
    For dissipative families (plasticity / damage) the logic is inverted: a
    permanent groove is EXPECTED. verify_results passes is_dissipative from the
    family; when run standalone we infer it from the metadata.
    """

    if nodes["deformed"].shape[0] == 0:
        return {"status": "SKIP", "message": "No node data"}

    y_def = nodes["deformed"][:, 1]

    # Robust residual (1st percentile of downward displacements) 
    y_neg = y_def[y_def < 0.0]
    residual = abs(float(np.percentile(y_neg, 1))) if y_neg.size else 0.0
    residual_raw = abs(min(float(np.min(y_def)), 0.0))   # kept for reference
    pile_up = max(float(np.max(y_def)), 0.0)

    # Reference depth (peak commanded depth; valid for the progressive ramp) 
    ref = abs(float(metadata.get("scratch_depth", 0.0)))
    ref_is_guess = ref < 1e-12
    if ref_is_guess:
        ref = metadata.get("tip_radius", 0.2) * 0.1
    rel = residual / ref * 100.0
    guess_note = (" [ref is a guess: scratch_depth absent from metadata]"
                  if ref_is_guess else "")

    # Dissipative families (plasticity / damage): a PERMANENT groove is
    # EXPECTED, so the pass/fail logic is inverted (residual ~ 0 is the anomaly).
    # Use the value passed by verify_results; fall back to metadata detection
    # (family tag or presence of a yield stress) for standalone use.
    if is_dissipative is None:
        family = str(metadata.get("family", "")).lower()
        is_dissipative = ("sigma_y0" in metadata
                          or any(t in family for t in
                                 ("j2", "mises", "plast", "semicryst", "glassy", "thermoset")))
    if is_dissipative:
        has_recovery = float(metadata.get("recovery_time", 0.0)) > 0.0
        recov_note = ("" if has_recovery else
                      " [no recovery step: elastic springback may be incomplete, "
                      "groove possibly overestimated]")
        passed = rel >= RESIDUAL_DEPTH_TOLERANCE * 100.0   # a groove is present
        verdict = ("OK — permanent groove present, consistent with plasticity"
                   if passed else
                   "No residual groove despite a dissipative model — check yield "
                   "level, depth or mesh") + guess_note + recov_note
        return {
            "status": "PASS" if passed else "WARN",
            "residual_depth_mm": residual,
            "residual_depth_raw_mm": residual_raw,
            "pile_up_mm": pile_up,
            "relative_percent": rel,
            "reference_mm": ref,
            "message": (
                "Residual depth = %.3e mm (%.1f%% of ref %.3f mm), pile-up = %.3e mm. %s"
                % (residual, rel, ref, pile_up, verdict)
            ),
        }

    # Recovery guard: last frame is relaxed only if a recovery step ran 
    has_recovery = float(metadata.get("recovery_time", 0.0)) > 0.0
    if not has_recovery:
        return {
            "status": "WARN",
            "residual_depth_mm": residual,
            "residual_depth_raw_mm": residual_raw,
            "pile_up_mm": pile_up,
            "relative_percent": rel,
            "reference_mm": ref,
            "message": (
                "Residual = %.3e mm (%.1f%% of ref %.3f mm) | raw min = %.3e mm. "
                "NO recovery step (recovery_time=0), not a relaxed state.%s"
                % (residual, rel, ref, residual_raw, guess_note)
            ),
        }

    passed = rel < RESIDUAL_DEPTH_TOLERANCE * 100.0
    verdict = ("OK — full hyperelastic recovery" if passed else
               "Residual groove without dissipation in the model : numerical artifact.") + guess_note

    return {
        "status": "PASS" if passed else "FAIL",
        "residual_depth_mm": residual,
        "residual_depth_raw_mm": residual_raw,
        "pile_up_mm": pile_up,
        "relative_percent": rel,
        "reference_mm": ref,
        "message": (
            "Residual depth = %.3e mm (%.1f%% of ref %.3f mm), pile-up = %.3e mm. %s"
            % (residual, rel, ref, pile_up, verdict)
        ),
    }

#  Master verification

#  Family-aware check selection
# families.py (Configuration package) is the source of truth for which checks
# apply to each family. We try to import it; if the package is not importable
# (standalone CSV verification), we fall back to this mirror.
_FALLBACK_FAMILIES = {
    "elastomer_mr": {
        "label": "Unfilled elastomer (Mooney-Rivlin)",
        "dissipative": False,
        "checks": ("quasi_static", "hourglass", "energy_total", "d1_validity",
                   "force_magnitude", "strain_level", "friction_physics", "recovery"),
    },
    "semicrystalline_j2": {
        "label": "Soft semicrystalline (linear elastic + J2 plasticity)",
        "dissipative": True,
        "checks": ("quasi_static", "hourglass", "energy_total",
                   "force_magnitude", "strain_level", "friction_physics", "recovery"),
    },
}
_DEFAULT_FAMILY = "elastomer_mr"


def _resolve_family(family_key):
    # Return {"label", "checks", "dissipative"} for a family key, preferring the
    # live definition in families.py and falling back to the local mirror.
    try:
        from ScratchSimulation.AbaqusModel.Configuration import get_family
        fam = get_family(family_key)
        mat = fam.build_config().material
        dissipative = (mat.plasticity.MODEL != "none" or mat.damage.MODEL != "none")
        return {"label": fam.label, "checks": list(fam.checks), "dissipative": dissipative}
    except Exception:
        return dict(_FALLBACK_FAMILIES.get(family_key, _FALLBACK_FAMILIES[_DEFAULT_FAMILY]))


def verify_results(filepath, print_report=True):
    """
    Run the checks declared for the simulation's polymer family on a results CSV.
    """

    if not os.path.exists(filepath):
        raise IOError("File not found: %s" % filepath)

    metadata, timeseries, nodes = parse_results_csv(filepath)

    family_key = str(metadata.get("family", _DEFAULT_FAMILY))
    fam = _resolve_family(family_key)

    report = {"file": filepath, "metadata": metadata,
              "family": family_key, "family_label": fam["label"], "checks": {}}

    # Check name -> (display label, zero-arg callable). Only the names listed in
    # the family's "checks" are run; recovery is told explicitly whether the
    # family is dissipative so its pass/fail logic matches the family.
    registry = {
        "quasi_static":     ("Quasi-static (KE/IE)",      lambda: check_quasi_static(timeseries)),
        "hourglass":        ("Hourglass (AE/IE)",         lambda: check_hourglass(timeseries)),
        "energy_total":     ("Energy total (ETOTAL)",     lambda: check_energy_total(timeseries)),
        "d1_validity":      ("D1 validity (K/mu window)", lambda: check_d1_validity(metadata)),
        "force_magnitude":  ("Force magnitude (Hertz)",   lambda: check_force_magnitude(timeseries, metadata, nodes)),
        "strain_level":     ("Strain level",              lambda: check_strain_level(timeseries, metadata, nodes)),
        "friction_physics": ("Friction physics (SCOF)",   lambda: check_friction_physics(timeseries, metadata)),
        "recovery":         ("Recovery",                  lambda: check_full_recovery(nodes, metadata, fam["dissipative"])),
    }

    for name in fam["checks"]:
        entry = registry.get(name)
        if entry is None:
            continue
        label, run = entry
        report["checks"][label] = run()

    if print_report:
        _print_report(report)

    return report


def _print_report(report):
    """
    Print a formatted verification report.
    """

    print("")
    print("-" * 60)
    print("  SCRATCH SIMULATION — Results verification")
    print("  Family: %s" % report.get("family_label", report.get("family", "?")))
    print("-" * 60)
    print("  File: %s" % report["file"])

    meta = report["metadata"]
    mat_keys = [k for k in ("rho", "C10", "C01", "D1", "E", "nu",
                            "sigma_y0", "mu_friction", "mu")
                if k in meta]
    
    if mat_keys:
        print("  Material: %s" % ", ".join("%s=%s" % (k, meta[k]) for k in mat_keys))
    if "tip_radius" in meta:
        print("  Indenter: R=%.2f mm, angle=%s deg"
              % (meta["tip_radius"], meta.get("cone_angle", "?")))
    if "wallclock" in meta:
        print("  Wallclock: %.1f s" % meta["wallclock"])

    counts = {}
    for name, result in report["checks"].items():
        status = result.get("status", "INFO")
        counts[status] = counts.get(status, 0) + 1
        print("")
        print("  [%4s]  %s" % (status, name))
        print("          %s" % result.get("message", ""))

    print("")
    print("-" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python results_verifier.py <path_to_Results.csv>")
        sys.exit(1)
    verify_results(sys.argv[1])