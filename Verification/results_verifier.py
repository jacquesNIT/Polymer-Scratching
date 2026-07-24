"""
    SIMULATION CONSISTENCY
    1. ALLKE/ALLIE < 5%
    2. ALLAE/ALLIE — target < 5% (WARN band up to 10%)
    3. ETOTAL, drift < 1%
    4. Artificial work    — ALLMW (mass scaling) & ALLPW (contact penalty)
                            small vs the physical energy scale

    MATERIAL CONSISTENCY
    5. D1 validity         — K/mu ratio in the numerically safe window (MR)
    6. Force magnitude     — peak normal force vs Hertz (elastic families) or
                             vs scratch hardness F ~ C*sigma_y*A (dissipative)
    7. Strain level & rate — Tabor characteristic strain, model validity range

    PHYSICAL CONSISTENCY
    8. Friction physics    — SCOF >= mu_input, bounded, low scatter
    9. Steady state        — RF/SCOF plateau over the second half of the scratch
   10. Settling            — kinetic energy decayed before the residual profile is trusted
   11. Recovery            — residual ~ 0 (hyperelastic) / groove expected (dissipative)
   12. Residual profile    — residual depth & pile-up reported WITHOUT a verdict (for viscoelastic families)

    The normal force is RF2 in displacement-controlled mode and CFN2 (contact
    pair) in force-controlled mode; see _normal_force_series.
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
RECOVERY_TAU_RATIO = 3.0         # recovery_time / tau_max needed before the residual profile of a viscoelastic family is considered converged
AE_IE_TARGET = 5.0               # [%]  hourglass PASS target (WARN band up to AE_IE_THRESHOLD)
MW_WARN_PCT = 1.0                # [%]  artificial work / physical energy (WARN above)
MW_FAIL_PCT = 5.0                # [%]  artificial work / physical energy (FAIL above)
PLATEAU_CV_WARN = 10.0           # [%]  force CV / trend over the scratch plateau (WARN above)
PLATEAU_CV_FAIL = 30.0           # [%]  force CV / trend over the scratch plateau (FAIL above)
SETTLE_KE_PCT = 1.0              # [%]  final ALLKE / peak ALLIE for a settled recovery
HARDNESS_TOLERANCE_FACTOR = 3.0  # dissipative families: force within x3 of C*sigma_y*A
CONSTRAINT_FACTOR = 2.0          # Tabor constraint factor C for polymers (~1.5-2.6)

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

def ab_properties(metadata):
    """
    Small-strain elastic properties from the Arruda-Boyce parameters
    (mu_AB, lambda_m, D_AB) found in the CSV metadata. The initial shear
    modulus includes the locking-stretch correction of the eight-chain model:
      mu_0 = mu*(1 + 3/(5 lm^2) + 99/(175 lm^4) + 513/(875 lm^6) + 42039/(67375 lm^8))
    Returns None if mu_AB / D_AB are missing.
    """
    if "mu_AB" not in metadata or "D_AB" not in metadata:
        return None

    mu = float(metadata["mu_AB"])
    D = float(metadata["D_AB"])
    lm = float(metadata.get("lambda_m", 0.0))

    corr = 1.0
    if lm > 0.0:
        l2 = lm * lm
        corr = (1.0 + 3.0 / (5.0 * l2) + 99.0 / (175.0 * l2 ** 2)
                + 513.0 / (875.0 * l2 ** 3) + 42039.0 / (67375.0 * l2 ** 4))

    mu0 = mu * corr
    K0 = 2.0 / D if D > 0 else float("inf")

    if K0 == float("inf"):
        E0, nu0 = 3.0 * mu0, 0.5
    else:
        E0 = 9.0 * K0 * mu0 / (3.0 * K0 + mu0)
        nu0 = (3.0 * K0 - 2.0 * mu0) / (2.0 * (3.0 * K0 + mu0))

    return {
        "mu_0": mu0, "K_0": K0, "E_0": E0, "nu_0": nu0,
        "K_mu_ratio": K0 / mu0 if mu0 > 0 else float("inf"),
    }

def yeoh_properties(metadata):
    """
    Small-strain elastic properties from the Yeoh parameters (C10_Y, D1_Y):
    mu_0 = 2*C10 (C20/C30 vanish at I1 = 3), K_0 = 2/D1.
    """
    if "C10_Y" not in metadata or "D1_Y" not in metadata:
        return None
    mu0 = 2.0 * float(metadata["C10_Y"])
    K0 = 2.0 / float(metadata["D1_Y"]) if float(metadata["D1_Y"]) > 0 else float("inf")
    return _props_from_mu_K(mu0, K0)

def ogden_properties(metadata):
    """
    Small-strain elastic properties from the Ogden parameters
    (mu1_O..muN_O, D1_O): in the Abaqus convention mu_0 = sum(mu_i).
    """
    if "D1_O" not in metadata:
        return None
    mus = []
    i = 1
    while ("mu%d_O" % i) in metadata:
        mus.append(float(metadata["mu%d_O" % i]))
        i += 1
    if not mus:
        return None
    mu0 = sum(mus)
    K0 = 2.0 / float(metadata["D1_O"]) if float(metadata["D1_O"]) > 0 else float("inf")
    return _props_from_mu_K(mu0, K0)


def _props_from_mu_K(mu0, K0):
    if K0 == float("inf"):
        E0, nu0 = 3.0 * mu0, 0.5
    else:
        E0 = 9.0 * K0 * mu0 / (3.0 * K0 + mu0)
        nu0 = (3.0 * K0 - 2.0 * mu0) / (2.0 * (3.0 * K0 + mu0))
    return {"mu_0": mu0, "K_0": K0, "E_0": E0, "nu_0": nu0,
            "K_mu_ratio": K0 / mu0 if mu0 > 0 else float("inf")}




def material_properties(metadata):
    """
    Small-strain isotropic elastic properties, from the linear-elastic
    parameters (E, nu) when present, else Mooney-Rivlin (C10, C01, D1), else
    Arruda-Boyce (mu_AB, lambda_m, D_AB), else Yeoh (C10_Y, D1_Y), else
    Ogden (mu*_O, D1_O). Returns None if no set is available.
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
    for fn in (mr_properties, ab_properties, yeoh_properties, ogden_properties):
        props = fn(metadata)
        if props is not None:
            return props
    return None




#  Checks — numerical quality

def _active_end(metadata, t_last):
    """
    End of the ACTIVE phase (indent + scratch) from the metadata, capped by
    the last time sample. Since the extractor now keeps the (low-frequency)
    energy history through unload/recovery, time[-1] is the RECOVERY end:
    every window that used to be anchored at time[-1] (scratch plateau, SCOF,
    quasi-static steady window, hourglass 'final') must be anchored here
    instead, or it silently slides into the unload/recovery phase.
    """
    st = float(metadata.get("scratch_time", 0.0) or 0.0)
    it = float(metadata.get("indentation_time", 0.0) or 0.0)
    constant = not str(metadata.get("depth_mode", "")).lower().startswith("prog")
    t_act = st + (it if constant else 0.0)
    if t_act <= 0.0:
        return t_last
    return min(t_last, t_act)


def check_quasi_static(timeseries, metadata=None):
    ke, ie = timeseries.get("ALLKE"), timeseries.get("ALLIE")
    time = timeseries.get("Time")

    if ke is None or ie is None:
        return {"status": "SKIP", "message": "ALLKE or ALLIE not in outputs"}

    mask = ie > 1e-20
    if not mask.any():
        return {"status": "SKIP", "message": "ALLIE is zero everywhere"}

    ratio = ke[mask] / ie[mask] * 100.0
    time_m = time[mask] if time is not None else np.arange(len(ratio))

    # Exclude the first 10% of time (contact-onset transient) and everything
    # AFTER the active phase: KE/IE during unload/recovery is settling's job,
    # and for a fully recovering elastomer IE -> 0 there (ratio blows up).
    t_max = time_m[-1] if len(time_m) else 1.0
    t_act = _active_end(metadata, t_max) if metadata is not None else t_max
    steady = (time_m > 0.1 * t_act) & (time_m <= t_act * (1.0 + 1e-9))
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


def check_hourglass(timeseries, metadata=None):
    ae, ie = timeseries.get("ALLAE"), timeseries.get("ALLIE")
    time = timeseries.get("Time")

    if ae is None or ie is None:
        return {"status": "SKIP", "message": "ALLAE or ALLIE not in outputs"}

    mask = ie > 1e-20
    if not mask.any():
        return {"status": "SKIP", "message": "ALLIE is zero everywhere"}

    ratio = ae[mask] / ie[mask] * 100.0
    # 'final' = last sample of the ACTIVE phase: with the energy history now
    # extending through unload/recovery, the very last sample sits after the
    # elastic release (IE has dropped), which would inflate AE/IE.
    final = ratio[-1]
    if metadata is not None and time is not None and len(time) == len(ae):
        t_act = _active_end(metadata, float(time[-1]))
        tm = time[mask]
        sel = np.nonzero(tm <= t_act * (1.0 + 1e-9))[0]
        if sel.size:
            final = ratio[sel[-1]]

    if final < AE_IE_TARGET:
        status, verdict = "PASS", "OK"
    elif final < AE_IE_THRESHOLD:
        status, verdict = "WARN", ("Acceptable but forces may be polluted by a few percent "
                                   "(consider ENHANCED hourglass control / finer mesh)")
    else:
        status, verdict = "FAIL", "Hourglass energy too high"
    return {
        "status": status,
        "final_percent": final,
        "message": (
            "AE/IE = %.2f%% (final, target < %.0f%%, hard limit %.0f%%). %s"
            % (final, AE_IE_TARGET, AE_IE_THRESHOLD, verdict)
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


def check_artificial_energy(timeseries):
    """
    Artificial work terms of the whole-model balance:

      * WM_ALLMW — mass-scaling work. The dominant contamination channel of
        the stiff (dissipative) families: the glassy MS study shows an ETOTAL
        drift of ~43% at MS 10000 that falls below ~5% only for MS <= 500.
      * WM_ALLPW — contact penalty work; should stay negligible.

    Both are normalised by the physical energy scale max(|ALLIE|, |WM_ALLWK|).
    """
    ie = timeseries.get("ALLIE")
    wk = timeseries.get("WM_ALLWK")
    mw = timeseries.get("WM_ALLMW")
    pw = timeseries.get("WM_ALLPW")

    if mw is None and pw is None:
        return {"status": "SKIP", "message": "WM_ALLMW / WM_ALLPW not in outputs"}

    e_ref = max(_peak(ie), _peak(wk))
    if e_ref < 1e-20:
        return {"status": "SKIP", "message": "No physical energy yet."}

    mw_pct = _peak(mw) / e_ref * 100.0 if mw is not None else 0.0
    pw_pct = _peak(pw) / e_ref * 100.0 if pw is not None else 0.0
    worst = max(mw_pct, pw_pct)

    if worst < MW_WARN_PCT:
        status, verdict = "PASS", "OK"
    elif worst < MW_FAIL_PCT:
        status, verdict = "WARN", "Artificial work is polluting the energy balance"
    else:
        status, verdict = "FAIL", ("Artificial work dominates -- reduce mass_scale, or switch to "
                                   "variable mass scaling (target_time_increment) on the fine zone only")

    return {
        "status": status,
        "allmw_percent": mw_pct,
        "allpw_percent": pw_pct,
        "message": (
            "ALLMW = %.2f%%, ALLPW = %.2f%% of the physical energy scale (%.3e) "
            "(warn %.0f%%, fail %.0f%%). %s"
            % (mw_pct, pw_pct, e_ref, MW_WARN_PCT, MW_FAIL_PCT, verdict)
        ),
    }


def check_steady_state(timeseries, metadata):
    """
    Steady-state plateau over the second half of the scratch phase.

    Without a plateau, the mean SCOF (and any single "peak force") is not a
    well-defined observable: a large CV reveals inertial ringing, a strong
    end-trend reveals a scratch too short to reach steady state or a boundary
    effect. Only meaningful in constant depth_mode (progressive mode ramps the
    depth throughout the scratch, so no plateau exists by design).
    """
    time = timeseries.get("Time")
    force, force_src = _normal_force_series(timeseries, metadata)
    st = metadata.get("scratch_time")

    if time is None or len(time) < 4:
        return {"status": "SKIP", "message": "No time data"}
    if force is None:
        return {"status": "SKIP", "message": "Normal force (RF2/CFN2) not in outputs"}
    if not st or float(st) <= 0.0:
        return {"status": "SKIP", "message": "scratch_time not in metadata"}

    if str(metadata.get("depth_mode", "")).lower().startswith("prog"):
        return {"status": "SKIP",
                "message": ("progressive depth_mode: depth ramps during the whole scratch, "
                            "no steady-state plateau expected")}

    # Anchor at the end of the ACTIVE phase (not time[-1] = recovery end).
    t_end = _active_end(metadata, float(time[-1]))
    mask = (time >= (t_end - 0.5 * float(st))) & (time <= t_end * (1.0 + 1e-9))
    if int(np.sum(mask)) < 10:
        return {"status": "SKIP", "message": "Fewer than 10 samples in the plateau window"}

    f = np.abs(force[mask])
    f_mean = float(np.mean(f))
    if f_mean < 1e-20:
        return {"status": "SKIP", "message": "Normal force is zero over the plateau window"}

    cv = float(np.std(f)) / f_mean * 100.0
    n = len(f)
    trend = (float(np.mean(f[3 * n // 4:])) / f_mean - 1.0) * 100.0

    # SCOF stability over the same window (when RF3 is available)
    scof_cv = None
    rf3 = timeseries.get("RF3")
    if rf3 is not None and len(rf3) == len(force):
        scof = np.abs(rf3[mask]) / np.maximum(f, 1e-20)
        if float(np.mean(scof)) > 0:
            scof_cv = float(np.std(scof) / np.mean(scof)) * 100.0

    worst = max(cv, abs(trend))
    if worst < PLATEAU_CV_WARN:
        status, verdict = "PASS", "Steady state reached"
    elif worst < PLATEAU_CV_FAIL:
        status, verdict = "WARN", "Noisy or drifting plateau (inertial ringing / scratch too short?)"
    else:
        status, verdict = "FAIL", "No steady state -- mean SCOF and peak force are not reliable"

    scof_msg = "" if scof_cv is None else " | SCOF CV = %.1f%%" % scof_cv
    return {
        "status": status,
        "force_source": force_src,
        "cv_percent": cv,
        "trend_percent": trend,
        "scof_cv_percent": scof_cv,
        "message": (
            "%s plateau (last 50%% of scratch): CV = %.1f%%, end-trend = %+.1f%%%s. %s"
            % (force_src, cv, trend, scof_msg, verdict)
        ),
    }


def check_settling(timeseries, metadata):
    """
    Before trusting the residual profile, the substrate kinetic energy must
    have decayed when the final frame is written: mass scaling slows the
    apparent settling by sqrt(mass_scale), so a short recovery step can freeze
    a still-ringing groove.

    NB: the substrate energy history is kept alive (low frequency) through
    unload/recovery and the extractor resamples it onto the master time axis
    without truncation, so this check is now effective. The WARN branch below
    is kept as a safety net for CSVs produced by the old extractor (which
    truncated the energy series at the end of the scratch).
    """
    time = timeseries.get("Time")
    ke = timeseries.get("ALLKE")
    ie = timeseries.get("ALLIE")
    recovery_time = float(metadata.get("recovery_time", 0.0) or 0.0)

    if recovery_time <= 0.0:
        return {"status": "SKIP", "message": "No recovery step (recovery_time = 0)"}
    if time is None or ke is None or ie is None or len(time) < 2:
        return {"status": "SKIP", "message": "Time/ALLKE/ALLIE not in outputs"}

    ie_peak = _peak(ie)
    if ie_peak < 1e-20:
        return {"status": "SKIP", "message": "ALLIE is zero everywhere"}

    # Expected end of the active (indent + scratch) phase from the metadata.
    st = float(metadata.get("scratch_time", 0.0) or 0.0)
    it = float(metadata.get("indentation_time", 0.0) or 0.0)
    constant_mode = not str(metadata.get("depth_mode", "")).lower().startswith("prog")
    t_active_end = st + (it if constant_mode else 0.0)

    covers_recovery = t_active_end > 0.0 and float(time[-1]) > 1.05 * t_active_end
    ke_final_pct = abs(float(ke[-1])) / ie_peak * 100.0

    if not covers_recovery:
        return {
            "status": "WARN",
            "ke_final_percent": ke_final_pct,
            "message": (
                "Energy history stops at the end of the scratch (deactivated in "
                "unload/recovery): settling of the recovery phase cannot be "
                "verified, so the residual depth is measured on a possibly "
                "still-ringing state (mass scaling slows settling by sqrt(MS)). "
                "Keep the substrate energy history active (low frequency) "
                "through recovery to enable this check."
            ),
        }

    passed = ke_final_pct < SETTLE_KE_PCT
    return {
        "status": "PASS" if passed else "FAIL",
        "ke_final_percent": ke_final_pct,
        "message": (
            "Final ALLKE = %.2f%% of peak ALLIE (threshold %.1f%%). %s"
            % (ke_final_pct, SETTLE_KE_PCT,
               "Settled -- residual profile trustworthy" if passed else
               "Still ringing -- extend recovery_time or reduce mass_scale "
               "before trusting the residual depth")
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

def _contact_radius(depth, R, cone_angle_deg=None):
    """
    Geometric contact radius of the Rockwell tip at a given penetration depth.
    cone_angle_deg is the HALF-apex angle measured from the axis (60 deg =>
    included angle 120 deg); the sphere-to-cone transition occurs at
    delta* = R*(1 - sin(alpha)) with the flank widening at tan(alpha) beyond.
    """
    depth = float(depth)
    if depth <= 0.0:
        return 0.0
    if cone_angle_deg:
        alpha = np.radians(float(cone_angle_deg))
        delta_star = R * (1.0 - np.sin(alpha))
    else:
        alpha, delta_star = None, float("inf")
    if alpha is None or depth <= delta_star:
        d = min(depth, R)
        return float(np.sqrt(max(2.0 * R * d - d * d, 0.0)))
    r_t = R * np.cos(alpha)
    return float(r_t + (depth - delta_star) * np.tan(alpha))


def _normal_force_series(timeseries, metadata):

    """
    Return (force_array, source_label) for the normal contact force.

    In displacement-controlled mode, u2 carries a DisplacementBC and RF2 is
    the physically meaningful reaction. In force-controlled mode, u2 carries
    no BC (only a ConcentratedForce / CF2 load) so RF2 has no reaction to
    report and reads ~0 throughout -- CFN2 (total normal contact force on
    the slave surface, from the dedicated contact-pair history region) is
    used instead. Falls back gracefully when one of the two columns is
    absent or identically zero (e.g. CSVs produced before this feature, or
    a CFN identifier mismatch), so older displacement-mode CSVs are
    completely unaffected.
    """

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




def check_force_magnitude(timeseries, metadata, nodes, is_dissipative=None):
    """
    Order-of-magnitude check of the peak normal force.

      * Elastic families (hyperelastic base): Hertz,
          F = (4/3) * E_star * sqrt(R) * depth^1.5,  E_star = E_0/(1-nu_0^2)
      * Dissipative families (plasticity): scratch hardness,
          F = C * sigma_y0 * A_proj,  A_proj = pi*a^2,  C ~ 2 (Tabor, polymers)
        Hertz is meaningless once the contact is dominated by plastic flow,
        which is why the old x10 tolerance detected almost nothing; the
        hardness estimate allows a x3 tolerance.

    The depth is the penetration at the instant of peak normal force
    (IndenterU2 trace, force and depth synchronised). The normal force is RF2
    in displacement-controlled mode or CFN2 in force-controlled mode (see
    _normal_force_series). Analytical estimates are halved (half-symmetry).
    """

    force, force_src = _normal_force_series(timeseries, metadata)
    props = material_properties(metadata)

    if force is None:
        return {"status": "SKIP", "message": "Normal force (RF2/CFN2) not in outputs or zero"}
    if props is None:
        return {"status": "SKIP", "message": "Material params not in metadata"}
    if "tip_radius" not in metadata:
        return {"status": "SKIP", "message": "Tip radius not in metadata"}

    if is_dissipative is None:
        family = str(metadata.get("family", "")).lower()
        is_dissipative = ("sigma_y0" in metadata
                          or any(t in family for t in
                                 ("j2", "mises", "plast", "semicryst", "glassy", "thermoset")))

    R = metadata["tip_radius"]
    f_peak = float(np.max(np.abs(force)))
    if f_peak < 1e-20:
        return {"status": "SKIP", "message": "%s is zero" % force_src}

    depth, dsrc = _penetration_depth(timeseries, metadata, nodes, at_peak_force=True)
    if depth < 1e-9:
        return {"status": "SKIP",
                "message": "No penetration depth available"}

    if is_dissipative and "sigma_y0" in metadata:
        a = _contact_radius(depth, R, metadata.get("cone_angle"))
        f_model = CONSTRAINT_FACTOR * metadata["sigma_y0"] * np.pi * a ** 2 / 2.0  # /2 half model
        model_name = "hardness C*sy*A (C=%.1f, a=%.4f mm)" % (CONSTRAINT_FACTOR, a)
        tol = HARDNESS_TOLERANCE_FACTOR
    else:
        E_star = props["E_0"] / (1.0 - props["nu_0"] ** 2)
        f_model = (4.0 / 3.0) * E_star * np.sqrt(R) * depth ** 1.5 / 2.0           # /2 half model
        model_name = "Hertz"
        tol = HERTZ_TOLERANCE_FACTOR

    ratio = f_peak / f_model if f_model > 0 else float("inf")
    ok = (1.0 / tol) < ratio < tol

    note = ""
    if "residual" in dsrc:
        note = " [depth is residual, not peak]"

    return {
        "status": "PASS" if ok else "WARN",
        "force_peak_N": f_peak,
        "force_source": force_src,
        "f_model_N": f_model,
        "model": model_name,
        "ratio": ratio,
        "depth_mm": depth,
        "depth_source": dsrc,
        "message": (
            "%s peak = %.3e N | %s = %.3e N (depth %.4f mm at %s) | ratio %.2f (tol x%.0f). %s%s"
            % (force_src, f_peak, model_name, f_model, depth, dsrc, ratio, tol,
               "Order of magnitude OK" if ok else
               "Force inconsistent with material strength/stiffness",
               note)
        ),
    }



def _penetration_depth(timeseries, metadata, nodes, at_peak_force=False):
    """
    Return (depth_mm, source) for the peak penetration. The synchronising
    force is RF2 or CFN2 depending on the control mode (_normal_force_series).
    """

    u2 = timeseries.get("IndenterU2")
    force, force_src = _normal_force_series(timeseries, metadata)
    if u2 is not None and len(u2) and float(np.max(np.abs(u2))) > 1e-12:
        if at_peak_force and force is not None and float(np.max(np.abs(force))) > 1e-20:
            idx = int(np.argmax(np.abs(force)))
            return abs(float(u2[idx])), "indenter U2 at peak %s" % force_src
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
        # cone_angle IS the half-apex angle measured from the axis (60 deg =>
        # included angle 120 deg, attack angle 30 deg) -- see Rockwell_coords.
        # The former /2 treated it as a full apex angle and misclassified the
        # regime (delta* = 0.100 mm instead of 0.027 mm for R = 0.2 mm).
        alpha = np.radians(float(cone_angle))              # half-apex from axis
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

    rf3 = timeseries.get("RF3")
    rf2, force_src = _normal_force_series(timeseries, metadata)
    mu_input = metadata.get("mu_friction", metadata.get("mu", None))

    if rf2 is None or rf3 is None:
        return {"status": "SKIP", "message": "Normal force (RF2/CFN2) or RF3 not in outputs"}
    if mu_input is None:
        return {"status": "SKIP", "message": "mu_friction not found in CSV metadata"}

    rf2_abs = np.abs(rf2)
    mask = rf2_abs > np.max(rf2_abs) * 0.10

    # Restrict to the scratch phase in constant depth_mode: during indentation
    # RF3 ~ 0 while RF2 already carries the full normal force, so including
    # those samples biases the mean SCOF downwards (false "SCOF < mu" alarms).
    time = timeseries.get("Time")
    st = metadata.get("scratch_time")
    constant_mode = not str(metadata.get("depth_mode", "")).lower().startswith("prog")
    if constant_mode and time is not None and st and float(st) > 0.0:
        # Scratch window anchored at the end of the ACTIVE phase, not at
        # time[-1] (= recovery end since the energy history covers recovery).
        t_act = _active_end(metadata, float(time[-1]))
        mask = mask & (time >= t_act - float(st)) & (time <= t_act * (1.0 + 1e-9))

    if not mask.any():
        return {"status": "SKIP", "message": "RF2 never exceeds 10% of peak"}

    scof = np.abs(rf3[mask]) / rf2_abs[mask]
    scof_mean, scof_std = np.mean(scof), np.std(scof)

    issues = []
    mu_p_dep = float(metadata.get("mu_pressure_dep", 0.0) or 0.0) > 0.5
    if scof_mean < mu_input * 0.95:
        if mu_p_dep:
            # Pressure-dependent (Briscoe) friction: mu(p) = tau0/p + alpha
            # decreases with contact pressure; metadata mu_friction stores the
            # asymptote alpha, so SCOF slightly below it only means the mean
            # pressure did not fully reach the asymptotic regime.
            issues.append("SCOF < mu asymptote (alpha) despite pressure-dependent "
                          "friction -- check the mu(p) table against the actual "
                          "contact pressures")
        else:
            issues.append("SCOF < mu_input — NON-PHYSICAL (ploughing cannot reduce friction)")
    if scof_mean > mu_input + 0.5:
        issues.append("Ploughing term too large")
    if scof_mean > 0 and scof_std / scof_mean > 0.30:
        issues.append("High SCOF scatter (std/mean=%.0f%%)"% (scof_std / scof_mean * 100))

    plough_pct = (scof_mean / mu_input - 1.0) * 100.0 if mu_input > 0 else 0.0

    return {
        "status": "PASS" if not issues else "WARN",
        "mu_input": mu_input,
        "normal_force_source": force_src,
        "scof_mean": scof_mean,
        "scof_std": scof_std,
        "ploughing_contribution_percent": plough_pct,
        "message": (
            "mu_input=%.2f | SCOF=|RF3|/|%s|=%.3f +/- %.3f | ploughing adds %.0f%%. %s"
            % (mu_input, force_src, scof_mean, scof_std, plough_pct,
               "Physically consistent" if not issues else " ; ".join(issues))
        ),
    }


def measure_residual_profile(nodes, metadata, timeseries=None):
    """
    Residual groove geometry -- MEASUREMENT ONLY, no pass/fail.

    Split out of check_full_recovery so that a family whose recovery
    verdict is undefined (viscoelastic: partial, time-dependent recovery)
    can still report its residual depth and pile-up. Returns None when no
    node data is available.

      residual_depth_mm      robust residual (1st percentile of downward disp.)
      residual_depth_raw_mm  deepest node (reference / outlier check)
      pile_up_mm             highest node above the original surface
      reference_mm           commanded depth (displacement mode) or measured
                             peak penetration (force mode)
      relative_percent       residual / reference * 100
      reference_note         provenance of the reference ("" if unambiguous)
    """

    if nodes["deformed"].shape[0] == 0:
        return None

    y_def = nodes["deformed"][:, 1]

    # Robust residual (1st percentile of downward displacements) 
    y_neg = y_def[y_def < 0.0]
    residual = abs(float(np.percentile(y_neg, 1))) if y_neg.size else 0.0
    residual_raw = abs(min(float(np.min(y_def)), 0.0))   # kept for reference
    pile_up = max(float(np.max(y_def)), 0.0)

    # Reference depth: peak commanded depth in displacement mode (valid for
    # the progressive ramp), measured peak penetration in force mode.
    control_mode = str(metadata.get("control_mode", "displacement"))
    if control_mode == "force":
        ref, ref_src = (0.0, "unavailable")
        if timeseries is not None:
            ref, ref_src = _penetration_depth(timeseries, metadata, nodes, at_peak_force=True)
        ref_is_guess = ref < 1e-12
        if ref_is_guess:
            ref = metadata.get("tip_radius", 0.2) * 0.1
            ref_src = "tip_radius guess (no usable measured depth)"
        guess_note = ((" [ref is a guess: %s]" % ref_src) if ref_is_guess else
                      (" [ref = measured peak depth, %s, force-controlled mode]" % ref_src))
    else:
        ref = abs(float(metadata.get("scratch_depth", 0.0)))
        ref_is_guess = ref < 1e-12
        if ref_is_guess:
            ref = metadata.get("tip_radius", 0.2) * 0.1
        guess_note = (" [ref is a guess: scratch_depth absent from metadata]"
                      if ref_is_guess else "")

    return {
        "residual_depth_mm": residual,
        "residual_depth_raw_mm": residual_raw,
        "pile_up_mm": pile_up,
        "reference_mm": ref,
        "relative_percent": residual / ref * 100.0,
        "reference_note": guess_note,
    }


def check_residual_profile(nodes, metadata, timeseries=None):
    """
    Report residual depth and pile-up WITHOUT a recovery verdict.

    For a family whose expected residual is neither ~0 (pure hyperelastic)
    nor permanent (plastic), the groove is still relaxing at the end of the
    recovery step: the measured residual is an UPPER BOUND that depends on
    recovery_time / tau_max. Status is INFO, except when the recovery step
    is absent or too short for the slowest Prony branch.
    """

    m = measure_residual_profile(nodes, metadata, timeseries)
    if m is None:
        return {"status": "SKIP", "message": "No node data"}

    t_rec = float(metadata.get("recovery_time", 0.0) or 0.0)
    tau_max = float(metadata.get("tau_max", 0.0) or 0.0)   # written by Prony_Config.params()

    status, note = "INFO", ""
    if t_rec <= 0.0:
        status = "WARN"
        note = (" NO recovery step (recovery_time=0): unloaded state, not a relaxed one.")
    elif tau_max > 0.0:
        n_tau = t_rec / tau_max
        if n_tau < RECOVERY_TAU_RATIO:
            status = "WARN"
            note = (" recovery_time = %.2f x tau_max (< %.0f): still relaxing, "
                    "residual is an UPPER BOUND, not a converged value."
                    % (n_tau, RECOVERY_TAU_RATIO))
        else:
            note = (" recovery_time = %.1f x tau_max: relaxation essentially complete."
                    % n_tau)

    out = dict(m)
    out["status"] = status
    out["message"] = (
        "Residual depth = %.3e mm (%.1f%% of ref %.3f mm), pile-up = %.3e mm "
        "| raw min = %.3e mm. Measurement only, no recovery verdict.%s%s"
        % (m["residual_depth_mm"], m["relative_percent"], m["reference_mm"],
           m["pile_up_mm"], m["residual_depth_raw_mm"],
           m["reference_note"], note))
    return out


def check_full_recovery(nodes, metadata, is_dissipative=None, timeseries=None):
    """
    Pure Mooney-Rivlin has no dissipation mechanism, so the groove must fully
    recover — residual surface depth ~ 0 once the material has relaxed.
    For dissipative families (plasticity / damage) the logic is inverted: a
    permanent groove is EXPECTED. verify_results passes is_dissipative from the
    family; when run standalone we infer it from the metadata.
    """

    # Geometry is measured by measure_residual_profile() -- single source of
    # truth, shared with check_residual_profile(). Verdict logic below is
    # unchanged.
    m = measure_residual_profile(nodes, metadata, timeseries)
    if m is None:
        return {"status": "SKIP", "message": "No node data"}

    residual     = m["residual_depth_mm"]
    residual_raw = m["residual_depth_raw_mm"]
    pile_up      = m["pile_up_mm"]
    ref          = m["reference_mm"]
    rel          = m["relative_percent"]
    guess_note   = m["reference_note"]



    # Dissipative families (plasticity / damage): pass/fail logic is inverted, groove expected
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
        "checks": ("quasi_static", "hourglass", "energy_total", "artificial_energy",
                   "d1_validity", "force_magnitude", "strain_level",
                   "friction_physics", "steady_state", "settling", "recovery"),
    },
    "semicrystalline_j2": {
        "label": "Soft semicrystalline (linear elastic + J2 plasticity)",
        "dissipative": True,
        "checks": ("quasi_static", "hourglass", "energy_total", "artificial_energy",
                   "force_magnitude", "strain_level",
                   "friction_physics", "steady_state", "settling", "recovery"),
    },
    "glassy_dp": {
        "label": "Glassy amorphous thermoplastic (linear elastic + Drucker-Prager)",
        "dissipative": True,
        "checks": ("quasi_static", "hourglass", "energy_total", "artificial_energy",
                   "force_magnitude", "strain_level",
                   "friction_physics", "steady_state", "settling", "recovery"),
    },
    "elastomer_ve": {
        "label": "Viscoelastic elastomer (Arruda-Boyce + Prony)",
        "dissipative": False,
        # recovery excluded: delayed viscoelastic recovery fits neither the
        # full-recovery nor the permanent-groove logic at finite recovery_time
        "checks": ("quasi_static", "hourglass", "energy_total", "artificial_energy",
                   "d1_validity", "force_magnitude", "strain_level",
                   "friction_physics", "steady_state", "settling", "profile"),
    },
    "glassy_pc": {
        "label": "Polycarbonate (elastic + Drucker-Prager, rate-dependent)",
        "dissipative": True,
        "checks": ("quasi_static", "hourglass", "energy_total", "artificial_energy",
                   "force_magnitude", "strain_level",
                   "friction_physics", "steady_state", "settling", "recovery"),
    },
    "glassy_pmma": {
        "label": "PMMA (elastic + Drucker-Prager, rate-dependent)",
        "dissipative": True,
        "checks": ("quasi_static", "hourglass", "energy_total", "artificial_energy",
                   "force_magnitude", "strain_level",
                   "friction_physics", "steady_state", "settling", "recovery"),
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
        "quasi_static":      ("Quasi-static (KE/IE)",             lambda: check_quasi_static(timeseries, metadata)),
        "hourglass":         ("Hourglass (AE/IE)",                lambda: check_hourglass(timeseries, metadata)),
        "energy_total":      ("Energy total (ETOTAL)",            lambda: check_energy_total(timeseries)),
        "artificial_energy": ("Artificial energy (ALLMW/ALLPW)",  lambda: check_artificial_energy(timeseries)),
        "d1_validity":       ("D1 validity (K/mu window)",        lambda: check_d1_validity(metadata)),
        "force_magnitude":   ("Force magnitude (Hertz/hardness)", lambda: check_force_magnitude(timeseries, metadata, nodes, fam["dissipative"])),
        "strain_level":      ("Strain level",                     lambda: check_strain_level(timeseries, metadata, nodes)),
        "friction_physics":  ("Friction physics (SCOF)",          lambda: check_friction_physics(timeseries, metadata)),
        "steady_state":      ("Steady state (scratch plateau)",   lambda: check_steady_state(timeseries, metadata)),
        "settling":          ("Settling (recovery phase)",        lambda: check_settling(timeseries, metadata)),
        "recovery":          ("Recovery",                         lambda: check_full_recovery(nodes, metadata, fam["dissipative"], timeseries)),
        "profile":           ("Residual profile (depth/pile-up)", lambda: check_residual_profile(nodes, metadata, timeseries)),
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
    control_mode = str(meta.get("control_mode", "displacement"))
    if control_mode == "force":
        print("  Control: force-driven, target = %.4g N" % meta.get("scratch_force", float("nan")))
    else:
        print("  Control: displacement-driven, target depth = %.4g mm" % abs(meta.get("scratch_depth", 0.0)))
    mat_keys = [k for k in ("rho", "he_model", "C10", "C01", "D1",
                            "mu_AB", "lambda_m", "C10_Y", "C20_Y", "C30_Y",
                            "mu1_O", "mu2_O", "alpha1_O", "alpha2_O",
                            "E", "nu", "sigma_y0", "mu_friction", "mu")
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