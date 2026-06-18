# Simulation results verifier.

"""
Reads the CSV output from the post-processor and runs validation checks:
    # 1. Energy balance       — ETOTAL drift (requires ETOTAL in outputs)
    2. Quasi-static check   — ALLKE / ALLIE ratio
    3. Hourglass check      — ALLAE / ALLIE ratio
    # 4. Contact penalty      — ALLPW / ALLIE ratio (requires ALLPW in outputs)
    5. Reaction forces      — magnitude, sign, symmetry (RF1 ≈ 0)
    6. Apparent friction    — RF3/RF2 vs input mu
    7. Time step            — regularity, sudden drops
    8. Groove profile       — residual depth (palier 1: should be ~0)
"""

import numpy as np
import os
import sys

#  CSV Parser
def parse_results_csv(filepath):
    # Parse the post-processor CSV into time-series and node data.

    metadata = {}
    header_cols = []
    ts_rows = []
    node_labels = []
    node_undef = []
    node_def = []

    with open(filepath, "r", encoding="latin-1") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip().replace("\r", "")
        if not line:
            continue

        # Metadata lines
        if line.startswith("#"):
            if "WallclockTime=" in line:
                try:
                    metadata["wallclock"] = float(line.split("=")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            if "Material parameters:" in line:
                metadata["material_str"] = line.split("Material parameters:")[1].strip()
            if "Indenter" in line or "indenter" in line:
                metadata["indenter_str"] = line.lstrip("# ").strip()
            continue

        # Column header line
        if "Time" in line and "RF1" in line:
            header_cols = [c.strip() for c in line.split(",")]
            continue

        # Data lines
        parts = line.split(",")
        if len(parts) < 2:
            continue

        # Time-series columns (non-empty Time field)
        if parts[0].strip():
            try:
                row = {}
                for ci, col in enumerate(header_cols):
                    if ci < len(parts) and parts[ci].strip():
                        row[col] = float(parts[ci])
                ts_rows.append(row)
            except ValueError:
                pass

        # Node data columns (non-empty NodeLabel field)
        label_idx = header_cols.index("NodeLabel") if "NodeLabel" in header_cols else 7
        if label_idx < len(parts) and parts[label_idx].strip():
            try:
                node_labels.append(int(parts[label_idx]))
                xu = float(parts[label_idx + 1])
                yu = float(parts[label_idx + 2])
                zu = float(parts[label_idx + 3])
                xd = float(parts[label_idx + 4])
                yd = float(parts[label_idx + 5])
                zd = float(parts[label_idx + 6])
                node_undef.append([xu, yu, zu])
                node_def.append([xd, yd, zd])
            except (ValueError, IndexError):
                pass

    # Build time-series arrays
    timeseries = {}
    if ts_rows:
        for col in ts_rows[0].keys():
            timeseries[col] = np.array([row.get(col, 0.0) for row in ts_rows])

    # Build node arrays
    nodes = {
        "labels": np.array(node_labels),
        "undeformed": np.array(node_undef) if node_undef else np.empty((0, 3)),
        "deformed": np.array(node_def) if node_def else np.empty((0, 3)),
    }

    return metadata, timeseries, nodes

#  Individual checks

# Thresholds
KE_IE_THRESHOLD = 5.0        # [%]
AE_IE_THRESHOLD = 10.0       # [%]
PW_IE_THRESHOLD = 5.0        # [%]
ETOTAL_DRIFT_THRESHOLD = 1.0  # [%]
SYMMETRY_THRESHOLD = 1.0     # [%] RF1/RF2
FRICTION_TOLERANCE = 0.15    # absolute tolerance on mu_apparent vs mu_input


def check_quasi_static(timeseries):
    """Check ALLKE/ALLIE ratio — must stay below 5%."""
    ke = timeseries.get("ALLKE")
    ie = timeseries.get("ALLIE")
    time = timeseries.get("Time")

    if ke is None or ie is None:
        return {"status": "SKIP", "message": "ALLKE or ALLIE not in outputs"}

    mask = ie > 1e-20
    if not mask.any():
        return {"status": "SKIP", "message": "ALLIE is zero everywhere"}

    ratio = ke[mask] / ie[mask] * 100.0
    max_ratio = np.max(ratio)
    max_idx = np.argmax(ratio)
    time_masked = time[mask] if time is not None else np.arange(len(ratio))

    # Also compute the ratio excluding the first 10% of time (transient at contact onset)
    t_max = time_masked[-1] if len(time_masked) > 0 else 1.0
    steady_mask = time_masked > 0.1 * t_max
    steady_ratio = np.max(ratio[steady_mask]) if steady_mask.any() else max_ratio

    passed = steady_ratio < KE_IE_THRESHOLD
    return {
        "status": "PASS" if passed else "FAIL",
        "max_ratio_percent": max_ratio,
        "max_ratio_time": time_masked[max_idx],
        "steady_state_max_percent": steady_ratio,
        "threshold_percent": KE_IE_THRESHOLD,
        "message": (
            "KE/IE = %.2f%% (steady-state max). %s"
            % (steady_ratio, "OK" if passed else "Ratio too important")
        ),
    }

def check_hourglass(timeseries):
    """Check ALLAE/ALLIE ratio — must stay below 10%."""
    ae = timeseries.get("ALLAE")
    ie = timeseries.get("ALLIE")

    if ae is None or ie is None:
        return {"status": "SKIP", "message": "ALLAE or ALLIE not in outputs"}

    mask = ie > 1e-20
    if not mask.any():
        return {"status": "SKIP", "message": "ALLIE is zero everywhere"}

    ratio = ae[mask] / ie[mask] * 100.0
    max_ratio = np.max(ratio)

    # Final ratio is most representative (cumulative energies)
    final_ratio = ratio[-1]

    passed = final_ratio < AE_IE_THRESHOLD
    return {
        "status": "PASS" if passed else "FAIL",
        "max_ratio_percent": max_ratio,
        "final_ratio_percent": final_ratio,
        "threshold_percent": AE_IE_THRESHOLD,
        "message": (
            "AE/IE = %.2f%% (final). %s"
            % (final_ratio, "OK" if passed else "Hourglass energy too high — refine mesh or check hourglass control")
        ),
    }


def check_contact_penalty(timeseries):
    """Check ALLPW/ALLIE ratio — must stay below 5%."""
    pw = timeseries.get("ALLPW")
    ie = timeseries.get("ALLIE")

    if pw is None:
        return {"status": "SKIP", "message": "ALLPW not in outputs — add it to history_energy_variables"}

    mask = ie > 1e-20
    if not mask.any():
        return {"status": "SKIP", "message": "ALLIE is zero everywhere"}

    ratio = np.abs(pw[mask]) / ie[mask] * 100.0
    final_ratio = ratio[-1]

    passed = final_ratio < PW_IE_THRESHOLD
    return {
        "status": "PASS" if passed else "FAIL",
        "final_ratio_percent": final_ratio,
        "threshold_percent": PW_IE_THRESHOLD,
        "message": (
            "PW/IE = %.2f%% (final). %s"
            % (final_ratio, "OK" if passed else "Contact penetration excessive — check contact formulation")
        ),
    }


def check_energy_balance(timeseries):
    """Check ETOTAL drift — must stay within 1%."""
    etotal = timeseries.get("ETOTAL")

    if etotal is None:
        return {"status": "SKIP", "message": "ETOTAL not in outputs — add it to history_energy_variables"}

    # Reference: max absolute value of ETOTAL
    ref = np.max(np.abs(etotal))
    if ref < 1e-30:
        return {"status": "PASS", "message": "ETOTAL is effectively zero — no energy in system yet"}

    drift = np.abs(etotal[-1] - etotal[0]) / ref * 100.0

    passed = drift < ETOTAL_DRIFT_THRESHOLD
    return {
        "status": "PASS" if passed else "FAIL",
        "drift_percent": drift,
        "threshold_percent": ETOTAL_DRIFT_THRESHOLD,
        "message": (
            "ETOTAL drift = %.3f%%. %s"
            % (drift, "OK" if passed else "Energy not conserved — numerical instability")
        ),
    }


def check_reaction_forces(timeseries):
    """Check reaction force magnitudes, signs, and symmetry."""
    rf1 = timeseries.get("RF1")
    rf2 = timeseries.get("RF2")
    rf3 = timeseries.get("RF3")

    if rf1 is None or rf2 is None or rf3 is None:
        return {"status": "SKIP", "message": "RF1/RF2/RF3 not all in outputs"}

    results = {}

    # RF2 should be negative (indenter pushing down into surface, reaction is upward = negative in Abaqus convention)
    rf2_min = np.min(rf2)
    rf2_max = np.max(rf2)
    results["RF2_min"] = rf2_min
    results["RF2_max"] = rf2_max
    results["RF2_sign_ok"] = rf2_min < 0

    # RF3 should be positive (friction opposes scratch direction)
    rf3_max = np.max(rf3)
    results["RF3_max"] = rf3_max

    # Symmetry: RF1 should be near zero
    rf2_scale = np.max(np.abs(rf2))
    if rf2_scale > 1e-20:
        rf1_ratio = np.max(np.abs(rf1)) / rf2_scale * 100.0
        symmetry_ok = rf1_ratio < SYMMETRY_THRESHOLD
        results["RF1_RF2_max_percent"] = rf1_ratio
        results["symmetry_ok"] = symmetry_ok
    else:
        rf1_ratio = 0.0
        symmetry_ok = True
        results["RF1_RF2_max_percent"] = 0.0
        results["symmetry_ok"] = True

    # Hertz estimate for order-of-magnitude check (Mooney-Rivlin small-strain)
    # This is a rough estimate — user should compare with their specific parameters
    results["message"] = (
        "RF2 range: [%.3e, %.3e] N | RF3 max: %.3e N | "
        "RF1/RF2 symmetry: %.2f%% %s"
        % (rf2_min, rf2_max, rf3_max, rf1_ratio,
           "OK" if symmetry_ok else "— ASYMMETRY DETECTED, check BCs")
    )
    results["status"] = "PASS" if (results["RF2_sign_ok"] and symmetry_ok) else "WARN"

    return results


def check_apparent_friction(timeseries, mu_input=0.3):
    """Compare apparent friction coefficient RF3/RF2 with input mu."""
    rf2 = timeseries.get("RF2")
    rf3 = timeseries.get("RF3")
    time = timeseries.get("Time")

    if rf2 is None or rf3 is None:
        return {"status": "SKIP", "message": "RF2 or RF3 not in outputs"}

    # Only compute where RF2 is significant (indenter in contact)
    rf2_abs = np.abs(rf2)
    threshold = np.max(rf2_abs) * 0.10
    mask = rf2_abs > threshold

    if not mask.any():
        return {"status": "SKIP", "message": "RF2 never exceeds 10% of peak — no significant contact"}

    apparent_mu = np.abs(rf3[mask]) / rf2_abs[mask]
    mu_mean = np.mean(apparent_mu)
    mu_std = np.std(apparent_mu)
    mu_min = np.min(apparent_mu)
    mu_max = np.max(apparent_mu)

    # Apparent mu should be >= input mu (ploughing adds to friction)
    # and not wildly different
    deviation = abs(mu_mean - mu_input)
    reasonable = deviation < FRICTION_TOLERANCE and mu_mean >= mu_input * 0.5

    return {
        "status": "PASS" if reasonable else "WARN",
        "mu_input": mu_input,
        "mu_apparent_mean": mu_mean,
        "mu_apparent_std": mu_std,
        "mu_apparent_range": (mu_min, mu_max),
        "message": (
            "mu_input=%.2f | mu_apparent=%.3f +/- %.3f [%.3f, %.3f]. %s"
            % (mu_input, mu_mean, mu_std, mu_min, mu_max,
               "OK — ploughing adds ~%.0f%% to interfacial friction" % ((mu_mean / mu_input - 1) * 100)
               if mu_mean > mu_input
               else "OK" if reasonable
               else "WARNING — apparent friction far from input, check contact")
        ),
    }


def check_time_stepping(timeseries):
    """Check for time step regularity and sudden drops."""
    time = timeseries.get("Time")

    if time is None or len(time) < 3:
        return {"status": "SKIP", "message": "Not enough time points"}

    dt = np.diff(time)
    dt_positive = dt[dt > 0]

    if len(dt_positive) < 2:
        return {"status": "SKIP", "message": "Cannot compute time step statistics"}

    dt_mean = np.mean(dt_positive)
    dt_min = np.min(dt_positive)
    dt_max = np.max(dt_positive)
    dt_ratio = dt_max / dt_min if dt_min > 0 else float("inf")

    # Note: these are output intervals, not solver increments
    # A large ratio suggests the output interval changed between steps (expected)
    # or that the solver struggled in some region

    return {
        "status": "INFO",
        "dt_mean": dt_mean,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "dt_ratio_max_min": dt_ratio,
        "n_points": len(time),
        "time_range": (time[0], time[-1]),
        "message": (
            "%d output points over [%.4e, %.4e] s | "
            "dt range: [%.3e, %.3e] s (ratio %.1f)"
            % (len(time), time[0], time[-1], dt_min, dt_max, dt_ratio)
        ),
    }


def check_recovery(nodes, scratch_depth=0.020):
    """
    Compute material recovery percentage after the scratch test.
    recovery_percent = (scratch_depth - residual_depth) / scratch_depth * 100
    scratch_depth : float — imposed scratch depth [mm] (positive, from Scratch_Config)
    """
    if nodes["deformed"].shape[0] == 0:
        return {"status": "SKIP", "message": "No node data"}

    y_def = nodes["deformed"][:, 1]
    residual_depth = abs(np.min(y_def))
    recovery_percent = (scratch_depth - residual_depth) / scratch_depth * 100.0
    recovery_percent = max(0.0, min(100.0, recovery_percent))

    return {
        "status": "INFO",
        "scratch_depth_mm": scratch_depth,
        "residual_depth_mm": residual_depth,
        "recovery_percent": recovery_percent,
        "n_nodes": len(y_def),
        "message": (
            "Residual depth=%.4e mm | Imposed depth=%.4e mm => Recovery=%.1f%%"
            % (residual_depth, scratch_depth, recovery_percent)
        ),
    }


# ============================================================
#  Master verification
# ============================================================

def verify_results(filepath, mu_input=0.3, scratch_depth=0.020, print_report=True):
    """
    Run all verification checks on a results CSV.

    Parameters
    ----------
    filepath     : str   — path to the _Results.csv file
    mu_input     : float — interfacial friction coefficient from Friction_Config
    print_report : bool  — if True, prints a formatted report

    Returns
    -------
    report : dict — all check results keyed by check name
    """
    if not os.path.exists(filepath):
        raise IOError("File not found: %s" % filepath)

    metadata, timeseries, nodes = parse_results_csv(filepath)

    report = {
        "file": filepath,
        "metadata": metadata,
        "checks": {},
    }

    # Run all checks
    checks = [
        ("Quasi-static (KE/IE)",      check_quasi_static(timeseries)),
        ("Hourglass (AE/IE)",          check_hourglass(timeseries)),
        ("Contact penalty (PW/IE)",    check_contact_penalty(timeseries)),
        ("Energy balance (ETOTAL)",    check_energy_balance(timeseries)),
        ("Reaction forces",            check_reaction_forces(timeseries)),
        ("Apparent friction",          check_apparent_friction(timeseries, mu_input)),
        ("Time stepping",              check_time_stepping(timeseries)),
        ("Recovery",                   check_recovery(nodes, scratch_depth)),
    ]

    for name, result in checks:
        report["checks"][name] = result

    if print_report:
        _print_report(report)

    return report


def _print_report(report):
    """Print a formatted verification report."""

    print("")
    print("=" * 72)
    print("  SIMULATION QUALITY REPORT")
    print("=" * 72)
    print("  File: %s" % report["file"])
    meta = report["metadata"]
    if "wallclock" in meta:
        print("  Wallclock: %.1f s" % meta["wallclock"])
    if "material_str" in meta:
        print("  Material: %s" % meta["material_str"])
    print("-" * 72)

    n_pass = 0
    n_fail = 0
    n_warn = 0
    n_skip = 0

    for name, result in report["checks"].items():
        status = result.get("status", "?")
        message = result.get("message", "")

        if status == "PASS":
            icon = "OK"
            n_pass += 1
        elif status == "FAIL":
            icon = "FAIL"
            n_fail += 1
        elif status == "WARN":
            icon = "WARN"
            n_warn += 1
        elif status == "SKIP":
            icon = "SKIP"
            n_skip += 1
        else:
            icon = "INFO"

        print("")
        print("  [%4s]  %s" % (icon, name))
        print("          %s" % message)

#  CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    filepath = sys.argv[1]
    mu = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    verify_results(filepath, mu_input=mu)