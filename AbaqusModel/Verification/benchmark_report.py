# -*- coding: utf-8 -*-
"""
CPython analysis of the benchmark outputs: Result are compared with a known reference.

    python3 -m ScratchSimulation.AbaqusModel.Verification.benchmark_report hertz runs/Bench_hertz_glassy_pc/BenchOutputs
    python3 -m ...benchmark_report level0 runs/Bench_level0_glassy_pc/BenchOutputs
    python3 -m ...benchmark_report time   runs/Bench_hertz_time_glassy_pc/BenchOutputs
    python3 -m ...benchmark_report scratch runs/MeshConvergence_glassy_pc_mesh9/SimDataOutputs

NB: The last mode re-analyses CSVs and produces:
  * the asymptotic-range gate on the mesh ladder (is a GCI legitimate at all?)
  * the contact-area inversion of the Briscoe SCOF
  * the amplitude-smoothing confound check on the IndenterU2 trace
"""

from __future__ import print_function

import glob
import os
import re
import sys

import numpy as np

try:
    from . import analytic as an
    from . import material_point as mp
except (ImportError, ValueError):      # direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from ScratchSimulation.AbaqusModel.Verification import analytic as an
    from ScratchSimulation.AbaqusModel.Verification import material_point as mp



# CSV reading
def read_csv(path):
    meta, header, rows = {}, None, []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            if line.startswith("#"):
                for k, v in re.findall(r"([A-Za-z_]\w*)\s*=\s*([^\s,]+)", line[1:]):
                    try:
                        meta[k] = float(v)
                    except ValueError:
                        meta[k] = v
                continue
            parts = [p.strip() for p in line.split(",")]
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            rows.append(parts)
    data = {}
    if header:
        for i, col in enumerate(header):
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[i]) if r[i] != "" else np.nan)
                except (ValueError, IndexError):
                    vals.append(np.nan)
            data[col] = np.array(vals)
    return meta, data


def _finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]



# Level 1 -- Hertz

def _col_stats(name, arr):
    if arr is None:
        return "%-16s ABSENT" % name
    a = np.asarray(arr, dtype=float)
    fin = np.isfinite(a)
    if not fin.any():
        return "%-16s %d rows, ALL NaN/empty" % (name, a.size)
    v = a[fin]
    return ("%-16s %d rows (%d finite)  min=%.6g  max=%.6g  |max|=%.6g"
            % (name, a.size, fin.sum(), v.min(), v.max(), np.abs(v).max()))


def _raise_empty_diagnostic(path, meta, d):
    """
    Fail with something actionable instead of an IndexError three frames up.

    An empty curve after filtering means the run produced no usable
    (depth, force) pair. The cause is almost always one of: the job never made
    contact, or the ODB history keys did not match what the extractor looks for
    so a column came out as zeros. Printing the actual column ranges settles
    which, in one look.
    """
    lines = ["", "=" * 74,
             "EMPTY CURVE after filtering: %s" % os.path.basename(path),
             "=" * 74,
             "No sample survived  isfinite(U2) & isfinite(RF2) & (U2 > 1e-6).",
             "", "Columns as read from the CSV:"]
    for key in ("Time", "IndenterU2", "RF2", "RF3", "CFN2", "CAREA",
                "SUB_ALLIE", "WM_ALLIE"):
        lines.append("  " + _col_stats(key, d.get(key)))
    lines += ["", "Relevant metadata:"]
    for key in ("family", "E", "nu", "tip_radius", "fine_size_x",
                "depth_max", "ramp_time", "hold_time",
                "mass_scale", "target_time_increment"):
        if key in meta:
            lines.append("  %-22s = %s" % (key, meta[key]))
    lines += ["",
              "Read it like this:",
              "  * IndenterU2 all zeros  -> the ODB history key for the",
              "    reference-point displacement did not match; the extractor",
              "    wrote zeros. The depth can be reconstructed (see the",
              "    fallback above) but the ODB key should be fixed.",
              "  * RF2 all zeros with a non-zero U2 -> the indenter moved but",
              "    never touched: check the initial gap and the contact pair.",
              "  * Both non-zero but |U2|max < 1e-6 mm -> the amplitude did not",
              "    span the step (the SmoothStep data is in step SECONDS).",
              "=" * 74, ""]
    raise SystemExit("\n".join(lines))


def inspect_csv(path):
    """CLI helper: dump every column's range so a bad run can be diagnosed
    without opening the file by hand."""
    meta, d = read_csv(path)
    print("\n=== %s ===" % os.path.basename(path))
    print("metadata:")
    for k in sorted(meta):
        print("  %-24s = %s" % (k, meta[k]))
    print("columns:")
    for k in d:
        print("  " + _col_stats(k, d[k]))


def analyse_hertz_run(path, contact_path=None):
    """
    One Hertz indentation run vs the closed form, over the WHOLE ramp.

    Because a = sqrt(R h) grows monotonically, a single job sweeps
    N_a = a / h_mesh continuously: the output is an error-vs-resolution CURVE,
    not one point.
    """
    meta, d = read_csv(path)
    E = float(meta.get("E", meta.get("E_0", np.nan)))
    nu = float(meta.get("nu", np.nan))
    if not np.isfinite(E):
        raise ValueError("%s: elastic constants (E, nu) absent from the metadata; "
                         "the material card was not linear elastic." % path)
    R = float(meta.get("tip_radius", 0.2))
    h_mesh = float(meta.get("fine_size_x", np.nan))

    t = d.get("Time")
    u2 = np.abs(d.get("IndenterU2"))
    rf2 = np.abs(d.get("RF2"))

    # The prescribed depth is KNOWN exactly (SmoothStep ramp of depth_max over
    # ramp_time), so if the U2 history came back degenerate -- all zeros, which
    # happens when the reference-point displacement key in the ODB does not
    # match what _pick() looks for -- we can reconstruct it instead of failing.
    u2_usable = (u2 is not None and np.isfinite(u2).any()
                 and np.nanmax(u2) > 1e-6)
    if not u2_usable:
        dmax = float(meta.get("depth_max", np.nan))
        tramp = float(meta.get("ramp_time", np.nan))
        if np.isfinite(dmax) and np.isfinite(tramp) and tramp > 0 \
                and t is not None and np.isfinite(t).any():
            xi = np.clip(np.asarray(t, dtype=float) / tramp, 0.0, 1.0)
            # Abaqus SMOOTH STEP quintic: a = xi^3 (10 - 15 xi + 6 xi^2)
            u2 = dmax * xi ** 3 * (10.0 - 15.0 * xi + 6.0 * xi ** 2)
            print("  ! %s: IndenterU2 was all zeros; depth reconstructed from "
                  "the prescribed SmoothStep ramp (depth_max=%.4g mm over "
                  "%.4g s). Check the ODB history key for U2."
                  % (os.path.basename(path), dmax, tramp))
        else:
            _raise_empty_diagnostic(path, meta, d)

    ok = np.isfinite(u2) & np.isfinite(rf2) & (u2 > 1e-6)
    depth, force = u2[ok], rf2[ok]

    p_exact = an.hertz_force(E, nu, R, depth, half_model=True)
    good = p_exact > 1e-12
    depth, force, p_exact = depth[good], force[good], p_exact[good]
    if depth.size == 0:
        _raise_empty_diagnostic(path, meta, d)
    err = 100.0 * (force - p_exact) / p_exact
    a = an.hertz_contact_radius(R, depth)
    N_a = a / h_mesh

    # resolution at which |error| first (and durably) drops below a threshold.
    # The first 10 % of the ramp is skipped: there P is a few micronewtons, the
    # contact spans one or two elements, and the RELATIVE error is dominated by
    # discretisation of the contact perimeter rather than by anything the
    # criterion is meant to measure.
    i0 = max(int(0.10 * depth.size), 1)

    def _n_at(thr):
        m = np.abs(err) <= thr
        m[:i0] = False
        if not m.any():
            return None
        # require the criterion to hold for the rest of the ramp, so a single
        # sign change on the way through zero is not credited as convergence
        for i in np.where(m)[0]:
            if m[i:].mean() > 0.9:
                return float(N_a[i])
        return None

    out = {
        "file": os.path.basename(path), "E": E, "nu": nu, "R": R,
        "h_mesh": h_mesh,
        "depth_max_mm": float(depth.max()) if depth.size else np.nan,
        "N_a_max": float(N_a.max()) if N_a.size else np.nan,
        "err_at_max_depth_pct": float(err[-1]) if err.size else np.nan,
        "N_a_for_5pct": _n_at(5.0),
        "N_a_for_2pct": _n_at(2.0),
        "N_a_for_1pct": _n_at(1.0),
        "hertz_validity": an.hertz_validity(R, float(depth.max()) if depth.size else 0.0),
        "curve": {"depth": depth, "N_a": N_a, "P_num": force,
                  "P_exact": p_exact, "err_pct": err},
    }

    # ---- energy: ALLIE must equal (2/5) P h, ALLPW must vanish -----------
    ie = d.get("SUB_ALLIE", d.get("WM_ALLIE"))
    pw = d.get("WM_ALLPW")
    ke = d.get("SUB_ALLKE")
    if ie is not None:
        ie_f = np.asarray(ie, dtype=float)[ok][good]
        u_exact = an.hertz_strain_energy(E, nu, R, depth, half_model=True)
        m = u_exact > 1e-15
        if m.any():
            out["energy_err_pct"] = float(100.0 * (ie_f[m][-1] - u_exact[m][-1])
                                          / u_exact[m][-1])
    if pw is not None and ie is not None:
        ief = _finite(ie)
        pwf = _finite(pw)
        if ief.size and pwf.size and ief.max() > 0:
            out["ALLPW_over_ALLIE_pct"] = float(100.0 * np.abs(pwf).max() / ief.max())
    if ke is not None and ie is not None:
        ief, kef = _finite(ie), _finite(ke)
        if ief.size and kef.size and ief.max() > 0:
            out["ALLKE_over_ALLIE_pct"] = float(100.0 * kef.max() / ief.max())

    # ---- contact radius: a_num vs sqrt(R h) ------------------------------
    if contact_path is None:
        guess = path.replace("_indent.csv", "_contact.csv")
        contact_path = guess if os.path.exists(guess) else None
    if contact_path:
        cm, cd = read_csv(contact_path)
        cu2 = np.abs(cd.get("IndenterU2", np.array([])))
        anum = cd.get("a_num", np.array([]))
        m = np.isfinite(cu2) & np.isfinite(anum) & (cu2 > 1e-6) & (anum > 0)
        if m.any():
            a_ex = an.hertz_contact_radius(R, cu2[m])
            out["contact_radius"] = {
                "a_num_final": float(anum[m][-1]),
                "a_exact_final": float(a_ex[-1]),
                "err_pct_final": float(100.0 * (anum[m][-1] - a_ex[-1]) / a_ex[-1]),
                "err_pct_all": 100.0 * (anum[m] - a_ex) / a_ex,
            }
    return out


def analyse_hertz_ladder(folder, pattern="Hertz_*_indent.csv"):
    """Mesh ladder: observed order, asymptotic-range gate, N_a criterion."""
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise SystemExit("No file matching %s in %s" % (pattern, folder))
    runs = [analyse_hertz_run(f) for f in files]
    runs.sort(key=lambda r: -r["h_mesh"])           # coarse -> fine
    h = [r["h_mesh"] for r in runs]
    f = [float(r["curve"]["P_num"][-1]) for r in runs]
    exact = float(runs[0]["curve"]["P_exact"][-1])
    rep = {
        "runs": runs, "h": h, "P": f, "P_exact": exact,
        "asymptotic": an.asymptotic_range_check(h, f) if len(h) >= 3 else None,
        "richardson": an.richardson_extrapolate(h, f) if len(h) >= 3 else None,
        "gci_pct": an.gci(h, f) if len(h) >= 3 else None,
    }
    if rep["richardson"] is not None and exact > 0:
        rep["richardson_vs_exact_pct"] = 100.0 * (rep["richardson"] - exact) / exact
    return rep


def print_hertz_ladder(rep):
    print("\n=== HERTZ MESH LADDER (level 1a) " + "=" * 45)
    print("%-10s %-8s %-14s %-14s %-9s %-8s %-8s"
          % ("h [mm]", "N_a", "P_num [N]", "P_exact [N]", "err [%]",
             "Na@2%", "Na@1%"))
    for r in rep["runs"]:
        c = r["curve"]
        print("%-10.4g %-8.2f %-14.6g %-14.6g %-9.2f %-8s %-8s"
              % (r["h_mesh"], r["N_a_max"], c["P_num"][-1], c["P_exact"][-1],
                 r["err_at_max_depth_pct"],
                 ("%.1f" % r["N_a_for_2pct"]) if r["N_a_for_2pct"] else "n/a",
                 ("%.1f" % r["N_a_for_1pct"]) if r["N_a_for_1pct"] else "n/a"))
    a = rep.get("asymptotic")
    if a:
        print("\nAsymptotic range : %s" % a["verdict"])
        print("  increments     : %s" % ", ".join("%.4g" % x for x in a["increments"]))
        print("  observed order : %s"
              % ("%.2f" % a["observed_order"] if a["observed_order"] else "unresolvable"))
    if rep.get("richardson") is not None:
        print("  Richardson     : %.6g N  (exact %.6g N, %.2f %%)"
              % (rep["richardson"], rep["P_exact"],
                 rep.get("richardson_vs_exact_pct", float("nan"))))
        print("  GCI (fine)     : %s"
              % ("%.2f %%" % rep["gci_pct"] if rep["gci_pct"] else "not legitimate"))
    else:
        print("  Richardson/GCI : NOT legitimate on this ladder.")
    for r in rep["runs"]:
        extras = []
        for k, lab in (("ALLPW_over_ALLIE_pct", "ALLPW/ALLIE"),
                       ("ALLKE_over_ALLIE_pct", "ALLKE/ALLIE"),
                       ("energy_err_pct", "ALLIE vs (2/5)Ph")):
            if k in r:
                extras.append("%s=%.2f%%" % (lab, r[k]))
        if "contact_radius" in r:
            extras.append("a_num err=%.1f%%" % r["contact_radius"]["err_pct_final"])
        if extras:
            print("  h=%-8.4g %s" % (r["h_mesh"], " | ".join(extras)))
    print("\nRead this as: the error floor reachable on a LINEAR ELASTIC problem")
    print("with a smooth sphere and an exact solution is a LOWER BOUND on the")
    print("error of the 40 um scratch. No tighter claim can be made downstream.")


# --------------------------------------------------------------------------
# Level 1b -- time / mass-scaling ladders
# --------------------------------------------------------------------------

def analyse_time_ladder(folder, pattern="HertzT*_indent.csv"):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise SystemExit("No file matching %s in %s" % (pattern, folder))
    rows = []
    for path in files:
        meta, d = read_csv(path)
        r = analyse_hertz_run(path)
        rows.append({
            "file": os.path.basename(path),
            "T": float(meta.get("ramp_time", np.nan)),
            "target_dt": float(meta.get("target_time_increment", 0.0)),
            "mass_scale": float(meta.get("mass_scale", 1.0)),
            "P": float(r["curve"]["P_num"][-1]),
            "P_exact": float(r["curve"]["P_exact"][-1]),
            "err_pct": r["err_at_max_depth_pct"],
            "KE_IE_pct": r.get("ALLKE_over_ALLIE_pct"),
            "PW_IE_pct": r.get("ALLPW_over_ALLIE_pct"),
        })
    rows.sort(key=lambda x: -x["T"])
    ref = rows[0]["P"]
    for x in rows:
        x["drift_vs_slowest_pct"] = 100.0 * (x["P"] - ref) / ref
    return rows


def print_time_ladder(rows, title="TIME LADDER (level 1b)"):
    print("\n=== %s " % title + "=" * (60 - len(title)))
    print("%-10s %-12s %-13s %-11s %-10s %-10s %-10s"
          % ("T [s]", "dt_target", "P_num [N]", "err vs Hertz",
             "drift [%]", "KE/IE %", "PW/IE %"))
    for x in rows:
        print("%-10.4g %-12.3e %-13.6g %-11.2f %-10.2f %-10s %-10s"
              % (x["T"], x["target_dt"], x["P"], x["err_pct"],
                 x["drift_vs_slowest_pct"],
                 "%.2f" % x["KE_IE_pct"] if x["KE_IE_pct"] is not None else "n/a",
                 "%.2f" % x["PW_IE_pct"] if x["PW_IE_pct"] is not None else "n/a"))
    span = max(abs(x["drift_vs_slowest_pct"]) for x in rows)
    print("\nDecision table:")
    print("  drift ~ the 14 % seen on the scratch  -> purely contact/inertia/mass")
    print("     scaling; the plastic pile-up hypothesis is REFUTED.")
    print("  drift ~ 0                             -> the mechanism requires")
    print("     plastic flow; keep the pile-up hypothesis and test it directly.")
    print("  measured span here: %.2f %%" % span)


# --------------------------------------------------------------------------
# Level 0 -- single element
# --------------------------------------------------------------------------

def _detect_yield_by_slope(le, s, E, frac=0.5, smooth=3):
    """
    Yield stress as the point where the tangent stiffness ds/d(le) has fallen
    below `frac` of its initial elastic value.

    Robust where the old "departure from s = E*le" test is not: under
    nlgeom=YES the elastic branch is not perfectly straight (finite strain,
    and near-incompressible kinematics at high nu), so an absolute-line test
    fires on the first increment. The tangent slope stays ~E through the
    elastic range whatever nu is, then drops sharply to the hardening slope at
    yield -- which is exactly the transition we want to locate.

    Returns the true stress at that strain, or None if no clear drop is seen
    (a genuinely elastic run, e.g. an applied strain below yield).
    """
    le = np.asarray(le, dtype=float)
    s = np.asarray(s, dtype=float)
    if le.size < 6:
        return None
    order = np.argsort(le)
    le, s = le[order], s[order]
    # Collapse duplicate strain values before differentiating. Explicit writes
    # several frames at (nearly) the same strain early in the ramp; np.gradient
    # then divides by a zero spacing and floods stderr with RuntimeWarnings.
    # Averaging s over each unique strain removes the singularity without
    # changing the slope anywhere it matters.
    uniq, inv = np.unique(np.round(le, 12), return_inverse=True)
    if uniq.size < le.size:
        s_avg = np.zeros_like(uniq)
        cnt = np.zeros_like(uniq)
        np.add.at(s_avg, inv, s)
        np.add.at(cnt, inv, 1.0)
        s = s_avg / np.maximum(cnt, 1.0)
        le = uniq
    if le.size < 6:
        return None
    # local slope by finite difference, lightly smoothed
    dsl = np.gradient(s, le)
    if smooth > 1 and dsl.size >= smooth:
        k = np.ones(smooth) / smooth
        dsl = np.convolve(dsl, k, mode="same")
    # reference elastic slope: median of the slopes below 0.3 % strain, or the
    # first few points if the ramp is coarse. Guard against a zero/negative.
    ref_mask = le < 3e-3
    ref = np.median(dsl[ref_mask]) if ref_mask.sum() >= 3 else np.median(dsl[:5])
    if not np.isfinite(ref) or ref <= 0:
        ref = float(E) if np.isfinite(E) and E > 0 else np.median(dsl[:5])
    below = np.where(dsl < frac * ref)[0]
    # ignore the very first couple of points (slope noise at s ~ 0)
    below = below[below >= 2]
    if below.size == 0:
        return None
    return float(s[below[0]])


def analyse_single_element(path):
    meta, d = read_csv(path)
    mode = str(meta.get("element_mode", "?"))
    E = float(meta.get("E", np.nan))
    nu = float(meta.get("nu", np.nan))
    sy0 = float(meta.get("sigma_y0", np.nan))
    beta = float(meta.get("friction_angle", np.nan))
    K = float(meta.get("flow_stress_ratio", np.nan))

    out = {"file": os.path.basename(path), "mode": mode, "E": E, "nu": nu}

    if mode in ("tension", "compression"):
        le = np.abs(d.get("LE33"))
        s = np.abs(d.get("S33"))
        m = np.isfinite(le) & np.isfinite(s)
        le, s = le[m], s[m]
        if le.size < 5:
            out["status"] = "SKIP"
            return out
        # Elastic slope. The window is the smaller of "below 0.2 % strain" and
        # "the first 5 % of the samples", so the fit degrades gracefully when a
        # run was written with few frames instead of silently disappearing.
        el = le < 2e-3
        if el.sum() < 3:
            # Few frames: fall back to the samples that are still linear, never
            # to a fixed fraction (which would reach past yield on a coarse
            # frame count and report a spuriously low modulus).
            lin = np.abs(s - s[0] - (s[1] - s[0]) / max(le[1] - le[0], 1e-30)
                         * (le - le[0])) / np.maximum(np.abs(s), 1e-9) < 0.02
            el = np.zeros_like(le, dtype=bool)
            n_lin = int(np.argmin(lin)) if (~lin).any() else le.size
            el[:max(3, min(n_lin, le.size))] = True
        if el.sum() >= 3 and le[el].max() > 0:
            slope = float(np.polyfit(le[el], s[el], 1)[0])
            out["E_measured"] = slope
            out["E_fit_strain_max"] = float(le[el].max())
            out["E_err_pct"] = 100.0 * (slope - E) / E if np.isfinite(E) else None
        out["sigma_at_max_strain"] = float(s[-1])
        out["strain_max"] = float(le[-1])
        if np.isfinite(sy0) and np.isfinite(beta) and np.isfinite(K):
            ratio = mp.dp_tension_over_compression(beta, K)
            out["dp_tc_ratio_expected"] = ratio
            out["dp_beta_implied_by_tc"] = beta
            out["flow_stress_ratio"] = K
            out["sigma_y_expected"] = sy0 * (1.0 if mode == "compression" else ratio)
            # Yield detection by TANGENT-STIFFNESS DROP, not by departure from a
            # theoretical elastic line s = E*le. Under nlgeom=YES the true-stress
            # vs log-strain curve is NOT exactly linear in the elastic range
            # (finite-strain and, for high nu like the 0.42 of semicrystalline_dp,
            # near-incompressible kinematics), so the old |s - E*le|/s > 2 %
            # test fired on the FIRST increment where s ~ 1e-6 MPa and reported a
            # nonsense "yield" of ~1e-6 with -100 %. The tangent slope, by
            # contrast, is ~E while elastic and collapses to the hardening slope
            # at yield regardless of nu. We look for the strain where the local
            # slope has fallen to a fraction of its initial elastic value.
            sy = _detect_yield_by_slope(le, s, E)
            if sy is not None:
                out["sigma_y_measured"] = float(sy)
                out["sigma_y_err_pct"] = 100.0 * (sy - out["sigma_y_expected"]) \
                    / out["sigma_y_expected"]
    elif mode == "shear":
        g = np.abs(d.get("LE12"))
        tau = np.abs(d.get("S12"))
        peeq = d.get("PEEQ")
        m = np.isfinite(g) & np.isfinite(tau)
        g, tau = g[m], tau[m]
        if g.size >= 5:
            out["tau_max"] = float(tau[-1])
            out["gamma_max"] = float(g[-1])
            # Did the element actually yield? PEEQ is the ground truth. If it
            # never leaves zero, the applied shear strain stayed below the
            # shear yield and tau_max is a purely ELASTIC value -- comparing it
            # with the card's shear yield is meaningless, so say so instead.
            peeq_max = float(np.nanmax(peeq)) if peeq is not None \
                and np.isfinite(peeq).any() else 0.0
            out["peeq_max"] = peeq_max
            out["yielded"] = bool(peeq_max > 1e-6)
            if np.isfinite(sy0) and np.isfinite(beta) and np.isfinite(K):
                out["tau_y_expected"] = mp.dp_shear_yield(sy0, beta, K)
                # Measured shear yield by the same tangent-drop detector used
                # for tension/compression, but only trust it if PEEQ confirms
                # plasticity actually occurred.
                tau_y = _detect_yield_by_slope(g, tau, mp.elastic_moduli_from_E_nu(
                    E, nu)["G"] if np.isfinite(E) and np.isfinite(nu) else None)
                if out["yielded"] and tau_y is not None:
                    out["tau_y_measured"] = float(tau_y)
                    out["tau_y_err_pct"] = 100.0 * (tau_y - out["tau_y_expected"]) \
                        / out["tau_y_expected"]
                # Strain needed to reach yield, to advise the suite if short.
                if np.isfinite(E) and np.isfinite(nu):
                    G = mp.elastic_moduli_from_E_nu(E, nu)["G"]
                    out["gamma_yield_needed"] = out["tau_y_expected"] / G
    out["status"] = "OK"
    return out


# --------------------------------------------------------------------------
# Level 0 -- hyperelastic and viscoelastic (Prony) checks
# --------------------------------------------------------------------------

def _parse_prony(meta):
    """Rebuild the Prony table written by the extractor as
    'prony_table=g:k:tau;g:k:tau;...' into [(g,k,tau), ...]."""
    raw = meta.get("prony_table")
    if not isinstance(raw, str) or not raw:
        return None
    table = []
    for triple in raw.split(";"):
        parts = triple.split(":")
        if len(parts) == 3:
            try:
                table.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                pass
    return table or None


def _he_true_stress(meta, model, lam):
    """Analytic uniaxial true stress for the hyperelastic MODEL in the header,
    reading the family's own constants out of the CSV metadata."""
    g = lambda *keys: next((float(meta[k]) for k in keys
                            if k in meta and isinstance(meta[k], float)), None)
    if model == "arruda_boyce":
        mu = g("mu_AB", "mu")
        lam_m = g("lambda_m", "lambdaL", "lambda_L")
        if mu is None or lam_m is None:
            return None
        return mp.arruda_boyce_uniaxial(mu, lam_m, lam)
    if model == "mooney_rivlin":
        c10, c01 = g("C10", "c10"), g("C01", "c01")
        if c10 is None or c01 is None:
            return None
        return mp.mooney_rivlin_uniaxial(c10, c01, lam)
    return None


def analyse_hyperelastic(path):
    """
    Uniaxial true-stress vs stretch, compared with the analytic hyperelastic
    law of the family. This is the transcription check for an elastomer: yield
    and T/C are meaningless here, but the whole stress-stretch curve is a
    stringent test of the card (a wrong mu or lambda_m shows up immediately).
    """
    meta, d = read_csv(path)
    model = str(meta.get("he_model", "?"))
    mode = str(meta.get("element_mode", "?"))
    out = {"file": os.path.basename(path), "mode": mode, "he_model": model,
           "status": "OK"}

    le = d.get("LE33")
    s = d.get("S33")
    if le is None or s is None:
        out["status"] = "SKIP"
        return out
    m = np.isfinite(le) & np.isfinite(s)
    le, s = le[m], s[m]
    if le.size < 5:
        out["status"] = "SKIP"
        return out

    # Abaqus LE33 is the logarithmic (true) strain; stretch = exp(LE).
    lam = np.exp(le)
    sig_exact = _he_true_stress(meta, model, lam)
    if sig_exact is None:
        out["status"] = "NO_REF"
        out["note"] = ("no analytic reference wired for he_model='%s'" % model)
        out["sigma_final"] = float(s[-1])
        out["stretch_final"] = float(lam[-1])
        return out

    # Compare where the signal is meaningful (away from lam ~ 1, where both
    # stresses vanish and the relative error is dominated by noise).
    sig_exact = np.asarray(sig_exact, dtype=float)
    big = np.abs(sig_exact) > 0.05 * np.nanmax(np.abs(sig_exact))
    rel = np.abs(s[big] - sig_exact[big]) / np.maximum(np.abs(sig_exact[big]), 1e-12)
    out["stretch_final"] = float(lam[-1])
    out["sigma_final_measured"] = float(s[-1])
    out["sigma_final_exact"] = float(sig_exact[-1])
    out["err_final_pct"] = 100.0 * (s[-1] - sig_exact[-1]) / sig_exact[-1] \
        if sig_exact[-1] != 0 else float("nan")
    out["err_max_pct"] = float(100.0 * np.nanmax(rel)) if rel.size else float("nan")
    out["err_mean_pct"] = float(100.0 * np.nanmean(rel)) if rel.size else float("nan")
    return out


def analyse_relaxation(path):
    """
    Prony relaxation: hold a constant strain and compare the normalised stress
    decay S(t)/S(0+) with the analytic G(t)/G0 of the *VISCOELASTIC card,
    time-scaled exactly as assignment._prony applies it. This is where a
    normalisation or time_scale_factor error hides -- invisible in a scratch.
    """
    meta, d = read_csv(path)
    out = {"file": os.path.basename(path), "mode": "relaxation", "status": "OK"}
    prony = _parse_prony(meta)
    if prony is None:
        out["status"] = "NO_REF"
        out["note"] = "no prony_table in metadata"
        return out
    tsf = float(meta.get("time_scale_factor", 1.0) or 1.0)

    t = d.get("Time")
    s = np.abs(d.get("S33"))
    m = np.isfinite(t) & np.isfinite(s)
    t, s = t[m], s[m]
    if t.size < 10:
        out["status"] = "SKIP"
        return out

    # The hold begins at the peak stress; normalise by it and shift time so the
    # hold starts at t = 0, which is what G(t)/G0 assumes.
    i_peak = int(np.argmax(s))
    t0 = t[i_peak]
    th = t[i_peak:] - t0
    sh = s[i_peak:]
    if sh.size < 5 or sh[0] <= 0:
        out["status"] = "SKIP"
        return out
    g_meas = sh / sh[0]
    g_exact = np.array([mp.prony_relaxation(prony, ti, G0=1.0,
                                            time_scale_factor=tsf) for ti in th])
    rel = np.abs(g_meas - g_exact)
    out["stability"] = mp.prony_summary(prony, time_scale_factor=tsf)
    out["g_inf_exact"] = float(g_exact[-1])
    out["g_inf_measured"] = float(g_meas[-1])
    out["err_max_abs"] = float(np.nanmax(rel))
    out["err_final_abs"] = float(abs(g_meas[-1] - g_exact[-1]))
    return out


def print_level0(folder):
    files = sorted(glob.glob(os.path.join(folder, "*_element.csv")))
    if not files:
        raise SystemExit("No *_element.csv in %s" % folder)
    print("\n=== LEVEL 0 -- MATERIAL POINT " + "=" * 48)

    # Route each file by what the card actually is. A family with plasticity
    # goes through the DP/J2 branch (E, yield, T/C); an elastomer goes through
    # the hyperelastic branch (stress-stretch curve); a relaxation run goes
    # through the Prony branch. The mode in the filename plus the metadata tell
    # us which, so the report is universal across all seven families.
    tension = compression = None
    for path in files:
        meta, _d = read_csv(path)
        he_model = str(meta.get("he_model", "none"))
        mode = str(meta.get("element_mode", "?"))
        has_plasticity = "yield_table" in meta or "sigma_y0" in meta \
            or isinstance(meta.get("sigma_y0"), float)
        is_relax = (mode == "relaxation")
        is_hyperelastic = (he_model in ("arruda_boyce", "mooney_rivlin",
                                        "yeoh", "ogden")
                           and not isinstance(meta.get("sigma_y0"), float))

        if is_relax:
            r = analyse_relaxation(path)
            print("\n%s  [relaxation]" % r["file"])
            if r["status"] != "OK":
                print("  (%s) %s" % (r["status"], r.get("note", "")))
                continue
            st = r["stability"]
            print("  Prony stability : sum(g_i)=%.3f -> %s"
                  % (st["sum_g"], st["message"]))
            print("  long-term modulus fraction G_inf/G0 : "
                  "expected %.4f | measured %.4f | abs err %.4f"
                  % (r["g_inf_exact"], r["g_inf_measured"], r["err_final_abs"]))
            print("  max |G(t)/G0 error| over the hold   : %.4f" % r["err_max_abs"])
            print("\n  This is the only place a Prony normalisation or a")
            print("  time_scale_factor error is visible: a scratch would hide it.")
            continue

        if is_hyperelastic:
            r = analyse_hyperelastic(path)
            print("\n%s  [%s, %s]" % (r["file"], r["he_model"], r["mode"]))
            if r["status"] == "NO_REF":
                print("  %s" % r.get("note", ""))
                print("  measured final stress %.4g MPa at stretch %.4g"
                      % (r.get("sigma_final", float("nan")),
                         r.get("stretch_final", float("nan"))))
                continue
            if r["status"] != "OK":
                print("  (%s)" % r["status"])
                continue
            print("  true stress at stretch %.3f : measured %.4g MPa | "
                  "analytic %.4g MPa | %+.2f %%"
                  % (r["stretch_final"], r["sigma_final_measured"],
                     r["sigma_final_exact"], r["err_final_pct"]))
            print("  curve error vs analytic law : max %.2f %% | mean %.2f %%"
                  % (r["err_max_pct"], r["err_mean_pct"]))
            print("\n  Elastomer transcription check: no yield, no T/C -- the")
            print("  whole stress-stretch curve IS the test. A wrong mu or")
            print("  lambda_m would show here immediately.")
            continue

        # default: plastic (DP / J2) branch
        r = analyse_single_element(path)
        print("\n%s  [%s]" % (r["file"], r["mode"]))
        if "E_measured" in r:
            print("  E : card %.4g MPa | measured %.4g MPa | %+0.2f %%"
                  % (r["E"], r["E_measured"], r["E_err_pct"]))
        if "sigma_y_measured" in r:
            print("  yield : expected %.4g MPa | measured %.4g MPa | %+0.2f %%"
                  % (r["sigma_y_expected"], r["sigma_y_measured"],
                     r["sigma_y_err_pct"]))
        if "tau_y_expected" in r:
            if r.get("yielded"):
                if "tau_y_measured" in r:
                    print("  shear yield : expected %.4g MPa | measured %.4g MPa "
                          "| %+0.2f %%  (PEEQ_max=%.3g)"
                          % (r["tau_y_expected"], r["tau_y_measured"],
                             r["tau_y_err_pct"], r.get("peeq_max", 0.0)))
                else:
                    print("  shear yield : expected %.4g MPa | plastic (PEEQ_max=%.3g)"
                          " but slope-drop not resolved; peak tau=%.4g MPa"
                          % (r["tau_y_expected"], r.get("peeq_max", 0.0),
                             r.get("tau_max", float("nan"))))
            else:
                need = r.get("gamma_yield_needed")
                print("  shear : ELASTIC only -- PEEQ stayed 0, peak tau=%.4g MPa "
                      "is G*gamma, not a yield." % r.get("tau_max", float("nan")))
                if need is not None:
                    print("          shear yield is %.4g MPa, reached at gamma~%.3f; "
                          "applied gamma_max=%.3f was too small."
                          % (r["tau_y_expected"], need, r.get("gamma_max", float("nan"))))
        if r["mode"] == "tension":
            tension = r
        if r["mode"] == "compression":
            compression = r

    if tension and compression and "sigma_y_measured" in tension \
            and "sigma_y_measured" in compression:
        m = tension["sigma_y_measured"] / compression["sigma_y_measured"]
        print("\n  MEASURED sigma_t / sigma_c = %.3f" % m)
        exp = tension.get("dp_tc_ratio_expected")
        if exp:
            print("  card-implied ratio         = %.3f" % exp)
        # Use the family's real K, not K=1, so the inverted beta is meaningful.
        K = tension.get("flow_stress_ratio", 1.0)
        try:
            beta_meas = mp.dp_beta_from_tc_ratio(m, K if K else 1.0)
            print("  beta implied by the measured ratio (K=%.2f) = %.2f deg"
                  % (K if K else 1.0, beta_meas))
        except Exception:
            pass
        print("\n  This is the only place the Drucker-Prager calibration can be")
        print("  checked directly: beta is never measured, it is DERIVED from a")
        print("  tension/compression ratio. Compare the number above with the")
        print("  T/C ratio of the source the beta came from.")


# --------------------------------------------------------------------------
# Re-analysis of EXISTING production scratch CSVs (no new simulation)
# --------------------------------------------------------------------------

def analyse_scratch_folder(folder, pattern="Mesh_*_Results.csv"):
    """
    Everything that can still be extracted from the mesh-study CSVs already on
    disk: asymptotic-range gate, plateau scatter (the error bars that are
    missing from the convergence plots), Briscoe contact-area inversion, and
    the amplitude-smoothing confound check.
    """
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise SystemExit("No file matching %s in %s" % (pattern, folder))
    rows = []
    for path in files:
        meta, d = read_csv(path)
        t = d.get("Time")
        rf2 = np.abs(d.get("RF2", np.array([])))
        rf3 = np.abs(d.get("RF3", np.array([])))
        u2 = d.get("IndenterU2", np.array([]))
        ie = d.get("ALLIE", np.array([]))
        ae = d.get("ALLAE", np.array([]))
        pw = d.get("WM_ALLPW", np.array([]))

        T = float(meta.get("scratch_time", np.nan))
        m = np.isfinite(t) & np.isfinite(rf2) & (rf2 > 0)
        if not m.any():
            continue
        tt, f2 = t[m], rf2[m]
        f3 = rf3[m] if rf3.size == rf2.size else np.full_like(f2, np.nan)
        # plateau = last 40 % of the scratch step (before unload)
        t_end = float(np.nanmax(tt[tt <= T])) if np.isfinite(T) else float(tt.max())
        win = (tt >= 0.6 * t_end) & (tt <= t_end)
        f2w, f3w = f2[win], f3[win]
        scof = f3w / np.maximum(f2w, 1e-30)

        row = {
            "file": os.path.basename(path),
            "h": float(meta.get("fine_size_x", np.nan)),
            "T": T,
            "RF2_mean": float(np.nanmean(f2w)),
            "RF2_std": float(np.nanstd(f2w)),
            "RF2_cv_pct": float(100.0 * np.nanstd(f2w) / np.nanmean(f2w)),
            "SCOF_mean": float(np.nanmean(scof)),
            "SCOF_std": float(np.nanstd(scof)),
            "U2_max": float(np.nanmax(np.abs(u2))) if u2.size else np.nan,
            "U2_commanded": abs(float(meta.get("scratch_depth", np.nan))),
        }
        if ie.size and ae.size:
            ief, aef = _finite(ie), _finite(ae)
            if ief.size and aef.size and ief.max() > 0:
                row["AE_IE_pct"] = float(100.0 * aef.max() / ief.max())
        if ie.size and pw.size:
            ief, pwf = _finite(ie), _finite(pw)
            if ief.size and pwf.size and ief.max() > 0:
                row["PW_IE_pct"] = float(100.0 * np.abs(pwf).max() / ief.max())
        row["mu_pressure_dep"] = float(meta.get("mu_pressure_dep", 0.0))
        row["mu_friction"] = float(meta.get("mu_friction", np.nan))
        rows.append(row)
    rows.sort(key=lambda r: -r["h"])
    return rows


def print_scratch_reanalysis(rows, tau0=None, alpha=None, mu_cap=None):
    print("\n=== RE-ANALYSIS OF EXISTING SCRATCH CSVs " + "=" * 37)
    print("%-9s %-7s %-12s %-9s %-9s %-9s %-9s %-9s"
          % ("h [mm]", "T [s]", "RF2 [N]", "CV [%]", "SCOF", "SCOF sd",
             "AE/IE %", "PW/IE %"))
    for r in rows:
        print("%-9.4g %-7.4g %-12.6g %-9.2f %-9.4f %-9.4f %-9s %-9s"
              % (r["h"], r["T"], r["RF2_mean"], r["RF2_cv_pct"],
                 r["SCOF_mean"], r["SCOF_std"],
                 "%.2f" % r["AE_IE_pct"] if "AE_IE_pct" in r else "n/a",
                 "%.2f" % r["PW_IE_pct"] if "PW_IE_pct" in r else "n/a"))

    h = [r["h"] for r in rows]
    for label, key in (("RF2", "RF2_mean"), ("SCOF", "SCOF_mean")):
        f = [r[key] for r in rows]
        if len(set(h)) >= 3:
            a = an.asymptotic_range_check(h, f)
            print("\n%s ladder : %s" % (label, a["verdict"]))
            print("  increments   : %s" % ", ".join("%.4g" % x for x in a["increments"]))
            noise = np.mean([r["RF2_cv_pct"] for r in rows])
            print("  mean plateau CV = %.2f %% -- any increment below this is NOISE,"
                  % noise)
            print("  not non-convergence. Error bars change the verdict, and they")
            print("  cost nothing: they are already in these CSVs.")

    # amplitude-smoothing confound
    print("\nCommanded vs achieved penetration (amplitude-smoothing confound):")
    for r in rows:
        if np.isfinite(r["U2_max"]) and np.isfinite(r["U2_commanded"]) \
                and r["U2_commanded"] > 0:
            dev = 100.0 * (r["U2_max"] - r["U2_commanded"]) / r["U2_commanded"]
            flag = "  <-- CHECK" if abs(dev) > 1.0 else ""
            print("  h=%-8.4g T=%-7.4g  |U2|max = %.5g mm  vs commanded %.5g mm "
                  "(%+.2f %%)%s" % (r["h"], r["T"], r["U2_max"],
                                    r["U2_commanded"], dev, flag))
    print("  The SMOOTH window of the tabular amplitudes is")
    print("  smooth * min(scratch_time, unload_time). unload_time is a FIXED")
    print("  0.01 s while scratch_time is swept, so the window is 5 % of the")
    print("  scratch at T=0.05 and 25 % at T=0.01. If |U2|max differs between")
    print("  the three T, the scratch-time 'sensitivity' is a load-path")
    print("  difference, not physics -- and not inertia either.")

    # Briscoe contact-area inversion
    if tau0 and alpha and any(r["mu_pressure_dep"] > 0 for r in rows):
        print("\nBriscoe contact-area inversion  A_c = (SCOF - alpha) * Fn / tau0")
        print("  (with Briscoe active the SCOF measures NOTHING BUT the contact area)")
        for r in rows:
            inv = an.contact_area_from_scof(r["SCOF_mean"], r["RF2_mean"],
                                            tau0, alpha, mu_cap)
            print("  h=%-8.4g A_c = %.5g mm^2" % (r["h"], float(inv["A_c"])))
        first = an.contact_area_from_scof(rows[0]["SCOF_mean"], rows[0]["RF2_mean"],
                                          tau0, alpha, mu_cap)
        last = an.contact_area_from_scof(rows[-1]["SCOF_mean"], rows[-1]["RF2_mean"],
                                         tau0, alpha, mu_cap)
        drift = 100.0 * (float(last["A_c"]) - float(first["A_c"])) / float(first["A_c"])
        print("  contact-area drift over the ladder : %+.1f %%" % drift)
        if "note" in first:
            print("  NB %s" % first["note"])
        print("  => the SCOF mesh drift IS the contact-area mesh drift. Request")
        print("     CAREA in the CSV (see the patch) and measure it directly.")


# --------------------------------------------------------------------------
# SCOF estimator identification
# --------------------------------------------------------------------------

def print_scof_signature(family="glassy_pc", depth=10e-3, R=0.2):
    from ScratchSimulation.AbaqusModel.Configuration.families import get_family
    from ScratchSimulation.AbaqusModel.Configuration.benchmarks import linear_moduli
    cfg = get_family(family).build_config()
    E, nu = linear_moduli(cfg.material)
    fric = cfg.material.friction
    if not getattr(fric, "mu_table", None):
        print("Family '%s' does not carry a tabulated mu(p); nothing to identify."
              % family)
        return
    p0 = float(an.hertz_p0(E, nu, R, depth))
    sig = an.scof_estimator_signature(fric.mu_table, p0)
    print("\n=== SCOF ESTIMATOR SIGNATURE (%s, Hertz at %g um) " % (family, depth * 1e3)
          + "=" * 12)
    print("  p0 = %.1f MPa, pbar = %.1f MPa" % (p0, 2.0 * p0 / 3.0))
    for k, v in sorted(sig.items()):
        print("  %-24s -> SCOF = %.4f" % (k, v))
    print("\n  Run ONE Hertz indentation with this friction table, read the")
    print("  simulated |RF3|/|RF2|, and the matching line names the estimator.")
    print("  results_verifier.check_friction_physics already computes the")
    print("  global force ratio, which is the correct one -- this test proves it")
    print("  end to end instead of assuming it.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    mode = argv[1]
    folder = argv[2] if len(argv) > 2 else "."
    if mode == "hertz":
        print_hertz_ladder(analyse_hertz_ladder(folder))
    elif mode == "time":
        print_time_ladder(analyse_time_ladder(folder, "HertzT_*_indent.csv"),
                          "TIME LADDER, constant mass scaling")
        try:
            print_time_ladder(analyse_time_ladder(folder, "HertzTP_*_indent.csv"),
                              "TIME LADDER, production mass scaling")
        except SystemExit:
            pass
    elif mode == "ms":
        print_time_ladder(analyse_time_ladder(folder, "HertzMS_*_indent.csv"),
                          "MASS-SCALING LADDER")
    elif mode == "level0":
        print_level0(folder)
    elif mode == "scratch":
        tau0 = float(argv[3]) if len(argv) > 3 else None
        alpha = float(argv[4]) if len(argv) > 4 else None
        cap = float(argv[5]) if len(argv) > 5 else None
        print_scratch_reanalysis(analyse_scratch_folder(folder), tau0, alpha, cap)
    elif mode == "scof":
        print_scof_signature(argv[2] if len(argv) > 2 else "glassy_pc")
    elif mode == "inspect":
        # Accepts a single CSV or a folder; dumps every column's range.
        targets = ([folder] if folder.lower().endswith(".csv")
                   else sorted(glob.glob(os.path.join(folder, "*.csv"))))
        if not targets:
            print("No CSV found at %s" % folder)
            return 1
        for p in targets:
            inspect_csv(p)
    else:
        print("Unknown mode '%s'." % mode)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))