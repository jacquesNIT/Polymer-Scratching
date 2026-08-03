# -*- coding: utf-8 -*-
"""
ODB -> CSV for the benchmarks.

Separate from the production extractor: that one is built around a scratch
(residual profile, plateau, force-mode fallbacks) and a benchmark needs
different columns. Merging them would tie the production CSV schema to the
benchmark suite.

Outputs
    <stem>_element.csv   strains, stresses, PEEQ, energies (level 0)
    <stem>_indent.csv    U2, RF, CAREA, energies      (levels 1a/1b/2)
    <stem>_contact.csv   a_num and p(r) per field frame
"""

from odbAccess import *
from abaqusConstants import *      # INTEGRATION_POINT etc. are not in odbAccess
import csv
import numpy as np
import os


# --------------------------------------------------------------------------
# History helpers
# --------------------------------------------------------------------------

def _key_matches(key, name):
    if key == name:
        return True
    cleaned = key.upper().replace(":", " ").replace(",", " ").replace(";", " ")
    return name.upper() in cleaned.split()


def _pick(data, name, default=None):
    if name in data:
        return data[name]
    cands = [k for k in data if _key_matches(k, name)]
    return data[cands[0]] if len(cands) == 1 else default


def _safe_xy(history_output):
    """(t, v) or (None, None). A job that aborts before its first output
    interval leaves the key present but .data None or empty; np.array(None) is
    0-D and a blind .T[0, :] then fails with an opaque index error instead of
    reporting that the job produced no output."""
    data = getattr(history_output, "data", None)
    if not data:
        return None, None
    arr = np.array(data, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return None, None
    return arr[:, 0], arr[:, 1]


def _history(odb, region_name):
    """Concatenate a history region across steps, rebasing step time."""
    keys = []
    for sname in odb.steps.keys():
        hrs = odb.steps[sname].historyRegions
        if region_name in hrs.keys():
            for k in hrs[region_name].historyOutputs.keys():
                if k not in keys:
                    keys.append(k)
    if not keys:
        return np.array([]), {}

    tparts, dparts, t_end, aborted = [], {}, 0.0, []
    for sname in odb.steps.keys():
        hrs = odb.steps[sname].historyRegions
        if region_name not in hrs.keys():
            continue
        hr = hrs[region_name]
        hk = list(hr.historyOutputs.keys())
        if not hk:
            continue
        t, _ = _safe_xy(hr.historyOutputs[hk[0]])
        if t is None or t.size == 0:
            aborted.append(sname)
            continue
        t = t + (t_end if (tparts and t[0] < 0.5 * t_end) else 0.0)
        start = 1 if (tparts and t.size > 1 and abs(t[0] - t_end) < 1e-12) else 0
        n = t.size - start
        if n <= 0:
            continue
        tparts.append(t[start:])
        for k in keys:
            col = np.zeros(n)
            if k in hk:
                _, v = _safe_xy(hr.historyOutputs[k])
                if v is not None:
                    m = min(n, max(v.size - start, 0))
                    col[:m] = v[start:start + m]
            dparts.setdefault(k, []).append(col)
        t_end = t[-1]

    if aborted:
        print(">>> WARNING: region '%s' has no output frames in step(s) %s. "
              "The job most likely ABORTED there (element distortion at fine "
              "meshes is the usual cause); check the .sta/.msg."
              % (region_name, ", ".join(aborted)))
    if not tparts:
        return np.array([]), {}
    return (np.concatenate(tparts),
            dict((k, np.concatenate(v)) for k, v in dparts.items()))


def _find_regions(odb):
    """Locate the reference-point, substrate, whole-model and contact regions
    by the variables they carry rather than by name."""
    found = {"rp": None, "sub": None, "whole": None, "contact": None}
    for sname in odb.steps.keys():
        for rk in odb.steps[sname].historyRegions.keys():
            hop = list(odb.steps[sname].historyRegions[rk].historyOutputs.keys())
            if found["rp"] is None and any(_key_matches(k, "RF2") for k in hop):
                found["rp"] = rk
            if found["whole"] is None and any(_key_matches(k, "ETOTAL") for k in hop):
                found["whole"] = rk
            if (found["sub"] is None and any(_key_matches(k, "ALLIE") for k in hop)
                    and not any(_key_matches(k, "ETOTAL") for k in hop)):
                found["sub"] = rk
            if found["contact"] is None and any(
                    _key_matches(k, n) for n in ("CAREA", "CFN2") for k in hop):
                found["contact"] = rk
    return found


def _resample_region(odb, region, keys, t_ref):
    if region is None:
        return dict((k, np.full_like(t_ref, np.nan)) for k in keys)
    tt, dd = _history(odb, region)
    if tt.size == 0:
        return dict((k, np.full_like(t_ref, np.nan)) for k in keys)
    zz = np.zeros_like(tt)
    return dict((k, np.interp(t_ref, tt, _pick(dd, k, zz))) for k in keys)


# --------------------------------------------------------------------------
# CSV writing
# --------------------------------------------------------------------------

def _write(path, header, columns, meta_lines):
    with open(path, "w") as f:
        for line in meta_lines:
            f.write("# %s\n" % line)
        w = csv.writer(f)
        w.writerow(header)
        n = max(len(c) for c in columns)
        for i in range(n):
            w.writerow([("" if i >= len(c) or c[i] != c[i] else c[i])
                        for c in columns])
    print("CSV written: %s" % path)


def _meta(cfg, bench, extra=None):
    """Header lines as '# key=value'. The report derives the effective mass
    factor, N and v_n/c from these, so E, nu, rho, mesh and ramp_time must all
    be present."""
    m = cfg.material
    lines = [
        "Benchmark output -- ScratchSimulation V&V suite",
        "family=%s" % getattr(m, "family", "?"),
        "bench_kind=%s bench_solver=%s" % (bench.kind, bench.solver_kind),
        "rho=%.6g" % m.rho,
        "he_model=%s" % m.hyperelastic.MODEL,
        "plasticity=%s viscoelastic=%s" % (m.plasticity.MODEL,
                                           m.viscoelastic.MODEL),
        "tip_radius=%.6g cone_angle=%.6g" % (cfg.indenter.tip_radius,
                                             cfg.indenter.cone_angle),
        "fine_size_x=%.6g fine_size_y=%.6g fine_size_z=%.6g"
        % (cfg.mesh.fine_size_x, cfg.mesh.fine_size_y, cfg.mesh.fine_size_z),
        "mass_scale=%.6g target_time_increment=%.6g"
        % (cfg.solver.mass_scale, cfg.solver.target_time_increment),
        "depth_max=%.6g ramp_time=%.6g hold_time=%.6g"
        % (bench.depth_max, bench.ramp_time, bench.hold_time),
        "element_mode=%s element_strain=%.6g"
        % (bench.element_mode, bench.element_strain),
    ]
    for k, v in (m.hyperelastic.params() or {}).items():
        lines.append("%s=%s" % (k, v))
    # Prony as flat g:k:tau triples, so the relaxation check can rebuild it.
    if getattr(m.viscoelastic, "prony_table", None):
        lines.append("prony_table=" + ";".join(
            "%.8g:%.8g:%.8g" % (float(r[0]), float(r[1]), float(r[2]))
            for r in m.viscoelastic.prony_table))
        lines.append("time_scale_factor=%.8g"
                     % float(getattr(cfg.solver, "time_scale_factor", 1.0) or 1.0))
    if getattr(m.plasticity, "yield_table", None):
        pl = m.plasticity
        lines.append("sigma_y0=%.6g" % pl.yield_table[0][0])
        if hasattr(pl, "friction_angle"):
            lines.append("friction_angle=%.6g flow_stress_ratio=%.6g "
                         "dilation_angle=%.6g"
                         % (pl.friction_angle, pl.flow_stress_ratio,
                            pl.dilation_angle))
    for k, v in (extra or {}).items():
        lines.append("%s=%s" % (k, v))
    return lines


# --------------------------------------------------------------------------
# Level 0
# --------------------------------------------------------------------------

def extract_single_element(job_name, stem, cfg, bench, out_dir="BenchOutputs"):
    odb = openOdb(path=job_name + ".odb", readOnly=True)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    elset = None
    for name in odb.rootAssembly.elementSets.keys():
        if "BENCHCUBESET" in name.upper():
            elset = odb.rootAssembly.elementSets[name]
            break
    if elset is None:
        elset = odb.rootAssembly.elementSets[
            list(odb.rootAssembly.elementSets.keys())[0]]

    series = dict((k, []) for k in
                  ("t", "S11", "S22", "S33", "S12",
                   "LE11", "LE22", "LE33", "LE12", "PEEQ"))
    for step in odb.steps.values():
        for frame in step.frames:
            fo = frame.fieldOutputs

            def one(name, comp=None):
                # No explicit position=: S and LE are integration-point
                # quantities by default and the constant spelling varies.
                if name not in fo.keys():
                    return float("nan")
                sub = fo[name].getSubset(region=elset)
                if not sub.values:
                    return float("nan")
                d = sub.values[0].data
                if comp is None:
                    return float(d)
                try:
                    return float(d[comp])
                except (TypeError, IndexError):
                    return float("nan")

            series["t"].append(float(frame.frameValue))
            for i, c in enumerate(("11", "22", "33", "12")):
                series["S" + c].append(one("S", i))
                series["LE" + c].append(one("LE", i))
            series["PEEQ"].append(one("PEEQ"))

    # The single-element step requests no ETOTAL, so locate the energy region
    # by ALLIE instead.
    reg = _find_regions(odb)
    tE, dE = _history(odb, reg["whole"] or reg["sub"] or "")
    e_keys = ("ALLKE", "ALLIE", "ALLSE", "ALLPD", "ALLCD", "ALLAE")
    zz = np.zeros_like(tE)

    t = np.array(series["t"])
    header = ["Time", "LE11", "LE22", "LE33", "LE12",
              "S11", "S22", "S33", "S12", "PEEQ"]
    cols = [series["t"]] + [series[k] for k in header[1:]]
    for k in e_keys:
        cols.append(list(np.interp(t, tE, _pick(dE, k, zz)))
                    if tE.size else [float("nan")] * len(t))
        header.append(k)

    path = os.path.join(out_dir, stem + "_element.csv")
    _write(path, header, cols, _meta(cfg, bench))
    odb.close()
    return path


# --------------------------------------------------------------------------
# Levels 1 / 2
# --------------------------------------------------------------------------

def extract_indentation(job_name, stem, cfg, bench, out_dir="BenchOutputs"):
    odb = openOdb(path=job_name + ".odb", readOnly=True)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    reg = _find_regions(odb)
    if reg["rp"] is None:
        odb.close()
        raise ValueError("No RF history region in %s.odb" % job_name)

    t, d = _history(odb, reg["rp"])
    if t.size == 0:
        odb.close()
        raise ValueError(
            "%s.odb: region '%s' produced zero usable output frames in every "
            "step. The job almost certainly ABORTED before its first "
            "increment; check the .sta/.msg before re-extracting."
            % (job_name, reg["rp"]))

    z = np.zeros_like(t)
    rf1, rf2, rf3 = (_pick(d, k, z) for k in ("RF1", "RF2", "RF3"))
    u2 = _pick(d, "U2", z)

    # _pick returns its default when a key is absent or ambiguous, and a column
    # of zeros then reaches the report as an empty curve. Abaqus does not
    # always put RF* and U* of a reference point in the same region, so scan.
    if not np.any(np.abs(np.asarray(u2, dtype=float)) > 1e-9):
        print(">>> U2 absent/zero in '%s'; scanning all history regions."
              % reg["rp"])
        found = None
        for sname in odb.steps.keys():
            for rk in odb.steps[sname].historyRegions.keys():
                tt, dd = _history(odb, rk)
                if tt.size == 0:
                    continue
                cand = _pick(dd, "U2", None)
                if cand is not None and np.any(np.abs(cand) > 1e-9):
                    u2, found = np.interp(t, tt, cand), rk
                    break
            if found:
                break
        if found:
            print(">>> U2 recovered from '%s' (|U2|max = %.6g mm)."
                  % (found, np.abs(u2).max()))
        else:
            print(">>> WARNING: no non-zero U2 in the ODB. The report can "
                  "reconstruct the depth from the prescribed ramp, but the "
                  "output request should be fixed.")

    sub_e = _resample_region(odb, reg["sub"],
                             ("ALLKE", "ALLIE", "ALLAE", "ALLSE", "ALLPD"), t)
    wm_e = _resample_region(odb, reg["whole"],
                            ("ALLKE", "ALLIE", "ALLSE", "ALLPD", "ALLVD",
                             "ALLFD", "ALLWK", "ALLPW", "ALLCW", "ALLMW",
                             "ALLSD", "ETOTAL"), t)
    cont = _resample_region(odb, reg["contact"],
                            ("CFN2", "CFN3", "CFS3", "CAREA"), t)

    header = ["Time", "IndenterU2", "RF1", "RF2", "RF3",
              "CFN2", "CFN3", "CFS3", "CAREA"]
    cols = [t, u2, rf1, rf2, rf3,
            cont["CFN2"], cont["CFN3"], cont["CFS3"], cont["CAREA"]]
    for k in ("ALLKE", "ALLIE", "ALLAE", "ALLSE", "ALLPD"):
        header.append("SUB_" + k)
        cols.append(sub_e[k])
    for k in ("ALLKE", "ALLIE", "ALLSE", "ALLPD", "ALLVD", "ALLFD",
              "ALLWK", "ALLPW", "ALLCW", "ALLMW", "ALLSD", "ETOTAL"):
        header.append("WM_" + k)
        cols.append(wm_e[k])

    path = os.path.join(out_dir, stem + "_indent.csv")
    _write(path, header, cols, _meta(cfg, bench))
    contact_path = _extract_contact_profile(odb, stem, cfg, bench, out_dir, t, u2)
    odb.close()
    return path, contact_path


def _extract_contact_profile(odb, stem, cfg, bench, out_dir, t_hist, u2_hist):
    """a_num and p(r) per field frame.

    a_num drives the ploughing force and the Briscoe SCOF and is the least
    converged output of the model -- far less than RF2 -- so measuring it
    against the exact sqrt(R*h) is the most informative part of the benchmark.
    """
    names, sub = cfg.naming, cfg.substrate
    z_tip = 0.5 * (sub.zs1 + sub.zs2)

    key = next((k for k in odb.rootAssembly.nodeSets.keys()
                if k.upper() == names.contact_region_nodes.upper()), None)
    if key is None:
        print(">>> contact node set not found; skipping a_num.")
        return None

    try:
        surf = odb.rootAssembly.surfaces[names.slave_surface.upper()]
        coords = dict((n.label, n.coordinates) for n in surf.nodes[0])
    except Exception as exc:
        print(">>> slave-surface coordinates unavailable (%s)." % exc)
        return None

    nset = odb.rootAssembly.nodeSets[key]
    rows = []
    for step in odb.steps.values():
        for frame in step.frames:
            if "CPRESS" not in frame.fieldOutputs.keys():
                continue
            vals = frame.fieldOutputs["CPRESS"].getSubset(region=nset).values
            profile, r_max, p_max, p_sum, n_c = [], 0.0, 0.0, 0.0, 0
            for v in vals:
                p = float(v.data)
                c = coords.get(v.nodeLabel)
                if p <= 1e-9 or c is None:
                    continue
                r = float(np.sqrt(c[0] ** 2 + (c[2] - z_tip) ** 2))
                n_c += 1
                p_sum += p
                r_max, p_max = max(r_max, r), max(p_max, p)
                profile.append((r, p))
            tf = float(frame.frameValue)
            rows.append({
                "time": tf,
                "u2": float(np.interp(tf, t_hist, u2_hist)) if len(t_hist) else float("nan"),
                "a_num": r_max, "n": n_c, "p_max": p_max,
                "p_mean": (p_sum / n_c if n_c else 0.0),
                "profile": sorted(profile)})

    path = os.path.join(out_dir, stem + "_contact.csv")
    with open(path, "w") as f:
        for line in _meta(cfg, bench, {"z_tip": z_tip}):
            f.write("# %s\n" % line)
        f.write("# a_num = max radius of a slave node with CPRESS > 0 "
                "(half model: r measured from the symmetry plane)\n")
        w = csv.writer(f)
        w.writerow(["Time", "IndenterU2", "a_num", "n_contact_nodes",
                    "p_max", "p_mean_nodal"])
        for r in rows:
            w.writerow([r["time"], r["u2"], r["a_num"], r["n"],
                        r["p_max"], r["p_mean"]])
        w.writerow([])
        w.writerow(["# pressure profile of the LAST frame"])
        w.writerow(["r_mm", "CPRESS_MPa"])
        if rows:
            for r, p in rows[-1]["profile"]:
                w.writerow([r, p])
    print("CSV written: %s" % path)
    return path