# -*- coding: utf-8 -*-
"""
ODB -> CSV extraction for the V&V benchmarks.

Kept separate from extractor.py on purpose: the production extractor is built
around a scratch (residual profile, plateau, contact-pair fallbacks) and a
benchmark needs different columns. Mixing them would make the production CSV
schema depend on the benchmark suite.

Outputs
  single element : Time, LE11..LE33, S11..S33, S12, PEEQ, energies
  indentation    : Time, U2, RF2, RF3, CAREA, energies  (history, dense)
                 + a separate <stem>_contact.csv with, per FIELD frame,
                   the numerical contact radius a_num and the p(r) profile
"""

from odbAccess import *
from abaqusConstants import *      # INTEGRATION_POINT etc. are NOT in odbAccess
import csv
import numpy as np
import os


# --------------------------------------------------------------------------
# shared helpers
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


def _history(odb, region_name):
    """Concatenate a history region across all steps (rebasing step time)."""
    keys = []
    for sname in odb.steps.keys():
        hrs = odb.steps[sname].historyRegions
        if region_name in hrs.keys():
            for k in hrs[region_name].historyOutputs.keys():
                if k not in keys:
                    keys.append(k)
    if not keys:
        return np.array([]), {}
    tparts, dparts, t_end = [], {}, 0.0
    for sname in odb.steps.keys():
        hrs = odb.steps[sname].historyRegions
        if region_name not in hrs.keys():
            continue
        hr = hrs[region_name]
        hk = list(hr.historyOutputs.keys())
        if not hk:
            continue
        t = np.array(hr.historyOutputs[hk[0]].data).T[0, :]
        if t.size == 0:
            continue
        offset = t_end if (tparts and t[0] < 0.5 * t_end) else 0.0
        t = t + offset
        start = 1 if (tparts and t.size > 1 and abs(t[0] - t_end) < 1e-12) else 0
        n = t.size - start
        if n <= 0:
            continue
        tparts.append(t[start:])
        for k in keys:
            col = (np.array(hr.historyOutputs[k].data).T[1, :][start:]
                   if k in hk else np.zeros(n))
            dparts.setdefault(k, []).append(col)
        t_end = t[-1]
    if not tparts:
        return np.array([]), {}
    return (np.concatenate(tparts),
            dict((k, np.concatenate(v)) for k, v in dparts.items()))


def _find_regions(odb):
    """Locate the indenter RP, substrate energy, whole-model and contact regions."""
    found = {"rp": None, "sub": None, "whole": None, "contact": None}
    for sname in odb.steps.keys():
        for rk in odb.steps[sname].historyRegions.keys():
            hop = list(odb.steps[sname].historyRegions[rk].historyOutputs.keys())
            if found["rp"] is None and any(_key_matches(k, "RF2") for k in hop):
                found["rp"] = rk
            if found["whole"] is None and any(_key_matches(k, "ETOTAL") for k in hop):
                found["whole"] = rk
            if (found["sub"] is None
                    and any(_key_matches(k, "ALLIE") for k in hop)
                    and not any(_key_matches(k, "ETOTAL") for k in hop)):
                found["sub"] = rk
            if found["contact"] is None and any(
                    _key_matches(k, n) for n in ("CAREA", "CFN2") for k in hop):
                found["contact"] = rk
    return found


def _write(path, header, columns, meta_lines):
    with open(path, "w") as f:
        for line in meta_lines:
            f.write("# %s\n" % line)
        w = csv.writer(f)
        w.writerow(header)
        n = max(len(c) for c in columns)
        for i in range(n):
            w.writerow([("" if i >= len(c) or c[i] != c[i] else c[i]) for c in columns])
    print("CSV written: %s" % path)


def _meta(cfg, bench, extra=None):
    m = cfg.material
    lines = [
        "Benchmark output -- ScratchSimulation V&V suite",
        "family=%s" % getattr(m, "family", "?"),
        "bench_kind=%s bench_solver=%s" % (bench.kind, bench.solver_kind),
        "rho=%.6g" % m.rho,
        "he_model=%s" % m.hyperelastic.MODEL,
        "plasticity=%s viscoelastic=%s" % (m.plasticity.MODEL, m.viscoelastic.MODEL),
        "tip_radius=%.6g cone_angle=%.6g" % (cfg.indenter.tip_radius,
                                             cfg.indenter.cone_angle),
        ("fine_size_x=%.6g fine_size_y=%.6g fine_size_z=%.6g"
         % (cfg.mesh.fine_size_x, cfg.mesh.fine_size_y, cfg.mesh.fine_size_z)),
        ("mass_scale=%.6g target_time_increment=%.6g"
         % (cfg.solver.mass_scale, cfg.solver.target_time_increment)),
        ("depth_max=%.6g ramp_time=%.6g hold_time=%.6g"
         % (bench.depth_max, bench.ramp_time, bench.hold_time)),
        ("element_mode=%s element_strain=%.6g"
         % (bench.element_mode, bench.element_strain)),
    ]
    for k, v in (m.hyperelastic.params() or {}).items():
        lines.append("%s=%s" % (k, v))
    if getattr(m.plasticity, "yield_table", None):
        pl = m.plasticity
        lines.append("sigma_y0=%.6g" % pl.yield_table[0][0])
        if hasattr(pl, "friction_angle"):
            lines.append("friction_angle=%.6g flow_stress_ratio=%.6g dilation_angle=%.6g"
                         % (pl.friction_angle, pl.flow_stress_ratio, pl.dilation_angle))
    for k, v in (extra or {}).items():
        lines.append("%s=%s" % (k, v))
    return lines


# --------------------------------------------------------------------------
# Level 0 -- single element
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

    t, s11, s22, s33, s12 = [], [], [], [], []
    le11, le22, le33, le12, peeq = [], [], [], [], []

    for step in odb.steps.values():
        for frame in step.frames:
            fo = frame.fieldOutputs

            def one(name, comp=None):
                if name not in fo.keys():
                    return float("nan")
                # No explicit position= : S and LE are integration-point
                # quantities by default, and forcing a position that a given
                # release spells differently is an avoidable failure mode.
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

            t.append(float(frame.frameValue))
            s11.append(one("S", 0)); s22.append(one("S", 1)); s33.append(one("S", 2))
            s12.append(one("S", 3))
            le11.append(one("LE", 0)); le22.append(one("LE", 1))
            le33.append(one("LE", 2)); le12.append(one("LE", 3))
            peeq.append(one("PEEQ"))

    # The single-element step requests no ETOTAL, so _find_regions()["whole"]
    # is None here: locate the energy region by ALLIE instead.
    _reg = _find_regions(odb)
    _energy_region = _reg["whole"] or _reg["sub"] or ""
    tE, dE = _history(odb, _energy_region)
    zz = np.zeros_like(tE)
    e_keys = ("ALLKE", "ALLIE", "ALLSE", "ALLPD", "ALLCD", "ALLAE")
    energies = dict((k, _pick(dE, k, zz)) for k in e_keys)

    path = os.path.join(out_dir, stem + "_element.csv")
    cols = [t, le11, le22, le33, le12, s11, s22, s33, s12, peeq]
    header = ["Time", "LE11", "LE22", "LE33", "LE12",
              "S11", "S22", "S33", "S12", "PEEQ"]
    # energies resampled onto the field-frame axis
    for k in e_keys:
        if tE.size:
            cols.append(list(np.interp(np.array(t), tE, energies[k])))
        else:
            cols.append([float("nan")] * len(t))
        header.append(k)

    _write(path, header, cols, _meta(cfg, bench))
    odb.close()
    return path


# --------------------------------------------------------------------------
# Level 1 / 2 -- indentation
# --------------------------------------------------------------------------

def extract_indentation(job_name, stem, cfg, bench, out_dir="BenchOutputs"):
    odb = openOdb(path=job_name + ".odb", readOnly=True)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    reg = _find_regions(odb)
    if reg["rp"] is None:
        odb.close()
        raise ValueError("No RF history region found in %s.odb" % job_name)

    t, d = _history(odb, reg["rp"])
    z = np.zeros_like(t)
    rf2 = _pick(d, "RF2", z)
    rf3 = _pick(d, "RF3", z)
    rf1 = _pick(d, "RF1", z)
    u2 = _pick(d, "U2", z)

    def _res(region, keys):
        if region is None:
            return dict((k, np.full_like(t, np.nan)) for k in keys)
        tt, dd = _history(odb, region)
        if tt.size == 0:
            return dict((k, np.full_like(t, np.nan)) for k in keys)
        zz = np.zeros_like(tt)
        return dict((k, np.interp(t, tt, _pick(dd, k, zz))) for k in keys)

    sub_e = _res(reg["sub"], ("ALLKE", "ALLIE", "ALLAE", "ALLSE", "ALLPD"))
    wm_e = _res(reg["whole"], ("ALLKE", "ALLIE", "ALLSE", "ALLPD", "ALLVD",
                               "ALLFD", "ALLWK", "ALLPW", "ALLCW", "ALLMW",
                               "ALLSD", "ETOTAL"))
    cont = _res(reg["contact"], ("CFN2", "CFN3", "CFS3", "CAREA"))

    header = ["Time", "IndenterU2", "RF1", "RF2", "RF3",
              "CFN2", "CFN3", "CFS3", "CAREA"]
    cols = [t, u2, rf1, rf2, rf3,
            cont["CFN2"], cont["CFN3"], cont["CFS3"], cont["CAREA"]]
    for k in ("ALLKE", "ALLIE", "ALLAE", "ALLSE", "ALLPD"):
        header.append("SUB_" + k); cols.append(sub_e[k])
    for k in ("ALLKE", "ALLIE", "ALLSE", "ALLPD", "ALLVD", "ALLFD",
              "ALLWK", "ALLPW", "ALLCW", "ALLMW", "ALLSD", "ETOTAL"):
        header.append("WM_" + k); cols.append(wm_e[k])

    path = os.path.join(out_dir, stem + "_indent.csv")
    _write(path, header, cols, _meta(cfg, bench))

    contact_path = _extract_contact_profile(odb, stem, cfg, bench, out_dir, t, u2)
    odb.close()
    return path, contact_path


def _extract_contact_profile(odb, stem, cfg, bench, out_dir, t_hist, u2_hist):
    """
    Per FIELD frame: the numerical contact radius a_num and the p(r) profile.

    a_num is the quantity that actually drives the ploughing force and the
    Briscoe SCOF, and it is the LEAST converged output of the model -- far
    less converged than RF2. Measuring it against the exact sqrt(R*h) is the
    single most informative thing the Hertz benchmark does.
    """
    names = cfg.naming
    sub = cfg.substrate
    z_tip = 0.5 * (sub.zs1 + sub.zs2)

    key = None
    for k in odb.rootAssembly.nodeSets.keys():
        if k.upper() == names.contact_region_nodes.upper():
            key = k
            break
    if key is None:
        print(">>> contact node set not found; skipping the a_num extraction.")
        return None
    nset = odb.rootAssembly.nodeSets[key]

    coords = {}
    try:
        surf = odb.rootAssembly.surfaces[names.slave_surface.upper()]
        for n in surf.nodes[0]:
            coords[n.label] = (n.coordinates[0], n.coordinates[1], n.coordinates[2])
    except Exception as exc:
        print(">>> could not read slave-surface coordinates (%s)." % exc)
        return None

    rows = []
    for step in odb.steps.values():
        for frame in step.frames:
            if "CPRESS" not in frame.fieldOutputs.keys():
                continue
            sub_f = frame.fieldOutputs["CPRESS"].getSubset(region=nset)
            r_max, n_c, p_max, p_sum = 0.0, 0, 0.0, 0.0
            profile = []
            for v in sub_f.values:
                p = float(v.data)
                if p <= 1e-9:
                    continue
                c = coords.get(v.nodeLabel)
                if c is None:
                    continue
                r = float(np.sqrt((c[0]) ** 2 + (c[2] - z_tip) ** 2))
                n_c += 1
                p_sum += p
                r_max = max(r_max, r)
                p_max = max(p_max, p)
                profile.append((r, p))
            tf = float(frame.frameValue)
            u2f = float(np.interp(tf, t_hist, u2_hist)) if len(t_hist) else float("nan")
            rows.append({"time": tf, "u2": u2f, "a_num": r_max,
                         "n_contact_nodes": n_c, "p_max": p_max,
                         "p_mean_nodal": (p_sum / n_c if n_c else 0.0),
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
            w.writerow([r["time"], r["u2"], r["a_num"], r["n_contact_nodes"],
                        r["p_max"], r["p_mean_nodal"]])
        w.writerow([])
        w.writerow(["# pressure profile of the LAST frame"])
        w.writerow(["r_mm", "CPRESS_MPa"])
        if rows:
            for r, p in rows[-1]["profile"]:
                w.writerow([r, p])
    print("CSV written: %s" % path)
    return path