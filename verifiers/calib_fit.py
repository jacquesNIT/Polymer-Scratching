"""Score every calibration sweep point against the laboratory, compared at equal groove width w0.

Example: python .\calib_fit.py --sim sweep_calib_curves_pc1 --design ..\designs\glassy_pc_calib.csv
             --lab '..\Lab results\Main_Batch\PC\4N\PC_4N_1Nm*' --material pc
"""

import argparse
import csv as _csv
import glob
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np


# Imports of the source modules
def _load_module(name, filenames):
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    for root in (here, Path.cwd(), here.parent, here.parent / "AbaqusModel"):
        for fn in filenames:
            cand = root / fn
            if cand.is_file():
                spec = importlib.util.spec_from_file_location(name, str(cand))
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
                return mod
    raise ImportError("%s not found next to calib_fit.py" % name)


bv = _load_module("bcrf_values", ["bcrf_values.py"])


# Observables that carry the comparison. `score` marks the three that enter
# the combined chi-square -- one per phenomenon, see the module header.
OBSERVABLES = [
    ("F_n", "Normal force", "N", True),
    ("A_ratio", "A_pile / A_groove", "-", True),
    ("SCOF", "Apparent friction coefficient", "-", True),
    ("h_r", "Residual depth", "um", False),
    ("h_p", "Lateral pile-up height", "um", False),
]
SCORED = [k for k, _, _, s in OBSERVABLES if s]

# Relative floor on the laboratory scatter
SIGMA_REL_FLOOR = 0.02

DEFAULT_SCRATCH_LENGTH = 2.0        # mm
DEFAULT_N_GRID = 25


# Laboratory: force files
_FORCE_COLS = {"z": 7, "Fz": 3, "Fx": 5, "CAP": 4, "t": 1}


def read_force_csv(path):
    """Read a TriboSoft force export and return (z [mm], F_n [N], F_t [N])."""
    raw = open(path, encoding="latin-1").read()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    body = [ln for ln in lines if ln.count(",") >= 10 and ln[:1].isdigit()]
    if not body:
        raise ValueError("no data rows in %s" % path)
    import io
    d = np.genfromtxt(io.StringIO("\n".join(body)), delimiter=",")
    z = np.abs(d[:, _FORCE_COLS["z"]] - d[0, _FORCE_COLS["z"]])
    fn = d[:, _FORCE_COLS["Fz"]]
    ft = np.abs(d[:, _FORCE_COLS["Fx"]])
    ok = np.isfinite(z) & np.isfinite(fn) & np.isfinite(ft)
    return z[ok], fn[ok], ft[ok]


# Laboratory: scans
def scan_area_profile(res, rezero=True):
    """A_groove and A_pile for every column of a levelled scan."""
    Zm, ref = res["Zm"], res["ref"]
    row_c, half, outer = res["row_c"], res["half"], res["outer"]
    y_um = res["y_um"]
    nx = Zm.shape[1]
    a_g = np.full(nx, np.nan)
    a_p = np.full(nx, np.nan)
    for j in np.flatnonzero(res["profiles"]["present"]):
        p = np.asarray(Zm[:, j], dtype=float)
        if rezero:
            sel = ref[:, j]
            if sel.sum() >= 10:
                base = np.nanmedian(p[sel])
                if np.isfinite(base):
                    p = p - base
        if not np.isfinite(p).any():
            continue
        sec = bv.section_values(p, y_um, row_c, half, outer, penetration=False)
        if sec is None:
            continue
        a_g[j] = sec["area_groove"]
        a_p[j] = sec["area_pileup"]
    return a_g, a_p


def anchor_index(res, mode="groove_end"):
    """Column index of the scan matching z = scratch_length.
    anchor: 'groove_end', 'mound' or 'deepest'.
    """
    prof = res["profiles"]
    present = prof["present"]
    if mode == "mound":
        mound = res.get("mound")
        if isinstance(mound, dict) and mound.get("x") is not None:
            return int(np.argmin(np.abs(res["x_um"] - float(mound["x"]))))
        mode = "groove_end"
    if mode == "deepest":
        return int(res["i_deep"])
    if not present.any():
        raise ValueError("no groove detected in %s" % res["path"])
    return int(np.max(np.nonzero(present)[0]))


def read_scan(path, material, anchor="groove_end", shift_um=0.0,
              scratch_length=DEFAULT_SCRATCH_LENGTH, **kw):
    """One .bcrf -> along-track curves on the z abscissa of the force file."""
    res = bv.analyse(path, material=material, **kw)
    a_g, a_p = scan_area_profile(res)
    prof = res["profiles"]
    i_end = anchor_index(res, anchor)
    x_end = res["x_um"][i_end] + float(shift_um)
    z_mm = scratch_length - (x_end - res["x_um"]) * 1e-3

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a_g > 0, a_p / a_g, np.nan)
    h_p = np.nanmean(np.vstack([prof["h_p_left"], prof["h_p_right"]]), axis=0)

    # Trim at the deepest column: past it the tip lifts off and w0 collapses.
    m = prof["present"].copy()
    i_deep = int(res["i_deep"])
    m[i_deep + 1:] = False

    w0 = prof["w0_s"][m] if "w0_s" in prof else prof["w0"][m]
    return {
        "path": path, "z": z_mm[m], "w0": w0, "w0_raw": prof["w0"][m],
        "h_r": prof["h_r"][m], "h_p": h_p[m],
        "A_groove": a_g[m], "A_pile": a_p[m], "A_ratio": ratio[m],
        "x_end": x_end, "i_deep": i_deep, "res": res,
    }


# Laboratory: reduction of the repeats
def _resample(x_src, y_src, x_dst):
    """Monotonic interpolation, NaN outside the support -- no extrapolation."""
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    ok = np.isfinite(x_src) & np.isfinite(y_src)
    if ok.sum() < 2:
        return np.full(np.shape(x_dst), np.nan)
    xs, ys = x_src[ok], y_src[ok]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    # collapse duplicates so np.interp stays single valued
    uniq, inv = np.unique(xs, return_inverse=True)
    if uniq.size < xs.size:
        ys = np.array([ys[inv == i].mean() for i in range(uniq.size)])
        xs = uniq
    out = np.interp(x_dst, xs, ys, left=np.nan, right=np.nan)
    return out


def isotonic(y, w=None):
    """Pool-adjacent-violators: nearest non-decreasing sequence to y."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y.copy()
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    vals, wts, sizes = [], [], []
    for i in range(n):
        vals.append(y[i])
        wts.append(w[i])
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, s2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, s1 = vals.pop(), wts.pop(), sizes.pop()
            wt = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wt if wt > 0 else 0.5 * (v1 + v2))
            wts.append(wt)
            sizes.append(s1 + s2)
    out = np.empty(n)
    k = 0
    for v, s in zip(vals, sizes):
        out[k:k + s] = v
        k += s
    return out


def monotone_axis(z, w0, min_step=1e-6):
    """Strictly increasing w0(z) and the mask of the points kept (one per plateau)."""
    ok = np.isfinite(z) & np.isfinite(w0)
    z, w0 = np.asarray(z, float)[ok], np.asarray(w0, float)[ok]
    order = np.argsort(z)
    z, w0 = z[order], w0[order]
    mono = isotonic(w0)
    keep = np.ones(mono.size, dtype=bool)
    keep[1:] = np.diff(mono) > min_step
    return z[keep], mono[keep], ok


def reduce_lab(force_paths, scan_paths, material, w0_grid=None,
               n_grid=DEFAULT_N_GRID, anchor="groove_end", shift_um=0.0,
               scratch_length=DEFAULT_SCRATCH_LENGTH, scan_kw=None):
    """Reduce each repeat separately, then average them into the laboratory reference on w0."""
    scan_kw = scan_kw or {}
    scans = [read_scan(p, material, anchor=anchor, shift_um=shift_um,
                       scratch_length=scratch_length, **scan_kw)
             for p in scan_paths]
    forces = [read_force_csv(p) for p in force_paths]

    if len(scans) != len(forces):
        print("  ! %d scan(s) but %d force file(s): pairing by sorted name"
              % (len(scans), len(forces)))
    n_rep = min(len(scans), len(forces))

    # Put everything on a common z grid first, then make w0(z) monotone and invert it.
    z_lo = max(np.nanmin(s["z"]) for s in scans[:n_rep])
    z_hi = min(np.nanmax(s["z"]) for s in scans[:n_rep])
    n_z = max(4 * n_grid, 100)
    z_grid = np.linspace(z_lo, z_hi, n_z)

    keys = ["w0"] + [k for k, _, _, _ in OBSERVABLES]
    on_z = {k: [] for k in keys}
    for i in range(n_rep):
        s = scans[i]
        zf, fn, ft = forces[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            scof = np.where(fn > 0, ft / fn, np.nan)
        on_z["w0"].append(_resample(s["z"], s["w0"], z_grid))
        on_z["F_n"].append(_resample(zf, fn, z_grid))
        on_z["SCOF"].append(_resample(zf, scof, z_grid))
        on_z["A_ratio"].append(_resample(s["z"], s["A_ratio"], z_grid))
        on_z["h_r"].append(_resample(s["z"], s["h_r"], z_grid))
        on_z["h_p"].append(_resample(s["z"], s["h_p"], z_grid))

    mean_z, sd_z, n_z_pts = {}, {}, {}
    for k in keys:
        stack = np.vstack(on_z[k])
        with np.errstate(invalid="ignore"):
            mean_z[k] = np.nanmean(stack, axis=0)
            sd_z[k] = (np.nanstd(stack, axis=0, ddof=1) if stack.shape[0] > 1
                       else np.full(stack.shape[1], np.nan))
            n_z_pts[k] = np.sum(np.isfinite(stack), axis=0)

    z_mono, w0_mono, _ = monotone_axis(z_grid, mean_z["w0"])
    if w0_grid is None:
        lo, hi = float(w0_mono[0]), float(w0_mono[-1])
        w0_grid = np.linspace(lo, hi, n_grid)
    w0_grid = np.asarray(w0_grid, dtype=float)
    z_of_w0 = np.interp(w0_grid, w0_mono, z_mono, left=np.nan, right=np.nan)

    lab = {"w0": w0_grid, "n_repeats": n_rep, "scans": scans,
           "anchor": anchor, "shift_um": shift_um,
           "z_grid": z_grid, "w0_of_z": mean_z["w0"], "w0_mono": w0_mono,
           "z_mono": z_mono}
    for k, _n, _u, _s in OBSERVABLES:
        lab[k] = _resample(z_grid, mean_z[k], z_of_w0)
        lab[k + "_sd"] = _resample(z_grid, sd_z[k], z_of_w0)
        lab[k + "_n"] = _resample(z_grid, n_z_pts[k].astype(float), z_of_w0)
    lab["z"] = z_of_w0
    lab["z_sd"] = np.full(w0_grid.size, np.nan)
    lab["z_n"] = np.full(w0_grid.size, float(n_rep))

    # Uncertainty at fixed w0: sigma^2(Y|w0) = sigma_direct^2(Y) + (dY/dw0)^2 sigma^2(w0)
    lab["w0_sd"] = _resample(z_grid, sd_z["w0"], z_of_w0)
    for k, _n, _u, _s in OBSERVABLES:
        y = lab[k]
        dydw = np.gradient(y, w0_grid)
        induced = np.abs(dydw) * lab["w0_sd"]
        direct = lab[k + "_sd"]
        with np.errstate(invalid="ignore"):
            lab[k + "_sd_direct"] = direct
            lab[k + "_sd_w0"] = induced
            lab[k + "_sd"] = np.sqrt(np.nan_to_num(direct) ** 2
                                     + np.nan_to_num(induced) ** 2)
    return lab

def lab_sigma(lab, key, rel_floor=SIGMA_REL_FLOOR):
    """Uncertainty used to weight the chi-square: repeat scatter, floored at a fraction of the value."""
    sd = np.asarray(lab.get(key + "_sd"), dtype=float)
    val = np.abs(np.asarray(lab[key], dtype=float))
    floor = rel_floor * val
    out = np.where(np.isfinite(sd), np.maximum(sd, floor), floor)
    return np.where(out > 0, out, np.nan)


# Simulation side
_ID_RE = re.compile(r"Design[_-](\d+)")


def read_sim_curves(path):
    """sim_values long CSV -> {run_id: {observable: array}} on w0."""
    runs = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for rec in _csv.DictReader(f):
            fname = rec.get("file", "")
            m = _ID_RE.search(fname)
            rid = m.group(1).zfill(5) if m else fname
            if str(rec.get("present", "")).strip() not in ("1", "True", "true"):
                continue
            runs.setdefault(rid, {"file": fname, "rows": []})["rows"].append(rec)

    def num(rec, key):
        v = rec.get(key, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    out = {}
    for rid, blk in runs.items():
        rows = blk["rows"]
        cur = {
            "file": blk["file"],
            "w0": np.array([num(r, "w0_s [um]") for r in rows]),
            "w0_raw": np.array([num(r, "w0 [um]") for r in rows]),
            "F_n": np.array([num(r, "F_n [N]") for r in rows]),
            "SCOF": np.array([num(r, "SCOF [-]") for r in rows]),
            "A_ratio": np.array([num(r, "A_ratio [-]") for r in rows]),
            "h_r": np.array([num(r, "h_r [um]") for r in rows]),
            "h_p": np.array([num(r, "h_p [um]") for r in rows]),
            "d_cmd": np.array([num(r, "d_cmd [um]") for r in rows]),
        }
        # Same monotone treatment and trimming as on the laboratory side.
        z = np.array([num(r, "z [mm]") for r in rows])
        order = np.argsort(z)
        for k in list(cur):
            if k != "file":
                cur[k] = cur[k][order]
        z = z[order]
        i_deep = int(np.nanargmin(cur["h_r"])) if np.isfinite(cur["h_r"]).any() \
            else cur["h_r"].size - 1
        sl = slice(0, i_deep + 1)
        for k in list(cur):
            if k != "file":
                cur[k] = cur[k][sl]
        z = z[sl]
        z_m, w_m, _ = monotone_axis(z, cur["w0"])
        cur["z"] = z
        cur["w0_mono"] = w_m
        cur["z_mono"] = z_m
        out[rid] = cur
    return out


def read_design(path):
    """Design CSV -> (rows_by_id, factor names). Comment lines are skipped."""
    meta, header, body = [], None, []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                meta.append(line)
                continue
            if header is None:
                header = [c.strip() for c in line.split(",")]
                continue
            body.append([c.strip() for c in line.split(",")])
    if header is None:
        raise SystemExit("no header row in %s" % path)

    factors = []
    for m in meta:
        if "active_factors=" in m:
            factors = [s.strip() for s in
                       m.split("active_factors=", 1)[1].strip().split(",")]
    if not factors:
        factors = [c[2:] for c in header if c.startswith("g_")]

    rows = {}
    for parts in body:
        rec = dict(zip(header, parts))
        rid = str(rec.get("id", "")).strip().zfill(5)
        if rid:
            rows[rid] = rec
    return rows, factors, meta


# Scoring
def score_runs(sim, lab, design_rows, factors, rel_floor=SIGMA_REL_FLOOR,
               min_overlap=5):
    """Chi-square and signed relative bias of every run against the laboratory reference."""
    w0 = lab["w0"]
    sig = {k: lab_sigma(lab, k, rel_floor) for k, _, _, _ in OBSERVABLES}

    results = []
    for rid in sorted(sim):
        cur = sim[rid]
        rec = {"id": rid, "file": cur["file"]}
        drow = design_rows.get(rid, {})
        for f in factors:
            try:
                rec["g_" + f] = float(drow.get("g_" + f, "nan"))
            except ValueError:
                rec["g_" + f] = np.nan
        rec["block"] = drow.get("block", "")

        n_ov = 0
        for key, _lab_name, _unit, _scored in OBSERVABLES:
            z_at = np.interp(w0, cur["w0_mono"], cur["z_mono"],
                             left=np.nan, right=np.nan)
            y_sim = _resample(cur["z"], cur[key], z_at)
            ok = np.isfinite(y_sim) & np.isfinite(lab[key]) & np.isfinite(sig[key])
            n_ov = max(n_ov, int(ok.sum()))
            if ok.sum() < min_overlap:
                rec["chi2_" + key] = np.nan
                rec["bias_" + key] = np.nan
                continue
            resid = (y_sim[ok] - lab[key][ok]) / sig[key][ok]
            rec["chi2_" + key] = float(np.mean(resid ** 2))
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = (y_sim[ok] - lab[key][ok]) / np.where(
                    np.abs(lab[key][ok]) > 0, np.abs(lab[key][ok]), np.nan)
            rec["bias_" + key] = float(np.nanmedian(rel))
        rec["n_overlap"] = n_ov
        rec["w0_sim_max"] = float(np.nanmax(cur["w0"])) if cur["w0"].size else np.nan

        parts = [rec["chi2_" + k] for k in SCORED if np.isfinite(rec.get("chi2_" + k, np.nan))]
        rec["chi2"] = float(np.mean(parts)) if len(parts) == len(SCORED) else np.nan
        results.append(rec)
    return results


def factor_marginals(results, factors):
    """Median score and median bias at each level of each factor (main effects)."""
    out = []
    for f in factors:
        col = np.array([r.get("g_" + f, np.nan) for r in results], dtype=float)
        levels = sorted(set(v for v in col if np.isfinite(v)))
        for lv in levels:
            m = np.isfinite(col) & (np.abs(col - lv) < 1e-9)
            row = {"factor": f, "level": lv, "n": int(m.sum())}
            with np.errstate(invalid="ignore"):
                row["chi2"] = float(np.nanmedian(
                    [results[i]["chi2"] for i in np.flatnonzero(m)]))
                for key, _l, _u, _s in OBSERVABLES:
                    row["bias_" + key] = float(np.nanmedian(
                        [results[i]["bias_" + key] for i in np.flatnonzero(m)]))
            out.append(row)
    return out


def ridge_report(results, factors, top_frac=0.25):
    """Spread of each factor among the best runs, to show which factors the data identifies."""
    ok = [r for r in results if np.isfinite(r.get("chi2", np.nan))]
    if not ok:
        return []
    ok.sort(key=lambda r: r["chi2"])
    n_top = max(3, int(round(top_frac * len(ok))))
    top = ok[:n_top]
    out = []
    for f in factors:
        allv = sorted(set(r["g_" + f] for r in ok if np.isfinite(r["g_" + f])))
        topv = sorted(set(r["g_" + f] for r in top if np.isfinite(r["g_" + f])))
        span_all = (max(allv) - min(allv)) if len(allv) > 1 else 0.0
        span_top = (max(topv) - min(topv)) if len(topv) > 1 else 0.0
        out.append({
            "factor": f, "n_top": len(top),
            "levels_all": allv, "levels_top": topv,
            "span_fraction": (span_top / span_all) if span_all > 0 else np.nan,
            "best": top[0]["g_" + f],
        })
    return out


# Outputs
def write_lab_csv(lab, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = ["z"] + [k for k, _, _, _ in OBSERVABLES]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        head = ["w0 [um]"]
        for k in keys:
            head += ["%s" % k, "%s_sd" % k, "%s_n" % k]
        w.writerow(head)
        for i in range(lab["w0"].size):
            row = ["%.6g" % lab["w0"][i]]
            for k in keys:
                for suffix in ("", "_sd", "_n"):
                    v = lab[k + suffix][i]
                    row.append("" if not np.isfinite(v) else "%.6g" % v)
            w.writerow(row)
    print("Wrote laboratory reference -> %s" % path)


def write_scores_csv(results, factors, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    head = (["id", "file", "block", "chi2"]
            + ["g_" + f for f in factors]
            + ["chi2_" + k for k, _, _, _ in OBSERVABLES]
            + ["bias_" + k for k, _, _, _ in OBSERVABLES]
            + ["n_overlap", "w0_sim_max"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(head)
        for r in sorted(results, key=lambda r: (np.isnan(r.get("chi2", np.nan)),
                                                r.get("chi2", np.inf))):
            row = []
            for h in head:
                v = r.get(h, "")
                if isinstance(v, float):
                    row.append("" if not np.isfinite(v) else "%.6g" % v)
                else:
                    row.append(v)
            w.writerow(row)
    print("Wrote run scores -> %s" % path)


def plot_overview(lab, sim, results, factors, out_png, n_best=3):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    keys = [k for k, _, _, _ in OBSERVABLES]
    labels = {k: "%s [%s]" % (n, u) for k, n, u, _ in OBSERVABLES}
    ok = [r for r in results if np.isfinite(r.get("chi2", np.nan))]
    ok.sort(key=lambda r: r["chi2"])
    best = ok[:n_best]

    fig, axes = plt.subplots(len(keys), 1, figsize=(8.0, 2.5 * len(keys)),
                             sharex=True)
    for ax, key in zip(np.atleast_1d(axes), keys):
        y, sd = lab[key], lab[key + "_sd"]
        ax.fill_between(lab["w0"], y - sd, y + sd, color="0.80", zorder=0,
                        label="lab, %d repeats (+/- 1 sd)" % lab["n_repeats"])
        ax.plot(lab["w0"], y, "k-", lw=1.8, label="lab mean")
        for r in best:
            cur = sim[r["id"]]
            y = _resample(cur["z"], cur[key],
                          np.interp(lab["w0"], cur["w0_mono"], cur["z_mono"],
                                    left=np.nan, right=np.nan))
            ax.plot(lab["w0"], y, lw=1.2,
                    label="%s  chi2=%.1f" % (r["id"], r["chi2"]))
        ax.set_ylabel(labels[key])
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="best")
    np.atleast_1d(axes)[-1].set_xlabel("residual groove width w0 [um]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Wrote %s" % out_png)


def plot_marginals(marginals, factors, out_png):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(factors), figsize=(3.0 * len(factors), 6.0),
                             squeeze=False)
    for j, f in enumerate(factors):
        rows = [m for m in marginals if m["factor"] == f]
        lv = [m["level"] for m in rows]
        axes[0][j].plot(lv, [m["chi2"] for m in rows], "o-", color="#1565c0")
        axes[0][j].set_title(f, fontsize=9)
        axes[0][j].set_ylabel("median chi2" if j == 0 else "")
        axes[0][j].grid(alpha=0.3)
        for key, _n, _u, scored in OBSERVABLES:
            if not scored:
                continue
            axes[1][j].plot(lv, [100.0 * m["bias_" + key] for m in rows],
                            "o-", lw=1.2, label=key)
        axes[1][j].axhline(0.0, color="k", lw=0.8)
        axes[1][j].set_xlabel("level")
        axes[1][j].set_ylabel("median bias [%]" if j == 0 else "")
        axes[1][j].grid(alpha=0.3)
    axes[1][0].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Wrote %s" % out_png)


def plot_lab_repeats(lab, out_png):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for s in lab["scans"]:
        ax.plot(s["z"], s["w0"], lw=1.1, label=Path(s["path"]).name)
    ax.set_xlabel("z along the scratch [mm] (anchored on %s)" % lab["anchor"])
    ax.set_ylabel("w0 [um]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Wrote %s" % out_png)


# Console report
def report(lab, results, marginals, ridge, factors, n_show=8):
    w0 = lab["w0"]
    print("\n" + "=" * 74)
    print("LABORATORY REFERENCE  (%d repeats, w0 = %.1f .. %.1f um, %d points)"
          % (lab["n_repeats"], w0[0], w0[-1], w0.size))
    print("=" * 74)
    print("  %-10s %10s %10s %10s" % ("observable", "at w0_lo", "at w0_hi",
                                      "median sd"))
    for key, name, unit, scored in OBSERVABLES:
        sd = lab[key + "_sd"]
        rel = np.nanmedian(np.abs(sd / np.where(np.abs(lab[key]) > 0,
                                                lab[key], np.nan)))
        print("  %-10s %10.4g %10.4g %9.1f %%   %s"
              % (key, lab[key][0], lab[key][-1], 100 * rel,
                 "scored" if scored else "diagnostic"))

    ok = [r for r in results if np.isfinite(r.get("chi2", np.nan))]
    ok.sort(key=lambda r: r["chi2"])
    print("\n" + "=" * 74)
    print("RANKING  (%d of %d runs scored on all three observables)"
          % (len(ok), len(results)))
    print("=" * 74)
    head = "  %-6s %8s" % ("id", "chi2")
    head += "".join("%9s" % ("g_" + f)[:9] for f in factors)
    head += "".join("%9s" % ("chi2_" + k)[:9] for k in SCORED)
    head += "".join("%8s" % ("d" + k)[:8] for k in SCORED)
    print(head)
    for r in ok[:n_show]:
        line = "  %-6s %8.1f" % (r["id"], r["chi2"])
        line += "".join("%9.3g" % r["g_" + f] for f in factors)
        line += "".join("%9.1f" % r["chi2_" + k] for k in SCORED)
        line += "".join("%7.0f%%" % (100 * r["bias_" + k]) for k in SCORED)
        print(line)
    print("  (d<obs> = median signed bias of the simulation against the lab)")

    print("\n" + "=" * 74)
    print("MAIN EFFECTS  (median over the other factors)")
    print("=" * 74)
    for f in factors:
        rows = [m for m in marginals if m["factor"] == f]
        print("  %s" % f)
        print("    %-10s %8s %9s %9s %9s"
              % ("level", "chi2", "dF_n", "dA_ratio", "dSCOF"))
        for m in rows:
            print("    %-10.4g %8.1f %8.0f%% %8.0f%% %8.0f%%"
                  % (m["level"], m["chi2"], 100 * m["bias_F_n"],
                     100 * m["bias_A_ratio"], 100 * m["bias_SCOF"]))

    print("\n" + "=" * 74)
    print("IDENTIFIABILITY  (spread of each factor among the best %d runs)"
          % (ridge[0]["n_top"] if ridge else 0))
    print("=" * 74)
    for r in ridge:
        frac = r["span_fraction"]
        verdict = ("constrained" if np.isfinite(frac) and frac <= 0.34
                   else "partly constrained" if np.isfinite(frac) and frac <= 0.67
                   else "NOT separated by these observables")
        print("  %-12s best=%-8.4g levels kept: %-24s span %s  -> %s"
              % (r["factor"], r["best"],
                 ",".join("%g" % v for v in r["levels_top"]),
                 ("%.0f %%" % (100 * frac)) if np.isfinite(frac) else "n/a",
                 verdict))


# CLI
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Score a simulation sweep against averaged laboratory "
                    "scratch tests, on the common w0 abscissa.")
    p.add_argument("--sim", required=True,
                   help="long CSV produced by sim_values.py")
    p.add_argument("--design", default=None,
                   help="design CSV, for the g_* factor levels "
                        "(e.g. designs/glassy_pmma_calib.csv)")
    p.add_argument("--lab", required=True,
                   help="prefix or glob of the laboratory repeats, e.g. "
                        "'PMMAXT_6N_1Nm*' -- .csv are the force files, "
                        ".bcrf the scans")
    p.add_argument("--material", default="pmma", choices=("pmma", "pc"),
                   help="recovery preset used by bcrf_values")
    p.add_argument("--out-dir", default="calib_fit")
    p.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID,
                   help="number of w0 points in the comparison grid")
    p.add_argument("--anchor", default="groove_end",
                   choices=("groove_end", "mound", "deepest"),
                   help="scan feature mapped onto z = scratch_length")
    p.add_argument("--anchor-shift", type=float, default=0.0,
                   help="manual shift of the anchor, in um")
    p.add_argument("--length", type=float, default=DEFAULT_SCRATCH_LENGTH,
                   help="scratch length in mm (default %.1f)"
                        % DEFAULT_SCRATCH_LENGTH)
    p.add_argument("--sigma-floor", type=float, default=SIGMA_REL_FLOOR,
                   help="relative floor on the laboratory scatter "
                        "(default %.2f)" % SIGMA_REL_FLOOR)
    p.add_argument("--rough", action="store_true",
                   help="pass --rough settings to bcrf_values (wavy surfaces)")
    p.add_argument("--n-show", type=int, default=8,
                   help="rows of the ranking printed to the console")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    args = p.parse_args(argv)

    pattern = args.lab
    force_paths = sorted(glob.glob(pattern + ".csv") + glob.glob(pattern))
    force_paths = sorted(set(p for p in force_paths if p.lower().endswith(".csv")))
    scan_paths = sorted(set(glob.glob(pattern + ".bcrf")
                            + [p for p in glob.glob(pattern)
                               if p.lower().endswith(".bcrf")]))
    if not force_paths or not scan_paths:
        raise SystemExit("no laboratory files matched %r (%d csv, %d bcrf)"
                         % (pattern, len(force_paths), len(scan_paths)))

    print("Laboratory: %d force file(s), %d scan(s)"
          % (len(force_paths), len(scan_paths)))
    for a, b in zip(force_paths, scan_paths):
        print("   %-28s <-> %s" % (Path(a).name, Path(b).name))

    scan_kw = {"measure_win": 60.0} if args.rough else {}
    lab = reduce_lab(force_paths, scan_paths, args.material,
                     n_grid=args.n_grid, anchor=args.anchor,
                     shift_um=args.anchor_shift, scratch_length=args.length,
                     scan_kw=scan_kw)

    sim = read_sim_curves(args.sim)
    print("Simulation: %d run(s) in %s" % (len(sim), Path(args.sim).name))

    if args.design:
        design_rows, factors, _meta = read_design(args.design)
        missing = sorted(set(design_rows) - set(sim))
        if missing:
            print("  ! %d design point(s) with no simulated curve: %s"
                  % (len(missing), ", ".join(missing)))
    else:
        design_rows, factors = {}, []

    results = score_runs(sim, lab, design_rows, factors,
                         rel_floor=args.sigma_floor)
    marginals = factor_marginals(results, factors) if factors else []
    ridge = ridge_report(results, factors) if factors else []

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    write_lab_csv(lab, os.path.join(out, "lab_reference.csv"))
    write_scores_csv(results, factors, os.path.join(out, "run_scores.csv"))
    if args.plot:
        plot_lab_repeats(lab, os.path.join(out, "lab_repeats.png"))
        plot_overview(lab, sim, results, factors,
                      os.path.join(out, "best_runs.png"))
        if marginals:
            plot_marginals(marginals, factors,
                           os.path.join(out, "main_effects.png"))

    report(lab, results, marginals, ridge, factors, n_show=args.n_show)
    return 0


if __name__ == "__main__":
    sys.exit(main())