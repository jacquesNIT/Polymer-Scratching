# Aggregate a sweep into a single tidy table (CPython, not Abaqus).
#
#   python3 sweep_collector.py /path/to/results_dir \
#           --design designs/glassy_pc_morris.csv \
#           --out sweep_glassy_pc.csv
#
# <results_dir> is walked recursively for *_Results.csv, so it can be any
# directory name and layout: point it at whatever you created on the cluster.

import argparse
import csv
import os
import re
import sys

import numpy as np

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.path.abspath(".")
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)


QOI_WINDOW_START = 0.90     # fraction of the scratch step where the QoI window opens
STATUS_RANK = {"PASS": 0, "INFO": 1, "SKIP": 1, "WARN": 2, "FAIL": 3}


def _import_results_verifier():
    candidates = [
        "ScratchSimulation.AbaqusModel.Verification.results_verifier",
        "ScratchSimulation.AbaqusModel.Postprocessing.results_verifier",
        "ScratchSimulation.AbaqusModel.results_verifier",
        "results_verifier",
    ]
    for name in candidates:
        try:
            module = __import__(name, fromlist=["*"])
            if hasattr(module, "parse_results_csv"):
                return module
        except ImportError:
            continue
    for root, _dirs, files in os.walk(os.path.dirname(_HERE)):
        if "results_verifier.py" in files:
            path = os.path.join(root, "results_verifier.py")
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("results_verifier", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            except Exception:
                break
    raise SystemExit("Could not import results_verifier.py. Pass its directory on PYTHONPATH.")


RV = _import_results_verifier()


# ----------------------------------------------------------------------
# Quantities of interest
# ----------------------------------------------------------------------

def _series(ts, *names):
    for n in names:
        a = ts.get(n)
        if a is not None and len(a) and float(np.max(np.abs(a))) > 1e-20:
            return np.asarray(a, dtype=float), n
    return None, "unavailable"


def _scratch_window(metadata):
    t_scr = float(metadata.get("scratch_time", 0.0) or 0.0)
    if t_scr <= 0.0:
        return None
    mode = str(metadata.get("depth_mode", "progressive")).lower()
    if mode.startswith("prog"):
        t0, t1 = 0.0, t_scr
    else:
        t_ind = float(metadata.get("indentation_time", 0.0) or 0.0)
        t0, t1 = t_ind, t_ind + t_scr
    return t0 + QOI_WINDOW_START * (t1 - t0), t1


def _contact_radius(metadata):
    R = float(metadata.get("tip_radius", 0.2) or 0.2)
    ang = float(metadata.get("cone_angle", 60.0) or 60.0)
    d = abs(float(metadata.get("scratch_depth", 0.0) or 0.0))
    if d <= 0.0:
        return float("nan")
    beta = np.radians(ang)
    h_t = R * (1.0 - np.sin(beta))
    if d <= h_t:
        return float(np.sqrt(max(2.0 * R * d - d * d, 0.0)))
    return float(R * np.cos(beta) + (d - h_t) * np.tan(beta))


def _peak(ts, name):
    a = ts.get(name)
    if a is None or not len(a):
        return float("nan")
    return float(np.max(np.abs(np.asarray(a, dtype=float))))


def _ratio(num, den):
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-20:
        return float("nan")
    return float(num / den)


def compute_qoi(metadata, timeseries, nodes):
    out = {}
    time = timeseries.get("Time")
    win = _scratch_window(metadata)

    fn, fn_src = _series(timeseries, "RF2", "CFN2")
    ft, ft_src = _series(timeseries, "RF3", "CFS3")
    out["force_source"] = fn_src

    if time is not None and win is not None and fn is not None:
        mask = (time >= win[0]) & (time <= win[1] + 1e-12)
        n = int(np.count_nonzero(mask))
        out["n_window"] = n
        if n >= 3:
            fnw = np.abs(fn[mask])
            out["Fn_half_N"] = float(np.mean(fnw))
            out["Fn_cv_pct"] = _ratio(float(np.std(fnw)), float(np.mean(fnw))) * 100.0
            if ft is not None and len(ft) == len(fn):
                ftw = np.abs(ft[mask])
                out["Ft_half_N"] = float(np.mean(ftw))
                out["scof"] = _ratio(float(np.mean(ftw)), float(np.mean(fnw)))
            a = _contact_radius(metadata)
            out["contact_radius_mm"] = a
            if np.isfinite(a) and a > 0.0:
                out["H_MPa"] = 4.0 * out["Fn_half_N"] / (np.pi * a * a)
    out.setdefault("n_window", 0)

    prof = None
    try:
        prof = RV.measure_residual_profile(nodes, metadata, timeseries)
    except Exception as exc:
        out["profile_error"] = str(exc)[:120]
    if prof:
        out["residual_depth_mm"] = prof.get("residual_depth_mm", float("nan"))
        out["residual_rel_pct"] = prof.get("relative_percent", float("nan"))
        out["pile_up_mm"] = prof.get("pile_up_mm", float("nan"))
        out["pile_up_max_mm"] = prof.get("pile_up_max_mm", float("nan"))
        out["pile_up_ratio"] = _ratio(prof.get("pile_up_mm", float("nan")),
                                      prof.get("residual_depth_mm", float("nan")))
        out["profile_method"] = prof.get("profile_method", "")
        out["profile_n_fail_flags"] = len(prof.get("profile_fail_flags", []))
        out["profile_n_warn_flags"] = len(prof.get("profile_warn_flags", []))

    ie = _peak(timeseries, "ALLIE")
    wm_ie = _peak(timeseries, "WM_ALLIE")
    out["AE_over_IE_pct"] = _ratio(_peak(timeseries, "ALLAE"), ie) * 100.0
    out["KE_over_IE_pct"] = _ratio(_peak(timeseries, "ALLKE"), ie) * 100.0
    out["PW_over_IE_pct"] = _ratio(_peak(timeseries, "WM_ALLPW"), wm_ie) * 100.0
    out["MW_over_IE_pct"] = _ratio(_peak(timeseries, "WM_ALLMW"), wm_ie) * 100.0
    out["ETOTAL_drift_pct"] = _ratio(_peak(timeseries, "ETOTAL"), wm_ie) * 100.0
    out["wallclock_s"] = float(metadata.get("wallclock", float("nan")))
    return out


def _slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.split("(")[0].strip().lower()).strip("_")


def collect_quality(path):
    out, worst = {}, "PASS"
    try:
        report = RV.verify_results(path, print_report=False)
    except Exception as exc:
        return {"verify_error": str(exc)[:160]}, "FAIL"
    out["family_resolved"] = int(bool(report.get("family_resolved", True)))
    for label, res in report.get("checks", {}).items():
        st = str(res.get("status", "INFO"))
        out["q_" + _slug(label)] = st
        if STATUS_RANK.get(st, 1) > STATUS_RANK.get(worst, 0):
            worst = st
    return out, worst


# ----------------------------------------------------------------------
# Design join
# ----------------------------------------------------------------------

def read_design(path):
    rows = {}
    with open(path, "r") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    for rec in csv.DictReader(lines):
        rows[str(rec["id"]).strip()] = rec
    return rows


def _run_id(filename):
    m = re.search(r"Design_(\w+?)_Results\.csv$", filename)
    if m:
        return m.group(1)
    return filename[:-len("_Results.csv")] if filename.endswith("_Results.csv") else filename


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate a sweep into a tidy table.")
    ap.add_argument("results_dir", help="directory containing the *_Results.csv (walked recursively)")
    ap.add_argument("--design", default=None, help="design CSV to join on the run id")
    ap.add_argument("--out", default="sweep_table.csv")
    ap.add_argument("--no-verify", action="store_true", help="skip the verifier checks (faster)")
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        raise SystemExit("Not a directory: %s" % args.results_dir)

    files = []
    for root, _dirs, names in os.walk(args.results_dir):
        for n in sorted(names):
            if n.endswith("_Results.csv"):
                files.append(os.path.join(root, n))
    if not files:
        raise SystemExit("No *_Results.csv found under %s" % args.results_dir)

    design = read_design(args.design) if args.design else {}
    records, n_err = [], 0

    for path in files:
        rid = _run_id(os.path.basename(path))
        rec = {"id": rid, "file": os.path.relpath(path, args.results_dir)}
        try:
            metadata, timeseries, nodes = RV.parse_results_csv(path)
            if not timeseries or "Time" not in timeseries:
                raise ValueError("no time-series rows: truncated or corrupt CSV")
            rec["family"] = str(metadata.get("family", ""))
            rec["fine_size_x"] = metadata.get("fine_size_x", float("nan"))
            rec["mass_scale"] = metadata.get("mass_scale", float("nan"))
            rec.update(compute_qoi(metadata, timeseries, nodes))
            if args.no_verify:
                rec["status"] = ""
            else:
                q, worst = collect_quality(path)
                rec.update(q)
                rec["status"] = worst
            rec["parse_error"] = ""
        except Exception as exc:
            n_err += 1
            rec["parse_error"] = str(exc)[:160]
            rec["status"] = "FAIL"
        if rid in design:
            for k, v in design[rid].items():
                if k not in ("id", "file"):
                    rec.setdefault(k, v)
        elif design:
            rec["design_missing"] = 1
        records.append(rec)

    lead = ["id", "family", "campaign", "method", "traj", "step", "moved", "sign", "status"]
    keys = []
    for name in lead:
        if any(name in r for r in records):
            keys.append(name)
    for r in records:
        for k in r:
            if k not in keys:
                keys.append(k)

    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="", extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    ok = sum(1 for r in records if r.get("status") not in ("FAIL",) and not r.get("parse_error"))
    print("Collected %d runs (%d usable, %d parse errors) -> %s"
          % (len(records), ok, n_err, args.out))
    if design:
        missing = [rid for rid in design if rid not in set(r["id"] for r in records)]
        if missing:
            print("MISSING %d design point(s) with no result: %s"
                  % (len(missing), ", ".join(sorted(missing)[:20])))


if __name__ == "__main__":
    main()