# Morris elementary-effects analysis of a collected sweep (CPython).
#
#   python3 morris_analysis.py sweep_glassy_pc.csv \
#           --design designs/glassy_pc_morris.csv \
#           --noise-floor noise_glassy_pc.json \
#           --out-dir analysis_glassy_pc
#
# noise-floor JSON maps a QoI name to its numerical standard deviation in the
# units of that QoI, e.g. {"Fn_half_N": 1.2e-3, "scof": 0.004}.

import argparse
import csv
import json
import os
import re

import numpy as np


DEFAULT_QOI = ["Fn_half_N", "Ft_half_N", "scof", "H_MPa",
               "residual_depth_mm", "pile_up_mm", "pile_up_ratio"]
N_BOOTSTRAP = 2000
CI = (5.0, 95.0)


def read_design(path):
    meta = {"delta": None, "active": [], "family": None,
            "campaign": None, "method": None, "r": None}
    rows = {}
    data_lines = []
    with open(path, "r") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                m = re.search(r"delta=([0-9.eE+-]+)", line)
                if m:
                    meta["delta"] = float(m.group(1))
                m = re.search(r"\br=(\d+)", line)
                if m and meta["r"] is None:
                    meta["r"] = int(m.group(1))
                m = re.search(r"active_factors=(\S+)", line)
                if m:
                    meta["active"] = [s for s in m.group(1).strip().split(",") if s]
                m = re.search(r"campaign=(\S+)\s+family=(\S+)", line)
                if m:
                    meta["campaign"], meta["family"] = m.group(1), m.group(2)
                m = re.search(r"method=(\w+)", line)
                if m and meta["method"] is None:
                    meta["method"] = m.group(1)
            else:
                data_lines.append(line)
    for rec in csv.DictReader(data_lines):
        rows[str(rec["id"]).strip()] = rec
    return meta, rows


def read_table(path):
    with open(path, "r") as f:
        return dict((str(r["id"]).strip(), r) for r in csv.DictReader(f))


def _num(rec, key):
    v = rec.get(key, "")
    if v in (None, "", "nan", "NaN"):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def elementary_effects(design_rows, table, qoi, delta, drop_status=("FAIL",)):
    """Return (list of (traj, factor, EE), n_missing)."""
    by_traj = {}
    for rid, d in design_rows.items():
        by_traj.setdefault(int(d["traj"]), {})[int(d["step"])] = (rid, d)

    effects, n_missing = [], 0
    for traj, steps in sorted(by_traj.items()):
        ordered = [steps[k] for k in sorted(steps)]
        values = []
        for rid, _d in ordered:
            rec = table.get(rid)
            if rec is None or str(rec.get("status", "")) in drop_status:
                values.append(float("nan"))
                n_missing += 1
            else:
                values.append(_num(rec, qoi))
        for k in range(1, len(ordered)):
            _rid, d = ordered[k]
            y0, y1 = values[k - 1], values[k]
            if not (np.isfinite(y0) and np.isfinite(y1)):
                continue
            sign = float(d.get("sign", 1.0) or 1.0)
            if sign == 0.0:
                continue
            effects.append((traj, d["moved"], (y1 - y0) / (sign * delta)))
    return effects, n_missing


def summarise(effects, factors, n_bootstrap=N_BOOTSTRAP, seed=0, ci_low=None):
    per_factor = dict((f, []) for f in factors)
    per_traj = {}
    for traj, fac, ee in effects:
        if fac in per_factor:
            per_factor[fac].append(ee)
            per_traj.setdefault(traj, []).append((fac, ee))

    rs = np.random.RandomState(seed)
    trajs = sorted(per_traj)
    boot = dict((f, []) for f in factors)
    if trajs and n_bootstrap > 0:
        for _ in range(n_bootstrap):
            pick = rs.randint(0, len(trajs), size=len(trajs))
            acc = dict((f, []) for f in factors)
            for idx in pick:
                for fac, ee in per_traj[trajs[idx]]:
                    acc[fac].append(abs(ee))
            for f in factors:
                boot[f].append(np.mean(acc[f]) if acc[f] else np.nan)

    lo_pct = CI[0] if ci_low is None else float(ci_low)
    out = {}
    for f in factors:
        vals = np.asarray(per_factor[f], dtype=float)
        if vals.size == 0:
            out[f] = {"mu_star": np.nan, "sigma": np.nan, "mu": np.nan, "n_eff": 0,
                      "mu_star_lo": np.nan, "mu_star_hi": np.nan}
            continue
        b = np.asarray(boot[f], dtype=float)
        b = b[np.isfinite(b)]
        out[f] = {
            "mu_star": float(np.mean(np.abs(vals))),
            "sigma": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "mu": float(np.mean(vals)),
            "n_eff": int(vals.size),
            "mu_star_lo": float(np.percentile(b, lo_pct)) if b.size else np.nan,
            "mu_star_hi": float(np.percentile(b, CI[1])) if b.size else np.nan,
        }
    return out


def _plot(summary, factors, qoi, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for f in factors:
        s = summary[f]
        if not np.isfinite(s["mu_star"]):
            continue
        ax.scatter(s["mu_star"], s["sigma"], s=45)
        ax.annotate(f, (s["mu_star"], s["sigma"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    lim = max([summary[f]["mu_star"] for f in factors
               if np.isfinite(summary[f]["mu_star"])] or [1.0])
    ax.plot([0, lim], [0, lim], ls="--", lw=0.8, color="0.6")
    ax.set_xlabel("mu*  (importance)")
    ax.set_ylabel("sigma  (non-linearity / interactions)")
    ax.set_title("Morris screening -- %s" % qoi)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Morris elementary-effects analysis.")
    ap.add_argument("table", help="tidy CSV produced by sweep_collector.py")
    ap.add_argument("--design", required=True, help="Morris design CSV")
    ap.add_argument("--qoi", default=None, help="comma-separated QoI columns")
    ap.add_argument("--noise-floor", default=None,
                    help="JSON written by noise_floor.py (or a flat {qoi: sigma})")
    ap.add_argument("--noise-mode", default="relative", choices=("relative", "absolute"),
                    help="relative: sigma is a FRACTION of the QoI, rescaled by the "
                         "mean of that QoI over the design (default, matches the "
                         "proportional error of Explicit)")
    ap.add_argument("--out-dir", default="analysis")
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--keep-failed", action="store_true",
                    help="keep runs whose verifier status is FAIL")
    args = ap.parse_args()

    meta, design_rows = read_design(args.design)
    if meta["method"] and meta["method"] != "morris":
        raise SystemExit("Design '%s' is a %s design; this script analyses Morris designs."
                         % (args.design, meta["method"]))
    if not meta["delta"] or not meta["active"]:
        raise SystemExit("Could not read delta / active_factors from the design header.")

    table = read_table(args.table)
    factors = meta["active"]
    delta = meta["delta"]
    drop = () if args.keep_failed else ("FAIL",)

    qois = [q.strip() for q in args.qoi.split(",")] if args.qoi else DEFAULT_QOI
    available = set()
    for rec in table.values():
        available.update(rec.keys())
    qois = [q for q in qois if q in available]
    if not qois:
        raise SystemExit("None of the requested QoI columns are present in %s" % args.table)

    noise, noise_is_rel = {}, (args.noise_mode == "relative")
    if args.noise_floor:
        with open(args.noise_floor, "r") as f:
            payload = json.load(f)
        if "sigma" in payload and isinstance(payload["sigma"], dict):
            noise = payload["sigma_rel" if noise_is_rel else "sigma"]
        else:
            noise = payload
            if noise_is_rel:
                raise SystemExit(
                    "Noise file '%s' is a flat {qoi: sigma} map with no sigma_rel. "
                    "Re-run noise_floor.py, or pass --noise-mode absolute."
                    % args.noise_floor)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    rows, retained = [], set()
    for qoi in qois:
        effects, n_missing = elementary_effects(design_rows, table, qoi, delta, drop)
        summary = summarise(effects, factors, n_bootstrap=args.bootstrap)
        mu_max = max([summary[f]["mu_star"] for f in factors
                      if np.isfinite(summary[f]["mu_star"])] or [np.nan])
        sig_num = float(noise[qoi]) if qoi in noise else None
        if sig_num is not None and noise_is_rel:
            scale = np.nanmean([_num(rec, qoi) for rec in table.values()])
            sig_num = sig_num * abs(scale)
        ee_noise = (sig_num * np.sqrt(2.0) / delta) if sig_num is not None else None
        # Under the null, EE ~ N(0, ee_noise) so mu* = mean|EE| has expectation
        # sqrt(2/pi)*ee_noise, NOT zero: that is the reference to test against.
        mu_null = (np.sqrt(2.0 / np.pi) * ee_noise) if ee_noise is not None else None

        print("")
        print("QoI %-20s  (%d elementary effects, %d missing points)"
              % (qoi, len(effects), n_missing))
        if ee_noise is not None:
            print("  numerical EE floor = %.4g  (sigma_num = %.4g, %s)"
                  % (ee_noise, sig_num, args.noise_mode))
        mde = None
        if mu_null is not None:
            spread = [s["mu_star_hi"] - s["mu_star_lo"] for s in summary.values()
                      if np.isfinite(s["mu_star_hi"]) and np.isfinite(s["mu_star_lo"])]
            mde = mu_null + (0.5 * float(np.median(spread)) if spread else 0.0)
            print("  mu* null reference = %.4g | minimum detectable mu* ~ %.4g"
                  % (mu_null, mde))
        print("  %-12s %11s %11s %11s %11s %6s %10s" %
              ("factor", "mu*", "mu*_lo", "mu*_hi", "sigma", "n_eff", "verdict"))
        for f in factors:
            s = summary[f]
            if mu_null is None:
                verdict = "n/a"
            elif not np.isfinite(s["mu_star_hi"]):
                verdict = "no data"
            elif s["mu_star_lo"] > mu_null:
                verdict = "RETAIN"
                retained.add(f)
            else:
                verdict = "freeze"
            print("  %-12s %11.5g %11.5g %11.5g %11.5g %6d %10s"
                  % (f, s["mu_star"], s["mu_star_lo"], s["mu_star_hi"],
                     s["sigma"], s["n_eff"], verdict))
            rows.append({
                "qoi": qoi, "factor": f,
                "mu_star": s["mu_star"], "mu_star_lo": s["mu_star_lo"],
                "mu_star_hi": s["mu_star_hi"], "sigma": s["sigma"], "mu": s["mu"],
                "n_eff": s["n_eff"],
                "rel_importance": (s["mu_star"] / mu_max) if mu_max else np.nan,
                "ee_noise_floor": ("" if ee_noise is None else ee_noise),
                "mu_star_null": ("" if mu_null is None else mu_null),
                "mu_star_mde": ("" if mde is None else mde),
                "verdict": verdict,
            })
        _plot(summary, factors, qoi, os.path.join(args.out_dir, "morris_%s.png" % qoi))

    summary_path = os.path.join(args.out_dir, "morris_summary.csv")
    with open(summary_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    keep = set(retained)
    fam = meta["family"] or "unknown"
    out_json = {
        "family": fam, "campaign": meta["campaign"], "delta": delta,
        "factors": factors, "qoi": qois,
        "noise_floor_supplied": bool(noise), "noise_mode": args.noise_mode,
        "retained": sorted(retained),
        "keep_for_sobol": sorted(keep),
        "frozen_candidates": sorted(set(factors) - keep),
    }
    json_path = os.path.join(args.out_dir, "retained_factors_%s.json" % fam)
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    print("")
    print("Summary -> %s" % summary_path)
    print("Retention -> %s" % json_path)
    if not noise:
        print("NOTE: no --noise-floor supplied, so no factor was retained or frozen; "
              "mu*/sigma are reported but the decision is left open.")
    else:
        print("Retained : %s" % (", ".join(sorted(retained)) or "(none)"))
        print("Freeze   : %s" % (", ".join(sorted(set(factors) - keep)) or "(none)"))
        print("A 'freeze' means no evidence of an effect above the noise floor, NOT "
              "proof of nullity: see mu_star_mde in morris_summary.csv for what this "
              "campaign was able to detect.")
        print("Next: generate_design.py %s --method sobol --n 1024 --only %s"
              % (fam, ",".join(sorted(keep)) or "..."))


if __name__ == "__main__":
    main()