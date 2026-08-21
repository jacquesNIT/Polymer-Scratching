# Morris elementary-effects analysis of a collected sweep (CPython).
#
#   python3 morris_analysis.py sweep_glassy_pc.csv \
#           --design designs/glassy_pc_morris.csv \
#           --out-dir analysis_glassy_pc
#
# # [PATCH:no-noise-floor] retention on a RELATIVE threshold: a factor is retained when
# mu*_lo / mu*_max exceeds --retain-frac on at least one QoI. No absolute
# noise scale is used anywhere.

import argparse
import csv
import json
import os
import re

import numpy as np


# [PATCH:no-noise-floor] Ft_half_N (= Fn * scof), H_MPa (aire de contact constante,
# donc r(H, Fn) = 1 exactement) et pile_up_ratio (sigma/mu* > 1.3 partout)
# sont retires : redondants ou non exploitables.
DEFAULT_QOI = ["Fn_half_N", "scof", "residual_depth_mm", "pile_up_mm"]
RETAIN_FRAC = 0.20
N_BOOTSTRAP = 4000
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


# [PATCH:screening-fixes] begin -- definition unique de "run exploitable".
def _usable(rec, drop_status=("FAIL",)):
    """
    Vrai si le run peut alimenter un effet elementaire.

    `_coverage` (dp_screening) excluait `parse_error` alors que
    `elementary_effects` ne l'excluait pas : le nombre de runs annonce en
    tete du rapport ne correspondait pas a celui reellement utilise. Les
    deux passent desormais par ici.
    """
    if rec is None:
        return False
    if str(rec.get("status", "")) in drop_status:
        return False
    if rec.get("parse_error"):
        return False
    return True


# [PATCH:screening-fixes] end
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
            # [PATCH:screening-fixes] original : if rec is None or str(rec.get("status", "")) in drop_status:
            if not _usable(rec, drop_status):
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
    # [PATCH:screening-fixes] l'intervalle doit etre symetrique du test applique :
    # hi restait fige a 95 % meme quand lo etait corrige a 0.833 %.
    hi_pct = CI[1] if ci_low is None else (100.0 - float(ci_low))
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
            # [PATCH:screening-fixes] original : float(np.percentile(b, CI[1]))
            "mu_star_hi": float(np.percentile(b, hi_pct)) if b.size else np.nan,
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
    # [PATCH:no-noise-floor] --noise-floor / --noise-mode retires.
    ap.add_argument("--retain-frac", type=float, default=RETAIN_FRAC,
                    help="relative retention threshold: keep a factor when "
                         "mu*_lo / mu*_max exceeds this fraction on at least "
                         "one QoI (default %.2f)." % RETAIN_FRAC)
    ap.add_argument("--out-dir", default="analysis")
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--keep-failed", action="store_true",
                    help="keep runs whose verifier status is FAIL")
    # [PATCH:screening-fixes] meme correction de multiplicite que dp_screening.py,
    # sans quoi les deux scripts rendaient des verdicts differents sur les
    # memes donnees. La regle 'retenu pour au moins une QoI' est une union
    # de n_qoi tests ; sans correction le risque de faux positif par facteur
    # atteint 26 % sur 6 QoI.
    ap.add_argument("--fwer", default="qoi", choices=("qoi", "none"),
                    help="multiplicity correction over the QoI union test. "
                         "'qoi' (default, matches dp_screening.py): bootstrap "
                         "threshold corrected to alpha/n_qoi. 'none': 5%% per test.")
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

    # [PATCH:no-noise-floor] plus de lecture de plancher de bruit.

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # [PATCH:screening-fixes] begin -- correction de multiplicite (etait absente ici).
    n_tests = len(qois) if args.fwer == "qoi" else 1
    ci_low = CI[0] / float(n_tests)
    if args.fwer == "qoi" and args.bootstrap * ci_low / 100.0 < 20:
        need = int(np.ceil(20 * 100.0 / ci_low))
        print("  bootstrap porte a %d pour resoudre le percentile corrige %.3f%%"
              % (need, ci_low))
        args.bootstrap = need
    # [PATCH:screening-fixes] end
    rows, retained, marginal = [], set(), set()
    for qoi in qois:
        effects, n_missing = elementary_effects(design_rows, table, qoi, delta, drop)
        # [PATCH:screening-fixes] original : summarise(effects, factors, n_bootstrap=args.bootstrap)
        summary = summarise(effects, factors, n_bootstrap=args.bootstrap,
                            ci_low=ci_low)
        # [PATCH:no-noise-floor] verdict sur seuil RELATIF (mu*_lo / mu*_max).
        mu_max = max([summary[f]["mu_star"] for f in factors
                      if np.isfinite(summary[f]["mu_star"])] or [np.nan])
        has_max = bool(mu_max) and np.isfinite(mu_max)
        thr = (args.retain_frac * mu_max) if has_max else float("nan")

        print("")
        print("QoI %-20s  (%d elementary effects, %d missing points)"
              % (qoi, len(effects), n_missing))
        if has_max:
            print("  retention threshold = %.4g  (%.0f%% of mu*_max)"
                  % (thr, 100 * args.retain_frac))
        print("  %-12s %11s %11s %11s %11s %8s %6s %10s" %
              ("factor", "mu*", "mu*_lo", "mu*_hi", "sigma", "rel_lo", "n_eff", "verdict"))
        for f in factors:
            s = summary[f]
            rel = (s["mu_star"] / mu_max) if has_max else float("nan")
            rel_lo = (s["mu_star_lo"] / mu_max) if has_max else float("nan")
            if not np.isfinite(rel):
                verdict = "no data"
            elif np.isfinite(rel_lo) and rel_lo >= args.retain_frac:
                verdict = "RETAIN"
                retained.add(f)
            elif rel >= args.retain_frac:
                verdict = "RETAIN?"
                marginal.add(f)
            else:
                verdict = "freeze"
            print("  %-12s %11.5g %11.5g %11.5g %11.5g %8.3f %6d %10s"
                  % (f, s["mu_star"], s["mu_star_lo"], s["mu_star_hi"],
                     s["sigma"], rel_lo, s["n_eff"], verdict))
            rows.append({
                "qoi": qoi, "factor": f,
                "mu_star": s["mu_star"], "mu_star_lo": s["mu_star_lo"],
                "mu_star_hi": s["mu_star_hi"], "sigma": s["sigma"], "mu": s["mu"],
                "n_eff": s["n_eff"],
                "rel_importance": rel, "rel_lo": rel_lo,
                "mu_star_threshold": thr,
                "verdict": verdict,
            })
        _plot(summary, factors, qoi, os.path.join(args.out_dir, "morris_%s.png" % qoi))

    summary_path = os.path.join(args.out_dir, "morris_summary.csv")
    with open(summary_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # [PATCH:no-noise-floor] marginal ajoute, cles de bruit retirees.
    marginal = marginal - retained
    keep = set(retained) | marginal
    fam = meta["family"] or "unknown"
    out_json = {
        "family": fam, "campaign": meta["campaign"], "delta": delta,
        "factors": factors, "qoi": qois,
        "retain_frac": args.retain_frac, "decision_rule": "relative",
        "retained": sorted(retained),
        "marginal": sorted(marginal),
        "keep_for_sobol": sorted(keep),
        "frozen_candidates": sorted(set(factors) - keep),
    }
    json_path = os.path.join(args.out_dir, "retained_factors_%s.json" % fam)
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    print("")
    print("Summary -> %s" % summary_path)
    print("Retention -> %s" % json_path)
    # [PATCH:no-noise-floor] resume sans reference au bruit.
    print("Retained : %s" % (", ".join(sorted(retained)) or "(none)"))
    print("Marginal : %s" % (", ".join(sorted(marginal)) or "(none)"))
    print("Freeze   : %s" % (", ".join(sorted(set(factors) - keep)) or "(none)"))
    print("The threshold is RELATIVE (mu*_lo / mu*_max >= %.2f): the top factor is "
          "retained by construction and a 'freeze' means SMALL COMPARED TO THE "
          "LARGEST, not null." % args.retain_frac)
    print("Next: generate_design.py %s --method sobol --n 1024 --only %s"
          % (fam, ",".join(sorted(keep)) or "..."))


if __name__ == "__main__":
    main()