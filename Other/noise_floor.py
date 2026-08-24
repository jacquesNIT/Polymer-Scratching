# Measure the numerical noise floor sigma_num of every QoI (CPython).
#
# Two modes, both feeding the JSON that morris_analysis.py consumes.
#
#   replicates : runs of the SAME nominal point that differ only by numerical
#                settings (mesh, target dt, mass scale). sigma = std over them.
#
#     python3 noise_floor.py replicates /path/to/noise_runs --out noise.json
#
#   scaling    : the exact stress-scale invariance test. Runs of the same point
#                with every stress-carrying constant multiplied by lambda.
#                Forces/energies must scale exactly by lambda and geometry must
#                be invariant, so the residual scatter after normalising is
#                numerical error with a KNOWN exact answer.
#
#     python3 noise_floor.py scaling /path/to/scale_runs --lambdas 1,2,4 \
#             --out noise.json
#
# In scaling mode the run directories (or filenames) must contain the tag
# lam<value>, e.g. Design_lam2_Results.csv or a subdirectory named lam2.

import argparse
import json
import os
import re
import sys

import numpy as np

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.path.abspath(".")
sys.path.insert(0, _HERE)

import sweep_collector as SC


FORCE_LIKE = ("Fn_half_N", "Ft_half_N", "H_MPa")
GEOMETRY_LIKE = ("residual_depth_mm", "pile_up_mm", "pile_up_max_mm",
                 "pile_up_ratio", "residual_rel_pct")
DIMENSIONLESS = ("scof",)
ALL_QOI = FORCE_LIKE + GEOMETRY_LIKE + DIMENSIONLESS


def _runs(root):
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for n in sorted(files):
            if n.endswith("_Results.csv"):
                out.append(os.path.join(dirpath, n))
    return out


def _qoi(path):
    metadata, timeseries, nodes = SC.RV.parse_results_csv(path)
    return SC.compute_qoi(metadata, timeseries, nodes)


def _lam_of(path, lambdas):
    m = re.search(r"lam([0-9.]+)", path)
    if m:
        return float(m.group(1))
    if len(lambdas) == 1:
        return lambdas[0]
    raise SystemExit("Cannot read a lam<value> tag from %s" % path)


def main():
    ap = argparse.ArgumentParser(description="Numerical noise floor per QoI.")
    ap.add_argument("mode", choices=("replicates", "scaling"))
    ap.add_argument("root", help="directory walked recursively for *_Results.csv")
    ap.add_argument("--lambdas", default="1", help="scaling mode: comma-separated lambda values")
    ap.add_argument("--qoi", default=None, help="comma-separated QoI subset")
    ap.add_argument("--out", default="noise_floor.json")
    ap.add_argument("--group-by-dir", action="store_true",
                    help="treat each sub-directory as one replicate group and keep the "
                         "WORST group (measure the floor at several points of the domain)")
    args = ap.parse_args()

    paths = _runs(args.root)
    if len(paths) < 2:
        raise SystemExit("Need at least 2 runs under %s (found %d)" % (args.root, len(paths)))
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    qois = [q.strip() for q in args.qoi.split(",")] if args.qoi else list(ALL_QOI)

    groups = {}
    for p in paths:
        key = os.path.dirname(p) if args.group_by_dir else "_all"
        try:
            vals = _qoi(p)
        except Exception as exc:
            print("  skipped %s (%s)" % (os.path.basename(p), str(exc)[:60]))
            continue
        lam = _lam_of(p, lambdas) if args.mode == "scaling" else 1.0
        acc = groups.setdefault(key, dict((q, []) for q in qois))
        for q in qois:
            v = vals.get(q, float("nan"))
            if not np.isfinite(v):
                continue
            if args.mode == "scaling" and q in FORCE_LIKE:
                v = v / lam
            acc[q].append(v)

    sigma, sigma_rel, means, report = {}, {}, {}, []
    for q in qois:
        best = None
        for key, acc in sorted(groups.items()):
            a = np.asarray(acc[q], dtype=float)
            if a.size < 2:
                continue
            sd, mean = float(np.std(a, ddof=1)), float(np.mean(a))
            rel = sd / abs(mean) if mean else float("nan")
            if best is None or rel > best[2]:
                best = (sd, mean, rel, a.size, os.path.basename(key))
        if best is None:
            report.append((q, 0, np.nan, np.nan, np.nan, "-"))
            continue
        sigma[q], means[q], sigma_rel[q] = best[0], best[1], best[2]
        report.append((q, best[3], best[1], best[0], 100.0 * best[2], best[4]))

    print("")
    print("  %-22s %5s %14s %14s %9s  %s"
          % ("QoI", "n", "mean", "sigma_num", "rel %", "worst group"))
    for q, n, mean, sd, rel, key in report:
        print("  %-22s %5d %14.6g %14.6g %8.3f%%  %s" % (q, n, mean, sd, rel, key))

    suspect = [q for q in sigma_rel if sigma_rel[q] > 0.20]
    if suspect:
        print("")
        print("  WARNING: relative scatter exceeds 20%% on %s." % ", ".join(sorted(suspect)))
        print("  These runs do not look like replicates of ONE point. Do not point this")
        print("  script at a sweep directory: replicates must differ ONLY by numerical")
        print("  settings (mesh, target dt, mass scale), never by material parameters.")

    payload = {"mode": args.mode, "n_runs": len(paths),
               "n_groups": len(groups), "grouped": bool(args.group_by_dir),
               "sigma": sigma, "sigma_rel": sigma_rel, "mean": means}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("")
    print("Wrote %s (mode=%s, %d runs, %d group(s))"
          % (args.out, args.mode, len(paths), len(groups)))
    print("Explicit numerical error is largely PROPORTIONAL to the QoI, so prefer "
          "morris_analysis.py --noise-mode relative, and measure the floor at "
          "several points of the domain with --group-by-dir.")
    if args.mode == "scaling":
        print("Reminder: in scaling mode the exact answer is known, so this sigma is a "
              "pure numerical-error estimate. Fixed-factor mass scaling breaks the "
              "invariance slightly -- use target_time_increment for this test.")


if __name__ == "__main__":
    main()