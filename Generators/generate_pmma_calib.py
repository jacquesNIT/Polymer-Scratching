# -*- coding: utf-8 -*-
# Second-generation calibration design for glassy_pmma -- explicit factorial.
#
#   python3 Generators/generate_pmma_calib.py              # -> designs/glassy_pmma_calib.csv
#   python3 Generators/generate_pmma_calib.py --blocks CORE
#
# Same format as generate_design.py, so run_parameter_study.py reads it
# unchanged:
#
#   abaqus cae noGUI=run_parameter_study.py -- design glassy_pmma_calib
#
# FILE NAME: designs/glassy_pmma_calib.csv, with no method suffix. This design is
# a full factorial, not a Sobol sequence and not a Morris trajectory set;
# naming it _sobol would put a false method in the file name and in every
# table that later joins on it. _design_path() in run_parameter_study.py
# takes the bare <family>.csv first for exactly that reason.
#
# WHAT CHANGED SINCE THE FIRST SWEEP
#   The 38-point sweep was scored against 4 laboratory repeats at 6 N /
#   1 N per min, compared at equal residual groove width. Three findings
#   drive this design.
#
#   1. mu_eff was the missing factor. tau_split only moved the SPLIT of the
#      friction at mu_eff = 0.22 frozen. But raising sigma_y raises the
#      contact pressure, and mu_eff = alpha + tau0/p falls with it: the SCOF
#      residual went from +2 % at sigma_scale = 1.0 to -22 % at 1.6. The
#      friction LEVEL is now a factor; the split (phi) is held at one level.
#
#   2. psi and sigma_scale both sat on their upper bound. psi = 10 deg won
#      every comparison with none of the best runs below it; sigma_scale
#      left a -4 % force residual at 1.6 against -13 % at 1.0. Both windows
#      now start where the old ones ended.
#
#   3. soft_drop is not identifiable here. It spanned its whole range among
#      the best runs. At the representative strain of a 20 um scratch the
#      softening is 80 % complete, so soft_drop and sigma_scale enter only
#      through their product. It is held at 25 MPa and probed on two runs;
#      identify it with a DEPTH sweep (2-3 points on the ridge x 4 depths in
#      8-26 um), not here.
#
# BLOCKS
#   BASE   1 run   run 00032 of the first sweep -- psi 10, soft_drop 25,
#                  sigma_scale 1.3, (tau0, alpha) = (60, 0.07) i.e.
#                  mu_eff 0.22 / phi 0.318. Best combined score of that
#                  sweep (chi2 = 23.1) and best topographic score (14.5).
#                  Re-running it here is what makes the two sweeps
#                  comparable: same mesh, same post-processing, one common
#                  point.
#   CORE  27 runs  full factorial psi(3) x sigma_scale(3) x mu_eff(3) at
#                  soft_drop = 25 and phi = 0.318. Corners first: the
#                  high-softening + high-friction corner aborted on the
#                  Morris screening, and psi 20 / sigma_scale 1.8 /
#                  mu_eff 0.40 is its analogue here. Find out on 8 jobs,
#                  not by discovering holes in a 27-point factorial.
#   SOFT   2 runs  soft_drop 18 and 8 at the centre of CORE. Not an attempt
#                  to calibrate it -- a check that freezing it at 25 does
#                  not move the other residuals.
#   SPLIT  2 runs  phi 0.545 and 0.773 at the centre of CORE, i.e. the two
#                  other Briscoe splits of the first sweep. Same purpose.
#
#   Unlike the first design, EVERY row is reachable through the g_* columns:
#   there is no beta probe and no off-grid friction placeholder, so nothing
#   here needs set: tokens and nothing can silently run a different point.
#   A beta sonde, if still wanted, has to be a separate depth/single job.

import argparse
import csv
import os
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.path.abspath(".")


def _locate_root(start):
    """Directory that CONTAINS the ScratchSimulation package.

    Walked up rather than hard coded so the script runs the same whether it
    sits in ScratchSimulation/Generators/ or, as it used to, directly in
    ScratchSimulation/. Nothing else in the tree has to know where the
    generators live.
    """
    d = start
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "ScratchSimulation", "AbaqusModel",
                                      "Configuration")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(start)          # pre-Generators layout


_ROOT = _locate_root(_HERE)                # holds ScratchSimulation/
_PKG = os.path.join(_ROOT, "ScratchSimulation")
sys.path.insert(0, _ROOT)

from ScratchSimulation.AbaqusModel.Configuration import get_family
from ScratchSimulation.AbaqusModel.Configuration.sampling import CAMPAIGNS

FAMILY = "glassy_pmma_calib"
DEFAULT_NAME = FAMILY

# Factor levels. psi and sigma_scale start where the first sweep stopped.
PSI = (10.0, 15.0, 20.0)
SCALE = (1.3, 1.55, 1.8)
MU = (0.22, 0.31, 0.40)

# Held levels, and the probes that check them.
SOFT_HELD = 25.0
SOFT_PROBE = (18.0, 8.0)
PHI_HELD = 0.3182                 # = the (60, 0.07) Briscoe split at mu 0.22
PHI_PROBE = (0.5455, 0.7727)      # = (40, 0.12) and (20, 0.17)

# Reference point: run 00032 of the first sweep.
BASE = {"psi": 10.0, "soft_drop": 25.0, "sigma_scale": 1.3,
        "mu_eff": 0.22, "phi": PHI_HELD}

# Designs stay in ScratchSimulation/designs: run_parameter_study.py
# resolves them relative to ITS own location, and it did not move.
DEFAULT_OUT = os.path.join(_PKG, "designs")


def blocks():
    """(block, priority, groups dict)."""
    rows = [("BASE", 0, dict(BASE))]

    centre = {"psi": PSI[1], "soft_drop": SOFT_HELD, "sigma_scale": SCALE[1],
              "mu_eff": MU[1], "phi": PHI_HELD}

    for psi in PSI:
        for sc in SCALE:
            for mu in MU:
                corner = (psi in (PSI[0], PSI[-1]) and sc in (SCALE[0], SCALE[-1])
                          and mu in (MU[0], MU[-1]))
                rows.append(("CORE", 1 if corner else 3,
                             {"psi": psi, "soft_drop": SOFT_HELD,
                              "sigma_scale": sc, "mu_eff": mu,
                              "phi": PHI_HELD}))

    for sd in SOFT_PROBE:
        g = dict(centre)
        g["soft_drop"] = sd
        rows.append(("SOFT", 2, g))

    for phi in PHI_PROBE:
        g = dict(centre)
        g["phi"] = phi
        rows.append(("SPLIT", 2, g))
    return rows


def write_design(path, spec, cfg, out, group_names):
    param_names = sorted(out[0]["params"].keys())
    header = (["id", "campaign", "family", "method", "block", "priority",
               "traj", "step", "moved", "sign"]
              + ["u_%s" % n for n in group_names]
              + ["g_%s" % n for n in group_names]
              + ["p_%s" % n for n in param_names])

    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, "w") as fh:
        fh.write("# Calibration design v2 -- generated by generate_pmma_calib.py\n")
        fh.write("# campaign=%s family=%s label=%s\n"
                 % (spec.campaign, FAMILY, spec.label))
        fh.write("# method=factorial n_runs=%d n_factors=%d\n"
                 % (len(out), len(group_names)))
        fh.write("# active_factors=%s\n" % ",".join(group_names))
        for f in spec.factors:
            fh.write("#   %s in [%g, %g] (%s) %s\n"
                     % (f.name, f.lo, f.hi, f.scale, f.description))
        fh.write("# levels: psi=%s sigma_scale=%s mu_eff=%s\n"
                 % (",".join("%g" % v for v in PSI),
                    ",".join("%g" % v for v in SCALE),
                    ",".join("%g" % v for v in MU)))
        fh.write("# held: soft_drop=%g (probed at %s) phi=%g (probed at %s)\n"
                 % (SOFT_HELD, ",".join("%g" % v for v in SOFT_PROBE),
                    PHI_HELD, ",".join("%g" % v for v in PHI_PROBE)))
        fh.write("# frozen=%s\n"
                 % ",".join("%s=%s" % (k, v) for k, v in sorted(spec.frozen.items())))
        fh.write("# geometry: tip_radius=%.4g mm cone_angle=%.4g deg "
                 "scratch_depth=%.4g mm\n"
                 % (cfg.indenter.tip_radius, cfg.indenter.cone_angle,
                    cfg.scratch.scratch_depth))
        fh.write("# NOTE: p_* columns are traceability only; the Abaqus side "
                 "re-derives\n#       them in-kernel from the g_* columns. "
                 "Every row here IS\n#       reachable through g_*, unlike "
                 "the v1 BASE and PROBE rows.\n")
        w = csv.writer(fh)
        w.writerow(header)
        for r in out:
            w.writerow([r["id"], spec.campaign, FAMILY, "factorial",
                        r["block"], r["priority"], 0, 0, "", 0]
                       + ["%.10g" % spec.factor(n).to_unit(r["groups"][n])
                          for n in group_names]
                       + ["%.10g" % r["groups"][n] for n in group_names]
                       + ["%.10g" % float(r["params"][n]) for n in param_names])
    print("Wrote %d runs -> %s" % (len(out), path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=None,
                    help="comma-separated subset, e.g. BASE,CORE")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--out-name", default=DEFAULT_NAME,
                    help="file stem (default: the family name, no method "
                         "suffix -- this design is a factorial)")
    args = ap.parse_args()

    spec = CAMPAIGNS[FAMILY]
    cfg = get_family(FAMILY).build_config()
    keep = set(s.strip().upper() for s in args.blocks.split(",")) if args.blocks else None

    rows = [r for r in blocks() if keep is None or r[0] in keep]
    rows.sort(key=lambda r: (r[1], r[0]))

    # BASE is deliberately a point of the new box, so it can collide with a
    # CORE corner. Drop the duplicate and keep the first occurrence, which
    # sorting by priority makes the BASE one: paying for the same job twice
    # buys nothing, and two rows with identical g_* would give the collector
    # two ids for one physical point.
    seen, unique = set(), []
    for blk, prio, groups in rows:
        key = tuple(round(float(groups[k]), 9) for k in sorted(groups))
        if key in seen:
            print("   dropped duplicate %s point %s"
                  % (blk, ", ".join("%s=%g" % (k, groups[k])
                                    for k in sorted(groups))))
            continue
        seen.add(key)
        unique.append((blk, prio, groups))
    rows = unique

    group_names = [f.name for f in spec.factors]
    out = []
    for i, (blk, prio, groups) in enumerate(rows):
        missing = [n for n in group_names if n not in groups]
        if missing:
            raise SystemExit("block %s is missing factor(s): %s"
                             % (blk, ", ".join(missing)))
        params = spec.derive(groups, cfg)
        out.append({"id": "%05d" % (i + 1), "block": blk, "priority": prio,
                    "groups": groups, "params": params})

    path = os.path.join(args.out_dir, "%s.csv" % args.out_name)
    write_design(path, spec, cfg, out, group_names)

    counts = {}
    for r in out:
        counts[r["block"]] = counts.get(r["block"], 0) + 1
    print("   blocks: %s" % ", ".join("%s=%d" % kv for kv in sorted(counts.items())))
    print("   tau0 spans %.1f - %.1f MPa, alpha %.3f - %.3f, sigma_y floor "
          "%.1f - %.1f MPa"
          % (min(r["params"]["tau0"] for r in out),
             max(r["params"]["tau0"] for r in out),
             min(r["params"]["alpha"] for r in out),
             max(r["params"]["alpha"] for r in out),
             min(r["params"]["sigma_y_floor"] for r in out),
             max(r["params"]["sigma_y_floor"] for r in out)))


if __name__ == "__main__":
    main()