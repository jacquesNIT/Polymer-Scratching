# -*- coding: utf-8 -*-
# Calibration design for glassy_pc (Exolon GP) -- explicit factorial.
#
#   python3 Generators/generate_pc_calib.py                # -> designs/glassy_pc_calib.csv
#   python3 Generators/generate_pc_calib.py --blocks CORE
#
#   abaqus cae noGUI=run_parameter_study.py -- design glassy_pc_calib
#
# FILE NAME: designs/glassy_pc_calib.csv, with no method suffix. This design is
# a full factorial, not a Sobol sequence and not a Morris trajectory set;
# naming it _sobol would put a false method in the file name and in every
# table that later joins on it. _design_path() in run_parameter_study.py
# takes the bare <family>.csv first for exactly that reason.
#
# WHY THIS IS NOT THE PMMA DESIGN WITH OTHER NUMBERS
#
#   1. p_ref is 230 MPa, not 400. The 400 MPa of the PMMA box is the
#      measured PMMA XT scratch pressure. Reduced laboratory data -- 4
#      repeats at 4 N / 1 N per min, compared at equal residual groove
#      width -- give PC 53-61 % of the PMMA normal force for the same w0
#      (2.56 N against 4.86 N at w0 = 110 um). The apparent contact
#      pressure follows, so PC's reference is about 230 MPa. Since the
#      whole (tau0, alpha) parametrisation is anchored on p_ref, reusing
#      400 MPa would define mu_eff at a pressure PC never reaches.
#
#   2. sigma_scale is centred on 1, not on 1.55. The families.py PC baseline
#      has sigma_y0 = 70 MPa against 103 for PMMA, a ratio of 0.68 -- close
#      to the measured force ratio. The PC starting point is probably
#      already good to 10-20 %, so the 1.55x that PMMA needs must NOT be
#      transposed here.
#
#   3. psi is centred lower and starts at 0. PC recovers far less than PMMA
#      (w0/|h_r| = 13.0 against 18.4) and its measured A_pile/A_groove is
#      slightly lower (0.87 against 0.93), so less displaced volume goes
#      missing and less dilatancy is needed to account for the balance.
#      families.py already puts PC at psi = 0.
#
#   4. The depth is 25 um, not 20. PC reaches w0 = 137 um in the laboratory
#      against 126 um for PMMA; the simulated range has to reach that far to
#      overlap it. 25 um is still under the 26.8 um sphere/cone tangency, so
#      the contact stays on the spherical cap. CHECK after the first runs
#      that the simulated w0 actually covers 137 um -- if it does not,
#      calib_fit.py will score on a truncated window.
#
# BLOCKS
#   BASE   1 run   the families.py PC point: psi 0, soft_drop 12,
#                  sigma_scale 1, and the Hossain friction pair
#                  (tau0 = 38.32 MPa, alpha = 0.06), which at p_ref = 230
#                  MPa is mu_eff = 0.227 and phi = 0.265. That this lands in
#                  the middle of the mu_eff window is the sanity check that
#                  the 230 MPa anchor is right.
#   CORE  27 runs  full factorial psi(3) x sigma_scale(3) x mu_eff(3) at
#                  soft_drop = 12 and phi = 0.318. Corners first.
#   SOFT   2 runs  soft_drop 6 and 18 at the centre of CORE -- a check that
#                  holding it does not move the other residuals, not an
#                  attempt to calibrate it. Identify it with a depth sweep.
#   SPLIT  2 runs  phi 0.545 and 0.773 at the centre of CORE.
#
#   Every row is reachable through the g_* columns; nothing here needs
#   set: tokens.

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

FAMILY = "glassy_pc_calib"
DEFAULT_NAME = FAMILY

# Factor levels. Centred on the families.py PC point, not on the PMMA one.
PSI = (0.0, 7.5, 15.0)
SCALE = (0.9, 1.15, 1.4)
MU = (0.23, 0.32, 0.41)

# Held levels, and the probes that check them.
SOFT_HELD = 12.0                  # families.py PC value
SOFT_PROBE = (6.0, 18.0)
PHI_HELD = 0.3182                 # same split as the PMMA design, for comparability
PHI_PROBE = (0.5455, 0.7727)

# Reference point: the families.py PC values, Hossain friction included.
#   tau0 = 38.32 MPa, alpha = 0.06 at p_ref = 230 MPa
#   -> mu_eff = 0.06 + 38.32/230 = 0.2266,  phi = 0.06/0.2266 = 0.2648
BASE = {"psi": 0.0, "soft_drop": 12.0, "sigma_scale": 1.0,
        "mu_eff": 0.2266, "phi": 0.2648}

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
        fh.write("# PC calibration design -- generated by generate_pc_calib.py\n")
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
                 "Every row here IS\n#       reachable through g_* -- no "
                 "set: token is needed for any of them.\n")
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