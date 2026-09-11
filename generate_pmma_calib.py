# -*- coding: utf-8 -*-
# Calibration design for glassy_pmma -- explicit factorial, not a sampler.
#
#   python3 generate_pmma_calib.py            # -> designs/glassy_pmma_calib.csv
#   python3 generate_pmma_calib.py --blocks CORE
#
# Emits exactly the format generate_design.py produces, so
# run_parameter_study.py reads it unchanged:
#
#   abaqus cae noGUI=run_parameter_study.py -- design glassy_pmma_calib
#
# BLOCKS
#   BASE   1 run   current families.py values (tau0=4, alpha=0.2). Reference
#                  point: everything else is read as a delta from it.
#   CORE  27 runs  full factorial psi(3) x soft_drop(3) x split(3) at
#                  sigma_scale = 1. A one-factor-at-a-time plan (3+3+3) plus
#                  two 3x3 blocks costs 21 unique points and gives no
#                  three-way information; the full 3^3 costs 27 and gives
#                  every main effect, every two-way interaction and the
#                  three-way. Six extra jobs for a complete picture.
#   LEVEL  8 runs  sigma_scale {1.3, 1.6} x soft_drop {25, 8} x split
#                  {(20,.17), (60,.07)}. Two reasons this block exists:
#                  (a) soft_drop alone CANNOT reach the lab target -- even
#                  soft_drop = 0 only lifts the effective yield from 78 to
#                  103 MPa, a factor 1.32, against a diagnosed force deficit
#                  of 1.55-1.67. The level has to be a factor of its own.
#                  (b) sigma_scale and friction are the structurally
#                  confounded pair (both raise Fn, both narrow w0), so their
#                  interaction is the one worth mapping.
#   PROBE  2 runs  beta = 30 deg at baseline and at sigma_scale = 1.3.
#                  PMMA is quoted as at least as pressure-sensitive as PC,
#                  yet families.py gives it beta = 25 against PC's 27.13.
#                  Under 400 MPa of hydrostatic compression a few degrees of
#                  beta move the effective flow stress a lot, so beta could
#                  absorb part of the 1.6x. Two jobs settle whether it is a
#                  first-order contributor before the whole deficit is
#                  pinned on sigma_y.
#
# RUN ORDER: BASE, then PROBE, then the four corner cells of CORE, then the
# rest. The Morris screening already flagged informative censoring in the
# high-softening + high-friction corner (element distortion, aborted jobs).
# soft_drop = 25 with tau0 = 60 IS that corner. Find out on 3 jobs, not by
# discovering holes in a 27-point factorial.

import argparse
import csv
import os
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.path.abspath(".")
sys.path.insert(0, os.path.dirname(_HERE))

from ScratchSimulation.AbaqusModel.Configuration import get_family
from ScratchSimulation.AbaqusModel.Configuration.sampling import (
    CAMPAIGNS, BRISCOE_SPLITS)

FAMILY = "glassy_pmma_calib"
PSI = (10.0, 5.0, 0.0)
SOFT = (25.0, 18.0, 8.0)
SPLIT = (0, 1, 2)                 # -> BRISCOE_SPLITS
DEFAULT_OUT = os.path.join(_HERE, "designs")


def blocks():
    """(block, priority, psi, soft_drop, tau_split, sigma_scale, beta_override)."""
    rows = []
    # BASE -- current families.py point. tau0=4/alpha=0.2 is not on the split
    # grid, so it is carried as its own split index -1 (see _split_of below).
    rows.append(("BASE", 0, 10.0, 25.0, -1, 1.0, None))

    # PROBE -- beta sonde
    rows.append(("PROBE", 1, 10.0, 25.0, 1, 1.0, 30.0))
    rows.append(("PROBE", 1, 10.0, 25.0, 1, 1.3, 30.0))

    # CORE -- full 3^3. Corners first (priority 2), interior after (3).
    for psi in PSI:
        for sd in SOFT:
            for sp in SPLIT:
                corner = (psi in (PSI[0], PSI[-1]) and sd in (SOFT[0], SOFT[-1])
                          and sp in (SPLIT[0], SPLIT[-1]))
                rows.append(("CORE", 2 if corner else 3, psi, sd, sp, 1.0, None))

    # LEVEL -- sigma_scale x soft_drop x split, reduced
    for sc in (1.3, 1.6):
        for sd in (25.0, 8.0):
            for sp in (0, 2):
                rows.append(("LEVEL", 4, 10.0, sd, sp, sc, None))
    return rows


def _split_of(idx):
    if idx < 0:
        return (4.0, 0.2)          # families.py placeholder
    return BRISCOE_SPLITS[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=None,
                    help="comma-separated subset, e.g. BASE,PROBE,CORE")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    spec = CAMPAIGNS[FAMILY]
    cfg = get_family(FAMILY).build_config()
    keep = set(s.strip().upper() for s in args.blocks.split(",")) if args.blocks else None

    rows = [r for r in blocks() if keep is None or r[0] in keep]
    rows.sort(key=lambda r: (r[1], r[0]))

    out = []
    for i, (blk, prio, psi, sd, sp, sc, beta) in enumerate(rows):
        groups = {"psi": psi, "soft_drop": sd,
                  "tau_split": float(max(sp, 0)), "sigma_scale": sc}
        params = spec.derive(groups, cfg)
        if sp < 0:                                  # BASE placeholder friction
            t0, al = _split_of(-1)
            params["tau0"], params["alpha"] = t0, al
            params["mu_eff"] = al + t0 / params["p_ref_MPa"]
            params["phi"] = al / params["mu_eff"]
        if beta is not None:
            params["friction_angle"] = beta
        out.append({"id": "%05d" % (i + 1), "block": blk, "priority": prio,
                    "groups": groups, "params": params})

    group_names = [f.name for f in spec.factors]
    param_names = sorted(out[0]["params"].keys())
    header = (["id", "campaign", "family", "method", "block", "priority",
               "traj", "step", "moved", "sign"]
              + ["u_%s" % n for n in group_names]
              + ["g_%s" % n for n in group_names]
              + ["p_%s" % n for n in param_names])

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    path = os.path.join(args.out_dir, "%s.csv" % FAMILY)

    with open(path, "w") as fh:
        fh.write("# Calibration design -- generated by generate_pmma_calib.py\n")
        fh.write("# campaign=%s family=%s label=%s\n"
                 % (spec.campaign, FAMILY, spec.label))
        fh.write("# method=factorial n_runs=%d n_factors=%d\n"
                 % (len(out), len(group_names)))
        fh.write("# active_factors=%s\n" % ",".join(group_names))
        for f in spec.factors:
            fh.write("#   %s in [%g, %g] (%s) %s\n"
                     % (f.name, f.lo, f.hi, f.scale, f.description))
        fh.write("# briscoe_splits=%s  (iso-mu_eff at p_ref)\n"
                 % ";".join("%g/%g" % s for s in BRISCOE_SPLITS))
        fh.write("# frozen=%s\n"
                 % ",".join("%s=%s" % (k, v) for k, v in sorted(spec.frozen.items())))
        fh.write("# geometry: tip_radius=%.4g mm cone_angle=%.4g deg "
                 "scratch_depth=%.4g mm\n"
                 % (cfg.indenter.tip_radius, cfg.indenter.cone_angle,
                    cfg.scratch.scratch_depth))
        fh.write("# NOTE: p_* columns are traceability only; the Abaqus side "
                 "re-derives\n#       them in-kernel from the g_* columns.\n")
        fh.write("# NOTE: p_friction_angle on the PROBE rows and p_tau0/p_alpha on\n"
                 "#       the BASE row are NOT reachable through g_* -- run those\n"
                 "#       with set: tokens (see the block comment).\n")
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


if __name__ == "__main__":
    main()