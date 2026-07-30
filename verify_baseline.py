# -*- coding: utf-8 -*-
"""
Baseline freeze and non-regression checker (plain CPython, no Abaqus).

This is the piece that answers "I change a lot of parameters and have no
reliable results to compare against". It creates one, then defends it.

    # after a run of  run_benchmarks.py -- baseline <family>
    python3 verify_baseline.py freeze  runs/Bench_baseline_glassy_pc/BenchOutputs
    python3 verify_baseline.py check   runs/Bench_baseline_glassy_pc/BenchOutputs
    python3 verify_baseline.py diff    runs/A/manifest_X.json runs/B/manifest_X.json
    python3 verify_baseline.py list

Rules enforced by 'freeze':
  * a baseline may NOT be created from a dirty git tree (use --allow-dirty
    only when you understand you are freezing something unreproducible);
  * every frozen case records its config_md5, so 'check' can tell
    "the answer moved" apart from "the question moved".

'check' prints, per quantity of interest, the deviation from the frozen value
and a verdict against a tolerance band. Run it BEFORE and AFTER every patch
campaign. Ten minutes of compute replaces the entire class of question
"is this difference my patch or something else?".
"""

from __future__ import print_function

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ScratchSimulation.AbaqusModel.Verification.benchmark_report import (       # noqa: E402
    read_csv, analyse_single_element)
from ScratchSimulation.AbaqusModel.Verification import analytic as an             # noqa: E402
from ScratchSimulation.AbaqusModel.Verification.manifest import (                 # noqa: E402
    read_manifest, print_diff, git_info)

BASELINE_DIR = os.path.join(_HERE, "baselines")
BASELINE_FILE = "baseline.json"

# Tolerance bands per quantity of interest [%]. Deliberately asymmetric:
# a material-point stress must be nearly exact, a scratch force need not be.
TOLERANCES = {
    "E_measured": 0.5,
    "sigma_y_measured": 1.0,
    "P_final": 1.0,
    "a_num_final": 3.0,
    "ALLIE_final": 2.0,
    "RF2_plateau": 2.0,
    "SCOF_plateau": 3.0,
    "residual_depth": 5.0,
}


# --------------------------------------------------------------------------
# Quantity extraction
# --------------------------------------------------------------------------

def qoi_from_element_csv(path):
    """
    Delegate to the report's analyser so the frozen quantities and the printed
    report can never disagree -- a baseline computed by a second, slightly
    different implementation is worse than no baseline.
    """
    meta, _d = read_csv(path)
    r = analyse_single_element(path)
    out = {}
    for k_src, k_dst in (("E_measured", "E_measured"),
                         ("sigma_y_measured", "sigma_y_measured"),
                         ("sigma_at_max_strain", "sigma_final"),
                         ("tau_max", "tau_final")):
        v = r.get(k_src)
        if isinstance(v, float) and np.isfinite(v):
            out[k_dst] = float(v)
    return meta, out


def qoi_from_indent_csv(path):
    meta, d = read_csv(path)
    out = {}
    rf2 = np.abs(d.get("RF2", np.array([])))
    u2 = np.abs(d.get("IndenterU2", np.array([])))
    m = np.isfinite(rf2) & np.isfinite(u2)
    if m.sum() > 3:
        out["P_final"] = float(rf2[m][-1])
        out["depth_final"] = float(u2[m][-1])
        E, nu = float(meta.get("E", np.nan)), float(meta.get("nu", np.nan))
        R = float(meta.get("tip_radius", 0.2))
        if np.isfinite(E) and np.isfinite(nu):
            pex = float(an.hertz_force(E, nu, R, out["depth_final"], half_model=True))
            out["P_exact"] = pex
            out["P_err_pct"] = 100.0 * (out["P_final"] - pex) / pex if pex else None
    ie = d.get("SUB_ALLIE", d.get("WM_ALLIE", np.array([])))
    if ie.size and np.isfinite(ie).any():
        out["ALLIE_final"] = float(np.nanmax(ie))
    contact = path.replace("_indent.csv", "_contact.csv")
    if os.path.exists(contact):
        _cm, cd = read_csv(contact)
        a = cd.get("a_num", np.array([]))
        if a.size and np.isfinite(a).any():
            out["a_num_final"] = float(np.nanmax(a))
    return meta, out


def qoi_from_scratch_csv(path):
    meta, d = read_csv(path)
    out = {}
    t = d.get("Time", np.array([]))
    rf2 = np.abs(d.get("RF2", np.array([])))
    rf3 = np.abs(d.get("RF3", np.array([])))
    T = float(meta.get("scratch_time", np.nan))
    m = np.isfinite(t) & np.isfinite(rf2) & (rf2 > 0)
    if m.sum() > 5:
        tt, f2 = t[m], rf2[m]
        f3 = rf3[m] if rf3.size == rf2.size else np.full_like(f2, np.nan)
        t_end = float(np.nanmax(tt[tt <= T])) if np.isfinite(T) else float(tt.max())
        w = (tt >= 0.6 * t_end) & (tt <= t_end)
        out["RF2_plateau"] = float(np.nanmean(f2[w]))
        out["RF2_plateau_std"] = float(np.nanstd(f2[w]))
        with np.errstate(invalid="ignore", divide="ignore"):
            out["SCOF_plateau"] = float(np.nanmean(f3[w] / np.maximum(f2[w], 1e-30)))
    return meta, out


def collect(folder):
    """Scan a benchmark output folder and return {case: {qoi: value}}."""
    cases = {}
    for path in sorted(glob.glob(os.path.join(folder, "*_element.csv"))):
        stem = os.path.basename(path)[:-len("_element.csv")]
        meta, q = qoi_from_element_csv(path)
        cases[stem] = {"kind": "element", "qoi": q, "meta_family": meta.get("family")}
    for path in sorted(glob.glob(os.path.join(folder, "*_indent.csv"))):
        stem = os.path.basename(path)[:-len("_indent.csv")]
        meta, q = qoi_from_indent_csv(path)
        cases[stem] = {"kind": "indent", "qoi": q, "meta_family": meta.get("family")}
    for path in sorted(glob.glob(os.path.join(folder, "*_Results.csv"))):
        stem = os.path.basename(path)[:-len("_Results.csv")]
        meta, q = qoi_from_scratch_csv(path)
        cases[stem] = {"kind": "scratch", "qoi": q, "meta_family": meta.get("family")}
    return cases


def attach_manifests(cases, folder):
    """Link each case to the manifest written next to it, if present."""
    for root in (folder, os.path.dirname(os.path.abspath(folder))):
        for path in glob.glob(os.path.join(root, "manifest_*.json")):
            stem = os.path.basename(path)[len("manifest_"):-len(".json")]
            if stem in cases:
                try:
                    man = read_manifest(path)
                    cases[stem]["config_md5"] = man.get("config_md5")
                    cases[stem]["git"] = man.get("git", {})
                    cases[stem]["derived"] = man.get("derived", {})
                except Exception as exc:
                    cases[stem]["manifest_error"] = str(exc)
    return cases


# --------------------------------------------------------------------------
# freeze / check
# --------------------------------------------------------------------------

def cmd_freeze(args):
    cases = attach_manifests(collect(args.folder), args.folder)
    if not cases:
        raise SystemExit("Nothing to freeze in %s" % args.folder)

    g = git_info(os.path.dirname(_HERE))
    if g.get("dirty") and not args.allow_dirty:
        raise SystemExit(
            "Refusing to freeze a baseline from a DIRTY working tree.\n"
            "A baseline that cannot be reproduced from a commit is not a\n"
            "baseline. Commit first, or pass --allow-dirty knowingly.\n"
            "Dirty files: %s" % ", ".join(g.get("dirty_files", [])[:10]))

    if not os.path.isdir(BASELINE_DIR):
        os.makedirs(BASELINE_DIR)
    name = args.name or os.path.basename(os.path.normpath(
        os.path.dirname(os.path.abspath(args.folder))))
    path = os.path.join(BASELINE_DIR, "%s_%s" % (name, BASELINE_FILE))
    payload = {"name": name, "source": os.path.abspath(args.folder),
               "git": g, "cases": cases, "note": args.note or ""}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    print("Baseline '%s' frozen: %d case(s) -> %s" % (name, len(cases), path))
    for k in sorted(cases):
        print("  %-32s %s" % (k, ", ".join("%s=%.6g" % (q, v)
                                           for q, v in sorted(cases[k]["qoi"].items())
                                           if isinstance(v, float))))
    return 0


def cmd_check(args):
    name = args.name or os.path.basename(os.path.normpath(
        os.path.dirname(os.path.abspath(args.folder))))
    path = os.path.join(BASELINE_DIR, "%s_%s" % (name, BASELINE_FILE))
    if not os.path.exists(path):
        raise SystemExit("No frozen baseline '%s' (%s). Run 'freeze' first."
                         % (name, path))
    with open(path, "r") as f:
        base = json.load(f)
    now = attach_manifests(collect(args.folder), args.folder)

    n_fail = n_warn = n_ok = 0
    print("Non-regression check against baseline '%s'" % name)
    print("  frozen at git %s%s"
          % ((base.get("git", {}).get("commit") or "n/a")[:10],
             "-DIRTY" if base.get("git", {}).get("dirty") else ""))
    print("")
    for case in sorted(set(base["cases"]) | set(now)):
        if case not in base["cases"]:
            print("  [NEW ] %s -- not in the baseline" % case)
            continue
        if case not in now:
            print("  [MISS] %s -- present in the baseline, absent now" % case)
            n_fail += 1
            continue
        b, c = base["cases"][case], now[case]
        cfg_moved = (b.get("config_md5") and c.get("config_md5")
                     and b["config_md5"] != c["config_md5"])
        if cfg_moved:
            print("  [CFG ] %s -- CONFIGURATION CHANGED (%s -> %s): a difference "
                  "below is expected, not a regression."
                  % (case, b["config_md5"][:8], c["config_md5"][:8]))
        for q in sorted(set(b["qoi"]) & set(c["qoi"])):
            vb, vc = b["qoi"][q], c["qoi"][q]
            if not (isinstance(vb, float) and isinstance(vc, float)) or vb == 0:
                continue
            dev = 100.0 * (vc - vb) / abs(vb)
            tol = TOLERANCES.get(q, 2.0)
            if abs(dev) <= 0.5 * tol:
                tag, n_ok = "OK  ", n_ok + 1
            elif abs(dev) <= tol:
                tag, n_warn = "WARN", n_warn + 1
            else:
                tag, n_fail = "FAIL", n_fail + 1
            if tag != "OK  " or args.verbose:
                print("  [%s] %-30s %-20s %.6g -> %.6g  (%+.2f %%, tol %.1f %%)"
                      % (tag, case, q, vb, vc, dev, tol))
    print("\n  %d OK, %d WARN, %d FAIL" % (n_ok, n_warn, n_fail))
    return 1 if n_fail else 0


def cmd_diff(args):
    a, b = read_manifest(args.a), read_manifest(args.b)
    print_diff(a, b, section="config")
    print("")
    print_diff(a, b, section="derived")
    return 0


def cmd_list(_args):
    if not os.path.isdir(BASELINE_DIR):
        print("No baseline yet.")
        return 0
    for path in sorted(glob.glob(os.path.join(BASELINE_DIR, "*_" + BASELINE_FILE))):
        with open(path, "r") as f:
            b = json.load(f)
        print("%-28s %3d case(s)  git %s%s  %s"
              % (b.get("name", "?"), len(b.get("cases", {})),
                 (b.get("git", {}).get("commit") or "n/a")[:10],
                 "-DIRTY" if b.get("git", {}).get("dirty") else "",
                 b.get("note", "")))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    f = sub.add_parser("freeze")
    f.add_argument("folder")
    f.add_argument("--name")
    f.add_argument("--note", default="")
    f.add_argument("--allow-dirty", action="store_true")
    f.set_defaults(func=cmd_freeze)

    c = sub.add_parser("check")
    c.add_argument("folder")
    c.add_argument("--name")
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=cmd_check)

    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    d.set_defaults(func=cmd_diff)

    l = sub.add_parser("list")
    l.set_defaults(func=cmd_list)

    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())