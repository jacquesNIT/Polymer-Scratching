# -*- coding: utf-8 -*-
"""
patch_calibration_jobs.py -- idempotent patch, three files.

WHAT I VERIFIED FIRST
---------------------
I checked your current sources rather than trusting my note, and my note was
wrong: NO calibration patch has been applied.

  * launch_cluster_jobs.py _OVERRIDE_ALIASES contains 25 keys, none of which
    is sigma_scale / tau0 / alpha / mu_cap / beta / psi / K. The 3 mm block
    (scratch_length, zs2, zs1, dpo_z) IS there, so patch_3mm_overrides was
    applied and patch_calibration_jobs was not.
  * run_parameter_study._apply_overrides is a plain dotted-path setattr walk
    with no pseudo-path interception, so patch_material_knobs was not applied
    either.

WHAT IS ALREADY REACHABLE WITHOUT ANY PATCH
-------------------------------------------
setattr on a dotted path already works for every knob that is a real
attribute. These need no patch at all, only a set: token:

    set:material.plasticity.dilation_angle=5.0     <- psi
    set:material.plasticity.friction_angle=30.0    <- beta
    set:material.plasticity.flow_stress_ratio=0.85 <- K
    set:material.hyperelastic.E=3300.0
    set:material.hyperelastic.nu=0.37

So I overstated the problem last time. Only the knobs CONSUMED AT
CONSTRUCTION are unreachable: sigma_y0, soft_drop, eps_soft and h are baked
into yield_table by gsell_jonas_table(), and tau0/alpha/mu_cap are baked
into mu_table by Friction_Config.briscoe(). Those are the ones that need
interception.

WHAT THIS PATCH DOES
--------------------
base.py                 gsell_jonas_table() returns a tuple SUBCLASS carrying
                        its own generating kwargs on .gsell. Still a tuple in
                        every respect (iteration, indexing, len, tuple()), so
                        nothing downstream changes -- but the table now knows
                        how it was built, which is what makes soft_drop
                        re-derivable without duplicating the literals from
                        families.py in a second registry that would drift.

run_parameter_study.py  _apply_overrides intercepts eight pseudo-paths under
                        material.*, collects them, and rebuilds the yield
                        table and the friction table ONCE at the end. Doing it
                        once rather than per token matters: sigma_scale and
                        soft_drop both touch the table, and applying them
                        sequentially would make the result order-dependent.

launch_cluster_jobs.py  A "Calibration" alias block (the five setattr-able
                        knobs plus the eight intercepted ones) and a commented
                        JOBS block for the campaign.

Discipline: exact-string anchors, CRLF preserved, .bak backups, ast.parse
validation, re-runnable (a second run is a no-op).

    python3 patch_calibration_jobs.py <path/to/AbaqusModel>
"""

import ast
import os
import shutil
import sys

MARK = "[calib-knobs-patch]"


# --------------------------------------------------------------------------
# base.py -- the yield table remembers how it was built
# --------------------------------------------------------------------------

BASE_CLASS = '''

# [calib-knobs-patch] begin
class GsellTable(tuple):
    """(sigma, eps) table that carries its own generating parameters.

    A tuple subclass, so it behaves exactly like the plain tuple the rest of
    the code expects: iteration, indexing, len(), tuple(t), Abaqus keyword
    writing are all unchanged. The only addition is .gsell, the kwargs dict
    gsell_jonas_table() was called with.

    Why: a calibration override such as soft_drop=8 has to REBUILD the table,
    which needs sigma_y0, h, Q, b, eps_soft, eps_max and n_points. Reading
    them back off the table would need a non-linear fit; duplicating them in
    a per-family registry would drift from families.py on the first edit.
    Carrying them is the only single-source option.

    Hand-written tables (glassy_dp) have no .gsell, and the override path
    says so explicitly instead of guessing.
    """
    gsell = None

    def with_kwargs(self, kw):
        self.gsell = dict(kw)
        return self
# [calib-knobs-patch] end

'''

BASE_EDITS = [
    ("def gsell_jonas_table(sigma_y0, h, Q=0.0, b=0.0, soft_drop=0.0, eps_soft=0.05,",
     BASE_CLASS.lstrip("\n")
     + "\ndef gsell_jonas_table(sigma_y0, h, Q=0.0, b=0.0, soft_drop=0.0, eps_soft=0.05,"),

    ("    return tuple((float(round(sv, 4)), float(round(ev, 6))) for sv, ev in zip(sig, eps))",
     "    # [calib-knobs-patch] original:\n"
     "    # return tuple((float(round(sv, 4)), float(round(ev, 6))) for sv, ev in zip(sig, eps))\n"
     "    _rows = tuple((float(round(sv, 4)), float(round(ev, 6)))\n"
     "                  for sv, ev in zip(sig, eps))\n"
     "    return GsellTable(_rows).with_kwargs(\n"
     "        {\"sigma_y0\": float(sigma_y0), \"h\": float(h), \"Q\": float(Q),\n"
     "         \"b\": float(b), \"soft_drop\": float(soft_drop),\n"
     "         \"eps_soft\": float(eps_soft), \"eps_max\": float(eps_max),\n"
     "         \"n_points\": int(n_points)})"),
]


# --------------------------------------------------------------------------
# run_parameter_study.py -- pseudo-path interception
# --------------------------------------------------------------------------

RPS_BLOCK = '''
# [calib-knobs-patch] begin -- material knobs that setattr cannot reach.
#
# sigma_y0 / soft_drop / eps_soft / h are arguments of gsell_jonas_table(),
# consumed into yield_table; tau0 / alpha / mu_cap are arguments of
# Friction_Config.briscoe(), consumed into mu_table. None of them survives as
# an attribute, so the dotted-path walk in _apply_overrides() cannot see them.
#
# They are intercepted here and applied ONCE, after every ordinary override,
# by rebuilding the two tables from merged parameters. Applying them one by
# one would make sigma_scale + soft_drop order-dependent, which is exactly
# the kind of silent coupling a calibration must not have.

_MATERIAL_KNOBS = (
    "sigma_scale",   # [-]   multiplies sigma_y0 AND soft_drop (pure stress scale)
    "sigma_y0",      # [MPa] absolute yield stress
    "soft_drop",     # [MPa] intrinsic softening depth
    "eps_soft",      # [-]   softening strain scale
    "h_gsell",       # [-]   G'Sell orientation hardening exponent
    "tau0",          # [MPa] Briscoe adhesive shear stress
    "alpha",         # [-]   Briscoe high-pressure asymptote
    "mu_cap",        # [-]   Briscoe table ceiling
)


def _briscoe_params(fric):
    """Recover (tau0, alpha, mu_cap) from an existing Briscoe table.

    briscoe() sets mu = alpha, so alpha is read directly. tau0 comes from any
    uncapped row: mu(p) = tau0/p + alpha. mu_cap is the plateau if one row is
    capped, otherwise the base.py default.
    """
    alpha = float(getattr(fric, "mu", 0.0))
    rows = list(getattr(fric, "mu_table", ()) or ())
    if not rows:
        return None
    mus = [float(m) for m, _ in rows]
    cap = max(mus)
    tau0 = 0.0
    for mu, p in reversed(rows):          # highest pressure = least likely capped
        if float(mu) < cap - 1e-9:
            tau0 = (float(mu) - alpha) * float(p)
            break
    return tau0, alpha, (cap if mus.count(cap) > 1 else 0.6)


def _apply_material_knobs(cfg, knobs):
    if not knobs:
        return
    mat = cfg.material

    wants_table = any(k in knobs for k in
                      ("sigma_scale", "sigma_y0", "soft_drop", "eps_soft", "h_gsell"))
    if wants_table:
        pl = getattr(mat, "plasticity", None)
        table = getattr(pl, "yield_table", None) if pl is not None else None
        kw = getattr(table, "gsell", None)
        if kw is None:
            raise SystemExit(
                "Material override %s: the yield table of family '%s' was not "
                "produced by gsell_jonas_table() (hand-written table, or base.py "
                "not patched), so it carries no generating parameters and cannot "
                "be rebuilt. Set the table in families.py instead."
                % (sorted(k for k in knobs if k != "tau0"), mat.family))
        kw = dict(kw)
        scale = float(knobs.get("sigma_scale", 1.0))
        if "sigma_y0" in knobs:
            kw["sigma_y0"] = float(knobs["sigma_y0"])
        if "soft_drop" in knobs:
            kw["soft_drop"] = float(knobs["soft_drop"])
        if "eps_soft" in knobs:
            kw["eps_soft"] = float(knobs["eps_soft"])
        if "h_gsell" in knobs:
            kw["h"] = float(knobs["h_gsell"])
        # pure stress scale: sigma_y0 and soft_drop together, so the SHAPE of
        # the post-yield branch is untouched and sigma_scale stays orthogonal
        # to soft_drop.
        kw["sigma_y0"] *= scale
        kw["soft_drop"] *= scale
        kw["Q"] *= scale
        if kw["soft_drop"] >= kw["sigma_y0"]:
            raise SystemExit(
                "Material override: soft_drop %.4g MPa >= sigma_y0 %.4g MPa; "
                "the yield stress would reach zero."
                % (kw["soft_drop"], kw["sigma_y0"]))
        pl.yield_table = gsell_jonas_table(**kw)
        print(">>> Override: yield table rebuilt %s"
              % ", ".join("%s=%.4g" % (k, v) for k, v in sorted(kw.items())))

    wants_fric = any(k in knobs for k in ("tau0", "alpha", "mu_cap"))
    if wants_fric:
        cur = _briscoe_params(getattr(mat, "friction", None))
        if cur is None:
            raise SystemExit(
                "Material override %s: family '%s' has no Briscoe friction "
                "table to rebuild (constant-mu friction?)."
                % (sorted(k for k in knobs if k in ("tau0", "alpha", "mu_cap")),
                   mat.family))
        tau0, alpha, cap = cur
        tau0 = float(knobs.get("tau0", tau0))
        alpha = float(knobs.get("alpha", alpha))
        cap = float(knobs.get("mu_cap", cap))
        mat.friction = Friction_Config.briscoe(
            tau0=tau0, alpha=alpha, mu_cap=cap, n_points=48,
            p_min=1.0, p_max=600.0)
        print(">>> Override: Briscoe rebuilt tau0=%.4g alpha=%.4g mu_cap=%.4g"
              % (tau0, alpha, cap))
# [calib-knobs-patch] end

'''

RPS_EDITS = [
    ("def _apply_overrides(cfg, overrides):",
     RPS_BLOCK.replace("\n", "\r\n").lstrip("\r\n")
     + "\r\ndef _apply_overrides(cfg, overrides):"),

    ("    for path, raw in (overrides or []):\r\n        obj = cfg\r\n        parts = path.split(\".\")",
     "    _knobs = {}                                  # [calib-knobs-patch]\r\n"
     "    for path, raw in (overrides or []):\r\n"
     "        # [calib-knobs-patch] material.<knob> pseudo-paths are collected,\r\n"
     "        # not setattr'd, and applied together once the loop is done.\r\n"
     "        if path.startswith(\"material.\") and path[9:] in _MATERIAL_KNOBS:\r\n"
     "            _knobs[path[9:]] = _parse_override_value(raw)\r\n"
     "            print(\">>> Override (material knob): %s = %r\"\r\n"
     "                  % (path[9:], _knobs[path[9:]]))\r\n"
     "            continue\r\n"
     "        obj = cfg\r\n        parts = path.split(\".\")"),

    ("        if path == \"solver.num_cpus\":     # keep MPI domains consistent\r\n"
     "            cfg.solver.num_domains = int(value)\r\n"
     "        print(\">>> Override: cfg.%s = %r\" % (path, value))",
     "        if path == \"solver.num_cpus\":     # keep MPI domains consistent\r\n"
     "            cfg.solver.num_domains = int(value)\r\n"
     "        print(\">>> Override: cfg.%s = %r\" % (path, value))\r\n"
     "    _apply_material_knobs(cfg, _knobs)           # [calib-knobs-patch]"),
]


# --------------------------------------------------------------------------
# launch_cluster_jobs.py -- aliases + example JOBS
# --------------------------------------------------------------------------

LAUNCH_ALIASES = '''
    # Calibration                                     [calib-knobs-patch]
    # Reachable by plain setattr (no interception needed):
    "psi":            "material.plasticity.dilation_angle",
    "beta":           "material.plasticity.friction_angle",
    "K":              "material.plasticity.flow_stress_ratio",
    "E":              "material.hyperelastic.E",
    "nu":             "material.hyperelastic.nu",
    # Intercepted in run_parameter_study._apply_overrides (consumed at
    # construction, so they are NOT real attributes):
    "sigma_scale":    "material.sigma_scale",
    "sigma_y0":       "material.sigma_y0",
    "soft_drop":      "material.soft_drop",
    "eps_soft":       "material.eps_soft",
    "h_gsell":        "material.h_gsell",
    "tau0":           "material.tau0",
    "alpha":          "material.alpha",
    "mu_cap":         "material.mu_cap",
'''

LAUNCH_JOBS = '''
    # ---- PMMA calibration [calib-knobs-patch] -------------------------------
    # The 38-point factorial runs through the design campaign, not here:
    #   ("design", "glassy_pmma_calib", {"tag": "CAL", "ALE": True, ...})
    # These entries are for the points the design CSV cannot carry, because
    # they are outside the campaign's factor set.
    #
    # BASE -- current families.py values. Needs NO override at all: tau0=4 and
    # alpha=0.2 already are the placeholder in _glassy_pmma_config().
    #("depth", "glassy_pmma", {"tag": "CAL_BASE", "indenter": "rockwell", "ALE": True, "distortion": False, "scratch_time": 0.01, "unload_time": 0.005, "recovery_time": 0.005, "scratch_depth": -0.020}),
    #
    # PROBE -- beta sonde. PMMA is quoted as at least as pressure-sensitive as
    # PC, yet families.py gives it 25 deg against PC's 27.13. Two jobs say
    # whether beta absorbs part of the 1.6x force deficit before the whole
    # deficit is pinned on sigma_y.
    #("depth", "glassy_pmma", {"tag": "CAL_BETA30", "indenter": "rockwell", "ALE": True, "distortion": False, "scratch_time": 0.01, "unload_time": 0.005, "recovery_time": 0.005, "scratch_depth": -0.020, "beta": 30.0}),
    #("depth", "glassy_pmma", {"tag": "CAL_BETA30_S13", "indenter": "rockwell", "ALE": True, "distortion": False, "scratch_time": 0.01, "unload_time": 0.005, "recovery_time": 0.005, "scratch_depth": -0.020, "beta": 30.0, "sigma_scale": 1.3}),
    #
    # CENSORING PROBE -- run this THIRD, before committing to the factorial.
    # Morris already flagged informative censoring in the high-softening +
    # high-friction corner (element distortion, aborted jobs). soft_drop=25
    # with tau0=60 IS that corner.
    #("depth", "glassy_pmma", {"tag": "CAL_CORNER", "indenter": "rockwell", "ALE": True, "distortion": False, "scratch_time": 0.01, "unload_time": 0.005, "recovery_time": 0.005, "scratch_depth": -0.020, "soft_drop": 25.0, "tau0": 60.0, "alpha": 0.07}),
'''

LAUNCH_EDITS = [
    ("    # 3mm Scratch \r\n    \"scratch_length\": \"scratch.scratch_length\",",
     LAUNCH_ALIASES.replace("\n", "\r\n").lstrip("\r\n")
     + "\r\n    # 3mm Scratch \r\n    \"scratch_length\": \"scratch.scratch_length\","),

    ("JOBS = [\r\n",
     "JOBS = [\r\n" + LAUNCH_JOBS.replace("\n", "\r\n").lstrip("\r\n")),
]


FILES = [
    ("base.py", BASE_EDITS, False),
    ("run_parameter_study.py", RPS_EDITS, True),
    ("launch_cluster_jobs.py", LAUNCH_EDITS, True),
]


def apply_to(path, edits, expect_crlf):
    with open(path, "rb") as fh:
        text = fh.read().decode("utf-8")
    if MARK in text:
        print("  %-26s already patched -- skipped" % os.path.basename(path))
        return
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            sys.exit("ANCHOR %s in %s (%d hits):\n  %r"
                     % ("NOT FOUND" if n == 0 else "NOT UNIQUE",
                        os.path.basename(path), n, old[:90]))
        text = text.replace(old, new, 1)
    ast.parse(text)
    lone = text.count("\n") - text.count("\r\n")
    if expect_crlf and lone:
        sys.exit("CRLF check failed in %s: %d bare LF" % (path, lone))
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))
    print("  %-26s patched (%d edits)" % (os.path.basename(path), len(edits)))


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    found = {}
    for name, _, _ in FILES:
        for cand in (os.path.join(root, name),
                     os.path.join(root, "Configuration", name)):
            if os.path.isfile(cand):
                found[name] = cand
                break
        else:
            sys.exit("not found under %s: %s" % (root, name))
    for name, edits, crlf in FILES:
        # base.py line endings are detected, not assumed
        with open(found[name], "rb") as fh:
            head = fh.read()
        crlf = (head.count(b"\r\n") > 0)
        e = edits
        if crlf and name == "base.py":
            e = [(a.replace("\n", "\r\n"), b.replace("\n", "\r\n"))
                 for a, b in edits]
        apply_to(found[name], e, crlf)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())