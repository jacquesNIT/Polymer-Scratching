# -*- coding: utf-8 -*-
"""
Named collections of benchmark cases.

ABAQUS-FREE. Each suite returns [(stem, factory), ...] where factory() yields
(cfg, bench). Separated from run.py so the suite matrix can be inspected and
costed without an Abaqus session.
"""

from .cases import (
    hertz_case, single_element_case, ploughing_case, standard_reference_case,
    MESH_LADDER, HERTZ_DEPTHS, INCREMENT_BUDGET, INCREMENT_LADDER,
    DT_SCALE_LADDER, MS_INCREMENTS, ELEMENT_MODES)
from ..AbaqusModel.Configuration.families import get_family

# Shear must be pushed past the shear yield tau_y = d/sqrt(3), reached at
# gamma_y = tau_y/G ~ 0.04-0.06 for these polymers. The old 0.05 sat on that
# threshold and often stayed elastic (PEEQ = 0), measuring nothing.
SHEAR_STRAIN = 0.20
AXIAL_STRAIN = 0.50

REF_MESH = 0.0102          # reference mesh for the 1b / ms sweeps
REF_DEPTH = 10e-3          # a/R = 0.22, Hertz reference good to ~1 %


def _tag(x):
    return ("%g" % x).replace(".", "p")


# --------------------------------------------------------------------------
# Level 0
# --------------------------------------------------------------------------

def suite_level0(family, **kw):
    """Material-point tests: E, yield, T/C ratio, shear yield, Prony.
    Relaxation is appended only when the card carries a Prony series, so one
    'level0' run fully exercises whatever the family is."""
    cases = [("L0_%s_%s" % (family, m),
              lambda f=family, mm=m, s=(SHEAR_STRAIN if m == "shear"
                                        else AXIAL_STRAIN):
              single_element_case(f, mode=mm, strain=s))
             for m in ELEMENT_MODES]
    try:
        ve = get_family(family).build_config().material.viscoelastic
        if getattr(ve, "MODEL", "none") not in ("none", None) \
                and getattr(ve, "prony_table", None):
            cases.append(("L0_%s_relaxation" % family,
                          lambda f=family: single_element_case(
                              f, mode="relaxation", strain=0.05,
                              element_time=1e-4)))
    except Exception as exc:
        print(">>> viscoelasticity check failed for %s (%s); relaxation "
              "case skipped." % (family, exc))
    return cases


def suite_level0_relax(family, **kw):
    """Prony relaxation alone."""
    return [("L0_%s_relaxation" % family,
             lambda f=family: single_element_case(
                 f, mode="relaxation", strain=0.05, element_time=1e-4))]


# --------------------------------------------------------------------------
# Level 1a -- spatial convergence
# --------------------------------------------------------------------------

def suite_hertz(family, depth=REF_DEPTH, budget=None, **kw):
    """Mesh ladder at fixed depth, unscaled (f = 1). Yields the N_a needed for
    a given accuracy and the observed order of convergence."""
    b = INCREMENT_BUDGET if budget is None else budget
    return [("Hertz_%s_h%s" % (family, _tag(h)),
             lambda f=family, hh=h, d=depth: hertz_case(
                 f, h_mesh=hh, depth_max=d, dt_scale=1.0, increment_budget=b))
            for h in MESH_LADDER]


def suite_hertz_depth(family, h_mesh=REF_MESH, **kw):
    """Depth ladder at fixed mesh: sensitivity to a/R, i.e. to the validity of
    the Hertz reference itself."""
    return [("HertzD_%s_d%s" % (family, _tag(d)),
             lambda f=family, dd=d, hh=h_mesh: hertz_case(
                 f, h_mesh=hh, depth_max=dd, dt_scale=1.0,
                 increment_budget=INCREMENT_BUDGET))
            for d in HERTZ_DEPTHS]


# --------------------------------------------------------------------------
# Level 1b -- quasi-staticity and mass scaling
# --------------------------------------------------------------------------

def _increment_sweep(family, h_mesh, depth, dt_scale, prefix):
    return [("%s_%s_N%d" % (prefix, family, N),
             lambda f=family, NN=N, hh=h_mesh, d=depth, s=dt_scale: hertz_case(
                 f, h_mesh=hh, depth_max=d, dt_scale=s,
                 increment_budget=NN, quasi_static_floor=False))
            for N in INCREMENT_LADDER]


def suite_hertz_time(family, h_mesh=REF_MESH, depth=REF_DEPTH, **kw):
    """Series 1: loading rate swept at CONSTANT mass scaling. The material is
    linear elastic, so no rate dependence is physically possible and every
    drift is numerical."""
    return _increment_sweep(family, h_mesh, depth, 15.0, "HertzN")


def suite_hertz_time_production(family, h_mesh=REF_MESH, depth=REF_DEPTH, **kw):
    """Series 2: same sweep with the mass scaling the production config
    computes. Comparing the two series separates 'depends on rate' from
    'depends on f, which happens to move with rate'."""
    return _increment_sweep(family, h_mesh, depth, None, "HertzNP")


def suite_hertz_ms(family, h_mesh=REF_MESH, depth=REF_DEPTH, **kw):
    """Mass-scaling ladder at fixed increment count: every point costs the
    same, so the only variable is the added mass. Gives a justified (N, f)
    production pair instead of one chosen by feel."""
    return [("HertzMS_%s_s%s" % (family, _tag(s)),
             lambda f=family, ss=s, hh=h_mesh, d=depth: hertz_case(
                 f, h_mesh=hh, depth_max=d, dt_scale=ss,
                 increment_budget=MS_INCREMENTS, quasi_static_floor=False))
            for s in DT_SCALE_LADDER]


# --------------------------------------------------------------------------
# Levels 1c and 2
# --------------------------------------------------------------------------

def suite_plough(family, **kw):
    """mu = 0 anchor plus one constant-mu point, on the real scratch.
    Briscoe must stay off: with a pressure-dependent mu the apparent friction
    is a function of the contact pressure field."""
    return [("Plough_%s_mu%s_h%s" % (family, _tag(mu), _tag(h)),
             lambda f=family, m=mu, hh=h: ploughing_case(f, h_mesh=hh, mu=m))
            for mu in (0.0, 0.3) for h in (0.020, 0.010)]


def suite_standard(family, **kw):
    """Quasi-static implicit reference. Start at the coarse mesh: the implicit
    contact construction is the least exercised part of the suite."""
    return [("Std_%s_h%s" % (family, _tag(h)),
             lambda f=family, hh=h: standard_reference_case(
                 f, h_mesh=hh, depth_max=40e-3, mu=0.0))
            for h in (0.020, 0.010)]


# --------------------------------------------------------------------------
# Regression baseline
# --------------------------------------------------------------------------

def suite_baseline(family, **kw):
    """Cheapest case from every level, frozen as the non-regression set. Run
    before and after any patch campaign."""
    cases = suite_level0(family)
    cases += [
        ("BL_Hertz_%s" % family,
         lambda f=family: hertz_case(f, h_mesh=REF_MESH, depth_max=REF_DEPTH,
                                     dt_scale=1.0, increment_budget=20000)),
        ("BL_HertzProd_%s" % family,
         lambda f=family: hertz_case(f, h_mesh=REF_MESH, depth_max=REF_DEPTH,
                                     dt_scale=None, increment_budget=20000)),
        ("BL_Plough_%s" % family,
         lambda f=family: ploughing_case(f, h_mesh=0.020, scratch_length=0.4,
                                         mu=0.0)),
    ]
    return cases


SUITES = {
    "level0": suite_level0,
    "level0_relax": suite_level0_relax,
    "hertz": suite_hertz,
    "hertz_depth": suite_hertz_depth,
    "hertz_time": suite_hertz_time,
    "hertz_time_prod": suite_hertz_time_production,
    "hertz_ms": suite_hertz_ms,
    "plough": suite_plough,
    "standard": suite_standard,
    "baseline": suite_baseline,
}