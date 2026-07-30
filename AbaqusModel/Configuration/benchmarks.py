# -*- coding: utf-8 -*-
"""
Benchmark configurations for the V&V pyramid.

ABAQUS-FREE (numpy only) -- same constraint as base.py / families.py, because
this module is imported by the Abaqus kernel AND by the CPython analysis
scripts.

    Level 0  single_element_case()   one C3D8R, uniaxial T / C / shear /
                                     relaxation. Truth = material_point.py.
    Level 1a hertz_case()            elastic indentation, Abaqus/Standard.
                                     Truth = analytic.hertz_*.
    Level 1b hertz_case(explicit)    the SAME model in Explicit with the
                                     production mass-scaling protocol.
                                     Truth = the same analytic solution.
    Level 1c ploughing_case()        full scratch, mu = 0, rigid-plastic
                                     ploughing anchor.
    Level 2  standard_reference_case() indentation with FULL plasticity in
                                     Abaqus/Standard: the true quasi-static
                                     reference for RF2 at depth.

Design rule: every benchmark reuses the PRODUCTION geometry, partitions and
mesh generator. A benchmark run on a bespoke mesh would validate the bespoke
mesh, not the model that actually produces the results.
"""

import numpy as np

from .base import (Simulation_Config, Material_Config, LinearElastic_Config,
                   P_Model_Config, VE_Model_Config, Damage_Config,
                   Friction_Config, Solver_Config, Scratch_Config,
                   Mesh_Config, Output_Config, elastic_moduli, natural_dt)
from .families import get_family


# --------------------------------------------------------------------------
# Bench_Config: what the benchmark builders need and Scratch_Config cannot say
# --------------------------------------------------------------------------

class Bench_Config(object):
    """
    Description of a benchmark load case. Deliberately NOT folded into
    Scratch_Config: a benchmark is an indentation or a material-point test,
    not a scratch, and overloading Scratch_Config would silently change the
    meaning of the production amplitudes.
    """

    HERTZ = "hertz"
    SINGLE_ELEMENT = "single_element"
    PLOUGHING = "ploughing"

    EXPLICIT = "explicit"
    STANDARD = "standard"

    def __init__(self,
                 kind=HERTZ,
                 solver_kind=EXPLICIT,
                 depth_max=10e-3,          # [mm] max penetration of the ramp
                 ramp_time=0.05,           # [s]  loading time (Explicit only)
                 hold_time=0.0,            # [s]  optional hold at max depth
                 n_history=400,            # history samples over the ramp
                 n_field=40,               # field frames over the ramp
                 element_mode="compression",   # single-element: tension|compression|shear|relaxation
                 element_strain=0.5,       # single-element: max |log strain|
                 element_time=0.01,        # [s] single-element ramp time
                 smoothing=None,           # amplitude smoothing (None = solver default)
                 label=""):
        self.kind = kind
        self.solver_kind = solver_kind
        self.depth_max = float(depth_max)
        self.ramp_time = float(ramp_time)
        self.hold_time = float(hold_time)
        self.n_history = int(n_history)
        self.n_field = int(n_field)
        self.element_mode = element_mode
        self.element_strain = float(element_strain)
        self.element_time = float(element_time)
        self.smoothing = smoothing
        self.label = label

    @property
    def total_time(self):
        return self.ramp_time + self.hold_time

    def to_dict(self):
        return {"bench_kind": self.kind, "bench_solver": self.solver_kind,
                "bench_depth_max": self.depth_max,
                "bench_ramp_time": self.ramp_time,
                "bench_hold_time": self.hold_time,
                "bench_element_mode": self.element_mode,
                "bench_element_strain": self.element_strain,
                "bench_smoothing": (-1.0 if self.smoothing is None
                                    else float(self.smoothing))}


# --------------------------------------------------------------------------
# Material helpers
# --------------------------------------------------------------------------

def linear_moduli(material):
    """(E, nu) of the small-strain base elasticity of ANY family, via
    base.elastic_moduli (which already dispatches on hyperelastic.MODEL)."""
    K, G = elastic_moduli(material)
    E = 9.0 * K * G / (3.0 * K + G)
    nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    return float(E), float(nu)


def strip_to_linear_elastic(cfg):
    """
    Replace the constitutive stack by the equivalent LINEAR ELASTIC base:
    no plasticity, no viscoelasticity, no damage.

    This is what makes the Hertz benchmark a benchmark: the reference is a
    closed form that exists only for linear elasticity, so anything else in
    the card would make a disagreement uninterpretable.
    """
    E, nu = linear_moduli(cfg.material)
    cfg.material = Material_Config(
        rho=cfg.material.rho,
        hyperelastic=LinearElastic_Config(E=E, nu=nu),
        viscoelastic=VE_Model_Config(),
        plasticity=P_Model_Config(),
        damage=Damage_Config(),
        friction=Friction_Config(mu=0.0),
        family=str(getattr(cfg.material, "family", "?")) + "_ELASTIC_BENCH",
    )
    return cfg


def set_frictionless(cfg):
    """mu = 0: the pure-ploughing anchor and the Hertz prerequisite."""
    cfg.material.friction = Friction_Config(mu=0.0)
    return cfg


def set_constant_mu(cfg, mu):
    """Constant Coulomb -- required before any mesh study (see README)."""
    cfg.material.friction = Friction_Config(mu=float(mu))
    return cfg


def retarget_dt(cfg, s=None):
    """
    Recompute solver.target_time_increment = s * natural_dt(material, L_min)
    for the CURRENT material and CURRENT mesh.

    Both are needed: strip_to_linear_elastic() changes the material (hence
    dt_nat), and any mesh change changes L_min. families.py evaluates this
    once, at build time, with the DEFAULT mesh -- so it must be recomputed
    whenever either is touched, or the effective mass-scaling factor drifts.
    s=None keeps the ratio the config already carries.
    """
    L_min = min(float(cfg.mesh.fine_size_x), float(cfg.mesh.fine_size_y),
                float(cfg.mesh.fine_size_z))
    dt_nat = float(natural_dt(cfg.material, L_min))
    if s is None:
        s = float(getattr(cfg.solver, "target_time_increment", 0.0) or 0.0)
        s = (s / dt_nat) if dt_nat > 0 and s > 0 else 0.0
    cfg.solver.target_time_increment = float(s) * dt_nat if s > 0 else 0.0
    return cfg


def set_mesh(cfg, h):
    """Isotropic fine mesh size, with the coarse bounds kept consistent.

    coarse_size_1 acts as the maxSize of the biased transitions in
    substrate.mesh_substrate(); when the fine size approaches it, seedEdgeByBias
    is asked for maxSize <= minSize and Abaqus aborts (the seedEdgeByBias crash
    seen previously). Guard it here rather than discovering it at build time.
    """
    h = float(h)
    cfg.mesh.fine_size_x = h
    cfg.mesh.fine_size_y = h
    cfg.mesh.fine_size_z = h
    c1 = float(cfg.mesh.coarse_size_1)
    if c1 <= h:
        raise ValueError(
            "set_mesh(%g): coarse_size_1=%g must be strictly larger than the "
            "fine size, otherwise seedEdgeByBias gets maxSize <= minSize and "
            "Abaqus aborts in mesh_substrate()." % (h, c1))
    return cfg


# --------------------------------------------------------------------------
# Level 1a / 1b: Hertz elastic indentation
# --------------------------------------------------------------------------

def hertz_case(family="glassy_pc", h_mesh=0.010, depth_max=10e-3,
               ramp_time=0.05, dt_scale=None, mass_scale=None,
               solver_kind=Bench_Config.EXPLICIT, smoothing=None,
               num_cpus=None):
    """
    Elastic indentation on the PRODUCTION substrate and mesh generator.

    depth_max defaults to 10 um:
      * a = sqrt(R*h) = 44.7 um  ->  a/R = 0.22, so the Hertz reference is
        itself good to ~1 % (see analytic.hertz_validity);
      * a single monotonic ramp then sweeps N_a = a/h_mesh continuously from
        0 to 4.5 (at h_mesh = 10 um), which is the plage the 40 um scratch
        actually lives in (a = 123 um -> N_a = 6.2 at h = 20 um).

    dt_scale = s sets variable mass scaling to target dt = s * dt_nat, i.e.
    an a-priori scaling factor f = s^2. Pass the family's production s to run
    benchmark 1b under the exact production protocol; pass s = 1 (or
    mass_scale = 1) for the near-unscaled reference of benchmark 1a.
    """
    cfg = get_family(family).build_config()
    s_production = None
    L0 = min(cfg.mesh.fine_size_x, cfg.mesh.fine_size_y, cfg.mesh.fine_size_z)
    dt0 = float(natural_dt(cfg.material, L0))
    t0 = float(getattr(cfg.solver, "target_time_increment", 0.0) or 0.0)
    if dt0 > 0 and t0 > 0:
        s_production = t0 / dt0

    strip_to_linear_elastic(cfg)
    set_mesh(cfg, h_mesh)

    if mass_scale is not None:
        cfg.solver.mass_scale = float(mass_scale)
        cfg.solver.target_time_increment = 0.0
    else:
        retarget_dt(cfg, s=(dt_scale if dt_scale is not None else s_production))

    cfg.solver.use_ALE = False          # ALE has no place in a reference case
    if num_cpus:
        cfg.solver.num_cpus = int(num_cpus)
        cfg.solver.num_domains = int(num_cpus)

    bench = Bench_Config(kind=Bench_Config.HERTZ, solver_kind=solver_kind,
                         depth_max=depth_max, ramp_time=ramp_time,
                         smoothing=smoothing,
                         label="Hertz_%s_h%g_d%g_T%g" % (family, h_mesh,
                                                         depth_max, ramp_time))
    cfg.job_name = "Bench" + bench.label.replace(".", "p")
    return cfg, bench


def hertz_reference_table(family="glassy_pc", depths=None, R=0.2,
                          h_meshes=(0.020, 0.015, 0.010, 0.0075, 0.005)):
    """
    The a-priori table that decides whether a benchmark point is worth
    running: exact P, a, p0, U at each depth, plus N_a on each candidate
    mesh, plus the Hertz-validity gate.
    """
    from ..Verification import analytic as an
    cfg = get_family(family).build_config()
    E, nu = linear_moduli(cfg.material)
    if depths is None:
        depths = (2e-3, 5e-3, 10e-3, 15e-3, 20e-3)
    rows = []
    for d in depths:
        a = float(an.hertz_contact_radius(R, d))
        rows.append({
            "depth_mm": float(d),
            "a_mm": a,
            "P_full_N": float(an.hertz_force(E, nu, R, d)),
            "P_half_N": float(an.hertz_force(E, nu, R, d, half_model=True)),
            "p0_MPa": float(an.hertz_p0(E, nu, R, d)),
            "pmean_MPa": float(an.hertz_pmean(E, nu, R, d)),
            "U_half_Nmm": float(an.hertz_strain_energy(E, nu, R, d, half_model=True)),
            "validity": an.hertz_validity(R, d),
            "N_a": dict((str(h), a / float(h)) for h in h_meshes),
        })
    return {"family": family, "E": E, "nu": nu, "E_star": float(an.estar(E, nu)),
            "R": R, "rows": rows}


# --------------------------------------------------------------------------
# Level 0: single element
# --------------------------------------------------------------------------

def single_element_case(family="glassy_pc", mode="compression",
                        strain=0.5, element_time=0.01, num_cpus=1):
    """
    One C3D8R with the family's EXACT material card.

    mode:
      "compression" the tabulated hardening curve must be reproduced exactly
                    (*DRUCKER PRAGER HARDENING defaults to TYPE=COMPRESSION)
      "tension"     discriminates DP from J2: J2 gives the same curve as in
                    compression, DP gives sigma_t/sigma_c = f(beta, K)
      "shear"       third invariant: probes K (flow_stress_ratio)
      "relaxation"  Prony series, constant strain hold
    """
    cfg = get_family(family).build_config()
    cfg.solver.use_ALE = False
    cfg.solver.mass_scale = 1.0
    cfg.solver.target_time_increment = 0.0     # no mass scaling in a material-point test
    cfg.solver.num_cpus = int(num_cpus)
    cfg.solver.num_domains = int(num_cpus)
    bench = Bench_Config(kind=Bench_Config.SINGLE_ELEMENT,
                         solver_kind=Bench_Config.EXPLICIT,
                         element_mode=mode, element_strain=strain,
                         element_time=element_time,
                         label="Elem_%s_%s" % (family, mode))
    cfg.job_name = "Bench" + bench.label
    return cfg, bench


# --------------------------------------------------------------------------
# Level 1c: ploughing anchor (mu = 0 full scratch)
# --------------------------------------------------------------------------

def ploughing_case(family="semicrystalline_j2", h_mesh=0.010,
                   scratch_time=0.05, mu=0.0):
    """
    Full production scratch with a CONSTANT mu (0.0 by default).

    mu = 0 isolates the geometric ploughing term, whose rigid-plastic value
    for the Rockwell C flank is (2/pi) cot(60 deg) = 0.368. The difference
    SCOF(mu) - SCOF(0) then isolates the interfacial term -- which is the only
    clean way to separate ploughing from adhesion, and the prerequisite to
    interpreting anything Briscoe does.
    """
    cfg = get_family(family).build_config()
    set_constant_mu(cfg, mu)
    set_mesh(cfg, h_mesh)
    retarget_dt(cfg)
    cfg.solver.use_ALE = False
    cfg.scratch.scratch_time = float(scratch_time)
    bench = Bench_Config(kind=Bench_Config.PLOUGHING,
                         label="Plough_%s_mu%g_h%g" % (family, mu, h_mesh))
    cfg.job_name = "Bench" + bench.label.replace(".", "p")
    return cfg, bench


# --------------------------------------------------------------------------
# Level 2: Abaqus/Standard quasi-static reference (full plasticity)
# --------------------------------------------------------------------------

def standard_reference_case(family="glassy_pc", h_mesh=0.010,
                            depth_max=40e-3, mu=0.0):
    """
    Indentation to the production depth with the FULL constitutive card, in
    Abaqus/Standard. No mass scaling, no inertia, no bulk viscosity, no
    time-step choice: the quasi-static answer by construction.

    RF2(depth_max) from this run minus RF2 from the Explicit scratch at the
    same depth IS the quasi-staticity error -- measured instead of estimated.
    Scratching is not attempted in Standard (sliding contact convergence);
    indentation is enough for the force comparison.
    """
    cfg = get_family(family).build_config()
    set_constant_mu(cfg, mu)
    set_mesh(cfg, h_mesh)
    cfg.solver.use_ALE = False
    cfg.solver.mass_scale = 1.0
    cfg.solver.target_time_increment = 0.0
    bench = Bench_Config(kind=Bench_Config.HERTZ,
                         solver_kind=Bench_Config.STANDARD,
                         depth_max=depth_max, ramp_time=1.0,
                         label="Std_%s_h%g_d%g" % (family, h_mesh, depth_max))
    cfg.job_name = "Bench" + bench.label.replace(".", "p")
    return cfg, bench


# --------------------------------------------------------------------------
# Default ladders
# --------------------------------------------------------------------------

# Mesh ladder for the Hertz study: uniform refinement ratio r = 1.4 so the
# observed order is resolvable (the production ladder 0.020/0.015/0.010/0.0075
# has non-uniform ratios 1.33/1.50/1.33, which weakens every triplet).
BENCH_MESH_LADDER = [0.020, 0.0143, 0.0102, 0.0073, 0.0052]

# Depths chosen so a/R stays <= 0.22 (Hertz good to ~1 %).
BENCH_HERTZ_DEPTHS = [2e-3, 5e-3, 10e-3]

# Scratch-time ladder for the quasi-staticity study (1b). Deliberately wider
# than the production one and ANCHORED ABOVE it: the production 0.05 s is not
# demonstrated to be quasi-static, so the ladder must bracket it from above.
BENCH_TIME_LADDER = [0.2, 0.1, 0.05, 0.025, 0.01]

# Mass-scaling ladder, expressed as s = dt_target / dt_nat, i.e. f = s^2.
BENCH_DT_SCALE_LADDER = [1.0, 5.0, 15.0, 30.0, 60.0]

# Level-0 matrix.
BENCH_ELEMENT_MODES = ["tension", "compression", "shear"]