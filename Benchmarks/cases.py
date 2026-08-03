# -*- coding: utf-8 -*-
"""
Benchmark configurations.


    level 0   single_element_case()      one C3D8R Element, material card vs hand integration of the same law
    level 1a  hertz_case(dt_scale=1)     elastic indentation vs closed form
    level 1b  hertz_case(budget=N)       same case, inertia/mass-scaling sweep
    level 1c  ploughing_case()           scratch at constant mu
    level 2   standard_reference_case()  implicit quasi-static reference

Design rules, all of them load-bearing:
  * benchmarks reuse the PRODUCTION mesh generator, so they validate the mesh
    that actually produces results;
  * the element size is never altered to make a case cheaper -- only the
    far-field volume and the number of increments are;
  * every cost knob is dimensionless (N_a = a/h, N increments, v_n/c_scaled)
    so a conclusion transfers to the scratch.
"""

import numpy as np

from ..AbaqusModel.Configuration.base import (
    Material_Config, LinearElastic_Config, P_Model_Config, VE_Model_Config,
    Damage_Config, Friction_Config, elastic_moduli, natural_dt)
from ..AbaqusModel.Configuration.families import get_family
from ..AbaqusModel.Verification.analytic import contact_radius_rockwell



# Ladders

# Uniform refinement ratio r = 1.4 to get an observable order
MESH_LADDER = [0.020, 0.0143, 0.0102, 0.0073, 0.0052]

# a/R <= 0.22 keeps the Hertz reference itself good to ~1 %.
HERTZ_DEPTHS = [2e-3, 5e-3, 10e-3]

# Increments per 1a run. Sits above the 200-wave-transit quasi-static floor
# (~23k at h=0.010 on the reduced box) at a quarter of the cost of a naive
# 0.05 s ramp. KE/IE in the report is the check that it was enough.
INCREMENT_BUDGET = 60000

# 1b ladder. The governing identity is
#     v_n / c_scaled = depth / (N * dt_nat * c0)
# which depends on N ALONE, not on the mass-scaling factor f. N therefore sets
# the inertia ratio, f sets the added mass, and N is also exactly the cost.
# This span (6.5e-4 ... 6.5e-6) brackets the production scratch's normal
# approach at ~9e-6.
INCREMENT_LADDER = [2000, 6000, 20000, 60000, 200000]

# Mass-scaling ladder as s = dt_target/dt_nat, i.e. f = s^2, swept at fixed
# increment count so every point costs the same and only the added mass moves.
DT_SCALE_LADDER = [1.0, 5.0, 15.0, 30.0, 60.0]
MS_INCREMENTS = 20000

ELEMENT_MODES = ["tension", "compression", "shear"]

QS_TRANSITS = 200.0        # wave transits of the domain for "quasi-static"
BOUNDARY_MARGIN = 10.0     # contact radii from contact to any boundary


# --------------------------------------------------------------------------
# Bench_Config
# --------------------------------------------------------------------------

class Bench_Config(object):
    """Load case description. Kept separate from Scratch_Config: a benchmark
    is an indentation or a material-point test, and overloading Scratch_Config
    would silently change the meaning of the production amplitudes."""

    HERTZ = "hertz"
    SINGLE_ELEMENT = "single_element"
    PLOUGHING = "ploughing"
    EXPLICIT = "explicit"
    STANDARD = "standard"

    def __init__(self, kind=HERTZ, solver_kind=EXPLICIT,
                 depth_max=10e-3, ramp_time=0.05, hold_time=0.0,
                 n_history=400, n_field=40,
                 element_mode="compression", element_strain=0.5,
                 element_time=0.01, label=""):
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
                "bench_element_strain": self.element_strain}


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------

def linear_moduli(material):
    """(E, nu) of the small-strain base elasticity of any family."""
    K, G = elastic_moduli(material)
    E = 9.0 * K * G / (3.0 * K + G)
    nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    return float(E), float(nu)


def strip_to_linear_elastic(cfg):
    """Reduce the constitutive stack to its linear elastic base. The Hertz
    reference exists only for linear elasticity, so anything else in the card
    would make a disagreement uninterpretable."""
    E, nu = linear_moduli(cfg.material)
    cfg.material = Material_Config(
        rho=cfg.material.rho,
        hyperelastic=LinearElastic_Config(E=E, nu=nu),
        viscoelastic=VE_Model_Config(),
        plasticity=P_Model_Config(),
        damage=Damage_Config(),
        friction=Friction_Config(mu=0.0),
        family=str(getattr(cfg.material, "family", "?")) + "_ELASTIC_BENCH")
    return cfg


def set_constant_mu(cfg, mu):
    """Constant Coulomb. Required before any mesh study: a pressure-dependent
    mu makes the apparent friction a function of the contact pressure field,
    which is the least converged output of the model."""
    cfg.material.friction = Friction_Config(mu=float(mu))
    return cfg


def set_mesh(cfg, h):
    """Isotropic fine size. Guards coarse_size_1 > h, because mesh_substrate()
    passes them to seedEdgeByBias as maxSize/minSize and Abaqus aborts when
    minSize >= maxSize."""
    h = float(h)
    cfg.mesh.fine_size_x = cfg.mesh.fine_size_y = cfg.mesh.fine_size_z = h
    if float(cfg.mesh.coarse_size_1) <= h:
        raise ValueError(
            "set_mesh(%g): coarse_size_1=%g must exceed the fine size, else "
            "seedEdgeByBias gets minSize >= maxSize and Abaqus aborts."
            % (h, cfg.mesh.coarse_size_1))
    return cfg


def retarget_dt(cfg, s=None):
    """Recompute solver.target_time_increment = s * dt_nat for the CURRENT
    material and mesh. families.py evaluates it once at build time with the
    default mesh, so it must be redone whenever either changes or the
    effective mass-scaling factor drifts."""
    L = min(cfg.mesh.fine_size_x, cfg.mesh.fine_size_y, cfg.mesh.fine_size_z)
    dt_nat = float(natural_dt(cfg.material, L))
    if s is None:
        cur = float(getattr(cfg.solver, "target_time_increment", 0.0) or 0.0)
        s = (cur / dt_nat) if (dt_nat > 0 and cur > 0) else 0.0
    cfg.solver.target_time_increment = float(s) * dt_nat if s > 0 else 0.0
    return cfg


def _effective_mass_factor(cfg):
    """f actually applied: (dt_target/dt_nat)^2 under variable scaling, else
    solver.mass_scale. Every family sets target_time_increment > 0, so reading
    mass_scale alone reports the wrong number."""
    L = min(cfg.mesh.fine_size_x, cfg.mesh.fine_size_y, cfg.mesh.fine_size_z)
    dt_nat = float(natural_dt(cfg.material, L))
    target = float(getattr(cfg.solver, "target_time_increment", 0.0) or 0.0)
    if target > 0.0:
        return max((target / dt_nat) ** 2, 1.0), dt_nat
    return max(float(getattr(cfg.solver, "mass_scale", 1.0) or 1.0), 1.0), dt_nat


def shrink_substrate_for_indentation(cfg, depth_max, margin=BOUNDARY_MARGIN,
                                     verbose=True):
    """Size the box for an indentation instead of a scratch.

    The production substrate is 3 mm long in z to hold the 2 mm scratch
    travel; an indentation has none, so that volume is pure cost. Hertz is a
    half-space solution, so the only requirement is that every boundary sits
    >= `margin` contact radii away. The ELEMENT SIZE is untouched -- N_a = a/h
    is what transfers to the scratch.

    The radius comes from contact_radius_rockwell, not sqrt(R*depth): past the
    sphere/cone transition h* = R(1-sin(alpha)) = 26.8 um the contact is on the
    conical flank and is much wider (0.123 mm at 40 um vs 0.089 mm), so the
    sphere formula would silently under-size the level-2 box.
    """
    a = float(contact_radius_rockwell(abs(float(depth_max)),
                                      float(cfg.indenter.tip_radius),
                                      float(cfg.indenter.cone_angle)))
    sub = cfg.substrate
    half = margin * a
    fine_half = max(3.0 * a, 4.0 * float(cfg.mesh.fine_size_x))

    sub.xs1 = sub.ys1 = sub.zs1 = 0.0
    sub.xs2 = max(half, 4.0 * fine_half)
    sub.ys2 = max(half, 4.0 * fine_half)
    sub.zs2 = max(2.0 * half, 6.0 * fine_half)
    sub.dpo_x = min(fine_half, 0.45 * sub.xs2)
    sub.dpo_y = min(fine_half, 0.45 * sub.ys2)
    sub.dpo_z = max((sub.zs2 - 2.0 * fine_half) / 2.0, 0.15 * sub.zs2)

    if verbose:
        fz = (sub.zs2 - sub.dpo_z) - (sub.zs1 + sub.dpo_z)
        h = float(cfg.mesh.fine_size_x)
        n = (sub.dpo_x / h) * (sub.dpo_y / h) * (fz / h)
        print(">>> domain: a=%.4f mm, box %.3f x %.3f x %.3f mm "
              "(%.0fa, %.0fa, %.0fa), ~%d fine elements"
              % (a, sub.xs2, sub.ys2, sub.zs2, sub.xs2 / a, sub.ys2 / a,
                 (sub.zs2 / 2.0) / a, int(n)))
        if min(sub.xs2, sub.ys2, sub.zs2 / 2.0) < 8.0 * a:
            print(">>> WARNING: a boundary is inside 8 contact radii; the "
                  "half-space assumption behind Hertz is strained.")
    return cfg


# --------------------------------------------------------------------------
# Level 1a / 1b -- Hertz elastic indentation
# --------------------------------------------------------------------------

def hertz_case(family="glassy_pc", h_mesh=0.0102, depth_max=10e-3,
               ramp_time=None, dt_scale=None, mass_scale=None,
               increment_budget=None, quasi_static_floor=True,
               solver_kind=Bench_Config.EXPLICIT, full_domain=False,
               num_cpus=None):
    """Elastic indentation on the production substrate and mesh generator.

    dt_scale = s targets dt = s*dt_nat, i.e. f = s^2. s=1 gives the unscaled
    reference (1a); the family's production s reproduces the production
    protocol (1b).

    increment_budget derives ramp_time = N * dt_eff. With f = 1 shortening the
    ramp keeps the answer EXACT (no added mass, unlike mass scaling) and only
    raises the kinetic energy, which the report tracks as KE/IE.

    quasi_static_floor=False lets 1b walk deliberately into the dynamic regime;
    clamping those runs would erase the effect being measured.
    """
    cfg = get_family(family).build_config()

    # Production s, captured before the material is stripped.
    f0, dt0 = _effective_mass_factor(cfg)
    s_production = np.sqrt(f0) if f0 > 1.0 else None

    strip_to_linear_elastic(cfg)
    set_mesh(cfg, h_mesh)
    if not full_domain:
        shrink_substrate_for_indentation(cfg, depth_max)

    if mass_scale is not None:
        cfg.solver.mass_scale = float(mass_scale)
        cfg.solver.target_time_increment = 0.0
    else:
        retarget_dt(cfg, s=(dt_scale if dt_scale is not None else s_production))

    if increment_budget is not None:
        f, dt_nat = _effective_mass_factor(cfg)
        dt_eff = dt_nat * np.sqrt(f)
        ramp_time = float(increment_budget) * dt_eff
        E_eq, _ = linear_moduli(cfg.material)
        c_scaled = np.sqrt(E_eq / cfg.material.rho) / np.sqrt(f)
        L = max(cfg.substrate.xs2, cfg.substrate.ys2, cfg.substrate.zs2)
        floor = QS_TRANSITS * L / c_scaled
        if quasi_static_floor and ramp_time < floor:
            ramp_time = floor
            print(">>> quasi-static floor applied: ramp -> %.3e s" % ramp_time)
        print(">>> hertz: f=%.1f, ramp=%.3e s (~%d incr), v_n/c=%.2e, "
              "%d transits"
              % (f, ramp_time, int(round(ramp_time / dt_eff)),
                 (depth_max / ramp_time) / c_scaled,
                 int(ramp_time / (L / c_scaled))))
    elif ramp_time is None:
        ramp_time = 0.05

    cfg.solver.use_ALE = False          # a reference case advects nothing
    if num_cpus:
        cfg.solver.num_cpus = cfg.solver.num_domains = int(num_cpus)

    bench = Bench_Config(
        kind=Bench_Config.HERTZ, solver_kind=solver_kind,
        depth_max=depth_max, ramp_time=ramp_time,
        label="Hertz_%s_h%g_d%g" % (family, h_mesh, depth_max))
    cfg.job_name = "Bench" + bench.label.replace(".", "p")
    return cfg, bench


def hertz_reference_table(family="glassy_pc", depths=None, R=0.2,
                          h_meshes=MESH_LADDER):
    """A-priori table deciding whether a benchmark point is worth running:
    exact P, a, p0, U per depth, N_a per candidate mesh, plus the a/R gate on
    the Hertz solution itself."""
    from ..AbaqusModel.Verification import analytic as an
    cfg = get_family(family).build_config()
    E, nu = linear_moduli(cfg.material)
    rows = []
    for d in (depths or HERTZ_DEPTHS):
        a = float(an.hertz_contact_radius(R, d))
        rows.append({
            "depth_mm": float(d),
            "a_mm": a,
            "P_half_N": float(an.hertz_force(E, nu, R, d, half_model=True)),
            "p0_MPa": float(an.hertz_p0(E, nu, R, d)),
            "U_half_Nmm": float(an.hertz_strain_energy(E, nu, R, d,
                                                       half_model=True)),
            "validity": an.hertz_validity(R, d),
            "N_a": dict((str(h), a / float(h)) for h in h_meshes)})
    return {"family": family, "E": E, "nu": nu,
            "E_star": float(an.estar(E, nu)), "R": R, "rows": rows}


# --------------------------------------------------------------------------
# Level 0 -- single element
# --------------------------------------------------------------------------

def single_element_case(family="glassy_pc", mode="compression",
                        strain=0.5, element_time=0.01, num_cpus=1):
    """One C3D8R with the family's exact material card.

    compression  reproduces the tabulated hardening curve exactly
                 (*DRUCKER PRAGER HARDENING defaults to TYPE=COMPRESSION)
    tension      discriminates DP from J2 via sigma_t/sigma_c = f(beta, K)
    shear        probes the third invariant, i.e. K
    relaxation   Prony series under a constant-strain hold

    No mass scaling: the element is 1 mm^3, and added mass would sit between
    the card and the stress it is being checked against.
    """
    cfg = get_family(family).build_config()
    cfg.solver.use_ALE = False
    cfg.solver.mass_scale = 1.0
    cfg.solver.target_time_increment = 0.0
    cfg.solver.num_cpus = cfg.solver.num_domains = int(num_cpus)
    bench = Bench_Config(kind=Bench_Config.SINGLE_ELEMENT,
                         element_mode=mode, element_strain=strain,
                         element_time=element_time,
                         label="Elem_%s_%s" % (family, mode))
    cfg.job_name = "Bench" + bench.label
    return cfg, bench


# --------------------------------------------------------------------------
# Level 1c -- ploughing anchor
# --------------------------------------------------------------------------

def ploughing_case(family="semicrystalline_j2", h_mesh=0.010,
                   scratch_length=0.8, mu=0.0, scratch_time=None):
    """Production scratch at constant mu.

    mu = 0 isolates the geometric ploughing term, whose rigid-plastic value
    for the Rockwell C flank is (2/pi)cot(60 deg) = 0.368; SCOF(mu) - SCOF(0)
    then isolates the interfacial term.

    Cost: the track is shortened from 2 mm to 0.8 mm and scratch_time scaled
    with it, so the sliding VELOCITY stays at its production value. 0.8 mm is
    still ~3 contact diameters of steady sliding, and the report's plateau
    window already discards the first 60 % of the step.
    """
    cfg = get_family(family).build_config()
    set_constant_mu(cfg, mu)
    set_mesh(cfg, h_mesh)

    L_prod, T_prod = float(cfg.scratch.scratch_length), float(cfg.scratch.scratch_time)
    L_new = float(scratch_length)
    T_new = (float(scratch_time) if scratch_time is not None
             else (T_prod * L_new / L_prod if L_prod > 0 else T_prod))
    cfg.scratch.scratch_length, cfg.scratch.scratch_time = L_new, T_new

    a = float(contact_radius_rockwell(abs(float(cfg.scratch.scratch_depth)),
                                      float(cfg.indenter.tip_radius),
                                      float(cfg.indenter.cone_angle)))
    sub = cfg.substrate
    sub.zs1 = 0.0
    sub.zs2 = 2.0 * sub.dpo_z + L_new + max(4.0 * a, 6.0 * cfg.mesh.fine_size_z)
    print(">>> plough: %.3f mm in %.4g s (v=%.1f mm/s, production %.1f), "
          "z=%.3f mm, %.1f contact diameters"
          % (L_new, T_new, L_new / T_new, L_prod / T_prod, sub.zs2,
             L_new / (2.0 * a)))

    retarget_dt(cfg)
    cfg.solver.use_ALE = False
    bench = Bench_Config(kind=Bench_Config.PLOUGHING,
                         label="Plough_%s_mu%g_h%g" % (family, mu, h_mesh))
    cfg.job_name = "Bench" + bench.label.replace(".", "p")
    return cfg, bench


# --------------------------------------------------------------------------
# Level 2 -- Abaqus/Standard quasi-static reference
# --------------------------------------------------------------------------

def standard_reference_case(family="glassy_pc", h_mesh=0.020,
                            depth_max=40e-3, mu=0.0):
    """Indentation to production depth with the FULL card, in Standard: no
    mass scaling, no inertia, no bulk viscosity, no time-step choice. The gap
    against the Explicit RF2 at the same depth IS the quasi-staticity error.

    Scratching is not attempted in Standard (sliding contact convergence);
    indentation suffices for the force comparison.
    """
    cfg = get_family(family).build_config()
    set_constant_mu(cfg, mu)
    set_mesh(cfg, h_mesh)
    shrink_substrate_for_indentation(cfg, depth_max)
    cfg.solver.use_ALE = False
    cfg.solver.mass_scale = 1.0
    cfg.solver.target_time_increment = 0.0
    bench = Bench_Config(kind=Bench_Config.HERTZ,
                         solver_kind=Bench_Config.STANDARD,
                         depth_max=depth_max, ramp_time=1.0,
                         label="Std_%s_h%g_d%g" % (family, h_mesh, depth_max))
    cfg.job_name = "Bench" + bench.label.replace(".", "p")
    return cfg, bench