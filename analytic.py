# -*- coding: utf-8 -*-
"""
Closed-form references for the numerical settings of the scratch model.

ABAQUS-FREE (numpy only): imported both by the Abaqus kernel (extractor.py
writes the derived quantities into the results CSV header) and by the plain
CPython analysis scripts.

Every function answers one question that no simulation output answers on its
own: what mass factor is ACTUALLY applied, how wide is the contact, how many
elements resolve it, and how much of the scratch step is eaten by amplitude
smoothing. Two CSVs written without these numbers are not comparable.

Location note: this module lives at the ScratchSimulation package root,
alongside launch_cluster_jobs.py and results_values.py. natural_dt is
imported from AbaqusModel/Configuration/base.py through a tolerant chain so
the module works whether it is reached as a package member, as a top-level
module, or by direct path execution.
"""

import numpy as np


# --------------------------------------------------------------------------
# natural_dt import: tolerant to the execution environment
# --------------------------------------------------------------------------

def _import_natural_dt():
    """Resolve base.natural_dt across the kernel / CPython / script paths."""
    try:
        from ScratchSimulation.AbaqusModel.Configuration.base import natural_dt
        return natural_dt
    except ImportError:
        pass
    try:
        from AbaqusModel.Configuration.base import natural_dt
        return natural_dt
    except ImportError:
        pass
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _cfg = os.path.join(_here, "AbaqusModel", "Configuration")
    if _cfg not in sys.path:
        sys.path.insert(0, _cfg)
    from base import natural_dt
    return natural_dt


# --------------------------------------------------------------------------
# Mass scaling
# --------------------------------------------------------------------------

def mass_scaling_factor(cfg, L_min=None):
    """
    Mass factor f ACTUALLY applied by the solver, and the resulting increment.

      * fixed scaling  (target_time_increment <= 0):  f = solver.mass_scale
      * variable scaling (target > 0, SEMI_AUTOMATIC / BELOW_MIN):
            f = (dt_target / dt_nat(L_min))^2

    physic_verifier.check_dynamics reads solver.mass_scale in BOTH cases, so
    on every family (all of which set target_time_increment > 0 in
    families.py) it reports the WRONG scaled wave speed.
    """
    natural_dt = _import_natural_dt()
    if L_min is None:
        L_min = min(float(cfg.mesh.fine_size_x),
                    float(cfg.mesh.fine_size_y),
                    float(cfg.mesh.fine_size_z))
    dt_nat = float(natural_dt(cfg.material, L_min))
    target = float(getattr(cfg.solver, "target_time_increment", 0.0) or 0.0)
    if target > 0.0:
        f = (target / dt_nat) ** 2
        mode = "variable (target_time_increment)"
    else:
        f = float(getattr(cfg.solver, "mass_scale", 1.0) or 1.0)
        mode = "fixed (mass_scale)"
    return {"f": float(f), "dm_over_m": float(f - 1.0), "dt_nat": dt_nat,
            "dt_eff": float(target if target > 0.0 else dt_nat * np.sqrt(max(f, 1.0))),
            "L_min": float(L_min), "mode": mode}


def scaled_wave_speed(E, rho, f):
    """c_scaled = sqrt(E/rho) / sqrt(f)  [mm/s]"""
    return float(np.sqrt(float(E) / float(rho)) / np.sqrt(max(float(f), 1.0)))


def contact_impedance_force(rho, E, f, area, v_normal):
    """
    Upper bound of the spurious DYNAMIC contact force introduced by mass
    scaling:  F_dyn ~ rho_eff * c_eff * A * v_n = rho * c0 * sqrt(f) * A * v_n.
    Grows as sqrt(f): the number to compare against the measured RF2 before
    blaming inertia for a force drift.
    """
    c0 = np.sqrt(float(E) / float(rho))
    return float(float(rho) * c0 * np.sqrt(max(float(f), 1.0)) * float(area) * float(v_normal))


# --------------------------------------------------------------------------
# Contact geometry
# --------------------------------------------------------------------------

def contact_radius_rockwell(depth, R, cone_angle):
    """
    Contact radius a(depth) [mm] of the Rockwell C tip: spherical cap below
    the sphere/cone transition, conical flank beyond it.

        delta* = R * (1 - sin(alpha))            transition depth
        a      = sqrt(2*R*d - d^2)               d <= delta*   (spherical)
        a      = R*cos(alpha) + (d-delta*)*tan(alpha)          (conical)

    Convention: cone_angle is the HALF-apex angle measured from the axis
    (60 deg here, i.e. 120 deg included), matching Indenter_Config.cone_angle.
    Do NOT divide by 2: an earlier convention treated it as the full apex
    angle and reported delta* = 0.100 mm instead of 0.027 mm at R = 0.2 mm.

    Past the transition the conical formula is what matters: at 40 um the
    contact is 0.123 mm wide, not the 0.089 mm the sphere formula returns.
    """
    depth = float(depth)
    R = float(R)
    if depth <= 0.0 or R <= 0.0:
        return 0.0
    alpha = np.radians(float(cone_angle))
    delta_star = R * (1.0 - np.sin(alpha))
    if depth <= delta_star:
        d = min(depth, R)
        return float(np.sqrt(max(2.0 * R * d - d * d, 0.0)))
    return float(R * np.cos(alpha) + (depth - delta_star) * np.tan(alpha))


def sphere_cone_transition_depth(R, cone_angle):
    """delta* = R * (1 - sin(alpha)) [mm] -- where the contact leaves the cap."""
    return float(float(R) * (1.0 - np.sin(np.radians(float(cone_angle)))))


def elements_per_contact_radius(a, h_mesh):
    """N_a = a / h -- the only mesh-resolution measure that transfers between
    the Hertz benchmark and the 40 um scratch."""
    return float(a) / float(h_mesh)


# --------------------------------------------------------------------------
# Amplitude smoothing
# --------------------------------------------------------------------------

def amplitude_smoothing_window(cfg):
    """
    Absolute half-width of the SMOOTH window that Abaqus applies at the
    scratch/unload kink of the tabular amplitudes, and its size RELATIVE to
    the scratch step:

        w = smooth * min(dt_before, dt_after)
          = smooth * min(scratch_time, unload_time)
        w_rel = w / scratch_time

    Why this matters: unload_time is a FIXED 0.01 s in polymer_default while
    scratch_time is swept. w_rel therefore GROWS as scratch_time shrinks
    (5 % at T=0.05, 10 % at T=0.025, 25 % at T=0.01) -- a displacement-path
    difference between the three runs of the scratch-time comparison, i.e. a
    confound that is NOT inertial and would bias a perfectly quasi-static,
    rate-independent material. Verify against the IndenterU2 trace before
    interpreting any scratch-time sensitivity.
    """
    sm = getattr(cfg.scratch, "amplitude_smoothing", None)
    if sm is None:
        return {"smooth": None, "w": None, "w_rel": None,
                "note": "solver default smoothing"}
    T = float(cfg.scratch.scratch_time)
    dt_after = float(cfg.scratch.unload_time)
    w = float(sm) * min(T, dt_after) if dt_after > 0 else float(sm) * T
    return {"smooth": float(sm), "w": w, "w_rel": (w / T if T > 0 else None),
            "note": ("SMOOTH window is %.1f %% of the scratch step" % (100.0 * w / T)
                     if T > 0 else "")}