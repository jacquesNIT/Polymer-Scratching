# -*- coding: utf-8 -*-
"""
Closed-form / quadrature reference solutions for the V&V benchmark suite.

ABAQUS-FREE (numpy only) -- imported by the Abaqus kernel AND by the plain
CPython analysis scripts, exactly like base.py / families.py.

Unit system: mm - tonne - s - MPa - N  (consistent with the rest of the project)
    length  mm
    force   N        (so 1 mN = 1e-3 in these units)
    stress  MPa
    density tonne/mm3

Contents
    1. Hertz sphere-on-halfspace              (benchmark 1a / 1b)
    2. Rockwell tip geometry                  (contact radius, sphere/cone)
    3. Ploughing friction                     (benchmark 1c)
    4. Abaqus tabular friction mu(p)          (SCOF estimator identification)
    5. SCOF of a Briscoe law on a Hertz field (three estimators, exact)
    6. Contact-area inversion of a measured SCOF
    7. Mesh-convergence algebra                (observed order, Richardson, GCI)
    8. Mass-scaling algebra                    (f, dm/m, wave speed, Courant)
"""

import numpy as np


def _trapz(y, x):
    """
    Trapezoidal quadrature, implemented once and locally: np.trapz was removed
    in numpy 2.0 and np.trapezoid does not exist in the older numpy shipped
    with the Abaqus kernel, so NEITHER name is portable across the two
    environments this module has to run in.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# --------------------------------------------------------------------------
# 1. Hertz: rigid sphere of radius R on an elastic half-space
# --------------------------------------------------------------------------
# Valid for a/R << 1 (frictionless, small strain, half-space). The systematic
# error of the Hertz solution itself is a few % once a/R > 0.2, which is why
# hertz_validity() below is used to gate the 1 % convergence criterion.

def estar(E, nu, E_ind=None, nu_ind=None):
    """
    Reduced modulus. With a RIGID indenter (the project's analytic rigid
    surface) 1/E* = (1-nu^2)/E, i.e. E* = E/(1-nu^2).
    """
    inv = (1.0 - nu ** 2) / float(E)
    if E_ind is not None:
        inv += (1.0 - (nu_ind or 0.0) ** 2) / float(E_ind)
    return 1.0 / inv


def hertz_contact_radius(R, h):
    """a = sqrt(R*h)  [mm]"""
    h = np.asarray(h, dtype=float)
    return np.sqrt(np.maximum(R * h, 0.0))


def hertz_force(E, nu, R, h, half_model=False):
    """
    P = (4/3) * E* * sqrt(R) * h^(3/2)   [N]

    half_model=True halves it, which is the convention used everywhere in this
    project (x = 0 symmetry plane, RF2 is the half-model reaction).
    """
    h = np.asarray(h, dtype=float)
    P = (4.0 / 3.0) * estar(E, nu) * np.sqrt(R) * np.maximum(h, 0.0) ** 1.5
    return 0.5 * P if half_model else P


def hertz_depth_from_force(E, nu, R, P, half_model=False):
    """Inverse of hertz_force (useful to compare at equal load)."""
    P = np.asarray(P, dtype=float)
    if half_model:
        P = 2.0 * P
    return (3.0 * P / (4.0 * estar(E, nu) * np.sqrt(R))) ** (2.0 / 3.0)


def hertz_p0(E, nu, R, h):
    """Peak (axial) contact pressure  p0 = (2/pi) * E* * sqrt(h/R)  [MPa]"""
    h = np.asarray(h, dtype=float)
    return (2.0 / np.pi) * estar(E, nu) * np.sqrt(np.maximum(h, 0.0) / R)


def hertz_pmean(E, nu, R, h):
    """Mean contact pressure  pbar = P/(pi a^2) = (2/3) p0  [MPa]"""
    return (2.0 / 3.0) * hertz_p0(E, nu, R, h)


def hertz_strain_energy(E, nu, R, h, half_model=False):
    """
    U = (2/5) * P * h   [N.mm = mJ]

    This is the EXACT internal energy of the half-space. On the benchmark it
    gives ALLIE a reference value, hence a direct measurement of the parasitic
    contact-penalty work ALLPW (which should go to zero).
    """
    return 0.4 * hertz_force(E, nu, R, h, half_model=half_model) * np.asarray(h, dtype=float)


def hertz_pressure_profile(E, nu, R, h, r):
    """p(r) = p0 * sqrt(1 - (r/a)^2), zero outside the contact."""
    a = hertz_contact_radius(R, h)
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    inside = np.abs(r) < a
    out[inside] = hertz_p0(E, nu, R, h) * np.sqrt(1.0 - (r[inside] / a) ** 2)
    return out


def hertz_validity(R, h, a_over_R_warn=0.20, a_over_R_fail=0.35):
    """
    a/R gate on the Hertz solution itself, so a benchmark point is never
    credited (or blamed) for an error that belongs to the analytical model.
        a/R <= 0.20  -> reference good to ~1 %, usable for a 1 % criterion
        a/R <= 0.35  -> reference good to a few %, usable for trends only
    """
    a = float(hertz_contact_radius(R, h))
    ratio = a / float(R)
    if ratio <= a_over_R_warn:
        status = "OK"
    elif ratio <= a_over_R_fail:
        status = "TREND_ONLY"
    else:
        status = "INVALID"
    return {"a_over_R": ratio, "status": status}


# --------------------------------------------------------------------------
# 2. Rockwell tip geometry (sphere + tangent cone)
# --------------------------------------------------------------------------

def sphere_cone_transition_depth(R, half_angle_deg):
    """
    Penetration at which the contact leaves the spherical cap and moves onto
    the conical flank:  h* = R * (1 - sin(alpha)),  alpha measured FROM THE
    AXIS (60 deg for a Rockwell C).  Matches base.Indenter_Config.Rockwell_coords
    (yc2 = R + R*sin(-alpha)).
    """
    alpha = np.radians(float(half_angle_deg))
    return float(R) * (1.0 - np.sin(alpha))


def contact_radius_rockwell(h, R, half_angle_deg=60.0):
    """
    Geometric contact radius of the Rockwell tip. Identical convention to
    results_verifier._contact_radius / physic_verifier._contact_radius.
    """
    h = float(h)
    if h <= 0.0:
        return 0.0
    alpha = np.radians(float(half_angle_deg))
    h_star = sphere_cone_transition_depth(R, half_angle_deg)
    if h <= h_star:
        d = min(h, R)
        return float(np.sqrt(max(2.0 * R * d - d * d, 0.0)))
    return float(R * np.cos(alpha) + (h - h_star) * np.tan(alpha))


def attack_angle_deg(half_angle_deg=60.0):
    """Angle between the conical flank and the free surface (30 deg here)."""
    return 90.0 - float(half_angle_deg)


# --------------------------------------------------------------------------
# 3. Ploughing friction (rigid-plastic, frictionless interface)
# --------------------------------------------------------------------------

def ploughing_scof_cone(half_angle_deg=60.0):
    """
    Rigid-plastic ploughing coefficient of a CONE, from the ratio of the
    frontal projected area to the horizontal projected area:

        mu_p = (2/pi) * cot(alpha) = (2/pi) * tan(theta)

    alpha = half-apex angle from the axis, theta = 90 - alpha = attack angle.
    Rockwell C (alpha = 60 deg, theta = 30 deg) -> mu_p = 0.368.

    Accuracy ~ +/-20 %: it ignores elastic recovery on the rear flank
    (which lowers it) and pile-up climbing the front flank (which raises it).
    Use it as an ORDER-OF-MAGNITUDE anchor for the mu = 0 benchmark case,
    never as a 1 % criterion.
    """
    alpha = np.radians(float(half_angle_deg))
    return float((2.0 / np.pi) / np.tan(alpha))


def ploughing_scof_sphere(h, R):
    """
    Spherical-cap regime (h < h*): mu_p = (2/pi) * (a/R) approximately,
    with a = sqrt(2Rh - h^2). Degenerates correctly as h -> 0.
    """
    h = float(h)
    a = float(np.sqrt(max(2.0 * R * h - h * h, 0.0)))
    return float((2.0 / np.pi) * a / float(R))


def ploughing_scof(h, R, half_angle_deg=60.0):
    """Dispatcher on the sphere/cone regime."""
    if h <= sphere_cone_transition_depth(R, half_angle_deg):
        return ploughing_scof_sphere(h, R)
    return ploughing_scof_cone(half_angle_deg)


# --------------------------------------------------------------------------
# 4. Abaqus tabular friction mu(p)
# --------------------------------------------------------------------------

def mu_from_table(p, mu_table):
    """
    Evaluate an Abaqus pressure-dependent friction table.

    mu_table rows follow the *FRICTION data line order used by
    base.Friction_Config.briscoe():  (mu, contact_pressure).
    Abaqus interpolates LINEARLY between table points and holds the END
    VALUES CONSTANT outside the tabulated range -- which np.interp does
    natively. Reproducing this exactly is the whole point: it is what makes
    the analytic SCOF below comparable to the simulated one.
    """
    tbl = np.asarray([[float(r[0]), float(r[1])] for r in mu_table], dtype=float)
    order = np.argsort(tbl[:, 1])
    return np.interp(np.asarray(p, dtype=float), tbl[order, 1], tbl[order, 0])


def briscoe_mu(p, tau0, alpha, mu_cap=None):
    """mu(p) = tau0/p + alpha, optionally capped (the continuous law, no table)."""
    p = np.maximum(np.asarray(p, dtype=float), 1e-12)
    mu = tau0 / p + alpha
    if mu_cap is not None:
        mu = np.minimum(mu, mu_cap)
    return mu


# --------------------------------------------------------------------------
# 5. SCOF of a friction law on a Hertz pressure field -- three estimators
# --------------------------------------------------------------------------
# Substitution u = (r/a)^2 maps the contact disc to u in [0, 1] with
#   dA = pi a^2 du     and     p(u) = p0 * sqrt(1 - u)
# so every integral below is a 1-D quadrature with no singularity.

_ESTIMATORS = ("global_force_ratio", "area_weighted_mu", "unweighted_node_mu")


def scof_on_hertz(mu_fn, p0, estimator="global_force_ratio", n=20001):
    """
    Apparent friction coefficient produced by mu_fn on a Hertz contact of
    peak pressure p0, for three candidate extraction methods:

      "global_force_ratio"  SCOF = (int mu p dA) / (int p dA) = Ft/Fn
                            <-- this is |RF3|/|RF2|, i.e. what
                                results_verifier.check_friction_physics
                                already computes, and the physically correct
                                one (it is what an experimentalist measures).

      "area_weighted_mu"    SCOF = (int mu dA) / A_c
                            an area-weighted mean of the LOCAL ratio.

      "unweighted_node_mu"  identical to area_weighted_mu in the limit of a
                            uniform Cartesian surface mesh (equal tributary
                            areas), which is exactly this project's mesh.

    All three are FINITE even for an uncapped Briscoe law (the 1/p singularity
    at the contact edge is integrable because the area vanishes there).
    Closed forms without cap and without table clipping:
        global      = alpha + 1.5 * tau0 / p0
        area-mean   = alpha + 2.0 * tau0 / p0
    """
    if estimator not in _ESTIMATORS:
        raise ValueError("estimator must be one of %s" % (_ESTIMATORS,))
    u = np.linspace(0.0, 1.0, int(n))
    s = np.sqrt(np.maximum(1.0 - u, 0.0))          # p/p0
    mu = np.asarray(mu_fn(p0 * s), dtype=float)
    if estimator == "global_force_ratio":
        num = _trapz(mu * s, u)
        den = _trapz(s, u)                          # = 2/3
        return float(num / den)
    return float(_trapz(mu, u))                     # area-weighted mean


def scof_estimator_signature(mu_table, p0, n=20001):
    """
    The discriminating table: run ONE benchmark indentation with the Briscoe
    table active, read the simulated SCOF, and compare it with these three
    numbers. The estimator that matches is the one the code implements.
    """
    fn = lambda p: mu_from_table(p, mu_table)
    return {e: scof_on_hertz(fn, p0, estimator=e, n=n) for e in _ESTIMATORS}


# --------------------------------------------------------------------------
# 6. Contact-area inversion of a measured SCOF (zero-cost diagnostic)
# --------------------------------------------------------------------------

def contact_area_from_scof(scof, F_n, tau0, alpha, mu_cap=None, p_bar=None):
    """
    For an UNCAPPED Briscoe law the tangential force integrates exactly:

        Ft = int (tau0 + alpha p) dA = tau0 * A_c + alpha * Fn
        =>  SCOF = Ft/Fn = alpha + tau0 * A_c / Fn
        =>  A_c  = (SCOF - alpha) * Fn / tau0

    i.e. with Briscoe active the SCOF measures NOTHING BUT THE CONTACT AREA.
    That is the single most useful consequence of the law for V&V: a mesh
    study run with Briscoe is a contact-area convergence study in disguise.

    CAVEAT (returned in the dict): where p < tau0/(mu_cap - alpha) the cap
    binds and tau = mu_cap * p < tau0 + alpha p, so the identity UNDER-
    estimates A_c. The returned "cap_pressure" is the pressure below which
    the cap is active; compare it with the actual contact pressures before
    trusting the number. The clean alternative is to output CAREA directly
    (already requested in Modelbuilder._request_contact_pair_history but not
    written to the CSV -- see the patch script).
    """
    scof = np.asarray(scof, dtype=float)
    F_n = np.asarray(F_n, dtype=float)
    A_c = (scof - float(alpha)) * F_n / float(tau0)
    out = {"A_c": A_c, "exact": mu_cap is None}
    if mu_cap is not None and mu_cap > alpha:
        p_c = float(tau0) / (float(mu_cap) - float(alpha))
        out["cap_pressure"] = p_c
        out["note"] = ("mu_cap=%.3f binds below p=%.1f MPa; A_c is a LOWER bound"
                       % (mu_cap, p_c))
    if p_bar is not None:
        out["p_bar_check"] = F_n / A_c
    return out


# --------------------------------------------------------------------------
# 7. Mesh-convergence algebra
# --------------------------------------------------------------------------

def observed_order(h, f):
    """
    Observed order of convergence p from three solutions on a (not necessarily
    uniform) mesh ladder, ordered from COARSE to FINE:

        r21 = h1/h2, r32 = h2/h3 (h1 coarsest)
        p solves  ln|eps32/eps21| + ln((r21^p - s)/(r32^p - s)) = p ln(r21)

    Returns None when the triplet is oscillatory (eps32/eps21 < 0) or the
    fixed point does not converge -- which is itself the useful answer:
    the ladder is NOT in the asymptotic range and no Richardson/GCI value
    may be quoted.
    """
    h = [float(x) for x in h]
    f = [float(x) for x in f]
    if len(h) != 3 or len(f) != 3:
        raise ValueError("observed_order needs exactly 3 (h, f) points")
    if not (h[0] > h[1] > h[2]):
        raise ValueError("h must be strictly decreasing (coarse -> fine)")
    e21 = f[1] - f[0]
    e32 = f[2] - f[1]
    if abs(e21) < 1e-30:
        return None
    ratio = e32 / e21
    if ratio <= 0.0:
        return None                       # oscillatory: outside asymptotic range
    r21 = h[0] / h[1]
    r32 = h[1] / h[2]
    s = 1.0 * np.sign(ratio)
    p = 2.0
    for _ in range(200):
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(np.log(abs(ratio)) + q) / np.log(r21)
        if not np.isfinite(p_new):
            return None
        if abs(p_new - p) < 1e-10:
            return float(p_new)
        p = 0.5 * (p + p_new)             # damped, avoids the usual oscillation
    return float(p)


def richardson_extrapolate(h, f, p=None):
    """
    f_exact ~ f_fine + (f_fine - f_coarse) / (r^p - 1)  on the two finest points.
    p defaults to the OBSERVED order (never to an assumed 2).
    """
    if p is None:
        # The order is always estimated on the THREE FINEST points: adding
        # coarse points to the fit would import error terms that are not in
        # the asymptotic range.
        p = observed_order(h[-3:], f[-3:]) if len(h) >= 3 else None
    if p is None:
        return None
    r = float(h[-2]) / float(h[-1])
    return float(f[-1] + (f[-1] - f[-2]) / (r ** p - 1.0))


def gci(h, f, p=None, Fs=1.25):
    """
    Roache Grid Convergence Index on the two finest points [%].
    Returns None when the order cannot be established -- deliberately: a GCI
    quoted outside the asymptotic range is meaningless.
    """
    if p is None:
        p = observed_order(h[-3:], f[-3:]) if len(h) >= 3 else None
    if p is None:
        return None
    r = float(h[-2]) / float(h[-1])
    e = abs((f[-1] - f[-2]) / f[-1])
    return float(Fs * e / (r ** p - 1.0) * 100.0)


def asymptotic_range_check(h, f):
    """
    The only honest gate before quoting any extrapolation:
      * increments must DECREASE monotonically in magnitude
      * they must keep the same sign
      * the observed order must be resolvable and in a sane band
    """
    h = [float(x) for x in h]
    f = [float(x) for x in f]
    d = [f[i + 1] - f[i] for i in range(len(f) - 1)]
    signs_ok = all(np.sign(d[i]) == np.sign(d[0]) for i in range(len(d))) if d else False
    shrink_ok = all(abs(d[i + 1]) < abs(d[i]) for i in range(len(d) - 1)) if len(d) > 1 else False
    p = observed_order(h[-3:], f[-3:]) if len(h) >= 3 else None
    ok = bool(signs_ok and shrink_ok and p is not None and 0.3 <= p <= 4.0)
    return {"in_asymptotic_range": ok, "increments": d, "same_sign": signs_ok,
            "monotone_shrink": shrink_ok, "observed_order": p,
            "verdict": ("asymptotic range reached" if ok else
                        "NOT in the asymptotic range -- no Richardson/GCI is legitimate")}


# --------------------------------------------------------------------------
# 8. Mass-scaling / dynamics algebra
# --------------------------------------------------------------------------

def mass_scaling_factor(cfg, L_min=None):
    """
    A-priori mass-scaling factor f actually applied by the solver.

      * fixed scaling  (target_time_increment <= 0):  f = solver.mass_scale
      * variable scaling (target > 0, SEMI_AUTOMATIC / BELOW_MIN):
            f = (dt_target / dt_nat(L_min))^2

    physic_verifier.check_dynamics currently uses solver.mass_scale in BOTH
    cases, so on every family (all of which set target_time_increment > 0 in
    families.py) it reports the WRONG scaled wave speed. See the patch script.
    """
    from ..Configuration.base import natural_dt          # Abaqus-free import
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


def elements_per_contact_radius(a, h_mesh):
    """N_a = a / h -- the only mesh-resolution measure that transfers between
    the Hertz benchmark and the 40 um scratch."""
    return float(a) / float(h_mesh)


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