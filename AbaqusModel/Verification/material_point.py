# -*- coding: utf-8 -*-
"""
Level-0 reference: the constitutive law integrated INDEPENDENTLY of Abaqus.

ABAQUS-FREE (numpy only). Every function here answers one question:
"given the numbers actually written into the Abaqus material card, what
stress should a single element return?" Comparing that with a one-element
Abaqus job isolates card-construction errors (units, column order, table
sampling, T/C ratio, Prony normalisation) from everything the scratch model
adds on top. None of those errors is visible in a scratch simulation.

Conventions match Abaqus exactly:
  * yield tables are TRUE stress vs LOG plastic strain
  * *DRUCKER PRAGER HARDENING defaults to TYPE=COMPRESSION, so the tabulated
    value IS the uniaxial COMPRESSIVE yield stress
  * hyperelastic uniaxial results are true (Cauchy) stress vs stretch
"""

import numpy as np


# --------------------------------------------------------------------------
# Hardening tables
# --------------------------------------------------------------------------

def gsell_jonas_eval(eps_p, sigma_y0, h, Q=0.0, b=0.0,
                     soft_drop=0.0, eps_soft=0.05):
    """
    Closed form of base.gsell_jonas_table -- the CONTINUOUS law, before
    tabulation. Comparing it with the tabulated version quantifies the
    sampling error introduced by n_points (which Abaqus then interpolates
    linearly, and extrapolates as a PERFECTLY PLASTIC plateau beyond the last
    point).
    """
    e = np.asarray(eps_p, dtype=float)
    return ((sigma_y0
             - soft_drop * (1.0 - np.exp(-e / max(eps_soft, 1e-9)))
             + Q * (1.0 - np.exp(-b * e))) * np.exp(h * e ** 2))


def table_eval(table, eps_p):
    """
    Abaqus interpretation of a (stress, plastic_strain) table: piecewise
    linear inside, CONSTANT (perfectly plastic) beyond the last point.
    """
    tbl = np.asarray([[float(r[0]), float(r[1])] for r in table], dtype=float)
    order = np.argsort(tbl[:, 1])
    return np.interp(np.asarray(eps_p, dtype=float), tbl[order, 1], tbl[order, 0])


def table_sampling_error(table, sigma_y0, h, Q=0.0, b=0.0,
                         soft_drop=0.0, eps_soft=0.05, n=2001):
    """Max relative error of the tabulated law vs the continuous one [%]."""
    tbl = np.asarray([[float(r[0]), float(r[1])] for r in table], dtype=float)
    e = np.linspace(0.0, float(tbl[:, 1].max()), int(n))
    exact = gsell_jonas_eval(e, sigma_y0, h, Q, b, soft_drop, eps_soft)
    approx = table_eval(table, e)
    rel = np.abs(approx - exact) / np.maximum(np.abs(exact), 1e-12)
    return {"max_rel_error_pct": float(100.0 * rel.max()),
            "at_eps_p": float(e[int(np.argmax(rel))]),
            "eps_p_max_tabulated": float(tbl[:, 1].max())}


# --------------------------------------------------------------------------
# Linear elasticity
# --------------------------------------------------------------------------

def elastic_uniaxial(E, nu, eps):
    """sigma_axial = E * eps ; lateral strain = -nu * eps (small strain)."""
    eps = np.asarray(eps, dtype=float)
    return {"sigma": float(E) * eps, "eps_lat": -float(nu) * eps}


def elastic_moduli_from_E_nu(E, nu):
    K = float(E) / (3.0 * (1.0 - 2.0 * float(nu)))
    G = float(E) / (2.0 * (1.0 + float(nu)))
    return {"K": K, "G": G, "M": K + 4.0 * G / 3.0}


# --------------------------------------------------------------------------
# J2 plasticity
# --------------------------------------------------------------------------

def j2_uniaxial(E, yield_table, eps):
    """
    Rate-independent J2, isotropic hardening, uniaxial. Identical in tension
    and compression by construction -- which is exactly the property that
    makes the T/C test below a DISCRIMINATING check between J2 and DP.
    """
    eps = np.asarray(eps, dtype=float)
    sy0 = float(yield_table[0][0])
    sig = np.empty_like(eps)
    for i, e in enumerate(eps):
        s = float(E) * e
        if s <= sy0:
            sig[i] = s
            continue
        # solve  s = E*(e - ep)  with  s = sy(ep)
        ep = max(e - sy0 / float(E), 0.0)
        for _ in range(100):
            sy = float(table_eval(yield_table, ep))
            ep_new = e - sy / float(E)
            if ep_new < 0.0:
                ep_new = 0.0
            if abs(ep_new - ep) < 1e-14:
                ep = ep_new
                break
            ep = 0.5 * (ep + ep_new)
        sig[i] = float(table_eval(yield_table, ep))
    return sig


# --------------------------------------------------------------------------
# Linear Drucker-Prager
# --------------------------------------------------------------------------
# Abaqus linear DP yield surface:
#     F = t - p*tan(beta) - d = 0
#     t = (q/2) * [ 1 + 1/K - (1 - 1/K) * (r/q)^3 ]
# with p = -trace(sigma)/3 (compression positive), q = von Mises,
# r = third deviatoric invariant.
#
#   uniaxial COMPRESSION:  r/q = -1  ->  t = q  ->  d = sigma_c (1 - tan(beta)/3)
#   uniaxial TENSION:      r/q = +1  ->  t = q/K
#   pure SHEAR:            r/q =  0  ->  t = (q/2)(1 + 1/K)

def dp_cohesion_from_compression(sigma_c, beta_deg):
    """d = sigma_c * (1 - tan(beta)/3)  [MPa]"""
    return float(sigma_c) * (1.0 - np.tan(np.radians(float(beta_deg))) / 3.0)


def dp_tension_over_compression(beta_deg, K=1.0):
    """
    sigma_t / sigma_c = (1 - tan(beta)/3) / (1/K + tan(beta)/3)

    This is THE number to check a Drucker-Prager calibration against, because
    beta is almost never measured directly: it is derived from a measured T/C
    ratio. Inverting (K = 1):  tan(beta) = 3 (1 - m) / (1 + m).
    """
    tb = np.tan(np.radians(float(beta_deg)))
    return float((1.0 - tb / 3.0) / (1.0 / float(K) + tb / 3.0))


def dp_beta_from_tc_ratio(m, K=1.0):
    """Inverse of dp_tension_over_compression: beta [deg] from sigma_t/sigma_c."""
    m = float(m)
    tb = 3.0 * (1.0 / float(K) * m - 1.0) / (-1.0 - m) if K != 1.0 else 3.0 * (1.0 - m) / (1.0 + m)
    if K != 1.0:
        # solve (1 - tb/3) = m (1/K + tb/3)  ->  tb (1/3 + m/3) = 1 - m/K
        tb = (1.0 - m / float(K)) / ((1.0 + m) / 3.0)
    return float(np.degrees(np.arctan(tb)))


def dp_shear_yield(sigma_c, beta_deg, K=1.0):
    """Pure-shear yield stress implied by the same card."""
    d = dp_cohesion_from_compression(sigma_c, beta_deg)
    return float(2.0 * d / (np.sqrt(3.0) * (1.0 + 1.0 / float(K))))


def dp_uniaxial(E, yield_table, beta_deg, K, eps, mode="compression"):
    """
    Uniaxial DP response. In compression the tabulated curve is reproduced
    EXACTLY (that is the definition of TYPE=COMPRESSION hardening); in tension
    it is scaled by dp_tension_over_compression. Any deviation of the Abaqus
    single-element result from these two curves is a card error.
    """
    eps = np.asarray(np.abs(eps), dtype=float)
    ratio = 1.0 if mode == "compression" else dp_tension_over_compression(beta_deg, K)
    sy0 = float(yield_table[0][0]) * ratio
    sig = np.empty_like(eps)
    for i, e in enumerate(eps):
        s = float(E) * e
        if s <= sy0:
            sig[i] = s
            continue
        ep = max(e - sy0 / float(E), 0.0)
        for _ in range(100):
            sy = float(table_eval(yield_table, ep)) * ratio
            ep_new = max(e - sy / float(E), 0.0)
            if abs(ep_new - ep) < 1e-14:
                ep = ep_new
                break
            ep = 0.5 * (ep + ep_new)
        sig[i] = float(table_eval(yield_table, ep)) * ratio
    return sig


def dp_convexity_check(K):
    """Abaqus requires 0.778 <= K <= 1.0 for the linear DP surface to be convex."""
    K = float(K)
    return {"K": K, "ok": bool(0.778 <= K <= 1.0),
            "message": ("convex" if 0.778 <= K <= 1.0 else
                        "K outside [0.778, 1.0]: non-convex yield surface, Abaqus will reject it")}


def dp_dilatancy_note(beta_deg, psi_deg):
    """Associated flow requires psi = beta; psi < beta is NON-associated."""
    associated = abs(float(beta_deg) - float(psi_deg)) < 1e-9
    return {"associated": associated,
            "message": ("associated flow" if associated else
                        "NON-associated (psi=%.1f != beta=%.1f): the tangent operator is "
                        "non-symmetric and the problem may lose ellipticity -- expect "
                        "mesh-dependent localisation" % (float(psi_deg), float(beta_deg)))}


# --------------------------------------------------------------------------
# Hyperelasticity (incompressible uniaxial, true stress vs stretch)
# --------------------------------------------------------------------------

_AB_C = (0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0, 519.0 / 673750.0)


def mooney_rivlin_uniaxial(C10, C01, lam):
    lam = np.asarray(lam, dtype=float)
    return 2.0 * (lam ** 2 - 1.0 / lam) * (float(C10) + float(C01) / lam)


def arruda_boyce_uniaxial(mu, lambda_m, lam):
    lam = np.asarray(lam, dtype=float)
    I1 = lam ** 2 + 2.0 / lam
    dWdI1 = np.zeros_like(lam)
    for i, Ci in enumerate(_AB_C, start=1):
        dWdI1 = dWdI1 + float(mu) * i * Ci * I1 ** (i - 1) / float(lambda_m) ** (2 * i - 2)
    return 2.0 * (lam ** 2 - 1.0 / lam) * dWdI1


def yeoh_uniaxial(C10, C20, C30, lam):
    lam = np.asarray(lam, dtype=float)
    I1 = lam ** 2 + 2.0 / lam
    d = (float(C10) + 2.0 * float(C20) * (I1 - 3.0)
         + 3.0 * float(C30) * (I1 - 3.0) ** 2)
    return 2.0 * (lam ** 2 - 1.0 / lam) * d


def ogden_uniaxial(mu, alpha, lam):
    lam = np.asarray(lam, dtype=float)
    s = np.zeros_like(lam)
    for m, a in zip(mu, alpha):
        s = s + (2.0 * float(m) / float(a)) * (lam ** float(a) - lam ** (-float(a) / 2.0))
    return s


# --------------------------------------------------------------------------
# Prony viscoelasticity
# --------------------------------------------------------------------------

def prony_relaxation(prony_table, t, G0=1.0, time_scale_factor=1.0):
    """
    Normalised shear relaxation modulus of a *VISCOELASTIC, TIME=PRONY card:

        G(t) = G0 * [ 1 - sum_i g_i (1 - exp(-t/tau_i)) ]

    time_scale_factor reproduces assignment._prony (tau -> tau / tsf), so the
    reference curve moves with the card and the check stays valid when the
    simulation compresses time.

    NB the hyperelastic constants of a Prony family are the INSTANTANEOUS
    moduli; the long-term modulus is G0 * (1 - sum g_i).
    """
    t = np.asarray(t, dtype=float)
    tsf = float(time_scale_factor) if time_scale_factor else 1.0
    g = np.ones_like(t)
    for row in prony_table:
        gi, _ki, taui = float(row[0]), float(row[1]), float(row[2]) / tsf
        g = g - gi * (1.0 - np.exp(-t / max(taui, 1e-30)))
    return float(G0) * g


def prony_summary(prony_table, time_scale_factor=1.0):
    tsf = float(time_scale_factor) if time_scale_factor else 1.0
    gsum = float(sum(float(r[0]) for r in prony_table))
    taus = [float(r[2]) / tsf for r in prony_table]
    return {"sum_g": gsum, "long_term_fraction": 1.0 - gsum,
            "tau_min": min(taus), "tau_max": max(taus),
            "stable": bool(0.0 < gsum < 1.0),
            "message": ("OK" if 0.0 < gsum < 1.0 else
                        "sum(g_i) = %.3f is outside (0, 1): unstable Prony series" % gsum)}


def deborah_numbers(prony_table, v, a, time_scale_factor=1.0):
    """De = tau * v / (2a): which Prony terms are actually active at the
    scratch contact time 2a/v. Terms with De outside [0.01, 100] are inert."""
    tsf = float(time_scale_factor) if time_scale_factor else 1.0
    tc = 2.0 * float(a) / float(v)
    return [{"tau": float(r[2]) / tsf, "De": (float(r[2]) / tsf) / tc,
             "active": bool(0.01 <= (float(r[2]) / tsf) / tc <= 100.0)}
            for r in prony_table]