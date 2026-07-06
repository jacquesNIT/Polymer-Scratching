"""
    PHYSICAL PRE-CHECKS

    Run checks on Simulation_Config before the run --
    material/model consistency, parameter plausibility, contact regime,
    discretisation, dynamics and time scales. Pure CPython + numpy (no Abaqus
    import), so it runs on the PC side and inside the cluster wrapper alike.

    CHECKS
    1. Model compatibility   -- Abaqus/Explicit exclusions (hyperelastic+plastic, viscoelastic+plastic), 
                                Drucker-Prager validity (linear form only in Explicit, K convexity, dilation vs friction angle), 
                                yield-table structure
    2. Parameter plausibility-- K/mu window, nu, sigma_y/E window, DP friction angle vs literature (PC ~15 deg, PMMA ~20 deg),
                                hardening-table extent vs scratch strains
    3. Contact regime        -- depth vs delta* = R(1-sin alpha), contact radius,
                                Tabor representative strain vs model validity,
                                hyperelastic identifiability at this depth
    4. Rheological factor    -- X = E0*tan(beta_attack)/sigma_y (Bucaille/Felder): elastic sliding / elastoplastic / fully plastic
    5. Discretisation        -- elements per contact radius & per depth, refined
                                zone extents, far-field box ratios, scratch path
    6. Dynamics              -- scaled wave speed vs scratch/indent velocities,
                                stable-increment & increment-count estimate
                                (single-precision drift risk)
    7. Rate & time scales    -- characteristic strain rate vs rate-independent
                                plasticity; Deborah numbers of the Prony terms
    8. Recovery settling     -- recovery_time vs the mass-scaled elastic transit
                                time (settling slows by sqrt(mass_scale))
    9. Force mode            -- CONSTANT depth_mode recommendation, expected
                                static penetration for the target force,
                                unload-time sanity


    " python physic_verifier.py <family_key> "        
"""

import os
import sys
import numpy as np

#  Thresholds & reference constants
K_MU_MIN = 10.0                   # min K/mu (below: artificially compressible)
K_MU_MAX = 100.0                  # max K/mu (above: single-precision noise risk)
SIGY_E_MIN = 0.005                # sigma_y0/E0 plausibility window for polymers
SIGY_E_MAX = 0.10
DP_BETA_WINDOW = (10.0, 30.0)     # [deg] literature window (PC ~15, PMMA ~20)
DP_K_MIN = 0.778                  # convexity limit of the linear DP yield surface
X_ELASTIC = 3.0                   # X below: elastic-dominated sliding
X_PLASTIC = 30.0                  # X above: fully plastic ploughing
ELEMS_PER_CONTACT_RADIUS = 8.0    # a / h target (WARN below, FAIL below half)
ELEMS_PER_DEPTH = 4.0             # depth / h_y target
FAR_FIELD_RATIO = 10.0            # box dimension / contact radius target
V_WAVE_WARN = 0.1                 # v / c_scaled (WARN above)
V_WAVE_FAIL = 0.3                 # v / c_scaled (FAIL above)
N_INC_SINGLE_PREC = 3.0e5         # increments above which single precision drifts
RATE_INDEP_WARN = 10.0            # [1/s] strain rate above which a rate-independent
                                  # yield table is a calibration inconsistency
DEBORAH_ACTIVE = (0.01, 100.0)    # window in which a Prony term actually works
SETTLING_TRANSITS = 20.0          # recovery_time / scaled transit time target
CONSTRAINT_FACTOR = 2.0           # Tabor constraint factor C for polymers (~1.5-2.6)

_HYPERELASTIC_MODELS = ("mooney_rivlin", "neo_hooke", "yeoh", "ogden", "arruda_boyce")


#  Small-strain elastic properties from the config material
def _ab_mu0_correction(lambda_m):
    """Initial-shear-modulus correction of the Arruda-Boyce eight-chain model."""
    if not lambda_m or lambda_m <= 0.0:
        return 1.0
    l2 = float(lambda_m) ** 2
    return (1.0 + 3.0 / (5.0 * l2) + 99.0 / (175.0 * l2 ** 2)
            + 513.0 / (875.0 * l2 ** 3) + 42039.0 / (67375.0 * l2 ** 4))


def _elastic_props(mat):
    """
    Small-strain isotropic properties (mu_0, K_0, E_0, nu_0, K/mu) from the
    base-elasticity block of a Material_Config. None if the model is unknown.
    """
    h = mat.hyperelastic
    model = getattr(h, "MODEL", "none")

    if model == "elastic":
        E0, nu0 = float(h.E), float(h.nu)
        mu0 = E0 / (2.0 * (1.0 + nu0))
        K0 = E0 / (3.0 * (1.0 - 2.0 * nu0)) if nu0 < 0.5 else float("inf")
    elif model == "mooney_rivlin":
        mu0 = 2.0 * (float(h.C10) + float(h.C01))
        K0 = 2.0 / float(h.D1) if h.D1 > 0 else float("inf")
    elif model == "arruda_boyce":
        mu0 = float(h.mu) * _ab_mu0_correction(getattr(h, "lambda_m", 0.0))
        K0 = 2.0 / float(h.D) if h.D > 0 else float("inf")
    elif model == "yeoh":
        mu0 = 2.0 * float(h.C10)                       # C20/C30 vanish at I1 = 3
        K0 = 2.0 / float(h.D1) if h.D1 > 0 else float("inf")
    elif model == "ogden":
        mu0 = float(sum(h.mu))                         # Abaqus convention: mu_0 = sum(mu_i)
        K0 = 2.0 / float(h.D1) if h.D1 > 0 else float("inf")

    else:
        return None

    if model != "elastic":
        if K0 == float("inf"):
            E0, nu0 = 3.0 * mu0, 0.5
        else:
            E0 = 9.0 * K0 * mu0 / (3.0 * K0 + mu0)
            nu0 = (3.0 * K0 - 2.0 * mu0) / (2.0 * (3.0 * K0 + mu0))

    return {"model": model, "mu_0": mu0, "K_0": K0, "E_0": E0, "nu_0": nu0,
            "K_mu_ratio": (K0 / mu0 if mu0 > 0 else float("inf"))}


def _yield_info(mat):
    """(sigma_y0, yield_table, model) of the plasticity block, or None."""
    p = mat.plasticity
    model = getattr(p, "MODEL", "none")
    if model == "none":
        return None
    table = tuple(getattr(p, "yield_table", ()) or ())
    sy0 = float(table[0][0]) if table else None
    return {"model": model, "sigma_y0": sy0, "table": table}


#  Indenter / contact geometry helpers
def _tip_geometry(cfg):
    """delta*, half-apex alpha [rad] and attack angle beta [rad] of the tip."""
    R = float(cfg.indenter.tip_radius)
    alpha = np.radians(float(cfg.indenter.cone_angle))   # HALF-apex from the axis
    delta_star = R * (1.0 - np.sin(alpha))
    beta_attack = np.pi / 2.0 - alpha
    return R, alpha, beta_attack, delta_star


def _contact_radius(depth, R, alpha):
    """Geometric contact radius of the sphere-cone tip at penetration depth."""
    depth = float(depth)
    if depth <= 0.0:
        return 0.0
    delta_star = R * (1.0 - np.sin(alpha))
    if depth <= delta_star:
        d = min(depth, R)
        return float(np.sqrt(max(2.0 * R * d - d * d, 0.0)))
    return float(R * np.cos(alpha) + (depth - delta_star) * np.tan(alpha))


def _invert_depth_from_force(F, R, alpha, props, sigma_y0):
    """
    Estimated static penetration [mm] for a physical normal force F [N]
    (full model force -- the /2 half-symmetry conventions cancel out).
    Plastic families: F = C*sigma_y*pi*a(d)^2 ; elastic families: Hertz.
    """
    if F <= 0.0:
        return 0.0, "n/a"
    if sigma_y0:
        a_t = np.sqrt(F / (CONSTRAINT_FACTOR * sigma_y0 * np.pi))
        r_t = R * np.cos(alpha)
        if a_t <= r_t:
            d = R - np.sqrt(max(R * R - a_t * a_t, 0.0))
        else:
            d = R * (1.0 - np.sin(alpha)) + (a_t - r_t) / np.tan(alpha)
        return float(d), "hardness inversion (C=%.1f)" % CONSTRAINT_FACTOR
    if props is None:
        return 0.0, "n/a"
    E_star = props["E_0"] / (1.0 - props["nu_0"] ** 2)
    d = (3.0 * F / (4.0 * E_star * np.sqrt(R))) ** (2.0 / 3.0)
    return float(d), "Hertz inversion"


def _contact_geometry(cfg):
    """
    Central geometric/kinematic picture shared by the checks: expected depth
    (commanded, or inverted from the target force), contact radius, regime,
    Tabor representative strain, scratch velocity, transit time, strain rate.
    """
    scratch = cfg.scratch
    mat = cfg.material
    R, alpha, beta_attack, delta_star = _tip_geometry(cfg)
    props = _elastic_props(mat)
    yinfo = _yield_info(mat)
    sy0 = yinfo["sigma_y0"] if yinfo else None

    if scratch.is_force_controlled:
        depth, dsrc = _invert_depth_from_force(
            float(scratch.scratch_force), R, alpha, props, sy0)
    else:
        depth, dsrc = abs(float(scratch.scratch_depth)), "commanded scratch_depth"

    a_geo = _contact_radius(depth, R, alpha)
    a_tabor = np.sqrt(depth * R) if depth > 0 else 0.0

    if depth <= 0.0:
        regime, eps_char = "n/a", 0.0
    elif depth <= delta_star:
        regime = "spherical"
        eps_char = 0.2 * a_tabor / R
    else:
        regime = "conical" if depth > 2.0 * delta_star else "hybrid (sphere-cone)"
        eps_char = 0.2 * np.tan(beta_attack)

    v = (float(scratch.scratch_length) / float(scratch.scratch_time)
         if scratch.scratch_time > 0 else 0.0)
    t_contact = 2.0 * a_geo / v if (v > 0 and a_geo > 0) else float("inf")
    eps_rate = eps_char / t_contact if t_contact < float("inf") else 0.0

    return {"R": R, "alpha_deg": np.degrees(alpha),
            "beta_attack_deg": np.degrees(beta_attack),
            "delta_star": delta_star, "depth": depth, "depth_source": dsrc,
            "a_geo": a_geo, "a_tabor": a_tabor, "regime": regime,
            "eps_char": eps_char, "v": v, "t_contact": t_contact,
            "eps_rate": eps_rate, "props": props, "yinfo": yinfo}


#  1. Model compatibility (Abaqus/Explicit exclusions and DP validity)
def check_model_compatibility(cfg):
    mat = cfg.material
    base = getattr(mat.hyperelastic, "MODEL", "none")
    plast = getattr(mat.plasticity, "MODEL", "none")
    visco = getattr(mat.viscoelastic, "MODEL", "none")

    issues, notes = [], []
    status = "PASS"

    if base in _HYPERELASTIC_MODELS and plast != "none":
        status = "FAIL"
        issues.append("hyperelastic base '%s' + plasticity '%s' is forbidden in Abaqus "
                      "(use a linear-elastic base, or the Parallel Rheological Framework)"
                      % (base, plast))

    if visco != "none" and plast != "none":
        status = "FAIL"
        issues.append("*VISCOELASTIC + any plasticity option is forbidden in "
                      "Abaqus/Explicit (confirmed from .dat); use *RATE DEPENDENT "
                      "or the Parallel Rheological Framework instead")

    if plast == "drucker_prager":
        p = mat.plasticity
        K = float(getattr(p, "flow_stress_ratio", 1.0))
        beta = float(getattr(p, "friction_angle", 0.0))
        psi = float(getattr(p, "dilation_angle", 0.0))
        notes.append("Explicit supports the LINEAR Drucker-Prager form only")
        if not (DP_K_MIN <= K <= 1.0):
            status = "FAIL"
            issues.append("flow_stress_ratio K=%.3f outside the convex range "
                          "[%.3f, 1.0]" % (K, DP_K_MIN))
        if beta >= 71.5:
            status = "FAIL"
            issues.append("friction_angle %.1f deg >= 71.5 deg (Abaqus hard limit)" % beta)
        if psi > beta:
            if status != "FAIL":
                status = "WARN"
            issues.append("dilation_angle (%.1f) > friction_angle (%.1f): "
                          "over-associated flow, unusual for polymers" % (psi, beta))

    yinfo = _yield_info(mat)
    if yinfo and yinfo["table"]:
        tbl = yinfo["table"]
        eps = [row[1] for row in tbl]
        sig = [row[0] for row in tbl]
        if abs(eps[0]) > 1e-12:
            status = "FAIL"
            issues.append("first plastic strain of the yield table must be 0.0 "
                          "(got %s)" % eps[0])
        if any(e2 <= e1 for e1, e2 in zip(eps, eps[1:])):
            status = "FAIL"
            issues.append("plastic strains of the yield table must be strictly increasing")
        if any(sv <= 0.0 for sv in sig):
            status = "FAIL"
            issues.append("yield stresses must be positive")
        if any(s2 < s1 for s1, s2 in zip(sig, sig[1:])):
            notes.append("softening branch in the yield table: physical for glassy "
                         "polymers but mesh-dependent (localisation) without regularisation")

    msg = "OK" if not issues else " ; ".join(issues)
    if notes:
        msg += " [" + " ; ".join(notes) + "]"
    return {"status": status,
            "message": "base=%s, plasticity=%s, viscoelastic=%s. %s"
                       % (base, plast, visco, msg)}


#  2. Parameter plausibility (literature windows)
def check_parameter_plausibility(cfg, geo):
    mat = cfg.material
    props = geo["props"]
    yinfo = geo["yinfo"]

    if props is None:
        return {"status": "SKIP", "message": "Unknown base-elasticity model"}

    issues = []
    status = "PASS"

    # Quasi-incompressibility window (hyperelastic bases)
    if props["model"] in _HYPERELASTIC_MODELS:
        ratio = props["K_mu_ratio"]
        if ratio < K_MU_MIN:
            status = "FAIL"
            issues.append("K/mu=%.1f < %.0f: artificially compressible (nu_0=%.3f)"
                          % (ratio, K_MU_MIN, props["nu_0"]))
        elif ratio > K_MU_MAX:
            status = "WARN"
            issues.append("K/mu=%.1f > %.0f: single-precision noise risk ('D too small')"
                          % (ratio, K_MU_MAX))
    else:
        nu = props["nu_0"]
        if not (0.0 < nu < 0.5):
            status = "FAIL"
            issues.append("nu=%.3f outside (0, 0.5)" % nu)

    # Yield level and hardening extent
    if yinfo and yinfo["sigma_y0"]:
        r = yinfo["sigma_y0"] / props["E_0"] if props["E_0"] > 0 else float("inf")
        if not (SIGY_E_MIN <= r <= SIGY_E_MAX):
            if status != "FAIL":
                status = "WARN"
            issues.append("sigma_y0/E0 = %.4f outside the polymer window [%.3f, %.2f]"
                          % (r, SIGY_E_MIN, SIGY_E_MAX))

        eps_last = yinfo["table"][-1][1] if yinfo["table"] else 0.0
        eps_local = 2.5 * geo["eps_char"]      # local strains ~2-3x the representative one
        if eps_last < eps_local:
            if status != "FAIL":
                status = "WARN"
            issues.append("hardening table ends at eps_p=%.2f < ~%.2f expected locally "
                          "under the bow-wave: Abaqus extrapolates a plateau (perfectly "
                          "plastic) -- consider a dense G'Sell-Jonas table up to eps_p~3"
                          % (eps_last, eps_local))

        if yinfo["model"] == "drucker_prager":
            beta = float(mat.plasticity.friction_angle)
            if not (DP_BETA_WINDOW[0] <= beta <= DP_BETA_WINDOW[1]):
                if status != "FAIL":
                    status = "WARN"
                issues.append("DP friction angle %.0f deg outside the literature window "
                              "[%.0f, %.0f] (PC ~15 deg, PMMA ~20 deg)"
                              % (beta, DP_BETA_WINDOW[0], DP_BETA_WINDOW[1]))

    verdict = "OK" if not issues else " ; ".join(issues)
    return {"status": status,
            "props": props,
            "message": ("E_0=%.3g MPa, nu_0=%.4f, mu_0=%.3g MPa, K/mu=%.1f%s. %s"
                        % (props["E_0"], props["nu_0"], props["mu_0"],
                           props["K_mu_ratio"],
                           (", sigma_y0=%.3g MPa" % yinfo["sigma_y0"])
                           if (yinfo and yinfo["sigma_y0"]) else "",
                           verdict))}


#  3. Contact regime & model validity at this depth
def check_contact_regime(cfg, geo):
    if geo["depth"] <= 0.0:
        return {"status": "SKIP", "message": "No usable penetration depth"}

    mat = cfg.material
    h = mat.hyperelastic
    model = getattr(h, "MODEL", "none")

    issues, notes = [], []
    status = "PASS"

    eps_local = 2.5 * geo["eps_char"]   # heuristic local peak vs representative

    if model == "mooney_rivlin":
        if eps_local > 1.0:
            status = "WARN"
            issues.append("estimated local strain ~%.2f beyond the MR validity "
                          "(~100-150%%)" % eps_local)
        if eps_local < 0.3:
            notes.append("at this depth MR/AB/neo-Hooke are nearly indistinguishable: "
                         "C01 is weakly identifiable (deeper scratch or sharper tip "
                         "needed to discriminate hyperelastic models)")
    elif model == "arruda_boyce":
        lam_local = 1.0 + eps_local
        lam_m = float(getattr(h, "lambda_m", 0.0) or 0.0)
        if lam_m > 0 and lam_local > 0.8 * lam_m:
            status = "WARN"
            issues.append("estimated local stretch %.2f approaches lambda_m=%.2f: "
                          "locking dominates, verify max(LE) in the field output"
                          % (lam_local, lam_m))
        if lam_m > 0 and eps_local < 0.3:
            notes.append("local strains well below locking: lambda_m weakly "
                         "identifiable at this depth (MR/AB nearly equivalent)")
            
    elif model in ("yeoh", "ogden"):
        # Both remain valid at large strains; the question is discrimination.
        if eps_local < 0.3:
            notes.append("at this depth the higher-order terms (C20/C30, alpha_i) "
                         "barely work: all matched hyperelastic models are nearly "
                         "equivalent -- deepen the scratch to discriminate them")



    verdict = "OK" if not issues else " ; ".join(issues)
    if notes:
        verdict += " [" + " ; ".join(notes) + "]"
    return {"status": status,
            "message": ("depth=%.4f mm (%s), delta*=%.4f mm -> %s regime | "
                        "a_geo=%.4f mm (a/R=%.2f) | eps_char=%.3f (local ~%.2f). %s"
                        % (geo["depth"], geo["depth_source"], geo["delta_star"],
                           geo["regime"], geo["a_geo"], geo["a_geo"] / geo["R"],
                           geo["eps_char"], eps_local, verdict))}


#  4. Rheological factor X (Bucaille / Felder)
def check_rheological_factor(cfg, geo):
    props = geo["props"]
    yinfo = geo["yinfo"]
    if props is None:
        return {"status": "SKIP", "message": "Unknown base-elasticity model"}

    if yinfo is None or not yinfo["sigma_y0"]:
        return {"status": "INFO",
                "message": ("No yield surface: pure elastic sliding regime. Full "
                            "rear-face recovery expected, residual depth ~ 0, "
                            "SCOF ~ mu_input + small recoverable ploughing term "
                            "(no rate/hysteretic friction without viscoelasticity).")}

    X = props["E_0"] * np.tan(np.radians(geo["beta_attack_deg"])) / yinfo["sigma_y0"]
    eps_e = yinfo["sigma_y0"] / props["E_0"]

    if X < X_ELASTIC:
        regime = ("elastic-dominated sliding: sink-in in front of the tip, strong "
                  "rear-face elastic recovery, little or no residual groove expected")
    elif X < X_PLASTIC:
        regime = ("elastoplastic transition: partial rear recovery AND permanent "
                  "groove/pile-up coexist -- the most model-sensitive regime")
    else:
        regime = ("fully plastic ploughing: pronounced pile-up, groove ~ geometric, "
                  "hardening law controls the bourrelet shape")

    return {"status": "INFO",
            "X": float(X),
            "message": ("X = E0*tan(beta)/sigma_y = %.1f (beta_attack=%.0f deg, "
                        "eps_e=sigma_y/E=%.4f) -> %s"
                        % (X, geo["beta_attack_deg"], eps_e, regime))}


#  5. Discretisation & box size (a priori mesh audit)
def check_discretization(cfg, geo):
    if geo["depth"] <= 0.0 or geo["a_geo"] <= 0.0:
        return {"status": "SKIP", "message": "No usable contact size"}

    sub = cfg.substrate
    msh = cfg.mesh
    a = geo["a_geo"]
    depth = geo["depth"]

    h_lat = max(float(msh.fine_size_x), float(msh.fine_size_z))
    h_y = float(msh.fine_size_y)

    n_a = a / h_lat if h_lat > 0 else float("inf")
    n_d = depth / h_y if h_y > 0 else float("inf")

    issues = []
    status = "PASS"

    if n_a < ELEMS_PER_CONTACT_RADIUS / 2.0:
        status = "FAIL"
        issues.append("only %.1f elements per contact radius (target >= %.0f): "
                      "forces/pile-up cannot converge" % (n_a, ELEMS_PER_CONTACT_RADIUS))
    elif n_a < ELEMS_PER_CONTACT_RADIUS:
        status = "WARN"
        issues.append("%.1f elements per contact radius (target >= %.0f)"
                      % (n_a, ELEMS_PER_CONTACT_RADIUS))

    if n_d < ELEMS_PER_DEPTH / 2.0:
        status = "FAIL"
        issues.append("only %.1f elements across the penetration depth (target >= %.0f)"
                      % (n_d, ELEMS_PER_DEPTH))
    elif n_d < ELEMS_PER_DEPTH:
        if status != "FAIL":
            status = "WARN"
        issues.append("%.1f elements across the penetration depth (target >= %.0f)"
                      % (n_d, ELEMS_PER_DEPTH))

    # Refined-zone extents (subsurface stress field ~ 2 contact radii)
    if float(sub.dpo_y) < 2.0 * a:
        if status != "FAIL":
            status = "WARN"
        issues.append("dpo_y=%.3f mm < 2a=%.3f mm: fine zone too shallow for the "
                      "subsurface stress field" % (sub.dpo_y, 2.0 * a))
    if float(sub.dpo_x) < 2.0 * a:
        if status != "FAIL":
            status = "WARN"
        issues.append("dpo_x=%.3f mm < 2a=%.3f mm: fine zone too narrow"
                      % (sub.dpo_x, 2.0 * a))

    # Scratch path: tip starts at z=dpo_z and must end inside the refined band
    z_end = float(sub.dpo_z) + float(cfg.scratch.scratch_length)
    if z_end > float(sub.zs2):
        status = "FAIL"
        issues.append("scratch end z=%.2f mm beyond the substrate (zs2=%.2f mm)"
                      % (z_end, sub.zs2))
    elif z_end > float(sub.zs2) - float(sub.dpo_z):
        if status != "FAIL":
            status = "WARN"
        issues.append("scratch ends outside the refined z-band (z_end=%.2f > %.2f mm)"
                      % (z_end, sub.zs2 - sub.dpo_z))

    # Far-field box ratios (encastre bottom / lateral face stiffen the response)
    for label, dim in (("depth ys2", float(sub.ys2)), ("half-width xs2", float(sub.xs2))):
        if dim / a < FAR_FIELD_RATIO:
            if status != "FAIL":
                status = "WARN"
            issues.append("%s/a = %.1f < %.0f: boundary stiffening of a few %% possible"
                          % (label, dim / a, FAR_FIELD_RATIO))

    verdict = "OK" if not issues else " ; ".join(issues)
    return {"status": status,
            "elements_per_contact_radius": float(n_a),
            "elements_per_depth": float(n_d),
            "message": ("a=%.4f mm, h_lat=%.3f, h_y=%.3f -> %.1f elems/a, %.1f elems/depth. %s"
                        % (a, h_lat, h_y, n_a, n_d, verdict))}


#  6. Dynamics: wave speed, velocities, increment budget
def check_dynamics(cfg, geo):
    props = geo["props"]
    if props is None:
        return {"status": "SKIP", "message": "Unknown base-elasticity model"}

    solver = cfg.solver
    scratch = cfg.scratch
    msh = cfg.mesh
    rho = float(cfg.material.rho)

    c0 = np.sqrt(props["E_0"] / rho)                      # [mm/s] bar wave speed
    f = max(float(solver.mass_scale), 1.0)
    c_scaled = c0 / np.sqrt(f)

    v = geo["v"]
    ratio_scratch = v / c_scaled if c_scaled > 0 else float("inf")

    if scratch.depth_mode == scratch.CONSTANT and scratch.indentation_time > 0:
        v_ind = geo["depth"] / float(scratch.indentation_time)
    else:
        v_ind = geo["depth"] / float(scratch.scratch_time) if scratch.scratch_time > 0 else 0.0
    ratio_ind = v_ind / c_scaled if c_scaled > 0 else float("inf")

    worst = max(ratio_scratch, ratio_ind)
    issues = []
    if worst >= V_WAVE_FAIL:
        status = "FAIL"
        issues.append("loading velocity reaches %.0f%% of the scaled wave speed: "
                      "inertia-dominated response" % (worst * 100.0))
    elif worst >= V_WAVE_WARN:
        status = "WARN"
        issues.append("loading velocity at %.0f%% of the scaled wave speed"
                      % (worst * 100.0))
    else:
        status = "PASS"

    # Increment budget (variable mass scaling targets dt directly)
    if solver.target_time_increment > 0.0:
        dt = float(solver.target_time_increment)
        dt_src = "target_time_increment"
    else:
        h_min = min(float(msh.fine_size_x), float(msh.fine_size_y), float(msh.fine_size_z))
        dt = h_min / c_scaled if c_scaled > 0 else 0.0
        dt_src = "h_min/c_scaled estimate"
    n_inc = float(scratch.total_time) / dt if dt > 0 else float("inf")
    if n_inc > N_INC_SINGLE_PREC:
        if status == "PASS":
            status = "WARN"
        issues.append("~%.1e increments expected (> %.0e): use DOUBLE precision "
                      "(explicitPrecision) to avoid round-off energy drift"
                      % (n_inc, N_INC_SINGLE_PREC))

    verdict = "OK" if not issues else " ; ".join(issues)
    return {"status": status,
            "c_scaled_mm_s": float(c_scaled),
            "n_increments_estimate": float(n_inc),
            "message": ("c0=%.3g mm/s, c_scaled=%.3g mm/s (MS=%g) | v_scratch=%.3g, "
                        "v_indent=%.3g mm/s | dt~%.2e s (%s), ~%.1e increments. %s"
                        % (c0, c_scaled, f, v, v_ind, dt, dt_src, n_inc, verdict))}


#  7. Rate & time-scale consistency
def check_rate_consistency(cfg, geo):
    mat = cfg.material
    yinfo = geo["yinfo"]
    visco = mat.viscoelastic
    visco_model = getattr(visco, "MODEL", "none")

    parts, issues = [], []
    status = "PASS"

    if geo["eps_rate"] > 0:
        parts.append("eps_rate ~ %.1e /s (contact transit 2a/v = %.2e s)"
                     % (geo["eps_rate"], geo["t_contact"]))

    rate_cfg = getattr(mat.plasticity, "rate_dependent", None) if yinfo else None

    if yinfo and rate_cfg is None and geo["eps_rate"] > RATE_INDEP_WARN:
        status = "WARN"
        issues.append("rate-INDEPENDENT yield table used at ~%.0f /s: a table "
                      "calibrated quasi-statically underestimates the flow stress "
                      "by ~5-10%% per decade (polymers, Eyring) -- calibrate at this "
                      "rate or add *RATE DEPENDENT" % geo["eps_rate"])
    elif yinfo and rate_cfg is not None and geo["eps_rate"] > 0:
        # Cowper-Symonds active: report the predicted overstress and verify the
        # working rate sits inside the Eyring fit window (the power-law fit is
        # only trustworthy inside it).
        try:
            R = rate_cfg.ratio(geo["eps_rate"])
            parts.append("Cowper-Symonds active: R = sigma_dyn/sigma_stat ~ %.2f "
                         "at %.0f /s (D=%.3g, n=%.2f)"
                         % (R, geo["eps_rate"], rate_cfg.D, rate_cfg.n))
            win = getattr(rate_cfg, "fit_window", None)
            if win and not (win[0] / 10.0 <= geo["eps_rate"] <= win[1] * 10.0):
                if status != "FAIL":
                    status = "WARN"
                issues.append("working strain rate %.1e /s is far outside the "
                              "Eyring fit window [%g, %g] /s: refit from_eyring "
                              "over the relevant decades"
                              % (geo["eps_rate"], win[0], win[1]))
        except Exception as exc:
            parts.append("rate model present but unreadable (%s)" % exc)

    if visco_model == "prony":
        table = tuple(getattr(visco, "prony_table", ()) or ())
        # Prony_Config stores LAB relaxation times; the material builder
        # divides them by solver.time_scale_factor at build time -- the
        # Deborah analysis must therefore use the SIMULATED taus.
        tsf = float(getattr(cfg.solver, "time_scale_factor", 1.0) or 1.0)
        de_list = []
        active = 0
        for (_g, _k, tau) in table:
            tau_sim = float(tau) / tsf
            de = tau_sim * geo["v"] / (2.0 * geo["a_geo"]) if geo["a_geo"] > 0 else 0.0
            de_list.append(de)
            if DEBORAH_ACTIVE[0] <= de <= DEBORAH_ACTIVE[1]:
                active += 1
        tsf_note = "" if tsf == 1.0 else " (lab taus / time_scale_factor=%g)" % tsf
        parts.append("Deborah numbers%s: %s"
                     % (tsf_note, ", ".join("%.2g" % d for d in de_list)))
        if table and active == 0:
            status = "WARN"
            issues.append("no Prony term with De in [%.2g, %.2g]: the viscoelasticity "
                          "is either frozen (De >> 1) or fully relaxed (De << 1) at "
                          "this scratch speed -- retune tau_i to the contact time, and "
                          "remember that shortening the simulated time requires "
                          "shifting the relaxation spectrum by the same factor"
                          % DEBORAH_ACTIVE)
    elif yinfo is None:
        parts.append("no viscoelasticity: the response is strictly rate-independent "
                     "(velocity parametric studies are non-predictive)")

    verdict = "OK" if not issues else " ; ".join(issues)
    return {"status": status,
            "eps_rate_per_s": float(geo["eps_rate"]),
            "message": "%s. %s" % (" | ".join(parts) if parts else "n/a", verdict)}


#  8. Recovery settling budget
def check_recovery_settling(cfg, geo):
    scratch = cfg.scratch
    props = geo["props"]
    if not scratch.has_recovery_step:
        return {"status": "SKIP", "message": "No recovery step (recovery_time = 0)"}
    if props is None:
        return {"status": "SKIP", "message": "Unknown base-elasticity model"}

    rho = float(cfg.material.rho)
    f = max(float(cfg.solver.mass_scale), 1.0)
    c_scaled = np.sqrt(props["E_0"] / rho) / np.sqrt(f)
    T_transit = 2.0 * float(cfg.substrate.ys2) / c_scaled if c_scaled > 0 else float("inf")
    n = float(scratch.recovery_time) / T_transit if T_transit > 0 else float("inf")

    if n >= SETTLING_TRANSITS:
        status, verdict = "PASS", "OK"
    elif n >= SETTLING_TRANSITS / 4.0:
        status, verdict = "WARN", ("marginal settling budget -- the residual profile "
                                   "may still be ringing at the final frame")
    else:
        status, verdict = "FAIL", ("recovery_time far too short for the mass-scaled "
                                   "dynamics: extend it or reduce mass_scale")

    return {"status": status,
            "transits": float(n),
            "message": ("recovery_time=%.3g s = %.0f elastic transits (2*ys2/c_scaled "
                        "= %.2e s; mass scaling slows settling by sqrt(MS)=%.1f; "
                        "target >= %.0f). %s"
                        % (scratch.recovery_time, n, T_transit, np.sqrt(f),
                           SETTLING_TRANSITS, verdict))}


#  9. Force-controlled mode specifics
def check_force_mode(cfg, geo):
    scratch = cfg.scratch
    if not scratch.is_force_controlled:
        return {"status": "SKIP", "message": "Displacement-controlled scratch"}

    issues = []
    status = "PASS"

    if scratch.depth_mode == scratch.PROGRESSIVE:
        status = "WARN"
        issues.append("PROGRESSIVE depth_mode in force control ramps the force during "
                      "the whole scratch: use CONSTANT so the force ramps during "
                      "indentation_time and holds at the plateau (over-indentation "
                      "/ inertial overshoot mitigation)")

    if scratch.unload_time < scratch.indentation_time:
        if status != "FAIL":
            status = "WARN"
        issues.append("unload_time (%.3g s) < indentation_time (%.3g s): abrupt force "
                      "removal excites the indenter inertia -- lengthen the unload"
                      % (scratch.unload_time, scratch.indentation_time))

    verdict = "OK" if not issues else " ; ".join(issues)
    return {"status": status,
            "expected_depth_mm": float(geo["depth"]),
            "message": ("F_target=%.3g N -> expected static penetration ~%.4f mm "
                        "(%s), regime %s. %s"
                        % (scratch.scratch_force, geo["depth"], geo["depth_source"],
                           geo["regime"], verdict))}


#  Master verification
def verify_physics(cfg, print_report=True):
    """
    Run the a-priori physical consistency checks on a Simulation_Config.
    Returns a report dict shaped like results_verifier.verify_results.
    """
    geo = _contact_geometry(cfg)
    family = str(getattr(cfg.material, "family", "?"))

    report = {"family": family, "job_name": getattr(cfg, "job_name", "?"),
              "geometry": geo, "checks": {}}

    registry = (
        ("Model compatibility (Abaqus)",  lambda: check_model_compatibility(cfg)),
        ("Parameter plausibility",        lambda: check_parameter_plausibility(cfg, geo)),
        ("Contact regime / validity",     lambda: check_contact_regime(cfg, geo)),
        ("Rheological factor (X)",        lambda: check_rheological_factor(cfg, geo)),
        ("Discretisation / box size",     lambda: check_discretization(cfg, geo)),
        ("Dynamics (wave speed, dt)",     lambda: check_dynamics(cfg, geo)),
        ("Rate & time scales",            lambda: check_rate_consistency(cfg, geo)),
        ("Recovery settling budget",      lambda: check_recovery_settling(cfg, geo)),
        ("Force-controlled mode",         lambda: check_force_mode(cfg, geo)),
    )
    for label, run in registry:
        report["checks"][label] = run()

    if print_report:
        _print_report(report)
    return report


def _print_report(report):
    print("")
    print("-" * 60)
    print("  SCRATCH SIMULATION — Physical pre-check (a priori)")
    print("  Family: %s | Job: %s" % (report["family"], report["job_name"]))
    print("-" * 60)
    geo = report["geometry"]
    print("  Tip: R=%.2f mm, half-apex=%.0f deg (attack %.0f deg), delta*=%.4f mm"
          % (geo["R"], geo["alpha_deg"], geo["beta_attack_deg"], geo["delta_star"]))
    print("  Expected depth: %.4f mm (%s) -> %s regime, a=%.4f mm"
          % (geo["depth"], geo["depth_source"], geo["regime"], geo["a_geo"]))
    print("  Scratch speed: %.3g mm/s | eps_char=%.3f | eps_rate ~ %.1e /s"
          % (geo["v"], geo["eps_char"], geo["eps_rate"]))

    for name, result in report["checks"].items():
        status = result.get("status", "INFO")
        print("")
        print("  [%4s]  %s" % (status, name))
        print("          %s" % result.get("message", ""))

    print("")
    print("-" * 60)


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))
    try:
        from ScratchSimulation.AbaqusModel.Configuration import get_family
    except ImportError:
        print("Could not import ScratchSimulation.AbaqusModel.Configuration -- "
              "run from the repository root or pass a Simulation_Config to "
              "verify_physics() directly.")
        sys.exit(1)

    key = sys.argv[1] if len(sys.argv) > 1 else "elastomer_mr"
    verify_physics(get_family(key).build_config())