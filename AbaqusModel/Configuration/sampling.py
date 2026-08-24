# Screening / sweep design layer -- Drucker-Prager families only.
# Abaqus-free (numpy only). Imported by families.py, by the Abaqus kernel and
# by the CPython design/analysis scripts. See SWEEP_METHODOLOGY.md.

import numpy as np

from .base import (LinearElastic_Config, DruckerPrager_Config,
                   Friction_Config, gsell_jonas_table, natural_dt)


# ----------------------------------------------------------------------
# Factor / campaign containers
# ----------------------------------------------------------------------

class Factor(object):

    def __init__(self, name, lo, hi, scale="lin", unit="-", description=""):
        if scale not in ("lin", "log"):
            raise ValueError("Factor '%s': scale must be 'lin' or 'log'" % name)
        if scale == "log" and (lo <= 0.0 or hi <= 0.0):
            raise ValueError("Factor '%s': log scale requires strictly positive bounds" % name)
        if hi <= lo:
            raise ValueError("Factor '%s': need hi > lo (got %g, %g)" % (name, lo, hi))
        self.name = name
        self.lo = float(lo)
        self.hi = float(hi)
        self.scale = scale
        self.unit = unit
        self.description = description

    def to_physical(self, u):
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        if self.scale == "log":
            return self.lo * (self.hi / self.lo) ** u
        return self.lo + u * (self.hi - self.lo)

    def to_unit(self, value):
        v = np.asarray(value, dtype=float)
        if self.scale == "log":
            return np.log(v / self.lo) / np.log(self.hi / self.lo)
        return (v - self.lo) / (self.hi - self.lo)

    def mid(self):
        return float(self.to_physical(0.5))

    def __repr__(self):
        return "Factor(%s, [%g, %g], %s)" % (self.name, self.lo, self.hi, self.scale)


class FamilySampling(object):

    def __init__(self, campaign, factors, frozen, derive, apply_fn,
                 expects=None, covers=(), label="", notes=""):
        self.campaign = campaign
        self.factors = list(factors)
        self.frozen = dict(frozen)
        self._derive = derive
        self._apply = apply_fn
        self.expects = dict(expects or {})
        self.covers = tuple(covers)
        self.label = label
        self.notes = notes
        seen = set()
        for f in self.factors:
            if f.name in seen:
                raise ValueError("Duplicate factor name '%s' in campaign '%s'" % (f.name, campaign))
            seen.add(f.name)

    @property
    def names(self):
        return [f.name for f in self.factors]

    @property
    def dim(self):
        return len(self.factors)

    def factor(self, name):
        for f in self.factors:
            if f.name == name:
                return f
        raise KeyError("Campaign '%s' has no factor '%s' (available: %s)"
                       % (self.campaign, name, ", ".join(self.names)))

    def unit_to_groups(self, unit_row):
        unit_row = np.asarray(unit_row, dtype=float).reshape(-1)
        if unit_row.size != self.dim:
            raise ValueError("Campaign '%s' expects %d coordinates, got %d"
                             % (self.campaign, self.dim, unit_row.size))
        return dict((f.name, float(f.to_physical(unit_row[i])))
                    for i, f in enumerate(self.factors))

    def derive(self, groups, cfg):
        missing = [n for n in self.names if n not in groups]
        if missing:
            raise KeyError("Campaign '%s': missing group(s) %s"
                           % (self.campaign, ", ".join(missing)))
        unknown = [k for k in groups if k not in self.names]
        if unknown:
            raise KeyError("Campaign '%s': unknown group(s) %s (declared: %s)"
                           % (self.campaign, ", ".join(sorted(unknown)), ", ".join(self.names)))
        return self._derive(groups, cfg)

    def apply(self, cfg, params):
        _assert_expected(cfg, self.expects, self.campaign)
        s = _target_dt_scale(cfg)
        self._apply(cfg, params)
        _restore_target_dt(cfg, s)
        return cfg

    def configure(self, cfg, groups):
        return self.apply(cfg, self.derive(groups, cfg))


# ----------------------------------------------------------------------
# Design generators
# ----------------------------------------------------------------------

def morris_delta(n_levels):
    n_levels = int(n_levels)
    if n_levels < 2 or n_levels % 2 != 0:
        raise ValueError("morris: n_levels must be an even integer >= 2 (got %s)" % n_levels)
    return n_levels / (2.0 * (n_levels - 1.0))


def _one_trajectory(d, n_levels, delta, rs):
    grid = np.arange(n_levels, dtype=float) / (n_levels - 1.0)
    allowed = grid[grid <= 1.0 - delta + 1e-12]
    xstar = rs.choice(allowed, size=d)
    B = np.tril(np.ones((d + 1, d), dtype=float), -1)
    Dstar = np.diag(rs.choice(np.array([-1.0, 1.0]), size=d))
    P = np.eye(d)[rs.permutation(d)]
    J1 = np.ones((d + 1, 1), dtype=float)
    Jd = np.ones((d + 1, d), dtype=float)
    Bstar = (J1.dot(xstar.reshape(1, d))
             + (delta / 2.0) * ((2.0 * B - Jd).dot(Dstar) + Jd)).dot(P)
    return np.clip(Bstar, 0.0, 1.0)


def _trajectory_distances(traj):
    M, k, d = traj.shape
    flat = traj.reshape(M * k, d)
    sq = np.sum(flat * flat, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * flat.dot(flat.T)
    Dpt = np.sqrt(np.maximum(D2, 0.0))
    return Dpt.reshape(M, k, M, k).sum(axis=(1, 3))


def _greedy_maximin(D, r):
    M = D.shape[0]
    best_set, best_score = None, -np.inf
    for seed in range(M):
        chosen = [seed]
        gain = D[seed].copy()
        gain[seed] = -np.inf
        for _ in range(r - 1):
            nxt = int(np.argmax(gain))
            chosen.append(nxt)
            gain = gain + D[nxt]
            gain[chosen] = -np.inf
        idx = np.array(chosen)
        score = float(D[np.ix_(idx, idx)].sum()) / 2.0
        if score > best_score:
            best_score, best_set = score, idx
    return np.sort(best_set), best_score


def morris_design(n_factors, n_trajectories, n_levels=4, seed=0, n_candidates=200):
    """
    Morris trajectory design with Campolongo-style maximin selection.
    Returns a dict; "unit" is (n_trajectories*(d+1), d) in [0, 1]^d.
    """
    d = int(n_factors)
    r = int(n_trajectories)
    if d < 1:
        raise ValueError("morris_design: n_factors must be >= 1")
    if r < 2:
        raise ValueError("morris_design: n_trajectories must be >= 2")
    M = max(int(n_candidates), r)
    delta = morris_delta(n_levels)

    rs = np.random.RandomState(int(seed))
    cand = np.array([_one_trajectory(d, n_levels, delta, rs) for _ in range(M)])

    if M > r:
        keep, score = _greedy_maximin(_trajectory_distances(cand), r)
    else:
        keep, score = np.arange(r), float("nan")
    sel = cand[keep]

    k = d + 1
    unit = sel.reshape(r * k, d)
    traj = np.repeat(np.arange(r), k)
    step = np.tile(np.arange(k), r)
    moved = np.full(r * k, -1, dtype=int)
    sign = np.zeros(r * k, dtype=float)
    for j in range(r):
        for s_ in range(1, k):
            diff = sel[j, s_] - sel[j, s_ - 1]
            i = int(np.argmax(np.abs(diff)))
            moved[j * k + s_] = i
            sign[j * k + s_] = 1.0 if diff[i] > 0 else -1.0

    return {"unit": unit, "traj": traj, "step": step, "moved": moved,
            "sign": sign, "delta": delta, "n_levels": int(n_levels),
            "n_trajectories": r, "n_factors": d, "seed": int(seed),
            "n_candidates": M, "maximin_score": score, "method": "morris"}


def sobol_design(n_factors, n_samples, seed=0):
    """
    Joint d-dimensional scrambled Sobol sequence. scipy is imported lazily so
    this module stays importable from the Abaqus kernel.
    """
    from scipy.stats import qmc
    d, n = int(n_factors), int(n_samples)
    unit = qmc.Sobol(d=d, scramble=True, seed=int(seed)).random(n=n)
    return {"unit": unit, "traj": np.full(n, -1, dtype=int),
            "step": np.arange(n), "moved": np.full(n, -1, dtype=int),
            "sign": np.zeros(n, dtype=float), "delta": float("nan"),
            "n_levels": 0, "n_trajectories": 0, "n_factors": d,
            "seed": int(seed), "n_candidates": 0,
            "maximin_score": float("nan"), "method": "sobol"}


# ----------------------------------------------------------------------
# Geometry / solver helpers
# ----------------------------------------------------------------------

def attack_angle_deg(cfg):
    # Angle between the indenter flank and the free surface [deg].
    return 90.0 - float(cfg.indenter.cone_angle)


def tangency_depth(cfg):
    R = float(cfg.indenter.tip_radius)
    return R * (1.0 - np.sin(np.radians(float(cfg.indenter.cone_angle))))


def contact_radius(cfg, depth=None):
    # Geometric contact radius [mm] of the sphero-cone at a given penetration.
    R = float(cfg.indenter.tip_radius)
    beta = np.radians(float(cfg.indenter.cone_angle))
    d = abs(float(cfg.scratch.scratch_depth if depth is None else depth))
    h_t = R * (1.0 - np.sin(beta))
    if d <= h_t:
        return float(np.sqrt(max(2.0 * R * d - d * d, 0.0)))
    return float(R * np.cos(beta) + (d - h_t) * np.tan(beta))


def _min_fine_size(cfg):
    return min(cfg.mesh.fine_size_x, cfg.mesh.fine_size_y, cfg.mesh.fine_size_z)


def _target_dt_scale(cfg):
    tgt = float(getattr(cfg.solver, "target_time_increment", 0.0) or 0.0)
    if tgt <= 0.0:
        return 0.0
    return tgt / natural_dt(cfg.material, _min_fine_size(cfg))


def _restore_target_dt(cfg, s):
    if s > 0.0:
        cfg.solver.target_time_increment = s * natural_dt(cfg.material, _min_fine_size(cfg))


def _assert_expected(cfg, expects, campaign):
    for slot, allowed in expects.items():
        obj = getattr(cfg.material, slot, None)
        model = getattr(obj, "MODEL", None)
        if model not in allowed:
            raise ValueError(
                "Campaign '%s' requires material.%s.MODEL in %s but family '%s' "
                "carries '%s'. Run the campaign on a family that matches, or fix "
                "the campaign registration in families.py."
                % (campaign, slot, list(allowed),
                   getattr(cfg.material, "family", "?"), model))


HARDNESS_FACTOR = 2.8         # p_mean ~ HARDNESS_FACTOR * sigma_y0 (Tabor)
MU_CAP = 0.6                  # Briscoe table ceiling, unchanged from base.py
SWEEP_FRICTION_POINTS = 48    # see SWEEP_METHODOLOGY.md, section "Briscoe table density"


def _friction_from(tau0, alpha, mu_cap, p_ref):
    if tau0 <= 1e-9:
        return Friction_Config(mu=alpha)
    return Friction_Config.briscoe(tau0=tau0, alpha=alpha,
                                   p_min=1.0, p_max=max(4.0 * p_ref, 600.0),
                                   n_points=SWEEP_FRICTION_POINTS, mu_cap=mu_cap)


# ----------------------------------------------------------------------
# Campaign 3 -- semicrystalline: linear elastic + Drucker-Prager + G'Sell/Voce
# ----------------------------------------------------------------------

C3_FROZEN = {
    "sigma_y0_ref_MPa": 28.0,
    "nu": 0.42,
    "rho": 0.95e-9,
    "b_voce": 8.0,
    "flow_stress_ratio": 1.0,
    "dilation_angle_deg": 0.0,
    "soft_drop_MPa": 0.0,
    "eps_soft": 0.05,
    "eps_max": 3.0,
    "n_points": 60,
}

C3_FACTORS = [
    Factor("X",      5.0, 60.0, "log", "-",   "Johnson parameter E* tan(theta) / sigma_y0"),
    Factor("h",      0.0, 0.45, "lin", "-",   "G'Sell orientation hardening exp(h eps^2)"),
    Factor("q",      0.0, 0.35, "lin", "-",   "Voce amplitude Q / sigma_y0"),
    Factor("beta",   1.0, 25.0, "lin", "deg", "Drucker-Prager friction angle (1 deg ~ J2)"),
    Factor("mu_eff", 0.05, 0.45, "lin", "-",  "effective friction at scratch pressure"),
    Factor("phi",    0.0, 1.0, "lin", "-",    "alpha / mu_eff, pressure-independent fraction"),
]


def _derive_dp(g, cfg, frozen, with_softening):
    sy = float(frozen["sigma_y0_ref_MPa"])
    nu = float(frozen["nu"])
    tan_att = float(np.tan(np.radians(attack_angle_deg(cfg))))
    E_star = g["X"] * sy / tan_att
    mu_eff, phi = g["mu_eff"], g["phi"]
    alpha = phi * mu_eff
    p_ref = HARDNESS_FACTOR * sy
    tau0 = p_ref * mu_eff * (1.0 - phi)
    out = {"rho": float(frozen["rho"]),
           "E": E_star * (1.0 - nu * nu), "nu": nu, "E_star": E_star,
           "sigma_y0": sy, "X": g["X"], "h": g["h"],
           "Q": g.get("q", 0.0) * sy, "b": float(frozen["b_voce"]),
           "soft_drop": (g["s"] * sy if with_softening else float(frozen["soft_drop_MPa"])),
           "eps_soft": (g["eps_soft"] if with_softening else float(frozen["eps_soft"])),
           "eps_max": float(frozen["eps_max"]), "n_points": float(frozen["n_points"]),
           "friction_angle": g["beta"],
           "flow_stress_ratio": (g["K"] if "K" in g
                                 else float(frozen["flow_stress_ratio"])),
           "dilation_angle": float(frozen["dilation_angle_deg"]),
           "tau0": tau0, "alpha": alpha, "mu_eff": mu_eff, "phi": phi,
           "p_ref_MPa": p_ref, "attack_angle_deg": attack_angle_deg(cfg)}
    return out


# [PATCH:exclusive-post-yield] begin -- exclusivite structurelle des termes post-seuil.
#
# Deux mecanismes post-seuil additifs et antagonistes ne sont pas
# separement identifiables : avec b_voce = 8 (echelle 0.125) et eps_soft
# jusqu'a 0.12, deux couples (q, s) de meme difference q - s donnent des
# tables d'ecrouissage separees par moins de 0.5 % de sigma_y0. Aucune des
# deux campagnes historiques ne les autorise ensemble (C3 gele
# soft_drop_MPa = 0, C4 gele b_voce = 0) et aucune famille calibree ne les
# porte simultanement. La regle est donc une INVARIANTE du modele, pas une
# preference de campagne, et elle est verifiee au point de passage commun.
EXCLUSIVE_PARAM_PAIRS = (
    ("Q", "soft_drop"),
)


def _assert_exclusive(p, campaign="?"):
    """Leve si deux parametres declares mutuellement exclusifs sont actifs."""
    for a, b in EXCLUSIVE_PARAM_PAIRS:
        va = abs(float(p.get(a, 0.0) or 0.0))
        vb = abs(float(p.get(b, 0.0) or 0.0))
        if va > 1e-12 and vb > 1e-12:
            raise ValueError(
                "Campaign '%s': '%s' (=%g) and '%s' (=%g) are mutually "
                "exclusive but both are active. They are antagonistic "
                "additive terms of the same hardening law and are not "
                "separately identifiable; use the signed factor 'w' "
                "instead of independent 'q' and 's'."
                % (campaign, a, va, b, vb))
    return p


# [PATCH:exclusive-post-yield] end
def _apply_dp(cfg, p):
    _assert_exclusive(p)
    cfg.material.rho = p["rho"]
    cfg.material.hyperelastic = LinearElastic_Config(E=p["E"], nu=p["nu"])
    cfg.material.plasticity = DruckerPrager_Config(
        friction_angle=p["friction_angle"],
        flow_stress_ratio=p["flow_stress_ratio"],
        dilation_angle=p["dilation_angle"],
        yield_table=gsell_jonas_table(
            sigma_y0=p["sigma_y0"], h=p["h"], Q=p["Q"], b=p["b"],
            soft_drop=p["soft_drop"], eps_soft=p["eps_soft"],
            eps_max=p["eps_max"], n_points=int(p["n_points"])),
        rate_dependent=None)
    cfg.material.friction = _friction_from(p["tau0"], p["alpha"], MU_CAP, p["p_ref_MPa"])


def _derive_c3(g, cfg):
    return _derive_dp(g, cfg, C3_FROZEN, with_softening=False)


SAMPLING_SEMICRYSTALLINE = FamilySampling(
    campaign="C3_semicrystalline_dp",
    factors=C3_FACTORS,
    frozen=C3_FROZEN,
    derive=_derive_c3,
    apply_fn=_apply_dp,
    expects={"plasticity": ("drucker_prager",), "viscoelastic": ("none",), "damage": ("none",)},
    covers=("semicrystalline_j2", "semicrystalline_dp"),
    label="Semicrystalline: elastic + Drucker-Prager + G'Sell/Voce",
    notes="beta -> 1 deg with K = 1 is the J2 limit, so semicrystalline_j2 is the "
          "low-beta corner of this campaign rather than a separate one.",
)


# ----------------------------------------------------------------------
# Campaign 4 -- glassy: linear elastic + Drucker-Prager + softening + G'Sell
# ----------------------------------------------------------------------

C4_FROZEN = {
    "sigma_y0_ref_MPa": 70.0,
    "nu": 0.37,
    "rho": 1.20e-9,
    "b_voce": 0.0,
    "dilation_angle_deg": 0.0,
    "eps_max": 2.5,
    "n_points": 60,
}

C4_FACTORS = [
    Factor("X",        5.0, 60.0, "log", "-",   "Johnson parameter E* tan(theta) / sigma_y0"),
    Factor("h",        0.0, 0.45, "lin", "-",   "G'Sell orientation hardening exp(h eps^2)"),
    Factor("s",        0.05, 0.35, "lin", "-",  "intrinsic softening drop / sigma_y0"),
    Factor("eps_soft", 0.02, 0.12, "lin", "-",  "softening strain scale"),
    Factor("beta",     8.0, 35.0, "lin", "deg", "Drucker-Prager friction angle"),
    Factor("K",        0.778, 1.0, "lin", "-",  "flow stress ratio (Abaqus lower bound 0.778)"),
    Factor("mu_eff",   0.05, 0.45, "lin", "-",  "effective friction at scratch pressure"),
    Factor("phi",      0.0, 1.0, "lin", "-",    "alpha / mu_eff, pressure-independent fraction"),
]


def _derive_c4(g, cfg):
    return _derive_dp(g, cfg, C4_FROZEN, with_softening=True)


SAMPLING_GLASSY = FamilySampling(
    campaign="C4_glassy_dp",
    factors=C4_FACTORS,
    frozen=C4_FROZEN,
    derive=_derive_c4,
    apply_fn=_apply_dp,
    expects={"plasticity": ("drucker_prager",), "viscoelastic": ("none",), "damage": ("none",)},
    covers=("glassy_dp", "glassy_pc", "glassy_pmma"),
    label="Glassy: elastic + Drucker-Prager + intrinsic softening + G'Sell",
    notes="Q/b frozen at 0 (glassy calibrations use softening + orientation "
          "hardening only), so 'q' is absent from this campaign.",
)


# ----------------------------------------------------------------------
# Campaign DP -- unified Drucker-Prager screening (supersedes C3 + C4)
# ----------------------------------------------------------------------
# One 9-factor box spanning both semicrystalline and glassy behaviour:
# s = 0 removes intrinsic softening (semicrystalline limit), q = 0 removes the
# Voce term (glassy limit), beta -> 1 deg with K = 1 is the J2 limit.

CDP_FROZEN = {
    "sigma_y0_ref_MPa": 50.0,
    "nu": 0.39,
    "rho": 1.10e-9,
    # [PATCH:exclusive-post-yield] b_voce n'est plus gele : il vaut 1/eps_c sur la
    # branche Voce. La valeur reste declaree pour la tracabilite et pour
    # les campagnes qui passent encore par _derive_dp sans aiguillage.
    "b_voce": 8.0,
    "b_voce_note": "superseded by 1/eps_c in CDP; kept for traceability",
    "dilation_angle_deg": 0.0,
    "eps_max": 2.5,
    "n_points": 60,
    "rationale_sigma_y0": "arbitrary: stress-scale invariance makes forces scale exactly",
    "rationale_nu": "first-order effect absorbed by X = E* tan(theta) / sigma_y0",
}

CDP_FACTORS = [
    # [PATCH:cdp-admissible] begin -- bornes restreintes au domaine bien pose.
    # Originaux :
    # Factor("X",        5.0, 60.0, "log", "-",   "Johnson parameter ..."),
    # Factor("eps_c", 0.02, 0.12, "lin", "-",  "strain scale ..."),
    # Factor("phi",      0.0, 1.0, "lin", "-",    "alpha / mu_eff, ..."),
    #
    # X = 5 <=> sigma_y0 / E = 0.136, soit une deformation d'ecoulement de
    # 13.6 % : un elastomere, pas un thermoplastique. Les familles visees se
    # situent entre X ~ 18 (PC) et X ~ 31 (PP) ; la borne 12 laisse de la
    # marge sous les polyolefines souples sans descendre dans un regime ou
    # (a) le sillon ne se forme plus et (b) l'adoucissement rend le probleme
    # mal pose. Voir _assert_post_yield_admissible plus bas.
    Factor("X",       12.0, 60.0, "log", "-",   "Johnson parameter E* tan(theta) / sigma_y0"),
    Factor("h",        0.0, 0.45, "lin", "-",   "G'Sell orientation hardening exp(h eps^2)"),
    # [PATCH:exclusive-post-yield] q / s / eps_soft remplaces par (w, eps_c).
    # Originaux -- q et s sont antagonistes et confondus dans le crochet :
    # Factor("q",        0.0, 0.35, "lin", "-",   "Voce amplitude Q / sigma_y0 (0 = glassy limit)"),
    # Factor("s",        0.0, 0.35, "lin", "-",   "intrinsic softening drop / sigma_y0 (0 = semicrystalline limit)"),
    # Factor("eps_soft", 0.02, 0.12, "lin", "-",  "softening strain scale"),
    Factor("w",     -0.35, 0.35, "lin", "-",  "signed post-yield amplitude / sigma_y0: "
                                             "w < 0 = intrinsic softening (glassy), "
                                             "w > 0 = Voce hardening (semicrystalline)"),
    Factor("eps_c", 0.05, 0.20, "lin", "-",  "strain scale of whichever post-yield "
                                             "branch is active (eps_soft if w < 0, "
                                             "1/b if w > 0)"),
    Factor("beta",     1.0, 35.0, "lin", "deg", "Drucker-Prager friction angle (1 deg ~ J2)"),
    Factor("K",        0.778, 1.0, "lin", "-",  "flow stress ratio (Abaqus lower bound 0.778)"),
    Factor("mu_eff",   0.05, 0.45, "lin", "-",  "effective friction at scratch pressure"),
    # phi = 1 exactement fait basculer _friction_from de la table de Briscoe
    # vers un Friction_Config(mu=alpha) constant : ce n'est pas une variation
    # continue du facteur mais un CHANGEMENT DE CLASSE DE MODELE au bord du
    # domaine, que Morris lit comme un effet elementaire. Le plafond 0.95
    # garde une composante tau0/p residuelle, donc une seule classe de modele
    # sur tout le domaine.
    Factor("phi",      0.0, 0.95, "lin", "-",   "alpha / mu_eff, pressure-independent fraction"),
    # [PATCH:cdp-admissible] end
]


# [PATCH:cdp-admissible] begin -- admissibilite de la branche post-seuil.
#
# Rapport de la pente post-seuil au module elastique :
#
#     |H| / E = |w| * sigma_y0 / (eps_c * E)
#             = |w| * tan(theta) / (X * eps_c * (1 - nu^2))
#
# Sur la branche adoucie (w < 0), |H| / E >= 1 signifie que le module
# tangent local est negatif ET plus raide que la reponse elastique : le
# probleme aux limites perd son caractere bien pose (perte d'ellipticite du
# probleme incremental). La deformation se localise sur la plus petite
# longueur disponible -- ici la taille d'element, faute de longueur interne
# de regularisation -- la solution devient purement dependante du maillage,
# et Abaqus/Explicit avorte sur distorsion.
#
# Le seuil 0.5 n'est PAS une garantie de non-localisation : sans
# regularisation, celle-ci existe des que H < 0. C'est un compromis
# pragmatique qui garde l'adoucissement assez doux pour que la bande
# s'etale sur plusieurs elements et que la reponse globale reste
# exploitable. Documenter la valeur retenue dans PHYSICS_CHOICES.md.
CDP_MAX_POST_YIELD_SLOPE = 0.5


def post_yield_slope_ratio(w, eps_c, X, nu, tan_attack):
    """|H_post-seuil| / E, sans dimension. Voir CDP_MAX_POST_YIELD_SLOPE."""
    eps_c = float(eps_c)
    X = float(X)
    if eps_c <= 0.0 or X <= 0.0:
        return float("inf")
    return abs(float(w)) * float(tan_attack) / (X * eps_c * (1.0 - float(nu) ** 2))


def _assert_post_yield_admissible(g, cfg, frozen, campaign):
    """
    Leve si le point echantillonne sort du domaine bien pose.

    Le garde-fou est place sur le chemin de derivation, PAS seulement dans
    les bornes des facteurs : `--only` gele les facteurs non listes au milieu
    de plage et une edition manuelle des bornes contournerait un simple
    controle de boite.
    """
    ratio = post_yield_slope_ratio(
        g["w"], g["eps_c"], g["X"], float(frozen["nu"]),
        float(np.tan(np.radians(attack_angle_deg(cfg)))))
    if ratio > CDP_MAX_POST_YIELD_SLOPE:
        raise ValueError(
            "Campaign '%s': post-yield slope |H|/E = %.3f exceeds the "
            "admissibility limit %.2f at (X=%.4g, w=%+.4g, eps_c=%.4g). "
            "Below this limit the softening branch stays milder than the "
            "elastic response; above it the local problem is ill-posed, the "
            "strain localises on a single element and Abaqus aborts. Either "
            "raise X, raise eps_c, reduce |w|, or add a regularisation "
            "length (rate dependence / damage energy) before sampling this "
            "corner."
            % (campaign, ratio, CDP_MAX_POST_YIELD_SLOPE,
               g["X"], g["w"], g["eps_c"]))
    return ratio
# [PATCH:cdp-admissible] end


# [PATCH:exclusive-post-yield] begin -- aiguillage exclusif des deux branches post-seuil.
def _split_post_yield(w, eps_c, sy):
    """
    Traduit (w, eps_c) en la paire de termes post-seuil de
    gsell_jonas_table, avec UN SEUL des deux actif.

        w < 0 : branche vitreuse    -> soft_drop = |w| * sy, eps_soft = eps_c,
                                       Q = 0
        w > 0 : branche semi-crist. -> Q = w * sy, b = 1 / eps_c,
                                       soft_drop = 0
        w = 0 : ni l'un ni l'autre (parfaitement plastique + orientation)

    Sur la grille Morris a p = 4 niveaux, w vaut -0.35, -0.1167, +0.1167
    ou +0.35 : w = 0 n'est jamais echantillonne, donc chaque point du plan
    appartient a une branche et a une seule.
    """
    w = float(w)
    eps_c = float(eps_c)
    if eps_c <= 0.0:
        raise ValueError("eps_c must be strictly positive (got %g)" % eps_c)
    if w < 0.0:
        return {"Q": 0.0, "b": 1.0 / eps_c,
                "soft_drop": -w * sy, "eps_soft": eps_c, "branch": -1.0}
    if w > 0.0:
        return {"Q": w * sy, "b": 1.0 / eps_c,
                "soft_drop": 0.0, "eps_soft": eps_c, "branch": 1.0}
    return {"Q": 0.0, "b": 1.0 / eps_c,
            "soft_drop": 0.0, "eps_soft": eps_c, "branch": 0.0}


def _derive_cdp(g, cfg):
    # [PATCH:exclusive-post-yield] original :
    # return _derive_dp(g, cfg, CDP_FROZEN, with_softening=True)
    sy = float(CDP_FROZEN["sigma_y0_ref_MPa"])
    br = _split_post_yield(g["w"], g["eps_c"], sy)
    inner = dict(g)
    inner.pop("w", None)
    inner.pop("eps_c", None)
    inner["s"] = br["soft_drop"] / sy
    inner["eps_soft"] = br["eps_soft"]
    out = _derive_dp(inner, cfg, CDP_FROZEN, with_softening=True)
    out["Q"] = br["Q"]
    out["b"] = br["b"]
    out["w"] = float(g["w"])
    out["eps_c"] = float(g["eps_c"])
    out["branch"] = br["branch"]
    # [PATCH:cdp-admissible] le point doit etre dans le domaine bien pose.
    out["post_yield_slope_ratio"] = _assert_post_yield_admissible(
        g, cfg, CDP_FROZEN, "CDP_drucker_prager_unified")
    return _assert_exclusive(out, "CDP_drucker_prager_unified")
# [PATCH:exclusive-post-yield] end


SAMPLING_DP_UNIFIED = FamilySampling(
    campaign="CDP_drucker_prager_unified",
    factors=CDP_FACTORS,
    frozen=CDP_FROZEN,
    derive=_derive_cdp,
    apply_fn=_apply_dp,
    expects={"plasticity": ("drucker_prager",), "viscoelastic": ("none",), "damage": ("none",)},
    covers=("semicrystalline_j2", "semicrystalline_dp",
            "glassy_dp", "glassy_pc", "glassy_pmma"),
    label="Unified Drucker-Prager (semicrystalline + glassy)",
    # [PATCH:exclusive-post-yield] note reecrite : la coexistence est desormais interdite.
    # notes="s=0 is the semicrystalline corner, q=0 the glassy corner, ...
    notes="w < 0 is the glassy branch (intrinsic softening), w > 0 the "
          "semicrystalline branch (Voce hardening); the two are mutually "
          "exclusive by construction and w = 0 is never sampled on the p=4 "
          "grid. beta=1 deg with K=1 remains the J2 corner. Orientation "
          "hardening h is shared by BOTH branches -- it is present in C3 and "
          "C4 alike and both calibrated glassy families carry h > 0 together "
          "with softening, so h is never exclusive with w.",
)


CAMPAIGNS = {
    "semicrystalline_dp": SAMPLING_DP_UNIFIED,
    "glassy_pc": SAMPLING_DP_UNIFIED,
}

# Superseded by SAMPLING_DP_UNIFIED but kept importable for the split campaigns.
LEGACY_CAMPAIGNS = {
    "semicrystalline_dp": SAMPLING_SEMICRYSTALLINE,
    "glassy_pc": SAMPLING_GLASSY,
}


def get_campaign(family_key):
    if family_key not in CAMPAIGNS:
        raise ValueError(
            "Family '%s' carries no sampling campaign. This module covers the "
            "Drucker-Prager families only; hosts are %s. The other DP families "
            "are corners of the same box (see SWEEP_METHODOLOGY.md)."
            % (family_key, ", ".join(sorted(CAMPAIGNS))))
    return CAMPAIGNS[family_key]