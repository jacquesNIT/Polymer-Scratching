# Unified driver for every scratch-test parameter study.
#
# " abaqus cae noGUI=run_parameter_study.py -- single "
# " abaqus cae noGUI=run_parameter_study.py -- mesh "
# " abaqus cae noGUI=run_parameter_study.py -- mass_scale "
# " abaqus cae noGUI=run_parameter_study.py -- material "
# " abaqus cae noGUI=run_parameter_study.py -- friction "
# " abaqus cae noGUI=run_parameter_study.py -- design <family> "
#
# JOB-SPLITTING TOKENS (after '--', any order; used by launch_cluster_jobs.py):
#   <study> [family] [i/N] [cpus=K] [tag=NAME]
#     i/N     -> run only the round-robin slice cases[i::N], in an ISOLATED
#                directory runs/<name>_c{i}of{N} with a unique job name
#                (mandatory for concurrent SLURM jobs: .odb/.sta/.lck collide
#                otherwise).
#     cpus=K  -> overrides solver.num_cpus/num_domains.
#     tag=X   -> free suffix (same study launched twice without collision).
#     set:PATH=VALUE -> per-job config override on any dotted cfg attribute,
#                applied AFTER the study's configure() so the per-job choice
#                wins (e.g. re-enable ALE on a mesh study). VALUE is parsed
#                as bool (true/false/on/off) / int / float, else kept as a
#                string. Combine with tag= to isolate the variants:
#                run dir runs/<Study>_<family>_<tag>, distinct job name.
#                NB: fields varied per-case by the study itself (apply_case)
#                are re-set at every case and cannot be overridden this way.
# Example: abaqus cae noGUI=run_parameter_study.py -- material elastomer_mr 3/8
# Example: abaqus cae noGUI=run_parameter_study.py -- mesh glassy_pc tag=mesh2 set:solver.use_ALE=True set:scratch.scratch_time=0.05

import sys
import os
import shutil

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:          # Abaqus CAE noGUI may not define __file__
    _HERE = os.path.abspath('.')
sys.path.insert(0, os.path.dirname(_HERE))

from ScratchSimulation.AbaqusModel.abaqus_env import *          
from ScratchSimulation.AbaqusModel.Configuration import Simulation_Config
from ScratchSimulation.AbaqusModel.Configuration import get_family
from ScratchSimulation.AbaqusModel.Configuration import matched_hyperelastic_set
from ScratchSimulation.AbaqusModel.Configuration import gsell_jonas_table
from ScratchSimulation.AbaqusModel.Configuration import natural_dt
from ScratchSimulation.AbaqusModel.Simulation import build_scratch_model
from ScratchSimulation.AbaqusModel.Material import SubstrateMaterialAssignment
from ScratchSimulation.AbaqusModel.Postprocessing import post_process
from ScratchSimulation.AbaqusModel.utils import run_job_and_wait, cleanup_abaqus_junk

class ParameterStudy(object):
    def __init__(self, name, cases, apply_case, label, configure=None):
        self.name = name
        self.cases = list(cases)
        self.apply_case = apply_case
        self.label = label
        self.configure = configure

def _makedirs_safe(path):
    # Race-safe mkdir -p (Python 2 has no exist_ok): concurrent SLURM jobs
    # starting together both try to create "runs/", and the loser used to
    # crash with OSError before its os.chdir -- leaving it in the repo root so
    # Abaqus wrote/looked for the .inp in the wrong place ("could not be
    # located"). Tolerating an already-existing directory removes the race.
    if path and not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise


def _parse_override_value(text):
    # bool / int / float auto-detection; falls back to the raw string.
    low = text.strip().lower()
    if low in ("true", "on", "yes"):
        return True
    if low in ("false", "off", "no"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _apply_overrides(cfg, overrides):
    # Per-job config overrides from set:PATH=VALUE tokens. Applied AFTER
    # study.configure() so the per-job choice wins over the study default
    # (e.g. ALE forced off by mesh_study can be re-enabled for one job).
    # Fails loudly on a typo instead of silently creating a dead attribute.
    for path, raw in (overrides or []):
        obj = cfg
        parts = path.split(".")
        for name in parts[:-1]:
            if not hasattr(obj, name):
                raise SystemExit("Override '%s=%s': cfg has no attribute '%s'."
                                 % (path, raw, name))
            obj = getattr(obj, name)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise SystemExit("Override '%s=%s': '%s' has no attribute '%s'."
                             % (path, raw, type(obj).__name__, leaf))
        value = _parse_override_value(raw)
        setattr(obj, leaf, value)
        if path == "solver.num_cpus":     # keep MPI domains consistent
            cfg.solver.num_domains = int(value)
        print(">>> Override: cfg.%s = %r" % (path, value))


def run_parameter_study(study, base_cfg=None, family=None, job_name=None,
                         output_subdir="SimDataOutputs", move_exts=(".sta", ".odb"),
                         chunk=None, cpus=None, tag=None, overrides=None):
    """
    chunk=(i, N) runs only the round-robin slice cases[i::N] inside an
    isolated directory (concurrent-job safe); cpus overrides the solver CPU
    count; tag adds a free suffix; overrides is a list of (dotted_path,
    raw_value) pairs applied by _apply_overrides() after study.configure().
    """

    cfg = base_cfg or get_family(family or DEFAULT_FAMILY).build_config()

    suffix = ""
    if tag:
        suffix += "_" + str(tag)

    cases = list(study.cases)
    n_global = len(cases)
    if chunk is not None:
        ci, cn = int(chunk[0]), int(chunk[1])
        if not (0 <= ci < cn):
            raise SystemExit("Invalid chunk %d/%d (need 0 <= i < N)." % (ci, cn))
        cases = cases[ci::cn]          # round-robin: balances heterogeneous case costs
        suffix += "_c%03dof%03d" % (ci, cn)
        print(">>> Chunk %d/%d: %d of %d cases (round-robin slice [%d::%d])."
              % (ci, cn, len(cases), n_global, ci, cn))
    if not cases:
        raise SystemExit("Chunk %s selects no case (study has %d cases)." % (chunk, n_global))

    # Family is part of the run directory AND the job name: two studies that
    # differ only by family (e.g. "mesh glassy_pc" vs "mesh elastomer_mr")
    # produce identical case labels, so without this they would overwrite each
    # other's CSVs in a shared runs/<study>/ folder.
    fam_tag = ("_" + str(family)) if family else ""
    cfg.job_name = (job_name or study.name) + fam_tag + suffix
    if study.configure:
        study.configure(cfg)

    _apply_overrides(cfg, overrides)

    if cpus:
        cfg.solver.num_cpus = int(cpus)
        cfg.solver.num_domains = int(cpus)
        print(">>> solver.num_cpus overridden to %d." % int(cpus))

    run_dir = os.path.join("runs", study.name + fam_tag + suffix)
    _makedirs_safe(run_dir)
    os.chdir(run_dir)
    run_dir_abs = os.path.abspath(os.getcwd())

    if output_subdir:
        _makedirs_safe(output_subdir)

    n_total = len(cases)
    for i, case in enumerate(cases, start=1):

        study.apply_case(cfg, case)
        stem = study.label(case)

        model, substrate_part = build_scratch_model(cfg)
        SubstrateMaterialAssignment(model, substrate_part, cfg).apply()



        run_job_and_wait(cfg.job_name, cfg)
        post_process(cfg.job_name, stem, cfg)

        mdb.close()   

        if output_subdir and stem != cfg.job_name:
            for ext in move_exts:
                src = cfg.job_name + ext
                if os.path.exists(src):
                    shutil.move(src, os.path.join(output_subdir, stem + ext))

        print(">>> [%d/%d] %s -> %s done." % (i, n_total, study.name, stem))

    cleanup_abaqus_junk(base_dir=run_dir_abs)



# Study definitions 
def single_study():
    return ParameterStudy(
        name="SingleScratch",
        cases=[None],
        apply_case=lambda cfg, _case: None,
        label=lambda _case: "SingleScratch",   
    )


def mesh_study(sizes):
    def apply(cfg, s):
        s_target = (cfg.solver.target_time_increment/ natural_dt(cfg.material, cfg.mesh.fine_size_x))   
        cfg.mesh.fine_size_x = s[0]
        cfg.mesh.fine_size_y = s[1]
        cfg.mesh.fine_size_z = s[2]
        cfg.solver.target_time_increment = (s_target * natural_dt(cfg.material, cfg.mesh.fine_size_x))      # For adaptative Mass Scaling with mesh size
    return ParameterStudy(
        name="MeshConvergence",
        cases=sizes,
        apply_case=apply,
        label=lambda s: "Mesh_%s_%s_%s" % (s[0], s[1], s[2]),
        configure=lambda cfg: setattr(cfg.solver, "use_ALE", False),
    )


def mass_scale_study(scales):
    return ParameterStudy(
        name="MassScaleConvergence",
        cases=scales,
        apply_case=lambda cfg, ms: setattr(cfg.solver, "mass_scale", ms),
        label=lambda ms: "MassScale%s" % ms,
        configure=lambda cfg: setattr(cfg.solver, "use_ALE", False),
    )


def target_dt_study(s_values):
    """
    Variable-mass-scaling convergence sweep: one scratch per target-dt scale
    factor s, with

        solver.target_time_increment = s * dt_nat(material, L_min)

    where dt_nat is the a-priori stable increment (Configuration.natural_dt)
    of the FAMILY's material at the fine mesh size, so the same s maps to a
    different absolute target per family (elastomers: K = 2/D1 governs;
    glassy/semicrystalline: M(E, nu)). Modelbuilder switches to
    SEMI_AUTOMATIC / THROUGHOUT_STEP / BELOW_MIN variable mass scaling
    (substrate scope) as soon as target_time_increment > 0.

    s <= 0 runs the BASELINE: fixed mass scaling with the family's default
    mass_scale (exactly what production runs use today) -- the reference case
    for the RF / energy / residual-profile comparison. Note the equivalence
    dt_fixed ~ sqrt(mass_scale) * dt_nat: with mass_scale = 500 the baseline
    already sits at s ~ 22, hence the default ladder brackets it.

    ALE forced on, mirroring mesh/mass_scale studies. Run e.g.:
        abaqus cae noGUI=run_parameter_study.py -- target_dt glassy_pc
    """
    def apply(cfg, s):
        L_min = min(cfg.mesh.fine_size_x, cfg.mesh.fine_size_y,
                    cfg.mesh.fine_size_z)
        dt_nat = natural_dt(cfg.material, L_min)
        s = float(s)
        if s > 0.0:
            cfg.solver.target_time_increment = s * dt_nat
            print(">>> target_dt: s=%g, dt_nat=%.3e s -> target dt=%.3e s "
                  "(variable mass scaling, substrate scope)"
                  % (s, dt_nat, cfg.solver.target_time_increment))
        else:
            cfg.solver.target_time_increment = 0.0   # fixed-factor baseline
            print(">>> target_dt: BASELINE fixed mass_scale=%g "
                  "(dt_nat=%.3e s, equivalent fixed dt ~ %.3e s, s_eq ~ %.1f)"
                  % (cfg.solver.mass_scale, dt_nat,
                     dt_nat * cfg.solver.mass_scale ** 0.5,
                     cfg.solver.mass_scale ** 0.5))

    return ParameterStudy(
        name="TargetDtConvergence",
        cases=s_values,
        apply_case=apply,
        label=lambda s: "TargetDt_s%g" % float(s),
        configure=lambda cfg: setattr(cfg.solver, "use_ALE", False),
    )


def friction_study(mu_values):
    def apply(cfg, mu):
        fric = cfg.material.friction
        if getattr(fric, "slip_rate_dependent", False):
            raise ValueError(
                "friction_study cannot sweep a slip-rate-dependent friction "
                "table (family '%s'): setting friction.mu leaves the table, "
                "and therefore the Abaqus input, unchanged."
                % getattr(cfg.material, "family", "?"))
        if getattr(fric, "pressure_dependent", False):
            fric.set_briscoe_alpha(mu)
            print(">>> friction: Briscoe alpha=%g (tau0=%g), mu(p) table rebuilt, "
                  "mu in [%.4f, %.4f]"
                  % (fric.briscoe_params["alpha"], fric.briscoe_params["tau0"],
                     min(r[0] for r in fric.mu_table),
                     max(r[0] for r in fric.mu_table)))
        else:
            fric.mu = mu
            print(">>> friction: constant Coulomb mu=%g" % mu)

    return ParameterStudy(
        name="Friction",
        cases=mu_values,
        apply_case=apply,
        label=lambda mu: "Mu_%s" % mu,
    )


def material_study(parameters):
    def apply(cfg, p):
        cfg.material.rho = p["rho"]
        cfg.material.hyperelastic.C10 = p["C10"]
        cfg.material.hyperelastic.C01 = p["C01"]
        cfg.material.hyperelastic.D1 = p["D1"]
        cfg.material.friction.mu = p["mu"]
    return ParameterStudy(
        name="MaterialSweep",
        cases=parameters,
        apply_case=apply,
        label=lambda p: "Material_%s" % p["id"],
    )

# Screening / sweep design produced by generate_design.py. Resolved relative to
# THIS script because run_parameter_study() chdirs into runs/ afterwards.
DEFAULT_DESIGN_DIR = "designs"


def _design_path(family_key, csv_path=None):
    if csv_path:
        return csv_path
    for method in ("morris", "sobol"):
        p = os.path.join(_HERE, DEFAULT_DESIGN_DIR, "%s_%s.csv" % (family_key, method))
        if os.path.exists(p):
            return p
    raise SystemExit(
        "No design file for family '%s' in %s (expected <family>_morris.csv or "
        "<family>_sobol.csv). Generate it with:\n"
        "    python3 generate_design.py %s"
        % (family_key, os.path.join(_HERE, DEFAULT_DESIGN_DIR), family_key))


def _load_design(family_key, csv_path=None):
    # Reads the g_* (dimensionless group) columns only: the material parameters
    # are re-derived in-kernel by families.py so the CSV cannot drift from it.
    import csv as _csv
    path = _design_path(family_key, csv_path)
    with open(path, "r") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    rows = []
    for rec in _csv.DictReader(lines):
        groups = {}
        for k, v in rec.items():
            if k and k.startswith("g_") and v not in (None, ""):
                groups[k[2:]] = float(v)
        if not groups:
            raise SystemExit("Design %s has no g_* column." % path)
        rows.append({"id": rec["id"], "groups": groups})
    print(">>> Loaded %d design points from %s" % (len(rows), path))
    return rows


def design_study(family_key, csv_path=None):
    """
    Screening / sweep driven by a design generated with generate_design.py.
    Every case is applied through the family's own sampling campaign, which
    validates the model form and rebuilds the yield table, the friction model
    and the target time increment. Run it as:
        abaqus cae noGUI=run_parameter_study.py -- design glassy_pc
    """
    fam = get_family(family_key)
    spec = getattr(fam, "sampling", None)
    if spec is None:
        raise SystemExit(
            "Family '%s' carries no sampling campaign. Campaign hosts are the "
            "families with sampling=... in families.py; the other families of "
            "the same model class are covered by their host's ranking."
            % family_key)
    cases = _load_design(family_key, csv_path)
    print(">>> Campaign %s (%s): %d factors %s"
          % (spec.campaign, spec.label, spec.dim, spec.names))

    def apply(cfg, case):
        spec.configure(cfg, case["groups"])

    return ParameterStudy(
        name="DesignSweep",
        cases=cases,
        apply_case=apply,
        label=lambda case: "Design_%s" % case["id"],
    )


def model_study(mu0=2.2, K_mu=55.0):
    """
    Hyperelastic model-form comparison: one scratch per constitutive model (Mooney-Rivlin, Arruda-Boyce, Yeoh, Ogden N=2), 
    mu_0, K_0 = K_mu*mu_0 matched by matched_hyperelastic_set(). 

    Only meaningful on hyperelastic families (elastomer_mr / elastomer_ve)
    """
    model_set = matched_hyperelastic_set(mu0=mu0, K_mu=K_mu)

    def apply(cfg, case):
        _label, he_cfg = case
        if cfg.material.plasticity.MODEL != "none":
            raise ValueError(
                "model_study swaps hyperelastic bases: family '%s' carries "
                "plasticity '%s' (forbidden combination in Abaqus). Run it on "
                "elastomer_mr / elastomer_ve."
                % (getattr(cfg.material, "family", "?"), cfg.material.plasticity.MODEL))
        cfg.material.hyperelastic = he_cfg

    return ParameterStudy(
        name="ModelComparison",
        cases=list(model_set),
        apply_case=apply,
        label=lambda case: "Model_%s" % case[0],
    )


def depth_study(depths):
    """
    Scratch-depth sweep: one scratch per prescribed penetration depth.
    Each case sets cfg.scratch.scratch_depth (in mm; negative = into the
    surface, matching Scratch_Config). ALE is forced on like the mesh/
    mass-scale studies, because deeper grooves drive larger element
    distortion in the contact zone.
    """
    return ParameterStudy(
        name="DepthSweep",
        cases=depths,
        apply_case=lambda cfg, d: setattr(cfg.scratch, "scratch_depth", d),
        label=lambda d: "Depth_%s" % d,
        configure=lambda cfg: setattr(cfg.solver, "use_ALE", False),
    )


# Base G'Sell-Jonas parameters held FIXED while h is swept -- only the
# orientation-hardening term exp(h*eps_p^2) changes between cases. These match
# the rigid semicrystalline (semicrystalline_j2) calibration; edit them to fit
# the family you actually run the study on.
GSELL_SIGMA_Y0  = 28.0    # initial yield stress [MPa]
GSELL_Q         = 5.0     # Voce initial-hardening amplitude [MPa]
GSELL_B         = 8.0     # Voce initial-hardening rate [-]
GSELL_SOFT_DROP = 0.0     # intrinsic post-yield softening [MPa] (0 = off)
GSELL_EPS_SOFT  = 0.05    # softening strain scale [-]
GSELL_EPS_MAX   = 3.0     # max plastic strain tabulated [-]
GSELL_N_POINTS  = 60      # points in the yield table


def gsell_h_study(h_values):
    """
    G'Sell strain-hardening sweep: one scratch per h in the G'Sell reference
    table (h = 0, 0.22, 0.45). Each case regenerates the J2 yield table with
    gsell_jonas_table(), holding sigma_y0/Q/b/... fixed (see GSELL_* above) and
    varying only the orientation-hardening term exp(h*eps_p^2) -- the term
    Bucaille et al. tie to pile-up and scratch resistance.

    Only meaningful on a family whose plasticity exposes a yield_table
    (palier 2, e.g. semicrystalline_j2). Run it as:
        abaqus cae noGUI=run_parameter_study.py -- gsell_h semicrystalline_j2
    """
    def apply(cfg, h):
        pl = cfg.material.plasticity
        if not hasattr(pl, "yield_table"):
            raise ValueError(
                "gsell_h_study rewrites the J2 yield table but family '%s' "
                "plasticity (%s) exposes no yield_table. Run it on a palier-2 "
                "family such as semicrystalline_j2."
                % (getattr(cfg.material, "family", "?"),
                   getattr(pl, "MODEL", type(pl).__name__)))
        pl.yield_table = gsell_jonas_table(
            sigma_y0=GSELL_SIGMA_Y0, h=h, Q=GSELL_Q, b=GSELL_B,
            soft_drop=GSELL_SOFT_DROP, eps_soft=GSELL_EPS_SOFT,
            eps_max=GSELL_EPS_MAX, n_points=GSELL_N_POINTS)

    return ParameterStudy(
        name="GSellHardening",
        cases=h_values,
        apply_case=apply,
        label=lambda h: "GSell_h_%s" % h,
    )


# Defaults + selection.
DEFAULT_FAMILY = "elastomer_mr" 
DEFAULT_MESH_SIZES = [
    #[0.04, 0.04, 0.04],
    #[0.03, 0.03, 0.03],
    [0.02, 0.02, 0.02],
    [0.015, 0.015, 0.015],
    [0.01, 0.01, 0.01],
    [0.007, 0.007, 0.007],
    [0.005, 0.005, 0.005],
]
DEFAULT_MASS_SCALES = [5000, 2000, 1000, 500]
DEFAULT_DT_SCALES = [30, 40, 80] # NB : For base MS = 500, sqrt(500) = 22, need more than ~20 to make a difference
DEFAULT_MU_VALUES = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
DEFAULT_DEPTHS = [-40e-3, -60e-3, -80e-3, -100e-3] 
DEFAULT_GSELL_H = [0.0, 0.11, 0.22, 0.33, 0.45] # For running (4-5h)
DEFAULT_STUDY = "single"

# QMC material sweep produced by MR_parameter_sampling.py, loaded from the
# CSV (no pandas in the Abaqus kernel); resolved relative to THIS script
# because run_parameter_study() chdirs into runs/ afterwards.
DEFAULT_SWEEP_CSV = os.path.join("material_parameters",
                                 "polymer_MR_material_parameter_sweep.csv")

def _load_material_parameters(csv_path=None):
    import csv as _csv
    path = csv_path or os.path.join(_HERE, DEFAULT_SWEEP_CSV)
    if not os.path.exists(path):
        raise SystemExit("Material sweep file not found: %s "
                         "(generate it with MR_parameter_sampling.py)" % path)
    rows = []
    with open(path, "r") as f:
        for r in _csv.DictReader(f):
            d = {"id": r["id"]}
            for k in ("rho", "nu", "C10", "C01", "D1", "mu", "r", "r_K"):
                if k in r and r[k] not in (None, ""):
                    d[k] = float(r[k])
            rows.append(d)
    print(">>> Loaded %d material-sweep cases from %s" % (len(rows), path))
    return rows

STUDIES = {
    "single":     lambda: single_study(),
    "mesh":       lambda: mesh_study(DEFAULT_MESH_SIZES),
    "mass_scale": lambda: mass_scale_study(DEFAULT_MASS_SCALES),
    "target_dt":  lambda: target_dt_study(DEFAULT_DT_SCALES),
    "friction":   lambda: friction_study(DEFAULT_MU_VALUES),
    "material":   lambda: material_study(_load_material_parameters()),
    "design":     lambda: design_study(_selected_family()),
    "models":     lambda: model_study(),
    "depth":      lambda: depth_study(DEFAULT_DEPTHS),
    "gsell_h":    lambda: gsell_h_study(DEFAULT_GSELL_H),
}

def _parse_cli(argv, default_study=DEFAULT_STUDY, default_family=DEFAULT_FAMILY):
    """
    Tokens after '--' in any order:
        <study> [family] [i/N] [cpus=K] [tag=X] [set:PATH=VALUE ...]
    Without '--', only study names are scanned in argv (legacy behaviour).
    """
    out = {"study": default_study, "family": default_family,
           "chunk": None, "cpus": None, "tag": None, "overrides": []}
    if "--" in argv:
        rest = argv[argv.index("--") + 1:]
    else:
        rest = [a for a in argv[1:] if a in STUDIES]

    positional = []
    for tok in rest:
        if tok.startswith("cpus="):
            out["cpus"] = int(tok.split("=", 1)[1])
        elif tok.startswith("tag="):
            out["tag"] = tok.split("=", 1)[1]
        elif tok.startswith("set:"):
            body = tok[4:]
            if "=" not in body:
                raise SystemExit("Bad override token '%s' "
                                 "(expected set:path=value)." % tok)
            path, val = body.split("=", 1)
            out["overrides"].append((path, val))
        elif tok.count("/") == 1 and all(p.isdigit() for p in tok.split("/")):
            i, n = tok.split("/")
            out["chunk"] = (int(i), int(n))
        else:
            positional.append(tok)
    if positional:
        out["study"] = positional[0]
    if len(positional) > 1:
        out["family"] = positional[1]
    return out


def _selected_study_name(default=DEFAULT_STUDY):
    argv = sys.argv
    if "--" in argv:
        rest = argv[argv.index("--") + 1:]
        if rest:
            return rest[0]
    for a in argv[1:]:
        if a in STUDIES:
            return a
    return default

def _selected_family(default=DEFAULT_FAMILY):
    # Optional second token after "--"
    argv = sys.argv
    if "--" in argv:
        rest = argv[argv.index("--") + 1:]
        if len(rest) >= 2:
            return rest[1]
    return default

if __name__ == "__main__":
    cli = _parse_cli(sys.argv)
    if cli["study"] not in STUDIES:
        raise SystemExit("Unknown study '%s'. Available: %s"
                         % (cli["study"], ", ".join(sorted(STUDIES))))
    run_parameter_study(STUDIES[cli["study"]](),
                        family=cli["family"],
                        chunk=cli["chunk"],
                        cpus=cli["cpus"],
                        tag=cli["tag"],
                        overrides=cli["overrides"])