# Unified driver for every scratch-test parameter study.
#
# " abaqus cae noGUI=run_parameter_study.py -- single "
# " abaqus cae noGUI=run_parameter_study.py -- mesh "
# " abaqus cae noGUI=run_parameter_study.py -- mass_scale "
# " abaqus cae noGUI=run_parameter_study.py -- material "
# " abaqus cae noGUI=run_parameter_study.py -- friction "
#
# JOB-SPLITTING TOKENS (after '--', any order; used by launch_cluster_jobs.py):
#   <study> [family] [i/N] [cpus=K] [tag=NAME]
#     i/N     -> run only the round-robin slice cases[i::N], in an ISOLATED
#                directory runs/<name>_c{i}of{N} with a unique job name
#                (mandatory for concurrent SLURM jobs: .odb/.sta/.lck collide
#                otherwise).
#     cpus=K  -> overrides solver.num_cpus/num_domains.
#     tag=X   -> free suffix (same study launched twice without collision).
# Example: abaqus cae noGUI=run_parameter_study.py -- material elastomer_mr 3/8

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


def run_parameter_study(study, base_cfg=None, family=None, job_name=None,
                         output_subdir="SimDataOutputs", move_exts=(".sta", ".odb"),
                         chunk=None, cpus=None, tag=None):
    """
    chunk=(i, N) runs only the round-robin slice cases[i::N] inside an
    isolated directory (concurrent-job safe); cpus overrides the solver CPU
    count; tag adds a free suffix.
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
        cfg.mesh.fine_size_x = s[0]
        cfg.mesh.fine_size_y = s[1]
        cfg.mesh.fine_size_z = s[2]
        cfg.mesh.coarse_size_0 = 2*s[0]
        cfg.mesh.coarse_size_1 = 4*s[0]
        cfg.mesh.coarse_size_2 = 8*s[0]
    return ParameterStudy(
        name="MeshConvergence",
        cases=sizes,
        apply_case=apply,
        label=lambda s: "Mesh_%s_%s_%s" % (s[0], s[1], s[2]),
        configure=lambda cfg: setattr(cfg.solver, "use_ALE", True),
    )


def mass_scale_study(scales):
    return ParameterStudy(
        name="MassScaleConvergence",
        cases=scales,
        apply_case=lambda cfg, ms: setattr(cfg.solver, "mass_scale", ms),
        label=lambda ms: "MassScale%s" % ms,
        configure=lambda cfg: setattr(cfg.solver, "use_ALE", True),
    )

def friction_study(mu_values):
    def apply(cfg, mu):
        cfg.material.friction.mu = mu
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

# Defaults + selection.
DEFAULT_FAMILY = "elastomer_mr" 
DEFAULT_MESH_SIZES = [
    [0.04, 0.04, 0.04],
    [0.03, 0.03, 0.03],
    [0.02, 0.02, 0.02],
    [0.015, 0.015, 0.015],
    #[0.01, 0.01, 0.01],
]
DEFAULT_MASS_SCALES = [5000]
DEFAULT_MU_VALUES = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
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
    "friction":   lambda: friction_study(DEFAULT_MU_VALUES),
    "material":   lambda: material_study(_load_material_parameters()),
    "models":     lambda: model_study(),
}

def _parse_cli(argv, default_study=DEFAULT_STUDY, default_family=DEFAULT_FAMILY):
    """
    Tokens after '--' in any order: <study> [family] [i/N] [cpus=K] [tag=X].
    Without '--', only study names are scanned in argv (legacy behaviour).
    """
    out = {"study": default_study, "family": default_family,
           "chunk": None, "cpus": None, "tag": None}
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
                        tag=cli["tag"])