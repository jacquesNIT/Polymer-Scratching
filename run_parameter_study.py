# Unified driver for every scratch-test parameter study.
#
# " abaqus cae noGUI=run_parameter_study.py -- single "
# " abaqus cae noGUI=run_parameter_study.py -- mesh "
# " abaqus cae noGUI=run_parameter_study.py -- mass_scale "
# " abaqus cae noGUI=run_parameter_study.py -- material "

import sys
import os
import shutil

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    sys.path.insert(0, os.path.abspath('..'))

from ScratchSimulation.AbaqusModel.abaqus_env import *          
from ScratchSimulation.AbaqusModel.Configuration import Simulation_Config
from ScratchSimulation.AbaqusModel.Configuration import get_family
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

def run_parameter_study(study, base_cfg=None, family=None, job_name=None,
                         output_subdir="SimDataOutputs", move_exts=(".sta", ".odb")):


    cfg = base_cfg or get_family(family or DEFAULT_FAMILY).build_config()
    cfg.job_name = job_name or study.name
    if study.configure:
        study.configure(cfg)

    run_dir = os.path.join("runs", study.name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    os.chdir(run_dir)

    if output_subdir and not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    n_total = len(study.cases)
    for i, case in enumerate(study.cases, start=1):

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

    cleanup_abaqus_junk()



# Study definitions (thin: just cases + how to apply + how to name).
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



# Defaults + selection.
DEFAULT_FAMILY = "semicrystalline_j2" 
DEFAULT_MESH_SIZES = [
    #[0.04, 0.04, 0.04],
    #[0.03, 0.03, 0.03],
    #[0.02, 0.02, 0.02],
    #[0.015, 0.015, 0.015],
    [0.01, 0.01, 0.01],
]
DEFAULT_MASS_SCALES = [300, 200 ,100]
DEFAULT_STUDY = "mass_scale"

# def _load_material_parameters():
#    from ScratchSimulation.mixed_material_parameter_sweep import parameters
#    return parameters

STUDIES = {
    "single":     lambda: single_study(),
    "mesh":       lambda: mesh_study(DEFAULT_MESH_SIZES),
    "mass_scale": lambda: mass_scale_study(DEFAULT_MASS_SCALES),
    # "material":   lambda: material_study(_load_material_parameters()),
}

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
    # Optional second token after "--": study then family.
    #   abaqus cae noGUI=run_parameter_study.py -- single semicrystalline_j2
    argv = sys.argv
    if "--" in argv:
        rest = argv[argv.index("--") + 1:]
        if len(rest) >= 2:
            return rest[1]
    return default

if __name__ == "__main__":
    name = _selected_study_name()
    if name not in STUDIES:
        raise SystemExit("Unknown study '%s'." % (name))
    run_parameter_study(STUDIES[name](), family=_selected_family())