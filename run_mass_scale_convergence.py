# Mass-scaling convergence study.

import sys
import os

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    sys.path.insert(0, os.path.abspath('..'))

from abaqus import *
from abaqusConstants import *
import shutil

from ScratchSimulation.AbaqusModel.Configuration import Simulation_Config
from ScratchSimulation.AbaqusModel.Simulation import build_scratch_model
from ScratchSimulation.AbaqusModel.Material import SubstrateMaterialAssignment
from ScratchSimulation.AbaqusModel.Postprocessing import post_process
from ScratchSimulation.AbaqusModel.utils import run_job_and_wait, cleanup_abaqus_junk

#  Configuration
cfg = Simulation_Config.polymer_default()
cfg.job_name = "MassScaleConvergence"

# ALE use
cfg.solver.use_ALE = True

#  Mass-scaling factors to test
mass_scales = [2000,1000,500,200,100]

#  Working directory
run_dir = os.path.join("runs", "MassScaleConvergence")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
os.chdir(run_dir)

#  Loop
n_total = len(mass_scales)
for i, ms in enumerate(mass_scales, start=1):

    file_name = "MassScale%s" % ms

    cfg.solver.mass_scale = ms

    model, substrate_part = build_scratch_model(cfg)

    material = SubstrateMaterialAssignment(model, substrate_part, cfg)
    material.apply()

    run_job_and_wait(cfg.job_name, cfg)
    post_process(cfg.job_name, file_name, cfg)

    mdb.close()
    for ext in [".sta", ".odb"]:
        src = cfg.job_name + ext
        dst = os.path.join("SimDataOutputs", file_name + ext)
        if os.path.exists(src):
            shutil.move(src, dst)

    print(">>> [%d/%d] mass_scale=%s — done." % (i, n_total, ms))

cleanup_abaqus_junk()
