# Run a single scratch test simulation.
import sys
import os

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    sys.path.insert(0, os.path.abspath('..'))

from abaqus import *
from abaqusConstants import *

from ScratchSimulation.AbaqusModel.Configuration import Simulation_Config
from ScratchSimulation.AbaqusModel.Simulation import build_scratch_model
from ScratchSimulation.AbaqusModel.Material import SubstrateMaterialAssignment
from ScratchSimulation.AbaqusModel.Postprocessing import post_process
from ScratchSimulation.AbaqusModel.utils import run_job_and_wait, cleanup_abaqus_junk

#  Configuration
cfg = Simulation_Config.polymer_default()
cfg.job_name = "SingleScratch"

#  Working directory
run_dir = os.path.join("runs", cfg.job_name)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
os.chdir(run_dir)

#  Build model, assign material
model, substrate_part = build_scratch_model(cfg)
material = SubstrateMaterialAssignment(model, substrate_part, cfg)
material.apply()

# run, post-process
run_job_and_wait(cfg.job_name, cfg)
post_process(cfg.job_name, cfg.job_name, cfg)
mdb.close()
cleanup_abaqus_junk()