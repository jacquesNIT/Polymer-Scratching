# Mesh convergence study
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
cfg.job_name = "MeshConvergence"

#  Mesh sizes to test  
mesh_sizes = [
    # [size_x,  size_y,  size_z]
    [0.060, 0.060, 0.060],
    [0.040, 0.040, 0.040],
    [0.03, 0.03, 0.03],
    [0.02, 0.02, 0.02],
    [0.015, 0.015, 0.015],
    [0.01, 0.01, 0.01],
    #[0.007, 0.007, 0.007],
    #[0.005, 0.005, 0.005],
]

#  Working directory
run_dir = os.path.join("runs", "MeshConvergence")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
os.chdir(run_dir)

#  Loop over mesh sizes
n_total = len(mesh_sizes)
for i, mesh_size in enumerate(mesh_sizes, start=1):

    file_name = "Mesh_%s_%s_%s" % (mesh_size[0], mesh_size[1], mesh_size[2])

    cfg.mesh.fine_size_x = mesh_size[0]
    cfg.mesh.fine_size_y = mesh_size[1]
    cfg.mesh.fine_size_z = mesh_size[2]

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

    print(">>> [%d/%d] mesh=%.3f mm — done." % (i, n_total, mesh_size[0]))

cleanup_abaqus_junk()
