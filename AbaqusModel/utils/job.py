# Abaqus job submission and wait.

from ScratchSimulation.AbaqusModel.abaqus_env import *
import os

def run_job_and_wait(job_name, cfg):

    solver = cfg.solver

    # Remove stale job from mdb to avoid "name already exists" error on rerun
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]

    # Remove .lck left by a crashed post_process to avoid job write failure
    lck = job_name + ".lck"
    if os.path.exists(lck):
        os.remove(lck)

    j = mdb.Job(
        activateLoadBalancing=False,
        atTime=None,
        contactPrint=OFF,
        description="",
        echoPrint=OFF,
        explicitPrecision=SINGLE,
        historyPrint=OFF,
        memory=90,
        memoryUnits=PERCENTAGE,
        model=cfg.naming.model_name,
        modelPrint=OFF,
        multiprocessingMode=MPI,
        name=job_name,
        nodalOutputPrecision=SINGLE,
        numCpus=solver.num_cpus,
        numDomains=solver.num_domains,
        parallelizationMethodExplicit=DOMAIN,
        queue=None,
        resultsFormat=ODB,
        scratch="",
        type=ANALYSIS,
        userSubroutine="",
        waitHours=0,
        waitMinutes=0,
    )

    print(">>> Submitting job '%s' ..." % job_name)
    j.submit(consistencyChecking=OFF)
    j.waitForCompletion()
    print(">>> Job '%s' COMPLETED." % job_name)
