# Abaqus job submission and wait.

from abaqus import *
from abaqusConstants import *
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from sketch import *
from visualization import *
from connectorBehavior import *
from odbAccess import *

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
        model="Model-1",
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

    # Submit and hand control straight to CAE. Do NOT insert manual sleep/.lck
    # polling between submit() and waitForCompletion(): that starves CAE's
    # job-monitor message loop, so the completion event fires during a sleep and
    # is missed, and waitForCompletion() then hangs forever waiting on it.
    print(">>> Submitting job '%s' ..." % job_name)
    j.submit(consistencyChecking=OFF)
    j.waitForCompletion()
    print(">>> Job '%s' COMPLETED." % job_name)
