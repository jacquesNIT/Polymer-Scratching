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
        explicitPrecision=DOUBLE,             # Important factor for high number of increments ( SINGLE for small sims, DOUBLE for bigger ones)
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
        scratch=os.environ.get("SLURM_TMPDIR", os.getcwd()),
        type=ANALYSIS,
        userSubroutine="",
        waitHours=0,
        waitMinutes=0,
    )

    print(">>> Submitting job '%s' ..." % job_name)
    j.submit(consistencyChecking=OFF)
    j.waitForCompletion()
    # [PATCH:abort-visibility] begin -- le statut n'etait pas verifie.
    # waitForCompletion() rend la main sur un job AVORTE exactement comme sur
    # un job reussi. post_process ouvrait alors un ODB tronque et remontait
    # une IndexError numpy, sans rapport visible avec la vraie cause.
    _check_job_status(j, job_name)
    # [PATCH:abort-visibility] end
    print(">>> Job '%s' COMPLETED." % job_name)


# [PATCH:abort-visibility] begin -- diagnostic d'abandon.
class JobAbortedError(RuntimeError):
    """Abaqus a rendu la main avec un statut autre que COMPLETED."""


_ABORT_HINTS = (
    "excessively distorted",
    "distorted element",
    "too many attempts",
    "time increment required is less than",
    "unstable",
    "negative eigenvalue",
    "out of memory",
    "insufficient",
    "disk",
    "quota",
    "error",
    "aborted",
)


def _tail_reason(job_name, ext, n_lines=400):
    """Derniere ligne informative d'un .sta / .msg / .log."""
    path = job_name + ext
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            lines = f.readlines()[-n_lines:]
    except (IOError, OSError):
        return None
    for line in reversed(lines):
        low = line.lower()
        if any(h in low for h in _ABORT_HINTS):
            return "%s: %s" % (ext, " ".join(line.split())[:200])
    return None


def _sta_says_success(job_name):
    """True / False / None (fichier illisible)."""
    path = job_name + ".sta"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            txt = f.read()
    except (IOError, OSError):
        return None
    return "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in txt.upper()


def _check_job_status(j, job_name):
    """
    Leve JobAbortedError si le job n'a pas abouti.

    `j.status` est la source primaire. Certaines versions/soumissions le
    laissent vide : on retombe alors sur le .sta, qui porte la phrase
    'THE ANALYSIS HAS COMPLETED SUCCESSFULLY'. Si NI l'un NI l'autre n'est
    exploitable on n'echoue pas (on avertit), pour ne pas casser une
    configuration qui fonctionne.
    """
    status = ""
    try:
        status = str(j.status).upper()
    except Exception:
        status = ""

    ok_status = ("COMPLET" in status) if status and status != "NONE" else None
    ok_sta = _sta_says_success(job_name)

    if ok_status is False or (ok_status is None and ok_sta is False):
        reasons = [r for r in (_tail_reason(job_name, ".msg"),
                               _tail_reason(job_name, ".sta"),
                               _tail_reason(job_name, ".log")) if r]
        raise JobAbortedError(
            "Abaqus job '%s' did NOT complete (status=%s, .sta success=%s). %s"
            % (job_name, status or "unknown", ok_sta,
               " | ".join(reasons) or "No abort reason found in .msg/.sta/.log."))

    if ok_status is None and ok_sta is None:
        print("Warning: could not determine the status of job '%s' "
              "(j.status empty and no readable .sta). Continuing." % job_name)
# [PATCH:abort-visibility] end