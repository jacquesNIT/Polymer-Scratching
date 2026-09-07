# Abaqus job submission and wait.

from ScratchSimulation.AbaqusModel.abaqus_env import *
import os


# [PATCH:prime-scratch] begin -- resolution du scratch local au noeud.
def _abaqus_scratch():
    """
    Repertoire de travail temporaire passe a mdb.Job(scratch=...).

    Ordre de resolution :
      1. /scratch/$SLURM_JOB_ID   -- convention de PRIME (cf. subabqpy2025)
      2. $SLURM_TMPDIR            -- convention Slurm generique
      3. $TMPDIR                  -- dernier recours local
      4. os.getcwd()              -- AVEC AVERTISSEMENT : c'est le home
                                     partage, les gros jobs y echoueront

    Chaque candidat est verifie en existence ET en ecriture : un chemin
    inscriptible mais inexistant renverrait exactement l'echec qu'on
    cherche a eviter.
    """
    candidates = []
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if job_id:
        candidates.append(os.path.join("/scratch", str(job_id)))
    for var in ("SLURM_TMPDIR", "TMPDIR"):
        value = os.environ.get(var)
        if value:
            candidates.append(value)

    for path in candidates:
        if os.path.isdir(path) and os.access(path, os.W_OK):
            print(">>> Abaqus scratch: %s" % path)
            return path

    fallback = os.getcwd()
    print("WARNING: no node-local scratch directory found (tried: %s). "
          "Falling back to %s. On a shared home this is where the .stt "
          "write failures come from."
          % (", ".join(candidates) or "<none>", fallback))
    return fallback
# [PATCH:prime-scratch] end


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
        # [PATCH:prime-scratch] PRIME expose /scratch/$SLURM_JOB_ID, pas
        # SLURM_TMPDIR. L'ancien repli os.getcwd() envoyait tout le
        # scratch Abaqus dans le home partage.
        scratch=_abaqus_scratch(),
        type=ANALYSIS,
        userSubroutine="",
        waitHours=0,
        waitMinutes=0,
    )

    # [PATCH:io-robustness] garde d'espace : un job qui ne peut pas ecrire son
    # .stt brule des heures de wallclock avant d'echouer.
    try:
        _st = os.statvfs(os.getcwd())
        _free_gb = (_st.f_bavail * _st.f_frsize) / float(1024 ** 3)
    except (AttributeError, OSError):
        _free_gb = None
    if _free_gb is not None:
        print(">>> Free space in run dir: %.1f GB." % _free_gb)
        _min_gb = float(os.environ.get("SCRATCHSIM_MIN_FREE_GB", "20"))
        if _free_gb < _min_gb:
            raise JobAbortedError(
                "Refusing to submit '%s': only %.1f GB free in %s "
                "(threshold %.1f GB). Abaqus would fail writing its "
                ".stt after hours of wallclock."
                % (job_name, _free_gb, os.getcwd(), _min_gb))

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
    # [PATCH:io-robustness] signatures d'echec d'ecriture.
    "check the disk space",
    "no space left",
    "unable to open the file",
    "write access",
    "fatal errors",
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
        # [PATCH:io-robustness] le .dat porte les erreurs de preprocesseur.
        reasons = [r for r in (_tail_reason(job_name, ".msg"),
                               _tail_reason(job_name, ".sta"),
                               _tail_reason(job_name, ".dat"),
                               _tail_reason(job_name, ".log")) if r]
        raise JobAbortedError(
            "Abaqus job '%s' did NOT complete (status=%s, .sta success=%s). %s"
            % (job_name, status or "unknown", ok_sta,
               " | ".join(reasons) or "No abort reason found in .msg/.sta/.log."))

    # [PATCH:io-robustness] begin -- un .sta absent n'est pas une incertitude.
    # Abaqus/Explicit cree le .sta des que l'analyse demarre. S'il
    # manque, le job est mort dans le preprocesseur (erreur de deck ou
    # d'ecriture) et le motif est dans le .dat, jamais consulte jusqu'ici.
    # L'ancien 'Continuing' laissait post_process ouvrir un ODB fantome,
    # qui remontait un KeyError: 'S_SURF-1' sans rapport avec la cause.
    if ok_status is None and ok_sta is None:
        reasons = [r for r in (_tail_reason(job_name, ".dat"),
                               _tail_reason(job_name, ".log"),
                               _tail_reason(job_name, ".msg")) if r]
        raise JobAbortedError(
            "Abaqus job '%s' produced no .sta: the analysis never "
            "started (input-file processor or I/O failure). %s"
            % (job_name, " | ".join(reasons)
               or "No reason found in .dat/.log/.msg."))
    # [PATCH:io-robustness] end
# [PATCH:abort-visibility] end