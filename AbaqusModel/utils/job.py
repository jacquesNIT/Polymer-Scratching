# Abaqus job submission and wait.

from ScratchSimulation.AbaqusModel.abaqus_env import *
import os

# [PATCH:quota-scratch] begin -- scratch Abaqus et memoire.
_SCRATCH_ENV_VARS = ("ABQ_SCRATCH", "SLURM_TMPDIR", "SLURM_SCRATCH_DIR", "TMPDIR")
_SCRATCH_ROOTS = ("/scratch", "/lscratch", "/localscratch", "/local", "/tmp")
_MIN_SCRATCH_GB = float(os.environ.get("SCRATCHSIM_MIN_SCRATCH_GB", "20"))
_SCRATCH_CACHE = []          # resolu une seule fois par process


def _free_gb(path):
    try:
        st = os.statvfs(path)
    except (AttributeError, OSError):
        return None
    return (st.f_bavail * st.f_frsize) / float(1024 ** 3)


def _usable(path, min_gb):
    if not path or not os.path.isdir(path) or not os.access(path, os.W_OK):
        return False
    gb = _free_gb(path)
    return (gb is None) or (gb >= min_gb)


def resolve_scratch_dir():
    """
    Repertoire scratch d'Abaqus, resolu une fois par process.

    L'ancienne ligne etait :
        scratch=os.environ.get("SLURM_TMPDIR", os.getcwd())
    SLURM_TMPDIR n'est PAS une variable SLURM standard (elle est propre a
    certains sites). Quand elle n'existe pas, le repli est le repertoire
    courant, c'est-a-dire runs/<etude>/ sur le home NFS : tout le trafic
    disque aleatoire d'Abaqus/Explicit y passe, six chunks en parallele.

    Ordre : $ABQ_SCRATCH / $SLURM_TMPDIR / $SLURM_SCRATCH_DIR / $TMPDIR,
    puis /scratch, /lscratch, /localscratch, /local, /tmp. Repli sur le cwd
    avec un avertissement explicite dans le .out SLURM.
    """
    if _SCRATCH_CACHE:
        return _SCRATCH_CACHE[0]

    jid = os.environ.get("SLURM_JOB_ID") or ("pid%d" % os.getpid())
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "abq"
    chosen = None

    for var in _SCRATCH_ENV_VARS:
        base = os.environ.get(var)
        if _usable(base, _MIN_SCRATCH_GB):
            chosen = os.path.join(base, "abq_%s" % jid)
            break
    if chosen is None:
        for root in _SCRATCH_ROOTS:
            if _usable(root, _MIN_SCRATCH_GB):
                chosen = os.path.join(root, user, "abq_%s" % jid)
                break

    if chosen is not None:
        try:
            if not os.path.isdir(chosen):
                os.makedirs(chosen)
        except OSError:
            chosen = None

    if chosen is None:
        chosen = os.getcwd()
        print("*** WARNING: aucun scratch local trouve (>= %.0f Go libres). "
              "Abaqus va utiliser le repertoire de run %s. Sur un home NFS "
              "partage, c'est la configuration qui produit les echecs "
              "'.stt / check the disk space'. Definir $ABQ_SCRATCH dans "
              "submit.sh." % (_MIN_SCRATCH_GB, chosen))
    else:
        gb = _free_gb(chosen)
        print(">>> Abaqus scratch: %s (%s Go libres)"
              % (chosen, "?" if gb is None else "%.0f" % gb))

    _SCRATCH_CACHE.append(chosen)
    return chosen


def abaqus_memory_setting():
    """
    memory=90 / PERCENTAGE = 90 % de la RAM PHYSIQUE du noeud, pas de
    l'allocation SLURM (-m 100). Plusieurs chunks sur un meme noeud
    sur-souscrivent alors la memoire, ce qui produit du swap : wallclock de
    plusieurs heures pour quelques secondes de CPU.
    """
    for var in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        raw = os.environ.get(var)
        if not raw:
            continue
        try:
            mb = float(raw)
        except ValueError:
            continue
        if var == "SLURM_MEM_PER_CPU":
            try:
                mb *= float(os.environ.get("SLURM_CPUS_ON_NODE", "1"))
            except ValueError:
                pass
        mb = int(mb * 0.80)
        if mb > 0:
            print(">>> Abaqus memory: %d Mo (80%% de l'allocation SLURM)." % mb)
            return mb, MEGA_BYTES
    return 90, PERCENTAGE
# [PATCH:quota-scratch] end

def run_job_and_wait(job_name, cfg):

    solver = cfg.solver

    # Remove stale job from mdb to avoid "name already exists" error on rerun
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]

    # Remove .lck left by a crashed post_process to avoid job write failure
    lck = job_name + ".lck"
    if os.path.exists(lck):
        os.remove(lck)

    _MEM_VALUE, _MEM_UNITS = abaqus_memory_setting()   # [PATCH:quota-scratch]

    j = mdb.Job(
        activateLoadBalancing=False,
        atTime=None,
        contactPrint=OFF,
        description="",
        echoPrint=OFF,
        explicitPrecision=DOUBLE,             # Important factor for high number of increments ( SINGLE for small sims, DOUBLE for bigger ones)
        historyPrint=OFF,
        # [PATCH:quota-scratch] memory=90/PERCENTAGE portait sur la RAM du noeud.
        memory=_MEM_VALUE,
        memoryUnits=_MEM_UNITS,
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
        # [PATCH:quota-scratch] original :
        #   scratch=os.environ.get("SLURM_TMPDIR", os.getcwd()),
        scratch=resolve_scratch_dir(),
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