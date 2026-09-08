# -*- coding: utf-8 -*-
"""
patch_quota_scratch.py -- suite de [PATCH:io-robustness].

Constat qui motive ce patch : lors de l'echec du 08/09,
_assert_disk_ok() s'est bien execute avant le cas Design_00013 et n'a PAS
declenche -- os.statvfs voyait >= 20 Go libres -- alors qu'Abaqus n'a pas
pu ecrire son .stt (2 h 02 de wallclock pour 16 s de CPU). statvfs sur un
montage NFS rapporte l'espace du SYSTEME DE FICHIERS, pas ce que le compte
a le droit d'y ecrire : un quota utilisateur ou une limite d'inodes est
invisible pour lui. Le garde-fou existant est donc structurellement aveugle
a la cause reelle.

Ce que fait ce patch (2 fichiers, 3 modifications) :

  Q1  run_parameter_study.py : _assert_disk_ok() double son test statvfs
      d'une SONDE D'ECRITURE REELLE (ecrire N Mo, fsync, effacer). C'est le
      seul test qui voit un quota, une limite d'inodes ou un montage
      read-only. Reglable par $SCRATCHSIM_PROBE_MB (0 = desactive).

  S1  job.py : `scratch=os.environ.get("SLURM_TMPDIR", os.getcwd())`.
      SLURM_TMPDIR n'est pas une variable SLURM standard ; si elle n'existe
      pas, tout le scratch d'Abaqus/Explicit atterrit dans le repertoire de
      run, donc sur le home NFS, en concurrence avec les 5 autres chunks.
      Remplace par une resolution explicite avec repli bruyant.

  M1  job.py : memory=90/PERCENTAGE = 90 %% de la RAM PHYSIQUE du noeud, pas
      de l'allocation SLURM. Cale sur $SLURM_MEM_PER_NODE quand il existe.

Ne touche PAS a RUNS_ROOT, KEEP_ODB, _purge_job_artifacts ni a l'isolation
par cas : [PATCH:io-robustness] les fournit deja.

Idempotent, conserve les fins de ligne, sauvegardes numerotees (.bak,
.bak2, ...), validation ast.parse avant ecriture.

Usage :  python patch_quota_scratch.py [--dry-run] [--revert] [--root=DIR]
"""

import ast
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "[PATCH:quota-scratch]"
_SCRATCH_LINE_RE = re.compile(r"^[ \t]+scratch\s*=.*\n", re.M)


# --------------------------------------------------------------------------
# utilitaires
# --------------------------------------------------------------------------
def read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    return raw.replace(b"\r\n", b"\n").decode("utf-8"), crlf


def write_text(path, text, crlf):
    data = text.encode("utf-8")
    if crlf:
        data = data.replace(b"\n", b"\r\n")
    with open(path, "wb") as f:
        f.write(data)


def backup(path):
    """N'ecrase jamais une sauvegarde existante : .bak, .bak2, .bak3, ..."""
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        return bak
    n = 2
    while os.path.exists("%s.bak%d" % (path, n)):
        n += 1
    bak = "%s.bak%d" % (path, n)
    shutil.copy2(path, bak)
    print("    (un .bak existait deja -> sauvegarde dans %s)"
          % os.path.basename(bak))
    return bak


def latest_backup(path):
    """La sauvegarde la plus recente = le plus grand suffixe (pas la mtime :
    shutil.copy2 recopie la date de la source)."""
    if not os.path.exists(path + ".bak"):
        return None
    latest, n = path + ".bak", 2
    while os.path.exists("%s.bak%d" % (path, n)):
        latest = "%s.bak%d" % (path, n)
        n += 1
    return latest


def context(text, anchor, max_lines=12):
    """Lignes les plus proches de l'ancre, classees par rarete des jetons."""
    tokens = set(t for t in re.split(r"[^A-Za-z_0-9]+", anchor) if len(t) > 3)
    if not tokens:
        return ""
    freq = dict((t, text.count(t)) for t in tokens)
    scored = []
    for i, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("#"):
            continue
        hit = [t for t in tokens if t in line]
        if not hit:
            continue
        scored.append((sum(2 if freq[t] <= 3 else 1 for t in hit), i,
                       line.rstrip()))
    if not scored:
        return "\n(aucune ligne comparable dans le fichier)"
    scored.sort(key=lambda r: (-r[0], r[1]))
    out = ["  %4d | %s" % (i, l) for _, i, l in scored[:max_lines]]
    if len(scored) > max_lines:
        out.append("  ... (%d autres lignes)" % (len(scored) - max_lines))
    return "\nLignes les plus proches de l'ancre :\n" + "\n".join(out)


def sub_once(text, anchor, replacement, label):
    n = text.count(anchor)
    if n != 1:
        raise SystemExit("[%s] ancre absente ou ambigue (%d occurrences) :\n%r%s"
                         % (label, n, anchor[:160], context(text, anchor)))
    return text.replace(anchor, replacement)


# --------------------------------------------------------------------------
# Q1 : sonde d'ecriture dans run_parameter_study.py
# --------------------------------------------------------------------------
ORIG_ASSERT = '''def _assert_disk_ok(stem):
    """
    Un job qui ne peut pas ecrire son .stt consomme 5 h de wallclock pour
    61 s de CPU avant d'echouer. Ce controle coute 3 ms et rend la main
    immediatement, avec le vrai motif.
    """
    free = _free_gb(".")
    if free is None:
        return
    if free < MIN_FREE_GB:
        raise SystemExit(
            "ABORT avant soumission de '%s' : %.1f Go libres dans %s, "
            "seuil = %.1f Go. Abaqus echouerait a l'ecriture du .stt apres "
            "plusieurs heures. Liberer de l'espace, ou pointer "
            "SCRATCHSIM_RUNS_ROOT vers un espace de travail, ou abaisser "
            "SCRATCHSIM_MIN_FREE_GB en connaissance de cause."
            % (stem, free, os.getcwd(), MIN_FREE_GB))
'''

NEW_ASSERT = '''# %(mark)s begin -- le test statvfs seul ne voit pas un quota.
# Le 08/09, _assert_disk_ok() s'est execute avant Design_00013 sans rien
# signaler (statvfs voyait >= 20 Go libres) et Abaqus a quand meme echoue a
# ecrire son .stt apres 2 h 02 de wallclock pour 16 s de CPU. Sur un montage
# NFS, statvfs rapporte l'espace du systeme de fichiers, pas le quota du
# compte ni le nombre d'inodes disponibles. Seule une ecriture reelle tranche.
PROBE_MB = int(os.environ.get("SCRATCHSIM_PROBE_MB", "64"))   # 0 = desactive


def _write_probe(path=".", mb=None):
    """Ecrit puis efface `mb` Mo. None si OK, sinon le message d'erreur."""
    mb = PROBE_MB if mb is None else mb
    if mb <= 0:
        return None
    probe = os.path.join(path, ".disk_probe_%%d.tmp" %% os.getpid())
    block = b"\\0" * (1024 * 1024)
    try:
        f = open(probe, "wb")
        try:
            for _ in range(mb):
                f.write(block)
            f.flush()
            os.fsync(f.fileno())
        finally:
            f.close()
        return None
    except (IOError, OSError) as exc:
        return "%%s: %%s" %% (type(exc).__name__, exc)
    finally:
        try:
            if os.path.exists(probe):
                os.remove(probe)
        except OSError:
            pass


# Original (conserve) :
#   def _assert_disk_ok(stem):
#       free = _free_gb(".")
#       if free is None:
#           return
#       if free < MIN_FREE_GB:
#           raise SystemExit(...)
def _assert_disk_ok(stem):
    """
    Un job qui ne peut pas ecrire son .stt consomme des heures de wallclock
    pour quelques secondes de CPU avant d'echouer. Deux tests :
      1. statvfs   -> voit un systeme de fichiers plein ;
      2. ecriture  -> voit EN PLUS un quota, une limite d'inodes, un
                      montage read-only. C'est le seul qui aurait attrape
                      l'echec du 08/09.
    """
    free = _free_gb(".")
    if free is not None and free < MIN_FREE_GB:
        raise SystemExit(
            "ABORT avant soumission de '%%s' : %%.1f Go libres dans %%s, "
            "seuil = %%.1f Go. Abaqus echouerait a l'ecriture du .stt apres "
            "plusieurs heures. Liberer de l'espace, ou pointer "
            "SCRATCHSIM_RUNS_ROOT vers un espace de travail, ou abaisser "
            "SCRATCHSIM_MIN_FREE_GB en connaissance de cause."
            %% (stem, free, os.getcwd(), MIN_FREE_GB))

    err = _write_probe(".")
    if err is not None:
        raise SystemExit(
            "ABORT avant soumission de '%%s' : impossible d'ecrire %%d Mo dans "
            "%%s alors que statvfs annonce %%s Go libres. Quota utilisateur "
            "atteint, limite d'inodes, ou montage en lecture seule -- c'est "
            "exactement le cas qui a fait echouer Design_00013 (.stt "
            "illisible apres 2 h). Verifier 'quota -s' et 'df -i'. "
            "Detail : %%s"
            %% (stem, PROBE_MB, os.getcwd(),
               "?" if free is None else "%%.1f" %% free, err))
# %(mark)s end
''' % {"mark": MARK}


def patch_rps(text):
    if MARK in text:
        return text, False
    return sub_once(text, ORIG_ASSERT, NEW_ASSERT, "rps/assert_disk_ok"), True


# --------------------------------------------------------------------------
# S1 + M1 : job.py
# --------------------------------------------------------------------------
JOB_HELPERS = '''
# %(mark)s begin -- scratch Abaqus et memoire.
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

    jid = os.environ.get("SLURM_JOB_ID") or ("pid%%d" %% os.getpid())
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "abq"
    chosen = None

    for var in _SCRATCH_ENV_VARS:
        base = os.environ.get(var)
        if _usable(base, _MIN_SCRATCH_GB):
            chosen = os.path.join(base, "abq_%%s" %% jid)
            break
    if chosen is None:
        for root in _SCRATCH_ROOTS:
            if _usable(root, _MIN_SCRATCH_GB):
                chosen = os.path.join(root, user, "abq_%%s" %% jid)
                break

    if chosen is not None:
        try:
            if not os.path.isdir(chosen):
                os.makedirs(chosen)
        except OSError:
            chosen = None

    if chosen is None:
        chosen = os.getcwd()
        print("*** WARNING: aucun scratch local trouve (>= %%.0f Go libres). "
              "Abaqus va utiliser le repertoire de run %%s. Sur un home NFS "
              "partage, c'est la configuration qui produit les echecs "
              "'.stt / check the disk space'. Definir $ABQ_SCRATCH dans "
              "submit.sh." %% (_MIN_SCRATCH_GB, chosen))
    else:
        gb = _free_gb(chosen)
        print(">>> Abaqus scratch: %%s (%%s Go libres)"
              %% (chosen, "?" if gb is None else "%%.0f" %% gb))

    _SCRATCH_CACHE.append(chosen)
    return chosen


def abaqus_memory_setting():
    """
    memory=90 / PERCENTAGE = 90 %% de la RAM PHYSIQUE du noeud, pas de
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
            print(">>> Abaqus memory: %%d Mo (80%%%% de l'allocation SLURM)." %% mb)
            return mb, MEGA_BYTES
    return 90, PERCENTAGE
# %(mark)s end
''' % {"mark": MARK}


def patch_job(text):
    if MARK in text:
        return text, False

    hits = _SCRATCH_LINE_RE.findall(text)
    if len(hits) == 1:
        orig = hits[0]
        indent = re.match(r"[ \t]*", orig).group(0)
        print("    job.py/scratch : remplace -> %s" % orig.strip())
        text = text.replace(
            orig,
            "%s# %s original :\n%s#   %s\n%sscratch=resolve_scratch_dir(),\n"
            % (indent, MARK, indent, orig.strip(), indent))
    elif not hits:
        print("    job.py/scratch : aucun kwarg scratch=, insertion.")
        text = sub_once(
            text,
            '        type=ANALYSIS,\n',
            '        # %s kwarg scratch absent : ajoute ici.\n'
            '        scratch=resolve_scratch_dir(),\n'
            '        type=ANALYSIS,\n' % MARK,
            "job.py/scratch-insert")
    else:
        raise SystemExit("[job.py/scratch] %d lignes 'scratch=' :\n%s"
                         % (len(hits),
                            "\n".join("  " + h.strip() for h in hits)))

    text = sub_once(
        text,
        'from ScratchSimulation.AbaqusModel.abaqus_env import *\nimport os\n',
        'from ScratchSimulation.AbaqusModel.abaqus_env import *\nimport os\n'
        + JOB_HELPERS,
        "job.py/helpers")

    text = sub_once(
        text,
        '        memory=90,\n        memoryUnits=PERCENTAGE,\n',
        '        # %s memory=90/PERCENTAGE portait sur la RAM du noeud.\n'
        '        memory=_MEM_VALUE,\n'
        '        memoryUnits=_MEM_UNITS,\n' % MARK,
        "job.py/memory")

    text = sub_once(
        text,
        '    j = mdb.Job(\n',
        '    _MEM_VALUE, _MEM_UNITS = abaqus_memory_setting()   # %s\n'
        '\n    j = mdb.Job(\n' % MARK,
        "job.py/memkw")

    return text, True


# --------------------------------------------------------------------------
# localisation des fichiers + main
# --------------------------------------------------------------------------
TARGETS = [
    ("job.py", patch_job, "def run_job_and_wait"),
    ("run_parameter_study.py", patch_rps, "def _assert_disk_ok"),
]

_SKIP_DIRS = set([".git", "__pycache__", "runs", "SimDataOutputs",
                  "material_parameters", ".idea", ".vscode", "build", "dist"])


def _walk_for(root, name, signature):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in _SKIP_DIRS and not d.startswith(".")]
        if name not in filenames:
            continue
        path = os.path.join(dirpath, name)
        try:
            with open(path, "rb") as f:
                blob = f.read()
        except (IOError, OSError):
            continue
        if signature.encode("utf-8") in blob.replace(b"\r\n", b"\n"):
            hits.append(os.path.realpath(path))
    return hits


def locate(name, signature, roots):
    seen = []
    for root in roots:
        for hit in _walk_for(root, name, signature):
            if hit not in seen:
                seen.append(hit)
        if seen:
            break
    if not seen:
        raise SystemExit(
            "Fichier introuvable : %s (aucun fichier de ce nom contenant %r "
            "sous %s).\nRelance avec --root=/chemin/vers/ScratchSimulation."
            % (name, signature, " ou ".join(roots)))
    if len(seen) > 1:
        raise SystemExit("Plusieurs candidats pour %s :\n  %s\nPrecise "
                         "--root=." % (name, "\n  ".join(seen)))
    return seen[0]


def search_roots(argv):
    for arg in argv:
        if arg.startswith("--root="):
            root = os.path.abspath(os.path.expanduser(arg.split("=", 1)[1]))
            if not os.path.isdir(root):
                raise SystemExit("--root : %s n'est pas un repertoire." % root)
            return [root]
    roots = [HERE]
    parent = os.path.dirname(HERE)
    if parent and parent != HERE:
        roots.append(parent)
    return roots


def main(argv):
    dry = "--dry-run" in argv
    revert = "--revert" in argv
    roots = search_roots(argv)

    resolved = [(n, fn, locate(n, sig, roots)) for n, fn, sig in TARGETS]
    for name, _, path in resolved:
        print("found %-24s -> %s" % (name, path))
    print("")

    if revert:
        for _, _, path in resolved:
            bak = latest_backup(path)
            if bak:
                shutil.copy2(bak, path)
                print("reverted %s  (depuis %s)"
                      % (path, os.path.basename(bak)))
            else:
                print("aucune sauvegarde pour %s" % path)
        return 0

    changed = 0
    for name, fn, path in resolved:
        text, crlf = read_text(path)
        new, applied = fn(text)
        if not applied:
            print("%-24s deja patche, ignore." % name)
            continue
        ast.parse(new, filename=name)
        if dry:
            print("%-24s OK (dry-run, %+d lignes)"
                  % (name, new.count("\n") - text.count("\n")))
            continue
        backup(path)
        write_text(path, new, crlf)
        print("%-24s patche (%s, %+d lignes)"
              % (name, "CRLF" if crlf else "LF",
                 new.count("\n") - text.count("\n")))
        changed += 1

    print("\n%d fichier(s) modifie(s)." % changed)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))