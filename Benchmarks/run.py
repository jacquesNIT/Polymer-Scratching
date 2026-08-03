# -*- coding: utf-8 -*-
"""
Benchmark driver.

    abaqus cae noGUI=Benchmarks/run.py -- level0   glassy_pc
    abaqus cae noGUI=Benchmarks/run.py -- hertz    glassy_pc
    abaqus cae noGUI=Benchmarks/run.py -- baseline glassy_pc

Tokens in any order: <suite> <family> [i/N] [cpus=K] [tag=X] [set:PATH=VALUE].
Every run directory receives a manifest.json (git commit, config fingerprint,
effective mass factor, N_a); without it two results are not comparable.
"""

import os
import shutil
import sys


def _bootstrap_path():
    """Put the directory CONTAINING the ScratchSimulation package on sys.path.

    Abaqus does not reliably define __file__ for a noGUI script, and the
    fallback cannot assume a fixed depth: guessing wrong shifts every level and
    inserts the wrong root. So search upward from every plausible starting
    point for a directory that holds ScratchSimulation/AbaqusModel, and take
    its parent. Works whether the script is launched from the repository root,
    from inside Benchmarks/, or by absolute path.
    """
    starts = []
    try:
        starts.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    starts.append(os.path.abspath(os.getcwd()))
    for a in sys.argv:
        if a.endswith("run.py") or a.endswith(".py"):
            cand = os.path.dirname(os.path.abspath(a))
            if cand:
                starts.append(cand)

    for start in starts:
        d = start
        for _ in range(6):
            # d itself is the package?
            if os.path.isdir(os.path.join(d, "AbaqusModel")) \
                    and os.path.basename(d) == "ScratchSimulation":
                parent = os.path.dirname(d)
                if parent not in sys.path:
                    sys.path.insert(0, parent)
                return d
            # d contains the package?
            pkg = os.path.join(d, "ScratchSimulation")
            if os.path.isdir(os.path.join(pkg, "AbaqusModel")):
                if d not in sys.path:
                    sys.path.insert(0, d)
                return pkg
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd

    raise SystemExit(
        "Cannot locate the ScratchSimulation package.\n"
        "Searched upward from: %s\n"
        "Launch from the repository root, e.g.\n"
        "    cd C:\\Stage2A\\ScratchSimulation\n"
        "    abaqus cae noGUI=Benchmarks/run.py -- level0 glassy_pc"
        % ", ".join(starts))


_PKG = _bootstrap_path()
REPO_DIR = os.path.dirname(_PKG)

from ScratchSimulation.AbaqusModel.abaqus_env import *
from ScratchSimulation.AbaqusModel.Configuration.families import FAMILIES
from ScratchSimulation.AbaqusModel.Material import SubstrateMaterialAssignment
from ScratchSimulation.AbaqusModel.Postprocessing import post_process
from ScratchSimulation.AbaqusModel.Simulation import build_scratch_model
from ScratchSimulation.AbaqusModel.Verification.manifest import write_manifest
from ScratchSimulation.AbaqusModel.utils import cleanup_abaqus_junk
from ScratchSimulation.Benchmarks.builder import (
    build_single_element_model, build_indentation_model,
    build_indentation_model_standard)
from ScratchSimulation.Benchmarks.cases import Bench_Config, retarget_dt
from ScratchSimulation.Benchmarks.extractor import (
    extract_single_element, extract_indentation)
from ScratchSimulation.Benchmarks.suites import SUITES

DEFAULT_SUITE = "hertz"
DEFAULT_FAMILY = "glassy_pc"

# Results always land in <package>/runs/, never relative to the launch
# directory, so a run started from inside Benchmarks/ does not create a second
# tree in the wrong place.
RUNS_ROOT = os.path.join(_PKG, "runs")

# Files Abaqus leaves in the working directory; purged before each submission
# so a later family never inherits an earlier one's deck.
_JOB_EXTS = (".inp", ".dat", ".sta", ".msg", ".com", ".prt", ".sim", ".lck",
             ".odb", ".res", ".mdl", ".stt", ".abq", ".pac", ".sel", ".ipm",
             ".log", ".fil")


# --------------------------------------------------------------------------
# Job handling
# --------------------------------------------------------------------------

def _submit(cfg, bench):
    name = cfg.job_name
    if name in mdb.jobs:
        del mdb.jobs[name]
    if os.path.exists(name + ".lck"):
        os.remove(name + ".lck")

    kwargs = dict(
        name=name, model=cfg.naming.model_name, description="V&V benchmark",
        type=ANALYSIS, numCpus=cfg.solver.num_cpus, memory=90,
        memoryUnits=PERCENTAGE, resultsFormat=ODB, nodalOutputPrecision=SINGLE,
        scratch=os.environ.get("SLURM_TMPDIR", os.getcwd()),
        contactPrint=OFF, echoPrint=OFF, historyPrint=OFF, modelPrint=OFF,
        queue=None, waitHours=0, waitMinutes=0, userSubroutine="",
        atTime=None, activateLoadBalancing=False)
    if bench.solver_kind == Bench_Config.EXPLICIT:
        kwargs.update(explicitPrecision=DOUBLE, multiprocessingMode=MPI,
                      parallelizationMethodExplicit=DOMAIN,
                      numDomains=cfg.solver.num_domains)
    else:
        kwargs.update(multiprocessingMode=THREADS, numGPUs=0)

    job = mdb.Job(**kwargs)
    print(">>> submitting '%s' (%s)" % (name, bench.solver_kind))
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    print(">>> '%s' completed." % name)


def _assign_material(model, part, cfg, with_friction=True):
    """create_material + assign_section, with the friction card optional.
    .apply() always calls update_friction(), which needs an interaction
    property; a material-point test has no interface and creating a dummy one
    would put a meaningless interaction in the model and in the ODB."""
    asg = SubstrateMaterialAssignment(model, part, cfg)
    asg.create_material()
    asg.assign_section()
    if with_friction:
        asg.update_friction()
    return asg


def _prepare_model(cfg):
    """One fresh model per case. The builders index mdb.models[model_name] but
    the project never creates that model, relying on CAE's default 'Model-1';
    with a per-case name it has to be made. Also guarantees no parts, materials
    or steps leak between families."""
    name = cfg.naming.model_name
    if name not in mdb.models:
        if "Model-1" in mdb.models and len(mdb.models) == 1:
            mdb.models.changeKey(fromName="Model-1", toName=name)
        else:
            mdb.Model(name=name, modelType=STANDARD_EXPLICIT)
    for k in list(mdb.models.keys()):
        if k != name:
            del mdb.models[k]


def _run_one(cfg, bench, stem):
    _prepare_model(cfg)

    if bench.kind == Bench_Config.SINGLE_ELEMENT:
        model, part = build_single_element_model(cfg, bench)
    elif bench.kind == Bench_Config.PLOUGHING:
        model, part = build_scratch_model(cfg)
    elif bench.solver_kind == Bench_Config.STANDARD:
        model, part = build_indentation_model_standard(cfg, bench)
    else:
        model, part = build_indentation_model(cfg, bench)

    _assign_material(model, part, cfg,
                     with_friction=(bench.kind != Bench_Config.SINGLE_ELEMENT))
    write_manifest(".", cfg, bench, extra={"case_stem": stem},
                   repo_dir=REPO_DIR, filename="manifest_%s.json" % stem)
    _submit(cfg, bench)

    if bench.kind == Bench_Config.SINGLE_ELEMENT:
        extract_single_element(cfg.job_name, stem, cfg, bench)
    elif bench.kind == Bench_Config.PLOUGHING:
        post_process(cfg.job_name, stem, cfg)
    else:
        extract_indentation(cfg.job_name, stem, cfg, bench)

    # No mdb.close(): it shuts the whole database and leaves the session
    # without a default model for the next case. Isolation is done by
    # _prepare_model instead.
    if stem != cfg.job_name:
        if not os.path.isdir("BenchOutputs"):
            os.makedirs("BenchOutputs")
        for ext in (".sta", ".odb", ".msg", ".dat", ".inp"):
            src = cfg.job_name + ext
            if os.path.exists(src):
                try:
                    shutil.move(src, os.path.join("BenchOutputs", stem + ext))
                except (OSError, shutil.Error):
                    pass


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_value(text):
    low = text.strip().lower()
    if low in ("true", "on", "yes"):
        return True
    if low in ("false", "off", "no"):
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _apply_overrides(cfg, overrides):
    for path, raw in overrides or []:
        obj, parts = cfg, path.split(".")
        for name in parts[:-1]:
            if not hasattr(obj, name):
                raise SystemExit("Override '%s': no attribute '%s'." % (path, name))
            obj = getattr(obj, name)
        if not hasattr(obj, parts[-1]):
            raise SystemExit("Override '%s': '%s' has no attribute '%s'."
                             % (path, type(obj).__name__, parts[-1]))
        value = _parse_value(raw)
        setattr(obj, parts[-1], value)
        if path == "solver.num_cpus":
            cfg.solver.num_domains = int(value)
        print(">>> override cfg.%s = %r" % (path, value))


def _parse_cli(argv):
    """Whitelist parsing: a token counts only if it is a known suite, a known
    family, or carries explicit syntax. Abaqus injects tokens that vary by
    release ('-cae', 'noGUI=...', the script path) and may or may not keep the
    '--' separator, so blacklisting them can never be complete."""
    out = {"suite": DEFAULT_SUITE, "family": DEFAULT_FAMILY,
           "chunk": None, "cpus": None, "tag": None, "overrides": []}
    toks = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]

    suite = family = None
    for tok in toks:
        if tok.startswith("cpus="):
            try:
                out["cpus"] = int(tok.split("=", 1)[1])
            except ValueError:
                pass
        elif tok.startswith("tag="):
            out["tag"] = tok.split("=", 1)[1]
        elif tok.startswith("set:"):
            body = tok[4:]
            if "=" not in body:
                raise SystemExit("Bad override token '%s'." % tok)
            out["overrides"].append(tuple(body.split("=", 1)))
        elif tok.count("/") == 1 and all(p.isdigit() for p in tok.split("/")):
            i, n = tok.split("/")
            out["chunk"] = (int(i), int(n))
        elif tok in SUITES and suite is None:
            suite = tok
        elif tok in FAMILIES and family is None:
            family = tok

    if suite:
        out["suite"] = suite
    if family:
        out["family"] = family
    print(">>> parsed CLI: suite=%s family=%s  (argv tail=%s)"
          % (out["suite"], out["family"], list(toks)))
    if family is None:
        print(">>> note: no known family token; using '%s'. Known: %s"
              % (out["family"], ", ".join(sorted(FAMILIES))))
    return out


def _cap_cpus(cfg):
    """Production configs ask for cluster core counts; mdb.Job() rejects
    numCpus above what the host has, and a benchmark must be runnable
    interactively."""
    try:
        import multiprocessing
        avail = multiprocessing.cpu_count()
        if int(getattr(cfg.solver, "num_cpus", 1) or 1) > avail:
            print(">>> capping num_cpus %d -> %d" % (cfg.solver.num_cpus, avail))
            cfg.solver.num_cpus = cfg.solver.num_domains = avail
    except Exception:
        pass


def _makedirs_safe(path):
    if path and not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise


def main():
    cli = _parse_cli(sys.argv)
    if cli["suite"] not in SUITES:
        raise SystemExit("Unknown suite '%s'. Available: %s"
                         % (cli["suite"], ", ".join(sorted(SUITES))))

    cases = SUITES[cli["suite"]](cli["family"])
    suffix = ("_" + cli["tag"]) if cli["tag"] else ""
    if cli["chunk"]:
        i, n = cli["chunk"]
        cases = cases[i::n]
        suffix += "_c%03dof%03d" % (i, n)
    if not cases:
        raise SystemExit("No case selected.")

    run_dir = os.path.join(RUNS_ROOT, "Bench_%s_%s%s"
                           % (cli["suite"], cli["family"], suffix))
    _makedirs_safe(run_dir)
    os.chdir(run_dir)
    run_dir_abs = os.path.abspath(os.getcwd())
    _makedirs_safe("BenchOutputs")
    print(">>> suite '%s' on '%s': %d case(s) in %s"
          % (cli["suite"], cli["family"], len(cases), run_dir_abs))

    for i, (stem, factory) in enumerate(cases, start=1):
        cfg, bench = factory()
        if cli["cpus"]:
            cfg.solver.num_cpus = cfg.solver.num_domains = int(cli["cpus"])
        _cap_cpus(cfg)
        _apply_overrides(cfg, cli["overrides"])
        # An override may have changed the mesh or material, so the mass
        # scaling target must be recomputed, never inherited.
        if cli["overrides"] and bench.kind != Bench_Config.SINGLE_ELEMENT:
            retarget_dt(cfg)

        cfg.job_name = "B" + stem[:60].replace(".", "p")
        cfg.naming.model_name = "M" + stem[:58].replace(".", "p").replace("-", "_")
        for ext in _JOB_EXTS:
            stale = cfg.job_name + ext
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass

        print("\n>>> [%d/%d] %s  (job=%s, model=%s)"
              % (i, len(cases), stem, cfg.job_name, cfg.naming.model_name))
        _run_one(cfg, bench, stem)

    cleanup_abaqus_junk(base_dir=run_dir_abs)
    print(">>> suite '%s' finished." % cli["suite"])
    return 0


if __name__ == "__main__":
    sys.exit(main())