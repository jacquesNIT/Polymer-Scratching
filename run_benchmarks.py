# -*- coding: utf-8 -*-
"""
Driver for the V&V benchmark suite. Same CLI conventions as
run_parameter_study.py so the cluster launcher needs no special case.

    abaqus cae noGUI=run_benchmarks.py -- level0   glassy_pc
    abaqus cae noGUI=run_benchmarks.py -- hertz    glassy_pc
    abaqus cae noGUI=run_benchmarks.py -- hertz_time glassy_pc
    abaqus cae noGUI=run_benchmarks.py -- hertz_ms glassy_pc
    abaqus cae noGUI=run_benchmarks.py -- plough   semicrystalline_j2
    abaqus cae noGUI=run_benchmarks.py -- standard glassy_pc
    abaqus cae noGUI=run_benchmarks.py -- baseline glassy_pc

Tokens after '--', any order:
    <suite> [family] [i/N] [cpus=K] [tag=X] [set:PATH=VALUE ...]

Every run directory receives a manifest.json (git commit + dirty flag,
config fingerprint, effective mass-scaling factor, dm/m, N_a, smoothing
window). Without it two results are not comparable, which is the whole point
of the exercise.
"""

import os
import shutil
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.path.abspath(".")
sys.path.insert(0, os.path.dirname(_HERE))

from ScratchSimulation.AbaqusModel.abaqus_env import *
from ScratchSimulation.AbaqusModel.Configuration.benchmarks import (
    Bench_Config, hertz_case, single_element_case, ploughing_case,
    standard_reference_case, set_mesh, retarget_dt,
    BENCH_MESH_LADDER, BENCH_TIME_LADDER, BENCH_DT_SCALE_LADDER,
    BENCH_ELEMENT_MODES, BENCH_HERTZ_DEPTHS)
from ScratchSimulation.AbaqusModel.Simulation.bench_builder import (
    build_single_element_model, build_indentation_model,
    build_indentation_model_standard)
from ScratchSimulation.AbaqusModel.Simulation import build_scratch_model
from ScratchSimulation.AbaqusModel.Material import SubstrateMaterialAssignment
from ScratchSimulation.AbaqusModel.Postprocessing.bench_extractor import (
    extract_single_element, extract_indentation)
from ScratchSimulation.AbaqusModel.Postprocessing import post_process
from ScratchSimulation.AbaqusModel.Verification.manifest import write_manifest
from ScratchSimulation.AbaqusModel.utils import cleanup_abaqus_junk

DEFAULT_FAMILY = "glassy_pc"
DEFAULT_SUITE = "hertz"
REPO_DIR = os.path.dirname(_HERE)


# --------------------------------------------------------------------------
# job submission
# --------------------------------------------------------------------------

def _submit(cfg, bench):
    name = cfg.job_name
    if name in mdb.jobs:
        del mdb.jobs[name]
    lck = name + ".lck"
    if os.path.exists(lck):
        os.remove(lck)

    kwargs = dict(
        name=name, model=cfg.naming.model_name, description="V&V benchmark",
        type=ANALYSIS, numCpus=cfg.solver.num_cpus,
        memory=90, memoryUnits=PERCENTAGE, resultsFormat=ODB,
        nodalOutputPrecision=SINGLE,
        scratch=os.environ.get("SLURM_TMPDIR", os.getcwd()),
        contactPrint=OFF, echoPrint=OFF, historyPrint=OFF, modelPrint=OFF,
        queue=None, waitHours=0, waitMinutes=0, userSubroutine="",
        atTime=None, activateLoadBalancing=False,
    )
    if bench.solver_kind == Bench_Config.EXPLICIT:
        kwargs.update(explicitPrecision=DOUBLE, multiprocessingMode=MPI,
                      parallelizationMethodExplicit=DOMAIN,
                      numDomains=cfg.solver.num_domains)
    else:
        # Abaqus/Standard: the explicit-only arguments are invalid.
        kwargs.update(multiprocessingMode=THREADS, numGPUs=0)

    j = mdb.Job(**kwargs)
    print(">>> submitting '%s' (%s)" % (name, bench.solver_kind))
    j.submit(consistencyChecking=OFF)
    j.waitForCompletion()
    print(">>> '%s' completed." % name)


def _assign_material(model, part, cfg, with_friction=True):
    """
    Material assignment, with the friction step made optional.

    SubstrateMaterialAssignment.apply() always calls update_friction(), which
    reads model.interactionProperties[cfg.naming.contact_property]. A
    material-point test HAS NO INTERFACE, so that property does not and should
    not exist: creating a dummy one just to satisfy apply() would put a
    meaningless interaction in the model and in the ODB.

    create_material() and assign_section() are the two public methods that
    matter here; they are called directly rather than through apply().
    """
    asg = SubstrateMaterialAssignment(model, part, cfg)
    asg.create_material()
    asg.assign_section()
    if with_friction:
        asg.update_friction()
    else:
        print(">>> friction card skipped: this benchmark has no contact interface.")
    return asg


def _run_one(cfg, bench, stem, run_dir):
    """Build + submit + extract + manifest for a single benchmark case."""
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
    write_manifest(run_dir, cfg, bench, extra={"case_stem": stem},
                   repo_dir=REPO_DIR,
                   filename="manifest_%s.json" % stem)
    _submit(cfg, bench)

    if bench.kind == Bench_Config.SINGLE_ELEMENT:
        extract_single_element(cfg.job_name, stem, cfg, bench)
    elif bench.kind == Bench_Config.PLOUGHING:
        post_process(cfg.job_name, stem, cfg)
    else:
        extract_indentation(cfg.job_name, stem, cfg, bench)

    mdb.close()
    for ext in (".sta", ".odb", ".msg"):
        src = cfg.job_name + ext
        if os.path.exists(src) and stem != cfg.job_name:
            dst_dir = "BenchOutputs"
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)
            shutil.move(src, os.path.join(dst_dir, stem + ext))


# --------------------------------------------------------------------------
# suites -- each returns a list of (stem, cfg_factory)
# --------------------------------------------------------------------------

def suite_level0(family, **kw):
    """One C3D8R per constitutive mode. Seconds per case."""
    cases = []
    for mode in BENCH_ELEMENT_MODES:
        strain = 0.05 if mode == "shear" else 0.5
        cases.append(("L0_%s_%s" % (family, mode),
                      lambda f=family, m=mode, s=strain: single_element_case(
                          f, mode=m, strain=s)))
    return cases


def suite_level0_relax(family, **kw):
    return [("L0_%s_relaxation" % family,
             lambda f=family: single_element_case(f, mode="relaxation",
                                                  strain=0.05,
                                                  element_time=1e-4))]


def suite_hertz(family, depth=10e-3, **kw):
    """
    Mesh ladder at fixed depth, near-unscaled (s = 1) -> benchmark 1a in
    Explicit. Answers: how many elements per contact radius does 1 % on the
    force actually require, and what is the OBSERVED order of convergence.
    """
    cases = []
    for h in BENCH_MESH_LADDER:
        cases.append(("Hertz_%s_h%s" % (family, ("%g" % h).replace(".", "p")),
                      lambda f=family, hh=h, d=depth: hertz_case(
                          f, h_mesh=hh, depth_max=d, ramp_time=0.05,
                          dt_scale=1.0)))
    return cases


def suite_hertz_depth(family, h_mesh=0.010, **kw):
    """Depth ladder at fixed mesh: a/R sensitivity of the Hertz reference."""
    return [("HertzD_%s_d%s" % (family, ("%g" % d).replace(".", "p")),
             lambda f=family, dd=d, hh=h_mesh: hertz_case(
                 f, h_mesh=hh, depth_max=dd, ramp_time=0.05, dt_scale=1.0))
            for d in BENCH_HERTZ_DEPTHS]


def suite_hertz_time(family, h_mesh=0.010, depth=10e-3, **kw):
    """
    BENCHMARK 1b, series 1: sweep the loading time with the mass-scaling
    factor HELD CONSTANT (dt_scale fixed). Linear elastic -> no rate
    dependence is possible, and the exact answer is known at every depth.
    Any residual drift here is purely numerical.
    """
    return [("HertzT_%s_T%s" % (family, ("%g" % T).replace(".", "p")),
             lambda f=family, TT=T, hh=h_mesh, d=depth: hertz_case(
                 f, h_mesh=hh, depth_max=d, ramp_time=TT, dt_scale=15.0))
            for T in BENCH_TIME_LADDER]


def suite_hertz_time_production(family, h_mesh=0.010, depth=10e-3, **kw):
    """
    BENCHMARK 1b, series 2: the SAME time sweep, but with the mass scaling
    left exactly as the production config computes it. Comparing series 1 and
    series 2 separates "the result depends on T" from "the result depends on
    the mass-scaling factor, which happens to move with T".
    """
    return [("HertzTP_%s_T%s" % (family, ("%g" % T).replace(".", "p")),
             lambda f=family, TT=T, hh=h_mesh, d=depth: hertz_case(
                 f, h_mesh=hh, depth_max=d, ramp_time=TT, dt_scale=None))
            for T in BENCH_TIME_LADDER]


def suite_hertz_ms(family, h_mesh=0.010, depth=10e-3, **kw):
    """
    Mass-scaling ladder at fixed T: the (T, f) admissibility map. Result is a
    JUSTIFIED production pair instead of a compromise chosen by feel.
    """
    return [("HertzMS_%s_s%s" % (family, ("%g" % s).replace(".", "p")),
             lambda f=family, ss=s, hh=h_mesh, d=depth: hertz_case(
                 f, h_mesh=hh, depth_max=d, ramp_time=0.05, dt_scale=ss))
            for s in BENCH_DT_SCALE_LADDER]


def suite_plough(family, **kw):
    """
    mu = 0 anchor + one constant-mu point, on the production scratch.
    SCOF(mu=0) is the pure ploughing coefficient, to be compared with
    (2/pi) cot(60 deg) = 0.368; SCOF(mu) - SCOF(0) isolates the interfacial
    term. Briscoe must stay OFF here.
    """
    cases = []
    for mu in (0.0, 0.3):
        for h in (0.020, 0.010):
            stem = "Plough_%s_mu%s_h%s" % (family, ("%g" % mu).replace(".", "p"),
                                           ("%g" % h).replace(".", "p"))
            cases.append((stem, lambda f=family, m=mu, hh=h: ploughing_case(
                f, h_mesh=hh, mu=m)))
    return cases


def suite_standard(family, **kw):
    """Level 2: quasi-static indentation reference with full plasticity."""
    return [("Std_%s_h%s" % (family, ("%g" % h).replace(".", "p")),
             lambda f=family, hh=h: standard_reference_case(
                 f, h_mesh=hh, depth_max=40e-3, mu=0.0))
            for h in (0.020, 0.010)]


def suite_baseline(family, **kw):
    """
    The frozen non-regression set: the cheapest cases from every level.
    Run it BEFORE and AFTER any patch campaign; ten minutes settles what
    would otherwise be an afternoon of archaeology.
    """
    cases = []
    cases += suite_level0(family)
    cases += [("BL_Hertz_%s" % family,
               lambda f=family: hertz_case(f, h_mesh=0.010, depth_max=10e-3,
                                           ramp_time=0.05, dt_scale=1.0))]
    cases += [("BL_HertzProd_%s" % family,
               lambda f=family: hertz_case(f, h_mesh=0.010, depth_max=10e-3,
                                           ramp_time=0.05, dt_scale=None))]
    cases += [("BL_Plough_%s" % family,
               lambda f=family: ploughing_case(f, h_mesh=0.020,
                                               scratch_time=0.05, mu=0.0))]
    return cases


SUITES = {
    "level0": suite_level0,
    "level0_relax": suite_level0_relax,
    "hertz": suite_hertz,
    "hertz_depth": suite_hertz_depth,
    "hertz_time": suite_hertz_time,
    "hertz_time_prod": suite_hertz_time_production,
    "hertz_ms": suite_hertz_ms,
    "plough": suite_plough,
    "standard": suite_standard,
    "baseline": suite_baseline,
}


# --------------------------------------------------------------------------
# CLI (identical grammar to run_parameter_study._parse_cli)
# --------------------------------------------------------------------------

def _parse_value(text):
    low = text.strip().lower()
    if low in ("true", "on", "yes"):
        return True
    if low in ("false", "off", "no"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _apply_overrides(cfg, overrides):
    for path, raw in overrides or []:
        obj = cfg
        parts = path.split(".")
        for name in parts[:-1]:
            if not hasattr(obj, name):
                raise SystemExit("Override '%s': no attribute '%s'." % (path, name))
            obj = getattr(obj, name)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise SystemExit("Override '%s': '%s' has no attribute '%s'."
                             % (path, type(obj).__name__, leaf))
        setattr(obj, leaf, _parse_value(raw))
        if path == "solver.num_cpus":
            cfg.solver.num_domains = int(_parse_value(raw))
        print(">>> Override cfg.%s = %r" % (path, _parse_value(raw)))


def _parse_cli(argv):
    out = {"suite": DEFAULT_SUITE, "family": DEFAULT_FAMILY,
           "chunk": None, "cpus": None, "tag": None, "overrides": []}
    rest = argv[argv.index("--") + 1:] if "--" in argv else \
        [a for a in argv[1:] if a in SUITES]
    positional = []
    for tok in rest:
        if tok.startswith("cpus="):
            out["cpus"] = int(tok.split("=", 1)[1])
        elif tok.startswith("tag="):
            out["tag"] = tok.split("=", 1)[1]
        elif tok.startswith("set:"):
            body = tok[4:]
            if "=" not in body:
                raise SystemExit("Bad override token '%s'." % tok)
            k, v = body.split("=", 1)
            out["overrides"].append((k, v))
        elif tok.count("/") == 1 and all(p.isdigit() for p in tok.split("/")):
            i, n = tok.split("/")
            out["chunk"] = (int(i), int(n))
        else:
            positional.append(tok)
    if positional:
        out["suite"] = positional[0]
    if len(positional) > 1:
        out["family"] = positional[1]
    return out


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
    if cli["chunk"] is not None:
        i, n = cli["chunk"]
        cases = cases[i::n]
        suffix += "_c%03dof%03d" % (i, n)
    if not cases:
        raise SystemExit("No case selected.")

    run_dir = os.path.join("runs", "Bench_%s_%s%s"
                           % (cli["suite"], cli["family"], suffix))
    _makedirs_safe(run_dir)
    os.chdir(run_dir)
    run_dir_abs = os.path.abspath(os.getcwd())
    _makedirs_safe("BenchOutputs")

    print(">>> benchmark suite '%s' on family '%s': %d case(s) in %s"
          % (cli["suite"], cli["family"], len(cases), run_dir_abs))

    for i, (stem, factory) in enumerate(cases, start=1):
        cfg, bench = factory()
        if cli["cpus"]:
            cfg.solver.num_cpus = int(cli["cpus"])
            cfg.solver.num_domains = int(cli["cpus"])
        _apply_overrides(cfg, cli["overrides"])
        # Any override may have changed the mesh or the material, so the
        # mass-scaling target must be recomputed -- never inherited.
        if bench.kind != Bench_Config.SINGLE_ELEMENT and cli["overrides"]:
            retarget_dt(cfg)
        cfg.job_name = "B" + stem[:60].replace(".", "p")
        print("\n>>> [%d/%d] %s" % (i, len(cases), stem))
        _run_one(cfg, bench, stem, ".")

    cleanup_abaqus_junk(base_dir=run_dir_abs)
    print(">>> suite '%s' finished." % cli["suite"])
    return 0


if __name__ == "__main__":
    sys.exit(main())