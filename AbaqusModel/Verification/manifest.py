# -*- coding: utf-8 -*-
"""
Run manifest: the traceability layer without which none of the benchmarks
above is worth anything.

ABAQUS-FREE (stdlib + numpy). Written by the kernel at build time, read by
the CPython analysis scripts.

The problem it solves is the one stated plainly: "I change a lot of
parameters and get a multitude of different results with nothing to compare
them to." Two results are only comparable if it is possible to PROVE what
differs between them. A manifest turns that into a hash comparison instead of
an archaeology exercise.

Every run directory gets a manifest.json containing:
  * git commit + dirty flag of the repository that produced it
  * a recursive fingerprint of the ENTIRE resolved Simulation_Config
    (values, never names: "glassy_pc" is not a specification, E = 2300 is)
  * the derived numbers that are never in the config but govern the result:
    dt_nat, effective mass-scaling factor f, dm/m, scaled wave speed,
    N_a = a/h, the amplitude-smoothing window, the ALE Courant number
  * environment: Abaqus version, host, SLURM ids, cpu count
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# Config fingerprint
# --------------------------------------------------------------------------

_SCALARS = (int, float, bool, str)
try:                                   # Python 2 kernels
    _SCALARS = _SCALARS + (unicode,)   # noqa: F821
except NameError:
    pass


def _plain(obj, _depth=0):
    """
    Recursively reduce a config object to plain JSON-able data, sorted, so
    that the same configuration always produces the same bytes.
    """
    if obj is None or isinstance(obj, _SCALARS):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_plain(o, _depth + 1) for o in obj]
    if isinstance(obj, dict):
        return dict((str(k), _plain(v, _depth + 1)) for k, v in sorted(obj.items()))
    if _depth > 8:
        return repr(obj)
    d = getattr(obj, "__dict__", None)
    if d is None:
        return repr(obj)
    out = {"__class__": type(obj).__name__}
    for k in sorted(d.keys()):
        if k.startswith("_"):
            continue
        out[k] = _plain(d[k], _depth + 1)
    # MODEL is a class attribute on every constitutive block, not an instance
    # one, so it would be lost without this: a fingerprint that cannot tell a
    # Mooney-Rivlin card from an Arruda-Boyce one is useless.
    model = getattr(type(obj), "MODEL", None)
    if model is not None:
        out["MODEL"] = str(model)
    return out


def config_fingerprint(cfg):
    """(plain_dict, md5_hex) of the fully resolved configuration."""
    plain = _plain(cfg)
    blob = json.dumps(plain, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return plain, hashlib.md5(blob).hexdigest()


# --------------------------------------------------------------------------
# Git / environment
# --------------------------------------------------------------------------

def git_info(repo_dir=None):
    """
    Commit + dirty flag. Never raises: a missing git is a WARN, not a crash.
    A run produced from a dirty tree is NOT reproducible and must be flagged
    as such, which is why 'dirty' is part of the manifest and checked by
    verify_baseline.py.
    """
    repo_dir = repo_dir or os.getcwd()
    out = {"commit": None, "dirty": None, "branch": None, "available": False}

    def _run(args):
        try:
            p = subprocess.Popen(args, cwd=repo_dir, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            so, _se = p.communicate()
            if p.returncode != 0:
                return None
            if not isinstance(so, str):
                so = so.decode("utf-8", "replace")
            return so.strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return out
    out["available"] = True
    out["commit"] = commit
    out["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    out["dirty"] = bool(status)
    if status:
        out["dirty_files"] = [l[3:] for l in status.splitlines()][:50]
    return out


def environment_info():
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": platform.node(),
        "cwd": os.getcwd(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    for k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_ON_NODE",
              "SLURM_JOB_NAME", "SLURM_NNODES"):
        if os.environ.get(k):
            env[k.lower()] = os.environ[k]
    try:
        from abaqus import version as _abq_version      # kernel only
        env["abaqus_version"] = str(_abq_version)
    except Exception:
        pass
    return env


# --------------------------------------------------------------------------
# Derived numbers
# --------------------------------------------------------------------------

def derived_quantities(cfg, bench=None):
    """
    The numbers that govern the result but appear in NO config field, so
    cannot be recovered later from the inputs alone.
    """
    from . import analytic as an
    out = {}
    try:
        ms = an.mass_scaling_factor(cfg)
        out["mass_scaling"] = ms
    except Exception as exc:
        out["mass_scaling"] = {"error": str(exc)}

    try:
        from ..Configuration.benchmarks import linear_moduli
        E, nu = linear_moduli(cfg.material)
        out["E_equiv_MPa"], out["nu_equiv"] = E, nu
        f = out.get("mass_scaling", {}).get("f", 1.0)
        out["c0_mm_s"] = an.scaled_wave_speed(E, cfg.material.rho, 1.0)
        out["c_scaled_mm_s"] = an.scaled_wave_speed(E, cfg.material.rho, f)
    except Exception as exc:
        out["moduli_error"] = str(exc)

    try:
        depth = abs(float(cfg.scratch.scratch_depth))
        R = float(cfg.indenter.tip_radius)
        a = an.contact_radius_rockwell(depth, R, cfg.indenter.cone_angle)
        h = min(float(cfg.mesh.fine_size_x), float(cfg.mesh.fine_size_z))
        out["contact_radius_mm"] = a
        out["N_a"] = an.elements_per_contact_radius(a, h)
        out["elements_per_depth"] = depth / float(cfg.mesh.fine_size_y)
        out["v_scratch_mm_s"] = (float(cfg.scratch.scratch_length)
                                 / max(float(cfg.scratch.scratch_time), 1e-30))
    except Exception as exc:
        out["geometry_error"] = str(exc)

    try:
        out["amplitude_smoothing"] = an.amplitude_smoothing_window(cfg)
    except Exception as exc:
        out["amplitude_smoothing"] = {"error": str(exc)}

    if getattr(cfg.solver, "use_ALE", False):
        try:
            from ..Configuration.base import ale_remesh_courant
            out["ale_courant"] = ale_remesh_courant(cfg)
        except Exception as exc:
            out["ale_courant"] = None
            out["ale_error"] = str(exc)

    if bench is not None:
        try:
            out["bench"] = bench.to_dict()
        except Exception:
            out["bench"] = repr(bench)
    return out


# --------------------------------------------------------------------------
# Write / read / compare
# --------------------------------------------------------------------------

def write_manifest(directory, cfg, bench=None, extra=None, repo_dir=None,
                   filename=MANIFEST_NAME):
    plain, digest = config_fingerprint(cfg)
    man = {
        "schema_version": SCHEMA_VERSION,
        "config_md5": digest,
        "config": plain,
        "derived": derived_quantities(cfg, bench),
        "git": git_info(repo_dir),
        "env": environment_info(),
    }
    if extra:
        man["extra"] = _plain(extra)
    if directory and not os.path.isdir(directory):
        try:
            os.makedirs(directory)
        except OSError:
            pass
    path = os.path.join(directory or ".", filename)
    with open(path, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    print(">>> manifest written: %s (config_md5=%s, git=%s%s)"
          % (path, digest[:10],
             (man["git"].get("commit") or "n/a")[:10],
             "-DIRTY" if man["git"].get("dirty") else ""))
    return path


def read_manifest(path):
    if os.path.isdir(path):
        path = os.path.join(path, MANIFEST_NAME)
    with open(path, "r") as f:
        return json.load(f)


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = "%s.%s" % (prefix, k) if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            out[key] = json.dumps(v, sort_keys=True)
        else:
            out[key] = v
    return out


def diff_manifests(a, b, section="config"):
    """
    Exactly what differs between two runs. This is the tool that answers
    "why are these two numbers different?" in ten seconds instead of an hour.
    """
    fa, fb = _flatten(a.get(section, {})), _flatten(b.get(section, {}))
    keys = sorted(set(fa) | set(fb))
    diffs = []
    for k in keys:
        va, vb = fa.get(k, "<absent>"), fb.get(k, "<absent>")
        if va != vb:
            diffs.append({"key": k, "a": va, "b": vb})
    return diffs


def print_diff(a, b, section="config"):
    diffs = diff_manifests(a, b, section=section)
    if not diffs:
        print("No difference in section '%s' (config_md5 %s vs %s)."
              % (section, a.get("config_md5", "?")[:10], b.get("config_md5", "?")[:10]))
        return diffs
    print("%d difference(s) in section '%s':" % (len(diffs), section))
    for d in diffs:
        print("  %-55s  %s  ->  %s" % (d["key"], d["a"], d["b"]))
    return diffs