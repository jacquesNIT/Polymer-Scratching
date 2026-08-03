# -*- coding: utf-8 -*-
# Multi-job launcher for ScratchSimulation.

# "python3 launch_cluster_jobs.py" (from the ScratchSimulation directory)

# For every JOBS: 
#  job_<label>.py - driver that runs run_parameter_study                           
#  submit_<label>.sh - copy of submit.sh with 'job_<label>'

import os
import re
import subprocess
import sys
import time

JOBS = [
    
    #("mesh", "glassy_pc", {"tag": "mesh12", "ALE": False, "scratch_time": 0.005, "distortion": True, "length": 0.1}),
    #("mesh", "glassy_pmma", {"tag": "mesh12", "ALE": False, "scratch_time": 0.005, "distortion": True, "length": 0.1}),
    #("mesh", "semicrystalline_j2", {"tag": "mesh12", "ALE": False, "scratch_time": 0.005, "distortion": True, "length": 0.1}),
    #("mesh", "semicrystalline_dp", {"tag": "mesh12", "ALE": False, "scratch_time": 0.005, "distortion": True, "length": 0.1}),

    ("mesh", "glassy_pc", {"tag": "mesh13", "ALE": False, "scratch_time": 0.01, "distortion": True, "length": 0.1}),
    ("mesh", "glassy_pmma", {"tag": "mesh13", "ALE": False, "scratch_time": 0.01, "distortion": True, "length": 0.1}),
    ("mesh", "semicrystalline_j2", {"tag": "mesh13", "ALE": False, "scratch_time": 0.01, "distortion": True, "length": 0.1}),
    ("mesh", "semicrystalline_dp", {"tag": "mesh13", "ALE": False, "scratch_time": 0.01, "distortion": True, "length": 0.1}),

    ("mesh", "glassy_pc", {"tag": "mesh14", "ALE": False, "scratch_time": 0.025, "distortion": True, "length": 0.1}),
    #("mesh", "glassy_pmma", {"tag": "mesh14", "ALE": False, "scratch_time": 0.025, "distortion": True, "length": 0.1}),
    ("mesh", "semicrystalline_j2", {"tag": "mesh14", "ALE": False, "scratch_time": 0.025, "distortion": True, "length": 0.1}),
    #("mesh", "semicrystalline_dp", {"tag": "mesh14", "ALE": False, "scratch_time": 0.025, "distortion": True, "length": 0.1}),

    ("mesh", "glassy_pc", {"tag": "mesh11", "ALE": False, "scratch_time": 0.003, "distortion": True, "length": 0.1}),
    ("mesh", "glassy_pmma", {"tag": "mesh11", "ALE": False, "scratch_time": 0.003, "distortion": True, "length": 0.1}),
    ("mesh", "semicrystalline_j2", {"tag": "mesh11", "ALE": False, "scratch_time": 0.003, "distortion": True, "length": 0.1}),
    ("mesh", "semicrystalline_dp", {"tag": "mesh11", "ALE": False, "scratch_time": 0.003, "distortion": True, "length": 0.1}),

    #("mesh", "elastomer_mr", {"tag": "mesh9", "ALE": False, "scratch_time": 0.025, "hourglass": "RELAX STIFFNESS", "distortion": True, "length": 0.1}),
    #("mesh", "elastomer_ve", {"tag": "mesh9", "ALE": False, "scratch_time": 0.025, "hourglass": "RELAX STIFFNESS", "distortion": True, "length": 0.1}),

    #("mesh", "glassy_pc", {"tag": "mesh8", "ALE": False, "scratch_time": 0.05, "distortion": True, "length": 0.1}),
    #("mesh", "glassy_dp", {"tag": "mesh8", "ALE": False, "scratch_time": 0.05, "distortion": True, "length": 0.1}),
    #("mesh", "glassy_pmma", {"tag": "mesh8", "ALE": False, "scratch_time": 0.05, "distortion": True, "length": 0.1}),
    #("mesh", "semicrystalline_j2", {"tag": "mesh8", "ALE": False, "scratch_time": 0.05, "distortion": True, "length": 0.1}),
    #("mesh", "semicrystalline_dp", {"tag": "mesh8", "ALE": False, "scratch_time": 0.05, "distortion": True, "length": 0.1}),
    #("mesh", "elastomer_mr", {"tag": "mesh8", "ALE": False, "scratch_time": 0.05, "hourglass": "RELAX STIFFNESS", "distortion": True, "length": 0.1}),
    #("mesh", "elastomer_ve", {"tag": "mesh8", "ALE": False, "scratch_time": 0.05, "hourglass": "RELAX STIFFNESS", "distortion": True, "length": 0.1}),

    #("mesh", "glassy_pc", {"tag": "mesh5", "ALE": True, "scratch_time": 0.1}),
    #("mesh", "glassy_dp", {"tag": "mesh5", "ALE": True, "scratch_time": 0.1}),
    #("mesh", "glassy_pmma", {"tag": "mesh5", "ALE": True, "scratch_time": 0.1}),
    #("mesh", "semicrystalline_j2", {"tag": "mesh5", "ALE": True, "scratch_time": 0.1}),
    #("mesh", "semicrystalline_dp", {"tag": "mesh5", "ALE": True, "scratch_time": 0.1}),
    #("mesh", "elastomer_mr", {"tag": "mesh5", "ALE": True, "scratch_time": 0.1, "hourglass": "RELAX STIFFNESS"}),
    #("mesh", "elastomer_ve", {"tag": "mesh5", "ALE": True, "scratch_time": 0.1, "hourglass": "RELAX STIFFNESS"}),

    #("mesh", "glassy_dp", {"tag": "ALE20", "ALE": True, "scratch_time": 0.05, "freq": 20, "sweeps": 1}),
    #("mesh", "glassy_dp", {"tag": "ALE200", "ALE": True, "scratch_time": 0.05, "freq": 200, "sweeps": 3}),
    #("mesh", "glassy_dp", {"tag": "ALE650", "ALE": True, "scratch_time": 0.05, "freq": 650, "sweeps": 3}),
    #("mesh", "glassy_dp", {"tag": "ALE1300", "ALE": True, "scratch_time": 0.05, "freq": 1300, "sweeps": 5}),


    #("single", "glassy_pc", {"tag": "quick_test", "ALE": False, "scratch_time": 0.03}),

    #("mesh", "glassy_pc", {"tag": "Test_NewMeshing", "ALE": True, "scratch_time": 0.05}),

    #("single", "glassy_pc", {"tag": "Test_NewMeshing_z10", "ALE": True, "scratch_time": 0.05, "z_size": 0.01}),
    #("single", "glassy_pc", {"tag": "Test_NewMeshing_z15", "ALE": True, "scratch_time": 0.05, "z_size": 0.015}),
    #("single", "glassy_pc", {"tag": "Test", "ALE": True, "scratch_time": 0.001}),
    #("single", "glassy_pc", {"tag": "Test_NewMeshing_z30", "ALE": True, "scratch_time": 0.05, "z_size": 0.030}),

    #("mesh", "glassy_pc", {"tag": "distortion_base", "ALE": False, "scratch_time": 0.05, "distortion": False, "length": 0.1}),
    #("mesh", "glassy_pc", {"tag": "distortionF2", "ALE": False, "scratch_time": 0.05, "distortion": False, "length": 0.2}),
    #("mesh", "glassy_pc", {"tag": "distortionF3", "ALE": False, "scratch_time": 0.05, "distortion": False, "length": 0.3}),
    #("mesh", "glassy_pc", {"tag": "distortionT1", "ALE": True, "scratch_time": 0.05, "distortion": True, "length": 0.1}),
    #("mesh", "glassy_pc", {"tag": "distortionT2", "ALE": True, "scratch_time": 0.05, "distortion": True, "length": 0.2}),
    #("mesh", "glassy_pc", {"tag": "distortionT3", "ALE": True, "scratch_time": 0.05, "distortion": True, "length": 0.3}),
]
SWEEP_JOBS = 8                  # number of jobs for the "material" sweep 
SUBMIT_TEMPLATE = "submit.sh"
RELAY_PARTITION = "q64"
DRY_RUN = False                 # True for testing only


# Valid tokens 
_VALID_STUDIES = ("single", "mesh", "mass_scale", "friction", "material",
                  "models", "depth", "gsell_h", "target_dt")

_VALID_FAMILIES = ("elastomer_mr", "elastomer_ve", "semicrystalline_j2",
                   "semicrystalline_dp", "glassy_dp", "glassy_pc", "glassy_pmma")

DRIVER_TEMPLATE = '''# Auto-generated by launch_cluster_jobs.py -- do not edit (overwritten).
# Job: {label}   Tokens: {tokens}
import sys, os
sys.dont_write_bytecode = True
SCRIPT = {script!r}
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT)))
sys.argv = [SCRIPT, "--"] + {tokens!r}
exec(compile(open(SCRIPT).read(), SCRIPT, "exec"),
     {{"__name__": "__main__", "__file__": SCRIPT}})
'''

def read_template():
    if not os.path.exists(SUBMIT_TEMPLATE):
        raise SystemExit("Template '%s' not found." % SUBMIT_TEMPLATE)
    with open(SUBMIT_TEMPLATE, "rb") as f:
        raw = f.read()
    if b"run_parameter_study" not in raw:
        raise SystemExit("'%s' does not contain the word 'run_parameter_study' " % SUBMIT_TEMPLATE)
    return raw


def wrapper_cpus(raw):
    """Read '-c' from the wrapper line so the Abaqus CPU count follows submit.sh automatically (cpus = token)."""
    m = re.search(rb"run_parameter_study", raw)
    line = raw[:m.start()].rsplit(b"\n", 1)[-1] + raw[m.start():].split(b"\n", 1)[0]
    m = re.search(rb"-c\s+(\d+)", line)
    return int(m.group(1)) if m else None


# Names for per-job overrides
_OVERRIDE_ALIASES = {

    # Main Tests
    "ALE":            "solver.use_ALE",
    "scratch_time":   "scratch.scratch_time",
    "scratch_depth":  "scratch.scratch_depth",
    "scratch_length": "scratch.scratch_length",
    "mass_scale":     "solver.mass_scale",
    "target_dt":      "solver.target_time_increment",
    "hourglass":      "mesh.hourglass_control",

    # Side Tests
    "freq":           "solver.ale_frequency",
    "sweeps":         "solver.ale_mesh_sweeps",
    "distortion":     "mesh.distortion_control",
    "length":         "mesh.length_ratio",
    "z_size":         "mesh.fine_size_z",
    "x_size":         "mesh.fine_size_x",
    "y_size":         "mesh.fine_size_y",
    "coarse1":        "mesh.coarse_size_1",
    "coarse2":        "mesh.coarse_size_2",

    # Load-path controls. The SMOOTH window of the tabular amplitudes is
    #     smooth * min(scratch_time, unload_time)
    # and unload_time is FIXED at 0.01 s in polymer_default while
    # scratch_time is swept: the window is 5 % of the scratch at T = 0.05 s
    # and 25 % at T = 0.01 s. A scratch-time study without these aliases
    # therefore compares three DIFFERENT load paths, which is not a
    # quasi-staticity test. Set smoothing=0 (or scale unload_time with
    # scratch_time) to hold the path fixed.
    "smoothing":      "scratch.amplitude_smoothing",
    "unload_time":    "scratch.unload_time",
    "recovery_time":  "scratch.recovery_time",
    "indent_time":    "scratch.indentation_time",
}

# mesh_substrate() compares hourglass_control against these EXACT strings and
# silently falls back to DEFAULT on anything else -- so validate here, loudly.
_HOURGLASS_VALUES = ("DEFAULT", "ENHANCED", "RELAX STIFFNESS")


def _normalize_hourglass(value, entry):
    v = str(value).upper().replace("_", " ").strip()
    if v not in _HOURGLASS_VALUES:
        raise SystemExit("Bad hourglass value %r in JOBS entry %r. Valid: %s"
                         % (value, entry, ", ".join(_HOURGLASS_VALUES)))
    return v

# mesh_substrate() resolves distortion_control via str().upper(), so bool True/False
# AND these strings all work. Normalising here to ON/OFF/DEFAULT survives
# _parse_override_value() (which turns "ON"->True, "OFF"->False, "DEFAULT"->str).
_DISTORTION_VALUES = ("ON", "TRUE", "YES", "OFF", "FALSE", "NO", "DEFAULT")


def _normalize_distortion(value, entry):
    v = str(value).upper().replace("_", " ").strip()
    if v not in _DISTORTION_VALUES:
        raise SystemExit("Bad distortion value %r in JOBS entry %r. "
                         "Valid: ON/True, OFF/False, DEFAULT." % (value, entry))
    if v in ("ON", "TRUE", "YES"):
        return "ON"
    if v in ("OFF", "FALSE", "NO"):
        return "OFF"
    return "DEFAULT"


def _validate_length_ratio(value, entry):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise SystemExit("Bad length value %r in JOBS entry %r: expected a "
                         "number in (0, 1]." % (value, entry))
    if not (0.0 < v <= 1.0):
        raise SystemExit("Bad length value %r in JOBS entry %r: need 0 < r <= 1."
                         % (value, entry))
    return v

_TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")   # tag lands in file + job names


def _format_override(value):
    # bool first: isinstance(True, int) is True in Python.
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _opt_tokens(opts, entry):
    """opts dict -> (tag or None, ['set:path=value', ...] tokens)."""
    if not isinstance(opts, dict):
        raise SystemExit("Bad JOBS entry %r: third element must be a dict."
                         % (entry,))
    tag = None
    tokens = []
    for key, value in opts.items():
        if key == "tag":
            tag = str(value)
            if not _TAG_RE.match(tag):
                raise SystemExit("Bad tag %r in JOBS entry %r "
                                 "(letters, digits, _ . - only)." % (tag, entry))
            continue
        path = _OVERRIDE_ALIASES.get(key, key if "." in key else None)
        if path is None:
            raise SystemExit(
                "Unknown override key '%s' in JOBS entry %r. Use an alias (%s) "
                "or a dotted cfg path such as 'solver.use_ALE'."
                % (key, entry, ", ".join(sorted(_OVERRIDE_ALIASES))))
        if path == "mesh.hourglass_control":
            value = _normalize_hourglass(value, entry)
        elif path == "mesh.distortion_control":
            value = _normalize_distortion(value, entry)
        elif path == "mesh.length_ratio":
            value = _validate_length_ratio(value, entry)
        tokens.append("set:%s=%s" % (path, _format_override(value)))
    return tag, tokens


def expand_jobs():
    """JOBS entries -> list of (label, tokens). 'material' is split. Expands
    per-job opts (tag + overrides), validates tokens, rejects duplicate labels."""
    out = []
    for entry in JOBS:
        if len(entry) not in (2, 3):
            raise SystemExit("Bad JOBS entry %r: expected (study, family) or "
                             "(study, family, opts)." % (entry,))
        study, family = entry[0], entry[1]
        opts = entry[2] if len(entry) == 3 else {}
        if study not in _VALID_STUDIES:
            raise SystemExit("Unknown study '%s' in JOBS. Valid: %s"
                             % (study, ", ".join(_VALID_STUDIES)))
        if family not in _VALID_FAMILIES:
            raise SystemExit("Unknown family '%s' in JOBS. Valid: %s"
                             % (family, ", ".join(_VALID_FAMILIES)))
        tag, over = _opt_tokens(opts, entry)
        base_label = "%s_%s" % (study, family) + (("_%s" % tag) if tag else "")
        tag_tok = (["tag=%s" % tag] if tag else [])
        if study == "material" and SWEEP_JOBS > 1:
            for i in range(SWEEP_JOBS):
                label = "%s_c%03dof%03d" % (base_label, i, SWEEP_JOBS)
                out.append((label, [study, family, "%d/%d" % (i, SWEEP_JOBS)]
                            + tag_tok + over))
        else:
            out.append((base_label, [study, family] + tag_tok + over))

    labels = [lbl for lbl, _ in out]
    dups = sorted(set(l for l in labels if labels.count(l) > 1))
    if dups:
        raise SystemExit(
            "Duplicate job label(s) %s -- give each variant a distinct "
            "'tag' in its opts dict." % ", ".join(dups))
    return out


def main():
    raw = read_template()
    cpus = wrapper_cpus(raw)
    here = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(here, "run_parameter_study.py")

    jobs = expand_jobs()
    print(">>> %d job(s) to submit (template: %s%s)"
          % (len(jobs), SUBMIT_TEMPLATE,
             ", -c %d detected -> cpus token" % cpus if cpus else ""))

    for label, tokens in jobs:
        if cpus:
            tokens = tokens + ["cpus=%d" % cpus]

        # 1. driver
        driver = "job_%s.py" % label
        with open(os.path.join(here, driver), "w") as f:
            f.write(DRIVER_TEMPLATE.format(label=label, tokens=tokens, script=script))

        # 2. submit script ()= submit.sh with the word substituted)
        submit = "submit_%s.sh" % label
        body = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        body = body.replace(b"run_parameter_study", driver[:-3].encode())
        with open(os.path.join(here, submit), "wb") as f:
            f.write(body)

        # 3. sbatch
        cmd = ["sbatch", "--partition", RELAY_PARTITION, submit]
        print("  [%s]  %s" % (label, " ".join(cmd)))
        if not DRY_RUN:
            rc = subprocess.call(cmd, cwd=here)
            if rc != 0:
                print("  !! sbatch returned %d for %s" % (rc, label))
            elif label != jobs[-1][0]:      
                delay = 10                  
                time.sleep(delay)

    if DRY_RUN:
        print(">>> DRY_RUN: files generated, nothing submitted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
