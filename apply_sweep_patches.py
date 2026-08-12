# Exact-string patcher for the sweep infrastructure.
#
#   python3 apply_sweep_patches.py --root /path/to/ScratchSimulation [--dry-run]
#
# Preserves each file's own line endings, is idempotent, and refuses to write
# anything if a single anchor is missing.

import argparse
import ast
import os
import sys


FAMILIES_IMPORT_ANCHOR = """from .base import (Simulation_Config, Material_Config,
                   LinearElastic_Config, J2Plasticity_Config,
                   DruckerPrager_Config, Prony_Config, Friction_Config,
                   RateDependent_Config, gsell_jonas_table, natural_dt)
"""

FAMILIES_IMPORT_NEW = FAMILIES_IMPORT_ANCHOR + """from .sampling import SAMPLING_DP_UNIFIED
"""

# All four Drucker-Prager families point at the same unified 9-factor campaign.
# The campaign overwrites every material parameter and restores the
# target_time_increment scale factor, so they yield a strictly identical
# configuration: the host choice is free. semicrystalline_j2 is excluded on
# purpose (von Mises base, the model-form guard fires). The hyperelastic
# families keep sampling=None: out of scope.
_FAM_SAMPLING = [
    ("_semicrystalline_dp_config", "_SEMICRYSTALLINE_CHECKS", "SAMPLING_DP_UNIFIED"),
    ("_glassy_config", "_GLASSY_CHECKS", "SAMPLING_DP_UNIFIED"),
    ("_glassy_pc_config", "_GLASSY_CHECKS", "SAMPLING_DP_UNIFIED"),
    ("_glassy_pmma_config", "_GLASSY_CHECKS", "SAMPLING_DP_UNIFIED"),
]

RPS_DESIGN_ANCHOR = "def model_study(mu0=2.2, K_mu=55.0):\n"

RPS_DESIGN_NEW = '''# Screening / sweep design produced by generate_design.py. Resolved relative to
# THIS script because run_parameter_study() chdirs into runs/ afterwards.
DEFAULT_DESIGN_DIR = "designs"


def _design_path(family_key, csv_path=None):
    if csv_path:
        return csv_path
    for method in ("morris", "sobol"):
        p = os.path.join(_HERE, DEFAULT_DESIGN_DIR, "%s_%s.csv" % (family_key, method))
        if os.path.exists(p):
            return p
    raise SystemExit(
        "No design file for family '%s' in %s (expected <family>_morris.csv or "
        "<family>_sobol.csv). Generate it with:\\n"
        "    python3 generate_design.py %s"
        % (family_key, os.path.join(_HERE, DEFAULT_DESIGN_DIR), family_key))


def _load_design(family_key, csv_path=None):
    # Reads the g_* (dimensionless group) columns only: the material parameters
    # are re-derived in-kernel by families.py so the CSV cannot drift from it.
    import csv as _csv
    path = _design_path(family_key, csv_path)
    with open(path, "r") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    rows = []
    for rec in _csv.DictReader(lines):
        groups = {}
        for k, v in rec.items():
            if k and k.startswith("g_") and v not in (None, ""):
                groups[k[2:]] = float(v)
        if not groups:
            raise SystemExit("Design %s has no g_* column." % path)
        rows.append({"id": rec["id"], "groups": groups})
    print(">>> Loaded %d design points from %s" % (len(rows), path))
    return rows


def design_study(family_key, csv_path=None):
    """
    Screening / sweep driven by a design generated with generate_design.py.
    Every case is applied through the family's own sampling campaign, which
    validates the model form and rebuilds the yield table, the friction model
    and the target time increment. Run it as:
        abaqus cae noGUI=run_parameter_study.py -- design glassy_pc
    """
    fam = get_family(family_key)
    spec = getattr(fam, "sampling", None)
    if spec is None:
        raise SystemExit(
            "Family '%s' carries no sampling campaign. Campaign hosts are the "
            "families with sampling=... in families.py; the other families of "
            "the same model class are covered by their host's ranking."
            % family_key)
    cases = _load_design(family_key, csv_path)
    print(">>> Campaign %s (%s): %d factors %s"
          % (spec.campaign, spec.label, spec.dim, spec.names))

    def apply(cfg, case):
        spec.configure(cfg, case["groups"])

    return ParameterStudy(
        name="DesignSweep",
        cases=cases,
        apply_case=apply,
        label=lambda case: "Design_%s" % case["id"],
    )


'''

RPS_LOOP_ANCHOR = '''    n_total = len(cases)
    for i, case in enumerate(cases, start=1):

        study.apply_case(cfg, case)
        stem = study.label(case)

        model, substrate_part = build_scratch_model(cfg)
        SubstrateMaterialAssignment(model, substrate_part, cfg).apply()



        run_job_and_wait(cfg.job_name, cfg)
        post_process(cfg.job_name, stem, cfg)

        mdb.close()   

        if output_subdir and stem != cfg.job_name:
            for ext in move_exts:
                src = cfg.job_name + ext
                if os.path.exists(src):
                    shutil.move(src, os.path.join(output_subdir, stem + ext))

        print(">>> [%d/%d] %s -> %s done." % (i, n_total, study.name, stem))

'''

RPS_LOOP_NEW = '''    n_total = len(cases)
    failed_cases = []
    for i, case in enumerate(cases, start=1):

        study.apply_case(cfg, case)
        stem = study.label(case)

        # A parameter sweep explores corners where Abaqus can legitimately
        # abort (element distortion, contact instability). Without this
        # guard a single aborted job would kill every remaining case of the
        # chunk. SystemExit is NOT caught: configuration errors must stop.
        case_error = None
        try:
            model, substrate_part = build_scratch_model(cfg)
            SubstrateMaterialAssignment(model, substrate_part, cfg).apply()

            run_job_and_wait(cfg.job_name, cfg)
            post_process(cfg.job_name, stem, cfg)
        except Exception as _exc:
            case_error = ("%s: %s" % (type(_exc).__name__, _exc))[:300]

        try:
            mdb.close()
        except Exception:
            pass

        if case_error is not None:
            failed_cases.append((stem, case_error))
            print(">>> [%d/%d] %s -> %s FAILED, continuing. %s"
                  % (i, n_total, study.name, stem, case_error))
            for ext in (move_exts + (".lck",)):
                stale = cfg.job_name + ext
                if os.path.exists(stale):
                    try:
                        os.remove(stale)
                    except OSError:
                        pass
            continue

        if output_subdir and stem != cfg.job_name:
            for ext in move_exts:
                src = cfg.job_name + ext
                if os.path.exists(src):
                    shutil.move(src, os.path.join(output_subdir, stem + ext))

        print(">>> [%d/%d] %s -> %s done." % (i, n_total, study.name, stem))

    if failed_cases:
        print(">>> %d of %d case(s) FAILED in this chunk:"
              % (len(failed_cases), n_total))
        for stem, msg in failed_cases:
            print(">>>    %s : %s" % (stem, msg))
        try:
            with open("FAILED_CASES.txt", "a") as _f:
                for stem, msg in failed_cases:
                    _f.write("%s\\t%s\\n" % (stem, msg))
        except (IOError, OSError):
            pass

'''

RPS_STUDIES_ANCHOR = '    "material":   lambda: material_study(_load_material_parameters()),\n'
RPS_STUDIES_NEW = (RPS_STUDIES_ANCHOR +
                   '    "design":     lambda: design_study(_selected_family()),\n')

RPS_DOC_ANCHOR = '# " abaqus cae noGUI=run_parameter_study.py -- friction "\n'
RPS_DOC_NEW = (RPS_DOC_ANCHOR +
               '# " abaqus cae noGUI=run_parameter_study.py -- design <family> "\n')

# The project has shipped this tuple both wrapped and on a single line, so try
# every known layout rather than assuming one.
_LAUNCH_TAIL = '"gsell_h", "target_dt")'
LAUNCH_FORMS = (
    '_VALID_STUDIES = ("single", "mesh", "mass_scale", "friction", "material", '
    '"models", "depth", ' + _LAUNCH_TAIL,
    '_VALID_STUDIES = ("single", "mesh", "mass_scale", "friction", "material",\n'
    '                  "models", "depth", ' + _LAUNCH_TAIL,
)

DESIGN_JOBS_DEFAULT = 5

LAUNCH_CONST_ANCHOR = ('SWEEP_JOBS = 8                  '
                       '# number of jobs for the "material" sweep \n')
LAUNCH_CONST_NEW = (LAUNCH_CONST_ANCHOR +
                    'DESIGN_JOBS = %d                  '
                    '# number of jobs for the "design" sweep\n'
                    % DESIGN_JOBS_DEFAULT)

# expand_jobs() hard-codes the split on study == "material"; generalise it so
# "design" is chunked the same way, each study keeping its own job count.
LAUNCH_SPLIT_ANCHOR = '''        if study == "material" and SWEEP_JOBS > 1:
            for i in range(SWEEP_JOBS):
                label = "%s_c%03dof%03d" % (base_label, i, SWEEP_JOBS)
                out.append((label, [study, family, "%d/%d" % (i, SWEEP_JOBS)]
                            + tag_tok + over))
        else:
            out.append((base_label, [study, family] + tag_tok + over))
'''

LAUNCH_SPLIT_NEW = '''        n_chunks = {"material": SWEEP_JOBS, "design": DESIGN_JOBS}.get(study, 1)
        if n_chunks > 1:
            for i in range(n_chunks):
                label = "%s_c%03dof%03d" % (base_label, i, n_chunks)
                out.append((label, [study, family, "%d/%d" % (i, n_chunks)]
                            + tag_tok + over))
        else:
            out.append((base_label, [study, family] + tag_tok + over))
'''


def _read(path):
    with open(path, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    return raw.decode("utf-8").replace("\r\n", "\n"), crlf


def _write(path, text, crlf):
    out = text.replace("\n", "\r\n") if crlf else text
    with open(path, "wb") as f:
        f.write(out.encode("utf-8"))


def _sub(text, anchor, new, label, report):
    if new in text:
        report.append(("skip", label))
        return text, True
    if anchor not in text:
        report.append(("MISS", label))
        return text, False
    if text.count(anchor) != 1:
        report.append(("AMBIG (%d matches)" % text.count(anchor), label))
        return text, False
    report.append(("apply", label))
    return text.replace(anchor, new, 1), True


def patch_families(text, report):
    ok = True
    text, o = _sub(text, FAMILIES_IMPORT_ANCHOR, FAMILIES_IMPORT_NEW,
                   "families.py: import sampling specs", report)
    ok = ok and o
    for factory, checks, spec in _FAM_SAMPLING:
        anchor = ("    config_factory=%s,\n    checks=%s,\n    sampling=None,\n"
                  % (factory, checks))
        new = ("    config_factory=%s,\n    checks=%s,\n    sampling=%s,\n"
               % (factory, checks, spec))
        text, o = _sub(text, anchor, new, "families.py: %s -> %s" % (factory, spec), report)
        ok = ok and o
    return text, ok


def patch_run_parameter_study(text, report):
    ok = True
    for anchor, new, label in (
            (RPS_DOC_ANCHOR, RPS_DOC_NEW, "run_parameter_study.py: usage header"),
            (RPS_DESIGN_ANCHOR, RPS_DESIGN_NEW + RPS_DESIGN_ANCHOR,
             "run_parameter_study.py: design_study()"),
            (RPS_LOOP_ANCHOR, RPS_LOOP_NEW,
             "run_parameter_study.py: fault-tolerant case loop"),
            (RPS_STUDIES_ANCHOR, RPS_STUDIES_NEW, "run_parameter_study.py: STUDIES entry")):
        text, o = _sub(text, anchor, new, label, report)
        ok = ok and o
    return text, ok


def patch_launcher(text, report):
    ok, done = True, False
    label = "launch_cluster_jobs.py: _VALID_STUDIES"
    for anchor in LAUNCH_FORMS:
        if anchor in text:
            text, o = _sub(text, anchor, anchor[:-1] + ', "design")',
                           label, report)
            ok, done = (ok and o), True
            break
    if not done:
        if '"target_dt", "design")' in text:
            report.append(("skip", label))
        else:
            report.append(("MISS", label + " (unrecognised layout)"))
            ok = False

    text, o = _sub(text, LAUNCH_CONST_ANCHOR, LAUNCH_CONST_NEW,
                   "launch_cluster_jobs.py: DESIGN_JOBS constant", report)
    ok = ok and o
    text, o = _sub(text, LAUNCH_SPLIT_ANCHOR, LAUNCH_SPLIT_NEW,
                   "launch_cluster_jobs.py: expand_jobs() chunking", report)
    ok = ok and o
    return text, ok


TARGETS = [
    ("families.py", ("AbaqusModel/Configuration", "Configuration", "."), patch_families),
    ("run_parameter_study.py", (".",), patch_run_parameter_study),
    ("launch_cluster_jobs.py", (".",), patch_launcher),
]


def _locate(root, name):
    for dirpath, _dirs, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="ScratchSimulation directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    report, staged, failed = [], [], False
    for name, _hints, fn in TARGETS:
        path = _locate(args.root, name)
        if path is None:
            report.append(("MISS", "%s not found under %s" % (name, args.root)))
            failed = True
            continue
        text, crlf = _read(path)
        new_text, ok = fn(text, report)
        if not ok:
            failed = True
            continue
        try:
            ast.parse(new_text)
        except SyntaxError as exc:
            report.append(("SYNTAX", "%s: %s" % (name, exc)))
            failed = True
            continue
        staged.append((path, new_text, crlf, text != new_text))

    width = max(len(s) for s, _ in report)
    for status, label in report:
        print("  [%-*s] %s" % (width, status, label))

    if failed:
        print("\nREFUSING to write: at least one anchor was missing, ambiguous or "
              "produced invalid syntax. No file was modified.")
        return 1

    changed = [s for s in staged if s[3]]
    if args.dry_run:
        print("\nDry run: %d file(s) would change." % len(changed))
        return 0
    for path, new_text, crlf, did_change in staged:
        if did_change:
            _write(path, new_text, crlf)
            print("\nWrote %s (%s)" % (path, "CRLF" if crlf else "LF"))
    if not changed:
        print("\nNothing to do: all patches already applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())