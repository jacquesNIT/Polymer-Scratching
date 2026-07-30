# -*- coding: utf-8 -*-
"""
Exact-string patcher for the ScratchSimulation V&V benchmark suite.

    python3 apply_benchmark_patches.py            # dry run, reports only
    python3 apply_benchmark_patches.py --write    # apply

Conventions kept from the previous patch campaigns:
  * per-file byte-level EOL preservation (every file below is CRLF);
  * idempotent -- a second run reports "already applied" and writes nothing;
  * refuses to write anything if ANY anchor is missing in ANY file;
  * AST-parses every patched file before committing it to disk;
  * comments rather than deletes.

Six patches, all of them consequences of the audit that produced the
benchmark suite:

  P1 extractor.py       CAREA / CFNM / CFSM written to the CSV.
                        CAREA is ALREADY requested by
                        Modelbuilder._request_contact_pair_history but never
                        extracted. With a Briscoe law
                            SCOF = alpha + tau0 * A_c / Fn
                        so the SCOF mesh drift IS the contact-area mesh drift.
                        Without CAREA that is inferred; with it, measured.

  P2 extractor.py       derived quantities in the CSV header: dt_nat, the
                        EFFECTIVE mass-scaling factor f, dm/m, N_a = a/h,
                        elements per depth, and the amplitude-smoothing
                        window. None of these is recoverable from the config
                        fields alone, and all of them govern the result.

  P3 physic_verifier.py check_dynamics used solver.mass_scale even under
                        VARIABLE mass scaling. Every family sets
                        target_time_increment > 0 in families.py, so the
                        reported scaled wave speed was wrong on EVERY
                        production run (mass_scale=500 assumed, while
                        f = (dt_target/dt_nat)^2 = 225 or 900 was applied).

  P4 substrate.py       build-time guard on coarse_size_1 vs the fine sizes.
                        seedEdgeByBias is called with maxSize=coarse_size_1
                        and minSize=fine_size_*; when the fine size reaches
                        coarse_size_1 Abaqus aborts. That is the
                        seedEdgeByBias:193 crash, and the two commented-out
                        entries of DEFAULT_MESH_SIZES (0.04, 0.03) would
                        reproduce it against coarse_size_1 = 0.028.

  P5 launch_cluster_jobs.py  new override aliases, notably
                        smoothing -> scratch.amplitude_smoothing and
                        unload_time / recovery_time. The SMOOTH window is
                        smooth * min(scratch_time, unload_time); unload_time
                        is FIXED at 0.01 s while scratch_time is swept, so the
                        window is 5 % of the scratch at T=0.05 and 25 % at
                        T=0.01. Without these aliases a scratch-time study
                        cannot be run at constant load path.

  P6 run_parameter_study.py  a manifest.json per case (git commit + dirty
                        flag, config fingerprint, derived quantities). This is
                        what makes two results comparable at all.
"""

from __future__ import print_function

import ast
import os
import sys

FILES = {
    "extractor.py": "Postprocessing",
    "physic_verifier.py": "Verification",
    "substrate.py": "Geometry",
    "launch_cluster_jobs.py": ".",
    "run_parameter_study.py": ".",
}


# --------------------------------------------------------------------------
# Patch definitions: (file, tag, anchor, replacement, sentinel)
# The sentinel is a string whose presence means the patch is already applied.
# --------------------------------------------------------------------------

P1_ANCHOR_A = '    cfs3 = _resample(t_cp, _pick(contact_data, "CFS3", z_cp), time_arr)\n'
P1_NEW_A = P1_ANCHOR_A + (
    '    # CAREA / CFNM / CFSM were already REQUESTED by\n'
    '    # Modelbuilder._request_contact_pair_history but never extracted. With a\n'
    '    # Briscoe law  SCOF = alpha + tau0 * A_c / Fn , so the contact area IS the\n'
    '    # SCOF: without this column the mesh drift of the SCOF can only be inferred.\n'
    '    cfnm = _resample(t_cp, _pick(contact_data, "CFNM", z_cp), time_arr)\n'
    '    cfsm = _resample(t_cp, _pick(contact_data, "CFSM", z_cp), time_arr)\n'
    '    carea = _resample(t_cp, _pick(contact_data, "CAREA", z_cp), time_arr)\n'
)

P1_ANCHOR_B = ('            "CFN1", "CFN2", "CFN3", "CFS1", "CFS2", "CFS3",'
               '  # contact-pair force (force-driven mode)\n')
P1_NEW_B = ('            "CFN1", "CFN2", "CFN3", "CFS1", "CFS2", "CFS3",'
            '  # contact-pair force (force-driven mode)\n'
            '            "CFNM", "CFSM", "CAREA",'
            '                     # contact magnitudes + contact AREA\n')

P1_ANCHOR_C = '            cfn1, cfn2, cfn3, cfs1, cfs2, cfs3,\n'
P1_NEW_C = '            cfn1, cfn2, cfn3, cfs1, cfs2, cfs3,\n            cfnm, cfsm, carea,\n'

P2_ANCHOR = '        f.write("# WallclockTime=%.2f s\\n" % wallclock)\n'
P2_NEW = P2_ANCHOR + '''
        # --- derived quantities -------------------------------------------
        # These govern the result but appear in NO config field, so two CSVs
        # written without them are not comparable. Written one per line as
        # "# key=value" so results_verifier.parse_results_csv picks them up.
        try:
            from ScratchSimulation.AbaqusModel.Verification.analytic import (
                mass_scaling_factor, amplitude_smoothing_window,
                contact_radius_rockwell, elements_per_contact_radius)
            _ms = mass_scaling_factor(cfg)
            f.write("# natural_dt=%.6e\\n" % _ms["dt_nat"])
            f.write("# mass_factor_eff=%.6e\\n" % _ms["f"])
            f.write("# dm_over_m=%.6e\\n" % _ms["dm_over_m"])
            f.write("# dt_effective=%.6e\\n" % _ms["dt_eff"])
            _depth = abs(float(scratch.scratch_depth))
            _a = contact_radius_rockwell(_depth, indenter.tip_radius,
                                         indenter.cone_angle)
            f.write("# contact_radius=%.6e\\n" % _a)
            f.write("# N_a=%.6e\\n" % elements_per_contact_radius(
                _a, min(mesh.fine_size_x, mesh.fine_size_z)))
            f.write("# elements_per_depth=%.6e\\n" % (_depth / mesh.fine_size_y))
            _sm = amplitude_smoothing_window(cfg)
            if _sm.get("w") is not None:
                f.write("# smooth_window=%.6e\\n" % _sm["w"])
                f.write("# smooth_window_rel=%.6e\\n" % _sm["w_rel"])
        except Exception:
            f.write("# derived_quantities=unavailable\\n")
'''

P3_ANCHOR = '''    c0 = np.sqrt(props["E_0"] / rho)                      # [mm/s] bar wave speed
    f = max(float(solver.mass_scale), 1.0)
    c_scaled = c0 / np.sqrt(f)
'''
P3_NEW = '''    c0 = np.sqrt(props["E_0"] / rho)                      # [mm/s] bar wave speed
    # OLD (wrong under VARIABLE mass scaling, i.e. on every family, since
    # families.py always sets target_time_increment > 0):
    #     f = max(float(solver.mass_scale), 1.0)
    # With SEMI_AUTOMATIC / THROUGHOUT_STEP / BELOW_MIN the solver drives dt to
    # target_time_increment, so the factor actually applied is
    #     f = (dt_target / dt_nat)^2
    # which for s = 15 is 225 and for s = 30 is 900 -- not mass_scale = 500.
    _target = float(getattr(solver, "target_time_increment", 0.0) or 0.0)
    if _target > 0.0:
        _h_min = min(float(msh.fine_size_x), float(msh.fine_size_y),
                     float(msh.fine_size_z))
        _dt_nat = _h_min / np.sqrt(props["E_0"] / rho)
        f = max((_target / _dt_nat) ** 2, 1.0)
    else:
        f = max(float(solver.mass_scale), 1.0)
    c_scaled = c0 / np.sqrt(f)
'''

P4_ANCHOR = '    # Edge seeds : biased transitions \n'
P4_NEW = '''    # Build-time guard on the biased transitions.
    # seedEdgeByBias below is called with maxSize=coarse_size_* and
    # minSize=fine_size_*. When a fine size reaches (or exceeds) the coarse
    # bound, Abaqus aborts inside seedEdgeByBias -- this is the crash seen at
    # this line, NOT a consequence of any material patch. With
    # polymer_default's coarse_size_1 = 0.028, the two commented-out entries
    # of DEFAULT_MESH_SIZES (0.04 and 0.03) would both reproduce it.
    # Failing here, with the numbers in the message, costs one second instead
    # of one queued job.
    _c1, _c2 = float(msh.coarse_size_1), float(msh.coarse_size_2)
    for _lbl, _fine, _coarse in (("fine_size_y", float(msh.fine_size_y), _c1),
                                 ("fine_size_z", float(msh.fine_size_z), _c1),
                                 ("fine_size_x", float(msh.fine_size_x), _c2)):
        if _fine >= _coarse:
            raise ValueError(
                "mesh_substrate: %s = %g is >= its biased-transition maxSize "
                "%g. seedEdgeByBias requires minSize < maxSize; increase "
                "coarse_size_1/coarse_size_2 or refine the fine mesh."
                % (_lbl, _fine, _coarse))

    # Edge seeds : biased transitions 
'''

P5_ANCHOR = '    "z_size":         "mesh.fine_size_z",\n'
P5_NEW = '''    "z_size":         "mesh.fine_size_z",
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
'''

P6_ANCHOR_IMPORT = ('from ScratchSimulation.AbaqusModel.utils import '
                    'run_job_and_wait, cleanup_abaqus_junk\n')
P6_NEW_IMPORT = (P6_ANCHOR_IMPORT +
                 'from ScratchSimulation.AbaqusModel.Verification.manifest '
                 'import write_manifest\n')

P6_ANCHOR_CALL = '        run_job_and_wait(cfg.job_name, cfg)\n'
P6_NEW_CALL = '''        # One manifest per case: git commit + dirty flag, a fingerprint of the
        # FULLY RESOLVED configuration, and the derived quantities (effective
        # mass-scaling factor, dm/m, N_a, smoothing window). Two results are
        # comparable only when it is possible to PROVE what differs between
        # them; without this, that is an archaeology exercise.
        try:
            write_manifest(".", cfg, extra={"case_stem": stem, "study": study.name},
                           filename="manifest_%s.json" % stem)
        except Exception as _exc:
            print("Warning: manifest not written for %s (%s)." % (stem, _exc))

        run_job_and_wait(cfg.job_name, cfg)
'''

PATCHES = [
    ("extractor.py", "P1a CAREA/CFNM/CFSM series", P1_ANCHOR_A, P1_NEW_A,
     'carea = _resample('),
    ("extractor.py", "P1b CSV header", P1_ANCHOR_B, P1_NEW_B, '"CFNM", "CFSM", "CAREA",'),
    ("extractor.py", "P1c CSV rows", P1_ANCHOR_C, P1_NEW_C, 'cfnm, cfsm, carea,'),
    ("extractor.py", "P2 derived quantities in header", P2_ANCHOR, P2_NEW,
     '# natural_dt=%.6e'),
    ("physic_verifier.py", "P3 effective mass-scaling factor", P3_ANCHOR, P3_NEW,
     '_target = float(getattr(solver, "target_time_increment"'),
    ("substrate.py", "P4 seedEdgeByBias guard", P4_ANCHOR, P4_NEW,
     'seedEdgeByBias requires minSize < maxSize'),
    ("launch_cluster_jobs.py", "P5 override aliases", P5_ANCHOR, P5_NEW,
     '"smoothing":      "scratch.amplitude_smoothing"'),
    ("run_parameter_study.py", "P6a manifest import", P6_ANCHOR_IMPORT,
     P6_NEW_IMPORT, 'from ScratchSimulation.AbaqusModel.Verification.manifest'),
    ("run_parameter_study.py", "P6b manifest call", P6_ANCHOR_CALL, P6_NEW_CALL,
     'write_manifest(".", cfg, extra={"case_stem": stem'),
]


# --------------------------------------------------------------------------
# Machinery
# --------------------------------------------------------------------------

def detect_eol(raw):
    if b"\r\n" in raw:
        return "\r\n"
    if b"\r" in raw:
        return "\r"
    return "\n"


def load(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    eol = detect_eol(raw)
    text = raw.decode("utf-8", "surrogateescape")
    return text.replace("\r\n", "\n").replace("\r", "\n"), eol, raw


def save(path, text, eol):
    body = text.replace("\n", eol) if eol != "\n" else text
    with open(path, "wb") as fh:
        fh.write(body.encode("utf-8", "surrogateescape"))


def resolve(root, filename):
    """Find the file anywhere under root (the package layout is nested)."""
    for dirpath, _dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None


def main(argv):
    write = "--write" in argv
    root = None
    for a in argv[1:]:
        if not a.startswith("--"):
            root = a
    root = root or os.getcwd()

    print("Root: %s" % os.path.abspath(root))
    print("Mode: %s\n" % ("WRITE" if write else "DRY RUN"))

    files = {}
    missing = []
    for fname in sorted(FILES):
        p = resolve(root, fname)
        if p is None:
            missing.append(fname)
        else:
            files[fname] = p
    if missing:
        print("ERROR: file(s) not found under the root: %s" % ", ".join(missing))
        return 2

    texts, eols = {}, {}
    for fname, path in files.items():
        texts[fname], eols[fname], _raw = load(path)
        print("  %-26s %s  (EOL %s)"
              % (fname, os.path.relpath(path, root),
                 "CRLF" if eols[fname] == "\r\n" else "LF"))
    print("")

    planned, skipped, failed = [], [], []
    for fname, tag, anchor, new, sentinel in PATCHES:
        text = texts[fname]
        if sentinel in text:
            skipped.append((fname, tag))
            continue
        n = text.count(anchor)
        if n != 1:
            failed.append((fname, tag, n))
            continue
        texts[fname] = text.replace(anchor, new, 1)
        planned.append((fname, tag))

    for fname, tag in skipped:
        print("  [SKIP] %-26s %s -- already applied" % (fname, tag))
    for fname, tag, n in failed:
        print("  [FAIL] %-26s %s -- anchor found %d time(s), expected 1"
              % (fname, tag, n))
    for fname, tag in planned:
        print("  [ OK ] %-26s %s" % (fname, tag))

    if failed:
        print("\nABORT: at least one anchor is missing or ambiguous. "
              "Nothing was written.\n"
              "The file has drifted from the version this patcher was written "
              "against; re-derive the anchor rather than forcing it.")
        return 3

    print("\nAST validation:")
    for fname in sorted(set(f for f, _ in planned)):
        try:
            ast.parse(texts[fname])
            print("  [ OK ] %s parses" % fname)
        except SyntaxError as exc:
            print("  [FAIL] %s: %s" % (fname, exc))
            return 4

    if not planned:
        print("\nNothing to do (all patches already applied).")
        return 0

    if not write:
        print("\nDry run complete: %d patch(es) ready. Re-run with --write."
              % len(planned))
        return 0

    for fname in sorted(set(f for f, _ in planned)):
        save(files[fname], texts[fname], eols[fname])
        raw = open(files[fname], "rb").read()
        eol_ok = detect_eol(raw) == eols[fname]
        print("  written: %-26s EOL preserved: %s" % (fname, eol_ok))
        if not eol_ok:
            print("    WARNING: line endings changed -- inspect before committing.")

    print("\n%d patch(es) applied." % len(planned))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))