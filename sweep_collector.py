# Aggregate a sweep into a single tidy table (CPython, not Abaqus).
#
#   python3 sweep_collector.py /path/to/results_dir \
#           --design designs/glassy_pc_morris.csv \
#           --out sweep_glassy_pc.csv
#
# <results_dir> is walked recursively for *_Results.csv, so it can be any
# directory name and layout: point it at whatever you created on the cluster.
#
# [PATCH:values-backend] the QoI are now produced by results_values.py
# (extract_values), which replaces results_verifier.py. Two consequences:
#
#   1. The QoI names change. Mapping from the previous pipeline:
#          Fn_half_N          -> F_n          [N]
#          Ft_half_N          -> F_t          [N]
#          scof               -> SCOF_mean    [-]   (+ SCOF_std)
#          residual_depth_mm  -> h_r          [mm]
#          pile_up_mm         -> h_p          [mm]
#          (new)                 h_fp         [mm]  frontal pile-up
#
#   2. results_values.py returns VALUES, not verdicts. There is therefore no
#      quality status any more, and NO RUN IS EVER EXCLUDED on a quality
#      criterion. `status` is now an INTEGRITY flag only:
#          OK      -- the file was produced and parsed
#          FAIL    -- the run aborted, the file is a stub, or parsing raised
#      The energy diagnostics (KE/IE, AE/IE, ETOTAL drift, ALLPW, settling)
#      are written as plain numeric columns and reported downstream as
#      INDICATORS. They never gate anything, so coarse-mesh test runs stay
#      in the table.

import argparse
import csv
import os
import re
import sys

import numpy as np

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.path.abspath(".")
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)


# [PATCH:values-backend] STATUS_RANK removed with the verifier: there is no
# longer a hierarchy of verdicts to aggregate.
# Original:
#   QOI_WINDOW_START = 0.90
#   STATUS_RANK = {"PASS": 0, "INFO": 1, "SKIP": 1, "WARN": 2, "FAIL": 3}


def _import_results_values():
    """
    Locate results_values.py.

    [PATCH:values-backend] replaces _import_results_verifier. The module is
    looked up under its package paths first, then by walking the tree, so the
    collector keeps working from the cluster layout and from a flat checkout.
    """
    candidates = [
        "ScratchSimulation.AbaqusModel.Postprocessing.results_values",
        "ScratchSimulation.AbaqusModel.Verification.results_values",
        "ScratchSimulation.AbaqusModel.results_values",
        "results_values",
    ]
    for name in candidates:
        try:
            module = __import__(name, fromlist=["*"])
            if hasattr(module, "extract_values"):
                return module
        except ImportError:
            continue
    for root, _dirs, files in os.walk(os.path.dirname(_HERE)):
        if "results_values.py" in files:
            path = os.path.join(root, "results_values.py")
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("results_values", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            except Exception:
                break
    raise SystemExit("Could not import results_values.py. Pass its directory on PYTHONPATH.")


RV = _import_results_values()


# ----------------------------------------------------------------------
# Quantities of interest
# ----------------------------------------------------------------------

# [PATCH:values-backend] the whole compute_qoi / _series / _scratch_window /
# _contact_radius / _peak / _ratio block is gone: every one of those
# quantities is now computed inside results_values.py. Keeping a second
# implementation here is exactly how the two used to drift apart.

# Energy diagnostics emitted by results_values.energy_values. Listed here so
# the collector can guarantee the columns exist (NaN when a series is
# missing), and so downstream code has one place to read the names from.
QUALITY_KEYS = (
    "KE_IE_steady_max",
    "KE_IE_overall_max",
    "AE_IE_final",
    "ETOTAL_drift",
    "ALLPW",
    "KE_final_over_IE_peak",
)

# Header metadata copied into the table.
METADATA_KEYS = (
    "family",
    "fine_size_x",
    "fine_size_y",
    "fine_size_z",
    "mass_scale",
    "target_time_increment",
    "scratch_time",
    "scratch_depth",
    "depth_mode",
)


def read_stub_status(path, max_lines=40):
    """
    (run_status, fail_reason) read from the header of a *_Results.csv.

    An aborted run writes a file reduced to a header carrying
    `# run_status=FAILED`. Reading it BEFORE parsing carries the abort reason
    into the table instead of an opaque "no time-series rows".
    """
    status, reason = "", ""
    try:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= max_lines or not line.startswith("#"):
                    break
                s = line[1:].strip()
                if s.startswith("run_status="):
                    status = s.split("=", 1)[1].strip()
                elif s.startswith("fail_reason="):
                    reason = s.split("=", 1)[1].strip()
    except (IOError, OSError):
        pass
    return status, reason


# ----------------------------------------------------------------------
# Design join
# ----------------------------------------------------------------------

def read_design(path):
    rows = {}
    with open(path, "r") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    for rec in csv.DictReader(lines):
        rows[str(rec["id"]).strip()] = rec
    return rows


def _run_id(filename):
    m = re.search(r"Design_(\w+?)_Results\.csv$", filename)
    if m:
        return m.group(1)
    return filename[:-len("_Results.csv")] if filename.endswith("_Results.csv") else filename


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate a sweep into a tidy table.")
    ap.add_argument("results_dir", help="directory containing the *_Results.csv (walked recursively)")
    ap.add_argument("--design", default=None, help="design CSV to join on the run id")
    ap.add_argument("--out", default="sweep_table.csv")
    # [PATCH:values-backend] --no-verify removed: there is nothing left to verify.
    ap.add_argument("--z", type=float, default=None,
                    help="z position [mm] where the residual profile is measured "
                         "(default: results_values.SCRATCH_LENGTH, i.e. the exit "
                         "cross-section)")
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        raise SystemExit("Not a directory: %s" % args.results_dir)

    files = []
    for root, _dirs, names in os.walk(args.results_dir):
        for n in sorted(names):
            if n.endswith("_Results.csv"):
                files.append(os.path.join(root, n))
    if not files:
        raise SystemExit("No *_Results.csv found under %s" % args.results_dir)

    design = read_design(args.design) if args.design else {}
    records, n_read_err = [], 0

    for path in files:
        rid = _run_id(os.path.basename(path))
        rec = {"id": rid, "file": os.path.relpath(path, args.results_dir)}

        # An aborted run leaves a header-only stub; recognise it before parsing.
        stub_status, stub_reason = read_stub_status(path)
        if stub_status:
            rec["run_status"] = stub_status
            rec["status"] = "FAIL"
            rec["parse_error"] = stub_reason or "run aborted (no results)"
            if rid in design:
                for k, v in design[rid].items():
                    if k not in ("id", "file"):
                        rec.setdefault(k, v)
            records.append(rec)
            continue

        try:
            # [PATCH:values-backend] parse_results_csv is called once for the
            # header metadata and extract_values once for the QoI. Parsing
            # twice costs a few ms per file and removes any chance of this
            # collector drifting away from results_values.extract_values.
            metadata, _timeseries, _nodes = RV.parse_results_csv(path)
            values = RV.extract_values(path, args.z)

            for k in METADATA_KEYS:
                if metadata.get(k) is not None:
                    rec[k] = metadata[k]
            for k, v in values.items():
                if k != "file":
                    rec[k] = v
            for k in QUALITY_KEYS:
                rec.setdefault(k, float("nan"))

            rec["run_status"] = "OK"
            rec["status"] = "OK"
            rec["parse_error"] = ""
        except Exception as exc:
            n_read_err += 1
            rec["run_status"] = "OK"      # the file exists; the reader failed
            rec["parse_error"] = str(exc)[:160]
            rec["status"] = "FAIL"

        if rid in design:
            for k, v in design[rid].items():
                if k not in ("id", "file"):
                    rec.setdefault(k, v)
        elif design:
            rec["design_missing"] = 1
        records.append(rec)

    # A design point with NO file at all must still exist in the table,
    # otherwise "never launched" and "aborted" are indistinguishable
    # downstream and the coverage figure in the report is wrong.
    if design:
        _seen = set(r["id"] for r in records)
        for _rid in sorted(design):
            if _rid in _seen:
                continue
            _rec = {"id": _rid, "file": "", "status": "FAIL",
                    "run_status": "MISSING",
                    "parse_error": "no result file produced for this design point"}
            for k, v in design[_rid].items():
                if k not in ("id", "file"):
                    _rec.setdefault(k, v)
            records.append(_rec)

    lead = ["id", "family", "campaign", "method", "traj", "step", "moved", "sign",
            "status", "run_status"]
    keys = []
    for name in lead:
        if any(name in r for r in records):
            keys.append(name)
    for r in records:
        for k in r:
            if k not in keys:
                keys.append(k)

    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="", extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    n_ok = sum(1 for r in records if r.get("status") == "OK")
    n_aborted = sum(1 for r in records if r.get("run_status") == "FAILED")
    n_absent = sum(1 for r in records if r.get("run_status") == "MISSING")
    print("Collected %d runs (%d usable, %d read errors, %d aborted, %d absent) -> %s"
          % (len(records), n_ok, n_read_err, n_aborted, n_absent, args.out))
    print("No quality gate is applied: every parsed run is usable. The energy "
          "diagnostics are reported as indicators only (%s)."
          % ", ".join(QUALITY_KEYS))


if __name__ == "__main__":
    main()