# -*- coding: utf-8 -*-
"""
convert_results_csv.py
======================

Convert extended result CSVs produced by `extractor.py` ("Mesh_*_Results.csv")
into the reduced legacy format used by `sim_test.csv`.

Output format
-------------
All leading comment lines of the source file are preserved verbatim, in their
original order (provenance banner, indenter type, material parameters, family,
simulation parameters, wallclock time, numerical diagnostics, ...).

Only the data block is reduced, to the following columns:
    Time, RF1, RF2, RF3, IE, KE, NodeLabel,
    x_undeformed, y_undeformed, z_undeformed,
    x_deformed, y_deformed, z_deformed

Energy mapping: IE <- ALLIE, KE <- ALLKE (substrate-scoped energies).
Falls back to WM_ALLIE / WM_ALLKE when the substrate-scoped columns are absent
(controlled by --energy-scope).

Ragged trailing rows (empty time/force fields, only nodal coordinates filled)
are kept as-is, matching the layout of `sim_test.csv`.

Usage
-----
    python convert_results_csv.py <input_dir>
    python convert_results_csv.py <input_dir> -o <output_dir>
    python convert_results_csv.py in/ -o out/ --recursive --suffix _legacy
    python convert_results_csv.py in/ --no-blank-lines --strict
    python convert_results_csv.py in/ -o out/ --header-mode legacy

The input must be a directory containing .csv files; a single file path is
rejected. Input files are never modified.
"""

from __future__ import print_function

import argparse
import csv
import io
import os
import re
import sys


# --------------------------------------------------------------------------
# Target format configuration
# --------------------------------------------------------------------------

# (output_name, [source candidates, in order of preference])
TARGET_COLUMNS = [
    ("Time",          ["Time"]),
    ("RF1",           ["RF1"]),
    ("RF2",           ["RF2"]),
    ("RF3",           ["RF3"]),
    ("IE",            ["ALLIE", "IE", "WM_ALLIE"]),
    ("KE",            ["ALLKE", "KE", "WM_ALLKE"]),
    ("NodeLabel",     ["NodeLabel"]),
    ("x_undeformed",  ["x_undeformed"]),
    ("y_undeformed",  ["y_undeformed"]),
    ("z_undeformed",  ["z_undeformed"]),
    ("x_deformed",    ["x_deformed"]),
    ("y_deformed",    ["y_deformed"]),
    ("z_deformed",    ["z_deformed"]),
]

# Overrides applied by --energy-scope
ENERGY_SCOPE_MAP = {
    "substrate": {"IE": ["ALLIE", "IE"], "KE": ["ALLKE", "KE"]},
    "whole":     {"IE": ["WM_ALLIE"],    "KE": ["WM_ALLKE"]},
    "auto":      None,  # use TARGET_COLUMNS as declared
}

# Comment lines kept when --header-mode legacy is requested
LEGACY_HEADER_PATTERNS = [
    re.compile(r"^\s*#\s*Indenter\s+type\s*:", re.IGNORECASE),
    re.compile(r"^\s*#\s*Material\s+parameters\s*:", re.IGNORECASE),
    re.compile(r"^\s*#\s*WallclockTime\s*=", re.IGNORECASE),
]

EOL = "\r\n"


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def read_source(path, header_mode="all", encoding="utf-8"):
    """Return (header_lines, column_names, data_rows).

    header_lines : comment lines to reproduce in the output
                   - header_mode="all"    : every leading comment line, verbatim
                   - header_mode="legacy" : only indenter / material / wallclock
                   - header_mode="none"   : no comment line at all
    column_names : column names found in the source file
    data_rows    : list of lists of raw strings (values are never parsed)
    """
    with io.open(path, "r", encoding=encoding, newline="") as fh:
        raw = fh.read()

    # Line endings are normalised for parsing only; output is always CRLF.
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    all_comments = []
    legacy_comments = {}
    idx_header_row = None

    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if not stripped.strip():
            continue
        if stripped.lstrip().startswith("#"):
            all_comments.append(stripped)
            for rank, pat in enumerate(LEGACY_HEADER_PATTERNS):
                if pat.match(stripped.lstrip()) and rank not in legacy_comments:
                    legacy_comments[rank] = stripped
            continue
        # First non-empty, non-comment line is the column header row.
        idx_header_row = i
        break

    if idx_header_row is None:
        raise ValueError("No column header row found in %s" % path)

    if header_mode == "all":
        header_lines = all_comments
    elif header_mode == "legacy":
        header_lines = [legacy_comments[r] for r in sorted(legacy_comments)]
    else:
        header_lines = []

    column_names = next(csv.reader([lines[idx_header_row]]))
    column_names = [c.strip() for c in column_names]

    data_rows = []
    for line in lines[idx_header_row + 1:]:
        if line.strip() == "":
            continue          # blank lines are separators in the legacy format
        if line.lstrip().startswith("#"):
            continue          # trailing comments, if any, are dropped
        data_rows.append(next(csv.reader([line])))

    return header_lines, column_names, data_rows


# --------------------------------------------------------------------------
# Transformation
# --------------------------------------------------------------------------

def build_column_plan(column_names, energy_scope="auto", strict=False):
    """Return (plan, missing).

    plan    : list of (output_name, source_index or None)
    missing : output names with no matching source column
    """
    lookup = {}
    for i, name in enumerate(column_names):
        lookup.setdefault(name, i)                 # first occurrence wins
        lookup.setdefault(name.lower(), i)

    override = ENERGY_SCOPE_MAP.get(energy_scope)
    plan, missing = [], []

    for out_name, candidates in TARGET_COLUMNS:
        if override and out_name in override:
            candidates = override[out_name]

        src_idx = None
        for cand in candidates:
            if cand in lookup:
                src_idx = lookup[cand]
                break
            if cand.lower() in lookup:
                src_idx = lookup[cand.lower()]
                break

        if src_idx is None:
            missing.append(out_name)
        plan.append((out_name, src_idx))

    if strict and missing:
        raise KeyError("Source columns not found: %s" % ", ".join(missing))

    return plan, missing


def project_rows(data_rows, plan):
    """Apply the column plan to every data row (ragged rows stay ragged)."""
    out = []
    n_plan = len(plan)
    for row in data_rows:
        n = len(row)
        new_row = [""] * n_plan
        for k, (_, src_idx) in enumerate(plan):
            if src_idx is not None and src_idx < n:
                new_row[k] = row[src_idx]
        out.append(new_row)
    return out


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def write_target(path, header_lines, rows, blank_lines=True, encoding="utf-8"):
    out_header = [name for name, _ in TARGET_COLUMNS]
    sep = EOL + EOL if blank_lines else EOL

    with io.open(path, "w", encoding=encoding, newline="") as fh:
        for line in header_lines:
            fh.write(line + EOL)
        fh.write(",".join(out_header) + sep)
        for i, row in enumerate(rows):
            fh.write(",".join(row))
            fh.write(sep if i < len(rows) - 1 else EOL)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def convert_file(src_path, dst_path, energy_scope="auto", strict=False,
                 blank_lines=True, header_mode="all", encoding="utf-8"):
    header_lines, column_names, data_rows = read_source(src_path, header_mode,
                                                        encoding)
    plan, missing = build_column_plan(column_names, energy_scope, strict)
    rows = project_rows(data_rows, plan)
    write_target(dst_path, header_lines, rows, blank_lines, encoding)
    return len(rows), missing


def collect_inputs(in_path, recursive=False, pattern=None):
    """Return a list of (absolute_path, path_relative_to_root).

    `in_path` must be a directory; single files are not accepted.
    """
    if not os.path.isdir(in_path):
        raise NotADirectoryError("Input must be a directory: %s" % in_path)

    rx = re.compile(pattern) if pattern else None
    found = []
    if recursive:
        for root, _dirs, files in os.walk(in_path):
            for f in sorted(files):
                if f.lower().endswith(".csv") and (rx is None or rx.search(f)):
                    full = os.path.join(root, f)
                    found.append((full, os.path.relpath(full, in_path)))
    else:
        for f in sorted(os.listdir(in_path)):
            full = os.path.join(in_path, f)
            if os.path.isfile(full) and f.lower().endswith(".csv") \
                    and (rx is None or rx.search(f)):
                found.append((full, f))
    return found


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert extended result CSVs to the reduced legacy format "
                    "(Time,RF1,RF2,RF3,IE,KE,NodeLabel,coords), keeping the "
                    "original header comments.")
    p.add_argument("input", help="Input directory containing .csv files.")
    p.add_argument("-o", "--output", default=None,
                   help="Output directory (default: <input>_converted).")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="Walk sub-directories, preserving the tree structure.")
    p.add_argument("--pattern", default=None,
                   help="Regex filter on the file name, e.g. '^Mesh_'.")
    p.add_argument("--suffix", default="",
                   help="Suffix appended to output file names, e.g. '_legacy'.")
    p.add_argument("--header-mode", choices=["all", "legacy", "none"],
                   default="all",
                   help="Comment lines to keep: 'all' (default, every leading "
                        "comment verbatim), 'legacy' (indenter / material / "
                        "wallclock only), or 'none'.")
    p.add_argument("--energy-scope", choices=["auto", "substrate", "whole"],
                   default="auto",
                   help="Source for the IE/KE columns (default: auto = "
                        "substrate-scoped with whole-model fallback).")
    p.add_argument("--no-blank-lines", dest="blank_lines", action="store_false",
                   help="Do not insert a blank line between data rows "
                        "(default reproduces sim_test.csv, which has them).")
    p.add_argument("--strict", action="store_true",
                   help="Fail if a target column has no source column.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files.")
    p.add_argument("--encoding", default="utf-8")
    args = p.parse_args(argv)

    in_path = os.path.abspath(args.input)
    if not os.path.exists(in_path):
        p.error("Path not found: %s" % in_path)
    if not os.path.isdir(in_path):
        p.error("Input must be a directory containing .csv files, not a single "
                "file: %s" % in_path)

    if args.output:
        out_root = os.path.abspath(args.output)
    else:
        out_root = in_path.rstrip(os.sep) + "_converted"

    if os.path.normpath(out_root) == os.path.normpath(in_path):
        p.error("Output directory must differ from the input directory.")

    items = collect_inputs(in_path, args.recursive, args.pattern)
    if not items:
        print("No .csv file found in %s" % in_path)
        return 1

    if not os.path.isdir(out_root):
        os.makedirs(out_root)

    n_ok = n_err = 0
    for src, rel in items:
        stem, ext = os.path.splitext(rel)
        dst = os.path.join(out_root, stem + args.suffix + ext)

        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)

        if os.path.exists(dst) and not args.overwrite:
            print("[SKIP] %s (already exists, use --overwrite)" % rel)
            continue

        try:
            n_rows, missing = convert_file(src, dst, args.energy_scope,
                                           args.strict, args.blank_lines,
                                           args.header_mode, args.encoding)
        except Exception as exc:                       # noqa: BLE001
            n_err += 1
            print("[FAIL] %s -> %s: %s" % (rel, type(exc).__name__, exc))
            continue

        n_ok += 1
        note = ""
        if missing:
            note = "  (empty columns: %s)" % ", ".join(missing)
        print("[OK]   %-55s %6d rows%s" % (rel, n_rows, note))

    print("\n%d file(s) converted, %d error(s). Output: %s"
          % (n_ok, n_err, out_root))
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())