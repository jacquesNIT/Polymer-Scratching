# Build a reduced design containing only the cases that are missing from a
# sweep, so they can be re-run without touching the ones that succeeded.
#
#   python3 retry_design.py --design designs/glassy_pc_morris.csv \
#           --results runs/Finalisazation/Morris1 \
#           --as-family glassy_dp
#
# Missing cases are found by comparing the design ids against the
# *_Results.csv actually present. FAILED_CASES.txt files can be added with
# --failed; ids from both sources are merged.
#
# The output keeps the ORIGINAL ids, which is what lets sweep_collector.py
# join the re-run results back onto the full design. Analysis is always done
# against the original design, never against the retry subset.

import argparse
import csv
import glob
import os
import re
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.path.abspath(".")

_DP_HOSTS = ("semicrystalline_dp", "glassy_dp", "glassy_pc", "glassy_pmma")


def read_design(path):
    header, data = [], []
    with open(path, "r") as f:
        for line in f:
            (header if line.lstrip().startswith("#") else data).append(line)
    return header, list(csv.DictReader(data)), data[0]


def ids_present(results_dir):
    found = set()
    for root, _dirs, names in os.walk(results_dir):
        for n in names:
            m = re.search(r"Design_(\w+?)_Results\.csv$", n)
            if m:
                found.add(m.group(1))
    return found


def ids_failed(patterns):
    out = set()
    for pat in patterns:
        for path in glob.glob(pat):
            with open(path, "r") as f:
                for line in f:
                    m = re.search(r"Design_(\w+)", line)
                    if m:
                        out.add(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser(description="Reduced design for re-running failed cases.")
    ap.add_argument("--design", required=True, help="original design CSV")
    ap.add_argument("--results", default=None,
                    help="results directory: any design id with no *_Results.csv is retried")
    ap.add_argument("--failed", nargs="*", default=None,
                    help="FAILED_CASES.txt file(s) or glob(s)")
    ap.add_argument("--ids", default=None, help="explicit comma-separated ids")
    ap.add_argument("--as-family", default=None,
                    help="host family the retry runs under (default: another DP host, "
                         "so the original design file is left untouched)")
    ap.add_argument("--out", default=None, help="output CSV (default: designs/<as-family>_morris.csv)")
    ap.add_argument("--jobs", type=int, default=5, help="DESIGN_JOBS you intend to use")
    args = ap.parse_args()

    header, rows, col_line = read_design(args.design)
    all_ids = [r["id"] for r in rows]
    src_family = rows[0].get("family", "")
    campaign = rows[0].get("campaign", "")

    missing = set()
    if args.results:
        if not os.path.isdir(args.results):
            raise SystemExit("Not a directory: %s" % args.results)
        missing |= (set(all_ids) - ids_present(args.results))
    if args.failed:
        missing |= ids_failed(args.failed)
    if args.ids:
        missing |= set(s.strip() for s in args.ids.split(",") if s.strip())
    if not (args.results or args.failed or args.ids):
        raise SystemExit("Give at least one of --results, --failed, --ids.")

    unknown = sorted(missing - set(all_ids))
    if unknown:
        raise SystemExit("Id(s) not in the design: %s" % ", ".join(unknown))
    retry = [r for r in rows if r["id"] in missing]
    if not retry:
        print("Nothing to retry: every design point has a result.")
        return 0

    target = args.as_family
    if target is None:
        for f in _DP_HOSTS:
            if f != src_family:
                target = f
                break
    if target == src_family and not args.out:
        raise SystemExit(
            "Retry family equals the source family (%s): the original design would be "
            "overwritten. Pass --as-family with a different DP host, or --out." % src_family)

    out = args.out or os.path.join(_HERE, "designs", "%s_morris.csv" % target)
    out_dir = os.path.dirname(out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    fields = list(rows[0].keys())
    with open(out, "w") as f:
        for line in header:
            f.write(line)
        f.write("# RETRY SUBSET of %s -- %d of %d cases, original ids preserved.\n"
                % (os.path.basename(args.design), len(retry), len(rows)))
        f.write("# Run it, then collect the ORIGINAL and the retry directories together;\n")
        f.write("# analyse against the ORIGINAL design, never against this subset.\n")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in retry:
            r = dict(r)
            if target:
                r["family"] = target
            w.writerow(r)

    per = (len(retry) + args.jobs - 1) // args.jobs
    print("Retrying %d of %d case(s): %s"
          % (len(retry), len(rows), ", ".join(sorted(missing)[:30])
             + (" ..." if len(missing) > 30 else "")))
    print("Wrote %s  (campaign %s, host %s)" % (out, campaign, target))
    print("")
    print("In launch_cluster_jobs.py:")
    print('    ("design", "%s", {"tag": "retry", "ALE": True, "freq": 250,' % target)
    print('                      "distortion": True, "length": 0.1}),')
    print("    DESIGN_JOBS = %d      -> %d job(s) of ~%d case(s)"
          % (args.jobs, args.jobs, per))
    if per < 2:
        print("    (fewer cases than jobs: lower DESIGN_JOBS to avoid empty chunks)")
    print("")
    print("Then collect BOTH directories in one pass:")
    print("    python3 sweep_collector.py <parent-dir-containing-both> \\")
    print("            --design %s --out sweep_dp.csv" % os.path.basename(args.design))
    return 0


if __name__ == "__main__":
    sys.exit(main())