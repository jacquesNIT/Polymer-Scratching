"""Batch companion to results_values.py -- one campaign folder in, one
exploratory report out.

    python3 results_values_batch.py <results_dir> [options]

    --design  designs/glassy_pc_morris.csv   join factor levels on the run id
    --out-dir batch_analysis                 destination of CSV + figures
    --z       2.0                            extraction plane (mm)
    --jobs    N                              parallel workers (default: cpu-1)
    --pattern "*_Results.csv"                glob applied recursively
    --no-plots                               table + console summary only

Every scalar is computed by results_values.extract_values(): this file adds NO
new physics, it only walks the folder, catches per-run failures, joins the
design and plots. Changing a formula in results_values.py changes it here too.

Written for CPython 3 (numpy + scipy + matplotlib). Not Abaqus-safe, and not
meant to be -- it runs on the retrieved CSVs, off-cluster.
"""

import argparse
import csv as _csv
import importlib.util
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# =====================================================================
#  Import of the single-file module -- the only source of formulas
# =====================================================================
def load_methods(explicit=None):
    """Import results_values.py from --methods, the CWD or this file's folder."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve().parent
    candidates += [here / "results_values.py", Path.cwd() / "results_values.py"]

    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("results_values", str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["results_values"] = mod
            spec.loader.exec_module(mod)
            return mod, path
    raise SystemExit(
        "results_values.py not found. Put it next to this script or pass "
        "--methods /path/to/results_values.py"
    )


RV = None          # populated in main(); worker processes reload it themselves
RV_PATH = None


# =====================================================================
#  Acceptance thresholds -- numerical quality, not physics
# =====================================================================
# Each entry: (upper bound, unit, one-line meaning). A run above ANY bound is
# tagged SUSPECT: its QoI are still written to the table (Morris needs the
# trajectory) but the run should not be trusted blindly in the ranking.
QUALITY_LIMITS = {
    "KE_IE_steady_max":     (5.0,  "%",  "kinetic / internal energy, steady window"),
    "AE_IE_final":          (2.0,  "%",  "hourglass / internal energy, end of active phase"),
    "ETOTAL_drift":         (1.0,  "%",  "energy-balance drift"),
    "ALLPW":                (2.0,  "%",  "contact penalty work"),
    # KE_final_over_IE_peak is still extracted and written to the table, but it
    # no longer gates the status: for the plastic families the final frame is
    # taken after unload + recovery, where IE has already collapsed, so the
    # ratio is large on almost every run and tags the whole campaign SUSPECT.
    # "KE_final_over_IE_peak": (2.0, "%", "residual vibration at the final frame"),
}

# QoI actually looked at in the figures, in display order.
QOI_KEYS = ["F_n", "F_t", "SCOF_mean", "SCOF_std", "h_r", "h_p", "h_fp", "wallclock"]

STATUS_ORDER = ["OK", "SUSPECT", "ERROR", "MISSING"]
STATUS_COLORS = {
    "OK": "#2e7d32", "SUSPECT": "#ef6c00", "ERROR": "#c62828", "MISSING": "#b0b0b0",
}


# =====================================================================
#  Collection
# =====================================================================
ID_PATTERNS = [
    re.compile(r"Design[_-](\d+)"),      # run_parameter_study label for sweeps
    re.compile(r"(?:^|[_-])(\d{3,})(?:[_-]|$)"),
]


def run_id(path):
    """Zero-padded run id read from the file stem, '' if none can be read."""
    stem = Path(path).name.replace("_Results.csv", "")
    for pat in ID_PATTERNS:
        m = pat.search(stem)
        if m:
            return m.group(1).zfill(5)
    return ""


def _extract_one(args):
    """Worker: one CSV -> one flat dict. Never raises, always returns a row."""
    filepath, z_value, methods_path = args
    global RV
    if RV is None:
        RV, _ = load_methods(methods_path)

    row = {"id": run_id(filepath), "file": Path(filepath).name,
           "path": str(filepath), "status": "OK", "error": ""}
    try:
        values = RV.extract_values(filepath, z_value)
        values.pop("file", None)
        row.update(values)
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = ("%s: %s" % (type(exc).__name__, exc))[:300]
        row["traceback"] = traceback.format_exc(limit=3)
    return row


def flag_quality(row):
    """Tag a successfully parsed run SUSPECT when a diagnostic exceeds its bound."""
    if row["status"] == "ERROR":
        return row

    # extract_values() does not raise on a truncated or header-only CSV: it
    # returns the two keys it always sets and nothing else. Without this guard
    # such a run would enter the table as a perfectly healthy OK with empty
    # columns.
    def _present(key):
        try:
            return np.isfinite(float(row[key]))
        except (KeyError, TypeError, ValueError):
            return False

    if not any(_present(k) for k in ("F_n", "SCOF_mean", "h_r", "ALLIE_max")):
        row["status"] = "ERROR"
        row["error"] = "parsed but empty: no force, no energy, no profile"
        return row

    breached = []
    for key, (limit, unit, _) in QUALITY_LIMITS.items():
        v = row.get(key)
        if v is None or not np.isfinite(v):
            continue
        if abs(float(v)) > limit:
            breached.append("%s=%.3g%s>%g" % (key, v, unit, limit))
    # A run whose profile could not be resolved (h_p NaN) carries no pile-up
    # information: the ValueError of calc_xy_peak_indexes is swallowed upstream
    # into a NaN, so it has to be caught here or it silently enters the ranking.
    if row.get("h_p") is not None and not np.isfinite(row.get("h_p", np.nan)):
        breached.append("h_p=NaN (groove not bracketed by 2 peaks)")
    if breached:
        row["status"] = "SUSPECT"
        row["quality_flags"] = "; ".join(breached)
    return row


def collect(results_dir, z_value, pattern, jobs, methods_path):
    files = sorted(Path(results_dir).rglob(pattern))
    if not files:
        raise SystemExit("No file matching %r under %s" % (pattern, results_dir))

    print("Found %d file(s) under %s" % (len(files), results_dir))
    payload = [(f, z_value, methods_path) for f in files]

    rows = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, row in enumerate(pool.map(_extract_one, payload), 1):
                rows.append(row)
                print("  [%3d/%3d] %-40s %s" % (i, len(files), row["file"], row["status"]))
    else:
        for i, args in enumerate(payload, 1):
            row = _extract_one(args)
            rows.append(row)
            print("  [%3d/%3d] %-40s %s" % (i, len(files), row["file"], row["status"]))

    rows = [flag_quality(r) for r in rows]
    rows.sort(key=lambda r: (r["id"], r["file"]))
    return rows


# =====================================================================
#  Design join
# =====================================================================
DESIGN_HINT = "# Sweep design"


def find_design(results_dir):
    """Look for the sweep design next to the results, in ./designs and in the CWD.

    A design file is identified by its first line, not by its name: the CSVs of
    the runs live in the same tree and would otherwise be picked up.
    """
    seen = []
    roots = [Path(results_dir), Path(results_dir) / "designs",
             Path.cwd(), Path.cwd() / "designs"]
    for root in roots:
        if not root.is_dir():
            continue
        for cand in sorted(root.glob("*.csv")):
            if cand in seen:
                continue
            seen.append(cand)
            try:
                with open(cand, "r", encoding="utf-8-sig", errors="replace") as f:
                    head = f.readline()
            except OSError:
                continue
            if head.startswith(DESIGN_HINT):
                return str(cand)
    return None


def read_design(design_csv):
    """Read the sweep design. Returns (rows_by_id, factor_names, meta_lines)."""
    meta, header, body = [], None, []
    with open(design_csv, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                meta.append(line)
                continue
            if header is None:
                header = [c.strip() for c in line.split(",")]
                continue
            body.append([c.strip() for c in line.split(",")])

    if header is None:
        raise SystemExit("No header row in %s" % design_csv)

    factors = []
    for m in meta:
        if "active_factors=" in m:
            factors = m.split("active_factors=", 1)[1].strip().split(",")
    if not factors:
        factors = [c[2:] for c in header if c.startswith("u_")]

    rows = {}
    for parts in body:
        rec = dict(zip(header, parts))
        rid = str(rec.get("id", "")).strip().zfill(5)
        if rid:
            rows[rid] = rec
    return rows, [f.strip() for f in factors], meta


def join_design(rows, design_rows, factors):
    """Attach traj/step/moved/sign and the u_*/g_* levels; report orphans."""
    seen = set()
    for r in rows:
        rec = design_rows.get(r["id"])
        if rec is None:
            r["in_design"] = 0
            continue
        seen.add(r["id"])
        r["in_design"] = 1
        for k in ("traj", "step", "moved", "sign"):
            if k in rec:
                r[k] = rec[k]
        for f in factors:
            for prefix in ("u_", "g_"):
                key = prefix + f
                if key in rec:
                    try:
                        r[key] = float(rec[key])
                    except ValueError:
                        r[key] = np.nan

    # Design rows with no CSV at all: they must appear in the campaign map,
    # otherwise a systematically crashing corner reads as an empty cell rather
    # than as a failure -- which is exactly the informative-censoring case.
    missing = []
    for rid, rec in design_rows.items():
        if rid in seen:
            continue
        row = {"id": rid, "file": "", "path": "", "status": "MISSING",
               "error": "no *_Results.csv for this design id", "in_design": 1}
        for k in ("traj", "step", "moved", "sign"):
            if k in rec:
                row[k] = rec[k]
        for f in factors:
            for prefix in ("u_", "g_"):
                if prefix + f in rec:
                    try:
                        row[prefix + f] = float(rec[prefix + f])
                    except ValueError:
                        row[prefix + f] = np.nan
        missing.append(row)

    rows = rows + missing
    rows.sort(key=lambda r: (r["id"], r["file"]))
    return rows


# =====================================================================
#  Table output
# =====================================================================
def write_table(rows, out_csv):
    lead = ["id", "status", "traj", "step", "moved", "sign", "file"]
    keys = [k for k in lead if any(k in r for r in rows)]
    keys += [k for k in dict.fromkeys(k for r in rows for k in r)
             if k not in keys and k not in ("path", "traceback")]

    units = getattr(RV, "UNITS", {})
    headers = [k + (" [%s]" % units[k] if k in units else "") for k in keys]

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])
    print("\nWrote %d row(s) to %s" % (len(rows), out_csv))


# =====================================================================
#  Small numerical helpers
# =====================================================================
def col(rows, key, statuses=("OK", "SUSPECT")):
    """Column of finite values for the selected statuses, plus the row indices."""
    vals, idx = [], []
    for i, r in enumerate(rows):
        if r.get("status") not in statuses:
            continue
        v = r.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
            idx.append(i)
    return np.array(vals), idx


def _rank(x):
    """Ranks with ties averaged. Plain argsort-of-argsort breaks ties in index
    order, which turns two CONSTANT series into a perfect correlation."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(x.size, dtype=float)
    xs = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    return ranks


def spearman(a, b):
    """Rank correlation on pairwise-complete data; NaN if fewer than 3 pairs
    or if either series is constant (no ranking, hence no correlation)."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ra, rb = _rank(a[m]), _rank(b[m])
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def aligned(rows, key_x, key_y):
    """Two aligned arrays over the runs where BOTH values are finite."""
    xs, ys = [], []
    for r in rows:
        if r.get("status") not in ("OK", "SUSPECT"):
            continue
        try:
            x, y = float(r.get(key_x)), float(r.get(key_y))
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)


# =====================================================================
#  Figures
# =====================================================================
def fig_campaign_map(rows, out_png):
    """Trajectory x step map of the run status, + failure rate per moved factor.

    This is the first thing to read on a Morris campaign: an elementary effect
    needs BOTH ends of a step, so a single red cell kills the effect of the
    factor moved at that step, and a factor whose failures cluster at one level
    is censored informatively -- its mu* is biased low, not small.
    """
    have = [r for r in rows if r.get("traj") not in (None, "")]
    if not have:
        return None

    trajs = sorted({int(float(r["traj"])) for r in have})
    steps = sorted({int(float(r["step"])) for r in have})
    grid = np.full((len(trajs), len(steps)), STATUS_ORDER.index("MISSING"), dtype=float)
    labels = np.full((len(trajs), len(steps)), "", dtype=object)

    for r in have:
        i = trajs.index(int(float(r["traj"])))
        j = steps.index(int(float(r["step"])))
        grid[i, j] = STATUS_ORDER.index(r.get("status", "MISSING"))
        labels[i, j] = str(r.get("moved", "") or "-")

    cmap = ListedColormap([STATUS_COLORS[s] for s in STATUS_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(STATUS_ORDER)), cmap.N)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(6 + 0.55 * len(steps) + 4, 1.2 + 0.42 * len(trajs)),
        gridspec_kw={"width_ratios": [2.1, 1.0]})

    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(steps)), [str(s) for s in steps])
    ax.set_yticks(range(len(trajs)), [str(t) for t in trajs])
    ax.set_xlabel("step in trajectory")
    ax.set_ylabel("trajectory")
    ax.set_title("Run status over the Morris design\n(cell label = factor moved at that step)")
    for i in range(len(trajs)):
        for j in range(len(steps)):
            ax.text(j, i, labels[i, j], ha="center", va="center",
                    fontsize=7, color="white")
    ax.set_xticks(np.arange(-0.5, len(steps)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(trajs)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLORS[s]) for s in STATUS_ORDER]
    ax.legend(handles, STATUS_ORDER, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=8)

    # Failure rate per moved factor, split by the direction of the move: this
    # is the censoring diagnostic, a factor that only fails at sign=+1 is at a
    # boundary of its admissible range.
    per = {}
    for r in have:
        f = str(r.get("moved", "") or "-")
        if f == "-":
            continue
        try:
            sign = int(float(r.get("sign", 0)))
        except (TypeError, ValueError):
            sign = 0
        bad = r.get("status") in ("ERROR", "MISSING")
        d = per.setdefault(f, {"n": 0, "bad_up": 0, "bad_dn": 0, "n_up": 0, "n_dn": 0})
        d["n"] += 1
        if sign >= 0:
            d["n_up"] += 1
            d["bad_up"] += int(bad)
        else:
            d["n_dn"] += 1
            d["bad_dn"] += int(bad)

    names = sorted(per, key=lambda f: -(per[f]["bad_up"] + per[f]["bad_dn"]) / max(per[f]["n"], 1))
    y = np.arange(len(names))
    up = [100.0 * per[f]["bad_up"] / max(per[f]["n_up"], 1) for f in names]
    dn = [100.0 * per[f]["bad_dn"] / max(per[f]["n_dn"], 1) for f in names]
    ax2.barh(y - 0.19, up, height=0.36, color="#c62828", label="move to upper level")
    ax2.barh(y + 0.19, dn, height=0.36, color="#7b1fa2", label="move to lower level")
    ax2.set_yticks(y, names)
    ax2.invert_yaxis()
    ax2.set_xlabel("failed or missing runs [%]")
    ax2.set_title("Censoring by factor and direction")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def fig_quality(rows, out_png):
    """One row per diagnostic: every run as a point, threshold as a red line."""
    keys = [k for k in QUALITY_LIMITS if col(rows, k)[0].size]
    if not keys:
        return None

    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 1.7 * len(keys)), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, key in zip(axes, keys):
        limit, unit, meaning = QUALITY_LIMITS[key]
        vals, idx = col(rows, key)
        colors = [STATUS_COLORS[rows[i]["status"]] for i in idx]
        jitter = np.random.default_rng(0).normal(0, 0.06, size=vals.size)
        ax.scatter(vals, jitter, c=colors, s=26, alpha=0.85, edgecolors="none")
        ax.axvline(limit, color="#c62828", ls="--", lw=1.4)
        ax.text(limit, 0.42, " limit %g%s" % (limit, unit), color="#c62828",
                fontsize=8, va="center")
        n_over = int(np.sum(np.abs(vals) > limit))
        ax.set_ylabel("%s\n[%s]" % (key, unit), fontsize=8)
        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)
        ax.set_title("%s -- %d/%d run(s) above the limit" % (meaning, n_over, vals.size),
                     fontsize=9, loc="left")
        ax.grid(axis="x", alpha=0.3)
        if vals.size and np.nanmax(vals) > 0 and np.nanmax(vals) / max(np.nanmedian(vals), 1e-12) > 50:
            ax.set_xscale("symlog", linthresh=max(limit * 0.1, 1e-6))

    fig.suptitle("Numerical quality -- a run above any limit is tagged SUSPECT", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def fig_distributions(rows, out_png):
    """Distribution of each QoI over the campaign, points coloured by status."""
    keys = [k for k in QOI_KEYS if col(rows, k)[0].size >= 2]
    if not keys:
        return None

    ncols = min(4, len(keys))
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, keys):
        vals, idx = col(rows, key)
        colors = [STATUS_COLORS[rows[i]["status"]] for i in idx]
        ax.boxplot(vals, vert=True, widths=0.45, showfliers=False,
                   medianprops={"color": "#1565c0"})
        x = 1 + np.random.default_rng(1).normal(0, 0.055, size=vals.size)
        ax.scatter(x, vals, c=colors, s=18, alpha=0.8, edgecolors="none", zorder=3)
        unit = getattr(RV, "UNITS", {}).get(key, "")
        ax.set_title("%s [%s]" % (key, unit), fontsize=9)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
        # A QoI whose spread is a rounding error carries no elementary effect.
        med = np.median(vals)
        if med:
            label = "n=%d  range/median=%.2g" % (vals.size, np.ptp(vals) / abs(med))
        else:
            label = "n=%d  median=0, range=%.2g" % (vals.size, np.ptp(vals))
        ax.set_xlabel(label, fontsize=7)

    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.suptitle("QoI distributions over the campaign", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def fig_qoi_correlation(rows, out_png):
    """Spearman matrix between QoI -- redundancy check before the ranking.

    Two QoI at |rho| ~ 1 are the same observable: keeping both doubles the
    weight of one physical response in the consolidated ranking without adding
    any information.
    """
    keys = [k for k in QOI_KEYS if k != "wallclock" and col(rows, k)[0].size >= 3]
    if len(keys) < 2:
        return None

    n = len(keys)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a, b = aligned(rows, keys[i], keys[j])
            mat[i, j] = spearman(a, b)

    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * n, 1.0 + 0.75 * n))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), keys, fontsize=8)
    for i in range(n):
        for j in range(n):
            if np.isfinite(mat[i, j]):
                strong = abs(mat[i, j]) > 0.6
                ax.text(j, i, "%.2f" % mat[i, j], ha="center", va="center",
                        fontsize=7.5, color="white" if strong else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.set_title("QoI redundancy (|rho| > 0.95 = duplicate observable)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def fig_qoi_vs_factors(rows, factors, out_png):
    """QoI (rows) against normalised factor levels (columns).

    A Morris design is not a scatter design: each column mixes trajectories, so
    a trend here is a MARGINAL trend, not an elementary effect. It is still the
    fastest way to see monotonicity, saturation, and the corner where a QoI
    blows up before running morris_analysis.py.
    """
    qois = [k for k in QOI_KEYS if k != "wallclock" and col(rows, k)[0].size >= 3]
    facs = [f for f in factors if col(rows, "u_" + f)[0].size >= 3]
    if not qois or not facs:
        return None

    fig, axes = plt.subplots(len(qois), len(facs), sharex=True,
                             figsize=(1.55 * len(facs) + 1.6, 1.5 * len(qois) + 1.0))
    axes = np.atleast_2d(axes)

    for i, q in enumerate(qois):
        for j, f in enumerate(facs):
            ax = axes[i, j]
            x, y = aligned(rows, "u_" + f, q)
            ax.scatter(x, y, s=13, alpha=0.65, color="#1565c0", edgecolors="none")
            # Median per design level: the Morris levels are discrete (p=4), so
            # the level medians read far better than a regression line.
            if x.size:
                for lvl in np.unique(np.round(x, 6)):
                    m = np.abs(x - lvl) < 1e-6
                    ax.plot([lvl], [np.median(y[m])], marker="_", ms=14,
                            color="#c62828", mew=2)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(f, fontsize=8)
            if j == 0:
                unit = getattr(RV, "UNITS", {}).get(q, "")
                ax.set_ylabel("%s\n[%s]" % (q, unit), fontsize=7)
            else:
                ax.set_yticklabels([])
            if i == len(qois) - 1:
                ax.set_xlabel("u", fontsize=7)

    fig.suptitle("Marginal response to each factor (normalised level u; red = level median)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# =====================================================================
#  Console summary
# =====================================================================
def summarise(rows, factors):
    counts = {s: sum(1 for r in rows if r.get("status") == s) for s in STATUS_ORDER}
    total = len(rows)
    print("\n" + "=" * 72)
    print("CAMPAIGN SUMMARY  (%d run(s))" % total)
    print("=" * 72)
    for s in STATUS_ORDER:
        if counts[s]:
            print("  %-8s %3d  (%4.1f %%)" % (s, counts[s], 100.0 * counts[s] / total))

    trajs = {}
    for r in rows:
        if r.get("traj") in (None, ""):
            continue
        trajs.setdefault(int(float(r["traj"])), []).append(r.get("status"))
    if trajs:
        complete = [t for t, st in trajs.items()
                    if all(s in ("OK", "SUSPECT") for s in st)]
        print("\n  Trajectories usable end to end: %d / %d  %s"
              % (len(complete), len(trajs), sorted(complete)))
        broken = sorted(set(trajs) - set(complete))
        if broken:
            print("  Broken trajectories (at least one step lost): %s" % broken)

    bad = [r for r in rows if r.get("status") in ("ERROR", "MISSING")]
    if bad:
        print("\n  Lost runs:")
        for r in bad[:25]:
            print("    %s  traj=%-3s step=%-3s moved=%-8s sign=%-3s  %s"
                  % (r["id"], r.get("traj", "-"), r.get("step", "-"),
                     r.get("moved", "-"), r.get("sign", "-"), r.get("error", "")[:80]))
        if len(bad) > 25:
            print("    ... and %d more (see the table)" % (len(bad) - 25))

    print("\n  QoI over the usable runs:")
    print("    %-14s %10s %10s %10s %10s %6s" % ("QoI", "min", "median", "max", "cv", "n"))
    for k in QOI_KEYS:
        v, _ = col(rows, k)
        if v.size < 2:
            continue
        med = np.median(v)
        cv = (np.std(v) / abs(med)) if med else np.nan
        print("    %-14s %10.4g %10.4g %10.4g %10.3f %6d"
              % (k, v.min(), med, v.max(), cv, v.size))

    # Redundancy is worth naming in the console too: it decides which QoI even
    # enter the screening.
    pairs = []
    keys = [k for k in QOI_KEYS if k != "wallclock" and col(rows, k)[0].size >= 3]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = aligned(rows, keys[i], keys[j])
            rho = spearman(a, b)
            if np.isfinite(rho) and abs(rho) > 0.95:
                pairs.append((keys[i], keys[j], rho))
    if pairs:
        print("\n  Near-duplicate QoI (|rho| > 0.95) -- keep one of each pair:")
        for a, b, rho in pairs:
            print("    %-14s %-14s rho = %+.3f" % (a, b, rho))
    print()


# =====================================================================
#  Entry point
# =====================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", help="folder holding the *_Results.csv (walked recursively)")
    p.add_argument("--design", default=None, help="sweep design CSV, joined on the run id")
    p.add_argument("--out-dir", default="batch_analysis")
    p.add_argument("--z", type=float, default=None, help="extraction plane in mm")
    p.add_argument("--jobs", type=int, default=0, help="parallel workers, 0 = cpu_count-1")
    p.add_argument("--pattern", default="*_Results.csv")
    p.add_argument("--methods", default=None, help="path to results_values.py")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    global RV, RV_PATH
    RV, RV_PATH = load_methods(args.methods)
    print("Formulas taken from %s" % RV_PATH)

    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    rows = collect(args.results_dir, args.z, args.pattern, jobs, args.methods)

    design_path = args.design or find_design(args.results_dir)
    factors = []
    if design_path:
        design_rows, factors, _ = read_design(design_path)
        rows = join_design(rows, design_rows, factors)
        print("Design: %s -- %d row(s), factors %s"
              % (design_path, len(design_rows), ",".join(factors)))
    else:
        # Figures 01 and 05 read traj / step / moved / u_*, which exist only in
        # the design file: without it they have nothing to plot and are skipped.
        print("\n!! No design file: figures 01 (campaign map) and 05 (QoI vs\n"
              "   factors) cannot be produced. Pass --design <sweep design csv>\n"
              "   (e.g. --design designs/glassy_pc_morris.csv).\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(rows, str(out_dir / "batch_values.csv"))
    summarise(rows, factors)

    if not args.no_plots:
        planned = [
            ("01_campaign_map.png", lambda p: fig_campaign_map(rows, p),
             "needs the design (traj / step / moved columns)"),
            ("02_numerical_quality.png", lambda p: fig_quality(rows, p),
             "no energy diagnostic could be read"),
            ("03_qoi_distributions.png", lambda p: fig_distributions(rows, p),
             "fewer than 2 usable runs per QoI"),
            ("04_qoi_correlation.png", lambda p: fig_qoi_correlation(rows, p),
             "fewer than 2 QoI with 3+ usable runs"),
            ("05_qoi_vs_factors.png", lambda p: fig_qoi_vs_factors(rows, factors, p),
             "needs the design (u_* factor levels)"),
        ]
        print("  figures:")
        for name, build, why in planned:
            made = build(str(out_dir / name))
            print("    %-28s %s" % (name, made if made else "SKIPPED -- " + why))


if __name__ == "__main__":
    main()