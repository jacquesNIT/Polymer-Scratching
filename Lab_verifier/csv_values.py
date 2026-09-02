# -*- coding: utf-8 -*-
"""lab_values.py -- in-situ profiles from a TriboSoft / Rtec MFT scratch export.

Reads a scratch CSV produced by the MFT software and plots, along the scratch
track:

    * normal force  F_n  and tangential force  F_t
    * scratch coefficient of friction  SCOF
    * in-situ penetration depth, with automatic detection of capacitive
      sensor saturation
    * SCOF and depth versus normal load

File layout expected (TriboSoft export)::

    line 0   recipe metadata keys      (Recipe, LogFrequency, Radius, ...)
    line 1   recipe metadata values
    line 2   channel header            (Step, Timestamp, DAQ.Fz (N), ...)
    line 3+  data rows, then two trailing rows (TotalTime, hh:mm:ss)

Usage::

    python lab_values.py Test2-PP_10N.csv
    python lab_values.py Test2-PP_10N.csv --out fig.png --export profiles.csv
    python lab_values.py *.csv --outdir figures/

Dependencies: numpy and matplotlib only. pandas and scipy are deliberately
not used, because their compiled extensions are blocked by the application
control policy on the lab workstation.

Units: forces in N, depths in um, distances in mm.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# channel resolution
# --------------------------------------------------------------------------

# Each entry maps a logical channel to the substrings that identify it.
# Matching is case-insensitive and ignores units, so "DAQ.Fz (N)" and
# "Fz" both resolve to "fn".
CHANNEL_KEYS = {
    "step":  ["step"],
    "time":  ["timestamp", "time (s)"],
    "fn":    ["fz"],
    "ft":    ["fx"],
    "cof":   ["cof"],
    "cap":   ["cap"],
    "x":     ["x position"],
    "y":     ["y position"],
    "zpos":  ["z position"],
    "zdep":  ["z depth"],
}

REQUIRED = ["fn", "ft", "x"]


def resolve_columns(columns):
    """Map logical channel names to the column index of the file."""
    found = {}
    for key, patterns in CHANNEL_KEYS.items():
        for i, col in enumerate(columns):
            low = col.lower()
            if any(p in low for p in patterns):
                found[key] = i
                break
    missing = [k for k in REQUIRED if k not in found]
    if missing:
        raise KeyError("channels not found in file: %s\navailable columns: %s"
                       % (", ".join(missing), ", ".join(columns)))
    return found


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def _to_float(text):
    """Parse a cell, tolerating blanks, commas as decimal marks and junk."""
    text = text.strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return np.nan


def read_scratch(path, step=None):
    """Return (dict of channel arrays, channel map, metadata).

    The trailing summary rows are dropped, all data cells are coerced to
    float, and only the longest recipe step is kept unless `step` is given.
    """
    with open(path, "r", newline="", errors="replace") as fh:
        rows = list(csv.reader(fh))

    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 4:
        raise ValueError("%s: file too short to be a scratch export" % path)

    meta = {}
    for k, v in zip(rows[0], rows[1]):
        k, v = k.strip(), v.strip()
        if k:
            meta[k] = v

    header = [c.strip() for c in rows[2]]
    ch = resolve_columns(header)
    ncol = len(header)

    # keep only rows whose Step field is an integer: this drops the trailing
    # 'TotalTime' and elapsed-time rows
    i_step = ch["step"]
    data = []
    for r in rows[3:]:
        if len(r) < ncol:
            r = r + [""] * (ncol - len(r))
        try:
            s = int(float(r[i_step].strip()))
        except (ValueError, IndexError):
            continue
        data.append((s, r))

    if not data:
        raise ValueError("%s: no numeric data rows found" % path)

    steps = np.array([s for s, _ in data])
    if step is None:
        vals, counts = np.unique(steps, return_counts=True)
        step = int(vals[int(np.argmax(counts))])

    sel = [r for s, r in data if s == step]
    if not sel:
        raise ValueError("step %s contains no data" % step)

    cols = {}
    for key, idx in ch.items():
        if key == "step":
            continue
        cols[key] = np.array([_to_float(r[idx]) for r in sel], dtype=float)

    # drop rows where a required channel is missing
    ok = np.ones(len(sel), dtype=bool)
    for key in REQUIRED:
        ok &= np.isfinite(cols[key])
    for key in cols:
        cols[key] = cols[key][ok]

    if cols["fn"].size == 0:
        raise ValueError("%s: no valid samples after cleaning" % path)

    return cols, ch, meta


# --------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------

def moving_average(v, win):
    """Centred moving average with edge replication (scipy-free)."""
    v = np.asarray(v, dtype=float)
    win = int(max(1, win))
    if win <= 1 or v.size == 0:
        return v.copy()
    if win > v.size:
        win = v.size
    half = win // 2

    # NaN-tolerant: average the finite samples only
    finite = np.isfinite(v)
    filled = np.where(finite, v, 0.0)

    pad_l, pad_r = half, win - 1 - half
    fp = np.concatenate([np.full(pad_l, filled[0]), filled,
                         np.full(pad_r, filled[-1])])
    wp = np.concatenate([np.full(pad_l, float(finite[0])),
                         finite.astype(float),
                         np.full(pad_r, float(finite[-1]))])

    cs = np.concatenate([[0.0], np.cumsum(fp)])
    cw = np.concatenate([[0.0], np.cumsum(wp)])
    num = cs[win:] - cs[:-win]
    den = cw[win:] - cw[:-win]

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def nanptp(v):
    return float(np.nanmax(v) - np.nanmin(v))


# --------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------

def build_profiles(cols, smooth_um=12.0, plateau_tol=1.0,
                   plateau_min_frac=0.02):
    """Compute the along-track profiles and the scalar summary.

    smooth_um        moving-average window, expressed in um of travel
    plateau_tol      depth window (um) used to detect a saturated CAP sensor
    plateau_min_frac a plateau shorter than this fraction of the track is
                     considered a genuine depth plateau, not a sensor limit
    """
    x = cols["x"]
    s = np.abs(x - x[0])                          # travel along the track, mm

    fn = cols["fn"]
    ft = np.abs(cols["ft"])

    cof = cols.get("cof")
    if cof is None or not np.isfinite(cof).any():
        with np.errstate(divide="ignore", invalid="ignore"):
            cof = np.where(fn > 1e-9, ft / fn, np.nan)

    # penetration: the capacitive gauge is the primary source, the stage Z
    # travel the fallback. Sign is resolved from the data, not assumed.
    pen, pen_src = None, None
    cap = cols.get("cap")
    zdep = cols.get("zdep")
    if cap is not None and np.isfinite(cap).any():
        pen = cap - cap[0] if cap[-1] > cap[0] else cap[0] - cap
        pen_src = "CAP"
    elif zdep is not None and np.isfinite(zdep).any():
        pen = np.abs(zdep - zdep[0]) * 1e3        # mm -> um
        pen_src = "stage Z"

    # smoothing window in samples
    step = np.abs(np.diff(s))
    ds = float(np.median(step[step > 0])) * 1e3 if np.any(step > 0) else 0.0
    win = max(3, int(round(smooth_um / ds))) if ds > 0 else 3

    prof = {
        "s": s, "fn": fn, "ft": ft, "cof": cof,
        "pen": pen, "pen_source": pen_src, "window": win,
        "smooth": lambda v: moving_average(v, win),
    }

    # SCOF averaged over 10-90 % of the normal-force ramp: this excludes the
    # initial elastic transient and the unloading tail.
    fmax = np.nanmax(fn)
    band = (fn >= 0.10 * fmax) & (fn <= 0.90 * fmax)
    prof["scof"] = float(np.nanmean(cof[band])) if band.any() else np.nan
    prof["scof_band"] = band

    # saturation of the depth channel
    prof["i_sat"] = None
    if pen is not None and np.isfinite(pen).any():
        flat = pen >= np.nanmax(pen) - plateau_tol
        i0 = int(np.argmax(flat))
        if flat[i0:].mean() > 0.9 and (pen.size - i0) / pen.size > plateau_min_frac:
            prof["i_sat"] = i0

    return prof


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def plot_profiles(prof, title="", out="scratch_profiles.png"):
    s, fn, ft, cof = prof["s"], prof["fn"], prof["ft"], prof["cof"]
    pen, sm = prof["pen"], prof["smooth"]
    i_sat = prof["i_sat"]

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.5))
    fig.subplots_adjust(hspace=0.32, wspace=0.26)

    def mark_sat(a):
        if i_sat is not None:
            a.axvline(s[i_sat], color="k", ls=":", lw=1)

    # --- forces ---------------------------------------------------------
    a = ax[0, 0]
    a.plot(s, fn, lw=0.6, color="0.82")
    a.plot(s, sm(fn), lw=1.6, color="C0", label=r"$F_n$")
    a.plot(s, ft, lw=0.6, color="0.86")
    a.plot(s, sm(ft), lw=1.6, color="C3", label=r"$F_t$")
    mark_sat(a)
    a.set_xlabel("distance along scratch  [mm]")
    a.set_ylabel("force  [N]")
    a.set_title("Normal and tangential force")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    # --- SCOF -----------------------------------------------------------
    a = ax[0, 1]
    a.plot(s, cof, lw=0.6, color="0.82")
    a.plot(s, sm(cof), lw=1.6, color="C2", label="SCOF")
    with np.errstate(divide="ignore", invalid="ignore"):
        a.plot(s, sm(np.where(fn > 1e-9, ft / fn, np.nan)), lw=1.1, ls="--",
               color="C1", label=r"$F_t/F_n$ recomputed")
    a.axhline(prof["scof"], color="k", lw=0.9, ls="-.",
              label="mean 10-90 %% = %.3f" % prof["scof"])
    mark_sat(a)
    a.set_xlabel("distance along scratch  [mm]")
    a.set_ylabel("COF  [-]")
    a.set_title("Scratch coefficient of friction")
    a.legend(fontsize=8.5)
    a.grid(alpha=0.3)

    # --- depth ----------------------------------------------------------
    a = ax[1, 0]
    if pen is None:
        a.text(0.5, 0.5, "no depth channel in this file", ha="center",
               va="center", transform=a.transAxes, fontsize=11, color="0.4")
        a.set_axis_off()
    else:
        a.plot(s, pen, lw=0.6, color="0.82")
        a.plot(s, sm(pen), lw=1.6, color="C4",
               label="penetration (%s)" % prof["pen_source"])
        if i_sat is not None:
            a.axvspan(s[i_sat], s[-1], color="crimson", alpha=0.10)
            a.axvline(s[i_sat], color="k", ls=":", lw=1)
            a.text(s[i_sat] + 0.03 * nanptp(s), 0.12 * np.nanmax(pen),
                   "depth channel saturated\n($F_n$ > %.1f N)" % fn[i_sat],
                   fontsize=8.5, color="crimson")
        a.set_xlabel("distance along scratch  [mm]")
        a.set_ylabel("penetration  [um]")
        a.set_title("In-situ penetration depth")
        a.legend(fontsize=9, loc="upper left")
        a.grid(alpha=0.3)

    # --- versus load ----------------------------------------------------
    a = ax[1, 1]
    a.plot(fn, sm(cof), lw=1.6, color="C2")
    a.set_xlabel(r"$F_n$  [N]")
    a.set_ylabel("COF  [-]", color="C2")
    a.tick_params(axis="y", labelcolor="C2")
    a.set_title("SCOF and penetration vs normal load")
    a.grid(alpha=0.3)
    if i_sat is not None:
        a.axvline(fn[i_sat], color="k", ls=":", lw=1)
    if pen is not None:
        a2 = a.twinx()
        a2.plot(fn, sm(pen), lw=1.6, color="C4")
        a2.set_ylabel("penetration  [um]", color="C4")
        a2.tick_params(axis="y", labelcolor="C4")

    if title:
        fig.suptitle(title, fontsize=12.5)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def describe(path, prof, meta):
    """One-line-per-quantity summary printed to stdout."""
    s, fn, ft, cof, pen = (prof["s"], prof["fn"], prof["ft"],
                           prof["cof"], prof["pen"])
    out = ["file          : %s" % os.path.basename(path)]
    if meta.get("Recipe"):
        out.append("recipe        : %s" % meta["Recipe"])
    if meta.get("OtherData"):
        out.append("loading       : %s" % meta["OtherData"])
    out.append("samples       : %d" % s.size)
    out.append("travel        : %.3f mm" % s[-1])
    out.append("Fn            : %.3f -> %.2f N" % (np.nanmin(fn), np.nanmax(fn)))
    out.append("Ft            : %.3f -> %.2f N" % (np.nanmin(ft), np.nanmax(ft)))
    out.append("COF           : %.3f -> %.3f" % (np.nanmin(cof), np.nanmax(cof)))
    out.append("SCOF 10-90 %%  : %.3f" % prof["scof"])
    if pen is None:
        out.append("penetration   : no depth channel")
    elif prof["i_sat"] is None:
        out.append("penetration   : %.1f um max (%s)"
                   % (np.nanmax(pen), prof["pen_source"]))
    else:
        i = prof["i_sat"]
        out.append("penetration   : %.1f um max (%s), SATURATED from "
                   "s = %.3f mm / Fn = %.2f N"
                   % (np.nanmax(pen), prof["pen_source"], s[i], fn[i]))
    return "\n".join(out)


def export_profiles(prof, path):
    """Write the along-track profiles as a tidy CSV (stdlib csv writer)."""
    names = ["s_mm", "Fn_N", "Ft_N", "COF"]
    cols = [prof["s"], prof["fn"], prof["ft"], prof["cof"]]
    if prof["pen"] is not None:
        names.append("penetration_um")
        cols.append(prof["pen"])
        if prof["i_sat"] is not None:
            flag = np.zeros(prof["s"].size, dtype=int)
            flag[prof["i_sat"]:] = 1
            names.append("depth_saturated")
            cols.append(flag)

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for row in zip(*cols):
            w.writerow(["%.6g" % v for v in row])
    return path


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def process(path, outdir=None, out=None, export=None, step=None,
            smooth_um=12.0, title=None):
    cols, ch, meta = read_scratch(path, step=step)
    prof = build_profiles(cols, smooth_um=smooth_um)

    stem = os.path.splitext(os.path.basename(path))[0]
    if out is None:
        out = os.path.join(outdir or os.path.dirname(path) or ".",
                           stem + "_profiles.png")
    if title is None:
        title = stem
        if meta.get("OtherData"):
            title += "   |   %s" % meta["OtherData"]

    plot_profiles(prof, title=title, out=out)
    print(describe(path, prof, meta))
    print("figure        : %s" % out)

    if export is not None:
        target = export
        if os.path.isdir(target):
            target = os.path.join(target, stem + "_profiles.csv")
        export_profiles(prof, target)
        print("profiles csv  : %s" % target)

    return prof


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Plot forces, SCOF and depth from an MFT scratch CSV.")
    p.add_argument("csv", nargs="+", help="scratch CSV file(s)")
    p.add_argument("--out", default=None,
                   help="output PNG (single input only)")
    p.add_argument("--outdir", default=None,
                   help="directory for the figures")
    p.add_argument("--export", default=None,
                   help="also write the profiles as CSV (file or directory)")
    p.add_argument("--step", type=int, default=None,
                   help="recipe step to analyse (default: the longest one)")
    p.add_argument("--smooth", type=float, default=12.0,
                   help="moving-average window in um of travel (default 12)")
    args = p.parse_args(argv)

    # cmd.exe does not expand wildcards, so do it here
    paths = []
    for pattern in args.csv:
        hits = sorted(glob.glob(pattern))
        paths.extend(hits if hits else [pattern])

    if args.out is not None and len(paths) > 1:
        p.error("--out cannot be used with several input files; use --outdir")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    failed = 0
    for path in paths:
        if len(paths) > 1:
            print("=" * 60)
        try:
            process(path, outdir=args.outdir, out=args.out, export=args.export,
                    step=args.step, smooth_um=args.smooth)
        except Exception as exc:                       # keep the batch going
            failed += 1
            print("FAILED %s: %s: %s"
                  % (os.path.basename(path), type(exc).__name__, exc))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())