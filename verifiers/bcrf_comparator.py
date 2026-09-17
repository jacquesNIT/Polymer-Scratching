# -*- coding: utf-8 -*-
"""bcrf_comparator.py -- overlay several .bcrf scans on common figures.

Reads every .bcrf of a directory, runs on each of them the exact analysis of bcrf_values.py and draws:

    <prefix>_track.png     residual depth h_r and lateral pile-up h_p of every file

    <prefix>_sections.png  the transverse section at the deepest point of each file

Usage:

    python bcrf_comparator.py scans/
    python bcrf_comparator.py scans/ --material pc --outdir figures/
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt



# 1. load the verifier module
def load_verifier(path=None):
    """Import bcrf_values.py"""
    if path:
        spec = importlib.util.spec_from_file_location("bcrf_values", path)
        if spec is None or spec.loader is None:
            raise SystemExit("cannot load the verifier from %r" % path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["bcrf_values"] = mod
        spec.loader.exec_module(mod)
        return mod

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    for name in ("bcrf_values", "bcrf_verifier", "lab_values2"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise SystemExit("bcrf_values.py not found next to bcrf_comparator.py; "
                     "pass it with --verifier /path/to/bcrf_values.py")



# 2. per-file analysis
def deepest_section(bv, res, section_win, section_rezero, degree):
    """Transverse profile at res['i_deep'], built by the verifier's own recipe.

    pick_sections() anchors its last section on the deepest point of a
    smoothed h_r, while res['i_deep'] is the minimum of the raw h_r over the
    detected groove (usually coincide). 
    """
    i_deep = res["i_deep"]
    if i_deep in res["sec_idx"]:
        k = list(res["sec_idx"]).index(i_deep)
        return np.asarray(res["sec_profiles"][k], dtype=float), i_deep

    Zm = res["Zm"]
    nx = Zm.shape[1]
    win_sec = max(1, int(round(section_win / res["dx"])))
    a, b = max(0, i_deep - win_sec // 2), min(nx, i_deep + win_sec // 2 + 1)
    p = np.nanmean(Zm[:, a:b], axis=1)

    if section_rezero:
        _, ref, mask, _ = bv.flatten(res["Z"], res["row_c"], res["outer"],
                                     degree=degree, up_cols=res["up_cols"])
        sel = mask[:, a:b].any(axis=1)
        if sel.sum() < 10:
            sel = ref[:, i_deep]
        p = p - np.nanmedian(p[sel])
    return p, i_deep


def analyse_one(bv, path, kw, section_win, section_rezero, degree):
    """Run the verifier on one file and collect what the two figures need."""
    res = bv.analyse(path, **kw)
    prof = res["profiles"]
    sec, i_deep = deepest_section(bv, res, section_win, section_rezero, degree)

    present = prof["present"]
    x_deep = float(res["x_um"][i_deep])
    x_start = (float(res["x_um"][int(np.min(np.nonzero(present)[0]))])
               if present.any() else np.nan)

    return {
        "path": path,
        "label": os.path.splitext(os.path.basename(path))[0],
        "res": res,
        "x_um": res["x_um"],
        "y_um": res["y_um"],
        "y_c_um": res["row_c"] * res["dy"],
        "h_r_s": prof["h_r_s"],
        "h_p_left_s": prof["h_p_left_s"],
        "h_p_right_s": prof["h_p_right_s"],
        "present": present,
        "threshold": prof["threshold"],
        "section": sec,
        "i_deep": i_deep,
        "x_deep": x_deep,
        "x_start": x_start,
        "mound": res["mound"],
    }


def x_shift(rec, align):
    """Origin subtracted from the along-track abscissa of one file."""
    if align == "deepest":
        return rec["x_deep"] if np.isfinite(rec["x_deep"]) else 0.0
    if align == "start":
        return rec["x_start"] if np.isfinite(rec["x_start"]) else 0.0
    return 0.0



# 3. figures
def plot_track(records, out, align="none", mask_absent=False,
               show_threshold=False, title=None):
    """h_r and the two lateral pile-up profiles of every file, one colour per
    file: solid = groove floor, dashed / dotted = pile-up on either side."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    cmap = plt.get_cmap("tab10")

    for n, rec in enumerate(records):
        c = cmap(n % 10)
        x = rec["x_um"] - x_shift(rec, align)
        keep = rec["present"] if mask_absent else np.ones(x.size, dtype=bool)

        def m(v):
            return np.where(keep, v, np.nan)

        ax.plot(x, m(rec["h_r_s"]), lw=1.7, color=c, label=rec["label"])
        ax.plot(x, m(rec["h_p_left_s"]), lw=1.1, ls="--", color=c, alpha=0.85)
        ax.plot(x, m(rec["h_p_right_s"]), lw=1.1, ls=":", color=c, alpha=0.85)
        if show_threshold:
            ax.axhline(rec["threshold"], color=c, lw=0.7, ls=":", alpha=0.4)
        if rec["mound"] is not None:
            mo = rec["mound"]
            ax.plot([mo["x"] - x_shift(rec, align)], [mo["height"]], "v",
                    ms=7, color=c, mec="k", mew=0.5)

    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("x  [um]   (scratch direction%s)"
                  % ("" if align == "none" else ", %s aligned" % align))
    ax.set_ylabel("Z  [um]")
    ax.set_title(title or "Residual depth and lateral pile-up along the track")
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    style = [plt.Line2D([], [], color="0.3", lw=1.7, label=r"$h_r$ groove floor"),
             plt.Line2D([], [], color="0.3", lw=1.1, ls="--",
                        label=r"$h_p$ pile-up, $y<y_c$"),
             plt.Line2D([], [], color="0.3", lw=1.1, ls=":",
                        label=r"$h_p$ pile-up, $y>y_c$")]
    leg = ax.legend(handles, labels, fontsize=8.5, ncol=2, loc="lower left")
    ax.add_artist(leg)
    ax.legend(handles=style, fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sections(records, out, centre_y=True, xlim=None, title=None):
    """The transverse section at the deepest point of each file, overlaid."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    cmap = plt.get_cmap("tab10")

    for n, rec in enumerate(records):
        c = cmap(n % 10)
        y = rec["y_um"] - (rec["y_c_um"] if centre_y else 0.0)
        ax.plot(y, rec["section"], lw=1.5, color=c,
                label="%s   (x = %.0f um)" % (rec["label"], rec["x_deep"]))

    ax.axhline(0, color="k", lw=0.6)
    if centre_y:
        ax.axvline(0.0, color="k", lw=0.7, ls="--")
    else:
        for n, rec in enumerate(records):
            ax.axvline(rec["y_c_um"], color=cmap(n % 10), lw=0.7, ls="--",
                       alpha=0.5)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel("y - y_c  [um]" if centre_y else "y  [um]")
    ax.set_ylabel("Z  [um]")
    ax.set_title(title or "Transverse section at the deepest point")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out



# 4. summary table
_SUMMARY_COLS = ["file", "x_deep_um", "h_r_um", "h_pen_um", "w0_um",
                 "h_p_left_um", "h_p_right_um", "area_groove_um2",
                 "area_pileup_um2", "area_ratio", "mound_um"]


def summary_row(bv, rec, section_win, section_rezero):
    """Section values of the deepest section, from the verifier's own
    section_values() so that the table and the curves cannot disagree."""
    res = rec["res"]
    i = rec["i_deep"]
    if i in res["sec_idx"]:
        s = res["sections"][list(res["sec_idx"]).index(i)]
    else:
        km = res.get("k_model", {})
        s = bv.section_values(
            rec["section"], rec["y_um"], res["row_c"], res["half"],
            res["outer"],
            k_slope=km.get("slope", bv.K_SLOPE),
            k_intercept=km.get("intercept", bv.K_INTERCEPT),
            k_quad=km.get("quad", bv.K_QUAD),
            k_clip=(km.get("k_min", bv.K_MIN), km.get("k_max", bv.K_MAX)),
            k_window=(km.get("d_lo", bv.K_VALID_UM[0]),
                      km.get("d_hi", bv.K_VALID_UM[1])))
    if s is None:
        s = {}
    nan = float("nan")
    return {
        "file": rec["label"],
        "x_deep_um": rec["x_deep"],
        "h_r_um": s.get("h_r", nan),
        "h_pen_um": s.get("h_pen", nan),
        "w0_um": s.get("w0", nan),
        "h_p_left_um": s.get("h_p_left", nan),
        "h_p_right_um": s.get("h_p_right", nan),
        "area_groove_um2": s.get("area_groove", nan),
        "area_pileup_um2": s.get("area_pileup", nan),
        "area_ratio": s.get("area_ratio", nan),
        "mound_um": (rec["mound"]["height"] if rec["mound"] else nan),
    }


def print_summary(rows):
    print("")
    print("deepest section of each file (same estimator as bcrf_values.py)")
    print("  %-28s %9s %8s %8s %8s %9s %10s %8s"
          % ("file", "x [um]", "h_r", "~h_pen", "w0", "h_p left",
             "h_p right", "A_p/A_g"))
    for r in rows:
        print("  %-28s %9.0f %8.2f %8.2f %8.1f %9.2f %10.2f %8.2f"
              % (r["file"][:28], r["x_deep_um"], r["h_r_um"], r["h_pen_um"],
                 r["w0_um"], r["h_p_left_um"], r["h_p_right_um"],
                 r["area_ratio"]))
    print("  (h in um, w0 in um, areas in um^2)")


def write_summary(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_SUMMARY_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.4f" % r[k] if isinstance(r[k], float) else r[k])
                        for k in _SUMMARY_COLS})
    return path



# 5. CLI
def collect(folder, pattern="*.bcrf", recursive=False):
    if os.path.isfile(folder):
        return [folder]
    pat = os.path.join(folder, "**", pattern) if recursive \
        else os.path.join(folder, pattern)
    return sorted(glob.glob(pat, recursive=recursive))


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Overlay the residual depth, the pile-up and the deepest "
                    "transverse section of every .bcrf of a folder, using "
                    "the bcrf_values.py analysis unchanged.")
    p.add_argument("folder", help="directory holding the .bcrf scans")
    p.add_argument("--pattern", default="*.bcrf",
                   help="glob applied inside the folder (default *.bcrf)")
    p.add_argument("--recursive", action="store_true",
                   help="descend into sub-directories")
    p.add_argument("--verifier", default=None,
                   help="path to bcrf_values.py if it is not importable")
    p.add_argument("--outdir", default=".", help="directory for the figures")
    p.add_argument("--prefix", default="comparison",
                   help="stem of the two output PNGs (default comparison)")
    p.add_argument("--summary", default=None,
                   help="write the deepest-section table to this CSV")
    # ---------------- comparison-specific display ----------------
    p.add_argument("--align", choices=("none", "deepest", "start"),
                   default="none",
                   help="common x origin: raw abscissa (default), the deepest "
                        "point of each track, or the first column where the "
                        "groove is detected")
    p.add_argument("--mask-absent", action="store_true",
                   help="draw the along-track curves only where the groove is "
                        "detected")
    p.add_argument("--show-threshold", action="store_true",
                   help="draw each file's detection threshold")
    p.add_argument("--no-centre-y", dest="centre_y", action="store_false",
                   help="plot the sections against the raw y instead of "
                        "y - y_c; tracks then no longer overlap")
    p.set_defaults(centre_y=True)
    p.add_argument("--sections-xlim", type=float, nargs=2, default=None,
                   metavar=("YLO", "YHI"),
                   help="transverse window of the sections figure, in um")
    p.add_argument("--title-track", default=None)
    p.add_argument("--title-sections", default=None)
    # ---------------- analysis: same names, same defaults ----------------
    p.add_argument("--smooth-x", type=float, default=25.0,
                   help="moving average along the track, in um (default 25)")
    p.add_argument("--smooth-y", type=float, default=4.0,
                   help="moving average across the track, in um (default 4)")
    p.add_argument("--sections", type=int, default=5,
                   help="number of transverse sections computed per file "
                        "(default 5; only the deepest one is drawn)")
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--threshold", type=float, default=3.0)
    p.add_argument("--outer", type=int, default=None)
    p.add_argument("--despike-k", type=float, default=5.0)
    p.add_argument("--despike-x", type=float, default=200.0)
    p.add_argument("--despike-y", type=float, default=9.0)
    p.add_argument("--despike-col-frac", type=float, default=0.50)
    p.add_argument("--median-y", type=float, default=8.0)
    p.add_argument("--hampel-len", type=float, default=120.0)
    p.add_argument("--hampel-k", type=float, default=4.0)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smooth", dest="no_smooth", action="store_false")
    g.add_argument("--no-smooth", dest="no_smooth", action="store_true",
                   help="draw the curves raw; nothing measured passes "
                        "through the smoothing either way")
    p.set_defaults(no_smooth=False)
    p.add_argument("--detect-win", type=float, default=25.0)
    p.add_argument("--section-win", type=float, default=25.0)
    p.add_argument("--measure-win", type=float, default=None)
    p.add_argument("--rough", action="store_true",
                   help="preset for wavy surfaces (cast PMMA GS): "
                        "--median-y 40 --measure-win 300 --threshold 2")
    p.add_argument("--no-section-rezero", dest="section_rezero",
                   action="store_false")
    p.set_defaults(section_rezero=True)
    p.add_argument("--upstream", type=float, default=None)
    p.add_argument("--centre-guard", type=float, default=0.25)
    p.add_argument("--track-y", type=float, default=None)
    p.add_argument("--deep-margin", type=float, default=0.05)
    p.add_argument("--deep-win", type=float, default=100.0)
    p.add_argument("--locate-win", type=float, default=4.0)
    p.add_argument("--tip-radius", type=float, default=None)
    p.add_argument("--tip-angle", type=float, default=None)
    p.add_argument("--k-slope", type=float, default=None)
    p.add_argument("--k-intercept", type=float, default=None)
    p.add_argument("--k-clip", type=float, nargs=2, default=None,
                   metavar=("KMIN", "KMAX"))
    p.add_argument("--material", default=None,
                   help="recovery-factor preset k(d) (pmma / pc)")
    p.add_argument("--k-quad", type=float, default=None)
    p.add_argument("--k-window", type=float, nargs=2, default=None,
                   metavar=("DLO", "DHI"))
    p.add_argument("--no-penetration", dest="penetration",
                   action="store_false")
    p.set_defaults(penetration=True)
    p.add_argument("--axial-band", type=float, default=0.0)
    args = p.parse_args(argv)

    bv = load_verifier(args.verifier)
    if args.material is None:
        args.material = bv.DEFAULT_MATERIAL

    # same --rough resolution as bcrf_values.main()
    if args.measure_win is None:
        args.measure_win = 300.0 if args.rough else 0.0
    if args.rough:
        if args.median_y == 8.0:
            args.median_y = 40.0
        if args.threshold == 3.0:
            args.threshold = 2.0

    paths = collect(args.folder, args.pattern, args.recursive)
    if not paths:
        raise SystemExit("no file matching %r in %s"
                         % (args.pattern, args.folder))

    kw = dict(
        smooth_x=0.0 if args.no_smooth else args.smooth_x,
        smooth_y=0.0 if args.no_smooth else args.smooth_y,
        degree=args.degree,
        n_sections=args.sections,
        depth_threshold=args.threshold,
        outer_px=args.outer,
        despike_k=args.despike_k,
        despike_x=args.despike_x,
        despike_y=args.despike_y,
        despike_col_frac=args.despike_col_frac,
        median_y=args.median_y,
        hampel_len=args.hampel_len,
        hampel_k=args.hampel_k,
        axial_band=args.axial_band,
        detect_win=args.detect_win,
        section_win=args.section_win,
        locate_win=args.locate_win,
        measure_win=args.measure_win,
        centre_guard=args.centre_guard,
        track_y=args.track_y,
        deep_margin=args.deep_margin,
        deep_win=args.deep_win,
        upstream=args.upstream,
        section_rezero=args.section_rezero,
        tip_radius=args.tip_radius,
        tip_angle=args.tip_angle,
        k_slope=args.k_slope,
        k_intercept=args.k_intercept,
        k_clip=args.k_clip,
        material=args.material,
        k_quad=args.k_quad,
        k_window=args.k_window,
        penetration=args.penetration,
    )

    records, failed = [], 0
    for path in paths:
        try:
            rec = analyse_one(bv, path, kw, args.section_win,
                              args.section_rezero, args.degree)
        except Exception as exc:
            failed += 1
            print("FAILED %s: %s: %s"
                  % (os.path.basename(path), type(exc).__name__, exc))
            continue
        records.append(rec)
        print("read %-30s track centre y = %6.1f um, deepest x = %6.0f um"
              % (rec["label"], rec["y_c_um"], rec["x_deep"]))
        if not rec["present"].any():
            print("   WARNING: no groove above the noise floor in this file; "
                  "its curves are noise, not measurements.")

    if not records:
        raise SystemExit("nothing could be analysed")

    os.makedirs(args.outdir, exist_ok=True)
    f1 = plot_track(records,
                    os.path.join(args.outdir, args.prefix + "_track.png"),
                    align=args.align, mask_absent=args.mask_absent,
                    show_threshold=args.show_threshold,
                    title=args.title_track)
    f2 = plot_sections(records,
                       os.path.join(args.outdir, args.prefix + "_sections.png"),
                       centre_y=args.centre_y, xlim=args.sections_xlim,
                       title=args.title_sections)

    rows = [summary_row(bv, r, args.section_win, args.section_rezero)
            for r in records]
    print_summary(rows)
    print("")
    print("track figure    : %s" % f1)
    print("sections figure : %s" % f2)
    if args.summary:
        print("summary csv     : %s" % write_summary(rows, args.summary))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())