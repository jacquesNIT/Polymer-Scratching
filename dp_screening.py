# Consolidated Morris screening report for the unified Drucker-Prager campaign.
#
#   python3 dp_screening.py /path/to/results \
#           --design designs/glassy_pc_morris.csv \
#           --out-dir screening_dp
#
# [PATCH:no-noise-floor] retention relies on a RELATIVE threshold
# (mu*_lo / mu*_max >= --retain-frac), not on an absolute noise floor.
#
# Runs the whole chain -- collect, elementary effects on every QoI, consolidated
# ranking -- and writes SCREENING_REPORT.md plus the figures next to it.
# Collection is skipped if --table points at an already collected tidy CSV.

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.path.abspath(".")
sys.path.insert(0, _HERE)

import morris_analysis as MA


# [PATCH:no-noise-floor] H_MPa, Ft_half_N and pile_up_ratio removed:
#   H_MPa           : contact_radius_mm is constant over the whole campaign, so
#                     H = Fn / fixed area and r(H, Fn) = 1.000000. Keeping it
#                     counted the same signal twice and inflated the
#                     multiplicity correction with a duplicate.
#   Ft_half_N       : Ft = Fn * scof, redundant with the two kept QoI.
#   pile_up_ratio   : sigma/mu* between 1.3 and 2.3 across all eight factors,
#                     the estimator is not usable at this number of EEs.
DEFAULT_QOI = [
    ("Fn_half_N", "Normal force (half-model)", "N"),
    ("scof", "Apparent friction coefficient", "-"),
    ("residual_depth_mm", "Residual depth", "mm"),
    ("pile_up_mm", "Lateral pile-up", "mm"),
]

SIGMA_RATIO_INTERACTION = 1.0    # sigma/mu* above this => interaction-dominated
RETAIN_FRAC = 0.10               # minimal mu*_lo / mu*_max to retain a factor


# ----------------------------------------------------------------------

def _collect(results_dir, design, out_csv):
    cmd = [sys.executable, os.path.join(_HERE, "sweep_collector.py"), results_dir,
           "--design", design, "--out", out_csv]
    print("  $ " + " ".join(cmd[1:]))
    r = subprocess.call(cmd)
    if r != 0 or not os.path.exists(out_csv):
        raise SystemExit("sweep_collector.py failed (exit %s)" % r)
    return out_csv


def _coverage(design_rows, table):
    ids = set(design_rows)
    # [PATCH:screening-fixes] original (duplicated definition, diverged from
    # elementary_effects which ignored parse_error):
    # have = set(k for k, v in table.items()
    #            if k in ids and not v.get("parse_error") and v.get("status") != "FAIL")
    have = set(k for k, v in table.items() if k in ids and MA._usable(v))
    per_traj = {}
    for rid, d in design_rows.items():
        per_traj.setdefault(int(d["traj"]), []).append(rid)
    complete = sum(1 for t, r in per_traj.items() if all(x in have for x in r))
    return {"n_design": len(ids), "n_usable": len(have),
            "n_missing": len(ids - have), "missing": sorted(ids - have),
            "n_traj": len(per_traj), "n_traj_complete": complete}


# [PATCH:no-noise-floor] decision without a noise floor.
#
# The old rule tested mu*_lo against mu*_null = sqrt(2/pi) * sqrt(2)/Delta * sigma_num,
# i.e. against an ABSOLUTE noise scale. Without a floor, no verdict was ever
# rendered. The rule that replaces it is RELATIVE:
#
#     rel_lo = mu*_lo / mu*_max      (per QoI)
#     RETAIN   if rel_lo >= retain_frac
#     RETAIN?  if rel    >= retain_frac but rel_lo < retain_frac
#     freeze   otherwise
#
# It keeps the bootstrap lower bound, so it stays sensitive to censoring: a
# factor with fewer surviving runs gets a wider interval and a lower rel_lo.
# But it ranks against the dominant factor of each QoI, it does not test
# against zero -- the top factor is retained by construction, and a 'freeze'
# means 'small compared to the largest', not 'null'.
def _analyse(design_rows, table, factors, delta, qoi_keys,
             bootstrap, ci_low, retain_frac=RETAIN_FRAC):
    out = {}
    for q in qoi_keys:
        effects, n_missing = MA.elementary_effects(design_rows, table, q, delta, ("FAIL",))
        if not effects:
            continue
        summary = MA.summarise(effects, factors, n_bootstrap=bootstrap, ci_low=ci_low)
        mu_max = max([summary[f]["mu_star"] for f in factors
                      if np.isfinite(summary[f]["mu_star"])] or [np.nan])
        usable_max = mu_max if (mu_max and np.isfinite(mu_max)) else np.nan
        for f in factors:
            s = summary[f]
            s["rel"] = (s["mu_star"] / usable_max) if np.isfinite(usable_max) else np.nan
            s["rel_lo"] = (s["mu_star_lo"] / usable_max) if np.isfinite(usable_max) else np.nan
            s["sigma_ratio"] = (s["sigma"] / s["mu_star"]) if s["mu_star"] else np.nan
            if not np.isfinite(s["rel"]):
                s["verdict"] = "no data"
            elif np.isfinite(s["rel_lo"]) and s["rel_lo"] >= retain_frac:
                s["verdict"] = "RETAIN"
            elif s["rel"] >= retain_frac:
                s["verdict"] = "RETAIN?"
            else:
                s["verdict"] = "freeze"
        out[q] = {"summary": summary, "n_effects": len(effects), "n_missing": n_missing,
                  "mu_max": mu_max, "retain_frac": retain_frac}
    return out


# [PATCH:screening-fixes] begin -- points 5 and 6.
CONFOUND_THRESHOLD = 0.55        # confounding index above which a warning is raised


def _confounding(per_qoi, factors):
    """
    POINT 5 -- exploits the SIGNED `mu`, until now computed and displayed but
    absent from every verdict.

    Two factors entering additively and antagonistically in the same law are
    not separately identifiable. The observable signature is: `mu` of opposite
    signs, `mu*` of the same order, and this CONSISTENTLY across QoI. Index
    per pair, averaged over QoI, of

        min(mu*_i, mu*_j) / max(mu*_i, mu*_j)  x  max(0, -sign(mu_i) sign(mu_j))

    1 = perfectly antagonistic and of equal weight; 0 = no signature.
    """
    out = []
    for a in range(len(factors)):
        for b in range(a + 1, len(factors)):
            fi, fj = factors[a], factors[b]
            ws = []
            for _q, blk in per_qoi.items():
                si, sj = blk["summary"].get(fi), blk["summary"].get(fj)
                if not si or not sj:
                    continue
                mi, mj = si["mu_star"], sj["mu_star"]
                if not (np.isfinite(mi) and np.isfinite(mj)) or max(mi, mj) <= 0:
                    continue
                bal = min(mi, mj) / max(mi, mj)
                opp = max(0.0, -((si["mu"] / mi) * (sj["mu"] / mj)))
                ws.append(bal * opp)
            if ws:
                out.append({"pair": (fi, fj), "index": float(np.mean(ws)),
                            "n_qoi": len(ws)})
    out.sort(key=lambda r: -r["index"])
    return out


# [PATCH:no-noise-floor] _structural_floor removed (noise floor).
def _consolidate(per_qoi, factors):
    rows = []
    for f in factors:
        rels, retained_in, ranks, sig_ratio, margins = [], [], [], [], []
        for q, blk in per_qoi.items():
            s = blk["summary"][f]
            if np.isfinite(s["rel"]):
                rels.append(s["rel"])
                order = sorted(factors, key=lambda g: -(blk["summary"][g]["mu_star"]
                                                        if np.isfinite(blk["summary"][g]["mu_star"]) else -1))
                ranks.append(order.index(f) + 1)
            if np.isfinite(s["sigma_ratio"]):
                sig_ratio.append(s["sigma_ratio"])
            # [PATCH:no-noise-floor] margin -> rel_lo.
            if s["verdict"].startswith("RETAIN"):
                retained_in.append(q)
                margins.append(s.get("rel_lo", np.nan))
        rows.append({
            "factor": f,
            "rel_max": max(rels) if rels else np.nan,
            "rel_mean": float(np.mean(rels)) if rels else np.nan,
            "rank_best": min(ranks) if ranks else None,
            "rank_mean": float(np.mean(ranks)) if ranks else np.nan,
            "sigma_ratio_max": max(sig_ratio) if sig_ratio else np.nan,
            "retained_in": retained_in,
            "n_retained": len(retained_in),
            # [PATCH:no-noise-floor] margin (mu*/noise threshold) -> rel_lo_best.
            "rel_lo_best": max([m for m in margins if np.isfinite(m)] or [np.nan]),
            "verdict": ("RETAIN" if retained_in else
                        ("RETAIN?" if any(
                            blk["summary"][f]["verdict"] == "RETAIN?"
                            for blk in per_qoi.values()) else "freeze")),
        })
    rows.sort(key=lambda r: (-(r["rel_max"] if np.isfinite(r["rel_max"]) else -1)))
    return rows


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _fig_heatmap(per_qoi, factors, qoi_keys, path):
    plt = _mpl()
    if plt is None:
        return False
    keys = [q for q in qoi_keys if q in per_qoi]
    M = np.array([[per_qoi[q]["summary"][f]["rel"] for q in keys] for f in factors], dtype=float)
    fig, ax = plt.subplots(figsize=(1.35 * len(keys) + 3.2, 0.48 * len(factors) + 2.0))
    im = ax.imshow(np.nan_to_num(M), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(factors, fontsize=9)
    for i in range(len(factors)):
        for j in range(len(keys)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=7,
                        color="white" if v < 0.6 else "black")
    ax.set_title("mu* normalised per QoI (1 = dominant factor)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _fig_consolidated(cons, path):
    plt = _mpl()
    if plt is None:
        return False
    rows = [r for r in cons if np.isfinite(r["rel_max"])][::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(rows) + 1.8))
    y = np.arange(len(rows))
    colors = ["#2f6f4e" if r["verdict"] == "RETAIN"
              else ("#c9a227" if r["verdict"] == "RETAIN?" else "#9aa0a6") for r in rows]
    ax.barh(y, [r["rel_max"] for r in rows], color=colors)
    ax.barh(y, [r["rel_mean"] for r in rows], height=0.32, color="black", alpha=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([r["factor"] for r in rows])
    ax.set_xlabel("normalised mu*  (bar = max over QoI, tick = mean)")
    ax.set_title("Consolidated ranking", fontsize=10)
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _fig_mu_sigma(blk, factors, qoi, path):
    plt = _mpl()
    if plt is None:
        return False
    s = blk["summary"]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for f in factors:
        if not np.isfinite(s[f]["mu_star"]):
            continue
        v = s[f]["verdict"]
        ax.scatter(s[f]["mu_star"], s[f]["sigma"], s=52,
                   color="#2f6f4e" if v == "RETAIN"
                   else ("#c9a227" if v == "RETAIN?" else "#9aa0a6"))
        ax.annotate(f, (s[f]["mu_star"], s[f]["sigma"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    lim = blk["mu_max"] if np.isfinite(blk["mu_max"]) else 1.0
    ax.plot([0, lim], [0, lim], ls="--", lw=0.8, color="0.6")
    # [PATCH:no-noise-floor] the vertical line marks the RELATIVE threshold, not a noise level.
    thr = blk.get("retain_frac")
    if thr and np.isfinite(lim):
        ax.axvline(thr * lim, ls=":", lw=1.0, color="#b3261e")
        ax.text(thr * lim, ax.get_ylim()[1] * 0.97, " threshold %.0f%% of mu*max" % (100 * thr),
                fontsize=7, color="#b3261e", va="top")
    ax.set_xlabel("mu*  (importance)")
    ax.set_ylabel("sigma  (non-linearity / interactions)")
    ax.set_title("Morris -- %s" % qoi)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def _fmt(v, n=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return ("%." + str(n) + "g") % v


def write_report(path, meta, cov, per_qoi, cons, factors, qoi_meta, args, figs,
                 extra=None):
    # [PATCH:screening-fixes] `extra` carries the blocks of points 5 and 6; defaults
    # to None so existing calls are not broken.
    L = []
    A = L.append
    keys = [q for q, _, _ in qoi_meta if q in per_qoi]
    retained = [r["factor"] for r in cons if r["verdict"] == "RETAIN"]
    marginal = [r["factor"] for r in cons if r["verdict"] == "RETAIN?"]
    frozen = [r["factor"] for r in cons if r["verdict"] == "freeze"]

    A("# Morris screening -- unified Drucker-Prager campaign")
    A("")
    A("| | |")
    A("|---|---|")
    A("| Campaign | `%s` |" % (meta["campaign"] or "?"))
    A("| Host family | `%s` |" % (meta["family"] or "?"))
    A("| Factors | %d : %s |" % (len(factors), ", ".join("`%s`" % f for f in factors)))
    A("| Trajectories r | %s |" % (meta["r"] or "?"))
    A("| Step Delta | %s |" % _fmt(meta["delta"], 6))
    A("| Planned / usable runs | %d / %d |" % (cov["n_design"], cov["n_usable"]))
    A("| Complete trajectories | %d / %d |" % (cov["n_traj_complete"], cov["n_traj"]))
    # [PATCH:no-noise-floor] 'noise floor' line replaced by the relative threshold.
    A("| Retention rule | mu*_lo / mu*_max >= %.2f (relative threshold) |"
      % args.retain_frac)
    A("| QoI analysed | %d |" % len(keys))
    A("| Multiplicity correction | %s |"
      % ("alpha/%d over QoI (bootstrap percentile %.3f%%)" % (len(keys), 5.0 / len(keys))
         if args.fwer == "qoi" else "none (5%% per test)"))
    A("")

    if cov["n_missing"]:
        A("> **%d missing or failed run(s).** Each missing point destroys two "
          "elementary effects. Affected ids: %s"
          % (cov["n_missing"], ", ".join(cov["missing"][:25])))
        A("")

    A("## 1. Consolidated ranking")
    A("")
    A("Rule applied: a factor is **retained if it exceeds the threshold for at "
      "least one QoI**. `mu*` is normalised per QoI (1 = dominant factor for "
      "that QoI), which makes the columns comparable across QoI of different units.")
    A("")
    A("This rule is a **union of %d tests per factor**. Without correction, the "
      "risk of wrongly retaining a null factor would reach %.0f%%; the bootstrap "
      "threshold is therefore corrected to alpha/%d." % (len(keys), 100 * (1 - 0.95 ** len(keys)), len(keys))
      if args.fwer == "qoi" else
      "No multiplicity correction is applied: over %d QoI, the per-factor false "
      "positive risk reaches %.0f%%." % (len(keys), 100 * (1 - 0.95 ** len(keys))))
    A("")
    # [PATCH:no-noise-floor] mu*/threshold column -> rel_lo.
    A("| Factor | mu* max | mu* mean | best rank | mean rank | sigma/mu* max | rel_lo max | deciding QoI | Verdict |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in cons:
        A("| `%s` | %s | %s | %s | %s | %s | %s | %s | **%s** |"
          % (r["factor"], _fmt(r["rel_max"], 3), _fmt(r["rel_mean"], 3),
             r["rank_best"] if r["rank_best"] else "-", _fmt(r["rank_mean"], 3),
             _fmt(r["sigma_ratio_max"], 3), _fmt(r["rel_lo_best"], 3),
             ", ".join("`%s`" % q for q in r["retained_in"]) or "-", r["verdict"]))
    A("")
    if figs.get("consolidated"):
        A("![Consolidated ranking](%s)" % figs["consolidated"])
        A("")
    if figs.get("heatmap"):
        A("![Normalised mu* map](%s)" % figs["heatmap"])
        A("")

    A("**Reading.** A `sigma/mu*` above %.1f flags a factor whose effect is "
      "dominated by interactions or by strong non-linearity: Morris does not "
      "distinguish the two, only Sobol will. A large gap between the best rank "
      "and the mean rank flags a factor specific to one QoI." % SIGMA_RATIO_INTERACTION)
    A("")

    A("## 2. Detail per QoI")
    A("")
    for q, lab, unit in qoi_meta:
        if q not in per_qoi:
            continue
        blk = per_qoi[q]
        A("### `%s` -- %s [%s]" % (q, lab, unit))
        A("")
        # [PATCH:no-noise-floor] no more noise threshold; the relative threshold is shown instead.
        A("%d elementary effects%s. Retention threshold: mu* >= %s "
          "(%.0f%% of the maximal mu* for this QoI)."
          % (blk["n_effects"],
             (", %d missing point(s)" % blk["n_missing"]) if blk["n_missing"] else "",
             _fmt(blk["retain_frac"] * blk["mu_max"]), 100 * blk["retain_frac"]))
        A("")
        A("| Factor | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | signed mu | n_eff | Verdict |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        order = sorted(factors, key=lambda f: -(blk["summary"][f]["mu_star"]
                                                if np.isfinite(blk["summary"][f]["mu_star"]) else -1))
        for f in order:
            s = blk["summary"][f]
            A("| `%s` | %s | %s | %s | %s | %s | %s | %s | %d | %s |"
              % (f, _fmt(s["mu_star"]), _fmt(s["mu_star_lo"]), _fmt(s["mu_star_hi"]),
                 _fmt(s["sigma"]), _fmt(s["sigma_ratio"], 3), _fmt(s.get("rel_lo"), 3),
                 _fmt(s["mu"]), s["n_eff"], s["verdict"]))
        A("")
        if figs.get(q):
            A("![Morris %s](%s)" % (q, figs[q]))
            A("")

    A("## 3. Decision")
    A("")
    # [PATCH:no-noise-floor] decision based on the relative threshold.
    A("**Retained (%d):** %s" % (len(retained), ", ".join("`%s`" % f for f in retained) or "none"))
    A("")
    A("**Retained but marginal (%d):** %s" % (len(marginal),
      ", ".join("`%s`" % f for f in marginal) or "none"))
    A("")
    A("**Frozen (%d):** %s" % (len(frozen), ", ".join("`%s`" % f for f in frozen) or "none"))
    A("")
    if marginal:
        A("> A `RETAIN?` factor has a `mu*` above the threshold but a bootstrap "
          "lower bound below it: the effect is plausible, this campaign does not "
          "resolve it. Either include them in the Sobol run, or increase r.")
        A("")
    A("> **The threshold is RELATIVE.** It compares each factor to the most "
      "influential one of the same QoI, it does not test against zero. Three "
      "consequences: the top factor is retained by construction; a `freeze` "
      "means *small compared to the largest*, **not** *null*; and if every "
      "factor had a real effect of the same order, the rule would freeze none "
      "of them. It reduces dimensionality, it does not prove any nullity.")
    A("")
    A("### Sobol")
    A("")
    A("```bash")
    A("python3 generate_design.py %s --method sobol --n 1024 \\" % (meta["family"] or "glassy_pc"))
    A("        --only %s" % (",".join(retained + marginal) if (retained or marginal) else "..."))
    A("```")
    A("")
    A("Marginal factors are included as a precaution. Unlisted factors are "
      "frozen at mid-range. Going from %d to %d factors is what makes 1024 "
      "points enough." % (len(factors), len(retained) + len(marginal)))
    A("")
    # [PATCH:screening-fixes] begin -- sections added (points 5 and 6).
    extra = extra or {}
    conf = extra.get("confounding") or []
    hot = [c for c in conf if c["index"] >= CONFOUND_THRESHOLD]
    A("## 3bis. Identifiability")
    A("")
    A("### Confounding signature between factors")
    A("")
    if hot:
        A("> **%d pair(s) above the threshold %.2f.** Opposite-signed `mu` and "
          "`mu*` of the same order, consistently across QoI: this is what two "
          "antagonistic additive terms of the same law produce. Only their "
          "difference is identifiable, and Morris assigns each of them half of "
          "a single effect. The `mu*` of these factors should not be read "
          "separately." % (len(hot), CONFOUND_THRESHOLD))
        A("")
    elif conf:
        A("No pair above the threshold %.2f: no confounding signature "
          "detected." % CONFOUND_THRESHOLD)
        A("")
    if conf:
        A("| Pair | index | QoI |")
        A("|---|---|---|")
        for c in conf[:8]:
            A("| `%s` / `%s` | %s | %d |"
              % (c["pair"][0], c["pair"][1], _fmt(c["index"], 3), c["n_qoi"]))
        A("")
    # [PATCH:no-noise-floor] 'structural noise floor' section removed.
    # [PATCH:screening-fixes] end
    A("## 4. Caveats")
    A("")
    A("- The ranking holds for the **Drucker-Prager class**, not for any single "
      "family: `semicrystalline_*` and `glassy_*` are points of the same "
      "dimensionless box. No `mu*` specific to PMMA or PC comes out of it.")
    A("- The unified box spans different physical regimes (softening present "
      "or absent). A high `sigma` may reflect this mixture rather than an "
      "interaction.")
    A("- `h` enters via `exp(h*eps^2)` evaluated up to eps_max: strong "
      "non-linearity on this factor is expected by construction of the model.")
    A("- At `phi = 1` the friction model switches from a tabulated table to "
      "constant Coulomb. Check that `phi = 0.99` and `phi = 1.00` give the "
      "same result before interpreting the `mu*` of `phi`.")
    A("")

    with open(path, "w") as f:
        f.write("\n".join(L))
    return retained, marginal, frozen


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Consolidated Morris screening report (DP).")
    ap.add_argument("results_dir", nargs="?", default=None,
                    help="folder of *_Results.csv (walked recursively)")
    ap.add_argument("--design", required=True)
    ap.add_argument("--table", default=None,
                    help="tidy CSV already produced by sweep_collector.py (skips collection)")
    # [PATCH:no-noise-floor] --noise-floor / --noise-mode / --gates removed.
    ap.add_argument("--qoi", default=None, help="comma-separated QoI subset")
    ap.add_argument("--out-dir", default="screening_dp")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--retain-frac", type=float, default=RETAIN_FRAC,
                    help="relative retention threshold: a factor is retained if "
                         "mu*_lo / mu*_max exceeds this fraction for at least "
                         "one QoI (default %.2f)." % RETAIN_FRAC)
    ap.add_argument("--fwer", default="qoi", choices=("qoi", "none"),
                    help="multiplicity correction. 'qoi' (default): the "
                         "'retained for at least one QoI' rule is a union of "
                         "n_qoi tests, so the bootstrap threshold is corrected "
                         "to alpha/n_qoi. 'none': 5%% per test, ~26%% error per "
                         "factor over 6 QoI.")
    args = ap.parse_args()

    if not args.table and not args.results_dir:
        raise SystemExit("Provide either results_dir or --table.")
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    meta, design_rows = MA.read_design(args.design)
    if not meta["delta"] or not meta["active"]:
        raise SystemExit("Could not read delta / active_factors from the design header.")
    factors = meta["active"]

    table_path = args.table or os.path.join(args.out_dir, "sweep_table.csv")
    if not args.table:
        print("Collecting...")
        _collect(args.results_dir, args.design, table_path)
    table = MA.read_table(table_path)

    cov = _coverage(design_rows, table)
    print("Coverage: %d/%d usable runs, %d/%d complete trajectories"
          % (cov["n_usable"], cov["n_design"], cov["n_traj_complete"], cov["n_traj"]))

    available = set()
    for rec in table.values():
        available.update(rec.keys())
    if args.qoi:
        wanted = [q.strip() for q in args.qoi.split(",")]
        qoi_meta = [(q, lab, u) for q, lab, u in DEFAULT_QOI if q in wanted]
        qoi_meta += [(q, q, "-") for q in wanted if q not in [x[0] for x in DEFAULT_QOI]]
    else:
        qoi_meta = list(DEFAULT_QOI)
    qoi_meta = [(q, lab, u) for q, lab, u in qoi_meta if q in available]
    if not qoi_meta:
        raise SystemExit("None of the requested QoI are present in %s" % table_path)

    # [PATCH:no-noise-floor] no more noise-floor reading.
    n_tests = len(qoi_meta) if args.fwer == "qoi" else 1
    ci_low = MA.CI[0] / float(n_tests)
    if args.fwer == "qoi" and args.bootstrap * ci_low / 100.0 < 20:
        need = int(np.ceil(20 * 100.0 / ci_low))
        print("  bootstrap raised to %d to resolve the corrected percentile %.3f%%"
              % (need, ci_low))
        args.bootstrap = need
    per_qoi = _analyse(design_rows, table, factors, meta["delta"],
                       [q for q, _, _ in qoi_meta],
                       args.bootstrap, ci_low, retain_frac=args.retain_frac)
    if not per_qoi:
        raise SystemExit("No elementary effect could be computed: check coverage.")
    cons = _consolidate(per_qoi, factors)

    figs = {}
    if _fig_consolidated(cons, os.path.join(args.out_dir, "consolidated.png")):
        figs["consolidated"] = "consolidated.png"
    if _fig_heatmap(per_qoi, factors, [q for q, _, _ in qoi_meta],
                    os.path.join(args.out_dir, "heatmap.png")):
        figs["heatmap"] = "heatmap.png"
    for q, _lab, _u in qoi_meta:
        if q in per_qoi and _fig_mu_sigma(per_qoi[q], factors, q,
                                          os.path.join(args.out_dir, "morris_%s.png" % q)):
            figs[q] = "morris_%s.png" % q

    csv_path = os.path.join(args.out_dir, "morris_summary.csv")
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        # [PATCH:no-noise-floor] mu_star_null / mu_star_mde columns removed.
        w.writerow(["qoi", "factor", "mu_star", "mu_star_lo", "mu_star_hi", "sigma",
                    "sigma_ratio", "rel", "rel_lo", "mu_signed", "n_eff",
                    "mu_star_threshold", "verdict"])
        for q, blk in per_qoi.items():
            for fac in factors:
                s = blk["summary"][fac]
                w.writerow([q, fac, s["mu_star"], s["mu_star_lo"], s["mu_star_hi"],
                            s["sigma"], s["sigma_ratio"], s["rel"], s.get("rel_lo"),
                            s["mu"], s["n_eff"],
                            blk["retain_frac"] * blk["mu_max"], s["verdict"]])

    cons_path = os.path.join(args.out_dir, "consolidated_ranking.csv")
    with open(cons_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["rank", "factor", "rel_max", "rel_mean", "rank_best", "rank_mean",
                    "sigma_ratio_max", "rel_lo_best", "n_qoi_retained", "retained_in",
                    "verdict"])
        for i, r in enumerate(cons, 1):
            w.writerow([i, r["factor"], r["rel_max"], r["rel_mean"], r["rank_best"],
                        r["rank_mean"], r["sigma_ratio_max"], r["rel_lo_best"],
                        r["n_retained"], ";".join(r["retained_in"]), r["verdict"]])

    # [PATCH:no-noise-floor] the 'structural' block (noise floor) is removed.
    extra = {"confounding": _confounding(per_qoi, factors)}
    for c in extra["confounding"][:1]:
        if c["index"] >= CONFOUND_THRESHOLD:
            print("  WARNING probable confounding: %s / %s (index %.3f)"
                  % (c["pair"][0], c["pair"][1], c["index"]))
    # [PATCH:screening-fixes] end
    md_path = os.path.join(args.out_dir, "SCREENING_REPORT.md")
    retained, marginal, frozen = write_report(md_path, meta, cov, per_qoi, cons,
                                    factors, qoi_meta, args, figs, extra=extra)

    json_path = os.path.join(args.out_dir, "retained_factors_%s.json" % (meta["family"] or "dp"))
    with open(json_path, "w") as f:
        json.dump({"family": meta["family"], "campaign": meta["campaign"],
                   "delta": meta["delta"], "factors": factors,
                   "qoi": [q for q, _, _ in qoi_meta],
                   "retain_frac": args.retain_frac, "decision_rule": "relative",
                   "retained": retained, "marginal": marginal,
                   "keep_for_sobol": retained + marginal,
                   "frozen_candidates": frozen,
                   "coverage": cov}, f, indent=2)

    print("")
    print("Report    -> %s" % md_path)
    print("Tables    -> %s , %s" % (csv_path, cons_path))
    print("Retention -> %s" % json_path)
    if retained or marginal:
        print("")
        print("Retained : %s" % (", ".join(retained) or "none"))
        print("Marginal : %s" % (", ".join(marginal) or "none"))
        print("Frozen   : %s" % (", ".join(frozen) or "none"))


if __name__ == "__main__":
    main()