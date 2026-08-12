# Consolidated Morris screening report for the unified Drucker-Prager campaign.
#
#   python3 dp_screening.py /mon/dossier/resultats \
#           --design designs/glassy_pc_morris.csv \
#           --noise-floor noise_dp.json \
#           --out-dir screening_dp
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

import ScratchSimulation.morris_analysis as MA


DEFAULT_QOI = [
    ("Fn_half_N", "Force normale (demi-modele)", "N"),
    ("scof", "Coefficient de frottement apparent", "-"),
    ("H_MPa", "Durete de rayage Fn/A_c", "MPa"),
    ("residual_depth_mm", "Profondeur residuelle", "mm"),
    ("pile_up_mm", "Bourrelet lateral", "mm"),
    ("pile_up_ratio", "Bourrelet / profondeur residuelle", "-"),
]

SIGMA_RATIO_INTERACTION = 1.0    # sigma/mu* above this => interaction-dominated
MARGIN_SCRUTINY = 2.0            # mu*/mu*_null below this => retention to scrutinise


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
    have = set(k for k, v in table.items()
               if k in ids and not v.get("parse_error") and v.get("status") != "FAIL")
    per_traj = {}
    for rid, d in design_rows.items():
        per_traj.setdefault(int(d["traj"]), []).append(rid)
    complete = sum(1 for t, r in per_traj.items() if all(x in have for x in r))
    return {"n_design": len(ids), "n_usable": len(have),
            "n_missing": len(ids - have), "missing": sorted(ids - have),
            "n_traj": len(per_traj), "n_traj_complete": complete}


def _analyse(design_rows, table, factors, delta, qoi_keys, noise, noise_rel,
             bootstrap, ci_low):
    out = {}
    for q in qoi_keys:
        effects, n_missing = MA.elementary_effects(design_rows, table, q, delta, ("FAIL",))
        if not effects:
            continue
        summary = MA.summarise(effects, factors, n_bootstrap=bootstrap, ci_low=ci_low)
        sig = noise.get(q)
        if sig is not None and noise_rel:
            scale = np.nanmean([MA._num(rec, q) for rec in table.values()])
            sig = sig * abs(scale)
        ee_noise = (sig * np.sqrt(2.0) / delta) if sig is not None else None
        mu_null = (np.sqrt(2.0 / np.pi) * ee_noise) if ee_noise is not None else None
        mu_max = max([summary[f]["mu_star"] for f in factors
                      if np.isfinite(summary[f]["mu_star"])] or [np.nan])
        spread = [summary[f]["mu_star_hi"] - summary[f]["mu_star_lo"] for f in factors
                  if np.isfinite(summary[f]["mu_star_hi"])]
        mde = (mu_null + 0.5 * float(np.median(spread))) if (mu_null is not None and spread) else None
        for f in factors:
            s = summary[f]
            s["rel"] = (s["mu_star"] / mu_max) if mu_max and np.isfinite(mu_max) else np.nan
            s["sigma_ratio"] = (s["sigma"] / s["mu_star"]) if s["mu_star"] else np.nan
            s["margin"] = (s["mu_star"] / mu_null) if mu_null else np.nan
            if mu_null is None:
                s["verdict"] = "n/a"
            elif np.isfinite(s["mu_star_lo"]) and s["mu_star_lo"] > mu_null:
                s["verdict"] = ("RETAIN" if s["margin"] >= MARGIN_SCRUTINY
                                else "RETAIN?")
            else:
                s["verdict"] = "freeze"
        out[q] = {"summary": summary, "n_effects": len(effects), "n_missing": n_missing,
                  "mu_null": mu_null, "ee_noise": ee_noise, "mde": mde,
                  "mu_max": mu_max, "sigma_num": sig}
    return out


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
            if s["verdict"].startswith("RETAIN"):
                retained_in.append(q)
                margins.append(s["margin"])
        rows.append({
            "factor": f,
            "rel_max": max(rels) if rels else np.nan,
            "rel_mean": float(np.mean(rels)) if rels else np.nan,
            "rank_best": min(ranks) if ranks else None,
            "rank_mean": float(np.mean(ranks)) if ranks else np.nan,
            "sigma_ratio_max": max(sig_ratio) if sig_ratio else np.nan,
            "retained_in": retained_in,
            "n_retained": len(retained_in),
            "margin_best": max([m for m in margins if np.isfinite(m)] or [np.nan]),
            "verdict": (("RETAIN" if max([m for m in margins if np.isfinite(m)] or [0])
                         >= MARGIN_SCRUTINY else "RETAIN?") if retained_in else
                        ("freeze" if any(blk["mu_null"] is not None for blk in per_qoi.values())
                         else "n/a")),
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
    ax.set_title("mu* normalise par QoI (1 = facteur dominant)", fontsize=10)
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
    ax.set_xlabel("mu* normalise  (barre = max sur les QoI, trait = moyenne)")
    ax.set_title("Classement consolide", fontsize=10)
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
    if blk["mu_null"] is not None:
        ax.axvline(blk["mu_null"], ls=":", lw=1.0, color="#b3261e")
        ax.text(blk["mu_null"], ax.get_ylim()[1] * 0.97, " seuil bruit",
                fontsize=7, color="#b3261e", va="top")
    ax.set_xlabel("mu*  (importance)")
    ax.set_ylabel("sigma  (non-linearite / interactions)")
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


def write_report(path, meta, cov, per_qoi, cons, factors, qoi_meta, args, figs):
    L = []
    A = L.append
    keys = [q for q, _, _ in qoi_meta if q in per_qoi]
    retained = [r["factor"] for r in cons if r["verdict"] == "RETAIN"]
    marginal = [r["factor"] for r in cons if r["verdict"] == "RETAIN?"]
    frozen = [r["factor"] for r in cons if r["verdict"] == "freeze"]

    A("# Criblage Morris -- campagne Drucker-Prager unifiee")
    A("")
    A("| | |")
    A("|---|---|")
    A("| Campagne | `%s` |" % (meta["campaign"] or "?"))
    A("| Famille hote | `%s` |" % (meta["family"] or "?"))
    A("| Facteurs | %d : %s |" % (len(factors), ", ".join("`%s`" % f for f in factors)))
    A("| Trajectoires r | %s |" % (meta["r"] or "?"))
    A("| Pas Delta | %s |" % _fmt(meta["delta"], 6))
    A("| Runs prevus / exploitables | %d / %d |" % (cov["n_design"], cov["n_usable"]))
    A("| Trajectoires completes | %d / %d |" % (cov["n_traj_complete"], cov["n_traj"]))
    A("| Plancher de bruit | %s |"
      % ("fourni (%s)" % args.noise_mode if args.noise_floor else "**absent : aucune decision rendue**"))
    A("| QoI analysees | %d |" % len(keys))
    A("| Correction de multiplicite | %s |"
      % ("alpha/%d sur les QoI (percentile bootstrap %.3f%%)" % (len(keys), 5.0 / len(keys))
         if args.fwer == "qoi" else "aucune (5%% par test)"))
    A("")

    if cov["n_missing"]:
        A("> **%d run(s) manquant(s) ou en echec.** Chaque point manquant detruit deux "
          "effets elementaires. Points concernes : %s"
          % (cov["n_missing"], ", ".join(cov["missing"][:25])))
        A("")

    A("## 1. Classement consolide")
    A("")
    A("Regle appliquee : un facteur est **retenu s'il depasse le seuil de bruit pour au "
      "moins une QoI**. `mu*` est normalise par QoI (1 = facteur dominant de cette QoI), "
      "ce qui rend les colonnes comparables entre QoI d'unites differentes.")
    A("")
    A("Cette regle est une **union de %d tests par facteur**. Sans correction, le risque "
      "de retenir a tort un facteur nul atteindrait %.0f%% ; le seuil bootstrap est donc "
      "corrige en alpha/%d." % (len(keys), 100 * (1 - 0.95 ** len(keys)), len(keys))
      if args.fwer == "qoi" else
      "Aucune correction de multiplicite n'est appliquee : sur %d QoI, le risque de "
      "faux positif par facteur atteint %.0f%%." % (len(keys), 100 * (1 - 0.95 ** len(keys))))
    A("")
    A("| Facteur | mu* max | mu* moyen | meilleur rang | rang moyen | sigma/mu* max | mu*/seuil | QoI decisives | Verdict |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in cons:
        A("| `%s` | %s | %s | %s | %s | %s | %s | %s | **%s** |"
          % (r["factor"], _fmt(r["rel_max"], 3), _fmt(r["rel_mean"], 3),
             r["rank_best"] if r["rank_best"] else "-", _fmt(r["rank_mean"], 3),
             _fmt(r["sigma_ratio_max"], 3), _fmt(r["margin_best"], 3),
             ", ".join("`%s`" % q for q in r["retained_in"]) or "-", r["verdict"]))
    A("")
    if figs.get("consolidated"):
        A("![Classement consolide](%s)" % figs["consolidated"])
        A("")
    if figs.get("heatmap"):
        A("![Carte mu* normalise](%s)" % figs["heatmap"])
        A("")

    A("**Lecture.** Un `sigma/mu*` superieur a %.1f signale un facteur dont l'effet est "
      "domine par les interactions ou par une forte non-linearite : Morris ne distingue "
      "pas les deux, seul le Sobol le fera. Un ecart important entre le meilleur rang et "
      "le rang moyen signale un facteur specifique a une QoI." % SIGMA_RATIO_INTERACTION)
    A("")

    A("## 2. Detail par QoI")
    A("")
    for q, lab, unit in qoi_meta:
        if q not in per_qoi:
            continue
        blk = per_qoi[q]
        A("### `%s` -- %s [%s]" % (q, lab, unit))
        A("")
        A("%d effets elementaires%s. %s"
          % (blk["n_effects"],
             (", %d point(s) manquant(s)" % blk["n_missing"]) if blk["n_missing"] else "",
             ("Seuil de bruit sur mu* : %s ; effet minimal detectable : %s."
              % (_fmt(blk["mu_null"]), _fmt(blk["mde"])))
             if blk["mu_null"] is not None else "Pas de plancher de bruit fourni."))
        A("")
        A("| Facteur | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | mu*/seuil | mu signe | n_eff | Verdict |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        order = sorted(factors, key=lambda f: -(blk["summary"][f]["mu_star"]
                                                if np.isfinite(blk["summary"][f]["mu_star"]) else -1))
        for f in order:
            s = blk["summary"][f]
            A("| `%s` | %s | %s | %s | %s | %s | %s | %s | %d | %s |"
              % (f, _fmt(s["mu_star"]), _fmt(s["mu_star_lo"]), _fmt(s["mu_star_hi"]),
                 _fmt(s["sigma"]), _fmt(s["sigma_ratio"], 3), _fmt(s.get("margin"), 3),
                 _fmt(s["mu"]), s["n_eff"], s["verdict"]))
        A("")
        if figs.get(q):
            A("![Morris %s](%s)" % (q, figs[q]))
            A("")

    A("## 3. Decision")
    A("")
    if not args.noise_floor:
        A("Aucun plancher de bruit n'a ete fourni : les `mu*` et `sigma` ci-dessus sont "
          "valides, mais **aucune retention ni gel n'est prononce**. Mesurer d'abord "
          "`sigma_num` avec `noise_floor.py`, puis relancer.")
    else:
        A("**Retenus (%d) :** %s" % (len(retained), ", ".join("`%s`" % f for f in retained) or "aucun"))
        A("")
        A("**Retenus mais marginaux (%d) :** %s" % (len(marginal),
          ", ".join("`%s`" % f for f in marginal) or "aucun"))
        A("")
        A("**Geles (%d) :** %s" % (len(frozen), ", ".join("`%s`" % f for f in frozen) or "aucun"))
        A("")
        if marginal:
            A("> Un facteur `RETAIN?` depasse le seuil de moins d'un facteur %.0f. Sur une "
              "QoI construite comme un **ratio** (`pile_up_ratio`), le plancher de bruit "
              "est heteroscedastique et sous-estime par une mesure faite en quelques "
              "points : ces retentions sont les plus susceptibles d'etre des faux "
              "positifs. Les traiter comme a verifier, soit en densifiant le plancher de "
              "bruit sur cette QoI, soit en les incluant dans le Sobol ou leur indice "
              "tranchera." % MARGIN_SCRUTINY)
            A("")
        A("Un gel signifie *aucune preuve d'effet au-dessus du bruit*, **pas** une preuve "
          "de nullite. Les colonnes `mde` de la section 2 indiquent ce que la campagne "
          "etait capable de detecter ; si le `mde` est du meme ordre que les `mu*` "
          "retenus, la campagne manquait de puissance et il faut augmenter r plutot que "
          "conclure.")
        A("")
        A("### Sobol")
        A("")
        A("```bash")
        A("python3 generate_design.py %s --method sobol --n 1024 \\" % (meta["family"] or "glassy_pc"))
        A("        --only %s" % (",".join(retained + marginal) if (retained or marginal) else "..."))
        A("```")
        A("")
        A("Les facteurs marginaux sont inclus par prudence. Les facteurs non listes sont "
          "geles en milieu de plage. Passer de %d a %d facteurs est ce qui rend 1024 "
          "points suffisants." % (len(factors), len(retained) + len(marginal)))
    A("")
    A("## 4. Reserves")
    A("")
    A("- Le classement vaut pour la **classe Drucker-Prager**, pas pour une famille "
      "particuliere : `semicrystalline_*` et `glassy_*` sont des points d'une meme boite "
      "adimensionnelle. Aucun `mu*` propre a PMMA ou PC n'en sort.")
    A("- La boite unifiee couvre des regimes physiques differents (adoucissement present "
      "ou absent). Un `sigma` eleve peut refleter ce melange plutot qu'une interaction.")
    A("- `h` intervient via `exp(h*eps^2)` evalue jusqu'a eps_max : une non-linearite forte "
      "sur ce facteur est attendue par construction du modele.")
    A("- A `phi = 1` le modele de frottement bascule de table tabulee vers Coulomb "
      "constant. Verifier que `phi = 0.99` et `phi = 1.00` donnent le meme resultat avant "
      "d'interpreter le `mu*` de `phi`.")
    A("")

    with open(path, "w") as f:
        f.write("\n".join(L))
    return retained, marginal, frozen


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Rapport de criblage Morris consolide (DP).")
    ap.add_argument("results_dir", nargs="?", default=None,
                    help="dossier des *_Results.csv (parcouru recursivement)")
    ap.add_argument("--design", required=True)
    ap.add_argument("--table", default=None,
                    help="CSV tidy deja produit par sweep_collector.py (evite la collecte)")
    ap.add_argument("--noise-floor", default=None)
    ap.add_argument("--noise-mode", default="relative", choices=("relative", "absolute"))
    ap.add_argument("--qoi", default=None, help="sous-ensemble de QoI, separees par des virgules")
    ap.add_argument("--out-dir", default="screening_dp")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--fwer", default="qoi", choices=("qoi", "none"),
                    help="correction de multiplicite. 'qoi' (defaut) : la regle "
                         "'retenu pour au moins une QoI' est une union de n_qoi tests, "
                         "le seuil bootstrap est donc corrige en alpha/n_qoi. "
                         "'none' : 5%% par test, ~26%% d'erreur par facteur sur 6 QoI.")
    args = ap.parse_args()

    if not args.table and not args.results_dir:
        raise SystemExit("Fournir soit results_dir, soit --table.")
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    meta, design_rows = MA.read_design(args.design)
    if not meta["delta"] or not meta["active"]:
        raise SystemExit("Impossible de lire delta / active_factors dans l'en-tete du design.")
    factors = meta["active"]

    table_path = args.table or os.path.join(args.out_dir, "sweep_table.csv")
    if not args.table:
        print("Collecte...")
        _collect(args.results_dir, args.design, table_path)
    table = MA.read_table(table_path)

    cov = _coverage(design_rows, table)
    print("Couverture : %d/%d runs exploitables, %d/%d trajectoires completes"
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
        raise SystemExit("Aucune QoI demandee n'est presente dans %s" % table_path)

    noise, noise_rel = {}, (args.noise_mode == "relative")
    if args.noise_floor:
        with open(args.noise_floor, "r") as f:
            payload = json.load(f)
        if "sigma" in payload and isinstance(payload["sigma"], dict):
            noise = payload["sigma_rel" if noise_rel else "sigma"]
        else:
            noise = payload
            if noise_rel:
                raise SystemExit("Fichier de bruit plat sans sigma_rel : utiliser "
                                 "--noise-mode absolute ou regenerer avec noise_floor.py.")

    n_tests = len(qoi_meta) if args.fwer == "qoi" else 1
    ci_low = MA.CI[0] / float(n_tests)
    if args.fwer == "qoi" and args.bootstrap * ci_low / 100.0 < 20:
        need = int(np.ceil(20 * 100.0 / ci_low))
        print("  bootstrap porte a %d pour resoudre le percentile corrige %.3f%%"
              % (need, ci_low))
        args.bootstrap = need
    per_qoi = _analyse(design_rows, table, factors, meta["delta"],
                       [q for q, _, _ in qoi_meta], noise, noise_rel,
                       args.bootstrap, ci_low)
    if not per_qoi:
        raise SystemExit("Aucun effet elementaire calculable : verifier la couverture.")
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
        w.writerow(["qoi", "factor", "mu_star", "mu_star_lo", "mu_star_hi", "sigma",
                    "sigma_ratio", "margin", "mu_signed", "n_eff", "rel", "mu_star_null",
                    "mu_star_mde", "verdict"])
        for q, blk in per_qoi.items():
            for fac in factors:
                s = blk["summary"][fac]
                w.writerow([q, fac, s["mu_star"], s["mu_star_lo"], s["mu_star_hi"],
                            s["sigma"], s["sigma_ratio"], s.get("margin"), s["mu"],
                            s["n_eff"], s["rel"], blk["mu_null"], blk["mde"], s["verdict"]])

    cons_path = os.path.join(args.out_dir, "consolidated_ranking.csv")
    with open(cons_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["rank", "factor", "rel_max", "rel_mean", "rank_best", "rank_mean",
                    "sigma_ratio_max", "margin_best", "n_qoi_retained", "retained_in",
                    "verdict"])
        for i, r in enumerate(cons, 1):
            w.writerow([i, r["factor"], r["rel_max"], r["rel_mean"], r["rank_best"],
                        r["rank_mean"], r["sigma_ratio_max"], r["margin_best"],
                        r["n_retained"], ";".join(r["retained_in"]), r["verdict"]])

    md_path = os.path.join(args.out_dir, "SCREENING_REPORT.md")
    retained, marginal, frozen = write_report(md_path, meta, cov, per_qoi, cons,
                                    factors, qoi_meta, args, figs)

    json_path = os.path.join(args.out_dir, "retained_factors_%s.json" % (meta["family"] or "dp"))
    with open(json_path, "w") as f:
        json.dump({"family": meta["family"], "campaign": meta["campaign"],
                   "delta": meta["delta"], "factors": factors,
                   "qoi": [q for q, _, _ in qoi_meta],
                   "noise_floor_supplied": bool(noise), "noise_mode": args.noise_mode,
                   "retained": retained, "marginal": marginal,
                   "keep_for_sobol": retained + marginal,
                   "frozen_candidates": frozen,
                   "coverage": cov}, f, indent=2)

    print("")
    print("Rapport   -> %s" % md_path)
    print("Tables    -> %s , %s" % (csv_path, cons_path))
    print("Retention -> %s" % json_path)
    if retained or marginal:
        print("")
        print("Retenus   : %s" % (", ".join(retained) or "aucun"))
        print("Marginaux : %s" % (", ".join(marginal) or "aucun"))
        print("Geles     : %s" % (", ".join(frozen) or "aucun"))


if __name__ == "__main__":
    main()