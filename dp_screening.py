# Consolidated Morris screening report for the unified Drucker-Prager campaign.
#
#   python3 dp_screening.py /mon/dossier/resultats \
#           --design designs/glassy_pc_morris.csv \
#           --out-dir screening_dp
#
# # [PATCH:no-noise-floor] la retention repose sur un seuil RELATIF
# (mu*_lo / mu*_max >= --retain-frac), pas sur un plancher de bruit absolu.
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


# [PATCH:no-noise-floor] H_MPa, Ft_half_N et pile_up_ratio retires :
#   H_MPa           : contact_radius_mm est constant sur la campagne, donc
#                     H = Fn / aire fixe et r(H, Fn) = 1.000000. Le garder
#                     comptait deux fois le meme signal et gonflait la
#                     correction de multiplicite d'un doublon.
#   Ft_half_N       : Ft = Fn * scof, redondant avec les deux conserves.
#   pile_up_ratio   : sigma/mu* entre 1.3 et 2.3 sur les huit facteurs,
#                     estimateur non exploitable a ce nombre d'EE.
DEFAULT_QOI = [
    ("Fn_half_N", "Force normale (demi-modele)", "N"),
    ("scof", "Coefficient de frottement apparent", "-"),
    ("residual_depth_mm", "Profondeur residuelle", "mm"),
    ("pile_up_mm", "Bourrelet lateral", "mm"),
]

SIGMA_RATIO_INTERACTION = 1.0    # sigma/mu* above this => interaction-dominated
RETAIN_FRAC = 0.10               # mu*_lo / mu*_max minimal pour retenir


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
    # [PATCH:screening-fixes] original (definition dupliquee, divergeait de
    # elementary_effects qui ignorait parse_error) :
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


# [PATCH:no-noise-floor] decision sans plancher de bruit.
#
# L'ancienne regle testait mu*_lo contre mu*_null = sqrt(2/pi) * sqrt(2)/Delta * sigma_num,
# c'est-a-dire contre une echelle de bruit ABSOLUE. Sans plancher, aucun
# verdict n'etait rendu. La regle qui la remplace est RELATIVE :
#
#     rel_lo = mu*_lo / mu*_max      (par QoI)
#     RETAIN   si rel_lo >= retain_frac
#     RETAIN?  si rel    >= retain_frac mais rel_lo < retain_frac
#     freeze   sinon
#
# Elle conserve la borne basse du bootstrap, donc elle reste sensible a la
# censure : un facteur ampute a un intervalle plus large et un rel_lo plus
# bas. Mais elle classe par rapport au facteur dominant de chaque QoI, elle
# ne teste pas contre zero -- le facteur de tete est retenu par construction
# et un 'freeze' signifie 'petit devant le plus grand', pas 'nul'.
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


# [PATCH:screening-fixes] begin -- points 5 et 6.
CONFOUND_THRESHOLD = 0.55        # indice de confusion au-dela duquel on alerte


def _confounding(per_qoi, factors):
    """
    POINT 5 -- exploite le `mu` SIGNE, jusqu'ici calcule et affiche mais
    absent de tout verdict.

    Deux facteurs entrant de facon additive et antagoniste dans la meme loi
    ne sont pas separement identifiables. La signature observable est : `mu`
    de signes opposes, `mu*` de meme ordre, et cela de facon COHERENTE sur
    les QoI. Indice par paire, moyenne sur les QoI de

        min(mu*_i, mu*_j) / max(mu*_i, mu*_j)  x  max(0, -sign(mu_i) sign(mu_j))

    1 = parfaitement antagonistes et de meme poids ; 0 = pas de signature.
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


# [PATCH:no-noise-floor] _structural_floor supprimee (plancher de bruit).
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
            # [PATCH:no-noise-floor] margin (mu*/seuil de bruit) -> rel_lo_best.
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
    # [PATCH:no-noise-floor] la ligne verticale marque le seuil RELATIF, pas un bruit.
    thr = blk.get("retain_frac")
    if thr and np.isfinite(lim):
        ax.axvline(thr * lim, ls=":", lw=1.0, color="#b3261e")
        ax.text(thr * lim, ax.get_ylim()[1] * 0.97, " seuil %.0f%% de mu*max" % (100 * thr),
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


def write_report(path, meta, cov, per_qoi, cons, factors, qoi_meta, args, figs,
                 extra=None):
    # [PATCH:screening-fixes] `extra` porte les blocs des points 5 et 6 ; valeur par
    # defaut None pour ne pas casser les appels existants.
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
    # [PATCH:no-noise-floor] ligne 'plancher de bruit' remplacee par le seuil relatif.
    A("| Regle de retention | mu*_lo / mu*_max >= %.2f (seuil relatif) |"
      % args.retain_frac)
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
    # [PATCH:no-noise-floor] colonne mu*/seuil -> rel_lo.
    A("| Facteur | mu* max | mu* moyen | meilleur rang | rang moyen | sigma/mu* max | rel_lo max | QoI decisives | Verdict |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in cons:
        A("| `%s` | %s | %s | %s | %s | %s | %s | %s | **%s** |"
          % (r["factor"], _fmt(r["rel_max"], 3), _fmt(r["rel_mean"], 3),
             r["rank_best"] if r["rank_best"] else "-", _fmt(r["rank_mean"], 3),
             _fmt(r["sigma_ratio_max"], 3), _fmt(r["rel_lo_best"], 3),
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
        # [PATCH:no-noise-floor] plus de seuil de bruit ; on affiche le seuil relatif.
        A("%d effets elementaires%s. Seuil de retention : mu* >= %s "
          "(%.0f%% du mu* maximal de cette QoI)."
          % (blk["n_effects"],
             (", %d point(s) manquant(s)" % blk["n_missing"]) if blk["n_missing"] else "",
             _fmt(blk["retain_frac"] * blk["mu_max"]), 100 * blk["retain_frac"]))
        A("")
        A("| Facteur | mu* | mu*_lo | mu*_hi | sigma | sigma/mu* | rel_lo | mu signe | n_eff | Verdict |")
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
    # [PATCH:no-noise-floor] decision fondee sur le seuil relatif.
    A("**Retenus (%d) :** %s" % (len(retained), ", ".join("`%s`" % f for f in retained) or "aucun"))
    A("")
    A("**Retenus mais marginaux (%d) :** %s" % (len(marginal),
      ", ".join("`%s`" % f for f in marginal) or "aucun"))
    A("")
    A("**Geles (%d) :** %s" % (len(frozen), ", ".join("`%s`" % f for f in frozen) or "aucun"))
    A("")
    if marginal:
        A("> Un facteur `RETAIN?` a un `mu*` au-dessus du seuil mais une borne basse "
          "bootstrap en dessous : l'effet est plausible, la campagne ne le resout pas. "
          "Les inclure dans le Sobol, ou augmenter r.")
        A("")
    A("> **Le seuil est RELATIF.** Il compare chaque facteur au plus influent de la "
      "meme QoI, il ne teste pas contre zero. Trois consequences : le facteur de "
      "tete est retenu par construction ; un `freeze` signifie *petit devant le plus "
      "grand*, **pas** *nul* ; et si tous les facteurs avaient un effet reel du meme "
      "ordre, la regle n'en gelerait aucun. Elle reduit la dimension, elle ne prouve "
      "aucune nullite.")
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
    # [PATCH:screening-fixes] begin -- sections ajoutees (points 5 et 6).
    extra = extra or {}
    conf = extra.get("confounding") or []
    hot = [c for c in conf if c["index"] >= CONFOUND_THRESHOLD]
    A("## 3bis. Identifiabilite")
    A("")
    A("### Signature de confusion entre facteurs")
    A("")
    if hot:
        A("> **%d paire(s) au-dessus du seuil %.2f.** `mu` de signes opposes et "
          "`mu*` de meme ordre, de facon coherente sur les QoI : c'est ce que "
          "produisent deux termes additifs antagonistes de la meme loi. Seule "
          "leur difference est identifiable, et Morris attribue a chacun la "
          "moitie d'un effet unique. Les `mu*` de ces facteurs ne doivent pas "
          "etre lus separement." % (len(hot), CONFOUND_THRESHOLD))
        A("")
    elif conf:
        A("Aucune paire au-dessus du seuil %.2f : pas de signature de "
          "confusion detectee." % CONFOUND_THRESHOLD)
        A("")
    if conf:
        A("| Paire | indice | QoI |")
        A("|---|---|---|")
        for c in conf[:8]:
            A("| `%s` / `%s` | %s | %d |"
              % (c["pair"][0], c["pair"][1], _fmt(c["index"], 3), c["n_qoi"]))
        A("")
    # [PATCH:no-noise-floor] section 'plancher structurel' supprimee.
    # [PATCH:screening-fixes] end
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
    # [PATCH:no-noise-floor] --noise-floor / --noise-mode / --gates retires.
    ap.add_argument("--qoi", default=None, help="sous-ensemble de QoI, separees par des virgules")
    ap.add_argument("--out-dir", default="screening_dp")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--retain-frac", type=float, default=RETAIN_FRAC,
                    help="seuil relatif de retention : un facteur est retenu si "
                         "mu*_lo / mu*_max depasse cette fraction sur au moins "
                         "une QoI (defaut %.2f)." % RETAIN_FRAC)
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

    # [PATCH:no-noise-floor] plus de lecture de plancher de bruit.
    n_tests = len(qoi_meta) if args.fwer == "qoi" else 1
    ci_low = MA.CI[0] / float(n_tests)
    if args.fwer == "qoi" and args.bootstrap * ci_low / 100.0 < 20:
        need = int(np.ceil(20 * 100.0 / ci_low))
        print("  bootstrap porte a %d pour resoudre le percentile corrige %.3f%%"
              % (need, ci_low))
        args.bootstrap = need
    per_qoi = _analyse(design_rows, table, factors, meta["delta"],
                       [q for q, _, _ in qoi_meta],
                       args.bootstrap, ci_low, retain_frac=args.retain_frac)
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
        # [PATCH:no-noise-floor] colonnes mu_star_null / mu_star_mde retirees.
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

    # [PATCH:no-noise-floor] le bloc 'structural' (plancher structurel) est retire.
    extra = {"confounding": _confounding(per_qoi, factors)}
    for c in extra["confounding"][:1]:
        if c["index"] >= CONFOUND_THRESHOLD:
            print("  ATTENTION confusion probable : %s / %s (indice %.3f)"
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