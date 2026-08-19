#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch_screening_pipeline.py
===========================

Corrige quatre defauts du pipeline de criblage Morris.

    python patch_screening_pipeline.py [dossier]   # defaut : repertoire courant
    python patch_screening_pipeline.py --check

POINT 2 -- verdicts divergents entre les deux points d'entree
-------------------------------------------------------------
`summarise()` accepte `ci_low`. `dp_screening.py` le passe (5/n_qoi, soit
0.833 % sur 6 QoI), `morris_analysis.py` ne le passe pas et reste a 5 %.
Lancer les deux sur les MEMES donnees produit deux listes de retention
differentes sans que rien ne le signale.

  -> `--fwer {qoi,none}` est ajoute a `morris_analysis.py`, avec la meme
     logique de relevement automatique du bootstrap que `dp_screening.py`.
  -> `mu_star_hi` devient symetrique de `mu_star_lo` (100 - ci_low au lieu
     de 95 fixe) : l'intervalle affiche correspond enfin au test applique.

POINT 3 -- deux definitions de "exploitable"
--------------------------------------------
`_coverage` exclut sur `parse_error` ET `status == FAIL` ; `elementary_effects`
n'exclut que sur `FAIL`. Un run avec `parse_error` mais un statut non-FAIL est
compte manquant dans l'en-tete du rapport et utilise dans le calcul.

  -> helper commun `_usable(rec, drop_status)`, utilise des deux cotes.
  -> `spread` dans `dp_screening._analyse` ne filtrait que `mu_star_hi` : un
     `mu_star_lo` NaN propageait un NaN et annulait silencieusement le `mde`.

POINT 5 -- `mu` signe calcule mais inexploite
---------------------------------------------
Deux facteurs entrant de facon additive et antagoniste dans la meme loi
produisent des `mu` de signes opposes avec des `mu*` de meme ordre, de facon
coherente sur toutes les QoI. C'est la signature de la confusion q / s, et
rien ne la testait.

  -> `_confounding()` classe toutes les paires par cet indice ; les paires
     au-dessus du seuil sont signalees dans le rapport (section 3).

POINT 6 -- plancher de bruit structurel inexploite
--------------------------------------------------
Un facteur peut etre INERTE par construction dans une partie du domaine
(`eps_soft` n'entre dans aucune equation quand `s = 0`). Les effets
elementaires mesures sur ces paires sont nuls par construction : ce qui
reste est du bruit, mesure sur la campagne reelle, sans run supplementaire.

  -> `--gates gate:inert` (defaut `s:eps_soft`) declare ces relations ;
     le rapport compare le plancher observe au `mu_star_null` fourni et
     previent si ce dernier est trop bas.

Discipline : ancres exactes, originaux commentes, CRLF preserves, validation
`ast.parse`, idempotent.
"""

import ast
import os
import shutil
import sys

MARKER = "# [PATCH:screening-fixes]"


# ======================================================================
# morris_analysis.py
# ======================================================================

MA_PATCHES = []

# ---- point 3 : helper commun -----------------------------------------
MA_PATCHES.append((
    "point 3 -- helper _usable",
    "def elementary_effects(design_rows, table, qoi, delta, drop_status=(\"FAIL\",)):\n",
    MARKER + " begin -- definition unique de \"run exploitable\".\n"
    "def _usable(rec, drop_status=(\"FAIL\",)):\n"
    "    \"\"\"\n"
    "    Vrai si le run peut alimenter un effet elementaire.\n"
    "\n"
    "    `_coverage` (dp_screening) excluait `parse_error` alors que\n"
    "    `elementary_effects` ne l'excluait pas : le nombre de runs annonce en\n"
    "    tete du rapport ne correspondait pas a celui reellement utilise. Les\n"
    "    deux passent desormais par ici.\n"
    "    \"\"\"\n"
    "    if rec is None:\n"
    "        return False\n"
    "    if str(rec.get(\"status\", \"\")) in drop_status:\n"
    "        return False\n"
    "    if rec.get(\"parse_error\"):\n"
    "        return False\n"
    "    return True\n"
    "\n"
    "\n"
    + MARKER + " end\n"
    "def elementary_effects(design_rows, table, qoi, delta, drop_status=(\"FAIL\",)):\n"))

MA_PATCHES.append((
    "point 3 -- elementary_effects utilise _usable",
    "            rec = table.get(rid)\n"
    "            if rec is None or str(rec.get(\"status\", \"\")) in drop_status:\n",
    "            rec = table.get(rid)\n"
    "            " + MARKER + " original : if rec is None or str(rec.get(\"status\", \"\")) in drop_status:\n"
    "            if not _usable(rec, drop_status):\n"))

# ---- point 2 : intervalle symetrique ---------------------------------
MA_PATCHES.append((
    "point 2 -- mu_star_hi symetrique de mu_star_lo",
    "    lo_pct = CI[0] if ci_low is None else float(ci_low)\n",
    "    lo_pct = CI[0] if ci_low is None else float(ci_low)\n"
    "    " + MARKER + " l'intervalle doit etre symetrique du test applique :\n"
    "    # hi restait fige a 95 % meme quand lo etait corrige a 0.833 %.\n"
    "    hi_pct = CI[1] if ci_low is None else (100.0 - float(ci_low))\n"))

MA_PATCHES.append((
    "point 2 -- percentile haut",
    "            \"mu_star_hi\": float(np.percentile(b, CI[1])) if b.size else np.nan,\n",
    "            " + MARKER + " original : float(np.percentile(b, CI[1]))\n"
    "            \"mu_star_hi\": float(np.percentile(b, hi_pct)) if b.size else np.nan,\n"))

# ---- point 2 : option --fwer -----------------------------------------
MA_PATCHES.append((
    "point 2 -- option --fwer",
    "    ap.add_argument(\"--keep-failed\", action=\"store_true\",\n"
    "                    help=\"keep runs whose verifier status is FAIL\")\n"
    "    args = ap.parse_args()\n",
    "    ap.add_argument(\"--keep-failed\", action=\"store_true\",\n"
    "                    help=\"keep runs whose verifier status is FAIL\")\n"
    "    " + MARKER + " meme correction de multiplicite que dp_screening.py,\n"
    "    # sans quoi les deux scripts rendaient des verdicts differents sur les\n"
    "    # memes donnees. La regle 'retenu pour au moins une QoI' est une union\n"
    "    # de n_qoi tests ; sans correction le risque de faux positif par facteur\n"
    "    # atteint 26 % sur 6 QoI.\n"
    "    ap.add_argument(\"--fwer\", default=\"qoi\", choices=(\"qoi\", \"none\"),\n"
    "                    help=\"multiplicity correction over the QoI union test. \"\n"
    "                         \"'qoi' (default, matches dp_screening.py): bootstrap \"\n"
    "                         \"threshold corrected to alpha/n_qoi. 'none': 5%% per test.\")\n"
    "    args = ap.parse_args()\n"))

MA_PATCHES.append((
    "point 2 -- ci_low applique dans main",
    "    rows, retained = [], set()\n"
    "    for qoi in qois:\n"
    "        effects, n_missing = elementary_effects(design_rows, table, qoi, delta, drop)\n"
    "        summary = summarise(effects, factors, n_bootstrap=args.bootstrap)\n",
    "    " + MARKER + " begin -- correction de multiplicite (etait absente ici).\n"
    "    n_tests = len(qois) if args.fwer == \"qoi\" else 1\n"
    "    ci_low = CI[0] / float(n_tests)\n"
    "    if args.fwer == \"qoi\" and args.bootstrap * ci_low / 100.0 < 20:\n"
    "        need = int(np.ceil(20 * 100.0 / ci_low))\n"
    "        print(\"  bootstrap porte a %d pour resoudre le percentile corrige %.3f%%\"\n"
    "              % (need, ci_low))\n"
    "        args.bootstrap = need\n"
    "    " + MARKER + " end\n"
    "    rows, retained = [], set()\n"
    "    for qoi in qois:\n"
    "        effects, n_missing = elementary_effects(design_rows, table, qoi, delta, drop)\n"
    "        " + MARKER + " original : summarise(effects, factors, n_bootstrap=args.bootstrap)\n"
    "        summary = summarise(effects, factors, n_bootstrap=args.bootstrap,\n"
    "                            ci_low=ci_low)\n"))


# ======================================================================
# dp_screening.py
# ======================================================================

DP_PATCHES = []

# ---- point 3 : coverage + spread --------------------------------------
DP_PATCHES.append((
    "point 3 -- _coverage utilise _usable",
    "    have = set(k for k, v in table.items()\n"
    "               if k in ids and not v.get(\"parse_error\") and v.get(\"status\") != \"FAIL\")\n",
    "    " + MARKER + " original (definition dupliquee, divergeait de\n"
    "    # elementary_effects qui ignorait parse_error) :\n"
    "    # have = set(k for k, v in table.items()\n"
    "    #            if k in ids and not v.get(\"parse_error\") and v.get(\"status\") != \"FAIL\")\n"
    "    have = set(k for k, v in table.items() if k in ids and MA._usable(v))\n"))

DP_PATCHES.append((
    "point 3 -- spread robuste aux NaN",
    "        spread = [summary[f][\"mu_star_hi\"] - summary[f][\"mu_star_lo\"] for f in factors\n"
    "                  if np.isfinite(summary[f][\"mu_star_hi\"])]\n",
    "        " + MARKER + " original : ne filtrait que mu_star_hi, donc un\n"
    "        # mu_star_lo NaN propageait un NaN dans la mediane et annulait le mde.\n"
    "        # spread = [summary[f][\"mu_star_hi\"] - summary[f][\"mu_star_lo\"] for f in factors\n"
    "        #           if np.isfinite(summary[f][\"mu_star_hi\"])]\n"
    "        spread = [summary[f][\"mu_star_hi\"] - summary[f][\"mu_star_lo\"] for f in factors\n"
    "                  if np.isfinite(summary[f][\"mu_star_hi\"])\n"
    "                  and np.isfinite(summary[f][\"mu_star_lo\"])]\n"))

# ---- points 5 et 6 : nouvelles fonctions -------------------------------
DP_PATCHES.append((
    "points 5 et 6 -- fonctions d'analyse",
    "def _consolidate(per_qoi, factors):\n",
    MARKER + " begin -- points 5 et 6.\n"
    "CONFOUND_THRESHOLD = 0.55        # indice de confusion au-dela duquel on alerte\n"
    "\n"
    "\n"
    "def _confounding(per_qoi, factors):\n"
    "    \"\"\"\n"
    "    POINT 5 -- exploite le `mu` SIGNE, jusqu'ici calcule et affiche mais\n"
    "    absent de tout verdict.\n"
    "\n"
    "    Deux facteurs entrant de facon additive et antagoniste dans la meme loi\n"
    "    ne sont pas separement identifiables. La signature observable est : `mu`\n"
    "    de signes opposes, `mu*` de meme ordre, et cela de facon COHERENTE sur\n"
    "    les QoI. Indice par paire, moyenne sur les QoI de\n"
    "\n"
    "        min(mu*_i, mu*_j) / max(mu*_i, mu*_j)  x  max(0, -sign(mu_i) sign(mu_j))\n"
    "\n"
    "    1 = parfaitement antagonistes et de meme poids ; 0 = pas de signature.\n"
    "    \"\"\"\n"
    "    out = []\n"
    "    for a in range(len(factors)):\n"
    "        for b in range(a + 1, len(factors)):\n"
    "            fi, fj = factors[a], factors[b]\n"
    "            ws = []\n"
    "            for _q, blk in per_qoi.items():\n"
    "                si, sj = blk[\"summary\"].get(fi), blk[\"summary\"].get(fj)\n"
    "                if not si or not sj:\n"
    "                    continue\n"
    "                mi, mj = si[\"mu_star\"], sj[\"mu_star\"]\n"
    "                if not (np.isfinite(mi) and np.isfinite(mj)) or max(mi, mj) <= 0:\n"
    "                    continue\n"
    "                bal = min(mi, mj) / max(mi, mj)\n"
    "                opp = max(0.0, -((si[\"mu\"] / mi) * (sj[\"mu\"] / mj)))\n"
    "                ws.append(bal * opp)\n"
    "            if ws:\n"
    "                out.append({\"pair\": (fi, fj), \"index\": float(np.mean(ws)),\n"
    "                            \"n_qoi\": len(ws)})\n"
    "    out.sort(key=lambda r: -r[\"index\"])\n"
    "    return out\n"
    "\n"
    "\n"
    "def _structural_floor(design_rows, table, factors, qoi_keys, delta, gates):\n"
    "    \"\"\"\n"
    "    POINT 6 -- plancher de bruit STRUCTUREL, gratuit.\n"
    "\n"
    "    Un facteur peut etre inerte par construction : `eps_soft` ne pilote que\n"
    "    la cinetique de l'adoucissement, donc quand `s` vaut sa borne basse il\n"
    "    n'entre dans aucune equation. Tout effet elementaire mesure sur ces\n"
    "    paires est nul par construction : ce qui reste est du bruit, mesure sur\n"
    "    la campagne reelle -- maillage, mass scaling et extraction reels --\n"
    "    sans aucun run supplementaire.\n"
    "\n"
    "    `gates` est une liste de couples (gate, inert).\n"
    "    \"\"\"\n"
    "    res = []\n"
    "    by_traj = {}\n"
    "    for rid, d in design_rows.items():\n"
    "        by_traj.setdefault(int(d[\"traj\"]), {})[int(d[\"step\"])] = (rid, d)\n"
    "    for gate, inert in gates:\n"
    "        col = \"g_\" + gate\n"
    "        if gate not in factors or inert not in factors:\n"
    "            continue\n"
    "        try:\n"
    "            lo = min(float(d[col]) for d in design_rows.values())\n"
    "        except (KeyError, ValueError):\n"
    "            continue\n"
    "        pairs = []\n"
    "        for _t, steps in sorted(by_traj.items()):\n"
    "            seq = [steps[k] for k in sorted(steps)]\n"
    "            for k in range(1, len(seq)):\n"
    "                rid1, d1 = seq[k]\n"
    "                rid0, _d0 = seq[k - 1]\n"
    "                if d1.get(\"moved\") != inert:\n"
    "                    continue\n"
    "                if abs(float(d1[col]) - lo) > 1e-9:\n"
    "                    continue\n"
    "                pairs.append((rid0, rid1))\n"
    "        floors = {}\n"
    "        for q in qoi_keys:\n"
    "            vals = []\n"
    "            for a, b in pairs:\n"
    "                ra, rb = table.get(a), table.get(b)\n"
    "                if not (MA._usable(ra) and MA._usable(rb)):\n"
    "                    continue\n"
    "                y0, y1 = MA._num(ra, q), MA._num(rb, q)\n"
    "                if np.isfinite(y0) and np.isfinite(y1):\n"
    "                    vals.append(abs(y1 - y0) / delta)\n"
    "            if vals:\n"
    "                v = np.asarray(vals, dtype=float)\n"
    "                floors[q] = {\"n\": int(v.size), \"mean\": float(v.mean()),\n"
    "                             \"max\": float(v.max())}\n"
    "        res.append({\"gate\": gate, \"inert\": inert, \"lo\": lo,\n"
    "                    \"n_pairs\": len(pairs), \"floors\": floors})\n"
    "    return res\n"
    "\n"
    "\n"
    + MARKER + " end\n"
    "def _consolidate(per_qoi, factors):\n"))

# ---- section de rapport ------------------------------------------------
DP_PATCHES.append((
    "points 5 et 6 -- section du rapport",
    "    A(\"## 4. Reserves\")\n",
    "    " + MARKER + " begin -- sections ajoutees (points 5 et 6).\n"
    "    extra = extra or {}\n"
    "    conf = extra.get(\"confounding\") or []\n"
    "    hot = [c for c in conf if c[\"index\"] >= CONFOUND_THRESHOLD]\n"
    "    A(\"## 3bis. Identifiabilite et plancher structurel\")\n"
    "    A(\"\")\n"
    "    A(\"### Signature de confusion entre facteurs\")\n"
    "    A(\"\")\n"
    "    if hot:\n"
    "        A(\"> **%d paire(s) au-dessus du seuil %.2f.** `mu` de signes opposes et \"\n"
    "          \"`mu*` de meme ordre, de facon coherente sur les QoI : c'est ce que \"\n"
    "          \"produisent deux termes additifs antagonistes de la meme loi. Seule \"\n"
    "          \"leur difference est identifiable, et Morris attribue a chacun la \"\n"
    "          \"moitie d'un effet unique. Les `mu*` de ces facteurs ne doivent pas \"\n"
    "          \"etre lus separement.\" % (len(hot), CONFOUND_THRESHOLD))\n"
    "        A(\"\")\n"
    "    elif conf:\n"
    "        A(\"Aucune paire au-dessus du seuil %.2f : pas de signature de \"\n"
    "          \"confusion detectee.\" % CONFOUND_THRESHOLD)\n"
    "        A(\"\")\n"
    "    if conf:\n"
    "        A(\"| Paire | indice | QoI |\")\n"
    "        A(\"|---|---|---|\")\n"
    "        for c in conf[:8]:\n"
    "            A(\"| `%s` / `%s` | %s | %d |\"\n"
    "              % (c[\"pair\"][0], c[\"pair\"][1], _fmt(c[\"index\"], 3), c[\"n_qoi\"]))\n"
    "        A(\"\")\n"
    "    A(\"### Plancher de bruit structurel\")\n"
    "    A(\"\")\n"
    "    sf = extra.get(\"structural\") or []\n"
    "    if not sf:\n"
    "        A(\"_Aucune relation d'inertie declaree (`--gates`)._\")\n"
    "        A(\"\")\n"
    "    for blk in sf:\n"
    "        A(\"`%s` est inerte par construction quand `%s` = %s : %d paire(s) dans \"\n"
    "          \"le plan. Les effets elementaires mesures y sont nuls par \"\n"
    "          \"construction, donc ce qui reste est du bruit -- mesure sur la \"\n"
    "          \"campagne reelle, sans run supplementaire.\"\n"
    "          % (blk[\"inert\"], blk[\"gate\"], _fmt(blk[\"lo\"], 3), blk[\"n_pairs\"]))\n"
    "        A(\"\")\n"
    "        if not blk[\"floors\"]:\n"
    "            A(\"_Aucune paire exploitable : les runs concernes sont manquants._\")\n"
    "            A(\"\")\n"
    "            continue\n"
    "        A(\"| QoI | n | mu* nul observe | max | seuil fourni | Coherence |\")\n"
    "        A(\"|---|---|---|---|---|---|\")\n"
    "        for q in keys:\n"
    "            fl = blk[\"floors\"].get(q)\n"
    "            if not fl:\n"
    "                continue\n"
    "            supplied = per_qoi[q][\"mu_null\"] if q in per_qoi else None\n"
    "            if supplied is None:\n"
    "                verdict = \"-\"\n"
    "            elif supplied < fl[\"mean\"]:\n"
    "                verdict = \"**seuil trop bas**\"\n"
    "            else:\n"
    "                verdict = \"ok\"\n"
    "            A(\"| `%s` | %d | %s | %s | %s | %s |\"\n"
    "              % (q, fl[\"n\"], _fmt(fl[\"mean\"]), _fmt(fl[\"max\"]),\n"
    "                 _fmt(supplied), verdict))\n"
    "        A(\"\")\n"
    "        A(\"> Un `seuil trop bas` signifie que le plancher fourni est inferieur \"\n"
    "          \"au bruit reellement mesure sur des effets nuls par construction : \"\n"
    "          \"des facteurs sont alors retenus a tort.\")\n"
    "        A(\"\")\n"
    "    " + MARKER + " end\n"
    "    A(\"## 4. Reserves\")\n"))

DP_PATCHES.append((
    "points 5 et 6 -- signature de write_report",
    "def write_report(path, meta, cov, per_qoi, cons, factors, qoi_meta, args, figs):\n",
    "def write_report(path, meta, cov, per_qoi, cons, factors, qoi_meta, args, figs,\n"
    "                 extra=None):\n"
    "    " + MARKER + " `extra` porte les blocs des points 5 et 6 ; valeur par\n"
    "    # defaut None pour ne pas casser les appels existants.\n"))

DP_PATCHES.append((
    "points 5 et 6 -- option --gates",
    "    ap.add_argument(\"--bootstrap\", type=int, default=4000)\n",
    "    ap.add_argument(\"--bootstrap\", type=int, default=4000)\n"
    "    " + MARKER + " point 6 : relations d'inertie structurelle.\n"
    "    ap.add_argument(\"--gates\", default=\"s:eps_soft\",\n"
    "                    help=\"couples gate:inert separes par des virgules. \"\n"
    "                         \"`inert` est sans effet quand `gate` est a sa borne \"\n"
    "                         \"basse, donc les effets elementaires mesures la sont \"\n"
    "                         \"du bruit pur. Vide pour desactiver.\")\n"))

DP_PATCHES.append((
    "points 5 et 6 -- calcul et passage a write_report",
    "    md_path = os.path.join(args.out_dir, \"SCREENING_REPORT.md\")\n"
    "    retained, marginal, frozen = write_report(md_path, meta, cov, per_qoi, cons,\n"
    "                                    factors, qoi_meta, args, figs)\n",
    "    " + MARKER + " begin -- points 5 et 6.\n"
    "    gates = []\n"
    "    for spec in (args.gates or \"\").split(\",\"):\n"
    "        if \":\" in spec:\n"
    "            g, i = [x.strip() for x in spec.split(\":\", 1)]\n"
    "            if g and i:\n"
    "                gates.append((g, i))\n"
    "    extra = {\n"
    "        \"confounding\": _confounding(per_qoi, factors),\n"
    "        \"structural\": _structural_floor(design_rows, table, factors,\n"
    "                                        [q for q, _, _ in qoi_meta],\n"
    "                                        meta[\"delta\"], gates),\n"
    "    }\n"
    "    for c in extra[\"confounding\"][:1]:\n"
    "        if c[\"index\"] >= CONFOUND_THRESHOLD:\n"
    "            print(\"  ATTENTION confusion probable : %s / %s (indice %.3f)\"\n"
    "                  % (c[\"pair\"][0], c[\"pair\"][1], c[\"index\"]))\n"
    "    " + MARKER + " end\n"
    "    md_path = os.path.join(args.out_dir, \"SCREENING_REPORT.md\")\n"
    "    retained, marginal, frozen = write_report(md_path, meta, cov, per_qoi, cons,\n"
    "                                    factors, qoi_meta, args, figs, extra=extra)\n"))


TARGETS = [("morris_analysis.py", MA_PATCHES), ("dp_screening.py", DP_PATCHES)]


# ----------------------------------------------------------------------

def _read(path):
    f = open(path, "rb")
    try:
        raw = f.read()
    finally:
        f.close()
    return raw.decode("utf-8", "replace").replace("\r\n", "\n"), (b"\r\n" in raw)


def _write(path, text, crlf):
    out = text.replace("\n", "\r\n") if crlf else text
    f = open(path, "wb")
    try:
        f.write(out.encode("utf-8"))
    finally:
        f.close()


def apply_to(path, patches, check_only):
    text, crlf = _read(path)
    print("\nCible : %s  (%s)" % (path, "CRLF" if crlf else "LF"))
    if MARKER in text:
        print("  deja patche -- rien a faire. IDEMPOTENT OK.")
        return True

    missing = []
    for name, anchor, replacement in patches:
        n = text.count(anchor)
        if n == 0:
            missing.append(name)
            print("  [ABSENT]  %s" % name)
            continue
        if n > 1:
            print("  [ERREUR]  ancre non unique (%d) : %s" % (n, name))
            return False
        text = text.replace(anchor, replacement, 1)
        print("  [ok]      %s" % name)

    if missing:
        print("  -> %d ancre(s) absente(s), rien n'a ete ecrit pour ce fichier."
              % len(missing))
        return False

    try:
        ast.parse(text)
    except SyntaxError as e:
        print("  [ERREUR]  la source patchee ne parse pas : %s" % e)
        return False
    print("  [ok]      validation ast.parse")

    if check_only:
        print("  --check : rien ecrit.")
        return True

    backup = path + ".bak_screening"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    _write(path, text, crlf)
    print("  [ok]      ecrit (%s preserve), sauvegarde -> %s"
          % ("CRLF" if crlf else "LF", os.path.basename(backup)))
    return True


def main(argv):
    check_only = "--check" in argv
    args = [a for a in argv[1:] if not a.startswith("--")]
    root = args[0] if args else "."

    ok = True
    for fname, patches in TARGETS:
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            print("ERREUR: %s introuvable." % path)
            return 2
        ok = apply_to(path, patches, check_only) and ok

    print("")
    if ok:
        print("Termine. Relancer le script pour confirmer l'idempotence.")
        print("")
        print("  python3 dp_screening.py <resultats> --design <plan> \\")
        print("          --noise-floor noise_dp.json --gates s:eps_soft")
    else:
        print("Termine avec des erreurs -- voir ci-dessus.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))