#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch_sampling_exclusive.py
===========================

Rend MUTUELLEMENT EXCLUSIFS l'adoucissement intrinseque (`s`) et le
durcissement de Voce (`q`) dans la campagne unifiee Drucker-Prager.

Pourquoi
--------
Les deux termes sont additifs et de signes opposes dans le meme crochet de
`gsell_jonas_table` :

    sigma_y0 * [ 1 - s*(1 - exp(-eps/eps_soft)) + q*(1 - exp(-b*eps)) ]

Avec `b_voce = 8` gele (echelle 0.125) et `eps_soft` jusqu'a 0.12, les deux
exponentielles se confondent : deux couples (q, s) de meme difference q - s
produisent des tables separees par moins de 0.5 % de sigma_y0. Seule la
difference est identifiable, et 47 des 100 points du plan actuel ont q > 0
ET s > 0 -- une combinaison qu'aucune des deux campagnes historiques ne
permet (`C3_FROZEN["soft_drop_MPa"] = 0`, `C4_FROZEN["b_voce"] = 0`).

Ce que fait le patch
--------------------
1. `CDP_FACTORS` : les trois facteurs `q`, `s`, `eps_soft` sont remplaces
   par DEUX facteurs

       w      in [-0.35, +0.35]   amplitude post-seuil signee / sigma_y0
                                  w < 0 -> adoucissement (vitreux)
                                  w > 0 -> Voce (semi-cristallin)
       eps_c  in [0.02, 0.12]     echelle de deformation de la branche active

   La grille Morris a p = 4 donne w dans {-0.35, -0.1167, +0.1167, +0.35} :
   w = 0 n'est jamais echantillonne, donc chaque point du plan appartient
   sans ambiguite a une branche et a une seule.

   d passe de 9 a 8 facteurs : r*(d+1) = 90 runs au lieu de 100.

2. `_derive_cdp` aiguille vers la branche active. `b_voce` cesse d'etre gele :
   il vaut 1/eps_c sur la branche Voce (soit b dans [8.3, 50], qui encadre
   l'ancienne valeur 8.0), et l'echelle d'adoucissement vaut eps_c sur la
   branche vitreuse. Les deux ne sont donc jamais actives ensemble.

3. GARDE-FOU DUR dans `_apply_dp` : toute configuration portant a la fois
   `Q > 0` et `soft_drop > 0` leve une exception. `_apply_dp` est le point de
   passage OBLIGE de toutes les campagnes (C3, C4, CDP) et des deux cotes
   (CPython pour le plan, noyau Abaqus pour l'execution) : aucun design ne
   peut atteindre Abaqus en violant l'exclusivite.

   Le garde-fou est declaratif : `EXCLUSIVE_PARAM_PAIRS` liste les couples
   interdits, il suffit d'y ajouter une ligne pour en declarer d'autres.

Contraintes respectees
----------------------
`sampling.py` est importe par le noyau Abaqus : pas de f-string, numpy
uniquement, compatible Python 2. Fins de ligne CRLF preservees, code
d'origine commente et non supprime, ancres exactes, validation `ast.parse`,
idempotent.

Usage
-----
    python patch_sampling_exclusive.py [chemin/vers/sampling.py]
    python patch_sampling_exclusive.py --check
"""

import ast
import os
import shutil
import sys

MARKER = "# [PATCH:exclusive-post-yield]"

DEFAULT_PATH = os.path.join(
    "ScratchSimulation", "AbaqusModel", "Configuration", "sampling.py")


# ----------------------------------------------------------------------
# 1. garde-fou dans _apply_dp (couvre TOUTES les campagnes)
# ----------------------------------------------------------------------

ANCHOR_APPLY = (
    "def _apply_dp(cfg, p):\n"
    "    cfg.material.rho = p[\"rho\"]\n"
)

REPLACE_APPLY = (
    MARKER + " begin -- exclusivite structurelle des termes post-seuil.\n"
    "#\n"
    "# Deux mecanismes post-seuil additifs et antagonistes ne sont pas\n"
    "# separement identifiables : avec b_voce = 8 (echelle 0.125) et eps_soft\n"
    "# jusqu'a 0.12, deux couples (q, s) de meme difference q - s donnent des\n"
    "# tables d'ecrouissage separees par moins de 0.5 % de sigma_y0. Aucune des\n"
    "# deux campagnes historiques ne les autorise ensemble (C3 gele\n"
    "# soft_drop_MPa = 0, C4 gele b_voce = 0) et aucune famille calibree ne les\n"
    "# porte simultanement. La regle est donc une INVARIANTE du modele, pas une\n"
    "# preference de campagne, et elle est verifiee au point de passage commun.\n"
    "EXCLUSIVE_PARAM_PAIRS = (\n"
    "    (\"Q\", \"soft_drop\"),\n"
    ")\n"
    "\n"
    "\n"
    "def _assert_exclusive(p, campaign=\"?\"):\n"
    "    \"\"\"Leve si deux parametres declares mutuellement exclusifs sont actifs.\"\"\"\n"
    "    for a, b in EXCLUSIVE_PARAM_PAIRS:\n"
    "        va = abs(float(p.get(a, 0.0) or 0.0))\n"
    "        vb = abs(float(p.get(b, 0.0) or 0.0))\n"
    "        if va > 1e-12 and vb > 1e-12:\n"
    "            raise ValueError(\n"
    "                \"Campaign '%s': '%s' (=%g) and '%s' (=%g) are mutually \"\n"
    "                \"exclusive but both are active. They are antagonistic \"\n"
    "                \"additive terms of the same hardening law and are not \"\n"
    "                \"separately identifiable; use the signed factor 'w' \"\n"
    "                \"instead of independent 'q' and 's'.\"\n"
    "                % (campaign, a, va, b, vb))\n"
    "    return p\n"
    "\n"
    "\n"
    + MARKER + " end\n"
    "def _apply_dp(cfg, p):\n"
    "    _assert_exclusive(p)\n"
    "    cfg.material.rho = p[\"rho\"]\n"
)


# ----------------------------------------------------------------------
# 2. facteurs de la campagne unifiee
# ----------------------------------------------------------------------

ANCHOR_FACTORS = (
    "    Factor(\"q\",        0.0, 0.35, \"lin\", \"-\",   \"Voce amplitude Q / sigma_y0 (0 = glassy limit)\"),\n"
    "    Factor(\"s\",        0.0, 0.35, \"lin\", \"-\",   \"intrinsic softening drop / sigma_y0 (0 = semicrystalline limit)\"),\n"
    "    Factor(\"eps_soft\", 0.02, 0.12, \"lin\", \"-\",  \"softening strain scale\"),\n"
)

REPLACE_FACTORS = (
    "    " + MARKER + " q / s / eps_soft remplaces par (w, eps_c).\n"
    "    # Originaux -- q et s sont antagonistes et confondus dans le crochet :\n"
    "    # Factor(\"q\",        0.0, 0.35, \"lin\", \"-\",   \"Voce amplitude Q / sigma_y0 (0 = glassy limit)\"),\n"
    "    # Factor(\"s\",        0.0, 0.35, \"lin\", \"-\",   \"intrinsic softening drop / sigma_y0 (0 = semicrystalline limit)\"),\n"
    "    # Factor(\"eps_soft\", 0.02, 0.12, \"lin\", \"-\",  \"softening strain scale\"),\n"
    "    Factor(\"w\",     -0.35, 0.35, \"lin\", \"-\",  \"signed post-yield amplitude / sigma_y0: \"\n"
    "                                             \"w < 0 = intrinsic softening (glassy), \"\n"
    "                                             \"w > 0 = Voce hardening (semicrystalline)\"),\n"
    "    Factor(\"eps_c\", 0.02, 0.12, \"lin\", \"-\",  \"strain scale of whichever post-yield \"\n"
    "                                             \"branch is active (eps_soft if w < 0, \"\n"
    "                                             \"1/b if w > 0)\"),\n"
)


# ----------------------------------------------------------------------
# 3. derivation de la campagne unifiee
# ----------------------------------------------------------------------

ANCHOR_DERIVE = (
    "def _derive_cdp(g, cfg):\n"
    "    return _derive_dp(g, cfg, CDP_FROZEN, with_softening=True)\n"
)

REPLACE_DERIVE = (
    MARKER + " begin -- aiguillage exclusif des deux branches post-seuil.\n"
    "def _split_post_yield(w, eps_c, sy):\n"
    "    \"\"\"\n"
    "    Traduit (w, eps_c) en la paire de termes post-seuil de\n"
    "    gsell_jonas_table, avec UN SEUL des deux actif.\n"
    "\n"
    "        w < 0 : branche vitreuse    -> soft_drop = |w| * sy, eps_soft = eps_c,\n"
    "                                       Q = 0\n"
    "        w > 0 : branche semi-crist. -> Q = w * sy, b = 1 / eps_c,\n"
    "                                       soft_drop = 0\n"
    "        w = 0 : ni l'un ni l'autre (parfaitement plastique + orientation)\n"
    "\n"
    "    Sur la grille Morris a p = 4 niveaux, w vaut -0.35, -0.1167, +0.1167\n"
    "    ou +0.35 : w = 0 n'est jamais echantillonne, donc chaque point du plan\n"
    "    appartient a une branche et a une seule.\n"
    "    \"\"\"\n"
    "    w = float(w)\n"
    "    eps_c = float(eps_c)\n"
    "    if eps_c <= 0.0:\n"
    "        raise ValueError(\"eps_c must be strictly positive (got %g)\" % eps_c)\n"
    "    if w < 0.0:\n"
    "        return {\"Q\": 0.0, \"b\": 1.0 / eps_c,\n"
    "                \"soft_drop\": -w * sy, \"eps_soft\": eps_c, \"branch\": -1.0}\n"
    "    if w > 0.0:\n"
    "        return {\"Q\": w * sy, \"b\": 1.0 / eps_c,\n"
    "                \"soft_drop\": 0.0, \"eps_soft\": eps_c, \"branch\": 1.0}\n"
    "    return {\"Q\": 0.0, \"b\": 1.0 / eps_c,\n"
    "            \"soft_drop\": 0.0, \"eps_soft\": eps_c, \"branch\": 0.0}\n"
    "\n"
    "\n"
    "def _derive_cdp(g, cfg):\n"
    "    " + MARKER + " original :\n"
    "    # return _derive_dp(g, cfg, CDP_FROZEN, with_softening=True)\n"
    "    sy = float(CDP_FROZEN[\"sigma_y0_ref_MPa\"])\n"
    "    br = _split_post_yield(g[\"w\"], g[\"eps_c\"], sy)\n"
    "    inner = dict(g)\n"
    "    inner.pop(\"w\", None)\n"
    "    inner.pop(\"eps_c\", None)\n"
    "    inner[\"s\"] = br[\"soft_drop\"] / sy\n"
    "    inner[\"eps_soft\"] = br[\"eps_soft\"]\n"
    "    out = _derive_dp(inner, cfg, CDP_FROZEN, with_softening=True)\n"
    "    out[\"Q\"] = br[\"Q\"]\n"
    "    out[\"b\"] = br[\"b\"]\n"
    "    out[\"w\"] = float(g[\"w\"])\n"
    "    out[\"eps_c\"] = float(g[\"eps_c\"])\n"
    "    out[\"branch\"] = br[\"branch\"]\n"
    "    return _assert_exclusive(out, \"CDP_drucker_prager_unified\")\n"
    + MARKER + " end\n"
)


# ----------------------------------------------------------------------
# 4. b_voce n'est plus gele + note de campagne
# ----------------------------------------------------------------------

ANCHOR_FROZEN = (
    "CDP_FROZEN = {\n"
    "    \"sigma_y0_ref_MPa\": 50.0,\n"
    "    \"nu\": 0.39,\n"
    "    \"rho\": 1.10e-9,\n"
    "    \"b_voce\": 8.0,\n"
)

REPLACE_FROZEN = (
    "CDP_FROZEN = {\n"
    "    \"sigma_y0_ref_MPa\": 50.0,\n"
    "    \"nu\": 0.39,\n"
    "    \"rho\": 1.10e-9,\n"
    "    " + MARKER + " b_voce n'est plus gele : il vaut 1/eps_c sur la\n"
    "    # branche Voce. La valeur reste declaree pour la tracabilite et pour\n"
    "    # les campagnes qui passent encore par _derive_dp sans aiguillage.\n"
    "    \"b_voce\": 8.0,\n"
    "    \"b_voce_note\": \"superseded by 1/eps_c in CDP; kept for traceability\",\n"
)

ANCHOR_NOTES = (
    "    notes=\"s=0 is the semicrystalline corner, q=0 the glassy corner, beta=1 deg \"\n"
    "          \"with K=1 the J2 corner. Softening and Voce hardening coexist in the \"\n"
    "          \"interior of the box, which is a superset of both calibrations.\",\n"
)

REPLACE_NOTES = (
    "    " + MARKER + " note reecrite : la coexistence est desormais interdite.\n"
    "    # notes=\"s=0 is the semicrystalline corner, q=0 the glassy corner, ...\n"
    "    notes=\"w < 0 is the glassy branch (intrinsic softening), w > 0 the \"\n"
    "          \"semicrystalline branch (Voce hardening); the two are mutually \"\n"
    "          \"exclusive by construction and w = 0 is never sampled on the p=4 \"\n"
    "          \"grid. beta=1 deg with K=1 remains the J2 corner. Orientation \"\n"
    "          \"hardening h is shared by BOTH branches -- it is present in C3 and \"\n"
    "          \"C4 alike and both calibrated glassy families carry h > 0 together \"\n"
    "          \"with softening, so h is never exclusive with w.\",\n"
)


PATCHES = [
    ("garde-fou _apply_dp",        ANCHOR_APPLY,   REPLACE_APPLY),
    ("CDP_FACTORS -> w, eps_c",    ANCHOR_FACTORS, REPLACE_FACTORS),
    ("_derive_cdp aiguillage",     ANCHOR_DERIVE,  REPLACE_DERIVE),
    ("CDP_FROZEN b_voce",          ANCHOR_FROZEN,  REPLACE_FROZEN),
    ("note de campagne",           ANCHOR_NOTES,   REPLACE_NOTES),
]


# ----------------------------------------------------------------------

def _read(path):
    f = open(path, "rb")
    try:
        raw = f.read()
    finally:
        f.close()
    crlf = b"\r\n" in raw
    return raw.decode("utf-8", "replace").replace("\r\n", "\n"), crlf


def _write(path, text, crlf):
    out = text.replace("\n", "\r\n") if crlf else text
    f = open(path, "wb")
    try:
        f.write(out.encode("utf-8"))
    finally:
        f.close()


def main(argv):
    check_only = "--check" in argv
    args = [a for a in argv[1:] if not a.startswith("--")]
    path = args[0] if args else DEFAULT_PATH

    if not os.path.exists(path):
        print("ERREUR: %s introuvable. Donner le chemin explicitement." % path)
        return 2

    text, crlf = _read(path)
    print("Cible : %s  (%d octets, %s)" % (path, len(text), "CRLF" if crlf else "LF"))

    if MARKER in text:
        print("Deja patche (marqueur present) -- rien a faire. IDEMPOTENT OK.")
        return 0

    applied, missing = [], []
    for name, anchor, replacement in PATCHES:
        n = text.count(anchor)
        if n == 0:
            missing.append(name)
            continue
        if n > 1:
            print("ERREUR: ancre '%s' non unique (%d occurrences)." % (name, n))
            return 3
        text = text.replace(anchor, replacement, 1)
        applied.append(name)

    for name in applied:
        print("  [ok]      %s" % name)
    for name in missing:
        print("  [ABSENT]  %s" % name)
    if missing:
        print("\nERREUR: %d/%d ancres absentes, rien n'a ete ecrit."
              % (len(missing), len(PATCHES)))
        return 4

    try:
        ast.parse(text)
    except SyntaxError as e:
        print("\nERREUR: la source patchee ne parse pas : %s" % e)
        return 5
    print("  [ok]      validation ast.parse")

    import re as _re
    if _re.search(r"(?<![\w.])f[\"']", text):
        print("  [!]       attention : f-string detectee (Abaqus est en Python 2)")
    else:
        print("  [ok]      aucune f-string (compatible noyau Abaqus Python 2)")

    if check_only:
        print("\n--check : aucun fichier ecrit.")
        return 0

    backup = path + ".bak_exclusive"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
        print("  [ok]      sauvegarde -> %s" % backup)

    _write(path, text, crlf)
    print("\nPatche %s (%s preserve)." % (path, "CRLF" if crlf else "LF"))
    print("")
    print("SUITE : le plan actuel (9 facteurs, q et s independants) est caduc.")
    print("        Regenerer :  python3 generate_design.py glassy_pc --r 10")
    print("        -> 8 facteurs, 90 runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))