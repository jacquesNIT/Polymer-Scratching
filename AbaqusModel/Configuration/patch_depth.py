# -*- coding: utf-8 -*-
"""
patch_depth_plateau.py

Corrige le sous-tir de profondeur en mode PROGRESSIVE (Scratch_Config, base.py).

Cause : la table d'amplitude de profondeur progressive a son pic (t2, 1.0) en point
INTERIEUR (immediatement suivi de la decharge). Le parametre 'smooth' d'Abaqus
arrondit ce sommet vers le bas -> la profondeur nominale n'est jamais atteinte, et
le manque ~ smooth_window/scratch_time depend donc du scratch_time. C'est l'origine
de la fausse dependance de RF2 au scratch_time (RF2(profondeur) se superpose).

Fix : donner a la rampe progressive un court PLATEAU plat au sommet (largeur
depth_hold = depth_hold_frac * scratch_time, comme le mode CONSTANT en a deja un).
L'interieur plat = depth_hold*(1 - 2*smooth) > 0 pour tout smooth < 0.5, donc la
valeur 1.0 est atteinte exactement ; profondeur nominale garantie ET profil normalise
invariant en scratch_time. t2/t3 inchanges -> pas de re-timing du step.

Convention : exact-string / regex-ancre, idempotent, CRLF preserve, code original
commente (non supprime), validation AST + round-trip. base.py reste Abaqus-free.
"""

import os
import re
import ast
import sys

PATH = os.environ.get("BASE_PY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "base.py"))


def read(path):
    with open(path, "rb") as f:
        raw = f.read()
    eol = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode("utf-8"), eol


def write(path, text):
    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))


def apply_edit(text, name, marker, pattern, repl, flags=re.MULTILINE):
    """Idempotent regex edit. marker present -> skip. Else require exactly one match."""
    if marker in text:
        print("  [skip] %-28s (deja applique)" % name)
        return text, False
    rx = re.compile(pattern, flags)
    matches = list(rx.finditer(text))
    if len(matches) != 1:
        raise RuntimeError("%s : attendu 1 correspondance, trouve %d" % (name, len(matches)))
    new = rx.sub(repl, text, count=1)
    if new == text:
        raise RuntimeError("%s : substitution sans effet" % name)
    print("  [ok]   %-28s applique" % name)
    return new, True


def main():
    text, eol = read(PATH)
    print("Fichier : %s  (%s)" % (PATH, "CRLF" if eol == "\r\n" else "LF"))
    # On travaille en \n interne puis on re-serialise avec l'EOL d'origine.
    text_lf = text.replace("\r\n", "\n")

    # -- E1 : parametre depth_hold_frac dans la signature __init__ -----------------
    text_lf, _ = apply_edit(
        text_lf, "E1 signature",
        marker="depth_hold_frac=",
        pattern=r"^(?P<ind>[ ]+)amplitude_smoothing=0\.25 \):(?P<cmt>.*)$",
        repl=(lambda m: "{ind}amplitude_smoothing=0.25,{cmt}\n"
                        "{ind}depth_hold_frac=0.05 ):"
                        "   # [-] PROGRESSIVE: plateau plat au sommet, fraction de scratch_time "
                        "(garantit la profondeur nominale malgre le lissage du pic ; cf depth_amplitude())."
              .format(ind=m.group("ind"), cmt=m.group("cmt"))),
    )

    # -- E2 : validation de depth_hold_frac ---------------------------------------
    text_lf, _ = apply_edit(
        text_lf, "E2 validation",
        marker="depth_hold_frac must be in",
        pattern=(r'^(?P<blk>[ ]+if amplitude_smoothing is not None and not \(0\.0 <= amplitude_smoothing <= 0\.5\):\n'
                 r'[ ]+raise ValueError\("amplitude_smoothing must be in \[0, 0\.5\] or None, got %s" % amplitude_smoothing\))$'),
        repl=(lambda m: m.group("blk") + "\n\n"
              "        if not (0.0 <= depth_hold_frac < 0.5):\n"
              "            raise ValueError(\"depth_hold_frac must be in [0, 0.5), got %s\" % depth_hold_frac)"),
    )

    # -- E3 : stockage de l'attribut ----------------------------------------------
    text_lf, _ = apply_edit(
        text_lf, "E3 attribut",
        marker="self.depth_hold_frac =",
        pattern=r"^(?P<line>[ ]+self\.amplitude_smoothing = amplitude_smoothing)$",
        repl=(lambda m: m.group("line") + "\n"
              "        self.depth_hold_frac = depth_hold_frac"),
    )

    # -- E4 : propriete depth_hold (juste apres t_scratch_end) ---------------------
    text_lf, _ = apply_edit(
        text_lf, "E4 propriete depth_hold",
        marker="def depth_hold(self)",
        pattern=(r"^(?P<blk>    @property\n"
                 r"    def t_scratch_end\(self\): # End of scratching phase \[s\]\.\n"
                 r"        return self\.t_indent_end \+ self\.scratch_time)$"),
        repl=(lambda m: m.group("blk") + "\n\n"
              "    @property\n"
              "    def depth_hold(self): # [s] Plateau tenu a la profondeur pic avant decharge (PROGRESSIVE). Fraction de scratch_time.\n"
              "        return self.depth_hold_frac * self.scratch_time"),
    )

    # -- E5 : plateau dans depth_amplitude() (branche PROGRESSIVE) -----------------
    # On capture le bloc exact (unique grace au 'lift_value' de la branche recovery)
    # et on commente l'original avant d'inserer la version avec plateau.
    e5_pattern = (
        r"^(?P<ind>        )if self\.depth_mode == self\.PROGRESSIVE:\n"
        r"            if not self\.has_recovery_step:\n"
        r"                return \(\(0\.0, 0\.0\),\(t2,  1\.0\),\(t3,  0\.0\)\)\n"
        r"            else:\n"
        r"                return \(\(0\.0,  0\.0\),\(t2,   1\.0\),\(t3,   lift_value\),\(t4,   lift_value\)\)$"
    )

    def e5_repl(m):
        ind = m.group("ind")  # 8 espaces
        lines = [
            ind + "if self.depth_mode == self.PROGRESSIVE:",
            ind + "    # --- PLATEAU FIX (sous-tir par arrondi du pic interieur) -------------",
            ind + "    # Original (pic (t2,1.0) interieur -> lisse VERS LE BAS, profondeur nominale",
            ind + "    # jamais atteinte ; manque ~ smooth_window/scratch_time -> fausse dependance",
            ind + "    # de RF2 au scratch_time) :",
            ind + "    #     if not self.has_recovery_step:",
            ind + "    #         return ((0.0, 0.0),(t2,  1.0),(t3,  0.0))",
            ind + "    #     else:",
            ind + "    #         return ((0.0,  0.0),(t2,   1.0),(t3,   lift_value),(t4,   lift_value))",
            ind + "    # Fix : sommet plat de largeur depth_hold (scale avec scratch_time). L'interieur",
            ind + "    # plat = depth_hold*(1 - 2*smooth) > 0 est atteint exactement pour tout smooth<0.5,",
            ind + "    # donc profondeur nominale garantie et profil normalise invariant. t2/t3 inchanges.",
            ind + "    t2h = t2 - self.depth_hold",
            ind + "    if not self.has_recovery_step:",
            ind + "        return ((0.0, 0.0),(t2h,  1.0),(t2,  1.0),(t3,  0.0))",
            ind + "    else:",
            ind + "        return ((0.0,  0.0),(t2h,  1.0),(t2,  1.0),(t3,  lift_value),(t4,  lift_value))",
        ]
        return "\n".join(lines)

    text_lf, _ = apply_edit(
        text_lf, "E5 depth_amplitude plateau",
        marker="PLATEAU FIX",
        pattern=e5_pattern,
        repl=e5_repl,
    )

    # -- Validation AST avant ecriture --------------------------------------------
    ast.parse(text_lf)
    print("AST : OK")

    out = text_lf.replace("\n", eol) if eol == "\r\n" else text_lf
    write(PATH, out)
    # Re-verif EOL preserve
    raw = open(PATH, "rb").read()
    assert (b"\r\n" in raw) == (eol == "\r\n"), "EOL non preserve"
    print("Ecrit : EOL preserve (%s)" % ("CRLF" if eol == "\r\n" else "LF"))


if __name__ == "__main__":
    main()