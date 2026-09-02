"""Evolution du SCOF le long du scratch.

    python scof.py <dossier_ou_csv> [options]

Options
    --csv <fichier>     ecrit la ou les courbes echantillonnees (z, F_n, F_t, SCOF)
    --png <fichier>     enregistre la figure
    --smooth <mm>       longueur de lissage gaussien du trace, en mm (defaut 0.0)
    --no-show           n'ouvre pas de fenetre (utile sur le cluster)
    --no-plot           calcul et export CSV seulement

results_values.py ne rend qu'un SCOF moyen sur la bande [10 %, 90 %] du pic de
force normale. Ce module reutilise EXACTEMENT les memes estimateurs -- ils sont
importes depuis results_values, pas recopies -- mais conserve la serie point par
point et la reporte sur l'abscisse z du sillon. La moyenne de la courbe restreinte
a la bande retenue redonne donc SCOF_mean au bit pres.

Convention d'abscisse identique a force_values : sur la fenetre ACTIVE, z avance
lineairement de 0 a scratch_length. Le facteur 2 du demi-modele s'annule dans le
rapport RF3/RF2 ; il n'est applique qu'aux forces tracees.
"""

import argparse
import csv as _csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


# =====================================================================
#  Import des estimateurs de results_values (source unique de verite)
# =====================================================================
def _load_results_values():
    """Importe results_values, d'abord par sys.path puis par chemin explicite.

    Aucune logique n'est recopiee : masques temporels, choix RF2/CFN2 et bornes
    de bande viennent tous du module d'origine. Toute correction apportee la-bas
    se propage ici sans intervention.
    """
    try:
        import results_values as rv  # noqa: F401
        return rv
    except ImportError:
        pass

    here = Path(__file__).resolve().parent
    candidates = [
        here / "results_values.py",
        Path.cwd() / "results_values.py",
        here.parent / "results_values.py",
    ]
    for cand in candidates:
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("results_values", str(cand))
            mod = importlib.util.module_from_spec(spec)
            sys.modules["results_values"] = mod
            spec.loader.exec_module(mod)
            return mod

    raise ImportError(
        "results_values.py introuvable. Placez scof.py a cote de results_values.py "
        "ou ajoutez son dossier a PYTHONPATH."
    )


rv = _load_results_values()


# Constantes de bande, relues depuis results_values pour rester synchronisees.
SCOF_LO_FRAC = rv.SCOF_LO_FRAC
SCOF_HI_FRAC = rv.SCOF_HI_FRAC


# =====================================================================
#  Geometrie de l'indenteur
# =====================================================================
def sphere_cone_transition_depth(metadata):
    """Profondeur h_t (mm) a laquelle le contact quitte la calotte spherique.

    Pour un cone de demi-angle theta compte depuis l'axe, raccorde tangentiellement
    a une sphere de rayon R, la tangence est atteinte a h_t = R (1 - sin theta).
    Rockwell C (R = 0.2 mm, theta = 60 deg) : h_t = 26.8 um.
    Retourne None si la geometrie n'est pas lisible dans l'entete.
    """
    R = metadata.get("tip_radius")
    theta = metadata.get("cone_angle")
    if R is None or theta is None:
        return None
    try:
        R = float(R)
        theta = float(theta)
    except (TypeError, ValueError):
        return None
    if R <= 0.0:
        return None  # pyramide equivalente : pas de calotte, pas de transition
    return R * (1.0 - np.sin(np.radians(theta)))


def sphere_cone_transition_z(metadata, scratch_length):
    """Abscisse z (mm) de la transition sphere -> cone, mode progressif seul.

    En depth_mode=progressive la consigne de profondeur monte lineairement avec
    la course : z_t = L * h_t / depth_max. En depth_mode=constant la profondeur
    est atteinte avant le scratch, la transition n'a pas d'abscisse le long du
    sillon et la fonction retourne None.
    """
    if str(metadata.get("depth_mode", "")).lower().startswith("prog") is False:
        return None
    h_t = sphere_cone_transition_depth(metadata)
    depth = metadata.get("scratch_depth")
    if h_t is None or depth is None:
        return None
    depth = abs(float(depth))
    if depth <= 0.0 or h_t >= depth:
        return None
    return scratch_length * h_t / depth


# =====================================================================
#  Coeur du calcul
# =====================================================================
def _nan_gaussian(y, sigma):
    """Lissage gaussien tolerant aux NaN (normalisation par le poids valide)."""
    from scipy.ndimage import gaussian_filter1d

    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if sigma <= 0.0 or not ok.any():
        return y.copy()
    filled = np.where(ok, y, 0.0)
    num = gaussian_filter1d(filled, sigma=sigma, mode="nearest")
    den = gaussian_filter1d(ok.astype(float), sigma=sigma, mode="nearest")
    out = np.full_like(y, np.nan)
    good = den > 1e-12
    out[good] = num[good] / den[good]
    out[~ok] = np.nan
    return out


def scof_curve(timeseries, metadata, smooth_mm=0.0):
    """SCOF point par point le long du sillon.

    Retourne un dict :
        z         (n,) abscisse le long du scratch [mm]
        scof      (n,) |RF3| / |RF2|, NaN la ou RF2 est nul
        scof_s    (n,) meme serie lissee si smooth_mm > 0, sinon copie de scof
        F_n, F_t  (n,) forces du modele COMPLET (facteur 2) [N]
        in_load   (n,) bool, echantillon anterieur au sommet de consigne
        in_band   (n,) bool, echantillon retenu par la moyenne de results_values
        z_t       transition sphere -> cone [mm] ou None
        SCOF_mean, SCOF_std, SCOF_n  moyenne de bande, identique a scof_values
    """
    rf3 = timeseries.get("RF3")
    rf2, rf2_src = rv._normal_force_series(timeseries, metadata)
    if rf2 is None or rf3 is None:
        return None

    active = rv._active_mask(timeseries, metadata)
    if len(active) != len(rf2) or not active.any():
        return None

    idx = np.flatnonzero(active)

    # scratch_length est lu dans l'entete quand il y est : un run a 3 mm est
    # trace sur 3 mm sans toucher a la constante de results_values.
    L = metadata.get("scratch_length")
    try:
        L = float(L)
    except (TypeError, ValueError):
        L = float(rv.SCRATCH_LENGTH)
    if not np.isfinite(L) or L <= 0.0:
        L = float(rv.SCRATCH_LENGTH)

    z = np.linspace(0.0, L, len(idx))

    load = rv._loading_mask(timeseries, metadata)
    in_load = load[idx] if len(load) == len(active) else np.ones(len(idx), dtype=bool)
    if not in_load.any():
        in_load = np.ones(len(idx), dtype=bool)

    rf2_abs = np.abs(np.asarray(rf2, dtype=float))[idx]
    rf3_abs = np.abs(np.asarray(rf3, dtype=float))[idx]

    peak = float(np.nanmax(rf2_abs[in_load]))
    if not np.isfinite(peak) or peak <= 0.0:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        scof = np.where(rf2_abs > 0.0, rf3_abs / rf2_abs, np.nan)
    scof[~np.isfinite(scof)] = np.nan

    in_band = (
        in_load
        & (rf2_abs >= SCOF_LO_FRAC * peak)
        & (rf2_abs <= SCOF_HI_FRAC * peak)
        & np.isfinite(scof)
    )

    sigma = 0.0
    if smooth_mm and len(z) > 2:
        dz = float(np.median(np.diff(z)))
        if np.isfinite(dz) and dz > 0.0:
            sigma = max(float(smooth_mm) / dz, 1e-6)
    scof_s = _nan_gaussian(scof, sigma) if sigma > 0.0 else scof.copy()

    out = {
        "z": z,
        "scof": scof,
        "scof_smooth": scof_s,
        "F_n": 2.0 * rf2_abs,
        "F_t": 2.0 * rf3_abs,
        "in_load": in_load,
        "in_band": in_band,
        "scratch_length": L,
        "rf2_source": rf2_src,
        "z_t": sphere_cone_transition_z(metadata, L),
        "smooth_mm": float(smooth_mm or 0.0),
    }
    if in_band.any():
        band = scof[in_band]
        out["SCOF_mean"] = float(np.mean(band))
        out["SCOF_std"] = float(np.std(band))
        out["SCOF_n"] = int(band.size)
    return out


def curve_from_file(filepath, smooth_mm=0.0):
    """Lit un *_Results.csv et renvoie (curve, metadata)."""
    metadata, timeseries, _nodes = rv.parse_results_csv(str(filepath))
    curve = scof_curve(timeseries, metadata, smooth_mm=smooth_mm)
    if curve is not None:
        curve["file"] = Path(filepath).name
    return curve, metadata


# =====================================================================
#  Sorties
# =====================================================================
def write_curve_csv(curves, out_csv):
    """Ecrit les courbes en format long : une ligne par (fichier, echantillon)."""
    parent = Path(out_csv).parent
    if str(parent):
        os.makedirs(str(parent), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["file", "z [mm]", "F_n [N]", "F_t [N]", "SCOF [-]",
                    "SCOF_smooth [-]", "in_load", "in_band"])
        for c in curves:
            for i in range(len(c["z"])):
                w.writerow([
                    c.get("file", ""),
                    "%.6g" % c["z"][i],
                    "%.6g" % c["F_n"][i],
                    "%.6g" % c["F_t"][i],
                    "" if not np.isfinite(c["scof"][i]) else "%.6g" % c["scof"][i],
                    "" if not np.isfinite(c["scof_smooth"][i])
                    else "%.6g" % c["scof_smooth"][i],
                    int(bool(c["in_load"][i])),
                    int(bool(c["in_band"][i])),
                ])
    print("Wrote %d curve(s) to %s" % (len(curves), out_csv))


def plot_curves(curves, png=None, show=True):
    """Deux panneaux partages en z : forces en haut, SCOF en bas."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_f, ax_s) = plt.subplots(
        2, 1, figsize=(8.0, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.4]},
    )
    single = len(curves) == 1

    for c in curves:
        label = c.get("file", "")
        z = c["z"]
        (ln,) = ax_f.plot(z, c["F_n"], lw=1.2,
                          label="F_n" if single else "F_n -- " + label)
        ax_f.plot(z, c["F_t"], lw=1.2, ls="--", color=ln.get_color(),
                  label="F_t" if single else "F_t -- " + label)

        # Serie brute en fond, serie lissee par-dessus quand un lissage est demande.
        if c["smooth_mm"] > 0.0:
            ax_s.plot(z, c["scof"], lw=0.8, alpha=0.30, color=ln.get_color())
            ax_s.plot(z, c["scof_smooth"], lw=1.6, color=ln.get_color(),
                      label=label if not single else "SCOF (lisse %.3g mm)"
                      % c["smooth_mm"])
        else:
            ax_s.plot(z, c["scof"], lw=1.2, color=ln.get_color(),
                      label=label if not single else "SCOF")

        # Bande de moyenne : hachures sur l'intervalle en z reellement retenu.
        if single and c["in_band"].any():
            zb = z[c["in_band"]]
            ax_s.axvspan(float(zb.min()), float(zb.max()), color="0.85", zorder=0,
                         label="bande [%.0f %%, %.0f %%] du pic de F_n"
                               % (100 * SCOF_LO_FRAC, 100 * SCOF_HI_FRAC))
            if "SCOF_mean" in c:
                ax_s.axhline(c["SCOF_mean"], color="k", lw=1.0, ls=":",
                             label="SCOF_mean = %.4f" % c["SCOF_mean"])

        if c.get("z_t") is not None:
            ax_s.axvline(c["z_t"], color="0.4", lw=1.0, ls="-.",
                         label="transition sphere/cone" if single else None)
            ax_f.axvline(c["z_t"], color="0.4", lw=1.0, ls="-.")

    ax_f.set_ylabel("Force [N]")
    ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=8, loc="upper left")

    ax_s.set_xlabel("z le long du scratch [mm]")
    ax_s.set_ylabel("SCOF = F_t / F_n [-]")
    ax_s.grid(alpha=0.3)
    ax_s.legend(fontsize=8, loc="lower right")
    ax_s.set_xlim(0.0, max(c["scratch_length"] for c in curves))

    # L'echelle est bornee sur la bande retenue : le rapport diverge a l'amorce,
    # ou F_n est encore quasi nul, et ecraserait tout le reste de la courbe.
    band_vals = np.concatenate(
        [c["scof"][c["in_band"]] for c in curves if c["in_band"].any()]
    ) if any(c["in_band"].any() for c in curves) else None
    if band_vals is not None and band_vals.size:
        lo, hi = float(np.nanmin(band_vals)), float(np.nanmax(band_vals))
        pad = 0.35 * max(hi - lo, 1e-3)
        ax_s.set_ylim(max(0.0, lo - pad), hi + pad)

    if single:
        fig.suptitle(curves[0].get("file", ""), fontsize=10)
    fig.tight_layout()

    if png:
        parent = Path(png).parent
        if str(parent):
            os.makedirs(str(parent), exist_ok=True)
        fig.savefig(png, dpi=150)
        print("Wrote figure to %s" % png)
    if show:
        plt.show()
    else:
        plt.close(fig)


# =====================================================================
#  CLI
# =====================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="Trace le SCOF le long du scratch.")
    p.add_argument("target", nargs="?", default=None,
                   help="fichier *_Results.csv ou dossier en contenant")
    p.add_argument("--csv", dest="out_csv", default=None,
                   help="export des courbes echantillonnees")
    p.add_argument("--png", dest="png", default=None, help="export de la figure")
    p.add_argument("--smooth", dest="smooth", type=float, default=0.0,
                   help="lissage gaussien du trace, en mm (defaut 0 = brut)")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="n'ouvre pas de fenetre")
    p.add_argument("--no-plot", dest="plot", action="store_false",
                   help="calcul et export seulement")
    args = p.parse_args(argv)

    target = Path(args.target) if args.target else Path.cwd()
    files = [target] if target.is_file() else sorted(target.glob("*_Results.csv"))
    if not files:
        print("No *_Results.csv found in %s" % target)
        return 1

    curves = []
    for f in files:
        curve, _meta = curve_from_file(f, smooth_mm=args.smooth)
        if curve is None:
            print("%-40s SCOF indisponible (RF2/RF3 manquants)" % Path(f).name)
            continue
        curves.append(curve)
        print("%-40s SCOF_mean=%.4f  SCOF_std=%.4f  n=%d  (F_n source: %s)"
              % (curve["file"], curve.get("SCOF_mean", np.nan),
                 curve.get("SCOF_std", np.nan), curve.get("SCOF_n", 0),
                 curve["rf2_source"]))

    if not curves:
        return 1
    if args.out_csv:
        write_curve_csv(curves, args.out_csv)
    if args.plot:
        plot_curves(curves, png=args.png, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())