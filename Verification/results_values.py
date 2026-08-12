"""
    python results_values_full.py <dossier_ou_csv> [<csv_sortie>] [<z_mm>]
"""

import csv as _csv
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# =====================================================================
#  Constantes de grille (recopiees de ScratchFeatures/constants.py)
# =====================================================================
# TARGET_SHAPE          : (lignes, colonnes) de la grille reguliere sur laquelle
#                         la topographie est reechantillonnee.
# SCRATCH_LENGTH        : longueur commandee du scratch (mm).
# SCRATCH_DOMAIN_WIDTH  : largeur TOTALE du domaine apres miroir (mm). Le modele
#                         est un demi-modele : x_undeformed va de 0 a 0.3, donc
#                         0.6 une fois la symetrie appliquee.
# SCRATCH_DOMAIN_LENGTH : longueur totale du domaine maille (mm).
TARGET_SHAPE = (80, 420)
SCRATCH_LENGTH = 2.0
SCRATCH_DOMAIN_WIDTH = 0.6
SCRATCH_DOMAIN_LENGTH = 2.5

# Tolerance relative des fenetres temporelles. Les horodatages Abaqus sont
# ecrits en float32 : le dernier echantillon de la phase active vaut
# 0.100000003 pour scratch_time = 0.1. La tolerance 1e-9 de results_verifier
# est plus serree que cette erreur d'arrondi et exclut donc ce dernier point.
TIME_TOL = 1e-6


# =====================================================================
#  Lecture du CSV — recopie de results_verifier.parse_results_csv
# =====================================================================
def parse_results_csv(filepath):
    """Parse le CSV du post-processeur.

    Les colonnes de series temporelles sont indexees par leur NOM dans la ligne
    d'entete, et les noeuds par la position de "NodeLabel" : le fichier peut
    donc porter les 27 colonnes de series sans que rien ne soit a adapter.

    metadata   : dict — parametres materiau et simulation lus dans les lignes '#'
    timeseries : dict — {colonne: np.ndarray} pour Time, RF1-3, CFN/CFS, energies
    nodes      : dict — {"labels", "undeformed" (Nx3), "deformed" (Nx3)}
    """
    metadata = {}
    header_cols = []
    ts_rows = []
    node_labels, node_undef, node_def = [], [], []

    with open(filepath, "r", encoding="latin-1") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().replace("\r", "")
        if not line:
            continue

        if line.startswith("#"):
            if "WallclockTime=" in line:
                m = re.search(r"WallclockTime=([\d\.eE+-]+)", line)
                if m:
                    metadata["wallclock"] = float(m.group(1))
            if "Material parameters:" in line or "Material:" in line:
                for m in re.finditer(r"(\w+)=([\d\.eE+-]+)", line):
                    try:
                        metadata[m.group(1)] = float(m.group(2))
                    except ValueError:
                        pass
            if "tip radius" in line.lower():
                m = re.search(r"tip radius\s*([\d\.eE+-]+)\s*mm", line, re.IGNORECASE)
                if m:
                    metadata["tip_radius"] = float(m.group(1))
                m = re.search(r"cone angle\s*([\d\.eE+-]+)", line, re.IGNORECASE)
                if m:
                    metadata["cone_angle"] = float(m.group(1))
            if "Simulation Parameters" in line:
                body = line.split("Parameters:", 1)[1]
                for k, v in re.findall(r"(\w+)=([A-Za-z0-9\.eE+-]+)", body):
                    try:
                        metadata[k] = float(v)
                    except ValueError:
                        metadata[k] = v

            if (line.count("=") == 1 and "parameters:" not in line.lower()
                    and "WallclockTime" not in line):
                m = re.match(r"#\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", line)
                if m:
                    key, val = m.group(1), m.group(2)
                    try:
                        metadata[key] = float(val)
                    except ValueError:
                        metadata[key] = val
            continue

        if "Time" in line and "RF1" in line:
            header_cols = [c.strip() for c in line.split(",")]
            continue

        parts = line.split(",")
        if len(parts) < 2 or not header_cols:
            continue

        if parts[0].strip():
            try:
                row = {}
                for ci, col in enumerate(header_cols):
                    if ci < len(parts) and parts[ci].strip():
                        row[col] = float(parts[ci])
                ts_rows.append(row)
            except ValueError:
                pass

        label_idx = header_cols.index("NodeLabel") if "NodeLabel" in header_cols else 7
        if label_idx < len(parts) and parts[label_idx].strip():
            try:
                node_labels.append(int(float(parts[label_idx])))
                node_undef.append([float(parts[label_idx + i]) for i in (1, 2, 3)])
                node_def.append([float(parts[label_idx + i]) for i in (4, 5, 6)])
            except (ValueError, IndexError):
                pass

    timeseries = {}
    if ts_rows:
        all_cols = set()
        for row in ts_rows:
            all_cols.update(row.keys())
        for col in all_cols:
            timeseries[col] = np.array([row.get(col, 0.0) for row in ts_rows])

    nodes = {
        "labels": np.array(node_labels),
        "undeformed": np.array(node_undef) if node_undef else np.empty((0, 3)),
        "deformed": np.array(node_def) if node_def else np.empty((0, 3)),
    }

    return metadata, timeseries, nodes


# =====================================================================
#  Helpers — recopies de results_verifier
# =====================================================================
WM_BALANCE_TERMS = ("WM_ALLIE", "WM_ALLVD", "WM_ALLFD", "WM_ALLKE",
                    "WM_ALLWK", "WM_ALLPW", "WM_ALLCW", "WM_ALLMW")


def _peak(x):
    """Valeur absolue maximale d'une serie, 0.0 si absente ou vide."""
    return float(np.max(np.abs(x))) if x is not None and len(x) else 0.0


def _active_end(metadata, t_last):
    """Fin de la phase ACTIVE (indentation + scratch), plafonnee par t_last.

    En depth_mode=constant l'indentation precede le scratch, donc la phase
    active dure indentation_time + scratch_time. En progressive l'indenteur
    descend pendant le scratch : la phase active s'arrete a scratch_time.
    Sans ce plafond, toute fenetre ancree sur time[-1] glisserait dans la
    decharge / le recovery.
    """
    st = float(metadata.get("scratch_time", 0.0) or 0.0)
    it = float(metadata.get("indentation_time", 0.0) or 0.0)
    constant = not str(metadata.get("depth_mode", "")).lower().startswith("prog")
    t_act = st + (it if constant else 0.0)
    if t_act <= 0.0:
        return t_last
    return min(t_last, t_act)


def _normal_force_series(timeseries, metadata):
    """Serie de force normale et son origine.

    En pilotage par deplacement, u2 porte une CL de deplacement et RF2 est la
    reaction physique. En pilotage par force, u2 ne porte qu'une charge
    concentree : RF2 est nul et c'est CFN2, la force normale totale sur la
    surface esclave, qui fait foi. Repli sur l'autre colonne si l'une est
    absente ou identiquement nulle.
    """
    rf2 = timeseries.get("RF2")
    cfn2 = timeseries.get("CFN2")
    control_mode = str(metadata.get("control_mode", "displacement"))

    def _nonzero(arr):
        return arr is not None and len(arr) and float(np.max(np.abs(arr))) > 1e-20

    if control_mode == "force":
        if _nonzero(cfn2):
            return cfn2, "CFN2"
        if _nonzero(rf2):
            return rf2, "RF2 (fallback, CFN2 unavailable/zero)"
        return None, "unavailable"

    if _nonzero(rf2):
        return rf2, "RF2"
    if _nonzero(cfn2):
        return cfn2, "CFN2 (fallback, RF2 unavailable/zero)"
    return None, "unavailable"


# =====================================================================
#  Grille et profils — recopies de ScratchFeatures
# =====================================================================
def extract_forces(rfs, z_values):
    """Nettoie les forces de reaction et les projette sur ``z_values``.

    ``rfs`` a la colonne 1 = force normale, colonne 2 = force tangentielle.
    Le temps restant est suppose avancer lineairement de z = 0 a
    z = SCRATCH_LENGTH sur les echantillons fournis.
    """
    # Demi-modele : x2 pour obtenir la force du modele complet.
    normal_force_clean = 2 * rfs[~np.isnan(rfs[:, 1]), 1]
    tangential_force_clean = 2 * rfs[~np.isnan(rfs[:, 2]), 2]
    rfs_len = len(normal_force_clean)

    z_force_domain = np.linspace(0, SCRATCH_LENGTH, rfs_len)

    normal_forces_mapped = np.interp(
        z_values, z_force_domain, normal_force_clean, right=np.nan
    )
    tangential_forces_mapped = np.interp(
        z_values, z_force_domain, tangential_force_clean, right=np.nan
    )

    return normal_forces_mapped, tangential_forces_mapped


def map_coords_to_new_grid(coords, method="linear", target_shape=None):
    """Projette le nuage de noeuds sur une grille reguliere z-x.

    Les coordonnees utilisees sont les coordonnees DEFORMEES (colonnes 3, 4, 5
    de ``coords``), c'est-a-dire une carte topographique eulerienne : un point
    de grille a z = 2.0 recoit la matiere qui s'y trouve a la fin du run,
    quelle que soit sa position initiale. Le demi-modele est d'abord replie
    par symetrie x -> -x.
    """
    x, y, z = coords[:, 3], coords[:, 4], coords[:, 5]

    # Miroir autour de x=0
    x = np.append(x, -x)
    y = np.append(y, y)
    z = np.append(z, z)

    points = np.column_stack((z.ravel(), x.ravel()))
    values = y.ravel()

    new_rows, new_cols = target_shape or TARGET_SHAPE

    x_new = np.linspace(-SCRATCH_DOMAIN_WIDTH / 2, SCRATCH_DOMAIN_WIDTH / 2, new_rows)
    z_new = np.linspace(0, SCRATCH_DOMAIN_LENGTH, new_cols)
    Z_new, X_new = np.meshgrid(z_new, x_new, indexing="xy")
    Y_new = griddata(points, values, (Z_new, X_new), method=method)

    return X_new, Y_new, Z_new


def get_profiles_from_coords(X, Y, Z, x_value=0.0, z_value=2.0):
    """Decoupe les profils xy (coupe transverse) et yz (ligne longitudinale).

    xy_profile : colonne de grille la plus proche de z_value -> (x, y)
    yz_profile : ligne de grille la plus proche de x_value  -> (y, z)
    """
    idx = np.argmin(np.abs(Z[0, :] - z_value))
    xy_profile = (X[:, idx].squeeze(), Y[:, idx].squeeze())

    idx = np.argmin(np.abs(X[:, 0] - x_value))
    y_coords = Y[idx, :]
    yz_profile = (y_coords, Z[0, :])

    return xy_profile, yz_profile


def _safe_nanmean(*args):
    """Moyenne des scalaires fournis en ignorant les NaN."""
    valid = [v for v in args if not np.isnan(v)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def calc_xy_peak_indexes(x, y, noise_floor_fraction=0.03):
    """Indices des deux cretes de bourrelet encadrant le sillon.

    1. Lissage gaussien fort (sigma=3 px) pour situer le creux macroscopique,
       insensible aux decrochages ponctuels : c'est l'ancre du sillon.
    2. Seuil de proeminence dynamique = 3 % du relief total (max - min).
    3. find_peaks sur le profil BRUT avec ce seuil.
    4. Partition gauche/droite par rapport a l'ancre ; sur chaque flanc, la
       crete retenue est la plus PROEMINENTE (pas la plus haute).
    """
    y_macro = gaussian_filter1d(y, sigma=3.0)
    idx_groove = int(np.argmin(y_macro))

    profile_relief = float(np.ptp(y))
    min_prom = noise_floor_fraction * profile_relief

    peaks, props = find_peaks(y, prominence=min_prom)
    prominences = props["prominences"]

    if len(peaks) < 2:
        raise ValueError(
            f"Profile topography resolved {len(peaks)} peaks; a minimum of 2 "
            "are required to bracket a scratch groove."
        )

    left_mask = peaks < idx_groove
    right_mask = peaks > idx_groove

    if not np.any(left_mask) or not np.any(right_mask):
        raise ValueError(
            f"Groove anchor at index {idx_groove} is not bracketed by peaks "
            "on both sides."
        )

    left_candidates = peaks[left_mask]
    left_prominences = prominences[left_mask]
    right_candidates = peaks[right_mask]
    right_prominences = prominences[right_mask]

    peak_idx_1 = int(left_candidates[np.argmax(left_prominences)])
    peak_idx_2 = int(right_candidates[np.argmax(right_prominences)])

    if peak_idx_1 > peak_idx_2:
        peak_idx_1, peak_idx_2 = peak_idx_2, peak_idx_1

    return peak_idx_1, peak_idx_2


def get_pile_up_height(x, y, peak_idx=None):
    """Hauteur de bourrelet = ordonnee au pic, ou maximum du profil si
    ``peak_idx`` est None. Mesuree par rapport a y = 0, la surface d'origine."""
    return y[peak_idx] if peak_idx is not None else np.nanmax(y)


def get_residual_depth(x, y):
    """Profondeur residuelle = valeur absolue du minimum du profil.

    Le abs() est applique sans controle de signe : si le profil est entierement
    au-dessus de y = 0 (sillon efface par la reprise elastique, ce qui arrive
    a z proche de scratch_length), la valeur retournee est le plus petit
    soulevement et non une profondeur.
    """
    h_r = np.nanmin(y)
    return np.abs(h_r)


def get_frontal_pile_up_height(yz_profile):
    """Hauteur de bourrelet frontal, branche h_fp de get_data_from_yz_profile.

    Le profil longitudinal pris a x ~ 0 (plan de symetrie) est restreint a
    z >= 2.0, puis h_fp = nanmax(y) sur ce segment. Le seuil 2.0 est ecrit en
    dur dans la source d'origine, il n'est pas indexe sur SCRATCH_LENGTH.
    """
    y, z = yz_profile
    mask = z >= 2.0
    y = y[mask]
    z = z[mask]
    return get_pile_up_height(z, y)


# =====================================================================
#  Assemblage des valeurs
# =====================================================================
def energy_values(timeseries, metadata):
    """Valeurs energetiques, formules de results_verifier sans les verdicts."""
    out = {}

    ke = timeseries.get("ALLKE")
    ie = timeseries.get("ALLIE")
    ae = timeseries.get("ALLAE")
    time = timeseries.get("Time")

    # --- KE/IE : ratio energie cinetique / energie interne du substrat, en %.
    # steady_max : maximum sur la fenetre 0.1*t_act < t <= t_act, ou t_act est
    #   la fin de la phase active. Les 10 % initiaux sont exclus car le ratio y
    #   est domine par la mise en mouvement, IE etant encore quasi nul.
    # overall_max : maximum sur toute la serie, fenetre comprise.
    if ke is not None and ie is not None:
        mask = ie > 1e-20
        if mask.any():
            ratio = ke[mask] / ie[mask] * 100.0
            time_m = time[mask] if time is not None else np.arange(len(ratio))
            t_max = time_m[-1] if len(time_m) else 1.0
            t_act = _active_end(metadata, t_max) if metadata is not None else t_max
            steady = (time_m > 0.1 * t_act) & (time_m <= t_act * (1.0 + 1e-9))
            out["KE_IE_steady_max"] = float(
                np.max(ratio[steady]) if steady.any() else np.max(ratio)
            )
            out["KE_IE_overall_max"] = float(np.max(ratio))

    # --- AE/IE : energie artificielle de hourglass rapportee a l'energie
    # interne, en %, lue au DERNIER echantillon de la phase active. Prendre le
    # tout dernier echantillon de la serie donnerait une valeur gonflee, IE
    # ayant chute apres la relaxation elastique.
    if ae is not None and ie is not None:
        mask = ie > 1e-20
        if mask.any():
            ratio = ae[mask] / ie[mask] * 100.0
            final = ratio[-1]
            if metadata is not None and time is not None and len(time) == len(ae):
                t_act = _active_end(metadata, float(time[-1]))
                tm = time[mask]
                sel = np.nonzero(tm <= t_act * (1.0 + 1e-9))[0]
                if sel.size:
                    final = ratio[sel[-1]]
            out["AE_IE_final"] = float(final)

    et = timeseries.get("ETOTAL")
    wk = timeseries.get("WM_ALLWK")

    # --- E_ref : echelle d'energie PHYSIQUE servant de denominateur a toutes
    # les grandeurs normalisees ci-dessous. C'est max(|ALLIE|, |WM_ALLWK|) :
    # l'energie cinetique du driver en est exclue.
    e_ref = max(_peak(ie), _peak(wk))
    if e_ref >= 1e-20:
        out["E_ref"] = float(e_ref)

        # Bilan reconstruit a partir des composantes whole-model :
        #   IE + VD + FD + KE - WK - PW - CW - MW, qui doit rester constant.
        have_wm = all(timeseries.get(k) is not None for k in WM_BALANCE_TERMS)
        recon = None
        if have_wm:
            recon = (
                timeseries["WM_ALLIE"] + timeseries["WM_ALLVD"]
                + timeseries["WM_ALLFD"] + timeseries["WM_ALLKE"]
                - timeseries["WM_ALLWK"] - timeseries["WM_ALLPW"]
                - timeseries["WM_ALLCW"] - timeseries["WM_ALLMW"]
            )

        bal = et if et is not None else recon
        if bal is not None:
            # Valeur du bilan a t=0 (energie cinetique initiale du driver).
            # Sert de reference au drift ci-dessous, non reportee en sortie.
            baseline = float(bal[0])
            # --- ETOTAL_drift : ecart MAXIMAL du bilan a sa valeur initiale,
            # rapporte a E_ref. Mesure la non-conservation.
            out["ETOTAL_drift"] = (
                float(np.max(np.abs(bal - baseline))) / e_ref * 100.0
            )
        if ie is not None:
            # --- Maximum de l'energie interne du substrat.
            out["ALLIE_max"] = float(np.max(ie))

        # --- ALLPW : travail de penalisation du contact rapporte a E_ref,
        # en %. Le numerateur est le PIC de la serie, pas sa valeur finale.
        pw = timeseries.get("WM_ALLPW")
        if pw is not None:
            out["ALLPW"] = float(_peak(pw) / e_ref * 100.0)

    # --- Settling : energie cinetique du substrat au DERNIER echantillon
    # (fin du recovery) rapportee au PIC d'energie interne, en %. Indique si
    # la surface vibre encore quand la frame finale est ecrite.
    if time is not None and ke is not None and ie is not None and len(time) >= 2:
        ie_peak = _peak(ie)
        if ie_peak >= 1e-20:
            out["KE_final_over_IE_peak"] = abs(float(ke[-1])) / ie_peak * 100.0

    return out


def scof_values(timeseries, metadata):
    """SCOF moyen sur le scratch, methode de check_friction_physics.

    Le masque retient les echantillons ou la force normale depasse 10 % de son
    pic, ce qui exclut l'approche et la decharge. En depth_mode=constant on
    restreint en plus a la fenetre [t_act - scratch_time, t_act] : pendant
    l'indentation RF3 est quasi nul alors que RF2 porte deja toute la charge,
    et inclure ces echantillons tirerait la moyenne vers le bas. En
    progressive, indentation et scratch sont concurrents : pas de restriction
    supplementaire.
    """
    out = {}
    rf3 = timeseries.get("RF3")
    rf2, _ = _normal_force_series(timeseries, metadata)
    if rf2 is None or rf3 is None:
        return out

    rf2_abs = np.abs(rf2)
    mask = rf2_abs > np.max(rf2_abs) * 0.10

    time = timeseries.get("Time")
    st = metadata.get("scratch_time")
    constant_mode = not str(metadata.get("depth_mode", "")).lower().startswith("prog")
    if constant_mode and time is not None and st and float(st) > 0.0:
        t_act = _active_end(metadata, float(time[-1]))
        mask = mask & (time >= t_act - float(st)) & (time <= t_act * (1.0 + 1e-9))

    if not mask.any():
        return out

    scof = np.abs(rf3[mask]) / rf2_abs[mask]
    out["SCOF_mean"] = float(np.mean(scof))
    out["SCOF_std"] = float(np.std(scof))
    return out


def force_values(timeseries, metadata, z_value):
    """Forces normale et tangentielle du modele complet a z = z_value.

    La serie est d'abord restreinte a la fenetre pendant laquelle l'indenteur
    avance (phase active en progressive, phase de scratch seule en constant),
    puis etiree lineairement sur z in [0, SCRATCH_LENGTH] et interpolee a
    z_value. Le facteur 2 corrige la symetrie du demi-modele.
    """
    out = {}
    rf3 = timeseries.get("RF3")
    rf2, _ = _normal_force_series(timeseries, metadata)
    time = timeseries.get("Time")
    if rf2 is None or rf3 is None:
        return out

    if time is not None and len(time) == len(rf2):
        t_act = _active_end(metadata, float(time[-1]))
        keep = time <= t_act * (1.0 + TIME_TOL)
        # En depth_mode=constant l'indentation precede le scratch : z reste
        # nul pendant indentation_time. Le mapping lineaire temps -> z de
        # extract_forces suppose que z avance de 0 a SCRATCH_LENGTH sur TOUS
        # les echantillons fournis ; garder la phase d'indentation decalerait
        # donc tout z intermediaire. On restreint a [t_act - scratch_time,
        # t_act], comme le fait deja scof_values. En progressive, indentation
        # et scratch sont concurrents : pas de restriction supplementaire.
        st = metadata.get("scratch_time")
        constant_mode = not str(
            metadata.get("depth_mode", "")
        ).lower().startswith("prog")
        if constant_mode and st and float(st) > 0.0:
            t_scratch_start = t_act - float(st)
            keep = keep & (time >= t_scratch_start - abs(t_act) * TIME_TOL)
    else:
        keep = np.ones(len(rf2), dtype=bool)
    if not keep.any():
        return out

    rfs = np.column_stack(
        [np.zeros(int(keep.sum())), np.asarray(rf2)[keep], np.asarray(rf3)[keep]]
    )
    F_n, F_t = extract_forces(rfs, [z_value])

    out["F_n"] = abs(float(np.asarray(F_n).ravel()[-1]))
    out["F_t"] = abs(float(np.asarray(F_t).ravel()[-1]))
    return out


def profile_values(nodes, z_value):
    """Topographie residuelle, methode du pipeline ScratchFeatures."""
    out = {}
    undef = nodes["undeformed"]
    deform = nodes["deformed"]
    if len(deform) == 0:
        return out

    # map_coords_to_new_grid attend les coordonnees deformees en colonnes 3-5.
    coords = np.hstack([undef, deform])

    X, Y, Z = map_coords_to_new_grid(coords)
    xy_profile, yz_profile = get_profiles_from_coords(X, Y, Z, z_value=z_value)
    x, y = xy_profile

    # --- h_r : |min| de la coupe transverse complete a z_value.
    out["h_r"] = float(get_residual_depth(x, y))

    # --- h_p : les deux cretes sont localisees par proeminence, le profil est
    # coupe au point le plus bas entre elles, et la hauteur de chaque flanc est
    # moyennee. Le miroir x -> -x rend les deux flancs identiques par
    # construction sur donnees simulees : la moyenne est alors sans effet.
    try:
        peak_idx_left, peak_idx_right = calc_xy_peak_indexes(x, y)
        idx_groove = peak_idx_left + int(
            np.argmin(y[peak_idx_left : peak_idx_right + 1])
        )
        h_p_left = get_pile_up_height(x[:idx_groove], y[:idx_groove], peak_idx_left)
        h_p_right = get_pile_up_height(
            x[idx_groove:], y[idx_groove:], peak_idx_right - idx_groove
        )
        out["h_p"] = float(_safe_nanmean(h_p_left, h_p_right))
    except ValueError:
        out["h_p"] = np.nan

    # --- h_fp : maximum du profil longitudinal a x ~ 0, pour z >= 2.0.
    out["h_fp"] = float(get_frontal_pile_up_height(yz_profile))
    return out


# Unites affichees en sortie. Systeme mm-tonne-s-MPa-N : energie en N.mm = mJ.
UNITS = {
    "z_extraction": "mm",
    "wallclock": "s",
    "KE_IE_steady_max": "%",
    "KE_IE_overall_max": "%",
    "AE_IE_final": "%",
    "E_ref": "mJ",
    "ETOTAL_drift": "%",
    "ALLIE_max": "mJ",
    "ALLPW": "%",
    "KE_final_over_IE_peak": "%",
    "F_n": "N",
    "F_t": "N",
    "SCOF_mean": "-",
    "SCOF_std": "-",
    "h_r": "mm",
    "h_p": "mm",
    "h_fp": "mm",
}


def extract_values(filepath, z_value=None):
    z_value = float(z_value) if z_value is not None else SCRATCH_LENGTH

    metadata, timeseries, nodes = parse_results_csv(filepath)

    values = {"file": Path(filepath).name, "z_extraction": z_value}
    # Temps de calcul du run, lu dans la ligne "# WallclockTime=" de l'entete.
    if metadata.get("wallclock") is not None:
        values["wallclock"] = float(metadata["wallclock"])
    values.update(energy_values(timeseries, metadata))
    values.update(force_values(timeseries, metadata, z_value))
    values.update(scof_values(timeseries, metadata))
    values.update(profile_values(nodes, z_value))
    return values


def _print_values(values):
    for k, v in values.items():
        unit = UNITS.get(k, "")
        if isinstance(v, float):
            print("%-28s %14.6g  %s" % (k, v, unit))
        else:
            print("%-28s %14s  %s" % (k, v, unit))
    print()


def main(target=None, out_csv=None, z_value=None):
    target = Path(target) if target else Path.cwd()
    files = [target] if target.is_file() else sorted(target.glob("*_Results.csv"))

    if not files:
        print(f"No *_Results.csv found in {target}")
        return

    rows = [extract_values(f, z_value) for f in files]
    for r in rows:
        _print_values(r)

    if out_csv:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        parent = Path(out_csv).parent
        if str(parent):
            os.makedirs(str(parent), exist_ok=True)
        # L'unite est portee par le nom de colonne : "F_n [N]", "h_r [mm]"...
        headers = [k + (" [%s]" % UNITS[k] if k in UNITS else "") for k in keys]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(headers)
            for r in rows:
                w.writerow([r.get(k, "") for k in keys])
        print(f"Wrote {len(rows)} row(s) to {out_csv}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else None,
        sys.argv[2] if len(sys.argv) > 2 else None,
        sys.argv[3] if len(sys.argv) > 3 else None,
    )