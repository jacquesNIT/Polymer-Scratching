import numpy as np
import pandas as pd
import os
from scipy.interpolate import griddata

# Import compatible dans les deux cas : module d'un package
# (``from ScratchFeatures.scratch_simulation_helpers import ...``) ou script
# executé directement depuis ce dossier. Remplace ``import ml_scratch.constants``.
try:
    from . import constants as C
except ImportError:  # exécution directe, hors package
    import constants as C


def simulation_main(
    simID,
    path=None,
    z_values=[2.0],
    extract_at_grid_z=False,
    target_shape=None,
):
    """Load one simulation's results and map them onto a regular grid.

    Args:
        simID (str): The ID of the data file (e.g., 'sim00001_Results.csv').
        path (str, optional): The path to the data files.
        z_values (list[float]): Along-scratch z-positions to sample the
            forces at. Ignored if ``extract_at_grid_z`` is True.
        extract_at_grid_z (bool): If True, sample forces at every
            along-scratch z in the mapped grid instead of ``z_values``.
        target_shape (tuple[int, int], optional): (rows, columns) of the
            output grid. Defaults to ``C.target_shape``.

    Returns:
        tuple: ``(X, Y, Z, F_n, F_t, parameters)`` -- the mapped coordinate
        grids, the normal and tangential forces interpolated onto the chosen
        z-values, and the simulation's material parameters.
    """
    parameters, _, rfs, _, _, coords = data_loader(simID, path)
    X, Y, Z = map_coords_to_new_grid(coords, target_shape=target_shape)

    if extract_at_grid_z:
        z_values = np.unique(Z)

    F_n, F_t = extract_forces(rfs, z_values)

    return X, Y, Z, F_n, F_t, parameters


def data_loader(simID, path=None):
    """Load a simulation results CSV and its material-parameters header comment.

    Args:
        simID (str): The ID of the data file (e.g., 'sim00001_Results.csv').
        path (str, optional): The path to the data files.

    Returns:
        tuple: ``(parameters, time, rfs, energies, nodeLabels, coords)`` --
        the material parameters dict, and the time, reaction-force,
        energy, node-label, and coordinate columns as arrays. ``None`` if
        the file is not found.

    Raises:
        ValueError: If the CSV has fewer columns than expected.
    """
    if path:
        file_path = os.path.join(path, simID)
    else:
        file_path = simID

    parameters = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("# Material parameters:"):
                    param_str = line.split(":", 1)[1].strip()
                    pairs = param_str.split(",")

                    for pair in pairs:
                        if "=" in pair:
                            key, value = pair.split("=")
                            key = key.strip()
                            value = value.strip()

                            # formatting
                            try:
                                if "." in value or "e" in value.lower():
                                    val_conv = float(value)
                                elif value.startswith("0") and len(value) > 1:
                                    val_conv = value  # Keep IDs as strings
                                else:
                                    val_conv = int(value)
                            except ValueError:
                                val_conv = value

                            parameters[key] = val_conv

                    # ensure consistent ordering of data
                    # Le tri d'origine servait AUSSI de filtre : tout ce qui
                    # n'etait pas dans sort_order etait supprime. Sur un header
                    # ScratchSimulation (glassy_pc) cela ne laissait que "E" et
                    # jetait rho, nu, sigma_y0, friction_angle, dilation_angle,
                    # mu_friction, mu_pressure_dep... Les cles connues restent en
                    # tete dans le meme ordre, les autres suivent.
                    sort_order = ["id", "E", "A", "B", "n", "mu"]
                    parameters = {
                        **{k: parameters[k] for k in sort_order if k in parameters},
                        **{k: v for k, v in parameters.items() if k not in sort_order},
                    }

                    break
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    df = pd.read_csv(file_path, comment="#")
    df_keys = df.keys()

    if len(df_keys) < 8:
        raise ValueError(f"CSV file has fewer columns ({len(df_keys)}) than expected.")

    time = df[df_keys[0]].to_numpy()
    rfs = df[df_keys[1:4]].to_numpy()
    energies = df[df_keys[4:6]].to_numpy()
    nodeLabels = df[df_keys[6]].to_numpy()
    coords = df[df_keys[7:]].to_numpy()

    return parameters, time, rfs, energies, nodeLabels, coords


def extract_forces(rfs, z_values):
    """Clean the raw reaction forces and map them onto ``z_values``.

    Args:
        rfs (np.ndarray): Reaction-force columns from :func:`data_loader`,
            with normal force in column 1 and tangential force in column 2.
        z_values (array-like): Along-scratch z-positions to sample the
            forces at.

    Returns:
        tuple[np.ndarray, np.ndarray]: Normal and tangential force,
        interpolated onto ``z_values``. ``NaN`` outside the recorded range.
    """
    # Simulation data is a half-model, so multiply by 2 to get the full force.
    normal_force_clean = 2 * rfs[~np.isnan(rfs[:, 1]), 1]
    tangential_force_clean = 2 * rfs[~np.isnan(rfs[:, 2]), 2]
    rfs_len = len(normal_force_clean)

    z_force_domain = np.linspace(0, C.scratch_length, rfs_len)

    normal_forces_mapped = np.interp(
        z_values, z_force_domain, normal_force_clean, right=np.nan
    )
    tangential_forces_mapped = np.interp(
        z_values, z_force_domain, tangential_force_clean, right=np.nan
    )

    return normal_forces_mapped, tangential_forces_mapped


def map_coords_to_new_grid(coords, method: str = "linear", target_shape=None):
    """Map the original mesh coordinates onto a new regular z-x grid.

    Interpolates the y-values (height) at each new grid point.

    Args:
        coords (np.ndarray): Array with shape (N, >=6). Columns [3], [4],
            [5] are x, y, z.
        method (str): Interpolation method for ``griddata``.
        target_shape (tuple[int, int], optional): (rows, columns) of the new
            grid. Defaults to ``C.target_shape``; pass a finer shape (e.g.
            for a smoother-looking surface plot) or a coarser one as needed.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(X_new, Y_new, Z_new)``
        -- the interpolated coordinates on the new grid.
    """
    x, y, z = coords[:, 3], coords[:, 4], coords[:, 5]

    # Mirror across x=0
    x = np.append(x, -x)
    y = np.append(y, y)
    z = np.append(z, z)

    points = np.column_stack((z.ravel(), x.ravel()))
    values = y.ravel()

    new_rows, new_cols = target_shape or C.target_shape

    x_new = np.linspace(
        -C.scratch_domain_width / 2, C.scratch_domain_width / 2, new_rows
    )
    z_new = np.linspace(0, C.scratch_domain_length, new_cols)
    Z_new, X_new = np.meshgrid(z_new, x_new, indexing="xy")
    Y_new = griddata(points, values, (Z_new, X_new), method=method)

    return X_new, Y_new, Z_new