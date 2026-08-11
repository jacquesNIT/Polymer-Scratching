import numpy as np
from numpy.polynomial import Polynomial

# ``np.trapezoid`` n'existe que depuis numpy 2.0 ; sous numpy 1.x c'est
# ``np.trapz`` (meme fonction, meme signature). Alias ajoute ici pour que les
# appels ci-dessous restent inchanges quelle que soit la version installee.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def feature_extraction_pipeline(
    X,
    Y,
    Z,
    F_n,
    F_t,
    parameters,
    z_value=2.0,
    plot=False,
    save_dir="reports/figures/feature_extraction/",
    get_additional_features=False,
    get_yz_profile_features=False,
    get_volume_features=False,
):
    """Extract tribological features from one cross-section of a scratch scan.

    Args:
        X (np.ndarray): 2-D across-scratch coordinate grid.
        Y (np.ndarray): 2-D height grid.
        Z (np.ndarray): 2-D along-scratch coordinate grid.
        F_n (array-like): Normal force samples; the last value is used.
        F_t (array-like): Tangential force samples; the last value is used.
        parameters (dict): Material/simulation parameters merged into the
            returned feature dict (e.g. ``E``, ``A``, ``B``, ``n``, ``mu``).
        z_value (float): Along-scratch position to extract the xy profile at.
        plot (bool): If True, generate and save a diagnostic figure per
            feature under ``save_dir``.
        save_dir (str): Directory for diagnostic figures when ``plot`` is
            True.
        get_additional_features (bool): If True, also compute w_p, A_p, A_g,
            k_p, k_g (PI9-PI13, level 2+). Level 1 (PI6-PI8) only needs h_p,
            h_r, w.
        get_yz_profile_features (bool): If True, also compute the frontal
            pile-up features (h_fp, w_fp, w_fp_d, A_fp, k_fp) from the yz
            profile.
        get_volume_features (bool): If True, also compute the pile-up and
            groove volumes (V_p, V_g) from the full 3-D grid.

    Returns:
        dict[str, float]: ``parameters`` merged with F_n, F_t, and the
        requested profile/volume features.
    """
    xy_profile, yz_profile = get_profiles_from_coords(X, Y, Z, z_value=z_value)

    axes_xy_dict = {}
    axes_yz_dict = {}
    if plot:
        xy_plot_configs = [
            ("peaks", "Identified Peaks"),
            ("h_p", "Pile-up height"),
            ("h_r", "Residual depth"),
            ("w", "Scratch width"),
        ]
        if get_additional_features:
            xy_plot_configs += [
                ("A_p", "Pile-up Area"),
                ("A_g", "Groove Area"),
                ("w_p", "Pile-up Width"),
                ("k_p", "Pile-up Curvature"),
                ("k_g", "Groove Curvature"),
            ]
        if get_yz_profile_features:
            yz_plot_configs = [
                ("h_fp", "Frontal Pile-up Height"),
                ("w_fp", "Frontal Pile-up Width"),
                ("w_fp_d", "Distance to Frontal Pile-up Peak"),
                ("A_fp", "Frontal Pile-up Area"),
                ("k_fp", "Frontal Pile-up Curvature"),
            ]

        for key, title in xy_plot_configs:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_title(title)
            ax.plot(
                xy_profile[0], xy_profile[1], color="black", linewidth=1.5, alpha=0.5
            )
            axes_xy_dict[key] = ax

        if get_yz_profile_features:
            for key, title in yz_plot_configs:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.set_title(title)
                ax.plot(
                    yz_profile[1],
                    yz_profile[0],
                    color="black",
                    linewidth=1.5,
                    alpha=0.5,
                )
                ax.set_xlim(2.0, 2.5)
                axes_yz_dict[key] = ax

    features_from_xy = get_data_from_xy_profile(
        xy_profile,
        axes_dict=axes_xy_dict if plot else None,
        get_additional_features=get_additional_features,
    )

    if get_yz_profile_features:
        features_from_yz = get_data_from_yz_profile(
            yz_profile, axes_dict=axes_yz_dict if plot else None
        )

    if get_volume_features:
        V_p = get_pile_up_volume(X, Y, Z)
        V_g = get_groove_volume(X, Y, Z)
        volume_features = {"V_p": V_p, "V_g": V_g}

    if plot:
        for key, ax in axes_xy_dict.items():
            if ax is not None:
                ax.legend(fontsize="small")
                ax.figure.tight_layout()
                fig = ax.figure
                filepath = os.path.join(save_dir, f"{key}.png")
                fig.savefig(filepath, dpi=300, bbox_inches="tight")
                plt.close(fig)

        if get_yz_profile_features:
            for key, ax in axes_yz_dict.items():
                if ax is not None:
                    ax.legend(fontsize="small")
                    ax.figure.tight_layout()
                    fig = ax.figure
                    filepath = os.path.join(save_dir, f"{key}.png")
                    fig.savefig(filepath, dpi=300, bbox_inches="tight")
                    plt.close(fig)

    features = {
        **parameters,
        "F_n": abs(F_n[-1]),
        "F_t": abs(F_t[-1]),
        **(volume_features if get_volume_features else {}),
        **features_from_xy,
        **(features_from_yz if get_yz_profile_features else {}),
    }

    return features


def get_profiles_from_coords(X, Y, Z, x_value=0.0, z_value=2.0):
    """Slice the xy and yz profiles out of a 3-D topography grid.

    Args:
        X (np.ndarray): 2-D across-scratch coordinate grid.
        Y (np.ndarray): 2-D height grid.
        Z (np.ndarray): 2-D along-scratch coordinate grid.
        x_value (float): Across-scratch position to extract the yz profile at.
        z_value (float): Along-scratch position to extract the xy profile at.

    Returns:
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        ``(xy_profile, yz_profile)``, each an ``(x, y)`` / ``(y, z)``
        coordinate pair at the nearest available grid position.
    """
    idx = np.argmin(np.abs(Z[0, :] - z_value))
    xy_profile = (X[:, idx].squeeze(), Y[:, idx].squeeze())

    idx = np.argmin(np.abs(X[:, 0] - x_value))
    y_coords = Y[idx, :]
    yz_profile = (y_coords, Z[0, :])

    return xy_profile, yz_profile


def _safe_nanmean(*args: float) -> float:
    """Mean of the given scalars, ignoring NaNs.

    Args:
        *args (float): Scalar values to average.

    Returns:
        float: Mean of the non-NaN values, or NaN if all are NaN.
    """
    valid = [v for v in args if not np.isnan(v)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def get_data_from_xy_profile(
    xy_profile: tuple[np.ndarray, np.ndarray],
    axes_dict: dict | None = None,
    get_additional_features: bool = False,
) -> dict[str, float]:
    """Extracts tribological metrics from a 1D scratch profile.

    Args:
        xy_profile (tuple[np.ndarray, np.ndarray]): Global coordinate tuple
            (x, y).
        axes_dict (dict | None): Dictionary mapping feature keys to
            Matplotlib Axes instances.
        get_additional_features (bool): If True, also computes w_p, A_p, A_g,
            k_p, k_g (needed for PI9-PI13, i.e. level 2+). Level 1 (PI6-PI8)
            only needs h_p, h_r, w.

    Returns:
        dict[str, float]: h_p, h_r, w, and (if requested) w_p, A_p, A_g, k_p,
        k_g.
    """
    x, y = xy_profile
    ax = axes_dict if axes_dict is not None else {}

    # 1. Resolve global peak coordinates
    peak_idx_left, peak_idx_right = calc_xy_peak_indexes(x, y, axes=ax.get("peaks"))

    # 2. Derive the true topological groove pivot strictly between the crests
    idx_groove = peak_idx_left + int(np.argmin(y[peak_idx_left : peak_idx_right + 1]))

    # 3. Partition profile into strict tribological flanks
    # Left flank domain: [0 ---> Groove Pivot]
    x_left, y_left = x[:idx_groove], y[:idx_groove]
    local_idx_left = peak_idx_left  # Slice offset is 0

    # Right flank domain: [Groove Pivot ---> End]
    x_right, y_right = x[idx_groove:], y[idx_groove:]
    local_idx_right = peak_idx_right - idx_groove  # Slice offset is idx_groove

    # 4. Flank-dependent intensive properties (Averaged via safe mean)
    h_p_left = get_pile_up_height(x_left, y_left, local_idx_left, axes=ax.get("h_p"))
    h_p_right = get_pile_up_height(
        x_right, y_right, local_idx_right, axes=ax.get("h_p")
    )
    h_p = _safe_nanmean(h_p_left, h_p_right)

    # 5. Extensive properties (Additive sums across flanks)
    w_left = get_scratch_width(x_left, y_left, local_idx_left, axes=ax.get("w"))
    w_right = get_scratch_width(x_right, y_right, local_idx_right, axes=ax.get("w"))
    w = float(np.sum([w_left, w_right]))

    # 6. Macroscopic continuous properties (Global operations)
    h_r = get_residual_depth(x, y, axes=ax.get("h_r"))

    features = {
        "h_p": h_p,
        "h_r": h_r,
        "w": w,
    }

    if not get_additional_features:
        return features

    w_p_left = get_pile_up_width(
        x_left, y_left, local_idx_left, threshold=2e-3, axes=ax.get("w_p")
    )
    w_p_right = get_pile_up_width(
        x_right, y_right, local_idx_right, threshold=2e-3, axes=ax.get("w_p")
    )
    w_p = _safe_nanmean(w_p_left, w_p_right)

    # Curvature operates on global arrays; pass global indices strictly
    k_p_left, _, _ = calculate_peak_curvature(
        x,
        y,
        peak_idx=peak_idx_left,
        num_span=10,
        max_degree=6,
        find_max=True,
        axes=ax.get("k_p"),
    )
    k_p_right, _, _ = calculate_peak_curvature(
        x,
        y,
        peak_idx=peak_idx_right,
        num_span=10,
        max_degree=6,
        find_max=True,
        axes=ax.get("k_p"),
    )
    k_p = _safe_nanmean(k_p_left, k_p_right)

    A_p_left = get_pile_up_area(x_left, y_left, axes=ax.get("A_p"))
    A_p_right = get_pile_up_area(x_right, y_right, axes=ax.get("A_p"))
    A_p = float(np.sum([A_p_left, A_p_right]))

    A_g = get_groove_area(x, y, axes=ax.get("A_g"))

    k_g, _, _ = calculate_peak_curvature(
        x,
        y,
        peak_idx=idx_groove,
        num_span=15,
        max_degree=4,
        find_max=False,
        axes=ax.get("k_g"),
    )

    features.update(
        {
            "w_p": w_p,
            "A_p": A_p,
            "A_g": A_g,
            "k_p": k_p,
            "k_g": k_g,
        }
    )
    return features


def calc_xy_peak_indexes(
    x: np.ndarray,
    y: np.ndarray,
    noise_floor_fraction: float = 0.03,
    axes=None,
) -> tuple[int, int]:
    """Identifies scratch test pile-up peaks anchored to the central groove.

    Args:
        x (np.ndarray): Input x-coordinate array.
        y (np.ndarray): Input y-coordinate array.
        noise_floor_fraction (float): Minimum prominence required to register
            as a valid peak, expressed as a fraction of the total profile
            relief (max y - min y).
        axes (matplotlib.axes.Axes | None): Optional axes instance for
            diagnostic verification.

    Returns:
        tuple[int, int]: Indices of the left and right pile-up peaks.

    Raises:
        ValueError: If fewer than two peaks are found, or the groove anchor
            is not bracketed by peaks on both sides.
    """
    # 1. Locate the central excavation anchor.
    # We apply a heavy Gaussian smoothing strictly to resolve the macroscopic
    # valley, protecting the anchor against localized downward sensor dropouts.
    y_macro = gaussian_filter1d(y, sigma=3.0)
    idx_groove = int(np.argmin(y_macro))

    # 2. Establish a dynamic, scale-invariant noise floor.
    profile_relief = float(np.ptp(y))  # Peak-to-peak amplitude (max - min)
    min_prom = noise_floor_fraction * profile_relief

    # 3. Detect all topological peaks and extract their exact prominences.
    peaks, props = find_peaks(y, prominence=min_prom)
    prominences = props["prominences"]

    if len(peaks) < 2:
        raise ValueError(
            f"Profile topography resolved {len(peaks)} peaks; a minimum of 2 "
            "are required to bracket a scratch groove. Inspect surface leveling."
        )

    # 4. Bipartite spatial partitioning relative to the groove anchor.
    left_mask = peaks < idx_groove
    right_mask = peaks > idx_groove

    if not np.any(left_mask) or not np.any(right_mask):
        raise ValueError(
            f"Groove anchor at index {idx_groove} (x={x[idx_groove]:.2f}) is "
            "not bracketed by peaks on both sides. The scan may have terminated "
            "prematurely inside the scratch path."
        )

    left_candidates = peaks[left_mask]
    left_prominences = prominences[left_mask]

    right_candidates = peaks[right_mask]
    right_prominences = prominences[right_mask]

    # 5. Select the primary topological pile-up on each flank.
    peak_idx_1 = int(left_candidates[np.argmax(left_prominences)])
    peak_idx_2 = int(right_candidates[np.argmax(right_prominences)])

    # Ensure consistent left-to-right index return order
    if peak_idx_1 > peak_idx_2:
        peak_idx_1, peak_idx_2 = peak_idx_2, peak_idx_1

    if axes is not None:
        distance = abs(x[peak_idx_2] - x[peak_idx_1])
        axes.scatter(
            x[peaks],
            y[peaks],
            label="Candidate surface peaks",
            color="gray",
            alpha=0.5,
            zorder=5,
        )
        axes.axvline(
            x[idx_groove],
            color="black",
            linestyle=":",
            label="Inferred groove anchor",
            alpha=0.6,
        )
        axes.scatter(
            [x[peak_idx_1], x[peak_idx_2]],
            [y[peak_idx_1], y[peak_idx_2]],
            label=f"Pile-up crests (Δx={distance:.2f})",
            color="blue",
            marker="^",
            s=80,
            zorder=6,
        )
        axes.legend(fontsize="small")

    return peak_idx_1, peak_idx_2


def get_data_from_yz_profile(yz_profile, axes_dict=None):
    """Extracts frontal pile-up metrics from a 1D yz scratch profile.

    Args:
        yz_profile (tuple[np.ndarray, np.ndarray]): Global coordinate tuple
            (y, z).
        axes_dict (dict | None): Dictionary mapping feature keys to
            Matplotlib Axes instances.

    Returns:
        dict[str, float]: h_fp, w_fp, w_fp_d, A_fp, k_fp.
    """
    ax = axes_dict if axes_dict is not None else {}
    y, z = yz_profile

    mask = z >= 2.0
    y = y[mask]
    z = z[mask]

    h_fp = get_pile_up_height(z, y, axes=ax.get("h_fp"))
    w_fp = get_pile_up_width(z, y, threshold=2e-3, axes=ax.get("w_fp"))
    w_fp_d = get_distance_to_frontal_pile_up_peak(z, y, axes=ax.get("w_fp_d"))
    A_fp = get_pile_up_area(z, y, axes=ax.get("A_fp"))
    k_fp, _, _ = calculate_peak_curvature(
        z,
        y,
        num_span=10,
        max_degree=6,
        find_max=True,
        axes=ax.get("k_fp"),
    )
    return {
        "h_fp": h_fp,
        "w_fp": w_fp,
        "w_fp_d": w_fp_d,
        "A_fp": A_fp,
        "k_fp": k_fp,
    }


def get_pile_up_height(x, y, peak_idx=None, axes=None):
    """Returns the pile-up height of a given profile.

    For the xy profile, this calculates one pile-up height -- to get both,
    split the profile and call separately. For the yz profile, this
    calculates the frontal pile-up height.

    Args:
        x (np.ndarray): 1D array of x-coordinates. If the profile is
            yz-oriented, this should be the z-coordinates.
        y (np.ndarray): 1D array of y-coordinates corresponding to the
            profile.
        peak_idx (int, optional): Index of the peak to calculate the height
            at. Defaults to the index of the profile's maximum.
        axes (matplotlib.axes.Axes, optional): If provided, plots the
            identified pile-up height on the given axes.

    Returns:
        float: The pile-up height, defined as the y-value at ``peak_idx``
        (or the profile maximum if ``peak_idx`` is None).
    """
    h_p = y[peak_idx] if peak_idx is not None else np.nanmax(y)
    if axes is not None:
        (
            axes.scatter(x[peak_idx], h_p, label=f"h_p={h_p:.5f}", zorder=5)
            if peak_idx is not None
            else axes.scatter(x[np.nanargmax(y)], h_p, label=f"h_p={h_p:.5f}", zorder=5)
        )
    return h_p


def get_scratch_width(x, y, peak_idx=None, axes=None):
    """Returns the sub-pixel x-position of one pile-up peak from centre.

    One flank's half-width: the caller sums the left and right calls to get
    the full scratch width.

    Args:
        x (np.ndarray): 1D array of x-coordinates.
        y (np.ndarray): 1D array of y-coordinates corresponding to the
            profile.
        peak_idx (int, optional): Index of the peak. Defaults to the index
            of the profile's maximum.
        axes (matplotlib.axes.Axes, optional): If provided, plots the
            identified peak position on the given axes.

    Returns:
        float: Absolute x-position of the peak.
    """
    if peak_idx is not None:
        idx = peak_idx
    else:
        idx = int(np.nanargmax(y))

    # Find the true peak x by interpolating between the two highest neighbours.
    # This handles flat-topped (mesh-discretised) peaks naturally: if the top
    # is flat, the two equal neighbours give the midpoint; if it is sharp, the
    # one dominant neighbour pulls the result toward itself.
    idx_left = max(idx - 1, 0)
    idx_right = min(idx + 1, len(y) - 1)

    candidates = [idx_left, idx, idx_right]
    # Pick the two highest among the three
    two_highest = sorted(candidates, key=lambda i: y[i], reverse=True)[:2]
    i1, i2 = sorted(two_highest)  # keep left-to-right order

    y1, y2 = y[i1], y[i2]
    x1, x2 = x[i1], x[i2]

    # Weighted average: higher point contributes more
    x_peak = float(np.average([x1, x2], weights=[y1, y2]))

    if axes is not None:
        axes.scatter(
            x_peak, y[idx], label=f"Scratch width={np.abs(x_peak):.5f}", zorder=5, s=1
        )

    return np.abs(x_peak)


def get_residual_depth(x, y, axes=None):
    """Returns the residual depth of a given profile, defined as the minimum y-value.

    For the xy profile, this is the groove depth at a given z-coordinate. For
    the yz profile, this is the residual scratch depth.

    Args:
        x (np.ndarray): 1D array of x-coordinates. If the profile is
            yz-oriented, this should be the z-coordinates.
        y (np.ndarray): 1D array of y-coordinates corresponding to the
            profile.
        axes (matplotlib.axes.Axes, optional): If provided, plots the
            identified residual depth on the given axes.

    Returns:
        float: The residual depth, defined as the minimum y-value of the
        profile.
    """
    h_r = np.nanmin(y)
    if axes is not None:
        axes.scatter(
            x[np.nanargmin(y)],
            h_r,
            label=f"h_r={h_r:.5f}",
            zorder=5,
        )
    return np.abs(h_r)


def get_pile_up_area(x, y, axes=None):
    """Returns the pile-up area of a given profile, defined as the area above the baseline (y=0).

    For the xy profile, this calculates the area of the pile-up region at a
    given z-coordinate. For the yz profile, this calculates the frontal
    pile-up area.

    Args:
        x (np.ndarray): 1D array of x-coordinates. If the profile is
            yz-oriented, this should be the z-coordinates.
        y (np.ndarray): 1D array of y-coordinates corresponding to the
            profile.
        axes (matplotlib.axes.Axes, optional): If provided, plots the
            identified pile-up area on the given axes.

    Returns:
        float: The pile-up area, calculated as the integral of the profile
        above the baseline (y=0).
    """
    mask_for_pile_up_region = y >= 0
    A_p = np.trapezoid(y[mask_for_pile_up_region], x[mask_for_pile_up_region])
    if axes is not None:
        axes.fill_between(
            x[mask_for_pile_up_region],
            y[mask_for_pile_up_region],
            alpha=0.3,
            label=f"Pile-up Area={A_p:.5f}",
        )
    return A_p


def get_groove_area(x, y, axes=None):
    """Returns the groove area of a given profile, defined as the area below the baseline (y=0).

    For the xy profile, this calculates the area of the groove region at a
    given z-coordinate. For the yz profile, this calculates the groove area
    in the yz plane.

    Args:
        x (np.ndarray): 1D array of x-coordinates. If the profile is
            yz-oriented, this should be the z-coordinates.
        y (np.ndarray): 1D array of y-coordinates corresponding to the
            profile.
        axes (matplotlib.axes.Axes, optional): If provided, plots the
            identified groove area on the given axes.

    Returns:
        float: The groove area, calculated as the integral of the profile
        below the baseline (y=0).
    """
    mask_for_groove_up_region = y <= 0
    A_g = np.trapezoid(y[mask_for_groove_up_region], x[mask_for_groove_up_region])
    if axes is not None:
        axes.fill_between(
            x[mask_for_groove_up_region],
            y[mask_for_groove_up_region],
            alpha=0.3,
            label=f"Groove Area={A_g:.5f}",
        )
    return np.abs(A_g)


def get_pile_up_width(x, y, peak_idx=None, threshold=1e-3, axes=None):
    """Returns the pile-up width of a given profile.

    Defined as the distance between the points where the profile crosses
    ``threshold`` on either side of the peak. For the xy profile, this
    calculates the width of the pile-up region at a given z-coordinate. For
    the yz profile, this calculates the width of the frontal pile-up region.

    Args:
        x (np.ndarray): 1D array of x-coordinates. If the profile is
            yz-oriented, this should be the z-coordinates.
        y (np.ndarray): 1D array of y-coordinates corresponding to the
            profile.
        peak_idx (int, optional): Index of the peak. Defaults to the index
            of the profile's maximum.
        threshold (float): Small threshold value determining the effective
            edge of the pile-up region; accounts for noise and gives a more
            robust width measurement.
        axes (matplotlib.axes.Axes, optional): If provided, plots the
            identified pile-up width on the given axes.

    Returns:
        float: The pile-up width, calculated as the distance between the
        points where the profile crosses ``threshold`` on either side of the
        peak. NaN if the profile never crosses the threshold on either side.
    """
    peak_idx = np.nanargmax(y) if peak_idx is None else peak_idx
    y_left = y[: peak_idx + 1][::-1]
    x_left = x[: peak_idx + 1][::-1]

    y_right = y[peak_idx:]
    x_right = x[peak_idx:]

    def _find_crossing(x_arr, y_arr, threshold):
        below_threshold_indices = np.where(y_arr < threshold)[0]
        if len(below_threshold_indices) == 0:
            return np.nan

        idx_below = below_threshold_indices[0]
        idx_above = idx_below - 1

        x1, y1 = x_arr[idx_above], y_arr[idx_above]
        x2, y2 = x_arr[idx_below], y_arr[idx_below]

        x_cross = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
        return x_cross

    x_start = _find_crossing(x_left, y_left, threshold)
    x_end = _find_crossing(x_right, y_right, threshold)

    if axes is not None:
        axes.scatter(
            [x_start, x_end],
            [threshold, threshold],
            label=f"Pile-up width={np.abs(x_start - x_end):.5f}",
            zorder=5,
        )

    return np.abs(x_start - x_end)


def get_pile_up_volume(X, Y, Z):
    """Integrates the pile-up region (Y >= 0) over the full 3-D grid.

    Args:
        X (np.ndarray): 2-D across-scratch coordinate grid.
        Y (np.ndarray): 2-D height grid.
        Z (np.ndarray): 2-D along-scratch coordinate grid.

    Returns:
        float: The pile-up volume.
    """
    Y_pileup = np.where(Y >= 0, Y, 0)
    slice_pileup_areas = np.trapezoid(Y_pileup, x=X, axis=0)
    z_coords = Z[0, :]
    return np.trapezoid(slice_pileup_areas, x=z_coords)


def get_groove_volume(X, Y, Z):
    """Integrates the groove region (Y <= 0) over the full 3-D grid.

    Args:
        X (np.ndarray): 2-D across-scratch coordinate grid.
        Y (np.ndarray): 2-D height grid.
        Z (np.ndarray): 2-D along-scratch coordinate grid.

    Returns:
        float: The groove volume (positive).
    """
    Y_groove = np.where(Y <= 0, Y, 0)
    slice_groove_areas = np.trapezoid(Y_groove, x=X, axis=0)
    z_coords = Z[0, :]
    return np.abs(np.trapezoid(slice_groove_areas, x=z_coords))


def get_distance_to_frontal_pile_up_peak(x, y, axes=None):
    """Returns the distance to the frontal pile-up peak of a given yz profile.

    Defined as the z-coordinate corresponding to the maximum y-value.

    Args:
        x (np.ndarray): 1D array of z-coordinates corresponding to the yz
            profile.
        y (np.ndarray): 1D array of y-coordinates corresponding to the yz
            profile.
        axes (matplotlib.axes.Axes, optional): If provided, plots the
            identified peak on the given axes.

    Returns:
        float: The z-coordinate of the frontal pile-up peak.
    """
    idx_max = np.nanargmax(y)
    x_peak = x[idx_max]

    if axes is not None:
        axes.scatter(
            x_peak,
            y[idx_max],
            label=f"Distance to frontal pile-up peak={x_peak:.5f}",
            zorder=5,
        )
    return x_peak


def calculate_peak_curvature(
    x: np.ndarray,
    y: np.ndarray,
    peak_idx: int | None = None,
    num_span: int = 10,
    max_degree: int = 5,
    find_max: bool = True,
    axes=None,
) -> tuple[float, float, Polynomial]:
    """Fits a local polynomial to an extremum and calculates analytical curvature.

    Args:
        x (np.ndarray): Input x-coordinate array.
        y (np.ndarray): Input y-coordinate array.
        peak_idx (int | None): Explicit index of the target extremum. If
            None, it is inferred.
        num_span (int): Number of indices to include on either side of the
            peak.
        max_degree (int): Maximum polynomial degree to test (selected via
            BIC).
        find_max (bool): If True, targets a local maximum; if False, targets
            a local minimum.
        axes (matplotlib.axes.Axes | None): Optional axes object for
            diagnostic plotting.

    Returns:
        tuple[float, float, Polynomial]: The curvature at the extremum, the
        fitted polynomial's R², and the fitted polynomial itself.

    Raises:
        ValueError: If fewer than 4 points fall in the local window.
    """
    # 1. Normalize orientation: magnitude of curvature is invariant to sign flips
    y_work = y if find_max else -y

    if peak_idx is None:
        peak_idx = int(np.nanargmax(y_work))

    # 2. Extract local window with strict boundary clamping
    start = max(0, peak_idx - num_span)
    end = min(len(x), peak_idx + num_span + 1)

    x_local = x[start:end]
    y_local = y_work[start:end]

    if len(x_local) < 4:
        raise ValueError("Insufficient data points in window to fit a curve.")

    # 3. Model selection via Bayesian Information Criterion (BIC)
    best_bic = np.inf
    best_poly = None
    n = len(x_local)

    for deg in range(2, max_degree + 1):
        poly = Polynomial.fit(x_local, y_local, deg=deg)
        residuals = y_local - poly(x_local)
        rss = np.sum(residuals**2)

        # Guard against log(0) in machine-exact synthetic fits
        rss = max(rss, 1e-16)

        k = deg + 1  # Number of estimated parameters
        bic = n * np.log(rss / n) + k * np.log(n)

        if bic < best_bic:
            best_bic = bic
            best_poly = poly

    # 4. Analytical vertex resolution
    deriv1 = best_poly.deriv(1)
    deriv2 = best_poly.deriv(2)

    roots = deriv1.roots()
    real_roots = roots[np.isreal(roots)].real

    # Filter roots situated strictly within the local window domain
    domain_roots = real_roots[(real_roots >= x_local[0]) & (real_roots <= x_local[-1])]

    if len(domain_roots) > 0:
        # Select the continuous root closest to the discrete observation
        x_star = domain_roots[np.argmin(np.abs(domain_roots - x[peak_idx]))]
    else:
        # Fallback to discrete coordinate if no real root exists in domain
        x_star = x[peak_idx]

    d1 = deriv1(x_star)
    d2 = deriv2(x_star)

    curvature = float(np.abs(d2) / (1.0 + d1**2) ** 1.5)

    # Calculate standard R2 of the selected model for user reporting
    ss_res = np.sum((y_local - best_poly(x_local)) ** 2)
    ss_tot = np.sum((y_local - np.mean(y_local)) ** 2)
    r2 = float(1.0 - (ss_res / max(ss_tot, 1e-16)))

    # 5. Diagnostic plotting
    if axes is not None:
        x_dense = np.linspace(x_local.min(), x_local.max(), 200)
        y_dense = best_poly(x_dense)
        axes.plot(
            x_dense,
            y_dense if find_max else -y_dense,
            linestyle="--",
            label=f"Fit (Deg={best_poly.degree()}, R²={r2:.3f}, κ={curvature:.2e})",
            zorder=4,
        )
        axes.legend(fontsize="small")

    return curvature, r2, best_poly