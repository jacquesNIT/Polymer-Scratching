import os

# === Scratch geometry ===
# Used by: data_processing.py, scratch_simulation_helpers.py,
# scratch_experimental_helpers.py, and several plotting/*.py scripts.
target_shape = (80, 420)
scratch_length = 2.0  # mm.
scratch_domain_width = 0.6  # mm. Full width.
scratch_domain_length = 2.5  # mm. Full length of the domain to be mapped to.
prescribed_max_depth = 0.04  # mm. The maximum depth of the scratch.


def prescribed_depth(z):
    """Indenter depth (mm) at along-scratch position ``z``.

    The simulations are displacement-driven: the indenter descends on a linear
    ramp from zero at z = 0 to ``prescribed_max_depth`` at ``scratch_length``,
    and the normal force is the response.  So unlike every other feature, this
    depth is *known* rather than measured off the residual topography.
    """
    return prescribed_max_depth * z / scratch_length


# === Experimental topography denoising ===
# Used by: scratch_experimental_helpers.py
# despike_size: median-filter window (pixels) applied to the raw ROI to reject
#   isolated profilometry spikes before resampling.
# smooth_sigma_(x|z): anisotropic Gaussian sigmas (pixels on the resampled
#   target grid). x is across the scratch, z is along it; the scratch is close
#   to translationally invariant along z, so we smooth more there and keep the
#   groove cross-section (x) sharp.
exp_despike_size = 3
exp_smooth_sigma_x = 1.0
exp_smooth_sigma_z = 2.0

# === Maximum-depth (feature-point) localisation along the scratch ===
# Used by: scratch_experimental_helpers.py, plotting/compare_scan_sources.py,
# plotting/experimental_cross_section_profile.py
# depth_band_frac: fraction of x-rows, centred on the deepest row, medianed into
#   a robust along-scratch depth profile (rejects per-row spikes/voids).
# depth_smooth_mm: Gaussian smoothing length (mm) applied to that profile so the
#   minimum is a stable low-frequency feature rather than a noisy pixel.
# depth_tol_frac: the max-depth location is the centre of the region within this
#   fraction of the deepest value -- the trough for a sharp groove, the plateau
#   centre for a flat constant-load bottom.
exp_depth_band_frac = 0.05
exp_depth_smooth_mm = 0.06
exp_depth_tol_frac = 0.03

# === Keyence VR height-CSV orientation detection ===
# Used by: scratch_experimental_helpers.py
# Its field of view is ~3x wider across the scratch than the .bcrf strip, so
# specimen tilt dominates the raw row means and the groove has to be found as
# a *local* depression instead.
# csv_detrend_mm: Gaussian length (mm, across the scratch) whose smoothed row
#   profile is subtracted as the tilt baseline. Must be well above the groove
#   width so the groove survives the subtraction as a residual dip.
# csv_groove_band_mm: half-width (mm) of the row band taken as "inside the
#   groove"; the surface reference is read from beyond 3x this distance.
exp_csv_detrend_mm = 0.25
exp_csv_groove_band_mm = 0.1
# csv_strip_half_width_mm: the CSV is cropped to a strip of this half-width around
#   the groove before being handed downstream, matching the .bcrf's 0.604 mm field
#   of view. Beyond geometric comparability this is what keeps
#   _find_max_depth_location working: its deepest-row search is a plain row-mean
#   argmin, which the full-width CSV's tilt sends to a frame edge.
exp_csv_strip_half_width_mm = 0.3

# === Specimen tilt removal (levelling) ===
# Used by: scratch_experimental_helpers.py
# The specimen never sits perfectly normal to the optical axis, so every scan
# carries a slope: measured across the processed ROI it reaches ~3 um across the
# scratch and ~9 um along it, against residual depths of only 12-33 um. Left in,
# it biases the deepest-row and deepest-column searches and -- because h_r, A_p
# and A_g are all defined against y = 0 -- shifts the baseline those features are
# read from. A plane is fitted to the undisturbed surface either side of the
# groove and subtracted, which removes the slope and puts y = 0 on the original
# surface: the same datum the simulation grids already use.
# level_band_mm: half-width (mm) of the band around the groove centre held out of
#   the fit. Must clear the groove *and* its pile-up shoulders; at 0.15 mm about
#   half the 0.60 mm field of view is still fitted, on both scan sources.
# level_clip_sigma: residual cut (in robust MAD sigmas) applied when refitting, to
#   shed debris, voids and profilometry spikes that survive the band exclusion.
# level_iters: number of clip-and-refit passes.
exp_level_band_mm = 0.15
exp_level_clip_sigma = 3.0
exp_level_iters = 3

# === Topography sources ===
# Used by: data_processing.py, scratch_experimental_helpers.py,
# plotting/ood_analysis.py
# Which topography sources each scratch test is processed from. Both are scans of
# the same groove, so a test listed here twice yields two specimens whose feature
# rows differ only by how the surface was measured -- the id carries a "_bcrf" or
# "_csv" suffix to tell them apart. Trim this list to process one source only.
exp_topography_sources = ["bcrf", "csv"]

# === Optuna search ===
# Used by: pipeline/optuna_search.py
n_trials = int(os.environ.get("ML_N_TRIALS", 30))
cv = int(os.environ.get("ML_CV", 3))
n_jobs = 1
scoring = "r2"
optuna_storage_dir = "optuna_studies/"
direction = "maximize"  # direction of optimisation of validation score


def optuna_storage_url(study_name):
    """Return a per-study SQLite storage URL.

    Each SLURM array task already has a unique ``study_name`` (it encodes
    model, dataset, target, feature_order, and seed), so giving every study
    its own SQLite file -- instead of sharing one file across all parallel
    array tasks -- avoids the writer contention ("database is locked") that a
    single shared file runs into under SLURM array concurrency.
    """
    return f"sqlite:///{optuna_storage_dir}{study_name}.sqlite3"


optuna_random_state = 42

# === Along-scratch sampling ===
# Used by: data_processing.py, data_processing/data_loader.py
interval_min = 1.0
interval_max = scratch_length


# === Data split sizes ===
# Used by: training_pipeline.py, analysis/sensitivity_analysis.py
simulation_data_test_size = 0.2
experimental_data_test_size = 1.0


# === Data paths ===
# Used by: data_processing.py, data_processing/data_loader.py,
# data_processing/make_dimensionless_groups.py, analysis/*.py,
# plotting/sr_equation_figure.py, training_pipeline.py
tensile_data_path = "data/tensile_data/"
scratch_data_path = "data/Scratch_tests/"
raw_path = "data/raw/"
processed_path = "data/processed/"
dimensionless_path = "data/dimensionless/"


# === Data-splitting seed generation ===
# Used by: main.py
random_state = 42
num_seeds = 10
low = 1
high = 100000


def feature_order_path(base_path, feature_order):
    """Nest ``base_path`` under a feature_order-specific subdirectory.

    Each feature 'level' generates and trains on its own copy of the data so
    runs do not clobber each other.  Returns a path ending in a separator so it
    can be string-concatenated with a filename or passed to ``os.path.join``.

    Example: ``feature_order_path("data/dimensionless/", 3)`` ->
    ``"data/dimensionless/feature_order_3/"``.

    Used by: training_pipeline.py, data_processing.py,
    analysis/sensitivity_analysis.py, analysis/OOD_analysis.py,
    plotting/sr_equation_figure.py
    """
    return os.path.join(base_path, f"feature_order_{feature_order}") + os.sep