import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import qmc

# ----------------------------------------------------------------------
# Mooney-Rivlin parameter sampling
#
# Strain energy:
#   W = C10 (I1_bar - 3) + C01 (I2_bar - 3) + (1/D1) (J_el - 1)^2
#
# Instead of sampling C01 and D1 freely (which produces non-physical combinations), we sample physically meaningful ratios and derive C01, D1 and nu from them:
#   - shear modulus   mu0 = 2 (C10 + C01)
#   - bulk modulus    K0  = 2 / D1
#   - Poisson ratio   nu  = (3 K0 - 2 mu0) / (2 (3 K0 + mu0))
#
# Variables sampled:
#   C10  : first MR constant             [MPa]
#   r    = C01 / C10                     [-]
#   r_K  = K0 / mu0                      [-]
#   mu   : Coulomb friction coefficient  [-]
#
# Variables derived :
#   C01 = r * C10                                [MPa]
#   D1  = 1 / (r_K * (C10 + C01))                [1/MPa]
#   nu  = (3 r_K - 2) / (2 (3 r_K + 1))          [-]
#
# r_K in [10, 100]  <=>  nu in [0.45, 0.499]
# C10 in [0.1, 2.0] <=> E in [0.6, 15]
#
# ----------------------------------------------------------------------

# Per-parameter configuration of the variables.
POLYMER_PARAMS = {
    "C10": {                        # First MR parameter [MPa]
        "range": (0.1, 2.0),
        "round": 2 },
    "r": {                          # C01 / C10 ratio [-]
        "range": (0.05, 0.25),
        "round": 3 },
    "r_K": {                        # K0 / mu0 ratio [-] compressibility
        "range": (10.0, 100.0),
        "round": 1 },
    "mu": {                         # Friction coefficient [-]
        "range": (0.3, 0.45),
        "round": 3 },
}

# Fixed parameter
POLYMER_FIXED = {"rho": 1.2e-9}

# Rounding
DERIVED_ROUND = {"C01": 4, "D1": 5, "nu": 4}

# Joint sampler
def _sample_unit_cube(method, n_samples, n_dims, seed=42):
    # Return an (n_samples, n_dims) array in [0, 1]^d from a single joint sequence.

    if method == "sobol":
        return qmc.Sobol(d=n_dims, scramble=True, seed=seed).random(n=n_samples)

    if method == "halton":
        return qmc.Halton(d=n_dims, scramble=True, seed=seed).random(n=n_samples)

    if method == "lhs":
        return qmc.LatinHypercube(d=n_dims, seed=seed).random(n=n_samples)

    if method == "random":
        return np.random.default_rng(seed).uniform(0.0, 1.0, size=(n_samples, n_dims))

    raise ValueError("Unknown joint sampling method: %s" % method)

# Map a [0, 1]^d sample onto the physical ranges (and discrete levels) of each variable
def _map_to_ranges(unit, param_config):
    names = list(param_config.keys())
    sample = np.empty_like(unit)
    for i, name in enumerate(names):
        cfg = param_config[name]
        if "levels" in cfg:
            # Discrete dimension: bin the joint coordinate onto the allowed levels
            levels = np.asarray(cfg["levels"], dtype=float)
            idx = np.clip((unit[:, i] * len(levels)).astype(int), 0, len(levels) - 1)
            sample[:, i] = levels[idx]
        elif cfg.get("log", False):
            # Log-uniform scaling (requires strictly positive bounds)
            lo, hi = cfg["range"]
            if lo <= 0:
                raise ValueError("log scaling requires lo > 0 (got %g for '%s')" % (lo, name))
            sample[:, i] = lo * (hi / lo) ** unit[:, i]
        else:
            # Linear scaling to [lo, hi]
            lo, hi = cfg["range"]
            sample[:, i] = lo + unit[:, i] * (hi - lo)
    return sample

# Grid sampler: Cartesian product (does not use a QMC sequence)
def _grid_sample(param_config, grid_points):
    if grid_points is None:
        raise ValueError(
            "method='grid' requires grid_points={'C10': n, 'r': n, ...}"
        )
    grids = []
    for name, cfg in param_config.items():
        if "levels" in cfg:
            grids.append(np.asarray(cfg["levels"], dtype=float))
        else:
            if name not in grid_points:
                raise ValueError("Missing grid_points entry for parameter '%s'" % name)
            lo, hi = cfg["range"]
            grids.append(np.linspace(lo, hi, grid_points[name]))
    sample = np.array(list(itertools.product(*grids)))
    print("Generated %d grid points." % sample.shape[0])
    return sample

#  Physical coupling: derive C01, D1, nu from the sampled design variables
def _derive_mooney_rivlin(df):
    df["C01"] = df["r"] * df["C10"]
    S = df["C10"] + df["C01"]                                   # = mu0 / 2
    df["D1"] = 1.0 / (df["r_K"] * S)                            # 2 / K0
    df["nu"] = (3.0 * df["r_K"] - 2.0) / (2.0 * (3.0 * df["r_K"] + 1.0))
    return df

#  Parameter generator
def generate_parameters(
    n_samples=10,
    method="sobol",                   # single joint sampler for every variable
    param_config=None,
    fixed_params=None,
    output_dir="material_parameters",
    tag="polymer",                    # prefix for output filenames
    seed=42,                          # one seed for the whole joint sequence
    grid_points=None ):               # only used when method == "grid"

    if param_config is None:
        param_config = POLYMER_PARAMS
    if fixed_params is None:
        fixed_params = POLYMER_FIXED

    names = list(param_config.keys())
    dim = len(names)

    #  1. Sample the design variables from joint sequence, then map to ranges
    if method == "grid":
        sample = _grid_sample(param_config, grid_points)
    else:
        unit = _sample_unit_cube(method, n_samples, dim, seed=seed)
        sample = _map_to_ranges(unit, param_config)

    df = pd.DataFrame(sample, columns=names)

    #  2. Round the sampled variables first, so derived values stay consistent
    sampled_round = {name: cfg["round"] for name, cfg in param_config.items() if "round" in cfg}
    if sampled_round:
        df = df.round(sampled_round)

    #  3. Derive the physically-coupled Mooney-Rivlin parameters
    df = _derive_mooney_rivlin(df)
    derived_round = {k: v for k, v in DERIVED_ROUND.items() if k in df.columns}
    if derived_round:
        df = df.round(derived_round)

    #  4. id + fixed columns right after id
    df.insert(0, "id", ["%05d" % i for i in range(1, len(df) + 1)])
    for col_idx, (col_name, col_val) in enumerate(fixed_params.items(), start=1):
        df.insert(col_idx, col_name, col_val)

    #  5. Column order: material-centric, raw ratios kept as trailing traceability
    preferred = ["id", "rho", "nu", "C10", "C01", "D1", "mu", "r", "r_K"]
    ordered = [c for c in preferred if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]

    #  6. Save
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, tag + "_material_parameter_sweep")
    df.to_csv(base + ".csv", index=False)

    records = df.to_dict(orient="records")
    with open(base + ".py", "w") as f:
        f.write("parameters = [\n")
        for row in records:
            f.write("    %s,\n" % row)
        f.write("]\n")

    print("Saved %d samples -> %s.csv" % (len(df), base))
    return records

if __name__ == "__main__":

    # Continuous joint Sobol sampling (default)
    generate_parameters(n_samples=1024, method="sobol", tag="polymer_MR")

    # Variant: mu drawn on discrete levels while the remaining variables keep the joint Sobol coverage.
    custom_config = {
        "C10": {"range": (0.1, 2.0),    "round": 2 },
        "r":   {"range": (0.05, 0.25),  "round": 3 },
        "r_K": {"range": (10.0, 100.0), "round": 1 },
        "mu":  {"range": (0.3, 0.45),
                "levels": [0.30, 0.33, 0.36, 0.39, 0.42, 0.45],
                "round": 3 },
    }
    generate_parameters(n_samples=1024, method="sobol",
                        param_config=custom_config, tag="polymer_MR_discrete_mu")