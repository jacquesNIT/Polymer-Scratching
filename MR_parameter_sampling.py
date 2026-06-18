import os
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

# Per-parameter configuration of the variables
POLYMER_PARAMS = {
    "C10": {                        # First MR parameter [MPa] 
        "range": (0.1, 2.0),
        "method": "sobol",
        "round": 2 },
    "r": {                          # C01 / C10 ratio [-]
        "range": (0.05, 0.25),
        "method": "sobol",          
        "round": 3 },
    "r_K": {                        # K0 / mu0 ratio [-] compressibility
        "range": (10.0, 100.0),
        "method": "sobol",          
        "round": 1 },
    "mu": {                         # Friction coefficient [-]
        "range": (0.3, 0.45),
        "method": "sobol",
        "round": 3 },
}

# Fixed parameter
POLYMER_FIXED = {"rho": 1.2e-9}

# Rounding 
DERIVED_ROUND = {"C01": 4, "D1": 5, "nu": 4}

# 1D samplers
def _sample_1d(method, n_samples, lo, hi, seed=42, **kwargs):
    # Return an array of 'n_samples' values in [lo, hi] using 'method'

    if method == "sobol":
        raw = qmc.Sobol(d=1, scramble=True, seed=seed).random(n=n_samples).ravel()
        return lo + raw * (hi - lo)

    if method == "halton":
        raw = qmc.Halton(d=1, scramble=True, seed=seed).random(n=n_samples).ravel()
        return lo + raw * (hi - lo)

    if method == "lhs":
        raw = qmc.LatinHypercube(d=1, seed=seed).random(n=n_samples).ravel()
        return lo + raw * (hi - lo)

    if method == "random":
        return np.random.default_rng(seed).uniform(lo, hi, size=n_samples)

    if method == "grid":
        # n_points evenly spaced values
        n_points = kwargs.get("n_points", n_samples)
        grid = np.linspace(lo, hi, n_points)
        return np.resize(grid, n_samples)

    if method == "exponential":
        # Log-uniform: good when lo and hi span orders of magnitude
        if lo <= 0:
            raise ValueError("exponential requires strictly positive bounds (got lo=%g)" % lo)
        raw = qmc.Sobol(d=1, scramble=True, seed=seed).random(n=n_samples).ravel()
        return lo * (hi / lo) ** raw

    if method == "discrete":
        # Map a Halton sequence onto discrete levels
        levels = np.asarray(kwargs["levels"])
        raw = qmc.Halton(d=1, scramble=True, seed=seed).random(n=n_samples).ravel()
        idx = np.clip((raw * len(levels)).astype(int), 0, len(levels) - 1)
        return levels[idx]

    raise ValueError("Unknown sampling method: %s" % method)

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
    param_config=None,
    fixed_params=None,
    output_dir="material_parameters",
    tag="polymer",                    # prefix for output filenames
    seed_offset=0 ):                    # added to per-param seed for reproducibility

    if param_config is None:
        param_config = POLYMER_PARAMS
    if fixed_params is None:
        fixed_params = POLYMER_FIXED

    #  1. Sample the independent design variables
    columns = {}
    for i, (name, cfg) in enumerate(param_config.items()):
        lo, hi = cfg["range"]
        method = cfg["method"]
        # Pass any extra keys (n_points, levels ...) as kwargs
        extra = {k: v for k, v in cfg.items() if k not in ("range", "method", "round")}
        columns[name] = _sample_1d(
            method, n_samples, lo, hi,
            seed=42 + seed_offset + i,   # unique seed per parameter
            **extra,
        )

    df = pd.DataFrame(columns)

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

    custom_config = {
        "C10": {
            "range": (0.1, 2.0),
            "method": "sobol",
            "round": 2 },
        "r": {
            "range": (0.05, 0.25),
            "method": "lhs",            
            "round": 3 },
        "r_K": {
            "range": (10.0, 100.0),
            "method": "lhs",           
            "round": 1 },
        "mu": {
            "range": (0.3, 0.45),
            "method": "discrete",
            "levels": [0.30, 0.33, 0.36, 0.39, 0.42, 0.45],
            "round": 3 },
    }
    generate_parameters(n_samples=1024, param_config=custom_config, tag="polymer_MR")