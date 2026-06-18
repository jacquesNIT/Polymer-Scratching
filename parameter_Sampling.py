import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import qmc

# Per-parameter configuration (lo, hi, method, decimals)
# Only for Mooney-Rivlin
POLYMER_PARAMS = {
    "E": {                         # [MPa]  Young's modulus
        "range": (1, 5),
        "method": "sobol",          
        "round": 2 },
    "C10": {                       # First MN parameter [MPa]
        "range": (100, 600), 
        "method": "sobol",
        "round": 0 },
    "C01": {                        # Second MN parameter [MPa]
        "range": (10, 60),
        "method": "sobol",
        "round": 0 },
    "D1": {                         # Compressibility parameter [1/MPa]
        "range": (0.001, 0.1),
        "method": "exponential",    # Prefered for multiple orders of magnitude
        "round": 3 },
    "mu": {                         # Friction coefficient
        "range": (0.3, 0.45),
        "method": "sobol",
        "round": 3 },
}

# Fixed parameters for every sample
POLYMER_FIXED = {"rho": 1e-9, "nu": 0.45}

#  1D samplers
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

#  Parameter generator 
def generate_parameters(
    n_samples=10,
    param_config=None,
    fixed_params=None,
    output_dir="material_parameters",
    tag="mixed",                        # prefix for output filenames
    seed_offset=0 ):                      # added to per-param seed for reproducibility
  
    if param_config is None:
        param_config = POLYMER_PARAMS
    if fixed_params is None:
        fixed_params = POLYMER_FIXED

    columns = {}
    for i, (name, cfg) in enumerate(param_config.items()):
        lo, hi = cfg["range"]
        method = cfg["method"]
        # Pass any extra keys (n_points, levels, value …) as kwargs
        extra = {k: v for k, v in cfg.items() if k not in ("range", "method", "round")}
        columns[name] = _sample_1d(
            method, n_samples, lo, hi,
            seed=42 + seed_offset + i,   # unique seed per parameter
            **extra,
        )

    df = pd.DataFrame(columns)
    df.insert(0, "id", ["%05d" % i for i in range(1, len(df) + 1)])

    # Fixed columns right after id
    for col_idx, (col_name, col_val) in enumerate(fixed_params.items(), start=1):
        df.insert(col_idx, col_name, col_val)

    # Rounding
    rounding = {name: cfg["round"] for name, cfg in param_config.items() if "round" in cfg}
    if rounding:
        df = df.round(rounding)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, tag + "_material_parameter_sweep")
    df.to_csv(base + ".csv", index=False)

    records = df.to_dict(orient="records")
    with open(base + ".py", "w") as f:
        f.write("parameters = [\n")
        for row in records:
            f.write("    %s,\n" % row)
        f.write("]\n")

    print("Saved %d samples → %s.csv" % (len(df), base))
    return records

if __name__ == "__main__":

    custom_config = {
        "E": {
            "range": (1, 5),
            "method": "sobol",
            "round": 2 },
        "C10": {
            "range": (100, 600),
            "method": "halton",
            "round": 0 },
        "C01": {
            "range": (10, 60),
            "method": "lhs",
            "round": 0 },
        "D1": {
            "range": (0.001, 0.1),
            "method": "exponential",
            "round": 3 },
        "mu": {
            "range": (0.3, 0.45),
            "method": "discrete",
            "levels": [0.30, 0.33, 0.36, 0.39, 0.42, 0.45],
            "round": 3 },
    }
    generate_parameters(n_samples=1024, param_config=custom_config, tag="mixed")
