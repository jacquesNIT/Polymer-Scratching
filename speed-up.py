import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. DATA INPUT (Same format as your Excel)
# ==========================================
# Add your values as you go. Leave np.nan for empty cells.
data = {
    "Mesh_Sizes": [3300, 10000, 32000, 83000, 240000],
    48: [np.nan, np.nan, np.nan, 56, 194],
    36: [6.0, 10.0, 21.0, 55.0, 200.0],
    24: [8.0, 11.0, 24.0, 69.0, 245],
    16: [4.0, 9.0, 27.0, 82.0, 307.0],
    12: [4.0, 9.0, 29.0, np.nan, np.nan],
    10: [np.nan, np.nan, np.nan, np.nan, np.nan],
    8: [np.nan, np.nan, np.nan, np.nan, np.nan],
    6: [np.nan, np.nan, np.nan, np.nan, np.nan],
    4: [np.nan, np.nan, np.nan, np.nan, np.nan],
    2: [np.nan, np.nan, np.nan, np.nan, np.nan],
}

# Convert to DataFrame
df_wide = pd.DataFrame(data)

# ==========================================
# 2. HPC TRANSFORMATION AND CALCULATIONS
# ==========================================
# Pivot the table from wide to long format for easier calculations
df_long = df_wide.melt(
    id_vars=["Mesh_Sizes"], var_name="CPUs", value_name="Walltime"
)
df_long["CPUs"] = df_long["CPUs"].astype(int)

# Temporarily drop rows that don't have a value yet (np.nan)
df_long = df_long.dropna(subset=["Walltime"]).sort_values(by=["Mesh_Sizes", "CPUs"])


def compute_hpc_metrics(group):
    # Find the minimum number of CPUs available for this specific mesh size
    min_cpu = group["CPUs"].min()
    ref_time = group.loc[group["CPUs"] == min_cpu, "Walltime"].values[0]

    # Calculate Speed-up relative to the minimum available CPU
    group["Speedup"] = ref_time / group["Walltime"]

    # Calculate relative Efficiency: S(P) / (P / P_min)
    group["Efficiency"] = group["Speedup"] / (group["CPUs"] / min_cpu)
    return group


# Apply calculations per Mesh Size group
df_metrics = df_long.groupby("Mesh_Sizes", group_keys=False).apply(
    compute_hpc_metrics
)

# ==========================================
# 3. PLOTTING THE CURVES
# ==========================================
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
mesh_sizes = df_metrics["Mesh_Sizes"].unique()

# Colors and markers to differentiate the mesh sizes
styles = {
    3300: {"color": "#0c54c4", "marker": "o"},
    10000: {"color": "#d48400", "marker": "s"},
    32000: {"color": "#00a676", "marker": "^"},
    83000: {"color": "#bf4f00", "marker": "D"},
    240000: {"color": "#c770c7", "marker": "X"},
}

for mesh in mesh_sizes:
    df_m = df_metrics[df_metrics["Mesh_Sizes"] == mesh]
    cpus = df_m["CPUs"]
    style = styles.get(
        mesh, {"color": None, "marker": "o"}
    )  # Default style if more mesh sizes are added

    # --- Plot 1: Simulation Time ---
    axs[0].plot(
        cpus,
        df_m["Walltime"],
        marker=style["marker"],
        color=style["color"],
        linewidth=2,
        label=f"{mesh} cells",
    )

    # --- Plot 2: Speed-up ---
    axs[1].plot(
        cpus,
        df_m["Speedup"],
        marker=style["marker"],
        color=style["color"],
        linewidth=2,
        label=f"{mesh} cells",
    )

    # --- Plot 3: Efficiency ---
    axs[2].plot(
        cpus,
        df_m["Efficiency"],
        marker=style["marker"],
        color=style["color"],
        linewidth=2,
        label=f"{mesh} cells",
    )

# --- Aesthetic Formatting ---

# Plot 1: Time
axs[0].set_title("Simulation Time", fontsize=12, fontweight="bold")
axs[0].set_xlabel("Number of CPUs")
axs[0].set_ylabel("Wallclock Time (seconds)")
axs[0].grid(True, linestyle="--", alpha=0.5)
axs[0].legend()

# Plot 2: Speed-up
axs[1].set_title("Speed-up (Relative to Min CPU)", fontsize=12, fontweight="bold")
axs[1].set_xlabel("Number of CPUs")
axs[1].set_ylabel("Speed-up $S(P)$")
axs[1].grid(True, linestyle="--", alpha=0.5)
axs[1].legend()

# Plot 3: Efficiency
axs[2].set_title("Parallel Efficiency", fontsize=12, fontweight="bold")
axs[2].set_xlabel("Number of CPUs")
axs[2].set_ylabel("Efficiency $E(P)$")
axs[2].set_ylim(0, 1.2)  # Limit scale from 0 to 120%
axs[2].grid(True, linestyle="--", alpha=0.5)
axs[2].legend()

plt.tight_layout()
plt.show()