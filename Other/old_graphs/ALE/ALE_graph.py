import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 1. DATA EXTRACTED FROM SIMULATION RESULTS (Tests with glassy_DP)
# ==============================================================================
data = [
    # --- Element size: 0.02 ---
    {"size": 0.02, "ALE": 20,   "AE_IE": 5.22, "E_drift": 0.525, "ALLPW": 7.46, "RF2": 4005, "Res_depth": 58.6, "Pile_up": 31.98},
    {"size": 0.02, "ALE": 200,  "AE_IE": 4.89, "E_drift": 0.752, "ALLPW": 8.42, "RF2": 3900, "Res_depth": 60.0, "Pile_up": 30.11},
    {"size": 0.02, "ALE": 650,  "AE_IE": 5.39, "E_drift": 0.125, "ALLPW": 5.77, "RF2": 4337, "Res_depth": 58.8, "Pile_up": 38.19},
    {"size": 0.02, "ALE": 1300, "AE_IE": 5.43, "E_drift": 0.097, "ALLPW": 5.68, "RF2": 4324, "Res_depth": 59.5, "Pile_up": 38.14},
    
    # --- Element size: 0.015 ---
    {"size": 0.015, "ALE": 20,   "AE_IE": 5.22, "E_drift": 0.865, "ALLPW": 9.34, "RF2": 3982, "Res_depth": 66.8, "Pile_up": 58.50},
    {"size": 0.015, "ALE": 200,  "AE_IE": 5.78, "E_drift": 0.184, "ALLPW": 7.87, "RF2": 4454, "Res_depth": 68.1, "Pile_up": 66.18},
    {"size": 0.015, "ALE": 650,  "AE_IE": 6.10, "E_drift": 0.563, "ALLPW": 7.08, "RF2": 4602, "Res_depth": 65.5, "Pile_up": 65.70},
    {"size": 0.015, "ALE": 1300, "AE_IE": 6.05, "E_drift": 0.511, "ALLPW": 7.61, "RF2": 4514, "Res_depth": 65.9, "Pile_up": 65.70},
    
    # --- Element size: 0.01 ---
    {"size": 0.01, "ALE": 20,   "AE_IE": 3.82, "E_drift": 0.646, "ALLPW": 8.63, "RF2": 4456, "Res_depth": 68.3, "Pile_up": 68.46},
    {"size": 0.01, "ALE": 200,  "AE_IE": 4.34, "E_drift": 0.081, "ALLPW": 8.10, "RF2": 4907, "Res_depth": 69.2, "Pile_up": 73.19},
    {"size": 0.01, "ALE": 650,  "AE_IE": 4.60, "E_drift": 0.657, "ALLPW": 7.49, "RF2": 5111, "Res_depth": 68.3, "Pile_up": 76.75},
    {"size": 0.01, "ALE": 1300, "AE_IE": 4.76, "E_drift": 0.651, "ALLPW": 7.01, "RF2": 5107, "Res_depth": 67.9, "Pile_up": 77.85},
    
    # --- Element size: 0.007 ---
    {"size": 0.007, "ALE": 20,   "AE_IE": 2.19, "E_drift": 0.868, "ALLPW": 5.89, "RF2": 5247, "Res_depth": 67.0, "Pile_up": 83.42},
    {"size": 0.007, "ALE": 200,  "AE_IE": 2.29, "E_drift": 1.436, "ALLPW": 5.12, "RF2": 5619, "Res_depth": 65.9, "Pile_up": 88.62},
    {"size": 0.007, "ALE": 650,  "AE_IE": np.nan, "E_drift": np.nan, "ALLPW": np.nan, "RF2": np.nan, "Res_depth": np.nan, "Pile_up": np.nan},
    {"size": 0.007, "ALE": 1300, "AE_IE": np.nan, "E_drift": np.nan, "ALLPW": np.nan, "RF2": np.nan, "Res_depth": np.nan, "Pile_up": np.nan},
]

df = pd.DataFrame(data)

# Plot formatting options
ale_values = [20, 200, 650, 1300]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']
sizes = [0.02, 0.015, 0.01, 0.007]

# ==============================================================================
# 2. FIGURE 1: ENERGY INDICATORS (AE/IE, E_drift, ALLPW) SIDE BY SIDE
# ==============================================================================
fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))

energy_metrics = [
    ('AE_IE', 'AE / IE (%)', 'Artificial Energy Ratio (AE / IE)'),
    ('E_drift', 'E drift (%)', 'Energy Drift (E drift)'),
    ('ALLPW', 'ALLPW (%)', 'Contact Penalty Work (ALLPW)')
]

for ax, (col, ylabel, title) in zip(axes1, energy_metrics):
    for ale, color, marker in zip(ale_values, colors, markers):
        sub_df = df[df['ALE'] == ale]
        ax.plot(sub_df['size'], sub_df[col], marker=marker, color=color, linewidth=2, markersize=7, label=f'ALE = {ale}')
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Element Size (Coarse → Fine)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(sizes)
    ax.set_xticklabels(['0.02', '0.015', '0.01', '0.007'])
    ax.invert_xaxis()  # Displays 0.02 on the left and 0.007 on the right (mesh refinement direction)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('energy_convergence.png', dpi=300)

# ==============================================================================
# 3. FIGURE 2: MECHANICAL RESPONSES (RF2, Res depth, Pile-up) SIDE BY SIDE
# ==============================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

mech_metrics = [
    ('RF2', 'RF2 (mN)', 'Reaction Force RF2'),
    ('Res_depth', 'Residual Depth (%)', 'Residual Depth'),
    ('Pile_up', 'Pile-up (µm)', 'Pile-up Height')
]

for ax, (col, ylabel, title) in zip(axes2, mech_metrics):
    for ale, color, marker in zip(ale_values, colors, markers):
        sub_df = df[df['ALE'] == ale]
        ax.plot(sub_df['size'], sub_df[col], marker=marker, color=color, linewidth=2, markersize=7, label=f'ALE = {ale}')
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Element Size (Coarse → Fine)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(sizes)
    ax.set_xticklabels(['0.02', '0.015', '0.01', '0.007'])
    ax.invert_xaxis()  # Displays 0.02 on the left and 0.007 on the right
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('mechanical_convergence.png', dpi=300)

plt.show()