import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Structure of the data extracted from the table (F1, F2, F3, base)
data = [
    # F1
    {"control": "F1", "size": 0.02,  "KE_IE": 0.01, "AE_IE": 8.36, "E_drift": 0.001, "ALLPW": 0.55, "SCOF": 0.525, "RF2": 5309, "Res_depth": 61.8, "Pile_up": 41.61, "wallclock": 1300},
    {"control": "F1", "size": 0.015, "KE_IE": 0.0, "AE_IE": 5.60, "E_drift": 0.003, "ALLPW": 0.38, "SCOF": 0.533, "RF2": 5441, "Res_depth": 60.6, "Pile_up": 45.33, "wallclock": 4400},
    {"control": "F1", "size": 0.01,  "KE_IE": 0.0, "AE_IE": 3.62, "E_drift": 0.003, "ALLPW": 0.25, "SCOF": 0.541, "RF2": 5634, "Res_depth": 61.4, "Pile_up": 48.24, "wallclock": 18000},
    {"control": "F1", "size": 0.007, "KE_IE": 0.0, "AE_IE": 1.17, "E_drift": 0.003, "ALLPW": 0.29, "SCOF": 0.543, "RF2": 5565, "Res_depth": 59.2, "Pile_up": 50.28, "wallclock": 120000},
    
    # F2
    {"control": "F2", "size": 0.02,  "KE_IE": 0.01, "AE_IE": 8.37, "E_drift": 0.001, "ALLPW": 0.56, "SCOF": 0.525, "RF2": 5303, "Res_depth": 62.3, "Pile_up": 41.42, "wallclock": 1200},
    {"control": "F2", "size": 0.015, "KE_IE": 0.0, "AE_IE": 5.59, "E_drift": 0.002, "ALLPW": 0.39, "SCOF": 0.532, "RF2": 5427, "Res_depth": 60.8, "Pile_up": 45.16, "wallclock": 4500},
    {"control": "F2", "size": 0.01,  "KE_IE": 0.0, "AE_IE": 3.63, "E_drift": 0.003, "ALLPW": 0.25, "SCOF": 0.541, "RF2": 5634, "Res_depth": 61.4, "Pile_up": 48.06, "wallclock": 18000},
    {"control": "F2", "size": 0.007, "KE_IE": 0.0, "AE_IE": 1.17, "E_drift": 0.003, "ALLPW": 0.29, "SCOF": 0.543, "RF2": 5564, "Res_depth": 58.7, "Pile_up": 50.32, "wallclock": 120000},
    
    # F3
    {"control": "F3", "size": 0.02,  "KE_IE": 0.01, "AE_IE": 8.32, "E_drift": 0.001, "ALLPW": 0.55, "SCOF": 0.525, "RF2": 5309, "Res_depth": 61.7, "Pile_up": 41.6, "wallclock": 1500},
    {"control": "F3", "size": 0.015, "KE_IE": 0.0, "AE_IE": 5.58, "E_drift": 0.003, "ALLPW": 0.37, "SCOF": 0.532, "RF2": 5448, "Res_depth": 61.1, "Pile_up": 45.3, "wallclock": 5400},
    {"control": "F3", "size": 0.01,  "KE_IE": 0.0, "AE_IE": 3.62, "E_drift": 0.003, "ALLPW": 0.25, "SCOF": 0.545, "RF2": 5642, "Res_depth": 61.4, "Pile_up": 48.2, "wallclock": 19900},
    {"control": "F3", "size": 0.007, "KE_IE": 0.0, "AE_IE": 1.17, "E_drift": 0.003, "ALLPW": 0.29, "SCOF": 0.543, "RF2": 5571, "Res_depth": 59.3, "Pile_up": 50.2, "wallclock": 125000},
    
    # Base
    {"control": "base", "size": 0.02,  "KE_IE": 0.1, "AE_IE": 9.56, "E_drift": 0.014, "ALLPW": 3.80, "SCOF": 0.5, "RF2": 4877,  "Res_depth": 55.1, "Pile_up": 40.41, "wallclock": 2000},
    {"control": "base", "size": 0.015, "KE_IE": 0.1, "AE_IE": 6.11, "E_drift": 0.065, "ALLPW": 4.39, "SCOF": 0.518, "RF2": 5170, "Res_depth": 57.8, "Pile_up": 43.37, "wallclock": 5000},
    {"control": "base", "size": 0.01,  "KE_IE": 0.09, "AE_IE": 3.93, "E_drift": 0.088, "ALLPW": 3.71, "SCOF": 0.535, "RF2": 5484, "Res_depth": 64.0, "Pile_up": 47.52, "wallclock": 20000},
    {"control": "base", "size": 0.007, "KE_IE": 0.04, "AE_IE": 1.63, "E_drift": 0.088, "ALLPW": 4.47, "SCOF": 0.542, "RF2": 5410, "Res_depth": 59.6, "Pile_up": 49.88, "wallclock": 100000},
]

df = pd.DataFrame(data)

# Style parameters
controls = ["base", "F1", "F2", "F3"]
colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']  # Red for base, blue/orange/green for tests
markers = ['D', 'o', 's', '^']
sizes = [0.02, 0.015, 0.01, 0.007]

# =========================================================
# FIGURE 1: Energy Indicators
# =========================================================
fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))

energy_metrics = [
    ('KE_IE', 'KE / IE (%)', "Kinetic Energy"),
    ('AE_IE', 'AE / IE (%)', "Artificial Energy"),
    ('E_drift', 'E drift (%)', "Energy Drift"),
    ('ALLPW', 'ALLPW (%)', "Contact Penalty Work")
]

for ax, (col, ylabel, title) in zip(axes1, energy_metrics):
    for ctrl, color, marker in zip(controls, colors, markers):
        sub_df = df[df['control'] == ctrl]
        ax.plot(sub_df['size'], sub_df[col], marker=marker, color=color, linewidth=2, markersize=7, label=f'Control = {ctrl}')
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Mesh Size (Coarse -> Fine)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(sizes)
    ax.set_xticklabels(['0.02', '0.015', '0.01', '0.007'])
    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('hourglass_energies.png', dpi=300)

# =========================================================
# FIGURE 2: Mechanical Responses
# =========================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

mech_metrics = [
    ('RF2', 'RF2 (mN)', 'Reaction Force RF2'),
    ('Res_depth', 'Residual Depth (%)', 'Residual Depth'),
    ('Pile_up', 'Pile-up (µm)', 'Pile-up')
]

for ax, (col, ylabel, title) in zip(axes2, mech_metrics):
    for ctrl, color, marker in zip(controls, colors, markers):
        sub_df = df[df['control'] == ctrl]
        ax.plot(sub_df['size'], sub_df[col], marker=marker, color=color, linewidth=2, markersize=7, label=f'Control = {ctrl}')
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Mesh Size (Coarse -> Fine)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(sizes)
    ax.set_xticklabels(['0.02', '0.015', '0.01', '0.007'])
    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('hourglass_mecanique.png', dpi=300)
plt.show()