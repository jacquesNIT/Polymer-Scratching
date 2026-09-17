"""
plot_hardening_tables.py

Plots the hardening tables (sigma_y, eps_p) of the polymer families
registered in Configuration/families.py.

Note: this script does NOT reimplement the G'Sell-Jonas formula. It builds
each config via family.build_config() (the same way Modelbuilder.py does)
and reads cfg.material.plasticity.yield_table directly -- so the plot is
guaranteed to match the code actually executed, not a copy of the formula
that could drift from the code over time.

Usage (run from the repo root, next to the Configuration/ package):
    python plot_hardening_tables.py
    python plot_hardening_tables.py --families glassy_pc semicrystalline_dp
    python plot_hardening_tables.py --out hardening_tables.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# --- direct import of the project package (no reimplementation) ---
# This script assumes it lives next to Configuration/ (repo root).
# Adjust PROJECT_ROOT below if that is not the case.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from Configuration.families import FAMILIES  # noqa: E402


# Families excluded by default (glassy_dp: coarse 3-point placeholder table;
# semicrystalline_j2: identical yield_table to semicrystalline_dp, redundant
# on this plot).
EXCLUDED_FAMILIES = ("glassy_dp", "semicrystalline_j2")

# Per-family style: color + marker shape only (no connecting line), so
# series stay distinguishable without relying on color alone.
STYLE = {
    "glassy_pc":          dict(color="#2a78d6", marker="o", label="glassy_pc (PC)"),
    "glassy_pmma":        dict(color="#eb6834", marker="s", label="glassy_pmma (PMMA)"),
    "semicrystalline_dp": dict(color="#4a3aa7", marker="v", label="semicrystalline_dp"),
}


def collect_tables(keys=None):
    """Builds each requested config and retrieves its yield_table if present."""
    keys = keys or [k for k in FAMILIES if k not in EXCLUDED_FAMILIES]
    tables = {}
    for key in keys:
        if key not in FAMILIES:
            print("  [skip] unknown family: %s" % key)
            continue
        cfg = FAMILIES[key].build_config()
        plasticity = getattr(cfg.material, "plasticity", None)
        yield_table = getattr(plasticity, "yield_table", None)
        if not yield_table:
            print("  [skip] %s: no yield_table (plasticity='%s', "
                  "purely hyperelastic/viscoelastic family)"
                  % (key, getattr(plasticity, "MODEL", "?")))
            continue
        tables[key] = yield_table
    return tables


def plot_tables(tables, out=None, log_x=False):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for key, table in sorted(tables.items()):
        eps = [pt[1] for pt in table]
        sig = [pt[0] for pt in table]
        style = STYLE.get(key, dict(color=None, marker="o", label=key))
        n_pts = len(table)
        ax.plot(eps, sig,
                 marker=style["marker"], linestyle="none",
                 color=style["color"],
                 markersize=8 if n_pts <= 6 else 3.5,   # bigger markers for sparse tables
                 label="%s (%d pts)" % (style["label"], n_pts))

    ax.set_xlabel(r"$\varepsilon_p$ -- equivalent plastic strain [-]")
    ax.set_ylabel(r"$\sigma_y$ -- flow stress [MPa]")
    ax.set_title("Hardening tables (*PLASTIC / *DRUCKER PRAGER HARDENING)")
    if log_x:
        ax.set_xscale("symlog", linthresh=0.01)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=200)
        print("Figure saved: %s" % out)
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", nargs="*", default=None,
                         help="Subset of families to plot (default: all FAMILIES except %s)"
                              % ", ".join(EXCLUDED_FAMILIES))
    parser.add_argument("--out", default=None,
                         help="Output path (.png/.pdf). Without this argument: interactive display (plt.show)")
    parser.add_argument("--log-x", action="store_true",
                         help="Plot eps_p on a symlog scale (useful if tables span very different ranges)")
    args = parser.parse_args()

    print("Available families: %s" % ", ".join(sorted(FAMILIES)))
    tables = collect_tables(args.families)
    if not tables:
        print("No hardening table found for the requested families.")
        return

    plot_tables(tables, out=args.out, log_x=args.log_x)


if __name__ == "__main__":
    main()