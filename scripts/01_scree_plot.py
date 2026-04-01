#!/usr/bin/env python3
"""01_scree_plot.py - Scree and cumulative variance explained plots.

Reads singular values, squares them to get eigenvalues, computes proportion
and cumulative variance explained, and produces a publication-quality figure.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import marchenko_pastur_pc_count_from_data_dir


def scree_plot(data_dir: str, output_dir: str, n_pcs: int = 0) -> None:
    """Create scree plot and cumulative variance explained plot.

    Parameters
    ----------
    data_dir : str
        Root data directory with ``ngspca_output/``.
    output_dir : str
        Directory where figure will be written.
    n_pcs : int
        Number of PCs to display (default 0 = all available PCs).
    """
    sv = pd.read_csv(os.path.join(data_dir, "ngspca_output", "svd.singularvalues.txt"), sep="\t")
    eigenvalues = sv["SINGULAR_VALUES"].values ** 2
    total_var = eigenvalues.sum()
    prop_var = eigenvalues / total_var
    cum_var = np.cumsum(prop_var)
    mp_n_pcs, _ = marchenko_pastur_pc_count_from_data_dir(data_dir, eigenvalues)

    n_show = len(eigenvalues) if n_pcs <= 0 else min(n_pcs, len(eigenvalues))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    ax1 = axes[0]
    ax1.bar(range(1, n_show + 1), prop_var[:n_show], color="steelblue", edgecolor="white")
    if mp_n_pcs <= n_show:
        ax1.axvline(mp_n_pcs + 0.5, linestyle="--", color="darkorange", linewidth=1.5,
                    label=f"MP cutoff: {mp_n_pcs} PCs")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Proportion of Variance Explained")
    ax1.set_title("Scree Plot")
    ax1.set_xlim(0.5, n_show + 0.5)
    if mp_n_pcs <= n_show:
        ax1.legend(loc="upper right")

    # Cumulative variance plot
    ax2 = axes[1]
    ax2.plot(range(1, n_show + 1), cum_var[:n_show], marker="o", markersize=3,
             color="darkred", linewidth=1.5)
    ax2.axhline(0.8, linestyle="--", color="grey", alpha=0.7, label="80 %")
    ax2.axhline(0.9, linestyle=":", color="grey", alpha=0.7, label="90 %")
    if mp_n_pcs <= n_show:
        ax2.axvline(mp_n_pcs, linestyle="--", color="darkorange", linewidth=1.5,
                    label=f"MP cutoff: {mp_n_pcs} PCs")
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()
    ax2.set_xlim(0.5, n_show + 0.5)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "scree_cumvar.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[01] Scree plot → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scree / cumulative variance plot")
    parser.add_argument("--data-dir", default=os.environ.get("NGSPCA_DATA_DIR", "1000G"))
    parser.add_argument("--output-dir", default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    parser.add_argument("--n-pcs", type=int, default=0,
                        help="Number of PCs to plot (0 = all available PCs)")
    args = parser.parse_args()
    scree_plot(args.data_dir, args.output_dir, args.n_pcs)


if __name__ == "__main__":
    main()
