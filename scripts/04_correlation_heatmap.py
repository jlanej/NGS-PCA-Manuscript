#!/usr/bin/env python3
"""04_correlation_heatmap.py - PC × QC variable association heatmaps.

For each categorical QC variable (batch, superpopulation, sex) computes η²
(eta-squared) between the variable and each PC.  For continuous QC variables
(MEAN_AUTOSOMAL_COV) computes Pearson r².  Results are displayed as heatmaps.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from utils import eta_squared


def correlation_heatmap(output_dir: str, n_pcs: int = 20) -> None:
    merged_path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    df = pd.read_csv(merged_path, sep="\t")
    df["RELEASE_BATCH"] = df["RELEASE_BATCH"].astype(str)

    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    available = [c for c in pc_cols if c in df.columns]

    cat_vars = ["RELEASE_BATCH", "SUPERPOPULATION", "INFERRED_SEX", "POPULATION"]
    cont_vars = ["MEAN_AUTOSOMAL_COV"]

    rows = []
    for var in cat_vars:
        valid = df[var].notna()
        for pc in available:
            eta2 = eta_squared(df.loc[valid, var], df.loc[valid, pc].values)
            rows.append({"Variable": var, "PC": pc, "Metric": "η²", "Value": eta2})
    for var in cont_vars:
        valid = df[var].notna()
        if valid.sum() < 10:
            continue
        for pc in available:
            r, _ = stats.pearsonr(df.loc[valid, var], df.loc[valid, pc])
            rows.append({"Variable": var, "PC": pc, "Metric": "r²", "Value": r ** 2})

    assoc = pd.DataFrame(rows)

    # Save summary table
    table_path = os.path.join(output_dir, "pc_qc_associations.tsv")
    assoc.to_csv(table_path, sep="\t", index=False)
    print(f"[04] Association table → {table_path}")

    # Heatmap
    pivot = assoc.pivot_table(index="Variable", columns="PC", values="Value")
    # Reorder PCs numerically
    pivot = pivot[[c for c in available if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(max(10, len(available) * 0.5), 4))
    sns.heatmap(pivot, annot=False, cmap="YlOrRd", vmin=0, vmax=1, ax=ax,
                linewidths=0.3, linecolor="white")
    ax.set_title("PC × QC Variable Associations (η² / r²)")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[04] Heatmap → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PC × QC association heatmaps")
    parser.add_argument("--output-dir", default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    parser.add_argument("--n-pcs", type=int, default=20)
    args = parser.parse_args()
    correlation_heatmap(args.output_dir, args.n_pcs)


if __name__ == "__main__":
    main()
