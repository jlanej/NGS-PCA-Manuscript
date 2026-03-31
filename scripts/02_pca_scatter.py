#!/usr/bin/env python3
"""02_pca_scatter.py - Pairwise PC scatter plots coloured by QC variables.

Reads the merged TSV produced by 00_merge_pcs_qc.py and creates scatter
plots of PC1 vs PC2, PC3 vs PC4, coloured by superpopulation, batch,
and inferred sex.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


PALETTE_SUPERPOP = {"AFR": "#E41A1C", "AMR": "#377EB8", "EAS": "#4DAF4A",
                    "EUR": "#984EA3", "SAS": "#FF7F00"}
PALETTE_BATCH = {"698": "#1B9E77", "2504": "#D95F02"}
PALETTE_SEX = {"M": "#4393C3", "F": "#D6604D"}


def _scatter(ax, x, y, labels, palette, title, xlabel, ylabel):
    for cat, colour in palette.items():
        mask = labels == cat
        if mask.any():
            ax.scatter(x[mask], y[mask], c=colour, label=cat, s=5, alpha=0.5,
                       edgecolors="none", rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=7, loc="best")


def pca_scatter(output_dir: str) -> None:
    merged_path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    df = pd.read_csv(merged_path, sep="\t")
    df["RELEASE_BATCH"] = df["RELEASE_BATCH"].astype(str)

    pc_pairs = [("PC1", "PC2"), ("PC3", "PC4")]
    overlays = [
        ("SUPERPOPULATION", PALETTE_SUPERPOP),
        ("RELEASE_BATCH", PALETTE_BATCH),
        ("INFERRED_SEX", PALETTE_SEX),
    ]

    for pcx, pcy in pc_pairs:
        fig, axes = plt.subplots(1, len(overlays), figsize=(6 * len(overlays), 5))
        for ax, (col, pal) in zip(axes, overlays):
            valid = df[col].notna()
            _scatter(ax, df.loc[valid, pcx].values, df.loc[valid, pcy].values,
                     df.loc[valid, col].values, pal,
                     title=col, xlabel=pcx, ylabel=pcy)
        fig.suptitle(f"{pcx} vs {pcy}", fontsize=14, y=1.02)
        fig.tight_layout()
        fname = f"pca_scatter_{pcx}_{pcy}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[02] PCA scatter → {os.path.join(output_dir, fname)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA scatter plots")
    parser.add_argument("--output-dir", default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    args = parser.parse_args()
    pca_scatter(args.output_dir)


if __name__ == "__main__":
    main()
