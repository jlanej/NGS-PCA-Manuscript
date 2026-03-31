#!/usr/bin/env python3
"""03_umap_projection.py - UMAP projections from selected PCs.

Reads the merged TSV and computes UMAP embeddings from the first *k* PCs
(default k=20).  Overlays superpopulation, batch, and sex.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap


PALETTE_SUPERPOP = {"AFR": "#E41A1C", "AMR": "#377EB8", "EAS": "#4DAF4A",
                    "EUR": "#984EA3", "SAS": "#FF7F00"}
PALETTE_BATCH = {"698": "#1B9E77", "2504": "#D95F02"}
PALETTE_SEX = {"M": "#4393C3", "F": "#D6604D"}


def umap_projection(output_dir: str, n_pcs: int = 20) -> None:
    merged_path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    df = pd.read_csv(merged_path, sep="\t")
    df["RELEASE_BATCH"] = df["RELEASE_BATCH"].astype(str)

    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    available = [c for c in pc_cols if c in df.columns]
    X = df[available].values

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30,
                        min_dist=0.3)
    embedding = reducer.fit_transform(X)
    df = df.copy()
    df["UMAP1"] = embedding[:, 0]
    df["UMAP2"] = embedding[:, 1]

    overlays = [
        ("SUPERPOPULATION", PALETTE_SUPERPOP),
        ("RELEASE_BATCH", PALETTE_BATCH),
        ("INFERRED_SEX", PALETTE_SEX),
    ]

    fig, axes = plt.subplots(1, len(overlays), figsize=(6 * len(overlays), 5))
    for ax, (col, pal) in zip(axes, overlays):
        for cat, colour in pal.items():
            mask = df[col] == cat
            if mask.any():
                ax.scatter(df.loc[mask, "UMAP1"], df.loc[mask, "UMAP2"],
                           c=colour, label=cat, s=5, alpha=0.5,
                           edgecolors="none", rasterized=True)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(f"UMAP ({n_pcs} PCs) — {col}")
        ax.legend(markerscale=3, fontsize=7, loc="best")

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"umap_{n_pcs}pcs.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[03] UMAP projection → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP projection from PCs")
    parser.add_argument("--output-dir", default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    parser.add_argument("--n-pcs", type=int, default=20)
    args = parser.parse_args()
    umap_projection(args.output_dir, args.n_pcs)


if __name__ == "__main__":
    main()
