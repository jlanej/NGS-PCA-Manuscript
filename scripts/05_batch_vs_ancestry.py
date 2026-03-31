#!/usr/bin/env python3
"""05_batch_vs_ancestry.py - Batch vs ancestry effect-size comparison.

Computes η² for RELEASE_BATCH and SUPERPOPULATION on every PC, then produces
a grouped bar chart comparing the two effect sizes side by side.  Also outputs
a summary TSV with the mean η² for each variable.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _eta_squared(categorical: pd.Series, continuous: np.ndarray) -> float:
    """Compute η² for a categorical-continuous pair."""
    groups = categorical.values
    unique = [g for g in np.unique(groups) if pd.notna(g)]
    if len(unique) < 2:
        return np.nan
    group_data = [continuous[groups == g] for g in unique]
    grand_mean = np.nanmean(continuous)
    ss_between = sum(len(g) * (np.nanmean(g) - grand_mean) ** 2 for g in group_data)
    ss_total = np.nansum((continuous - grand_mean) ** 2)
    if ss_total == 0:
        return np.nan
    return ss_between / ss_total


def batch_vs_ancestry(output_dir: str, n_pcs: int = 20) -> pd.DataFrame:
    merged_path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    df = pd.read_csv(merged_path, sep="\t")
    df["RELEASE_BATCH"] = df["RELEASE_BATCH"].astype(str)

    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    available = [c for c in pc_cols if c in df.columns]

    records = []
    for pc in available:
        valid_batch = df["RELEASE_BATCH"].notna()
        valid_anc = df["SUPERPOPULATION"].notna()
        eta2_batch = _eta_squared(df.loc[valid_batch, "RELEASE_BATCH"],
                                  df.loc[valid_batch, pc].values)
        eta2_anc = _eta_squared(df.loc[valid_anc, "SUPERPOPULATION"],
                                df.loc[valid_anc, pc].values)
        records.append({"PC": pc, "Batch_eta2": eta2_batch,
                        "Ancestry_eta2": eta2_anc})

    result = pd.DataFrame(records)

    # Summary
    mean_batch = result["Batch_eta2"].mean()
    mean_ancestry = result["Ancestry_eta2"].mean()
    max_batch = result["Batch_eta2"].max()
    max_ancestry = result["Ancestry_eta2"].max()

    summary = pd.DataFrame([
        {"Variable": "RELEASE_BATCH", "Mean_eta2": mean_batch, "Max_eta2": max_batch},
        {"Variable": "SUPERPOPULATION", "Mean_eta2": mean_ancestry, "Max_eta2": max_ancestry},
    ])
    summary_path = os.path.join(output_dir, "batch_vs_ancestry_summary.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    result_path = os.path.join(output_dir, "batch_vs_ancestry_detail.tsv")
    result.to_csv(result_path, sep="\t", index=False)
    print(f"[05] Summary → {summary_path}")

    # Grouped bar chart
    x = np.arange(len(available))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(available) * 0.6), 5))
    ax.bar(x - width / 2, result["Batch_eta2"], width, label="Batch (η²)",
           color="#D95F02")
    ax.bar(x + width / 2, result["Ancestry_eta2"], width, label="Ancestry (η²)",
           color="#1B9E77")
    ax.set_xticks(x)
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("η² (Effect Size)")
    ax.set_title("Batch vs Ancestry Effect Sizes per PC")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(output_dir, "batch_vs_ancestry.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[05] Bar chart → {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch vs ancestry effect comparison")
    parser.add_argument("--output-dir", default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    parser.add_argument("--n-pcs", type=int, default=20)
    args = parser.parse_args()
    batch_vs_ancestry(args.output_dir, args.n_pcs)


if __name__ == "__main__":
    main()
