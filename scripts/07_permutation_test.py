#!/usr/bin/env python3
"""07_permutation_test.py - Permutation test of batch and ancestry η² significance.

For each PC and each grouping variable (RELEASE_BATCH, SUPERPOPULATION), tests
whether the observed η² is larger than expected by chance using a non-parametric
permutation test.  Produces:

  - permutation_eta2_results.tsv   (observed η², mean null, p-value per PC)
  - permutation_eta2_batch.png     (bar chart with significance annotations)
  - permutation_eta2_nulldist.png  (null distribution overlay for top PCs)
  - permutation_eta2_null_distributions.npz  (optional, --save-nulls)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import eta_squared, marchenko_pastur_pc_count_from_data_dir


# ------------------------------------------------------------------
# Core permutation logic
# ------------------------------------------------------------------

def _compute_observed_eta2(
    df: pd.DataFrame,
    pc_cols: list[str],
    variable: str,
) -> np.ndarray:
    """Return observed η² for *variable* on each PC."""
    valid = df[variable].notna()
    labels = df.loc[valid, variable]
    values = df.loc[valid, pc_cols].values
    return np.array([eta_squared(labels, values[:, i]) for i in range(len(pc_cols))])


def _permutation_null(
    df: pd.DataFrame,
    pc_cols: list[str],
    variable: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n_permutations × n_pcs) matrix of permuted η² values."""
    valid = df[variable].notna()
    labels_arr = df.loc[valid, variable].values.copy()
    values = df.loc[valid, pc_cols].values
    n_pcs = len(pc_cols)
    null_matrix = np.empty((n_permutations, n_pcs))
    # Reuse a single Series object to avoid repeated allocation
    shuffled_series = pd.Series(labels_arr.copy())
    for i in range(n_permutations):
        shuffled_series[:] = rng.permutation(labels_arr)
        for j in range(n_pcs):
            null_matrix[i, j] = eta_squared(shuffled_series, values[:, j])
    return null_matrix


def _empirical_pvalues(
    observed: np.ndarray,
    null_matrix: np.ndarray,
) -> np.ndarray:
    """Phipson & Smyth (2010) conservative empirical p-values."""
    n_perm = null_matrix.shape[0]
    counts = np.sum(null_matrix >= observed[np.newaxis, :], axis=0)
    return (counts + 1) / (n_perm + 1)


# ------------------------------------------------------------------
# Significance label helper
# ------------------------------------------------------------------

def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ------------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------------

def permutation_test(
    output_dir: str,
    data_dir: str = "1000G",
    n_pcs: int = 0,
    n_permutations: int = 10_000,
    seed: int = 42,
    save_nulls: bool = False,
) -> pd.DataFrame:
    """Run permutation test and write results + figures."""
    rng = np.random.default_rng(seed)

    # ---- Load data ------------------------------------------------
    merged_path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    df = pd.read_csv(merged_path, sep="\t")
    df["RELEASE_BATCH"] = df["RELEASE_BATCH"].astype(str)

    # ---- Determine PCs via Marchenko–Pastur -----------------------
    eigenvalues = pd.read_csv(
        os.path.join(data_dir, "ngspca_output", "svd.singularvalues.txt"),
        sep="\t",
    )["SINGULAR_VALUES"].values ** 2
    max_pcs = n_pcs if n_pcs > 0 else None
    mp_n_pcs, _ = marchenko_pastur_pc_count_from_data_dir(
        data_dir, eigenvalues, max_pcs=max_pcs,
    )
    pc_cols = [f"PC{i}" for i in range(1, mp_n_pcs + 1)]
    pc_cols = [c for c in pc_cols if c in df.columns]
    print(f"[07] Using {len(pc_cols)} MP-selected PCs")

    # ---- Run permutation tests for each variable ------------------
    variables = ["RELEASE_BATCH", "SUPERPOPULATION"]
    records: list[dict] = []
    null_matrices: dict[str, np.ndarray] = {}

    for var in variables:
        print(f"[07] Permutation test for {var} ({n_permutations} permutations) …")
        observed = _compute_observed_eta2(df, pc_cols, var)
        null_mat = _permutation_null(df, pc_cols, var, n_permutations, rng)
        pvals = _empirical_pvalues(observed, null_mat)
        null_matrices[var] = null_mat

        for k, pc in enumerate(pc_cols):
            records.append({
                "Variable": var,
                "PC": pc,
                "observed_eta2": observed[k],
                "mean_null_eta2": null_mat[:, k].mean(),
                "p_value": pvals[k],
                "n_permutations": n_permutations,
            })

    results = pd.DataFrame(records)

    # ---- Write results TSV ----------------------------------------
    results_path = os.path.join(output_dir, "permutation_eta2_results.tsv")
    results.to_csv(results_path, sep="\t", index=False)
    print(f"[07] Results → {results_path}")

    # ---- Optionally save null distributions -----------------------
    if save_nulls:
        npz_path = os.path.join(output_dir, "permutation_eta2_null_distributions.npz")
        np.savez_compressed(
            npz_path,
            RELEASE_BATCH=null_matrices["RELEASE_BATCH"],
            SUPERPOPULATION=null_matrices["SUPERPOPULATION"],
            pc_cols=np.array(pc_cols),
        )
        print(f"[07] Null distributions → {npz_path}")

    # ---- Figure A: grouped bar chart with significance stars ------
    _plot_bar_chart(results, pc_cols, output_dir)

    # ---- Figure B: null distribution overlay (top PCs) ------------
    _plot_null_distributions(results, null_matrices, pc_cols, output_dir)

    return results


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

def _plot_bar_chart(
    results: pd.DataFrame,
    pc_cols: list[str],
    output_dir: str,
) -> None:
    batch = results[results["Variable"] == "RELEASE_BATCH"].reset_index(drop=True)
    ancestry = results[results["Variable"] == "SUPERPOPULATION"].reset_index(drop=True)

    x = np.arange(len(pc_cols))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(pc_cols) * 0.8), 5))
    bars_b = ax.bar(x - width / 2, batch["observed_eta2"], width,
                    label="Batch (η²)", color="#D95F02")
    bars_a = ax.bar(x + width / 2, ancestry["observed_eta2"], width,
                    label="Ancestry (η²)", color="#1B9E77")

    # Annotate significance
    y_max = max(batch["observed_eta2"].max(), ancestry["observed_eta2"].max())
    offset = y_max * 0.03
    for i in range(len(pc_cols)):
        lbl_b = _sig_label(batch.iloc[i]["p_value"])
        lbl_a = _sig_label(ancestry.iloc[i]["p_value"])
        ax.text(x[i] - width / 2, batch.iloc[i]["observed_eta2"] + offset,
                lbl_b, ha="center", va="bottom", fontsize=7)
        ax.text(x[i] + width / 2, ancestry.iloc[i]["observed_eta2"] + offset,
                lbl_a, ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(pc_cols, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("η² (Effect Size)")
    ax.set_title("Batch vs Ancestry η² with Permutation Significance")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(output_dir, "permutation_eta2_batch.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[07] Bar chart → {out_path}")


def _plot_null_distributions(
    results: pd.DataFrame,
    null_matrices: dict[str, np.ndarray],
    pc_cols: list[str],
    output_dir: str,
    n_top: int = 5,
) -> None:
    """Histogram of permuted η² with observed value overlay for top PCs."""
    show_pcs = pc_cols[:n_top]
    n_show = len(show_pcs)
    variables = ["RELEASE_BATCH", "SUPERPOPULATION"]

    fig, axes = plt.subplots(n_show, len(variables),
                             figsize=(5 * len(variables), 3 * n_show),
                             squeeze=False)

    for j, var in enumerate(variables):
        sub = results[results["Variable"] == var].reset_index(drop=True)
        for i, pc in enumerate(show_pcs):
            ax = axes[i][j]
            k = pc_cols.index(pc)
            null_vals = null_matrices[var][:, k]
            obs_val = sub.iloc[k]["observed_eta2"]
            p_val = sub.iloc[k]["p_value"]

            ax.hist(null_vals, bins=50, color="grey", alpha=0.7, edgecolor="white")
            ax.axvline(obs_val, color="red", linewidth=1.5,
                       label=f"obs={obs_val:.4f}")
            ax.set_title(f"{pc} — {var}\np = {p_val:.4g}", fontsize=9)
            ax.set_xlabel("η²")
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "permutation_eta2_nulldist.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[07] Null distribution overlay → {out_path}")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Permutation test of batch and ancestry η² significance",
    )
    parser.add_argument("--output-dir",
                        default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    parser.add_argument("--data-dir",
                        default=os.environ.get("NGSPCA_DATA_DIR", "1000G"))
    parser.add_argument("--n-pcs", type=int, default=0,
                        help="Number of PCs (0 = Marchenko–Pastur auto)")
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-nulls", action="store_true",
                        help="Save full null distribution matrices as .npz")
    args = parser.parse_args()
    permutation_test(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        n_pcs=args.n_pcs,
        n_permutations=args.n_permutations,
        seed=args.seed,
        save_nulls=args.save_nulls,
    )


if __name__ == "__main__":
    main()
