#!/usr/bin/env python3
"""08_variance_partitioning.py - Partial η² / variance partitioning per PC.

For each Marchenko-Pastur-selected PC fits the linear model:

    PC_i ~ RELEASE_BATCH + SUPERPOPULATION + FAMILY_ROLE + MEAN_AUTOSOMAL_COV

Then computes (via leave-one-out R² differences, i.e. Type III sums of squares):

  - Unique η² for batch           (R²_full - R²_no_batch)
  - Unique η² for ancestry        (R²_full - R²_no_ancestry)
  - Unique η² for family role     (R²_full - R²_no_family_role)
  - Unique η² for coverage        (R²_full - R²_no_coverage)
  - Shared / confounded variance  = max(0, R²_full - Σ unique)
  - Residual                      = max(0, 1 - R²_full)

Produces:
  - variance_partitioning.tsv   (unique/shared/residual variance per PC)
  - variance_partitioning.png   (stacked bar chart per PC)
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
from utils import marchenko_pastur_pc_count_from_data_dir


# ---------------------------------------------------------------------------
# Linear model helpers
# ---------------------------------------------------------------------------

def _fit_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Fit OLS with an implicit intercept and return R² ∈ [0, 1].

    Returns 0.0 when SS_total is zero (constant response).
    """
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    # Add intercept column
    X_int = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
    y_pred = X_int @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def variance_partitioning(
    output_dir: str,
    data_dir: str = "1000G",
    n_pcs: int = 0,
) -> pd.DataFrame:
    """Compute partial η² decomposition and write outputs.

    Parameters
    ----------
    output_dir : str
        Directory containing ``merged_pcs_qc.tsv`` (pipeline outputs).
    data_dir : str
        Root data directory (for Marchenko–Pastur PC selection).
    n_pcs : int
        Maximum number of PCs to analyse (0 = Marchenko–Pastur auto).

    Returns
    -------
    pd.DataFrame
        One row per PC with columns for each variance component.
    """
    # ---- Load merged data -------------------------------------------------
    merged_path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    df = pd.read_csv(merged_path, sep="\t")
    df["RELEASE_BATCH"] = df["RELEASE_BATCH"].astype(str)

    # ---- Determine PCs via Marchenko–Pastur --------------------------------
    sv_path = os.path.join(data_dir, "ngspca_output", "svd.singularvalues.txt")
    eigenvalues = pd.read_csv(sv_path, sep="\t")["SINGULAR_VALUES"].values ** 2
    max_pcs = n_pcs if n_pcs > 0 else None
    mp_n_pcs, _ = marchenko_pastur_pc_count_from_data_dir(
        data_dir, eigenvalues, max_pcs=max_pcs,
    )
    pc_cols = [f"PC{i}" for i in range(1, mp_n_pcs + 1)]
    pc_cols = [c for c in pc_cols if c in df.columns]
    print(f"[08] Using {len(pc_cols)} MP-selected PCs")

    # ---- Predictor columns required ---------------------------------------
    required = ["RELEASE_BATCH", "SUPERPOPULATION", "FAMILY_ROLE", "MEAN_AUTOSOMAL_COV"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[08] Missing required columns: {missing}")

    valid = df[required].notna().all(axis=1)
    sub = df.loc[valid].reset_index(drop=True)
    print(f"[08] {valid.sum()} of {len(df)} samples have complete predictor data")

    # ---- Encode categorical predictors as dummy variables -----------------
    X_batch = pd.get_dummies(sub["RELEASE_BATCH"], drop_first=True,
                              prefix="BATCH").values.astype(float)
    X_anc   = pd.get_dummies(sub["SUPERPOPULATION"], drop_first=True,
                              prefix="POP").values.astype(float)
    X_fam   = pd.get_dummies(sub["FAMILY_ROLE"], drop_first=True,
                              prefix="FAM").values.astype(float)

    # Standardise continuous predictor
    cov_raw = sub["MEAN_AUTOSOMAL_COV"].values.astype(float)
    cov_std = (cov_raw - np.mean(cov_raw)) / (np.std(cov_raw) + 1e-10)
    X_cov   = cov_std.reshape(-1, 1)

    # Build full and leave-one-out design matrices
    X_full    = np.column_stack([X_batch, X_anc, X_fam, X_cov])
    X_no_batch = np.column_stack([X_anc,   X_fam, X_cov])
    X_no_anc   = np.column_stack([X_batch, X_fam, X_cov])
    X_no_fam   = np.column_stack([X_batch, X_anc, X_cov])
    X_no_cov   = np.column_stack([X_batch, X_anc, X_fam])

    # ---- Compute decomposition per PC ------------------------------------
    records = []
    for pc in pc_cols:
        y = sub[pc].values.astype(float)

        r2_full     = _fit_r2(X_full,     y)
        r2_no_batch = _fit_r2(X_no_batch, y)
        r2_no_anc   = _fit_r2(X_no_anc,   y)
        r2_no_fam   = _fit_r2(X_no_fam,   y)
        r2_no_cov   = _fit_r2(X_no_cov,   y)

        unique_batch    = max(0.0, r2_full - r2_no_batch)
        unique_ancestry = max(0.0, r2_full - r2_no_anc)
        unique_family   = max(0.0, r2_full - r2_no_fam)
        unique_coverage = max(0.0, r2_full - r2_no_cov)

        # Shared: the portion of R²_full not attributable to any single predictor
        # alone; clipped to [0, 1] (can be slightly negative due to floating point)
        shared   = max(0.0, r2_full - unique_batch - unique_ancestry
                       - unique_family - unique_coverage)
        residual = max(0.0, 1.0 - r2_full)

        records.append({
            "PC": pc,
            "r2_full": r2_full,
            "unique_batch": unique_batch,
            "unique_ancestry": unique_ancestry,
            "unique_family_role": unique_family,
            "unique_coverage": unique_coverage,
            "shared": shared,
            "residual": residual,
        })

    result = pd.DataFrame(records)

    # ---- Write TSV -------------------------------------------------------
    tsv_path = os.path.join(output_dir, "variance_partitioning.tsv")
    result.to_csv(tsv_path, sep="\t", index=False)
    print(f"[08] Summary TSV → {tsv_path}")

    # ---- Stacked bar chart -----------------------------------------------
    _plot_stacked_bar(result, pc_cols, output_dir)

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COMPONENT_COLORS = {
    "unique_batch":       "#D95F02",
    "unique_ancestry":    "#1B9E77",
    "unique_family_role": "#7570B3",
    "unique_coverage":    "#E7298A",
    "shared":             "#A6A6A6",
    "residual":           "#F0F0F0",
}

_COMPONENT_LABELS = {
    "unique_batch":       "Unique: Batch",
    "unique_ancestry":    "Unique: Ancestry",
    "unique_family_role": "Unique: Family Role",
    "unique_coverage":    "Unique: Coverage",
    "shared":             "Shared / Confounded",
    "residual":           "Residual",
}

_COMPONENT_ORDER = [
    "unique_batch", "unique_ancestry", "unique_family_role",
    "unique_coverage", "shared", "residual",
]


def _plot_stacked_bar(result: pd.DataFrame, pc_cols: list, output_dir: str) -> None:
    n = len(pc_cols)
    x = np.arange(n)

    # Normalise each PC's components to sum to 1.0 for a clean 100 % stacked bar.
    # When predictors are correlated the raw unique values can sum to > R²_full,
    # so we proportionally rescale unique contributions to R²_full, then add residual.
    plot_df = result.copy()
    sum_unique = (plot_df["unique_batch"] + plot_df["unique_ancestry"]
                  + plot_df["unique_family_role"] + plot_df["unique_coverage"])
    # Where sum > r2_full, scale unique components down proportionally
    over = sum_unique > plot_df["r2_full"]
    if over.any():
        scale = np.where(over, plot_df["r2_full"] / sum_unique.clip(lower=1e-12), 1.0)
        for col in ["unique_batch", "unique_ancestry", "unique_family_role", "unique_coverage"]:
            plot_df[col] = plot_df[col] * scale
        plot_df["shared"] = 0.0
    # Residual stays as 1 - r2_full
    plot_df["residual"] = (1.0 - plot_df["r2_full"]).clip(lower=0.0)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), 5))

    bottom = np.zeros(n)
    for comp in _COMPONENT_ORDER:
        vals = plot_df[comp].values
        ax.bar(
            x, vals, bottom=bottom,
            color=_COMPONENT_COLORS[comp],
            label=_COMPONENT_LABELS[comp],
            edgecolor="white", linewidth=0.4,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pc_cols, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Proportion of Total Variance")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Variance Partitioning per PC\n"
        "(PC ~ Batch + Ancestry + Family Role + Coverage)"
    )
    ax.legend(
        loc="upper right", fontsize=8,
        framealpha=0.9, edgecolor="#cbd5e1",
    )
    fig.tight_layout()

    out_path = os.path.join(output_dir, "variance_partitioning.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[08] Stacked bar chart → {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partial η² / variance partitioning per PC",
    )
    parser.add_argument("--output-dir",
                        default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    parser.add_argument("--data-dir",
                        default=os.environ.get("NGSPCA_DATA_DIR", "1000G"))
    parser.add_argument("--n-pcs", type=int, default=0,
                        help="Number of PCs (0 = Marchenko–Pastur auto)")
    args = parser.parse_args()
    variance_partitioning(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        n_pcs=args.n_pcs,
    )


if __name__ == "__main__":
    main()
