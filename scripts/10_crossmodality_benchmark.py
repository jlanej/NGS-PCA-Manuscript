#!/usr/bin/env python3
"""10_crossmodality_benchmark.py — Cross-modality validation of NGS-PCA vs. array-based PCs.

Compare principal components derived from NGS coverage depth (ngsPCA) with
those derived from Illumina array genotyping to determine whether NGS-PCA's
ancestry signal reflects real population structure or reference-bias artefact.

Analyses
--------
1. **PC–PC correlation matrix** (Pearson & Spearman) — heatmap visualisation.
2. **Canonical Correlation Analysis (CCA)** — overall subspace alignment.
3. **Procrustes rotation** — geometric alignment and disparity.
4. **Residual batch test** — after regressing array-PCs out of each NGS-PC,
   test residuals ~ RELEASE_BATCH to detect residual technical signal.
5. **Overlay scatter plots** — NGS-PC1 vs NGS-PC2 coloured by array-PC1
   and vice versa.

Outputs (written to *output_dir*)
----------------------------------
- ``crossmodality_correlation.tsv``   — Pearson & Spearman r for each pair
- ``crossmodality_summary.tsv``       — CCA canonical correlations + Procrustes
- ``crossmodality_residual_batch.tsv`` — batch η² before/after adjustment
- ``crossmodality_heatmap.png``       — correlation heatmap
- ``crossmodality_scatter.png``       — overlay scatter plots
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import procrustes as scipy_procrustes
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(__file__))
from utils import eta_squared, marchenko_pastur_pc_count_from_data_dir


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_array_pcs(data_dir: str) -> pd.DataFrame:
    """Load array-based ancestry PCs from compiled_sample_sheet.tsv.

    Returns a DataFrame with columns: sample_id, ARRAY_PC1 … ARRAY_PC20,
    keeping only samples that were not excluded prior to array PCA.
    """
    path = os.path.join(
        data_dir, "illumina_idat_processing", "compiled_sample_sheet.tsv"
    )
    cols_needed = ["sample_id", "pre_pca_excluded"] + [
        f"PC{i}" for i in range(1, 21)
    ]
    df = pd.read_csv(path, sep="\t", usecols=cols_needed)
    # Retain only samples that participated in the array PCA computation
    df = df[df["pre_pca_excluded"] == 0].copy()
    df = df.drop(columns=["pre_pca_excluded"])
    rename = {f"PC{i}": f"ARRAY_PC{i}" for i in range(1, 21)}
    df = df.rename(columns=rename)
    return df


def _load_ngs_merged(output_dir: str) -> pd.DataFrame:
    """Load merged NGS-PCA + QC data."""
    path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    return pd.read_csv(path, sep="\t")


def _merge_ngs_array(
    ngs: pd.DataFrame, array: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join NGS-PCA and array-PCA data on sample ID."""
    merged = ngs.merge(array, left_on="SAMPLE", right_on="sample_id", how="inner")
    merged = merged.drop(columns=["sample_id"])
    return merged


# ---------------------------------------------------------------------------
# Analysis 1: PC–PC correlation matrix
# ---------------------------------------------------------------------------

def _compute_pc_correlations(
    df: pd.DataFrame, ngs_cols: list[str], array_cols: list[str]
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations for all NGS-PC × ARRAY-PC pairs."""
    records = []
    for nc in ngs_cols:
        for ac in array_cols:
            x = df[nc].values
            y = df[ac].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 3:
                records.append(
                    dict(
                        NGS_PC=nc, ARRAY_PC=ac,
                        pearson_r=np.nan, pearson_p=np.nan,
                        spearman_r=np.nan, spearman_p=np.nan,
                    )
                )
                continue
            pr, pp = pearsonr(x[mask], y[mask])
            sr, sp = spearmanr(x[mask], y[mask])
            records.append(
                dict(
                    NGS_PC=nc, ARRAY_PC=ac,
                    pearson_r=pr, pearson_p=pp,
                    spearman_r=sr, spearman_p=sp,
                )
            )
    return pd.DataFrame(records)


def _plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    ngs_cols: list[str],
    array_cols: list[str],
    output_path: str,
) -> None:
    """Save a Pearson |r| heatmap of NGS-PCs × array-PCs."""
    mat = corr_df.pivot(index="NGS_PC", columns="ARRAY_PC", values="pearson_r")
    mat = mat.reindex(index=ngs_cols, columns=array_cols)

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(array_cols) * 0.6 + 2),
                                             max(5, len(ngs_cols) * 0.4 + 1)))

    # Absolute correlation heatmap
    sns.heatmap(
        mat.abs(), ax=axes[0], cmap="YlOrRd", vmin=0, vmax=1,
        annot=True, fmt=".2f", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "|Pearson r|", "shrink": 0.8},
    )
    axes[0].set_title("|Pearson r|", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Array PC")
    axes[0].set_ylabel("NGS PC")

    # Signed correlation heatmap
    sns.heatmap(
        mat, ax=axes[1], cmap="RdBu_r", vmin=-1, vmax=1,
        annot=True, fmt=".2f", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )
    axes[1].set_title("Signed Pearson r", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Array PC")
    axes[1].set_ylabel("NGS PC")

    fig.suptitle(
        "NGS-PCA vs. Array-PCA PC Correlation",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis 2: Canonical Correlation Analysis
# ---------------------------------------------------------------------------

def _compute_cca(
    df: pd.DataFrame,
    ngs_cols: list[str],
    array_cols: list[str],
    n_components: int | None = None,
) -> dict:
    """Run CCA between the two PC sets and return canonical correlations."""
    X = df[ngs_cols].values.copy()
    Y = df[array_cols].values.copy()

    # Standardise columns (CCA is sensitive to scale)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-12)

    n_comp = n_components or min(len(ngs_cols), len(array_cols))
    # CCA requires n_components <= min(n_features_X, n_features_Y, n_samples)
    n_comp = min(n_comp, X.shape[0], X.shape[1], Y.shape[1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cca = CCA(n_components=n_comp, max_iter=1000)
        X_c, Y_c = cca.fit_transform(X, Y)

    canon_corrs = [
        float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]) for i in range(n_comp)
    ]
    return {
        "n_components": n_comp,
        "canonical_correlations": canon_corrs,
        "mean_canonical_corr": float(np.mean(canon_corrs)),
    }


# ---------------------------------------------------------------------------
# Analysis 3: Procrustes rotation
# ---------------------------------------------------------------------------

def _compute_procrustes(
    df: pd.DataFrame, ngs_cols: list[str], array_cols: list[str]
) -> dict:
    """Procrustes alignment of NGS-PC and array-PC embeddings.

    Both matrices are trimmed to the same number of dimensions
    (min of the two PC counts) and the same samples.
    """
    n_dim = min(len(ngs_cols), len(array_cols))
    X = df[ngs_cols[:n_dim]].values.copy()
    Y = df[array_cols[:n_dim]].values.copy()

    # scipy.spatial.procrustes requires same shape and handles centering/scaling
    _, _, disparity = scipy_procrustes(X, Y)
    return {
        "n_dimensions": n_dim,
        "procrustes_disparity": float(disparity),
    }


# ---------------------------------------------------------------------------
# Analysis 4: Residual batch test
# ---------------------------------------------------------------------------

def _compute_residual_batch(
    df: pd.DataFrame, ngs_cols: list[str], array_cols: list[str],
    n_adjust: int | None = None,
) -> pd.DataFrame:
    """For each NGS-PC, compute batch η² before and after regressing out array PCs.

    Parameters
    ----------
    n_adjust : int, optional
        Number of top array PCs to regress out (default: all available).
    """
    adj_cols = array_cols[:n_adjust] if n_adjust else array_cols
    Z = df[adj_cols].values

    valid = df["RELEASE_BATCH"].notna()
    batch = df.loc[valid, "RELEASE_BATCH"].astype(str)

    records = []
    for pc in ngs_cols:
        y = df[pc].values
        y_valid = y[valid]

        # Original η²
        eta2_orig = eta_squared(batch, y_valid)

        # Regress out array PCs
        Z_valid = Z[valid]
        reg = LinearRegression().fit(Z_valid, y_valid)
        residuals = y_valid - reg.predict(Z_valid)
        eta2_resid = eta_squared(batch, residuals)

        records.append(
            dict(
                NGS_PC=pc,
                eta2_batch_original=eta2_orig,
                eta2_batch_residual=eta2_resid,
                eta2_change=eta2_orig - eta2_resid,
                n_array_pcs_adjusted=len(adj_cols),
            )
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Analysis 5: Overlay scatter plots
# ---------------------------------------------------------------------------

def _plot_overlay_scatter(
    df: pd.DataFrame, output_path: str
) -> None:
    """Create overlay scatter plots: NGS-PC1 vs NGS-PC2 coloured by array-PC1,
    and array-PC1 vs array-PC2 coloured by NGS-PC1."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: NGS-PC1 vs NGS-PC2, colour = array-PC1
    sc1 = axes[0].scatter(
        df["PC1"], df["PC2"],
        c=df["ARRAY_PC1"], cmap="coolwarm", s=6, alpha=0.7, edgecolors="none",
    )
    axes[0].set_xlabel("NGS PC1", fontsize=10)
    axes[0].set_ylabel("NGS PC2", fontsize=10)
    axes[0].set_title("NGS-PC space, coloured by Array PC1", fontsize=11, fontweight="bold")
    plt.colorbar(sc1, ax=axes[0], label="Array PC1", shrink=0.8)

    # Panel B: array-PC1 vs array-PC2, colour = NGS-PC1
    sc2 = axes[1].scatter(
        df["ARRAY_PC1"], df["ARRAY_PC2"],
        c=df["PC1"], cmap="coolwarm", s=6, alpha=0.7, edgecolors="none",
    )
    axes[1].set_xlabel("Array PC1", fontsize=10)
    axes[1].set_ylabel("Array PC2", fontsize=10)
    axes[1].set_title("Array-PC space, coloured by NGS PC1", fontsize=11, fontweight="bold")
    plt.colorbar(sc2, ax=axes[1], label="NGS PC1", shrink=0.8)

    fig.suptitle(
        "Cross-Modality Overlay: NGS-PCA vs Array-PCA",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def crossmodality_benchmark(
    output_dir: str,
    data_dir: str = "1000G",
    n_pcs: int = 0,
    seed: int = 42,
) -> None:
    """Run all cross-modality validation analyses.

    Parameters
    ----------
    output_dir : str
        Directory for output files.
    data_dir : str
        Root data directory containing ``illumina_idat_processing/`` and
        ``ngspca_output/``.
    n_pcs : int
        Number of NGS-PCs to use. 0 = auto-select via Marchenko–Pastur.
    seed : int
        Random seed (not currently used, reserved for future permutation tests).
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    # -- Load data --
    print("[10] Loading NGS-PCA merged data …")
    ngs = _load_ngs_merged(output_dir)

    print("[10] Loading array-based ancestry PCs …")
    array = _load_array_pcs(data_dir)

    print("[10] Merging datasets …")
    merged = _merge_ngs_array(ngs, array)
    n_samples = len(merged)
    print(f"[10]   {n_samples} overlapping samples after merge")
    if n_samples < 10:
        print("[10] WARNING: too few overlapping samples for meaningful analysis.")
        return

    # -- Determine significant NGS PCs via Marchenko–Pastur --
    sv_path = os.path.join(data_dir, "ngspca_output", "svd.singularvalues.txt")
    sv_df = pd.read_csv(sv_path, sep="\t")
    eigenvalues = sv_df["SINGULAR_VALUES"].values ** 2

    if n_pcs > 0:
        n_ngs_pcs = n_pcs
    else:
        n_ngs_pcs, _ = marchenko_pastur_pc_count_from_data_dir(
            data_dir, eigenvalues
        )
    ngs_cols = [f"PC{i}" for i in range(1, n_ngs_pcs + 1)]
    # Ensure all NGS PC columns exist
    ngs_cols = [c for c in ngs_cols if c in merged.columns]
    array_cols = [f"ARRAY_PC{i}" for i in range(1, 21)]
    array_cols = [c for c in array_cols if c in merged.columns]
    print(f"[10]   Using {len(ngs_cols)} NGS-PCs × {len(array_cols)} array-PCs")

    # -- Analysis 1: Correlation matrix --
    print("[10] Computing PC–PC correlations …")
    corr_df = _compute_pc_correlations(merged, ngs_cols, array_cols)
    corr_path = os.path.join(output_dir, "crossmodality_correlation.tsv")
    corr_df.to_csv(corr_path, sep="\t", index=False)
    print(f"[10]   → {corr_path}")

    heatmap_path = os.path.join(output_dir, "crossmodality_heatmap.png")
    _plot_correlation_heatmap(corr_df, ngs_cols, array_cols, heatmap_path)
    print(f"[10]   → {heatmap_path}")

    # -- Analysis 2: CCA --
    print("[10] Running Canonical Correlation Analysis …")
    cca_results = _compute_cca(merged, ngs_cols, array_cols)
    print(f"[10]   CCA mean canonical corr = {cca_results['mean_canonical_corr']:.4f}")

    # -- Analysis 3: Procrustes --
    print("[10] Computing Procrustes rotation …")
    proc_results = _compute_procrustes(merged, ngs_cols, array_cols)
    print(f"[10]   Procrustes disparity = {proc_results['procrustes_disparity']:.6f}")

    # -- Save summary TSV (CCA + Procrustes) --
    summary_records = []
    for i, cc in enumerate(cca_results["canonical_correlations"], 1):
        summary_records.append(
            dict(metric="canonical_correlation", component=i, value=cc)
        )
    summary_records.append(
        dict(
            metric="mean_canonical_correlation",
            component=np.nan,
            value=cca_results["mean_canonical_corr"],
        )
    )
    summary_records.append(
        dict(
            metric="procrustes_disparity",
            component=np.nan,
            value=proc_results["procrustes_disparity"],
        )
    )
    summary_records.append(
        dict(metric="n_overlapping_samples", component=np.nan, value=n_samples)
    )
    summary_records.append(
        dict(metric="n_ngs_pcs", component=np.nan, value=len(ngs_cols))
    )
    summary_records.append(
        dict(metric="n_array_pcs", component=np.nan, value=len(array_cols))
    )
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(output_dir, "crossmodality_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"[10]   → {summary_path}")

    # -- Analysis 4: Residual batch test --
    print("[10] Running residual batch association test …")
    resid_df = _compute_residual_batch(merged, ngs_cols, array_cols)
    resid_path = os.path.join(output_dir, "crossmodality_residual_batch.tsv")
    resid_df.to_csv(resid_path, sep="\t", index=False)
    print(f"[10]   → {resid_path}")

    # -- Analysis 5: Overlay scatter plots --
    print("[10] Generating overlay scatter plots …")
    scatter_path = os.path.join(output_dir, "crossmodality_scatter.png")
    _plot_overlay_scatter(merged, scatter_path)
    print(f"[10]   → {scatter_path}")

    print("[10] Cross-modality benchmark complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-modality validation: NGS-PCA vs. array-based PCs",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"),
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("NGSPCA_DATA_DIR", "1000G"),
    )
    parser.add_argument(
        "--n-pcs", type=int, default=0,
        help="Number of NGS-PCs to use (0 = Marchenko–Pastur auto-select)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    crossmodality_benchmark(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        n_pcs=args.n_pcs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
