#!/usr/bin/env python3
"""11_reference_bias_audit.py — Audit genome-wide reference-genome bias using QC metrics.

GRCh38 is EUR-dominated, and residual reference-genome (mappability) bias
may cause ancestry-correlated gradients in coverage summary statistics even
after bin curation.  This script uses per-sample genome-wide and autosomal QC
metrics as proxies for such bias, without returning to single-bin granularity.

Analyses
--------
1. **OLS regression** of each key QC metric on SUPERPOPULATION + RELEASE_BATCH.
   Significant SUPERPOPULATION terms after adjusting for batch signal possible
   reference-driven bias.  Partial η² quantifies effect sizes.
2. **Feature–feature correlation matrix** — Pearson correlations among NGS PCs,
   array-based genotype PCs, and coverage QC metrics.  Hierarchical clustering
   reveals whether QC metrics co-cluster with ancestry-sensitive PCs.

Outputs (written to *output_dir*)
----------------------------------
- ``reference_bias_regression.tsv``   — per-metric regression coefficients and
  partial η² for SUPERPOPULATION and RELEASE_BATCH.
- ``reference_bias_feature_corr.tsv`` — pairwise absolute Pearson correlation
  matrix among NGS PCs, array PCs, and QC metrics.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(__file__))
from utils import eta_squared, marchenko_pastur_pc_count_from_data_dir


# ---------------------------------------------------------------------------
# QC metrics used throughout the audit
# ---------------------------------------------------------------------------
QC_METRICS = [
    "MEAN_AUTOSOMAL_COV",
    "MEDIAN_GENOME_COV",
    "SD_COV",
    "MAD_COV",
    "IQR_COV",
    "HQ_MEDIAN_COV",
    "HQ_IQR_COV",
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_ngs_merged(output_dir: str) -> pd.DataFrame:
    """Load merged NGS-PCA + QC data."""
    path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    df = pd.read_csv(path, sep="\t")
    df["RELEASE_BATCH"] = df["RELEASE_BATCH"].astype(str)
    return df


def _load_array_pcs(data_dir: str) -> pd.DataFrame | None:
    """Load array-based ancestry PCs from compiled_sample_sheet.tsv.

    Returns None if the file is missing.
    """
    path = os.path.join(
        data_dir, "illumina_idat_processing", "compiled_sample_sheet.tsv",
    )
    if not os.path.isfile(path):
        return None
    cols_needed = ["sample_id", "pre_pca_excluded"] + [
        f"PC{i}" for i in range(1, 21)
    ]
    df = pd.read_csv(path, sep="\t", usecols=cols_needed)
    # Remove stray header rows that appear when multiple cohort files are
    # concatenated (the duplicated header has sample_id='IID').
    df = df[df["sample_id"] != "IID"].copy()
    df["pre_pca_excluded"] = pd.to_numeric(df["pre_pca_excluded"], errors="coerce")
    df = df[df["pre_pca_excluded"] == 0].copy()
    df = df.drop(columns=["pre_pca_excluded"])
    for i in range(1, 21):
        df[f"PC{i}"] = pd.to_numeric(df[f"PC{i}"], errors="coerce")
    rename = {f"PC{i}": f"ARRAY_PC{i}" for i in range(1, 21)}
    df = df.rename(columns=rename)
    return df


# ---------------------------------------------------------------------------
# Analysis 1: OLS regression of QC metrics ~ SUPERPOPULATION + RELEASE_BATCH
# ---------------------------------------------------------------------------

def compute_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Regress each QC metric on SUPERPOPULATION and RELEASE_BATCH.

    Returns a DataFrame with columns:
        metric, predictor, partial_eta2, f_stat, p_value, df_between, df_within
    """
    available_metrics = [m for m in QC_METRICS if m in df.columns]
    rows = []
    for metric in available_metrics:
        valid = df[metric].notna() & df["SUPERPOPULATION"].notna() & df["RELEASE_BATCH"].notna()
        sub = df.loc[valid].copy()
        if len(sub) < 20:
            continue
        y = sub[metric].values.astype(float)

        for predictor in ["SUPERPOPULATION", "RELEASE_BATCH"]:
            # Partial η²: residualise y for the other predictor, then compute η²
            other = "RELEASE_BATCH" if predictor == "SUPERPOPULATION" else "SUPERPOPULATION"
            # Group-centre y by the 'other' variable to partial it out
            other_means = sub.groupby(other)[metric].transform("mean").values
            y_adj = y - other_means

            eta2 = eta_squared(sub[predictor], y_adj)

            # F-test on adjusted values
            groups = [y_adj[sub[predictor].values == g]
                      for g in sub[predictor].unique() if len(y_adj[sub[predictor].values == g]) > 0]
            if len(groups) >= 2:
                f_stat, p_val = sp_stats.f_oneway(*groups)
                f_stat = float(f_stat) if np.isfinite(f_stat) else np.nan
                p_val = float(p_val) if np.isfinite(p_val) else 1.0
            else:
                f_stat, p_val = np.nan, 1.0

            k = len(groups)
            n = len(y_adj)
            rows.append({
                "metric": metric,
                "predictor": predictor,
                "partial_eta2": float(eta2) if np.isfinite(eta2) else 0.0,
                "f_stat": f_stat,
                "p_value": p_val,
                "df_between": k - 1,
                "df_within": n - k,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 2: Feature–feature correlation matrix
# ---------------------------------------------------------------------------

def compute_feature_correlation(
    df: pd.DataFrame,
    array_df: pd.DataFrame | None,
    ngs_pc_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Compute absolute Pearson correlation among NGS PCs, array PCs, and QC metrics.

    Returns (correlation DataFrame, ordered feature list).
    """
    features = {}

    # NGS PCs
    for col in ngs_pc_cols:
        if col in df.columns:
            features[col] = df[col].values.astype(float)

    # QC metrics
    available_metrics = [m for m in QC_METRICS if m in df.columns]
    for m in available_metrics:
        features[m] = pd.to_numeric(df[m], errors="coerce").values

    # Array PCs (up to 10 to keep heatmap readable)
    n_array_pcs = 0
    if array_df is not None:
        merged = df[["SAMPLE"]].merge(array_df, left_on="SAMPLE", right_on="sample_id", how="inner")
        if len(merged) > 20:
            for i in range(1, 11):
                col = f"ARRAY_PC{i}"
                if col in merged.columns:
                    vals = pd.to_numeric(merged[col], errors="coerce").values
                    if not np.all(np.isnan(vals)):
                        # Align to full df by reindex
                        arr_map = dict(zip(merged["SAMPLE"], vals))
                        features[col] = np.array([
                            arr_map.get(s, np.nan) for s in df["SAMPLE"]
                        ])
                        n_array_pcs += 1

    # Filter out constant features (zero variance → undefined correlation)
    feature_names = [
        k for k, v in features.items()
        if np.nanstd(v) > 0
    ]
    features = {k: features[k] for k in feature_names}

    # Build correlation matrix
    n = len(feature_names)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            xi = features[feature_names[i]]
            xj = features[feature_names[j]]
            mask = ~(np.isnan(xi) | np.isnan(xj))
            if mask.sum() >= 2:
                r, _ = sp_stats.pearsonr(xi[mask], xj[mask])
                mat[i, j] = abs(r)

    corr_df = pd.DataFrame(mat, index=feature_names, columns=feature_names)
    return corr_df, feature_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_audit(data_dir: str, output_dir: str) -> None:
    """Run the full reference-bias audit and write outputs."""
    print("[11] Loading merged data …")
    df = _load_ngs_merged(output_dir)

    # Determine MP-selected PCs
    sv_path = os.path.join(data_dir, "ngspca_output", "svd.singularvalues.txt")
    sv_df = pd.read_csv(sv_path, sep="\t")
    eigenvalues = sv_df["SINGULAR_VALUES"].values ** 2
    mp_pcs, _ = marchenko_pastur_pc_count_from_data_dir(data_dir, eigenvalues)
    ngs_pc_cols = [f"PC{i}" for i in range(1, mp_pcs + 1)]
    ngs_pc_cols = [c for c in ngs_pc_cols if c in df.columns]

    # --- Regression ---
    print("[11] Computing OLS regressions of QC metrics ~ SUPERPOPULATION + RELEASE_BATCH …")
    reg_df = compute_regression(df)
    reg_path = os.path.join(output_dir, "reference_bias_regression.tsv")
    reg_df.to_csv(reg_path, sep="\t", index=False)
    print(f"[11]   → {reg_path}  ({len(reg_df)} rows)")

    n_sig = (reg_df["p_value"] < 0.05).sum()
    print(f"[11]   {n_sig}/{len(reg_df)} predictor–metric pairs significant at p < 0.05")

    # --- Feature correlation ---
    print("[11] Computing feature–feature correlation matrix …")
    array_df = _load_array_pcs(data_dir)
    corr_df, feature_names = compute_feature_correlation(df, array_df, ngs_pc_cols)
    corr_path = os.path.join(output_dir, "reference_bias_feature_corr.tsv")
    corr_df.to_csv(corr_path, sep="\t")
    print(f"[11]   → {corr_path}  ({len(feature_names)} features)")

    print("[11] Reference bias audit complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit genome-wide reference-genome bias using QC metrics",
    )
    parser.add_argument("--data-dir",
                        default=os.environ.get("NGSPCA_DATA_DIR", "1000G"))
    parser.add_argument("--output-dir",
                        default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_audit(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
