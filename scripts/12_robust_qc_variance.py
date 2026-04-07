#!/usr/bin/env python3
"""12_robust_qc_variance.py — Variance partitioning of QC coverage metrics.

Extends the Reference-Genome Bias Audit by robustly quantifying how much
variance in genome-wide coverage metrics (mean/median, MAD/IQR) is explained
by ancestry (SUPERPOPULATION) after conditioning on batch (RELEASE_BATCH),
and vice versa.

For each coverage QC metric, fits the OLS model::

    QC_metric ~ RELEASE_BATCH + SUPERPOPULATION

Then computes (via leave-one-out R² differences, i.e. Type III sums of
squares):

  - Unique R² for ancestry   (R²_full − R²_batch-only)
  - Unique R² for batch      (R²_full − R²_ancestry-only)
  - Shared / confounded      = max(0, R²_full − unique_ancestry − unique_batch)
  - Residual                 = 1 − R²_full

This is analogous to ``08_variance_partitioning.py`` but applied to QC
coverage metrics rather than PCs, directly answering whether ancestry-
correlated coverage gradients survive after conditioning on batch.

Outputs (written to *output_dir*)
----------------------------------
- ``robust_qc_variance.tsv``  — per-metric variance decomposition.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# QC metrics to partition
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
# Linear-model helper (identical to 08_variance_partitioning)
# ---------------------------------------------------------------------------

def _fit_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Fit OLS with an implicit intercept and return R² ∈ [0, 1]."""
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    X_int = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
    y_pred = X_int @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def robust_qc_variance(output_dir: str) -> pd.DataFrame:
    """Compute leave-one-out R² decomposition for each QC metric.

    Parameters
    ----------
    output_dir : str
        Directory containing ``merged_pcs_qc.tsv`` (pipeline outputs).

    Returns
    -------
    pd.DataFrame
        One row per QC metric with columns for each variance component.
    """
    merged_path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    df = pd.read_csv(merged_path, sep="\t")
    df["RELEASE_BATCH"] = df["RELEASE_BATCH"].astype(str)

    available_metrics = [m for m in QC_METRICS if m in df.columns]

    # Filter to samples with valid predictors
    valid = df["SUPERPOPULATION"].notna() & df["RELEASE_BATCH"].notna()
    sub = df.loc[valid].reset_index(drop=True)
    print(f"[12] {valid.sum()} of {len(df)} samples have valid ancestry + batch")

    # One-hot encode predictors
    X_batch = pd.get_dummies(sub["RELEASE_BATCH"], drop_first=True,
                              prefix="BATCH").values.astype(float)
    X_anc = pd.get_dummies(sub["SUPERPOPULATION"], drop_first=True,
                            prefix="POP").values.astype(float)
    X_full = np.column_stack([X_batch, X_anc])

    records = []
    for metric in available_metrics:
        metric_valid = sub[metric].notna()
        s = sub.loc[metric_valid].reset_index(drop=True)
        if len(s) < 20:
            continue

        y = s[metric].values.astype(float)
        Xb = pd.get_dummies(s["RELEASE_BATCH"], drop_first=True,
                             prefix="BATCH").values.astype(float)
        Xa = pd.get_dummies(s["SUPERPOPULATION"], drop_first=True,
                             prefix="POP").values.astype(float)
        Xf = np.column_stack([Xb, Xa])

        r2_full = _fit_r2(Xf, y)
        r2_batch_only = _fit_r2(Xb, y)
        r2_anc_only = _fit_r2(Xa, y)

        unique_ancestry = max(0.0, r2_full - r2_batch_only)
        unique_batch = max(0.0, r2_full - r2_anc_only)
        shared = max(0.0, r2_full - unique_ancestry - unique_batch)
        residual = max(0.0, 1.0 - r2_full)

        records.append({
            "metric": metric,
            "r2_full": r2_full,
            "unique_ancestry": unique_ancestry,
            "unique_batch": unique_batch,
            "shared": shared,
            "residual": residual,
            "n_samples": len(s),
        })

    result = pd.DataFrame(records)

    # Write TSV
    tsv_path = os.path.join(output_dir, "robust_qc_variance.tsv")
    result.to_csv(tsv_path, sep="\t", index=False)
    print(f"[12] Variance partitioning of QC metrics → {tsv_path}")

    # Summary
    if len(result) > 0:
        mean_anc = result["unique_ancestry"].mean()
        mean_batch = result["unique_batch"].mean()
        print(f"[12]   Mean unique ancestry R²: {mean_anc:.4f}")
        print(f"[12]   Mean unique batch R²:    {mean_batch:.4f}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Variance partitioning of QC coverage metrics (ancestry vs batch)",
    )
    parser.add_argument("--output-dir",
                        default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    robust_qc_variance(args.output_dir)


if __name__ == "__main__":
    main()
