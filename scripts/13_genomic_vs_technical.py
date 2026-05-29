#!/usr/bin/env python3
"""13_genomic_vs_technical.py — The decisive test: real genomic ancestry vs. technical artefact.

This script answers a single, intuitive question for every NGS-PCA principal
component:

    Does the PC's correlation with *genuine* genomic ancestry survive after we
    remove everything the *technical* coverage/QC metrics can explain?

Ancestry is, by definition, a property of the genome.  The gold-standard,
coverage-independent measurement of ancestry is the **SNP-array genotype PCs**
(block ``G``), which are computed from allele frequencies and are immune to
read-depth artefacts.  Technical / batch variation is summarised by the
**coverage QC metrics** (block ``T``).  For each NGS-PCA PC we fit three nested
ordinary-least-squares models and decompose the variance via *commonality
analysis* (Type-III / leave-one-block-out R² differences):

    R²_G   = PC ~ G            (genotype-ancestry signal, in total)
    R²_T   = PC ~ T            (technical signal, in total)
    R²_GT  = PC ~ G + T        (both blocks together)

    unique_genomic    U_G = R²_GT − R²_T   (ancestry signal NOT explainable by QC)
    unique_technical  U_T = R²_GT − R²_G   (technical signal NOT explainable by ancestry)
    shared            C   = R²_G + R²_T − R²_GT  (ancestry–QC confounded variance)

The decisive scalar is the **genomic retention** of each PC:

    retention = U_G / R²_G

    retention ≈ 1  → the ancestry correlation is *irreducible genomic signal*
                     (QC cannot account for it)  → the PC captures ancestry
                     directly via real genomic signal.
    retention ≈ 0  → the ancestry correlation is *entirely mediated by QC*
                     → the PC captures coverage/batch artefacts that merely
                     correlate with ancestry, not genomic ancestry itself.

To make ``U_G`` robust to the number of genotype predictors (adding any block
of predictors inflates R²), we build a **permutation null**: the genotype block
is row-shuffled relative to the PC + QC data, breaking the genomic association
while preserving the technical association and the predictor count.  This yields
an empirical one-sided p-value for ``U_G`` and a bias-corrected estimate
``unique_genomic_corrected = max(0, U_G − mean(U_G_null))``.

Outputs (written to *output_dir*)
----------------------------------
- ``genomic_vs_technical.tsv``  — per-PC commonality decomposition + p-values
- ``genomic_vs_technical.png``  — two-panel decisive figure
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

# Candidate technical / coverage QC metrics (block T).  Columns that are absent
# or (near-)constant in the cohort are dropped automatically.
QC_METRICS = [
    "MEAN_AUTOSOMAL_COV", "MEDIAN_GENOME_COV", "MEDIAN_BIN_COV",
    "HQ_MEDIAN_COV", "SD_COV", "MAD_COV", "IQR_COV",
    "HQ_SD_COV", "HQ_MAD_COV", "HQ_IQR_COV",
    "PCT_GENOME_COV_10X", "PCT_GENOME_COV_20X",
    "X_COV_RATIO", "Y_COV_RATIO", "MITO_COV_RATIO", "MTDNA_CN",
]


# ---------------------------------------------------------------------------
# Data loading (mirrors scripts/10_crossmodality_benchmark.py conventions)
# ---------------------------------------------------------------------------

def _load_array_pcs(data_dir: str, n_array_pcs: int) -> pd.DataFrame | None:
    """Load array-based genotype ancestry PCs (the genomic ground truth).

    Returns a DataFrame with columns ``sample_id`` and ``ARRAY_PC1 … ARRAY_PCk``
    for non-excluded samples, or ``None`` if the file is unavailable.
    """
    path = os.path.join(
        data_dir, "illumina_idat_processing", "compiled_sample_sheet.tsv"
    )
    if not os.path.isfile(path):
        return None
    cols_needed = ["sample_id", "pre_pca_excluded"] + [
        f"PC{i}" for i in range(1, n_array_pcs + 1)
    ]
    df = pd.read_csv(path, sep="\t", usecols=cols_needed)
    df = df[df["sample_id"] != "IID"].copy()
    df["pre_pca_excluded"] = pd.to_numeric(df["pre_pca_excluded"], errors="coerce")
    df = df[df["pre_pca_excluded"] == 0].copy()
    df = df.drop(columns=["pre_pca_excluded"])
    for i in range(1, n_array_pcs + 1):
        df[f"PC{i}"] = pd.to_numeric(df[f"PC{i}"], errors="coerce")
    rename = {f"PC{i}": f"ARRAY_PC{i}" for i in range(1, n_array_pcs + 1)}
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Linear-model helpers
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


def _standardize(mat: np.ndarray) -> np.ndarray:
    """Z-score each column; constant columns become all-zero."""
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd_safe = np.where(sd < 1e-12, 1.0, sd)
    out = (mat - mu) / sd_safe
    out[:, sd < 1e-12] = 0.0
    return out


def _orthonormal_residual_basis(G: np.ndarray, M_T: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the part of ``G`` that is orthogonal to ``M_T``.

    ``M_T`` already includes an intercept column.  The returned matrix ``Q`` has
    orthonormal columns spanning the residualised genotype block, so the unique
    genomic R² of a vector ``e`` (itself residualised on ``M_T``) is simply
    ``||Qᵀ e||² / SS_tot``.  Row-permutations of ``Q`` preserve orthonormality,
    which makes the permutation null both exact and fast.
    """
    # Residualise each column of G on M_T.
    coef, _, _, _ = np.linalg.lstsq(M_T, G, rcond=None)
    G_res = G - M_T @ coef
    # Orthonormal basis via QR; drop numerically-rank-deficient columns.
    q, r = np.linalg.qr(G_res)
    keep = np.abs(np.diag(r)) > 1e-8
    return q[:, keep]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def genomic_vs_technical(
    output_dir: str,
    data_dir: str = "1000G",
    n_pcs: int = 0,
    n_array_pcs: int = 10,
    n_permutations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame | None:
    """Run the genomic-vs-technical commonality test and write outputs.

    Returns the per-PC results DataFrame, or ``None`` when the array genotype
    PCs are unavailable (in which case no outputs are written).
    """
    # ---- Load NGS-PCA + QC -------------------------------------------------
    merged_path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    df = pd.read_csv(merged_path, sep="\t")

    # ---- Load genotype-array ancestry PCs (genomic ground truth) -----------
    array = _load_array_pcs(data_dir, n_array_pcs)
    if array is None:
        print("[13] Array genotype PCs unavailable — skipping genomic-vs-technical test")
        return None

    merged = df.merge(array, left_on="SAMPLE", right_on="sample_id", how="inner")
    merged = merged.drop(columns=["sample_id"])

    array_cols = [c for c in (f"ARRAY_PC{i}" for i in range(1, n_array_pcs + 1))
                  if c in merged.columns]
    qc_cols = [c for c in QC_METRICS if c in merged.columns]

    # ---- Determine NGS PCs via Marchenko–Pastur ----------------------------
    sv_path = os.path.join(data_dir, "ngspca_output", "svd.singularvalues.txt")
    eigenvalues = pd.read_csv(sv_path, sep="\t")["SINGULAR_VALUES"].values ** 2
    max_pcs = n_pcs if n_pcs > 0 else None
    mp_n_pcs, _ = marchenko_pastur_pc_count_from_data_dir(
        data_dir, eigenvalues, max_pcs=max_pcs,
    )
    pc_cols = [f"PC{i}" for i in range(1, mp_n_pcs + 1) if f"PC{i}" in merged.columns]

    # ---- Drop rows with any missing predictor / response -------------------
    needed = pc_cols + array_cols + qc_cols
    valid = merged[needed].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    sub = merged.loc[valid].reset_index(drop=True)
    n_samples = int(len(sub))
    print(f"[13] {n_samples} samples overlap NGS-PCA, array genotype PCs and QC metrics")
    print(f"[13] Using {len(pc_cols)} MP-selected NGS PCs · "
          f"{len(array_cols)} array PCs (G) · {len(qc_cols)} QC metrics (T)")
    if n_samples < 20 or len(array_cols) < 2 or len(qc_cols) < 2:
        print("[13] Insufficient overlapping data — skipping")
        return None

    # ---- Build standardised predictor blocks -------------------------------
    G = _standardize(sub[array_cols].values.astype(float))
    # Drop QC columns that are constant within the analysed sample set.
    T_raw = sub[qc_cols].values.astype(float)
    nonconst = T_raw.std(axis=0) > 1e-12
    qc_cols = [c for c, k in zip(qc_cols, nonconst) if k]
    T = _standardize(T_raw[:, nonconst])

    # Orthonormal basis of G residualised on [1, T] — used for the fast,
    # exact permutation null of the unique genomic component.
    M_T = np.column_stack([np.ones(n_samples), T])
    Q_Gres = _orthonormal_residual_basis(G, M_T)

    rng = np.random.default_rng(seed)
    perm_indices = [rng.permutation(n_samples) for _ in range(n_permutations)]

    # ---- Per-PC commonality decomposition ----------------------------------
    records = []
    for pc in pc_cols:
        y = sub[pc].values.astype(float)
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))

        r2_g = _fit_r2(G, y)
        r2_t = _fit_r2(T, y)
        r2_gt = _fit_r2(np.column_stack([G, T]), y)

        unique_genomic = max(0.0, r2_gt - r2_t)
        unique_technical = max(0.0, r2_gt - r2_g)
        shared = r2_g + r2_t - r2_gt  # may be slightly negative (suppression)
        retention = unique_genomic / r2_g if r2_g > 1e-12 else np.nan

        # Permutation null on the unique genomic component.  Residualise y on
        # [1, T] once, then project the (row-permuted) genomic residual basis.
        coef_y, _, _, _ = np.linalg.lstsq(M_T, y, rcond=None)
        e_y = y - M_T @ coef_y  # residual of PC after removing technical block
        if Q_Gres.shape[1] == 0 or ss_tot == 0.0 or n_permutations == 0:
            p_value = np.nan
            mean_null = np.nan
            corrected = unique_genomic
        else:
            obs_proj = float(np.sum((Q_Gres.T @ e_y) ** 2)) / ss_tot
            null = np.empty(n_permutations)
            for i, pi in enumerate(perm_indices):
                proj = Q_Gres[pi].T @ e_y
                null[i] = float(np.sum(proj ** 2)) / ss_tot
            p_value = float((np.sum(null >= obs_proj) + 1) / (n_permutations + 1))
            mean_null = float(np.mean(null))
            corrected = max(0.0, unique_genomic - mean_null)

        records.append({
            "PC": pc,
            "r2_genomic": r2_g,
            "r2_technical": r2_t,
            "r2_full": r2_gt,
            "unique_genomic": unique_genomic,
            "unique_technical": unique_technical,
            "shared": shared,
            "retention": retention,
            "unique_genomic_corrected": corrected,
            "mean_null_unique_genomic": mean_null,
            "p_value": p_value,
            "n_array_pcs": len(array_cols),
            "n_qc_metrics": len(qc_cols),
            "n_samples": n_samples,
            "n_permutations": n_permutations,
        })

    result = pd.DataFrame(records)

    os.makedirs(output_dir, exist_ok=True)
    tsv_path = os.path.join(output_dir, "genomic_vs_technical.tsv")
    result.to_csv(tsv_path, sep="\t", index=False)
    print(f"[13] Per-PC commonality decomposition → {tsv_path}")

    n_sig = int((result["p_value"] < 0.05).sum())
    print(f"[13]   {n_sig}/{len(result)} PCs carry significant irreducible "
          f"genomic ancestry signal (unique genomic p < 0.05)")
    print(f"[13]   mean genomic retention = {np.nanmean(result['retention']):.3f}")

    _plot(result, output_dir)
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COL_GENOMIC = "#1B9E77"
_COL_TECHNICAL = "#D95F02"
_COL_SHARED = "#A6A6A6"


def _sig_label(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _plot(result: pd.DataFrame, output_dir: str) -> None:
    pcs = result["PC"].tolist()
    x = np.arange(len(pcs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(pcs) * 0.8), 5))

    # --- Panel A: commonality decomposition (stacked) -----------------------
    ug = result["unique_genomic"].values
    ut = result["unique_technical"].values
    sh = np.clip(result["shared"].values, 0.0, None)
    ax1.bar(x, ug, color=_COL_GENOMIC, label="Unique genomic (real ancestry)")
    ax1.bar(x, sh, bottom=ug, color=_COL_SHARED, label="Shared / confounded")
    ax1.bar(x, ut, bottom=ug + sh, color=_COL_TECHNICAL, label="Unique technical (QC/batch)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pcs, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Variance explained (R²)")
    ax1.set_title("Variance partition per PC\n(genotype ancestry vs. coverage QC)")
    ax1.legend(fontsize=8, framealpha=0.9)

    # --- Panel B: ancestry correlation before vs. after QC adjustment -------
    width = 0.4
    ax2.bar(x - width / 2, result["r2_genomic"].values, width,
            color=_COL_GENOMIC, alpha=0.45,
            label="R² genotype-ancestry (raw)")
    ax2.bar(x + width / 2, result["unique_genomic"].values, width,
            color=_COL_GENOMIC,
            label="surviving QC adjustment (unique genomic)")
    ymax = max(1e-6, float(result["r2_genomic"].max()))
    for xi, (_, row) in zip(x, result.iterrows()):
        lbl = _sig_label(row["p_value"])
        if lbl:
            ax2.text(xi + width / 2, row["unique_genomic"] + 0.02 * ymax, lbl,
                     ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pcs, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Genotype-ancestry R²")
    ax2.set_title("Does the ancestry signal survive QC adjustment?\n"
                  "(*** p<0.001 ** p<0.01 * p<0.05 ns)")
    ax2.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "genomic_vs_technical.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[13] Decisive figure → {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genomic (real ancestry) vs. technical (QC) commonality test per PC",
    )
    parser.add_argument("--output-dir",
                        default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    parser.add_argument("--data-dir",
                        default=os.environ.get("NGSPCA_DATA_DIR", "1000G"))
    parser.add_argument("--n-pcs", type=int, default=0,
                        help="Number of NGS PCs (0 = Marchenko–Pastur auto)")
    parser.add_argument("--n-array-pcs", type=int, default=10,
                        help="Number of genotype-array PCs to use as the genomic block")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Permutations for the unique-genomic null (0 = skip)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    genomic_vs_technical(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        n_pcs=args.n_pcs,
        n_array_pcs=args.n_array_pcs,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
