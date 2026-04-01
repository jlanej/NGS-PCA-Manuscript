#!/usr/bin/env python3
"""06_interactive_report.py - Generate an interactive single-page HTML report.

Reads all pipeline outputs and produces a single self-contained HTML file
with interactive Plotly charts and statistical tables covering:
  - Introduction to NGS-PCA methodology
  - Scree / cumulative variance with Marchenko–Pastur cutoff
  - PCA scatter plots (PC1-PC2, PC3-PC4)
  - UMAP projection
  - Confounding assessment (batch × population, batch × sex)
  - PC × QC association heatmap (η², Cramér's V, point-biserial r)
  - Sex association across PCs
  - Batch vs ancestry comparison
  - Summary statistics tables

The report is written to ``docs/index.html`` by default.
"""

import argparse
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import umap as umap_module

sys.path.insert(0, os.path.dirname(__file__))
from utils import eta_squared, marchenko_pastur_pc_count_from_data_dir


# ---------------------------------------------------------------------------
# Colour palettes (matching the static plots)
# ---------------------------------------------------------------------------
PALETTE_SUPERPOP = {
    "AFR": "#E41A1C", "AMR": "#377EB8", "EAS": "#4DAF4A",
    "EUR": "#984EA3", "SAS": "#FF7F00",
}
PALETTE_BATCH = {"698": "#1B9E77", "2504": "#D95F02"}
PALETTE_SEX = {"M": "#4393C3", "F": "#D6604D"}
POP_TO_SUPERPOP = {
    "ACB": "AFR", "ASW": "AFR", "ESN": "AFR", "GWD": "AFR",
    "LWK": "AFR", "MSL": "AFR", "YRI": "AFR",
    "CLM": "AMR", "MXL": "AMR", "PEL": "AMR", "PUR": "AMR",
    "CDX": "EAS", "CHB": "EAS", "CHS": "EAS", "JPT": "EAS", "KHV": "EAS",
    "CEU": "EUR", "FIN": "EUR", "GBR": "EUR", "IBS": "EUR", "TSI": "EUR",
    "BEB": "SAS", "GIH": "SAS", "ITU": "SAS", "PJL": "SAS", "STU": "SAS",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_singular_values(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "ngspca_output", "svd.singularvalues.txt")
    return pd.read_csv(path, sep="\t")


def _load_merged(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    df = pd.read_csv(path, sep="\t")
    df["RELEASE_BATCH"] = df["RELEASE_BATCH"].astype(str)
    return df


def _load_full_merged_from_data_dir(data_dir: str) -> pd.DataFrame:
    pcs_path = os.path.join(data_dir, "ngspca_output", "svd.pcs.txt")
    qc_path = os.path.join(data_dir, "qc_output", "sample_qc.tsv")
    pcs = pd.read_csv(pcs_path, sep="\t")
    pcs["SAMPLE"] = pcs["SAMPLE"].str.replace(r"\.by1000\.$", "", regex=True)
    qc = pd.read_csv(qc_path, sep="\t")
    if "SUPERPOPULATION" in qc.columns:
        qc = qc.rename(columns={"SUPERPOPULATION": "FAMILY_ROLE"})
    qc["SUPERPOPULATION"] = qc["POPULATION"].map(POP_TO_SUPERPOP)
    merged = pcs.merge(qc, left_on="SAMPLE", right_on="SAMPLE_ID", how="inner")
    if "SAMPLE_ID" in merged.columns:
        merged = merged.drop(columns=["SAMPLE_ID"])
    merged["RELEASE_BATCH"] = merged["RELEASE_BATCH"].astype(str)
    return merged


def _count_available_samples(data_dir: str) -> int:
    pcs_path = os.path.join(data_dir, "ngspca_output", "svd.pcs.txt")
    qc_path = os.path.join(data_dir, "qc_output", "sample_qc.tsv")
    pcs = pd.read_csv(pcs_path, sep="\t")
    qc = pd.read_csv(qc_path, sep="\t")
    pcs_samples = set(pcs["SAMPLE"].str.replace(r"\.by1000\.$", "", regex=True))
    qc_samples = set(qc["SAMPLE_ID"])
    return len(pcs_samples & qc_samples)


def _compute_variance(sv_df: pd.DataFrame, n_pcs: int = 0):
    eigenvalues = sv_df["SINGULAR_VALUES"].values ** 2
    total = eigenvalues.sum()
    prop = eigenvalues / total
    cum = np.cumsum(prop)
    n = len(eigenvalues) if n_pcs <= 0 else min(n_pcs, len(eigenvalues))
    return prop[:n].tolist(), cum[:n].tolist(), n


def _compute_umap(df: pd.DataFrame, n_pcs: int):
    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    available = [c for c in pc_cols if c in df.columns]
    X = df[available].values
    reducer = umap_module.UMAP(
        n_components=2, random_state=42, n_neighbors=30, min_dist=0.3,
    )
    emb = reducer.fit_transform(X)
    return emb[:, 0].tolist(), emb[:, 1].tolist()


def _prepare_scatter_data(df: pd.DataFrame):
    """Return a lightweight dict for the scatter plots (PC1-10 + numerics)."""
    pc_cols = [f"PC{i}" for i in range(1, 11)]
    meta_cols = ["SAMPLE", "SUPERPOPULATION", "RELEASE_BATCH",
                 "INFERRED_SEX", "POPULATION", "RELATEDNESS", "FAMILY_ROLE"]
    numeric_cols = [
        "MEAN_AUTOSOMAL_COV", "X_COV_RATIO", "Y_COV_RATIO",
        "MITO_COV_RATIO", "MEDIAN_GENOME_COV",
        "PCT_GENOME_COV_10X", "PCT_GENOME_COV_20X",
        "SD_COV", "MAD_COV", "IQR_COV", "MEDIAN_BIN_COV",
        "HQ_MEDIAN_COV", "HQ_SD_COV", "HQ_MAD_COV", "HQ_IQR_COV",
        "MTDNA_CN",
    ]
    want = meta_cols + pc_cols + numeric_cols
    have = [c for c in want if c in df.columns]
    sub = df[have].copy()
    found_numeric = [c for c in numeric_cols if c in df.columns]
    return sub.to_dict(orient="list"), found_numeric


def _compute_associations(df: pd.DataFrame, n_pcs: int = 20):
    """Compute η²/r² associations and p-values for the heatmap.

    Categorical variables use η² (from one-way ANOVA F-test) with the
    corresponding p-value.  Continuous variables use Pearson r² with the
    two-sided t-test p-value from ``scipy.stats.pearsonr``.

    η² (eta-squared) is the proportion of variance in a PC score that is
    explained by the categorical grouping.  It is the natural analogue of
    R² for categorical predictors and equals SSbetween / SStotal from a
    one-way ANOVA.  Unlike Cramér's V (which operates on contingency
    tables of two categorical variables), η² directly measures how much
    of a *continuous* variable's spread is captured by group membership,
    making it the appropriate effect-size metric when the outcome (PC
    score) is numeric and the predictor is categorical.
    """
    from scipy import stats as sp_stats

    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    available = [c for c in pc_cols if c in df.columns]
    cat_vars = ["RELEASE_BATCH", "SUPERPOPULATION", "INFERRED_SEX",
                "POPULATION", "RELATEDNESS"]
    # Include only categorical vars actually present
    cat_vars = [v for v in cat_vars if v in df.columns]
    cont_vars = [
        "MEAN_AUTOSOMAL_COV", "X_COV_RATIO", "Y_COV_RATIO",
        "MITO_COV_RATIO", "MEDIAN_GENOME_COV",
        "PCT_GENOME_COV_10X", "PCT_GENOME_COV_20X",
        "SD_COV", "MAD_COV", "IQR_COV", "MEDIAN_BIN_COV",
        "HQ_MEDIAN_COV", "HQ_SD_COV", "HQ_MAD_COV", "HQ_IQR_COV",
        "MTDNA_CN",
    ]
    cont_vars = [v for v in cont_vars if v in df.columns]
    rows = []
    for var in cat_vars:
        valid = df[var].notna()
        if valid.sum() < 10:
            continue
        for pc in available:
            eta2 = eta_squared(df.loc[valid, var], df.loc[valid, pc].values)
            # ANOVA F-test for p-value
            groups = [
                df.loc[valid & (df[var] == g), pc].values
                for g in df.loc[valid, var].unique()
                if len(df.loc[valid & (df[var] == g)]) > 0
            ]
            if len(groups) >= 2:
                f_stat, p_val = sp_stats.f_oneway(*groups)
                p_val = float(p_val) if np.isfinite(p_val) else 1.0
            else:
                p_val = 1.0
            rows.append({"Variable": var, "PC": pc, "Value": float(eta2),
                         "Metric": "η²", "p_value": p_val})
    for var in cont_vars:
        valid = df[var].notna()
        if valid.sum() < 10:
            continue
        # Skip constant columns (e.g. all-zero metrics)
        if df.loc[valid, var].nunique() < 2:
            continue
        for pc in available:
            r, p_val = sp_stats.pearsonr(df.loc[valid, var], df.loc[valid, pc])
            rows.append({"Variable": var, "PC": pc, "Value": float(r ** 2),
                         "Metric": "r²", "p_value": float(p_val)})
    return rows, [c for c in available]


def _compute_batch_ancestry(df: pd.DataFrame, n_pcs: int = 20):
    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    available = [c for c in pc_cols if c in df.columns]
    records = []
    for pc in available:
        vb = df["RELEASE_BATCH"].notna()
        va = df["SUPERPOPULATION"].notna()
        eta_b = eta_squared(df.loc[vb, "RELEASE_BATCH"], df.loc[vb, pc].values)
        eta_a = eta_squared(df.loc[va, "SUPERPOPULATION"], df.loc[va, pc].values)
        records.append({"PC": pc,
                        "Batch_eta2": float(eta_b),
                        "Ancestry_eta2": float(eta_a)})
    return records, available


def _compute_confounding(df: pd.DataFrame):
    """Compute chi-squared tests for confounding between batch, population, and sex."""
    from scipy.stats import chi2_contingency

    results = []
    pairs = [
        ("RELEASE_BATCH", "SUPERPOPULATION", "Batch × Superpopulation"),
        ("RELEASE_BATCH", "INFERRED_SEX", "Batch × Sex"),
        ("INFERRED_SEX", "SUPERPOPULATION", "Sex × Superpopulation"),
    ]
    for var1, var2, label in pairs:
        valid = df[var1].notna() & df[var2].notna()
        sub = df.loc[valid, [var1, var2]]
        ct = pd.crosstab(sub[var1], sub[var2])
        chi2, p, dof, expected = chi2_contingency(ct)
        n = ct.values.sum()
        k = min(ct.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            std_resid = (ct.values - expected) / np.sqrt(expected)
            std_resid = np.where(np.isfinite(std_resid), std_resid, 0.0)
        results.append({
            "label": label,
            "var1": var1,
            "var2": var2,
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "cramers_v": cramers_v,
            "rows": [str(r) for r in ct.index.tolist()],
            "cols": [str(c) for c in ct.columns.tolist()],
            "observed": ct.values.tolist(),
            "expected": [[float(x) for x in row] for row in expected.tolist()],
            "std_residuals": [[float(x) for x in row] for row in std_resid.tolist()],
        })
    return results


def _compute_sex_pc_associations(df: pd.DataFrame, n_pcs: int = 20):
    """Compute point-biserial correlation and Kruskal-Wallis for sex across PCs."""
    from scipy import stats as sp_stats

    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    available = [c for c in pc_cols if c in df.columns]
    valid = df["INFERRED_SEX"].notna()
    sub = df.loc[valid].copy()
    sex_binary = (sub["INFERRED_SEX"] == "M").astype(float).values

    records = []
    for pc in available:
        pc_vals = sub[pc].values
        r_pb, p_pb = sp_stats.pointbiserialr(sex_binary, pc_vals)
        groups = [sub.loc[sub["INFERRED_SEX"] == g, pc].values
                  for g in ["M", "F"] if g in sub["INFERRED_SEX"].values]
        if len(groups) == 2:
            h_stat, p_kw = sp_stats.kruskal(*groups)
        else:
            h_stat, p_kw = np.nan, np.nan
        eta2 = eta_squared(sub["INFERRED_SEX"], pc_vals)
        records.append({
            "PC": pc,
            "eta2": float(eta2),
            "r_pointbiserial": float(r_pb),
            "p_pointbiserial": float(p_pb),
            "kruskal_H": float(h_stat) if np.isfinite(h_stat) else None,
            "p_kruskal": float(p_kw) if np.isfinite(p_kw) else None,
        })
    return records


def _compute_sample_summary(df: pd.DataFrame):
    """Compute dataset summary statistics."""
    spop_counts = df["SUPERPOPULATION"].value_counts().sort_index()
    batch_counts = df["RELEASE_BATCH"].value_counts().sort_index()
    sex_counts = df["INFERRED_SEX"].value_counts().sort_index()
    pop_counts = df["POPULATION"].value_counts().sort_index()
    return {
        "superpop_counts": {str(k): int(v) for k, v in spop_counts.items()},
        "batch_counts": {str(k): int(v) for k, v in batch_counts.items()},
        "sex_counts": {str(k): int(v) for k, v in sex_counts.items()},
        "pop_counts": {str(k): int(v) for k, v in pop_counts.items()},
        "n_total": int(len(df)),
    }


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Compute R² for OLS regression of *y* on *X* (no intercept in X).

    An intercept column is added automatically.
    """
    n = X.shape[0]
    X_int = np.column_stack([np.ones(n), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
        y_hat = X_int @ beta  # requires Python ≥3.5 / NumPy ≥1.10
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    except np.linalg.LinAlgError:
        return 0.0


def _compute_variance_partitioning(df: pd.DataFrame, n_pcs: int = 20):
    """Variance partitioning: unique batch, unique ancestry, shared per PC.

    Uses the inclusion/exclusion R² decomposition:
      unique_batch    = R²_full − R²_ancestry_only
      unique_ancestry = R²_full − R²_batch_only
      shared          = R²_batch + R²_ancestry − R²_full
      residual        = 1 − R²_full
    """
    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    available = [c for c in pc_cols if c in df.columns]

    valid = df["RELEASE_BATCH"].notna() & df["SUPERPOPULATION"].notna()
    sub = df.loc[valid].copy()

    batch_dum = pd.get_dummies(sub["RELEASE_BATCH"], prefix="b", dtype=float)
    anc_dum = pd.get_dummies(sub["SUPERPOPULATION"], prefix="a", dtype=float)
    # Drop one level per factor to avoid the dummy-variable trap
    # (perfect multicollinearity) in OLS regression.
    if batch_dum.shape[1] > 1:
        batch_dum = batch_dum.iloc[:, 1:]
    if anc_dum.shape[1] > 1:
        anc_dum = anc_dum.iloc[:, 1:]

    records = []
    for pc in available:
        y = sub[pc].values
        if np.sum((y - y.mean()) ** 2) == 0:
            records.append({"PC": pc, "unique_batch": 0, "unique_ancestry": 0,
                            "shared": 0, "residual": 1,
                            "r2_full": 0, "r2_batch": 0, "r2_ancestry": 0})
            continue
        X_full = np.column_stack([batch_dum.values, anc_dum.values])
        r2_full = _ols_r2(X_full, y)
        r2_batch = _ols_r2(batch_dum.values, y)
        r2_ancestry = _ols_r2(anc_dum.values, y)
        # Inclusion/exclusion R² decomposition (Venn-diagram approach):
        #   unique_batch    = R²_full − R²_ancestry   (batch adds beyond ancestry)
        #   unique_ancestry = R²_full − R²_batch       (ancestry adds beyond batch)
        #   shared          = R²_batch + R²_ancestry − R²_full  (confounded)
        unique_batch = r2_full - r2_ancestry
        unique_ancestry = r2_full - r2_batch
        shared = r2_batch + r2_ancestry - r2_full
        records.append({
            "PC": pc,
            "unique_batch": float(max(0, unique_batch)),
            "unique_ancestry": float(max(0, unique_ancestry)),
            "shared": float(max(0, shared)),
            "residual": float(1.0 - r2_full),
            "r2_full": float(r2_full),
            "r2_batch": float(r2_batch),
            "r2_ancestry": float(r2_ancestry),
        })
    return records


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _build_html(
    variance_prop,
    variance_cum,
    n_scree,
    mp_cutoff_pcs,
    scatter_data,
    numeric_cols,
    umap1,
    umap2,
    n_umap_pcs,
    assoc_rows,
    assoc_pcs,
    batch_records,
    batch_pcs,
    confounding_results,
    sex_pc_records,
    sample_summary,
    variance_partitioning,
    n_samples,
    n_populations,
    n_superpops,
):
    """Return a complete HTML string with embedded Plotly charts."""

    # Convert NaN to null for JSON
    def _clean(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    data_json = json.dumps(_clean({
        "variance_prop": variance_prop,
        "variance_cum": variance_cum,
        "n_scree": n_scree,
        "mp_cutoff_pcs": mp_cutoff_pcs,
        "scatter": scatter_data,
        "numeric_cols": numeric_cols,
        "umap1": umap1,
        "umap2": umap2,
        "n_umap_pcs": n_umap_pcs,
        "assoc_rows": assoc_rows,
        "assoc_pcs": assoc_pcs,
        "batch_records": batch_records,
        "batch_pcs": batch_pcs,
        "confounding": confounding_results,
        "sex_pc": sex_pc_records,
        "sample_summary": sample_summary,
        "variance_partitioning": variance_partitioning,
        "n_samples": n_samples,
        "n_populations": n_populations,
        "n_superpops": n_superpops,
    }))

    html = textwrap.dedent("""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NGS-PCA Interactive Report</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
    <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface2: #f1f5f9;
      --border: #cbd5e1;
      --text: #0f172a;
      --text-dim: #475569;
      --accent: #0ea5e9;
      --accent2: #6366f1;
      --green: #059669;
      --amber: #d97706;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }
    .header {
      background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
      border-bottom: 1px solid var(--border);
      padding: 2rem 2rem 1rem;
    }
    .header h1 {
      font-size: 1.75rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 0.25rem;
    }
    .header p { color: var(--text-dim); font-size: 0.95rem; }
    .stats-bar {
      display: flex;
      gap: 1.5rem;
      margin-top: 1.25rem;
      flex-wrap: wrap;
    }
    .stat-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.75rem 1.25rem;
      min-width: 140px;
    }
    .stat-card .label { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-card .value { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
    nav.toc {
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      gap: 0;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      overflow-x: auto;
      padding: 0 1rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    nav.toc a {
      color: var(--text-dim);
      padding: 0.75rem 1rem;
      font-size: 0.85rem;
      text-decoration: none;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
      white-space: nowrap;
    }
    nav.toc a:hover { color: var(--text); background: var(--surface2); }
    nav.toc a.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
      font-weight: 600;
    }
    .report-section {
      padding: 2rem 2rem 2.5rem;
      max-width: 1400px;
      margin: 0 auto;
      scroll-margin-top: 48px;
    }
    .report-section + .report-section {
      border-top: 1px solid var(--border);
    }
    .report-section h2 {
      font-size: 1.35rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
      color: var(--text);
    }
    .report-section .description {
      color: var(--text-dim);
      font-size: 0.92rem;
      margin-bottom: 1.25rem;
      max-width: 900px;
      line-height: 1.7;
    }
    .report-section .description a {
      color: var(--accent);
      text-decoration: underline;
    }
    .plot-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 1.5rem;
    }
    .plot-card-header {
      padding: 1rem 1.25rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .plot-card-header h3 { font-size: 1rem; font-weight: 600; }
    .plot-card-body { padding: 0.5rem; }
    .controls { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
    .controls button {
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text-dim);
      padding: 0.35rem 0.85rem;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.15s;
      font-family: inherit;
    }
    .controls button:hover { border-color: var(--accent); color: var(--text); }
    .controls button.active { background: var(--accent); color: var(--bg); border-color: var(--accent); font-weight: 600; }
    .controls select {
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.35rem 0.6rem;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      font-family: inherit;
    }
    .controls label {
      font-size: 0.8rem;
      color: var(--text-dim);
      display: flex;
      align-items: center;
      gap: 0.3rem;
      white-space: nowrap;
    }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
    /* Summary tables */
    .summary-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      margin-bottom: 1.25rem;
      background: var(--surface);
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .summary-table th, .summary-table td {
      padding: 0.6rem 0.9rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    .summary-table th {
      background: var(--surface2);
      font-weight: 600;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: var(--text-dim);
    }
    .summary-table td { color: var(--text); }
    .summary-table tr:last-child td { border-bottom: none; }
    .summary-table tr:hover td { background: #f0f9ff; }
    .summary-table .num { text-align: right; font-variant-numeric: tabular-nums; }
    .summary-table .sig { color: #dc2626; font-weight: 600; }
    .summary-table .ns { color: #16a34a; }
    .confound-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      margin-bottom: 1.25rem;
    }
    /* Dual-handle range slider for continuous colour scales */
    .range-slider-wrap {
      display: none;
      align-items: center;
      gap: 0.5rem;
      margin-top: 0.4rem;
      font-size: 0.8rem;
      color: var(--text-dim);
    }
    .range-slider-wrap.visible { display: flex; flex-wrap: wrap; }
    .range-slider-wrap input[type=range] {
      -webkit-appearance: none;
      appearance: none;
      width: 130px;
      height: 6px;
      background: var(--border);
      border-radius: 3px;
      outline: none;
    }
    .range-slider-wrap input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none; appearance: none;
      width: 14px; height: 14px;
      border-radius: 50%; background: var(--accent);
      cursor: pointer;
    }
    .range-slider-wrap input[type=range]::-moz-range-thumb {
      width: 14px; height: 14px; border:none;
      border-radius: 50%; background: var(--accent); cursor: pointer;
    }
    .range-slider-wrap .range-label { min-width: 3.5em; text-align: right; font-variant-numeric: tabular-nums; }
    /* Filter-removed points info banner (shown when slider excludes points) */
    .filter-info {
      display: none;
      font-size: 0.8rem;
      padding: 0.28rem 0.75rem;
      margin: 0.3rem 0 0 0;
      border-radius: 0.3rem;
      background: #fffbeb;
      border-left: 3px solid var(--amber);
      color: #92400e;
    }
    .filter-info.visible { display: block; }
    .confound-card h4 { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; }
    .confound-card .stat-row {
      display: flex;
      gap: 1.5rem;
      flex-wrap: wrap;
      margin-bottom: 0.75rem;
      font-size: 0.88rem;
      color: var(--text-dim);
    }
    .confound-card .stat-row span { font-weight: 600; color: var(--text); }
    .residual-table { font-size: 0.82rem; }
    .residual-table td.pos { background: rgba(220,38,38,0.08); }
    .residual-table td.neg { background: rgba(22,163,74,0.08); }
    .intro-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem 1.75rem;
      margin-bottom: 1.25rem;
      line-height: 1.75;
    }
    .intro-box h3 {
      font-size: 1.05rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
      color: var(--accent2);
    }
    .intro-box p { margin-bottom: 0.75rem; font-size: 0.92rem; }
    .intro-box ul { margin: 0.5rem 0 0.75rem 1.5rem; font-size: 0.92rem; }
    .intro-box li { margin-bottom: 0.3rem; }
    footer {
      text-align: center;
      padding: 2rem;
      color: var(--text-dim);
      font-size: 0.8rem;
      border-top: 1px solid var(--border);
    }
    </style>
    </head>
    <body>

    <div class="header">
      <h1>NGS-PCA Interactive Report</h1>
      <p>Principal Component Analysis of 1000 Genomes Project — Batch Effects &amp; Population Structure</p>
      <div class="stats-bar">
        <div class="stat-card"><div class="label">Samples</div><div class="value" id="stat-samples">—</div></div>
        <div class="stat-card"><div class="label">Populations</div><div class="value" id="stat-pops">—</div></div>
        <div class="stat-card"><div class="label">Superpopulations</div><div class="value" id="stat-spops">—</div></div>
        <div class="stat-card"><div class="label">PCs computed</div><div class="value" id="stat-pcs">—</div></div>
        <div class="stat-card"><div class="label">MP-significant PCs</div><div class="value" id="stat-mp">—</div></div>
      </div>
    </div>

    <nav class="toc" id="toc-nav">
      <a href="#section-intro" class="active">Introduction</a>
      <a href="#section-scree">Variance Explained</a>
      <a href="#section-pca">PCA Scatter</a>
      <a href="#section-umap">UMAP</a>
      <a href="#section-confounding">Confounding</a>
      <a href="#section-partitioning">Variance Partitioning</a>
      <a href="#section-heatmap">PC–QC Associations</a>
      <a href="#section-sex">Sex × PC</a>
      <a href="#section-batch">Batch vs Ancestry</a>
      <a href="#section-summary">Summary</a>
    </nav>

    <!-- Introduction -->
    <div class="report-section" id="section-intro">
      <h2>Introduction &amp; Background</h2>
      <div class="intro-box">
        <h3>What is NGS-PCA?</h3>
        <p>
          <strong><a href="https://github.com/jlanej/NGS-PCA" target="_blank" rel="noopener">NGS-PCA</a></strong>
          performs principal component analysis (PCA) directly on next-generation sequencing (NGS) summary
          statistics — specifically, read-depth signals across the genome — without requiring genotype calls.
          It applies a randomized singular value decomposition (SVD) to a samples × genomic-bins coverage
          matrix, producing orthogonal principal components that capture the dominant axes of variation in
          sequencing data.
        </p>
        <h3>Why PCA on sequencing data?</h3>
        <p>
          NGS-PCA is designed to detect and characterise <strong>technical sources of variation</strong>
          in sequencing data — primarily batch effects arising from differences in library preparation,
          sequencing runs, and read-depth profiles — without requiring genotype calls. It is intended as
          a sequencing quality-control and diagnostics view of the data, highlighting dominant coverage-level
          structure that can otherwise be hard to detect from summary metrics alone. Key properties:
        </p>
        <ul>
          <li><strong>Efficient</strong> — SVD is run directly on an existing samples × genomic-bins
              coverage matrix, making turnaround practical for routine QC workflows.</li>
          <li><strong>Coverage-signal based</strong> — analyses are derived from read-depth patterns, so
              no genotype-calling step is required for this technical assessment.</li>
          <li><strong>Batch-effect focused</strong> — principal components primarily capture technical
              variation in sequencing depth, enabling quality-control diagnostics for sequencing
              studies.</li>
        </ul>
        <h3>About this report</h3>
        <p>
          This interactive report applies NGS-PCA to the
          <strong>1000 Genomes Project</strong> dataset. We decompose the coverage matrix, visualise
          the resulting PCs, and quantify how much of the variation in each PC is attributable to
          technical factors (sequencing batch) versus biological factors (continental ancestry, sex).
          Because NGS-PCA PCs primarily reflect coverage-level technical variation, batch effects are
          expected to dominate the leading components. However, when batches are enriched for specific
          populations or sexes, technical and biological signals can become <em>confounded</em>. We
          formally test for such enrichment and present effect-size metrics (η², Cramér's V,
          point-biserial <em>r</em>) so that readers can assess the severity of any confounding.
        </p>
      </div>
    </div>

    <!-- Scree -->
    <div class="report-section" id="section-scree">
      <h2>Variance Explained by Principal Components</h2>
      <p class="description">
        These plots summarize the variance structure of the <em>k</em> components retained by the
        randomized SVD (RSVD). Because RSVD is a truncated decomposition that computes only the
        leading <em>k</em> singular values, the percentages shown are normalized to the sum of the
        retained eigenvalues (Σ σ<sub>i</sub><sup>2</sup>, <em>i</em> = 1…<em>k</em>),
        <strong>not</strong> to the full data-matrix variance (which requires all singular values).
        As a result, the cumulative curve reaches 100% at the last retained PC by construction of
        the normalization — it does <em>not</em> imply that the retained components explain all
        variance in the original data matrix.
        The dashed orange line marks the Marchenko–Pastur (MP) cutoff — PCs to the left carry more
        variance than expected under a random-noise null model and are considered statistically
        significant.
      </p>
      <div class="grid-2">
        <div class="plot-card">
          <div class="plot-card-header"><h3>Scree Plot</h3></div>
          <div class="plot-card-body"><div id="scree-bar" style="height:420px"></div></div>
        </div>
        <div class="plot-card">
          <div class="plot-card-header"><h3>Cumulative Variance</h3></div>
          <div class="plot-card-body"><div id="scree-cum" style="height:420px"></div></div>
        </div>
      </div>
    </div>

    <!-- PCA Scatter -->
    <div class="report-section" id="section-pca">
      <h2>PCA Scatter Plots</h2>
      <p class="description">
        Interactive 3D scatter plot of principal components (select up to three PCs from PC1–10).
        Toggle the colour overlay to explore batch effects, population labels, sex differences,
        or any continuous QC metric on a heatmap scale.
        When a categorical variable is selected, the second panel shows boxplots of PC
        distributions by group; when a continuous metric is selected, it shows Pearson and
        Spearman correlation bar charts (computed at runtime).
      </p>
      <div class="plot-card">
        <div class="plot-card-header">
          <h3>PCA 3D Scatter</h3>
          <div class="controls">
            <label>X <select id="pca-x"></select></label>
            <label>Y <select id="pca-y"></select></label>
            <label>Z <select id="pca-z"></select></label>
            <label>Color <select id="pca-color"></select></label>
          </div>
        </div>
        <div class="range-slider-wrap" id="pca-range-wrap">
          <span class="range-label" id="pca-range-lo-label">0</span>
          <input type="range" id="pca-range-lo" min="0" max="100" value="0" step="0.1">
          <span style="color:var(--text);">–</span>
          <input type="range" id="pca-range-hi" min="0" max="100" value="100" step="0.1">
          <span class="range-label" id="pca-range-hi-label">100</span>
          <button id="pca-range-reset" style="font-size:0.75rem;padding:0.2rem 0.5rem;">Reset</button>
        </div>
        <div class="filter-info" id="pca-filter-info"></div>
        <div class="plot-card-body"><div id="pca-scatter" style="height:560px"></div></div>
      </div>
      <div class="plot-card">
        <div class="plot-card-header"><h3 id="pca-panel2-title">Distribution</h3></div>
        <div class="plot-card-body"><div id="pca-panel2" style="height:560px"></div></div>
      </div>
    </div>

    <!-- UMAP -->
    <div class="report-section" id="section-umap">
      <h2>UMAP Projection</h2>
      <p class="description">
        Two-dimensional UMAP embedding computed from a Marchenko–Pastur-selected number of
        principal components. Colour by categorical grouping or a continuous QC metric.
        The second panel shows boxplots (categorical) or correlation bar charts (continuous).
      </p>
      <div class="grid-2">
        <div class="plot-card">
          <div class="plot-card-header">
            <h3 id="umap-title">UMAP (PCs)</h3>
            <div class="controls">
              <label>Color <select id="umap-color"></select></label>
            </div>
          </div>
          <div class="range-slider-wrap" id="umap-range-wrap">
            <span class="range-label" id="umap-range-lo-label">0</span>
            <input type="range" id="umap-range-lo" min="0" max="100" value="0" step="0.1">
            <span style="color:var(--text);">–</span>
            <input type="range" id="umap-range-hi" min="0" max="100" value="100" step="0.1">
            <span class="range-label" id="umap-range-hi-label">100</span>
            <button id="umap-range-reset" style="font-size:0.75rem;padding:0.2rem 0.5rem;">Reset</button>
          </div>
          <div class="filter-info" id="umap-filter-info"></div>
          <div class="plot-card-body"><div id="umap-plot" style="height:560px"></div></div>
        </div>
        <div class="plot-card">
          <div class="plot-card-header"><h3 id="umap-panel2-title">Distribution</h3></div>
          <div class="plot-card-body"><div id="umap-panel2" style="height:560px"></div></div>
        </div>
      </div>
    </div>

    <!-- Confounding Assessment -->
    <div class="report-section" id="section-confounding">
      <h2>Confounding Assessment</h2>
      <p class="description">
        Before interpreting NGS-PCA results, it is important to check whether technical variables
        (sequencing batch) are confounded with biological variables (ancestry, sex). Because
        NGS-PCA PCs primarily capture coverage-level technical variation, batch effects are expected
        to drive the leading components. However, if batches are enriched for particular populations
        or sexes, apparent "batch effects" on PCs may be intertwined with biological signal, and vice
        versa. We use Pearson's χ² test of independence for each pair and report Cramér's V as an
        effect size. Standardised residuals identify which specific cells are over- or
        under-represented.
      </p>
      <div id="confounding-cards"></div>
    </div>

    <!-- Variance Partitioning -->
    <div class="report-section" id="section-partitioning">
      <h2>Variance Partitioning: Disentangling Batch and Ancestry</h2>
      <p class="description">
        To assess whether batch PCs are erroneously capturing ancestry — or whether ancestry
        is simply confounded with batch — we decompose each PC's variance into four components
        using an inclusion/exclusion R² approach.
        <strong>Unique Batch</strong> = variance explained only by batch (not by ancestry);
        <strong>Unique Ancestry</strong> = variance explained only by ancestry (not by batch);
        <strong>Shared (Confounded)</strong> = variance attributable to either factor due to
        their correlation;
        <strong>Residual</strong> = variance explained by neither.
        Large "Shared" bars indicate PCs where batch and ancestry effects are inseparable.
      </p>
      <div class="plot-card">
        <div class="plot-card-header"><h3>R² Variance Partitioning per PC</h3></div>
        <div class="plot-card-body"><div id="partitioning-bar" style="height:460px"></div></div>
      </div>
      <div id="partitioning-table-container"></div>
    </div>

    <!-- Heatmap -->
    <div class="report-section" id="section-heatmap">
      <h2>PC × QC Variable Associations</h2>
      <p class="description">
        Effect sizes between each PC and every available QC variable.
        <strong>Categorical</strong> variables (batch, population, sex, relatedness) use
        <strong>η²</strong> (eta-squared) — the proportion of PC-score variance explained by
        group membership, equivalent to SS<sub>between</sub> / SS<sub>total</sub> from a
        one-way ANOVA.  η² is the natural R²-analogue for a categorical predictor on a
        continuous outcome and is therefore the appropriate metric here: it directly answers
        "how much of this PC's spread does the grouping capture?".  Cramér's V, by contrast,
        measures association between two <em>categorical</em> variables and would require
        discretising the PC scores, losing information.
        <strong>Continuous</strong> variables (coverage metrics) use Pearson <strong>r²</strong>.
        P-values from the corresponding ANOVA F-test (categorical) or Pearson t-test
        (continuous) are shown on hover.  High values indicate that a QC variable explains
        substantial variance in that PC, suggesting the PC captures variation driven by that
        variable.
      </p>
      <div class="plot-card">
        <div class="plot-card-header"><h3>Association Heatmap (η² / r²)</h3></div>
        <div class="plot-card-body"><div id="heatmap-plot" style="height:600px"></div></div>
      </div>
    </div>

    <!-- Sex × PC -->
    <div class="report-section" id="section-sex">
      <h2>Sex Association Across Principal Components</h2>
      <p class="description">
        Biological sex can correlate with principal components through genuine chromosomal
        differences or through sequencing artefacts (e.g., coverage variation on sex chromosomes).
        We quantify the association using three complementary metrics: η² (ANOVA-based), the
        point-biserial correlation coefficient <em>r</em> (signed, so direction matters), and the
        Kruskal–Wallis <em>H</em>-test (non-parametric rank-based test).
      </p>
      <div class="plot-card">
        <div class="plot-card-header"><h3>Point-Biserial <em>r</em>: Sex × PC</h3></div>
        <div class="plot-card-body"><div id="sex-pc-bar" style="height:400px"></div></div>
      </div>
      <div id="sex-pc-table-container"></div>
    </div>

    <!-- Batch vs Ancestry -->
    <div class="report-section" id="section-batch">
      <h2>Batch vs Ancestry Effect Sizes</h2>
      <p class="description">
        Compares η² (eta-squared) effect sizes of sequencing batch and continental ancestry
        on each PC. Because NGS-PCA PCs capture coverage-level technical variation, batch effects
        are expected to dominate the leading components. Large ancestry η² on the same PCs may
        indicate that batches are non-randomly distributed across populations, warranting further
        investigation.
      </p>
      <div class="plot-card">
        <div class="plot-card-header"><h3>η² per Principal Component</h3></div>
        <div class="plot-card-body"><div id="batch-bar" style="height:460px"></div></div>
      </div>
    </div>

    <!-- Summary -->
    <div class="report-section" id="section-summary">
      <h2>Summary Statistics</h2>
      <p class="description">
        Key dataset characteristics and analysis metrics at a glance.
      </p>
      <div class="grid-2">
        <div>
          <h3 style="font-size:1rem;margin-bottom:0.75rem;">Samples per Superpopulation</h3>
          <table class="summary-table" id="tbl-superpop"></table>
        </div>
        <div>
          <h3 style="font-size:1rem;margin-bottom:0.75rem;">Samples per Batch</h3>
          <table class="summary-table" id="tbl-batch"></table>
        </div>
      </div>
      <div style="margin-top:0.5rem;">
        <h3 style="font-size:1rem;margin-bottom:0.75rem;">Samples per Sex</h3>
        <table class="summary-table" id="tbl-sex"></table>
      </div>
    </div>

    <footer>
      Generated by the <a href="https://github.com/jlanej/NGS-PCA" target="_blank" rel="noopener">NGS-PCA</a> analysis pipeline · Plotly.js interactive charts
    </footer>

    <script>
    /* ------------------------------------------------------------------ */
    /*  DATA  (injected by Python)                                        */
    /* ------------------------------------------------------------------ */
    const DATA = __DATA_JSON__;

    /* ------------------------------------------------------------------ */
    /*  PLOTLY THEME                                                       */
    /* ------------------------------------------------------------------ */
    const LAYOUT_BASE = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: '#ffffff',
      font: { color: '#0f172a', family: 'Inter, Segoe UI, system-ui, sans-serif', size: 12 },
      margin: { t: 30, r: 20, b: 50, l: 60 },
      xaxis: { gridcolor: '#e2e8f0', zerolinecolor: '#cbd5e1' },
      yaxis: { gridcolor: '#e2e8f0', zerolinecolor: '#cbd5e1' },
    };
    const CFG = { responsive: true, displayModeBar: true, displaylogo: false,
                  modeBarButtonsToRemove: ['lasso2d','select2d'] };

    const PAL_SUPERPOP = { AFR:'#E41A1C', AMR:'#377EB8', EAS:'#4DAF4A', EUR:'#984EA3', SAS:'#FF7F00' };
    const PAL_BATCH    = { '698':'#1B9E77', '2504':'#D95F02' };
    const PAL_SEX      = { M:'#4393C3', F:'#D6604D' };
    const CAT_PALS = { SUPERPOPULATION: PAL_SUPERPOP, RELEASE_BATCH: PAL_BATCH, INFERRED_SEX: PAL_SEX };
    const PCA_PCS = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'].filter(p => DATA.scatter[p]);
    const NUMERIC_COLS = DATA.numeric_cols || [];

    /* ------------------------------------------------------------------ */
    /*  STATS                                                              */
    /* ------------------------------------------------------------------ */
    document.getElementById('stat-samples').textContent = DATA.n_samples;
    document.getElementById('stat-pops').textContent    = DATA.n_populations;
    document.getElementById('stat-spops').textContent   = DATA.n_superpops;
    document.getElementById('stat-pcs').textContent     = DATA.n_scree;
    document.getElementById('stat-mp').textContent      = DATA.mp_cutoff_pcs;

    /* ------------------------------------------------------------------ */
    /*  SCROLLSPY FOR TOC                                                  */
    /* ------------------------------------------------------------------ */
    (function() {
      const sections = document.querySelectorAll('.report-section');
      const links = document.querySelectorAll('#toc-nav a');
      const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            links.forEach(l => l.classList.remove('active'));
            const active = document.querySelector('#toc-nav a[href="#' + entry.target.id + '"]');
            if (active) active.classList.add('active');
          }
        });
      }, { rootMargin: '-80px 0px -60% 0px', threshold: 0 });
      sections.forEach(s => observer.observe(s));
    })();

    /* ------------------------------------------------------------------ */
    /*  HELPER: populate <select> and colour <select>                      */
    /* ------------------------------------------------------------------ */
    function populateSelect(id, options, selected) {
      const sel = document.getElementById(id);
      options.forEach(o => {
        const opt = document.createElement('option');
        opt.value = o; opt.textContent = o;
        if (o === selected) opt.selected = true;
        sel.appendChild(opt);
      });
    }

    function populateColorSelect(id) {
      const sel = document.getElementById(id);
      const catGroup = document.createElement('optgroup');
      catGroup.label = 'Categorical';
      [{v:'SUPERPOPULATION',t:'Superpopulation'},{v:'RELEASE_BATCH',t:'Batch'},{v:'INFERRED_SEX',t:'Sex'},
       {v:'POPULATION',t:'Population'},{v:'RELATEDNESS',t:'Relatedness'}]
        .filter(function(item) { return DATA.scatter[item.v]; })
        .forEach(function(item) {
          var o = document.createElement('option');
          o.value = 'cat:' + item.v; o.textContent = item.t;
          catGroup.appendChild(o);
        });
      sel.appendChild(catGroup);
      if (NUMERIC_COLS.length > 0) {
        const numGroup = document.createElement('optgroup');
        numGroup.label = 'Continuous (heatmap)';
        NUMERIC_COLS.forEach(function(c) {
          var o = document.createElement('option');
          o.value = 'num:' + c; o.textContent = c;
          numGroup.appendChild(o);
        });
        sel.appendChild(numGroup);
      }
    }

    /* ------------------------------------------------------------------ */
    /*  HELPER: dual-handle range slider for continuous colour scales      */
    /* ------------------------------------------------------------------ */
    function initRangeSlider(prefix, plotFn) {
      var lo = document.getElementById(prefix + '-range-lo');
      var hi = document.getElementById(prefix + '-range-hi');
      var loLabel = document.getElementById(prefix + '-range-lo-label');
      var hiLabel = document.getElementById(prefix + '-range-hi-label');
      var wrap = document.getElementById(prefix + '-range-wrap');
      var resetBtn = document.getElementById(prefix + '-range-reset');
      var state = {dataMin:0, dataMax:1, lo:0, hi:1};

      function update(fromSlider) {
        var loVal = parseFloat(lo.value);
        var hiVal = parseFloat(hi.value);
        if (loVal > hiVal) { if (fromSlider === 'lo') hi.value = lo.value; else lo.value = hi.value; }
        loVal = parseFloat(lo.value); hiVal = parseFloat(hi.value);
        var range = state.dataMax - state.dataMin;
        state.lo = state.dataMin + (loVal / 100) * range;
        state.hi = state.dataMin + (hiVal / 100) * range;
        loLabel.textContent = state.lo.toPrecision(4);
        hiLabel.textContent = state.hi.toPrecision(4);
        plotFn();
      }
      lo.addEventListener('input', function() { update('lo'); });
      hi.addEventListener('input', function() { update('hi'); });
      resetBtn.addEventListener('click', function() { lo.value = 0; hi.value = 100; update('lo'); });

      return {
        show: function(dataMin, dataMax) {
          state.dataMin = dataMin; state.dataMax = dataMax;
          lo.value = 0; hi.value = 100;
          state.lo = dataMin; state.hi = dataMax;
          loLabel.textContent = dataMin.toPrecision(4);
          hiLabel.textContent = dataMax.toPrecision(4);
          wrap.classList.add('visible');
        },
        hide: function() { wrap.classList.remove('visible'); },
        getRange: function() { return [state.lo, state.hi]; }
      };
    }

    /* ------------------------------------------------------------------ */
    /*  HELPER: filter-removed-points info banner                         */
    /* ------------------------------------------------------------------ */
    function updateFilterInfo(prefix, totalN, loVal, hiVal, dataMin, dataMax) {
      var el = document.getElementById(prefix + '-filter-info');
      if (!el) return;
      if (loVal <= dataMin && hiVal >= dataMax) { el.classList.remove('visible'); return; }
      el.innerHTML = 'Colorscale clamped to <strong>' + loVal.toPrecision(4) + '</strong>'
        + ' \u2013 <strong>' + hiVal.toPrecision(4) + '</strong>'
        + ' \u2014 all <strong>' + totalN + '</strong> points shown';
      el.classList.add('visible');
    }
    function hideFilterInfo(prefix) {
      var el = document.getElementById(prefix + '-filter-info');
      if (el) el.classList.remove('visible');
    }

    /* ------------------------------------------------------------------ */
    /*  HELPER: Pearson and Spearman correlation at runtime                */
    /* ------------------------------------------------------------------ */
    function pearsonCorr(x, y) {
      /* Require ≥10 valid pairs to guard against misleading estimates. */
      var n = x.length, sumX=0, sumY=0, sumXY=0, sumX2=0, sumY2=0, cnt=0;
      for (var i=0; i<n; i++) {
        if (x[i]==null || y[i]==null || isNaN(x[i]) || isNaN(y[i])) continue;
        sumX+=x[i]; sumY+=y[i]; sumXY+=x[i]*y[i];
        sumX2+=x[i]*x[i]; sumY2+=y[i]*y[i]; cnt++;
      }
      if (cnt < 10) return 0;
      var num = cnt*sumXY - sumX*sumY;
      var den = Math.sqrt((cnt*sumX2-sumX*sumX)*(cnt*sumY2-sumY*sumY));
      return den === 0 ? 0 : num/den;
    }

    function rankArray(arr) {
      /* Tied values receive the average of the ranks they span (mid-rank). */
      var indexed = [];
      for (var i=0; i<arr.length; i++) {
        if (arr[i]!=null && !isNaN(arr[i])) indexed.push({v:arr[i], i:i});
      }
      indexed.sort(function(a,b){return a.v-b.v;});
      var ranks = new Array(arr.length).fill(null);
      for (var i=0; i<indexed.length; ) {
        var j = i;
        while (j < indexed.length && indexed[j].v === indexed[i].v) j++;
        var avgRank = (i+j-1)/2 + 1;
        for (var k=i; k<j; k++) ranks[indexed[k].i] = avgRank;
        i = j;
      }
      return ranks;
    }

    function spearmanCorr(x, y) { return pearsonCorr(rankArray(x), rankArray(y)); }

    /* ------------------------------------------------------------------ */
    /*  SCREE                                                              */
    /* ------------------------------------------------------------------ */
    (function() {
      const pcs = DATA.variance_prop.map((_, i) => 'PC' + (i + 1));
      const pctProp = DATA.variance_prop.map(v => (v * 100));
      const pctCum  = DATA.variance_cum.map(v => (v * 100));
      const cutoffPc = DATA.mp_cutoff_pcs;
      const cutoffLabel = 'PC' + cutoffPc;

      Plotly.newPlot('scree-bar', [{
        x: pcs, y: pctProp, type: 'bar',
        marker: { color: pctProp, colorscale: [[0,'#38bdf8'],[1,'#818cf8']], line: { width: 0 } },
        hovertemplate: '%{x}: %{y:.2f}%<extra></extra>',
      }, {
        x: [cutoffLabel, cutoffLabel],
        y: [0, Math.max(...pctProp) * 1.03],
        type: 'scatter',
        mode: 'lines',
        line: { color: '#f97316', width: 2, dash: 'dash' },
        name: 'MP cutoff (' + cutoffPc + ' PCs)',
        hovertemplate: 'Marchenko-Pastur cutoff: ' + cutoffPc + ' PCs<extra></extra>',
      }], {
        ...LAYOUT_BASE,
        xaxis: { ...LAYOUT_BASE.xaxis, title: 'Principal Component', tickangle: -45, dtick: 5 },
        yaxis: { ...LAYOUT_BASE.yaxis, title: '% Variance (Retained PCs)' },
        legend: { x: 0.65, y: 0.95, bgcolor: 'rgba(0,0,0,0)' },
      }, CFG);

      Plotly.newPlot('scree-cum', [{
        x: pcs, y: pctCum, type: 'scatter', mode: 'lines+markers',
        line: { color: '#34d399', width: 2.5 },
        marker: { size: 4, color: '#34d399' },
        hovertemplate: '%{x}: %{y:.1f}%<extra></extra>',
      }, {
        x: pcs, y: pcs.map(() => 80), mode: 'lines', line: { dash: 'dash', color: '#94a3b8', width: 1 },
        showlegend: true, name: '80%',
        hoverinfo: 'skip',
      }, {
        x: pcs, y: pcs.map(() => 90), mode: 'lines', line: { dash: 'dot', color: '#94a3b8', width: 1 },
        showlegend: true, name: '90%',
        hoverinfo: 'skip',
      }, {
        x: [cutoffLabel, cutoffLabel],
        y: [0, 105],
        mode: 'lines',
        line: { color: '#f97316', width: 2, dash: 'dash' },
        showlegend: true,
        name: 'MP cutoff (' + cutoffPc + ' PCs)',
        hovertemplate: 'Marchenko-Pastur cutoff: ' + cutoffPc + ' PCs<extra></extra>',
      }], {
        ...LAYOUT_BASE,
        xaxis: { ...LAYOUT_BASE.xaxis, title: 'Number of PCs', tickangle: -45, dtick: 5 },
        yaxis: { ...LAYOUT_BASE.yaxis, title: 'Cumulative % (Retained PCs)', range: [0, 105] },
        legend: { x: 0.7, y: 0.3, bgcolor: 'rgba(0,0,0,0)' },
      }, CFG);
    })();

    /* ------------------------------------------------------------------ */
    /*  PCA 3D SCATTER + PANEL 2 (boxplots / correlation bars)            */
    /* ------------------------------------------------------------------ */
    populateSelect('pca-x', PCA_PCS, 'PC1');
    populateSelect('pca-y', PCA_PCS, 'PC2');
    populateSelect('pca-z', PCA_PCS, 'PC3');
    populateColorSelect('pca-color');
    var pcaSlider = initRangeSlider('pca', plotPCA);

    function plotPCA() {
      var xKey = document.getElementById('pca-x').value;
      var yKey = document.getElementById('pca-y').value;
      var zKey = document.getElementById('pca-z').value;
      var colorVal = document.getElementById('pca-color').value;
      var parts = colorVal.split(':');
      var cType = parts[0], col = parts[1];
      var S = DATA.scatter;

      if (cType === 'cat') {
        pcaSlider.hide();
        hideFilterInfo('pca');
        var pal = CAT_PALS[col] || {};
        var groups = {};
        for (var i=0; i<S.SAMPLE.length; i++) {
          var g = S[col] ? S[col][i] : null;
          if (g == null) continue;
          if (!groups[g]) groups[g] = {x:[],y:[],z:[],text:[]};
          groups[g].x.push(S[xKey][i]); groups[g].y.push(S[yKey][i]); groups[g].z.push(S[zKey][i]);
          groups[g].text.push(S.SAMPLE[i]);
        }
        var traces = Object.entries(groups).map(function(e) {
          var name = e[0], d = e[1];
          return {
            x:d.x, y:d.y, z:d.z, text:d.text, type:'scatter3d', mode:'markers', name:name,
            marker: {color: pal[name]||'#888', size:3, opacity:0.8},
            hovertemplate: '%{text}<br>'+xKey+':%{x:.3f}<br>'+yKey+':%{y:.3f}<br>'+zKey+':%{z:.3f}<extra>'+name+'</extra>',
          };
        });
        Plotly.react('pca-scatter', traces, {
          ...LAYOUT_BASE, margin:{t:15,r:15,b:15,l:15},
          scene: {xaxis:{title:xKey}, yaxis:{title:yKey}, zaxis:{title:zKey}},
        }, CFG);
      } else {
        var vals = S[col] || [];
        var finite = vals.filter(function(v){return v!=null && isFinite(v);});
        var dMin = finite.length ? Math.min.apply(null, finite) : 0;
        var dMax = finite.length ? Math.max.apply(null, finite) : 1;
        if (!pcaSlider._inited || pcaSlider._col !== col) {
          pcaSlider.show(dMin, dMax); pcaSlider._inited = true; pcaSlider._col = col;
        }
        var rng = pcaSlider.getRange();
        updateFilterInfo('pca', finite.length, rng[0], rng[1], dMin, dMax);
        Plotly.react('pca-scatter', [{
          x:S[xKey], y:S[yKey], z:S[zKey], text:S.SAMPLE,
          type:'scatter3d', mode:'markers',
          marker: {color:vals, colorscale:'Viridis', size:3, opacity:0.8,
                   cmin:rng[0], cmax:rng[1],
                   colorbar:{title:col,titleside:'right'}},
          hovertemplate: '%{text}<br>'+xKey+':%{x:.3f}<br>'+yKey+':%{y:.3f}<br>'+zKey+':%{z:.3f}<br>'+col+':%{marker.color:.3f}<extra></extra>',
        }], {
          ...LAYOUT_BASE, margin:{t:15,r:15,b:15,l:15},
          scene: {xaxis:{title:xKey}, yaxis:{title:yKey}, zaxis:{title:zKey}},
        }, CFG);
      }
      updatePCAPanel2();
    }

    function updatePCAPanel2() {
      var colorVal = document.getElementById('pca-color').value;
      var parts = colorVal.split(':');
      var cType = parts[0], col = parts[1];
      var S = DATA.scatter;
      var xKey = document.getElementById('pca-x').value;
      var yKey = document.getElementById('pca-y').value;
      var zKey = document.getElementById('pca-z').value;
      var selPCs = [xKey, yKey, zKey];

      if (cType === 'cat') {
        var pal = CAT_PALS[col] || {};
        var cats = [];
        var seen = {};
        (S[col]||[]).forEach(function(v){ if(v!=null && !seen[v]){seen[v]=1;cats.push(v);} });
        cats.sort();
        var traces = [];
        // Violin+box traces (top subplot)
        selPCs.forEach(function(pc, pi) {
          cats.forEach(function(cat) {
            var vals = [];
            for (var i=0; i<S.SAMPLE.length; i++) {
              if (S[col][i] === cat) vals.push(S[pc][i]);
            }
            traces.push({
              y:vals, type:'violin', name:cat,
              x:vals.map(function(){return pc;}),
              marker:{color:pal[cat]||'#888'},
              box:{visible:true}, meanline:{visible:true}, scalemode:'count',
              legendgroup:cat, showlegend:pi===0,
              xaxis:'x', yaxis:'y',
            });
          });
        });
        // Proportion bar traces (bottom subplot)
        var totalN = S.SAMPLE.length;
        cats.forEach(function(cat) {
          var n = 0;
          for (var i=0; i<S.SAMPLE.length; i++) if (S[col][i]===cat) n++;
          traces.push({
            x:[100*n/totalN], y:[col], type:'bar', orientation:'h',
            name:cat, marker:{color:pal[cat]||'#888'},
            legendgroup:cat, showlegend:false,
            xaxis:'x2', yaxis:'y2',
            hovertemplate:cat+': '+n+' samples (<b>%{x:.1f}%</b>)<extra></extra>',
          });
        });
        document.getElementById('pca-panel2-title').textContent = col + ' \u2014 Distributions & Proportions';
        Plotly.react('pca-panel2', traces, {
          ...LAYOUT_BASE, violinmode:'group', barmode:'stack',
          xaxis:{domain:[0,1], ...LAYOUT_BASE.xaxis, title:'Principal Component'},
          yaxis:{domain:[0.38,1], ...LAYOUT_BASE.yaxis, title:'PC Score'},
          xaxis2:{domain:[0,1], ...LAYOUT_BASE.xaxis, title:'% of samples', range:[0,100]},
          yaxis2:{domain:[0,0.3], ...LAYOUT_BASE.yaxis},
          legend:{bgcolor:'rgba(0,0,0,0)', font:{size:10}},
          margin:{t:15,r:20,b:50,l:80},
        }, CFG);
      } else {
        var metric = S[col] || [];
        var rng = pcaSlider.getRange();
        var allVals = [];
        for (var i=0; i<metric.length; i++) {
          if (metric[i]!=null && isFinite(metric[i])) allVals.push(metric[i]);
        }
        var pearsonVals = PCA_PCS.map(function(pc){return pearsonCorr(S[pc], metric);});
        var spearmanVals = PCA_PCS.map(function(pc){return spearmanCorr(S[pc], metric);});
        var distTraces = [];
        if (allVals.length > 0) {
          distTraces.push({
            y:allVals, type:'violin', name:col, x0:col,
            marker:{color:'#6366f1'}, box:{visible:true}, meanline:{visible:true},
            xaxis:'x', yaxis:'y', showlegend:false,
          });
        }
        distTraces.push({
          x:PCA_PCS, y:pearsonVals, type:'bar', name:'Pearson r',
          marker:{color:'#6366f1'},
          hovertemplate:'%{x}: r = %{y:.4f}<extra>Pearson</extra>',
          xaxis:'x2', yaxis:'y2',
        });
        distTraces.push({
          x:PCA_PCS, y:spearmanVals, type:'bar', name:'Spearman \\u03c1',
          marker:{color:'#f59e0b'},
          hovertemplate:'%{x}: \\u03c1 = %{y:.4f}<extra>Spearman</extra>',
          xaxis:'x2', yaxis:'y2',
        });
        document.getElementById('pca-panel2-title').textContent = col + ' \u2014 Distribution & PC Correlation';
        Plotly.react('pca-panel2', distTraces, {
          ...LAYOUT_BASE, barmode:'group',
          xaxis:{domain:[0,1], ...LAYOUT_BASE.xaxis, title:col},
          yaxis:{domain:[0.42,1], ...LAYOUT_BASE.yaxis, title:'Value'},
          xaxis2:{domain:[0,1], ...LAYOUT_BASE.xaxis, title:'Principal Component'},
          yaxis2:{domain:[0,0.35], ...LAYOUT_BASE.yaxis, title:'Correlation'},
          shapes:[
            {type:'line', y0:rng[0], y1:rng[0], x0:0, x1:1, xref:'paper', yref:'y',
             line:{color:'#f97316', width:2, dash:'dash'}},
            {type:'line', y0:rng[1], y1:rng[1], x0:0, x1:1, xref:'paper', yref:'y',
             line:{color:'#f97316', width:2, dash:'dash'}},
            {type:'line', x0:-0.5, x1:PCA_PCS.length-0.5, y0:0, y1:0,
             xref:'x2', yref:'y2', line:{color:'#94a3b8', width:1, dash:'dot'}},
          ],
          legend:{bgcolor:'rgba(0,0,0,0)', font:{size:10}},
          margin:{t:15,r:20,b:50,l:60},
        }, CFG);
      }
    }

    document.getElementById('pca-color').addEventListener('change', function() { pcaSlider._inited = false; plotPCA(); });
    ['pca-x','pca-y','pca-z'].forEach(function(id){
      document.getElementById(id).addEventListener('change', plotPCA);
    });
    plotPCA();

    /* ------------------------------------------------------------------ */
    /*  UMAP (2D) + PANEL 2                                               */
    /* ------------------------------------------------------------------ */
    populateColorSelect('umap-color');
    var umapSlider = initRangeSlider('umap', plotUmap);

    function plotUmap() {
      var colorVal = document.getElementById('umap-color').value;
      var parts = colorVal.split(':');
      var cType = parts[0], col = parts[1];
      var S = DATA.scatter, ids = S['SAMPLE'];

      if (cType === 'cat') {
        umapSlider.hide();
        hideFilterInfo('umap');
        var pal = CAT_PALS[col] || {};
        var groups = {};
        for (var i=0; i<DATA.umap1.length; i++) {
          var g = S[col] ? S[col][i] : null;
          if (g == null) continue;
          if (!groups[g]) groups[g] = {x:[],y:[],text:[]};
          groups[g].x.push(DATA.umap1[i]); groups[g].y.push(DATA.umap2[i]);
          groups[g].text.push(ids[i]);
        }
        var traces = Object.entries(groups).map(function(e) {
          var name = e[0], d = e[1];
          return {
            x:d.x, y:d.y, text:d.text, type:'scattergl', mode:'markers', name:name,
            marker:{color:pal[name]||'#888', size:5, opacity:0.8},
            hovertemplate:'%{text}<br>UMAP-1:%{x:.2f}<br>UMAP-2:%{y:.2f}<extra>'+name+'</extra>',
          };
        });
        Plotly.react('umap-plot', traces, {
          ...LAYOUT_BASE,
          xaxis:{...LAYOUT_BASE.xaxis, title:'UMAP-1'},
          yaxis:{...LAYOUT_BASE.yaxis, title:'UMAP-2'},
          legend:{bgcolor:'rgba(0,0,0,0)', font:{size:11}},
          margin:{t:15,r:15,b:50,l:60},
        }, CFG);
      } else {
        var vals = S[col] || [];
        var finite = vals.filter(function(v){return v!=null && isFinite(v);});
        var dMin = finite.length ? Math.min.apply(null, finite) : 0;
        var dMax = finite.length ? Math.max.apply(null, finite) : 1;
        if (!umapSlider._inited || umapSlider._col !== col) {
          umapSlider.show(dMin, dMax); umapSlider._inited = true; umapSlider._col = col;
        }
        var rng = umapSlider.getRange();
        updateFilterInfo('umap', finite.length, rng[0], rng[1], dMin, dMax);
        Plotly.react('umap-plot', [{
          x:DATA.umap1, y:DATA.umap2, text:ids,
          type:'scattergl', mode:'markers',
          marker:{color:vals, colorscale:'Viridis', size:5, opacity:0.8,
                  cmin:rng[0], cmax:rng[1],
                  colorbar:{title:col,titleside:'right'}},
          hovertemplate:'%{text}<br>UMAP-1:%{x:.2f}<br>UMAP-2:%{y:.2f}<br>'+col+':%{marker.color:.3f}<extra></extra>',
        }], {
          ...LAYOUT_BASE,
          xaxis:{...LAYOUT_BASE.xaxis, title:'UMAP-1'},
          yaxis:{...LAYOUT_BASE.yaxis, title:'UMAP-2'},
          margin:{t:15,r:15,b:50,l:60},
        }, CFG);
      }
      updateUmapPanel2();
    }

    function updateUmapPanel2() {
      var colorVal = document.getElementById('umap-color').value;
      var parts = colorVal.split(':');
      var cType = parts[0], col = parts[1];
      var S = DATA.scatter;
      var dims = ['UMAP-1','UMAP-2'];
      var dimData = [DATA.umap1, DATA.umap2];

      if (cType === 'cat') {
        var pal = CAT_PALS[col] || {};
        var cats = [];
        var seen = {};
        (S[col]||[]).forEach(function(v){ if(v!=null && !seen[v]){seen[v]=1;cats.push(v);} });
        cats.sort();
        var traces = [];
        // Violin+box traces (top subplot)
        dims.forEach(function(dim, di) {
          cats.forEach(function(cat) {
            var vals = [];
            for (var i=0; i<S.SAMPLE.length; i++) {
              if (S[col][i] === cat) vals.push(dimData[di][i]);
            }
            traces.push({
              y:vals, type:'violin', name:cat,
              x:vals.map(function(){return dim;}),
              marker:{color:pal[cat]||'#888'},
              box:{visible:true}, meanline:{visible:true}, scalemode:'count',
              legendgroup:cat, showlegend:di===0,
              xaxis:'x', yaxis:'y',
            });
          });
        });
        // Proportion bar (bottom subplot)
        var totalN = S.SAMPLE.length;
        cats.forEach(function(cat) {
          var n = 0;
          for (var i=0; i<S.SAMPLE.length; i++) if (S[col][i]===cat) n++;
          traces.push({
            x:[100*n/totalN], y:[col], type:'bar', orientation:'h',
            name:cat, marker:{color:pal[cat]||'#888'},
            legendgroup:cat, showlegend:false,
            xaxis:'x2', yaxis:'y2',
            hovertemplate:cat+': '+n+' samples (<b>%{x:.1f}%</b>)<extra></extra>',
          });
        });
        document.getElementById('umap-panel2-title').textContent = col + ' \u2014 Distributions & Proportions';
        Plotly.react('umap-panel2', traces, {
          ...LAYOUT_BASE, violinmode:'group', barmode:'stack',
          xaxis:{domain:[0,1], ...LAYOUT_BASE.xaxis, title:'UMAP Dimension'},
          yaxis:{domain:[0.38,1], ...LAYOUT_BASE.yaxis, title:'Coordinate'},
          xaxis2:{domain:[0,1], ...LAYOUT_BASE.xaxis, title:'% of samples', range:[0,100]},
          yaxis2:{domain:[0,0.3], ...LAYOUT_BASE.yaxis},
          legend:{bgcolor:'rgba(0,0,0,0)', font:{size:10}},
          margin:{t:15,r:20,b:50,l:80},
        }, CFG);
      } else {
        var metric = S[col] || [];
        var rng = umapSlider.getRange();
        var allValsU = [];
        for (var i=0; i<metric.length; i++) {
          if (metric[i]!=null && isFinite(metric[i])) allValsU.push(metric[i]);
        }
        var pVals = dims.map(function(_,di){return pearsonCorr(dimData[di], metric);});
        var sVals = dims.map(function(_,di){return spearmanCorr(dimData[di], metric);});
        var distTracesU = [];
        if (allValsU.length > 0) {
          distTracesU.push({
            y:allValsU, type:'violin', name:col, x0:col,
            marker:{color:'#6366f1'}, box:{visible:true}, meanline:{visible:true},
            xaxis:'x', yaxis:'y', showlegend:false,
          });
        }
        distTracesU.push({
          x:dims, y:pVals, type:'bar', name:'Pearson r',
          marker:{color:'#6366f1'},
          hovertemplate:'%{x}: r = %{y:.4f}<extra>Pearson</extra>',
          xaxis:'x2', yaxis:'y2',
        });
        distTracesU.push({
          x:dims, y:sVals, type:'bar', name:'Spearman \\u03c1',
          marker:{color:'#f59e0b'},
          hovertemplate:'%{x}: \\u03c1 = %{y:.4f}<extra>Spearman</extra>',
          xaxis:'x2', yaxis:'y2',
        });
        document.getElementById('umap-panel2-title').textContent = col + ' \u2014 Distribution & UMAP Correlation';
        Plotly.react('umap-panel2', distTracesU, {
          ...LAYOUT_BASE, barmode:'group',
          xaxis:{domain:[0,1], ...LAYOUT_BASE.xaxis, title:col},
          yaxis:{domain:[0.42,1], ...LAYOUT_BASE.yaxis, title:'Value'},
          xaxis2:{domain:[0,1], ...LAYOUT_BASE.xaxis, title:'Dimension'},
          yaxis2:{domain:[0,0.35], ...LAYOUT_BASE.yaxis, title:'Correlation'},
          shapes:[
            {type:'line', y0:rng[0], y1:rng[0], x0:0, x1:1, xref:'paper', yref:'y',
             line:{color:'#f97316', width:2, dash:'dash'}},
            {type:'line', y0:rng[1], y1:rng[1], x0:0, x1:1, xref:'paper', yref:'y',
             line:{color:'#f97316', width:2, dash:'dash'}},
            {type:'line', x0:-0.5, x1:1.5, y0:0, y1:0,
             xref:'x2', yref:'y2', line:{color:'#94a3b8', width:1, dash:'dot'}},
          ],
          legend:{bgcolor:'rgba(0,0,0,0)', font:{size:10}},
          margin:{t:15,r:20,b:50,l:60},
        }, CFG);
      }
    }

    document.getElementById('umap-title').textContent = 'UMAP (' + DATA.n_umap_pcs + ' PCs, MP)';
    document.getElementById('umap-color').addEventListener('change', function() { umapSlider._inited = false; plotUmap(); });
    plotUmap();

    /* ------------------------------------------------------------------ */
    /*  CONFOUNDING ASSESSMENT                                             */
    /* ------------------------------------------------------------------ */
    (function() {
      const container = document.getElementById('confounding-cards');
      DATA.confounding.forEach(c => {
        const pStr = c.p_value < 0.001 ? c.p_value.toExponential(2)
                   : c.p_value < 0.01  ? c.p_value.toFixed(4)
                   : c.p_value.toFixed(3);
        const sigClass = c.p_value < 0.05 ? 'sig' : 'ns';
        const sigLabel = c.p_value < 0.05 ? 'Significant' : 'Not significant';

        let html = '<div class="confound-card">';
        html += '<h4>' + c.label + '</h4>';
        html += '<div class="stat-row">';
        html += '<div>\\u03c7\\u00b2 = <span>' + c.chi2.toFixed(2) + '</span></div>';
        html += '<div>df = <span>' + c.dof + '</span></div>';
        html += '<div><em>p</em> = <span class="' + sigClass + '">' + pStr + '</span> (' + sigLabel + ')</div>';
        html += "<div>Cram\\u00e9r\\u2019s V = <span>" + c.cramers_v.toFixed(3) + '</span></div>';
        html += '</div>';

        /* Standardised residuals table */
        html += '<details><summary style="cursor:pointer;font-size:0.85rem;color:var(--text-dim);margin-bottom:0.5rem;">Show standardised residuals</summary>';
        html += '<table class="summary-table residual-table"><thead><tr><th>' + c.var1 + ' \\\\ ' + c.var2 + '</th>';
        c.cols.forEach(col => { html += '<th class="num">' + col + '</th>'; });
        html += '</tr></thead><tbody>';
        c.rows.forEach((row, ri) => {
          html += '<tr><td>' + row + '</td>';
          c.std_residuals[ri].forEach(v => {
            const cls = v > 2 ? 'pos' : (v < -2 ? 'neg' : '');
            html += '<td class="num ' + cls + '">' + v.toFixed(2) + '</td>';
          });
          html += '</tr>';
        });
        html += '</tbody></table></details>';
        html += '</div>';
        container.innerHTML += html;
      });
    })();

    /* ------------------------------------------------------------------ */
    /*  VARIANCE PARTITIONING                                              */
    /* ------------------------------------------------------------------ */
    (function() {
      var vp = DATA.variance_partitioning;
      if (!vp || vp.length === 0) return;
      var pcs = vp.map(function(r){return r.PC;});
      Plotly.newPlot('partitioning-bar', [{
        x:pcs, y:vp.map(function(r){return r.unique_batch;}), type:'bar', name:'Unique Batch',
        marker:{color:'#D95F02'}, hovertemplate:'%{x}<br>Unique Batch: %{y:.4f}<extra></extra>',
      },{
        x:pcs, y:vp.map(function(r){return r.shared;}), type:'bar', name:'Shared (Confounded)',
        marker:{color:'#94a3b8'}, hovertemplate:'%{x}<br>Shared: %{y:.4f}<extra></extra>',
      },{
        x:pcs, y:vp.map(function(r){return r.unique_ancestry;}), type:'bar', name:'Unique Ancestry',
        marker:{color:'#1B9E77'}, hovertemplate:'%{x}<br>Unique Ancestry: %{y:.4f}<extra></extra>',
      },{
        x:pcs, y:vp.map(function(r){return r.residual;}), type:'bar', name:'Residual',
        marker:{color:'#e2e8f0'}, hovertemplate:'%{x}<br>Residual: %{y:.4f}<extra></extra>',
      }], {
        ...LAYOUT_BASE, barmode:'stack',
        xaxis:{...LAYOUT_BASE.xaxis, title:'Principal Component', tickangle:-45},
        yaxis:{...LAYOUT_BASE.yaxis, title:'Proportion of Variance (R\\u00b2)', range:[0,1.05]},
        legend:{bgcolor:'rgba(0,0,0,0)', x:0.7, y:0.95},
      }, CFG);

      /* detailed table */
      var tc = document.getElementById('partitioning-table-container');
      var html = '<table class="summary-table"><thead><tr>';
      html += '<th>PC</th><th class="num">R\\u00b2 Full</th><th class="num">R\\u00b2 Batch</th>';
      html += '<th class="num">R\\u00b2 Ancestry</th><th class="num">Unique Batch</th>';
      html += '<th class="num">Unique Ancestry</th><th class="num">Shared</th>';
      html += '<th class="num">Residual</th></tr></thead><tbody>';
      vp.forEach(function(r) {
        html += '<tr><td>' + r.PC + '</td>';
        html += '<td class="num">' + r.r2_full.toFixed(4) + '</td>';
        html += '<td class="num">' + r.r2_batch.toFixed(4) + '</td>';
        html += '<td class="num">' + r.r2_ancestry.toFixed(4) + '</td>';
        html += '<td class="num">' + r.unique_batch.toFixed(4) + '</td>';
        html += '<td class="num">' + r.unique_ancestry.toFixed(4) + '</td>';
        html += '<td class="num">' + r.shared.toFixed(4) + '</td>';
        html += '<td class="num">' + r.residual.toFixed(4) + '</td></tr>';
      });
      html += '</tbody></table>';
      tc.innerHTML = html;
    })();

    /* ------------------------------------------------------------------ */
    /*  HEATMAP                                                            */
    /* ------------------------------------------------------------------ */
    (function() {
      const variables = [...new Set(DATA.assoc_rows.map(r => r.Variable))];
      const pcs = DATA.assoc_pcs;
      const z = variables.map(v =>
        pcs.map(pc => {
          const r = DATA.assoc_rows.find(a => a.Variable === v && a.PC === pc);
          return r ? r.Value : 0;
        })
      );
      const hovertext = variables.map((v, vi) =>
        pcs.map((pc, pi) => {
          const r = DATA.assoc_rows.find(a => a.Variable === v && a.PC === pc);
          const val = r ? r.Value : 0;
          const metric = r && r.Metric ? r.Metric : '';
          const pVal = r && r.p_value != null
            ? (r.p_value < 0.001 ? r.p_value.toExponential(2)
               : r.p_value < 0.01 ? r.p_value.toFixed(4)
               : r.p_value.toFixed(3))
            : 'N/A';
          const sigStr = r && r.p_value != null && r.p_value < 0.05 ? ' *' : '';
          return v + ' \\u00d7 ' + pc + '<br>' + metric + ' = ' + val.toFixed(4) + '<br><i>p</i> = ' + pVal + sigStr;
        })
      );
      Plotly.newPlot('heatmap-plot', [{
        z: z, x: pcs, y: variables, type: 'heatmap',
        colorscale: [
          [0,    '#ffffff'],
          [0.15, '#312e81'],
          [0.35, '#7c3aed'],
          [0.55, '#c026d3'],
          [0.75, '#e11d48'],
          [1,    '#fbbf24']
        ],
        zmin: 0, zmax: 1,
        hovertext: hovertext, hoverinfo: 'text',
        colorbar: { title: 'Effect size', titleside: 'right', tickformat: '.2f',
                    bgcolor: 'rgba(0,0,0,0)', tickcolor: '#0f172a', tickfont: { color: '#0f172a' } },
      }], {
        ...LAYOUT_BASE,
        xaxis: { ...LAYOUT_BASE.xaxis, title: 'Principal Component', tickangle: -45, side: 'bottom' },
        yaxis: { ...LAYOUT_BASE.yaxis, title: '', automargin: true },
        margin: { t: 15, r: 90, b: 80, l: 160 },
      }, CFG);
    })();

    /* ------------------------------------------------------------------ */
    /*  SEX × PC ASSOCIATIONS                                              */
    /* ------------------------------------------------------------------ */
    (function() {
      const pcs = DATA.sex_pc.map(r => r.PC);
      const rpb = DATA.sex_pc.map(r => r.r_pointbiserial);
      const colors = rpb.map(v => v >= 0 ? '#4393C3' : '#D6604D');

      Plotly.newPlot('sex-pc-bar', [{
        x: pcs, y: rpb, type: 'bar',
        marker: { color: colors, line: { width: 0 } },
        hovertemplate: '%{x}<br>Point-biserial r: %{y:.4f}<extra></extra>',
      }], {
        ...LAYOUT_BASE,
        xaxis: { ...LAYOUT_BASE.xaxis, title: 'Principal Component', tickangle: -45 },
        yaxis: { ...LAYOUT_BASE.yaxis, title: 'Point-biserial r (M vs F)' },
        shapes: [{ type: 'line', x0: -0.5, x1: pcs.length - 0.5, y0: 0, y1: 0,
                   line: { color: '#94a3b8', width: 1, dash: 'dot' } }],
      }, CFG);

      /* sex association table */
      const tc = document.getElementById('sex-pc-table-container');
      let html = '<table class="summary-table"><thead><tr>';
      html += '<th>PC</th><th class="num">\\u03b7\\u00b2</th><th class="num">Point-biserial <em>r</em></th>';
      html += '<th class="num"><em>p</em> (PB)</th><th class="num">Kruskal\\u2013Wallis <em>H</em></th>';
      html += '<th class="num"><em>p</em> (KW)</th></tr></thead><tbody>';
      DATA.sex_pc.forEach(r => {
        const pbSig = r.p_pointbiserial != null && r.p_pointbiserial < 0.05 ? 'sig' : 'ns';
        const kwSig = r.p_kruskal != null && r.p_kruskal < 0.05 ? 'sig' : 'ns';
        const pPB = r.p_pointbiserial != null
          ? (r.p_pointbiserial < 0.001 ? r.p_pointbiserial.toExponential(2) : r.p_pointbiserial.toFixed(4))
          : 'N/A';
        const pKW = r.p_kruskal != null
          ? (r.p_kruskal < 0.001 ? r.p_kruskal.toExponential(2) : r.p_kruskal.toFixed(4))
          : 'N/A';
        html += '<tr>';
        html += '<td>' + r.PC + '</td>';
        html += '<td class="num">' + r.eta2.toFixed(4) + '</td>';
        html += '<td class="num">' + r.r_pointbiserial.toFixed(4) + '</td>';
        html += '<td class="num ' + pbSig + '">' + pPB + '</td>';
        html += '<td class="num">' + (r.kruskal_H != null ? r.kruskal_H.toFixed(2) : 'N/A') + '</td>';
        html += '<td class="num ' + kwSig + '">' + pKW + '</td>';
        html += '</tr>';
      });
      html += '</tbody></table>';
      tc.innerHTML = html;
    })();

    /* ------------------------------------------------------------------ */
    /*  BATCH VS ANCESTRY                                                  */
    /* ------------------------------------------------------------------ */
    (function() {
      const pcs = DATA.batch_records.map(r => r.PC);
      const batch = DATA.batch_records.map(r => r.Batch_eta2);
      const ancestry = DATA.batch_records.map(r => r.Ancestry_eta2);

      Plotly.newPlot('batch-bar', [{
        x: pcs, y: batch, type: 'bar', name: 'Batch (\\u03b7\\u00b2)',
        marker: { color: '#D95F02', line: { width: 0 } },
        hovertemplate: '%{x}<br>Batch \\u03b7\\u00b2: %{y:.4f}<extra></extra>',
      }, {
        x: pcs, y: ancestry, type: 'bar', name: 'Ancestry (\\u03b7\\u00b2)',
        marker: { color: '#1B9E77', line: { width: 0 } },
        hovertemplate: '%{x}<br>Ancestry \\u03b7\\u00b2: %{y:.4f}<extra></extra>',
      }], {
        ...LAYOUT_BASE,
        barmode: 'group',
        xaxis: { ...LAYOUT_BASE.xaxis, title: 'Principal Component', tickangle: -45 },
        yaxis: { ...LAYOUT_BASE.yaxis, title: '\\u03b7\\u00b2 (Effect Size)' },
        legend: { bgcolor: 'rgba(0,0,0,0)', x: 0.75, y: 0.95 },
      }, CFG);
    })();

    /* ------------------------------------------------------------------ */
    /*  SUMMARY TABLES                                                     */
    /* ------------------------------------------------------------------ */
    (function() {
      function buildCountTable(id, data, labelCol) {
        const tbl = document.getElementById(id);
        let html = '<thead><tr><th>' + labelCol + '</th><th class="num">Count</th><th class="num">%</th></tr></thead><tbody>';
        const total = Object.values(data).reduce((a, b) => a + b, 0);
        Object.entries(data).forEach(([k, v]) => {
          html += '<tr><td>' + k + '</td><td class="num">' + v + '</td><td class="num">' + (100 * v / total).toFixed(1) + '%</td></tr>';
        });
        html += '<tr style="font-weight:600;background:var(--surface2)"><td>Total</td><td class="num">' + total + '</td><td class="num">100%</td></tr>';
        html += '</tbody>';
        tbl.innerHTML = html;
      }
      buildCountTable('tbl-superpop', DATA.sample_summary.superpop_counts, 'Superpopulation');
      buildCountTable('tbl-batch', DATA.sample_summary.batch_counts, 'Batch');
      buildCountTable('tbl-sex', DATA.sample_summary.sex_counts, 'Sex');
    })();

    </script>
    </body>
    </html>
    """)

    # Inject the data
    html = html.replace("__DATA_JSON__", data_json, 1)
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(
    data_dir: str,
    output_dir: str,
    report_dir: str,
    n_pcs_scree: int = 0,
    n_pcs_assoc: int = 20,
    n_pcs_umap_max: int = 0,
) -> str:
    """Generate the interactive HTML report.

    Returns the path to the written HTML file.
    """
    print("[06] Loading data …")
    n_samples = _count_available_samples(data_dir)
    sv_df = _load_singular_values(data_dir)
    merged = _load_merged(output_dir)
    if len(merged) != n_samples:
        print(
            "[06] Detected subsetted merged input "
            f"({len(merged)} rows); rebuilding full merged table ({n_samples} rows) for report …"
        )
        merged = _load_full_merged_from_data_dir(data_dir)

    n_populations = merged["POPULATION"].nunique()
    n_superpops = merged["SUPERPOPULATION"].nunique()

    print("[06] Computing variance explained …")
    var_prop, var_cum, n_scree = _compute_variance(sv_df, n_pcs_scree)
    eigenvalues = sv_df["SINGULAR_VALUES"].values ** 2
    max_pcs = n_pcs_umap_max if n_pcs_umap_max > 0 else None
    mp_cutoff_pcs, _ = marchenko_pastur_pc_count_from_data_dir(
        data_dir, eigenvalues, max_pcs=max_pcs
    )
    n_umap_pcs = mp_cutoff_pcs

    print("[06] Preparing scatter data …")
    scatter_data, numeric_cols = _prepare_scatter_data(merged)

    print("[06] Computing UMAP embedding …")
    umap1, umap2 = _compute_umap(merged, n_umap_pcs)

    print("[06] Computing PC–QC associations …")
    assoc_rows, assoc_pcs = _compute_associations(merged, n_pcs_assoc)

    print("[06] Computing batch vs ancestry …")
    batch_records, batch_pcs = _compute_batch_ancestry(merged, n_pcs_assoc)

    print("[06] Computing confounding assessment …")
    confounding_results = _compute_confounding(merged)

    print("[06] Computing sex × PC associations …")
    sex_pc_records = _compute_sex_pc_associations(merged, n_pcs_assoc)

    print("[06] Computing variance partitioning …")
    variance_partitioning = _compute_variance_partitioning(merged, n_pcs_assoc)

    print("[06] Computing sample summary …")
    sample_summary = _compute_sample_summary(merged)

    print("[06] Generating HTML …")
    html = _build_html(
        var_prop, var_cum, n_scree,
        mp_cutoff_pcs,
        scatter_data, numeric_cols, umap1, umap2, n_umap_pcs,
        assoc_rows, assoc_pcs,
        batch_records, batch_pcs,
        confounding_results, sex_pc_records, sample_summary,
        variance_partitioning,
        n_samples, n_populations, n_superpops,
    )

    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[06] Interactive report → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML dashboard from pipeline outputs",
    )
    parser.add_argument("--data-dir",
                        default=os.environ.get("NGSPCA_DATA_DIR", "1000G"))
    parser.add_argument("--output-dir",
                        default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    parser.add_argument("--report-dir", default="docs",
                        help="Directory for the HTML report (default: docs)")
    parser.add_argument("--n-pcs-scree", type=int, default=0,
                        help="Number of PCs for scree/cumulative plots (0 = all available PCs)")
    parser.add_argument("--n-pcs-assoc", type=int, default=20)
    parser.add_argument("--n-pcs-umap-max", type=int, default=0,
                        help="Optional upper bound for MP-selected UMAP PCs (0 = no cap)")
    args = parser.parse_args()
    generate_report(args.data_dir, args.output_dir, args.report_dir,
                    args.n_pcs_scree, args.n_pcs_assoc, args.n_pcs_umap_max)


if __name__ == "__main__":
    main()
