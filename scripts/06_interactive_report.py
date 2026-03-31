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


def _compute_variance(sv_df: pd.DataFrame, n_pcs: int = 50):
    eigenvalues = sv_df["SINGULAR_VALUES"].values ** 2
    total = eigenvalues.sum()
    prop = eigenvalues / total
    cum = np.cumsum(prop)
    n = min(n_pcs, len(eigenvalues))
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
    """Return a lightweight dict for the scatter plots."""
    cols = ["SAMPLE", "PC1", "PC2", "PC3", "PC4",
            "SUPERPOPULATION", "RELEASE_BATCH", "INFERRED_SEX", "POPULATION"]
    sub = df[[c for c in cols if c in df.columns]].copy()
    return sub.to_dict(orient="list")


def _compute_associations(df: pd.DataFrame, n_pcs: int = 20):
    """Compute η²/r² associations for the heatmap."""
    from scipy import stats as sp_stats

    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    available = [c for c in pc_cols if c in df.columns]
    cat_vars = ["RELEASE_BATCH", "SUPERPOPULATION", "INFERRED_SEX", "POPULATION"]
    cont_vars = ["MEAN_AUTOSOMAL_COV"]
    rows = []
    for var in cat_vars:
        valid = df[var].notna()
        for pc in available:
            eta2 = eta_squared(df.loc[valid, var], df.loc[valid, pc].values)
            rows.append({"Variable": var, "PC": pc, "Value": float(eta2)})
    for var in cont_vars:
        valid = df[var].notna()
        if valid.sum() < 10:
            continue
        for pc in available:
            r, _ = sp_stats.pearsonr(df.loc[valid, var], df.loc[valid, pc])
            rows.append({"Variable": var, "PC": pc, "Value": float(r ** 2)})
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


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _build_html(
    variance_prop,
    variance_cum,
    n_scree,
    mp_cutoff_pcs,
    scatter_data,
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
    .controls { display: flex; gap: 0.5rem; flex-wrap: wrap; }
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
          Population genetic studies routinely use PCA to reveal ancestry structure and detect outlier
          samples. Traditionally this requires variant calling, which is computationally expensive and may
          introduce genotyping artefacts. NGS-PCA bypasses genotype calls entirely, operating on raw
          coverage profiles. This makes it:
        </p>
        <ul>
          <li><strong>Fast</strong> — SVD on a compact coverage matrix is orders of magnitude cheaper than
              whole-genome variant calling.</li>
          <li><strong>Genotype-free</strong> — no allele-frequency assumptions, no call-rate filters, no
              linkage-disequilibrium pruning.</li>
          <li><strong>Sensitive to batch effects</strong> — technical variation in sequencing depth is
              captured by PCs, enabling quality-control diagnostics alongside population structure.</li>
        </ul>
        <h3>About this report</h3>
        <p>
          This interactive report applies NGS-PCA to the
          <strong>1000 Genomes Project</strong> dataset. We decompose the coverage matrix, visualise
          the resulting PCs, and quantify how much of the variation in each PC is attributable to
          biological factors (continental ancestry, sex) versus technical factors (sequencing batch).
          A key concern is <em>confounding</em> — if batches are enriched for specific populations or
          sexes, batch effects can masquerade as biological signal. We formally test for such enrichment
          and present effect-size metrics (η², Cramér's V, point-biserial <em>r</em>) so that readers
          can assess the severity of any confounding.
        </p>
      </div>
    </div>

    <!-- Scree -->
    <div class="report-section" id="section-scree">
      <h2>Variance Explained by Principal Components</h2>
      <p class="description">
        The scree plot shows the proportion of total variance captured by each PC.
        The cumulative curve indicates how many PCs are needed to reach key variance thresholds.
        The dashed orange line marks the Marchenko–Pastur (MP) cutoff — PCs to the left of this line
        carry more variance than expected from random noise.
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
        Interactive scatter plots of principal component pairs.
        Toggle the colour overlay to explore population structure, batch effects, and sex differences.
        Clustering by superpopulation is expected on the leading PCs; visible batch separation may
        indicate technical confounding.
      </p>
      <div class="grid-2">
        <div class="plot-card">
          <div class="plot-card-header">
            <h3>PC1 vs PC2</h3>
            <div class="controls" id="ctrl-pca12">
              <button class="active" data-color="superpop">Superpopulation</button>
              <button data-color="batch">Batch</button>
              <button data-color="sex">Sex</button>
            </div>
          </div>
          <div class="plot-card-body"><div id="pca-12" style="height:500px"></div></div>
        </div>
        <div class="plot-card">
          <div class="plot-card-header">
            <h3>PC3 vs PC4</h3>
            <div class="controls" id="ctrl-pca34">
              <button class="active" data-color="superpop">Superpopulation</button>
              <button data-color="batch">Batch</button>
              <button data-color="sex">Sex</button>
            </div>
          </div>
          <div class="plot-card-body"><div id="pca-34" style="height:500px"></div></div>
        </div>
      </div>
    </div>

    <!-- UMAP -->
    <div class="report-section" id="section-umap">
      <h2>UMAP Projection</h2>
      <p class="description">
        Two-dimensional UMAP embedding computed from a Marchenko–Pastur-selected number of
        principal components. UMAP preserves both local neighbourhood structure and global cluster
        separation, providing an intuitive visualisation of sample relationships.
      </p>
      <div class="plot-card">
        <div class="plot-card-header">
          <h3 id="umap-title">UMAP (PCs)</h3>
          <div class="controls" id="ctrl-umap">
            <button class="active" data-color="superpop">Superpopulation</button>
            <button data-color="batch">Batch</button>
            <button data-color="sex">Sex</button>
          </div>
        </div>
        <div class="plot-card-body"><div id="umap-plot" style="height:560px"></div></div>
      </div>
    </div>

    <!-- Confounding Assessment -->
    <div class="report-section" id="section-confounding">
      <h2>Confounding Assessment</h2>
      <p class="description">
        Before interpreting PCA results, it is essential to check whether technical variables
        (sequencing batch) are confounded with biological variables (ancestry, sex). If a batch
        is enriched for a particular population, apparent "batch effects" on PCs may partly reflect
        real ancestry structure, and vice versa. We use Pearson's χ² test of independence for each
        pair and report Cramér's V as an effect size. Standardised residuals identify which specific
        cells are over- or under-represented.
      </p>
      <div id="confounding-cards"></div>
    </div>

    <!-- Heatmap -->
    <div class="report-section" id="section-heatmap">
      <h2>PC × QC Variable Associations</h2>
      <p class="description">
        Effect sizes (η² for categorical variables, r² for continuous) between each PC and QC variable.
        High values indicate that a QC variable explains substantial variance in that PC, suggesting
        that the PC captures variation driven by that variable.
      </p>
      <div class="plot-card">
        <div class="plot-card-header"><h3>Association Heatmap (η² / r²)</h3></div>
        <div class="plot-card-body"><div id="heatmap-plot" style="height:400px"></div></div>
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
        on each PC. Ideally, ancestry effects should dominate the leading PCs; large batch effects
        indicate technical confounding that may warrant correction.
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
      <div class="grid-2" style="margin-top:0.5rem;">
        <div>
          <h3 style="font-size:1rem;margin-bottom:0.75rem;">Samples per Sex</h3>
          <table class="summary-table" id="tbl-sex"></table>
        </div>
        <div>
          <h3 style="font-size:1rem;margin-bottom:0.75rem;">Top PCs by Variance Explained</h3>
          <table class="summary-table" id="tbl-toppc"></table>
        </div>
      </div>
      <div style="margin-top:0.5rem;">
        <h3 style="font-size:1rem;margin-bottom:0.75rem;">Samples per Population</h3>
        <table class="summary-table" id="tbl-pop"></table>
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
        yaxis: { ...LAYOUT_BASE.yaxis, title: '% Variance Explained' },
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
        yaxis: { ...LAYOUT_BASE.yaxis, title: 'Cumulative Variance (%)', range: [0, 105] },
        legend: { x: 0.7, y: 0.3, bgcolor: 'rgba(0,0,0,0)' },
      }, CFG);
    })();

    /* ------------------------------------------------------------------ */
    /*  PCA SCATTER HELPER                                                 */
    /* ------------------------------------------------------------------ */
    function scatterTraces(xKey, yKey, colorBy) {
      const S = DATA.scatter;
      let pal, col;
      if (colorBy === 'superpop') { pal = PAL_SUPERPOP; col = 'SUPERPOPULATION'; }
      else if (colorBy === 'batch') { pal = PAL_BATCH; col = 'RELEASE_BATCH'; }
      else { pal = PAL_SEX; col = 'INFERRED_SEX'; }

      const xArr = S[xKey], yArr = S[yKey], cArr = S[col], ids = S['SAMPLE'];
      const groups = {};
      for (let i = 0; i < xArr.length; i++) {
        const g = cArr[i];
        if (g == null) continue;
        if (!groups[g]) groups[g] = { x: [], y: [], text: [] };
        groups[g].x.push(xArr[i]);
        groups[g].y.push(yArr[i]);
        groups[g].text.push(ids[i]);
      }
      return Object.entries(groups).map(([name, d]) => ({
        x: d.x, y: d.y, text: d.text, type: 'scattergl', mode: 'markers',
        name: name,
        marker: { color: pal[name] || '#888', size: 4.5, opacity: 0.8 },
        hovertemplate: '%{text}<br>' + xKey + ': %{x:.4f}<br>' + yKey + ': %{y:.4f}<extra>' + name + '</extra>',
      }));
    }

    function plotScatter(divId, xKey, yKey, colorBy) {
      const traces = scatterTraces(xKey, yKey, colorBy);
      Plotly.react(divId, traces, {
        ...LAYOUT_BASE,
        xaxis: { ...LAYOUT_BASE.xaxis, title: xKey },
        yaxis: { ...LAYOUT_BASE.yaxis, title: yKey },
        legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
        margin: { t: 15, r: 15, b: 50, l: 60 },
      }, CFG);
    }

    /* PCA scatter initial render */
    plotScatter('pca-12', 'PC1', 'PC2', 'superpop');
    plotScatter('pca-34', 'PC3', 'PC4', 'superpop');

    /* colour toggle */
    function wireControls(ctrlId, divId, xKey, yKey) {
      document.querySelectorAll('#' + ctrlId + ' button').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('#' + ctrlId + ' button').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          plotScatter(divId, xKey, yKey, btn.dataset.color);
        });
      });
    }
    wireControls('ctrl-pca12', 'pca-12', 'PC1', 'PC2');
    wireControls('ctrl-pca34', 'pca-34', 'PC3', 'PC4');

    /* ------------------------------------------------------------------ */
    /*  UMAP                                                               */
    /* ------------------------------------------------------------------ */
    function plotUmap(colorBy) {
      const S = DATA.scatter;
      let pal, col;
      if (colorBy === 'superpop') { pal = PAL_SUPERPOP; col = 'SUPERPOPULATION'; }
      else if (colorBy === 'batch') { pal = PAL_BATCH; col = 'RELEASE_BATCH'; }
      else { pal = PAL_SEX; col = 'INFERRED_SEX'; }

      const cArr = S[col], ids = S['SAMPLE'];
      const groups = {};
      for (let i = 0; i < DATA.umap1.length; i++) {
        const g = cArr[i];
        if (g == null) continue;
        if (!groups[g]) groups[g] = { x: [], y: [], text: [] };
        groups[g].x.push(DATA.umap1[i]);
        groups[g].y.push(DATA.umap2[i]);
        groups[g].text.push(ids[i]);
      }
      const traces = Object.entries(groups).map(([name, d]) => ({
        x: d.x, y: d.y, text: d.text, type: 'scattergl', mode: 'markers',
        name: name,
        marker: { color: pal[name] || '#888', size: 5, opacity: 0.8 },
        hovertemplate: '%{text}<br>UMAP-1: %{x:.2f}<br>UMAP-2: %{y:.2f}<extra>' + name + '</extra>',
      }));
      Plotly.react('umap-plot', traces, {
        ...LAYOUT_BASE,
        xaxis: { ...LAYOUT_BASE.xaxis, title: 'UMAP-1' },
        yaxis: { ...LAYOUT_BASE.yaxis, title: 'UMAP-2' },
        legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
        margin: { t: 15, r: 15, b: 50, l: 60 },
      }, CFG);
    }
    plotUmap('superpop');
    document.getElementById('umap-title').textContent = 'UMAP (' + DATA.n_umap_pcs + ' PCs, MP)';
    document.querySelectorAll('#ctrl-umap button').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#ctrl-umap button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        plotUmap(btn.dataset.color);
      });
    });

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
        html += '<div>χ² = <span>' + c.chi2.toFixed(2) + '</span></div>';
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
        pcs.map((pc, pi) => v + ' × ' + pc + '<br>Effect size: ' + (z[vi][pi] != null ? z[vi][pi].toFixed(4) : 'N/A'))
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
      html += '<th>PC</th><th class="num">η²</th><th class="num">Point-biserial <em>r</em></th>';
      html += '<th class="num"><em>p</em> (PB)</th><th class="num">Kruskal–Wallis <em>H</em></th>';
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
        x: pcs, y: batch, type: 'bar', name: 'Batch (η²)',
        marker: { color: '#D95F02', line: { width: 0 } },
        hovertemplate: '%{x}<br>Batch η²: %{y:.4f}<extra></extra>',
      }, {
        x: pcs, y: ancestry, type: 'bar', name: 'Ancestry (η²)',
        marker: { color: '#1B9E77', line: { width: 0 } },
        hovertemplate: '%{x}<br>Ancestry η²: %{y:.4f}<extra></extra>',
      }], {
        ...LAYOUT_BASE,
        barmode: 'group',
        xaxis: { ...LAYOUT_BASE.xaxis, title: 'Principal Component', tickangle: -45 },
        yaxis: { ...LAYOUT_BASE.yaxis, title: 'η² (Effect Size)' },
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
      buildCountTable('tbl-pop', DATA.sample_summary.pop_counts, 'Population');

      /* top PCs table */
      const tpc = document.getElementById('tbl-toppc');
      let html = '<thead><tr><th>PC</th><th class="num">Variance (%)</th><th class="num">Cumulative (%)</th></tr></thead><tbody>';
      const topN = Math.min(10, DATA.variance_prop.length);
      for (let i = 0; i < topN; i++) {
        html += '<tr><td>PC' + (i + 1) + '</td>';
        html += '<td class="num">' + (DATA.variance_prop[i] * 100).toFixed(2) + '%</td>';
        html += '<td class="num">' + (DATA.variance_cum[i] * 100).toFixed(1) + '%</td></tr>';
      }
      html += '</tbody>';
      tpc.innerHTML = html;
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
    n_pcs_scree: int = 50,
    n_pcs_assoc: int = 20,
    n_pcs_umap_max: int = 0,
) -> str:
    """Generate the interactive HTML report.

    Returns the path to the written HTML file.
    """
    print("[06] Loading data …")
    sv_df = _load_singular_values(data_dir)
    merged = _load_merged(output_dir)

    n_samples = len(merged)
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
    scatter_data = _prepare_scatter_data(merged)

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

    print("[06] Computing sample summary …")
    sample_summary = _compute_sample_summary(merged)

    print("[06] Generating HTML …")
    html = _build_html(
        var_prop, var_cum, n_scree,
        mp_cutoff_pcs,
        scatter_data, umap1, umap2, n_umap_pcs,
        assoc_rows, assoc_pcs,
        batch_records, batch_pcs,
        confounding_results, sex_pc_records, sample_summary,
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
    parser.add_argument("--n-pcs-scree", type=int, default=50)
    parser.add_argument("--n-pcs-assoc", type=int, default=20)
    parser.add_argument("--n-pcs-umap-max", type=int, default=0,
                        help="Optional upper bound for MP-selected UMAP PCs (0 = no cap)")
    args = parser.parse_args()
    generate_report(args.data_dir, args.output_dir, args.report_dir,
                    args.n_pcs_scree, args.n_pcs_assoc, args.n_pcs_umap_max)


if __name__ == "__main__":
    main()
