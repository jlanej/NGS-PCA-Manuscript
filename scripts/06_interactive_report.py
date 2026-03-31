#!/usr/bin/env python3
"""06_interactive_report.py - Generate an interactive HTML dashboard.

Reads all pipeline outputs and produces a single self-contained HTML file
with interactive Plotly charts covering every analysis step:
  - Scree / cumulative variance
  - PCA scatter plots (PC1-PC2, PC3-PC4)
  - UMAP projection
  - PC × QC association heatmap
  - Batch vs ancestry comparison

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
from utils import eta_squared


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


def _compute_umap(df: pd.DataFrame, n_pcs: int = 20):
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


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _build_html(
    variance_prop,
    variance_cum,
    n_scree,
    scatter_data,
    umap1,
    umap2,
    assoc_rows,
    assoc_pcs,
    batch_records,
    batch_pcs,
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
        "scatter": scatter_data,
        "umap1": umap1,
        "umap2": umap2,
        "assoc_rows": assoc_rows,
        "assoc_pcs": assoc_pcs,
        "batch_records": batch_records,
        "batch_pcs": batch_pcs,
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
    nav {
      display: flex;
      gap: 0;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      overflow-x: auto;
      padding: 0 1rem;
    }
    nav button {
      background: none;
      border: none;
      color: var(--text-dim);
      padding: 0.85rem 1.25rem;
      font-size: 0.9rem;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
      white-space: nowrap;
      font-family: inherit;
    }
    nav button:hover { color: var(--text); background: var(--surface2); }
    nav button.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
      font-weight: 600;
    }
    .tab-content { display: none; padding: 1.5rem 2rem 3rem; max-width: 1400px; margin: 0 auto; }
    .tab-content.active { display: block; }
    .tab-content h2 {
      font-size: 1.25rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: var(--text);
    }
    .tab-content .description {
      color: var(--text-dim);
      font-size: 0.9rem;
      margin-bottom: 1.25rem;
      max-width: 800px;
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
      </div>
    </div>

    <nav id="tab-nav">
      <button class="active" data-tab="scree">Variance Explained</button>
      <button data-tab="pca">PCA Scatter</button>
      <button data-tab="umap">UMAP Projection</button>
      <button data-tab="heatmap">PC–QC Associations</button>
      <button data-tab="batch">Batch vs Ancestry</button>
    </nav>

    <!-- Scree -->
    <div class="tab-content active" id="tab-scree">
      <h2>Variance Explained by Principal Components</h2>
      <p class="description">
        The scree plot shows the proportion of total variance captured by each PC.
        The cumulative curve indicates how many PCs are needed to reach key variance thresholds.
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
    <div class="tab-content" id="tab-pca">
      <h2>PCA Scatter Plots</h2>
      <p class="description">
        Interactive scatter plots of principal component pairs.
        Toggle the colour overlay to explore population structure, batch effects, and sex differences.
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
    <div class="tab-content" id="tab-umap">
      <h2>UMAP Projection</h2>
      <p class="description">
        Two-dimensional UMAP embedding computed from the top 20 principal components.
        UMAP preserves both local neighbourhood structure and global cluster separation.
      </p>
      <div class="plot-card">
        <div class="plot-card-header">
          <h3>UMAP (20 PCs)</h3>
          <div class="controls" id="ctrl-umap">
            <button class="active" data-color="superpop">Superpopulation</button>
            <button data-color="batch">Batch</button>
            <button data-color="sex">Sex</button>
          </div>
        </div>
        <div class="plot-card-body"><div id="umap-plot" style="height:560px"></div></div>
      </div>
    </div>

    <!-- Heatmap -->
    <div class="tab-content" id="tab-heatmap">
      <h2>PC × QC Variable Associations</h2>
      <p class="description">
        Effect sizes (η² for categorical variables, r² for continuous) between each PC and QC variable.
        High values indicate that a QC variable explains substantial variance in that PC.
      </p>
      <div class="plot-card">
        <div class="plot-card-header"><h3>Association Heatmap (η² / r²)</h3></div>
        <div class="plot-card-body"><div id="heatmap-plot" style="height:400px"></div></div>
      </div>
    </div>

    <!-- Batch vs Ancestry -->
    <div class="tab-content" id="tab-batch">
      <h2>Batch vs Ancestry Effect Sizes</h2>
      <p class="description">
        Compares η² (eta-squared) effect sizes of sequencing batch and continental ancestry
        on each PC.  Ideally, ancestry effects should dominate; large batch effects indicate
        technical confounding that warrants correction.
      </p>
      <div class="plot-card">
        <div class="plot-card-header"><h3>η² per Principal Component</h3></div>
        <div class="plot-card-body"><div id="batch-bar" style="height:460px"></div></div>
      </div>
    </div>

    <footer>
      Generated by the NGS-PCA analysis pipeline · Plotly.js interactive charts
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

    /* ------------------------------------------------------------------ */
    /*  TABS                                                               */
    /* ------------------------------------------------------------------ */
    document.querySelectorAll('#tab-nav button').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#tab-nav button').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
        window.dispatchEvent(new Event('resize'));   // trigger Plotly relayout
      });
    });

    /* ------------------------------------------------------------------ */
    /*  SCREE                                                              */
    /* ------------------------------------------------------------------ */
    (function() {
      const pcs = DATA.variance_prop.map((_, i) => 'PC' + (i + 1));
      const pctProp = DATA.variance_prop.map(v => (v * 100));
      const pctCum  = DATA.variance_cum.map(v => (v * 100));

      Plotly.newPlot('scree-bar', [{
        x: pcs, y: pctProp, type: 'bar',
        marker: { color: pctProp, colorscale: [[0,'#38bdf8'],[1,'#818cf8']], line: { width: 0 } },
        hovertemplate: '%{x}: %{y:.2f}%<extra></extra>',
      }], {
        ...LAYOUT_BASE,
        xaxis: { ...LAYOUT_BASE.xaxis, title: 'Principal Component', tickangle: -45, dtick: 5 },
        yaxis: { ...LAYOUT_BASE.yaxis, title: '% Variance Explained' },
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
    document.querySelectorAll('#ctrl-umap button').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#ctrl-umap button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        plotUmap(btn.dataset.color);
      });
    });

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

    print("[06] Preparing scatter data …")
    scatter_data = _prepare_scatter_data(merged)

    print("[06] Computing UMAP embedding …")
    umap1, umap2 = _compute_umap(merged)

    print("[06] Computing PC–QC associations …")
    assoc_rows, assoc_pcs = _compute_associations(merged, n_pcs_assoc)

    print("[06] Computing batch vs ancestry …")
    batch_records, batch_pcs = _compute_batch_ancestry(merged, n_pcs_assoc)

    print("[06] Generating HTML …")
    html = _build_html(
        var_prop, var_cum, n_scree,
        scatter_data, umap1, umap2,
        assoc_rows, assoc_pcs,
        batch_records, batch_pcs,
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
    args = parser.parse_args()
    generate_report(args.data_dir, args.output_dir, args.report_dir,
                    args.n_pcs_scree, args.n_pcs_assoc)


if __name__ == "__main__":
    main()
