# NGS-PCA Manuscript

Analyses and figures for the manuscript describing the **NGS-PCA** method on
the 1000 Genomes (1000G) dataset. The pipeline produces publication-quality
figures that highlight how NGS-PCA detects batch effects in whole-genome
sequencing data.

---

## Repository Layout

```
.
├── 1000G/                          # Input data
│   ├── ngspca_output/              # SVD / PCA output
│   │   ├── svd.pcs.txt            #   200 PCs for 3 202 samples
│   │   ├── svd.singularvalues.txt #   Singular values
│   │   ├── svd.samples.txt        #   Sample IDs
│   │   └── svd.bins.txt           #   Genomic bins
│   └── qc_output/
│       └── sample_qc.tsv          #   Sample metadata & QC metrics
├── scripts/                        # Modular analysis scripts
│   ├── 00_merge_pcs_qc.py         #   Merge PCs + QC, map superpopulations
│   ├── 01_scree_plot.py           #   Scree & cumulative variance
│   ├── 02_pca_scatter.py          #   Pairwise PC scatter plots
│   ├── 03_umap_projection.py      #   UMAP from top PCs
│   ├── 04_correlation_heatmap.py  #   PC × QC association heatmaps
│   └── 05_batch_vs_ancestry.py    #   Batch vs ancestry effect sizes
├── tests/
│   └── test_analysis.py           # pytest test suite
├── run_all.sh                      # Orchestrator script
├── Dockerfile                      # Containerised environment
├── requirements.txt                # Python dependencies
└── .github/workflows/
    ├── build-and-publish.yml       # Build & push Docker image to GHCR
    └── ci-analysis.yml             # Run analysis + tests on subset
```

## Quick Start

### 1. Local (virtualenv)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full run (all 3 202 samples)
bash run_all.sh

# Quick subset run (1000 samples)
NGSPCA_SUBSET=1000 bash run_all.sh

# Run tests
NGSPCA_OUTPUT_DIR=output pytest tests/ -v
```

### 2. Docker

```bash
docker build -t ngspca-manuscript .

# Mount data and output directories
docker run --rm \
  -v "$(pwd)/1000G:/data" \
  -v "$(pwd)/output:/output" \
  ngspca-manuscript

# Subset run
docker run --rm \
  -v "$(pwd)/1000G:/data" \
  -v "$(pwd)/output:/output" \
  -e NGSPCA_SUBSET=1000 \
  ngspca-manuscript
```

### 3. Apptainer / Singularity (HPC)

```bash
# Pull from GHCR (after CI publishes)
apptainer pull docker://ghcr.io/jlanej/ngs-pca-manuscript:main

# Run
apptainer exec \
  --bind 1000G:/data,output:/output \
  ngs-pca-manuscript_main.sif \
  bash /app/run_all.sh
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NGSPCA_DATA_DIR` | `1000G` | Root directory containing `ngspca_output/` and `qc_output/` |
| `NGSPCA_OUTPUT_DIR` | `output` | Directory for all output figures and tables |
| `NGSPCA_SUBSET` | `0` | Number of samples to use (0 = all) |

## Outputs

| File | Description |
|---|---|
| `merged_pcs_qc.tsv` | Merged PC scores + sample metadata |
| `scree_cumvar.png` | Scree plot and cumulative variance explained |
| `pca_scatter_PC1_PC2.png` | PC1 vs PC2 coloured by superpopulation, batch, sex |
| `pca_scatter_PC3_PC4.png` | PC3 vs PC4 coloured by superpopulation, batch, sex |
| `umap_<N>pcs.png` | UMAP projection from Marchenko–Pastur-selected PCs |
| `correlation_heatmap.png` | PC × QC variable association heatmap (η² / r²) |
| `pc_qc_associations.tsv` | Association statistics table |
| `batch_vs_ancestry.png` | Batch vs ancestry η² grouped bar chart |
| `batch_vs_ancestry_detail.tsv` | Per-PC batch and ancestry η² |
| `batch_vs_ancestry_summary.tsv` | Mean and max η² summary |

## CI / CD

- **`build-and-publish.yml`** – Builds the Docker image and pushes to GHCR on
  pushes to `main` and version tags.
- **`ci-analysis.yml`** – Runs the full pipeline on a 1000-sample subset,
  executes the test suite, and uploads output artifacts.

## License

MIT – see [LICENSE](LICENSE).
