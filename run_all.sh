#!/usr/bin/env bash
set -euo pipefail
# run_all.sh - Orchestrator for the full NGS-PCA analysis pipeline.
#
# Environment variables (all optional):
#   NGSPCA_DATA_DIR   - root data directory       (default: 1000G)
#   NGSPCA_OUTPUT_DIR - output directory           (default: output)
#   NGSPCA_SUBSET     - subsample N rows (0=all)   (default: 0)
#
# Usage:
#   ./run_all.sh                          # full run
#   NGSPCA_SUBSET=200 ./run_all.sh        # quick CI run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${NGSPCA_DATA_DIR:-1000G}"
OUTPUT_DIR="${NGSPCA_OUTPUT_DIR:-output}"
SUBSET="${NGSPCA_SUBSET:-0}"

echo "=== NGS-PCA Analysis Pipeline ==="
echo "  Data dir:   ${DATA_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Subset:     ${SUBSET} (0 = all samples)"
echo ""

echo "[Step 0] Merging PCs and QC data …"
python "${SCRIPT_DIR}/scripts/00_merge_pcs_qc.py" \
    --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" --n-samples "${SUBSET}"

echo ""
echo "[Step 1] Scree / cumulative variance plot …"
python "${SCRIPT_DIR}/scripts/01_scree_plot.py" \
    --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"

echo ""
echo "[Step 2] PCA scatter plots …"
python "${SCRIPT_DIR}/scripts/02_pca_scatter.py" --output-dir "${OUTPUT_DIR}"

echo ""
echo "[Step 3] UMAP projection …"
python "${SCRIPT_DIR}/scripts/03_umap_projection.py" --output-dir "${OUTPUT_DIR}"

echo ""
echo "[Step 4] Correlation heatmap …"
python "${SCRIPT_DIR}/scripts/04_correlation_heatmap.py" --output-dir "${OUTPUT_DIR}"

echo ""
echo "[Step 5] Batch vs ancestry …"
python "${SCRIPT_DIR}/scripts/05_batch_vs_ancestry.py" --output-dir "${OUTPUT_DIR}"

echo ""
echo "[Step 6] Interactive HTML report …"
python "${SCRIPT_DIR}/scripts/06_interactive_report.py" \
    --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}"

echo ""
echo "=== Pipeline complete.  Outputs in ${OUTPUT_DIR}/ ==="
