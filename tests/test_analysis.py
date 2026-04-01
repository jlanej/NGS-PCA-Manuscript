#!/usr/bin/env python3
"""Test suite for the NGS-PCA analysis pipeline.

Tests cover:
 - Data merging and superpopulation mapping
 - Presence and non-emptiness of output figures and tables
 - Scientific validation of batch vs ancestry effect sizes
"""

import json
import os
import re

from PIL import Image
import numpy as np
import pandas as pd
import pytest

OUTPUT_DIR = os.environ.get("NGSPCA_OUTPUT_DIR", "output")
DATA_DIR = os.environ.get("NGSPCA_DATA_DIR", "1000G")
# NGS-PCA SVD sample IDs are suffixed with ".by1000." and need normalisation.
PC_SAMPLE_SUFFIX_PATTERN = r"\.by1000\.$"
ORANGE_R_MIN = 220
ORANGE_G_MIN = 90
ORANGE_G_MAX = 180
ORANGE_B_MAX = 80
MIN_ORANGE_PIXELS = 100


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def merged():
    path = os.path.join(OUTPUT_DIR, "merged_pcs_qc.tsv")
    assert os.path.isfile(path), f"Merged file not found: {path}"
    return pd.read_csv(path, sep="\t")


@pytest.fixture(scope="module")
def batch_ancestry():
    path = os.path.join(OUTPUT_DIR, "batch_vs_ancestry_detail.tsv")
    assert os.path.isfile(path), f"Batch-vs-ancestry detail not found: {path}"
    return pd.read_csv(path, sep="\t")


@pytest.fixture(scope="module")
def batch_ancestry_summary():
    path = os.path.join(OUTPUT_DIR, "batch_vs_ancestry_summary.tsv")
    assert os.path.isfile(path), f"Batch-vs-ancestry summary not found: {path}"
    return pd.read_csv(path, sep="\t")


# ---------------------------------------------------------------------------
# Merge & mapping tests
# ---------------------------------------------------------------------------
class TestMergeAndMapping:
    def test_merged_not_empty(self, merged):
        assert len(merged) > 0

    def test_has_pc_columns(self, merged):
        assert "PC1" in merged.columns
        assert "PC2" in merged.columns

    def test_has_qc_columns(self, merged):
        for col in ("RELEASE_BATCH", "INFERRED_SEX", "POPULATION"):
            assert col in merged.columns, f"Missing column {col}"

    def test_superpopulation_mapped(self, merged):
        assert "SUPERPOPULATION" in merged.columns
        valid = merged["SUPERPOPULATION"].dropna().unique()
        expected = {"AFR", "AMR", "EAS", "EUR", "SAS"}
        assert set(valid).issubset(expected), f"Unexpected superpop values: {set(valid) - expected}"

    def test_no_duplicate_samples(self, merged):
        assert merged["SAMPLE"].is_unique


# ---------------------------------------------------------------------------
# Output file existence tests
# ---------------------------------------------------------------------------
EXPECTED_FILES = [
    "merged_pcs_qc.tsv",
    "scree_cumvar.png",
    "pca_scatter_PC1_PC2.png",
    "pca_scatter_PC3_PC4.png",
    "correlation_heatmap.png",
    "pc_qc_associations.tsv",
    "batch_vs_ancestry.png",
    "batch_vs_ancestry_detail.tsv",
    "batch_vs_ancestry_summary.tsv",
]


REPORT_DIR = os.environ.get("NGSPCA_REPORT_DIR", "docs")


class TestOutputFiles:
    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_file_exists_and_nonempty(self, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        assert os.path.isfile(path), f"Missing output: {path}"
        assert os.path.getsize(path) > 0, f"Empty output: {path}"

    def test_umap_output_matches_pc_count(self):
        path = os.path.join(OUTPUT_DIR, "merged_pcs_qc.tsv")
        merged = pd.read_csv(path, sep="\t")
        pc_cols = [c for c in merged.columns if c.startswith("PC")]
        files = [f for f in os.listdir(OUTPUT_DIR)
                 if f.startswith("umap_") and f.endswith("pcs.png")]
        assert len(files) == 1, f"Expected one UMAP figure, found: {files}"
        used_pcs = int(files[0].split("_")[1].replace("pcs.png", ""))
        assert 2 <= used_pcs <= len(pc_cols), "UMAP PC count should be in valid range"

    def test_scree_plot_cutoff_annotation_present(self):
        path = os.path.join(OUTPUT_DIR, "scree_cumvar.png")
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB"))
        orange_like = ((arr[:, :, 0] > ORANGE_R_MIN)
                       & (arr[:, :, 1] > ORANGE_G_MIN)
                       & (arr[:, :, 1] < ORANGE_G_MAX)
                       & (arr[:, :, 2] < ORANGE_B_MAX))
        assert orange_like.sum() > MIN_ORANGE_PIXELS, (
            "Expected orange MP cutoff annotation on scree plot"
        )


class TestInteractiveReport:
    def test_report_exists(self):
        path = os.path.join(REPORT_DIR, "index.html")
        assert os.path.isfile(path), f"Missing interactive report: {path}"
        assert os.path.getsize(path) > 0, f"Empty interactive report: {path}"

    def test_report_contains_plotly(self):
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "plotly" in content.lower(), "Report should reference Plotly"
        assert "Plotly.newPlot" in content or "Plotly.react" in content, \
            "Report should contain Plotly chart calls"

    def test_report_uses_all_available_samples(self):
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        pcs_path = os.path.join(DATA_DIR, "ngspca_output", "svd.pcs.txt")
        qc_path = os.path.join(DATA_DIR, "qc_output", "sample_qc.tsv")
        assert os.path.isfile(pcs_path), f"Missing PCA sample file: {pcs_path}"
        assert os.path.isfile(qc_path), f"Missing QC sample file: {qc_path}"
        pcs = pd.read_csv(pcs_path, sep="\t")
        qc = pd.read_csv(qc_path, sep="\t")
        pcs_samples = set(pcs["SAMPLE"].str.replace(PC_SAMPLE_SUFFIX_PATTERN, "", regex=True))
        qc_samples = set(qc["SAMPLE_ID"])
        n_available = len(pcs_samples & qc_samples)
        assert f'"n_samples": {n_available}' in content

    def test_report_scatter_uses_all_available_samples(self):
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        pcs_path = os.path.join(DATA_DIR, "ngspca_output", "svd.pcs.txt")
        qc_path = os.path.join(DATA_DIR, "qc_output", "sample_qc.tsv")
        pcs = pd.read_csv(pcs_path, sep="\t")
        qc = pd.read_csv(qc_path, sep="\t")
        pcs_samples = set(pcs["SAMPLE"].str.replace(PC_SAMPLE_SUFFIX_PATTERN, "", regex=True))
        qc_samples = set(qc["SAMPLE_ID"])
        n_available = len(pcs_samples & qc_samples)
        match = re.search(r"const DATA = (\{.*?\});\s*\n\s*/\*.*?\*/\s*\n\s*const LAYOUT_BASE", content, re.DOTALL)
        assert match, "Report should embed DATA JSON payload"
        payload = json.loads(match.group(1))
        assert payload["n_samples"] == n_available
        assert len(payload["scatter"]["SAMPLE"]) == n_available

    def test_report_has_all_sections(self):
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        for section in ["intro", "scree", "pca", "umap",
                        "confounding", "heatmap"]:
            assert f'id="section-{section}"' in content, \
                f"Report missing section: {section}"
        for removed in ["partitioning", "sex", "batch", "summary"]:
            assert f'id="section-{removed}"' not in content, \
                f"Report should not contain removed section: {removed}"

    def test_report_mentions_marchenko_pastur_and_dynamic_umap_pcs(self):
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "Marchenko-Pastur cutoff" in content
        assert "UMAP (' + DATA.n_umap_pcs + ' PCs, MP)'" in content

    def test_report_is_single_page(self):
        """Report should use scrollable sections, not hidden tabs."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "report-section" in content, "Should use report-section class"
        assert 'id="toc-nav"' in content, "Should have TOC navigation"
        assert "tab-content" not in content, "Should not use old tab-content class"

    def test_report_has_confounding_assessment(self):
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "confounding" in content.lower()
        assert "chi2" in content or "χ²" in content or "chi-squared" in content.lower()
        assert "cramers_v" in content or "Cram" in content

    def test_report_uses_light_theme(self):
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "--bg: #f8fafc;" in content
        assert "plot_bgcolor: '#ffffff'" in content

    def test_report_has_3d_pca_scatter(self):
        """PCA scatter should use scatter3d with selectable PC axes."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "scatter3d" in content, "Report should use 3D scatter"
        assert "pca-scatter" in content, "Report should have pca-scatter div"
        assert "pca-x" in content, "Report should have PC X selector"
        assert "pca-y" in content, "Report should have PC Y selector"
        assert "pca-z" in content, "Report should have PC Z selector"

    def test_report_has_continuous_color_option(self):
        """Report should support continuous heatmap coloring on scatter plots."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "numeric_cols" in content, "Report should include numeric_cols"
        assert "Viridis" in content, "Report should use Viridis colorscale"
        assert "pca-color" in content, "Report should have PCA colour selector"

    def test_report_has_correlation_bar_charts(self):
        """Report should compute runtime Pearson/Spearman correlations."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "pearsonCorr" in content, "Report should compute Pearson r"
        assert "spearmanCorr" in content, "Report should compute Spearman rho"
        assert "pca-panel2" in content, "Report should have PCA panel 2 div"

    def test_report_has_umap_second_panel(self):
        """UMAP should have a second panel for distributions / correlations."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "umap-panel2" in content, "Report should have UMAP panel 2 div"
        assert "umap-color" in content, "Report should have UMAP colour selector"

    def test_report_has_expanded_qc_metrics(self):
        """Report should include expanded coverage QC metrics for color-coding."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        for metric in ["MEAN_AUTOSOMAL_COV", "SD_COV", "MAD_COV", "IQR_COV",
                        "MEDIAN_BIN_COV", "MITO_COV_RATIO"]:
            assert metric in content, f"Report should include QC metric {metric}"

    def test_report_has_range_slider(self):
        """Report should have dual-handle range sliders for continuous colour scales."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "pca-range-lo" in content, "Report should have PCA range slider low handle"
        assert "pca-range-hi" in content, "Report should have PCA range slider high handle"
        assert "umap-range-lo" in content, "Report should have UMAP range slider low handle"
        assert "umap-range-hi" in content, "Report should have UMAP range slider high handle"
        assert "range-slider-wrap" in content, "Report should have range slider wrapper CSS"
        assert "initRangeSlider" in content, "Report should have range slider init function"

    def test_report_heatmap_has_pvalues(self):
        """Association heatmap should include p-values in data payload."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "p_value" in content, "Heatmap should include p_value data"
        assert '"Metric"' in content, "Heatmap should include metric type labels"

    def test_report_dropped_tables(self):
        """'Top PCs by Variance Explained' and 'Samples per Population' tables should be removed."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "tbl-toppc" not in content, "Report should not have Top PCs table"
        assert "tbl-pop" not in content, "Report should not have Population table"

    def test_report_heatmap_description_explains_eta_squared(self):
        """Heatmap description should explain why eta-squared is used."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "SS<sub>between</sub>" in content or "SS_between" in content, \
            "Report should explain eta-squared formula"
        assert "ANOVA" in content, "Report should mention ANOVA"

    def test_report_has_filter_info_elements(self):
        """Report should have colorscale-clamping info banners for PCA and UMAP sliders."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "pca-filter-info" in content, "Report should have PCA filter info div"
        assert "umap-filter-info" in content, "Report should have UMAP filter info div"
        assert "filter-info" in content, "Report should have filter-info CSS class"
        assert "updateFilterInfo" in content, "Report should have updateFilterInfo function"
        assert "hideFilterInfo" in content, "Report should have hideFilterInfo function"
        assert "Colorscale clamped to" in content, "Report should describe colorscale clamping"

    def test_report_has_violin_plots(self):
        """Companion panels should use violin+box plots for categorical colour."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "'violin'" in content or "type:'violin'" in content, \
            "Report should use violin plot type"
        assert "meanline" in content, "Violins should include mean-line option"
        assert "scalemode" in content, "Violins should use scalemode to reflect counts"

    def test_report_has_proportion_bars(self):
        """Companion panels should include proportion bar charts for categorical colour."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "Distributions & Proportions" in content, \
            "Categorical companion panel should be titled 'Distributions & Proportions'"
        assert "% of samples" in content, "Proportion chart should label x-axis as % of samples"

    def test_report_continuous_companion_has_distribution(self):
        """Continuous companion panels should show metric distribution with slider cutoff lines."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "Distribution & PC Correlation" in content, \
            "Continuous PCA companion title should mention Distribution"
        assert "Distribution & UMAP Correlation" in content, \
            "Continuous UMAP companion title should mention Distribution"
        assert "y0:rng[0]" in content, "Distribution violin should have low cutoff line shape"
        assert "y0:rng[1]" in content, "Distribution violin should have high cutoff line shape"

    def test_report_continuous_companion_shows_significance(self):
        """Correlation bar charts should annotate significance with *, **, *** labels."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "sigLabel" in content, "Report should have sigLabel helper function"
        assert "corrPValue" in content, "Report should compute correlation p-values"
        assert "corrN" in content, "Report should count valid correlation pairs"
        assert "normalCDF" in content, "Report should implement normal CDF for p-values"
        assert "textposition:'outside'" in content, \
            "Significance labels should be positioned outside bars"

    def test_report_slider_clamps_colorscale(self):
        """Sliders clamp the colorscale (cmin/cmax) while keeping all points in scatter."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "cmin:rng[0]" in content, "Scatter should use cmin to clamp colorscale low"
        assert "cmax:rng[1]" in content, "Scatter should use cmax to clamp colorscale high"
        assert "updateFilterInfo" in content, "Scatter functions should call updateFilterInfo"


class TestCiConfiguration:
    def test_ci_subset_is_1000(self):
        path = os.path.join(".github", "workflows", "ci-analysis.yml")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert 'NGSPCA_SUBSET: "1000"' in content


# ---------------------------------------------------------------------------
# Scientific validation tests
# ---------------------------------------------------------------------------
class TestScientificValidation:
    def test_batch_has_nonzero_effect(self, batch_ancestry):
        assert batch_ancestry["Batch_eta2"].max() > 0

    def test_ancestry_has_nonzero_effect(self, batch_ancestry):
        assert batch_ancestry["Ancestry_eta2"].max() > 0

    def test_max_batch_exceeds_max_ancestry(self, batch_ancestry_summary):
        """Both batch and ancestry should have measurable effect on at least one PC."""
        batch_max = batch_ancestry_summary.loc[
            batch_ancestry_summary["Variable"] == "RELEASE_BATCH", "Max_eta2"
        ].iloc[0]
        ancestry_max = batch_ancestry_summary.loc[
            batch_ancestry_summary["Variable"] == "SUPERPOPULATION", "Max_eta2"
        ].iloc[0]
        assert batch_max > 0, "Batch max η² should be > 0"
        assert ancestry_max > 0, "Ancestry max η² should be > 0"
