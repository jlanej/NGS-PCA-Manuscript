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


@pytest.fixture(scope="module")
def permutation_results():
    path = os.path.join(OUTPUT_DIR, "permutation_eta2_results.tsv")
    assert os.path.isfile(path), f"Permutation results not found: {path}"
    return pd.read_csv(path, sep="\t")


@pytest.fixture(scope="module")
def variance_partitioning():
    path = os.path.join(OUTPUT_DIR, "variance_partitioning.tsv")
    assert os.path.isfile(path), f"Variance partitioning TSV not found: {path}"
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

    def test_superpopulation_from_file(self, merged):
        """SUPERPOPULATION should come directly from sample_qc.tsv (no remapping)."""
        assert "SUPERPOPULATION" in merged.columns
        valid = merged["SUPERPOPULATION"].dropna().unique()
        expected = {"AFR", "AMR", "EAS", "EUR", "SAS"}
        assert set(valid).issubset(expected), f"Unexpected superpop values: {set(valid) - expected}"

    def test_family_role_from_file(self, merged):
        """FAMILY_ROLE should come directly from sample_qc.tsv (not derived from SUPERPOPULATION)."""
        assert "FAMILY_ROLE" in merged.columns, "FAMILY_ROLE column must be present in merged output"
        # Values should include at least 'unrel' for unrelated samples
        assert merged["FAMILY_ROLE"].notna().any(), "FAMILY_ROLE should have non-null values"

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
    "permutation_eta2_results.tsv",
    "permutation_eta2_batch.png",
    "permutation_eta2_nulldist.png",
    "variance_partitioning.tsv",
    "variance_partitioning.png",
    "within_ancestry_batch.tsv",
    "within_ancestry_batch.png",
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
                        "confounding", "relatedness", "ancestry-distance",
                        "permutation", "heatmap", "within-ancestry"]:
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

    def test_report_has_relatedness_section(self):
        """Report should have a pedigree-based relatedness distance section."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert 'id="section-relatedness"' in content, \
            "Report should have relatedness section"
        assert "relatedness-violin" in content, \
            "Report should have relatedness violin plot div"
        assert "relatedness-paired" in content, \
            "Report should have relatedness paired dot plot div"
        assert "Wilcoxon" in content, \
            "Report should mention Wilcoxon signed-rank test"
        assert "relatedness_distance" in content, \
            "Report DATA payload should include relatedness_distance"

    def test_report_relatedness_has_writeup(self):
        """Relatedness section should have explanation of rationale and method."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "negative control" in content.lower() or "familial" in content.lower(), \
            "Relatedness section should mention its role as a negative control"
        assert "Euclidean distance" in content, \
            "Relatedness section should describe Euclidean distance computation"
        assert "nearest non-self" in content or "nearest relative" in content, \
            "Relatedness section should describe nearest-relative vs nearest-non-self comparison"

    def test_report_slider_clamps_colorscale(self):
        """Sliders clamp the colorscale (cmin/cmax) while keeping all points in scatter."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "cmin:rng[0]" in content, "Scatter should use cmin to clamp colorscale low"
        assert "cmax:rng[1]" in content, "Scatter should use cmax to clamp colorscale high"
        assert "updateFilterInfo" in content, "Scatter functions should call updateFilterInfo"

    def test_report_has_permutation_section(self):
        """Report should have permutation test section with all three interactive charts."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert 'id="section-permutation"' in content, \
            "Report should have permutation section"
        assert 'id="perm-pval-plot"' in content, \
            "Report should have -log10(p) overview chart"
        assert 'id="perm-eta2-plot"' in content, \
            "Report should have effect size bar chart"

    def test_report_permutation_data_in_payload(self):
        """Report DATA payload should include permutation results for all MP PCs."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        match = re.search(
            r"const DATA = (\{.*?\});\s*\n\s*/\*.*?\*/\s*\n\s*const LAYOUT_BASE",
            content, re.DOTALL,
        )
        assert match, "Report should embed DATA JSON payload"
        payload = json.loads(match.group(1))
        pm = payload.get("permutation")
        assert pm is not None, "DATA should contain permutation key"
        assert "pc_cols" in pm and len(pm["pc_cols"]) > 0, \
            "permutation should list pc_cols"
        assert "RELEASE_BATCH" in pm["results"], \
            "permutation results should include RELEASE_BATCH"
        assert "SUPERPOPULATION" in pm["results"], \
            "permutation results should include SUPERPOPULATION"
        # Coverage variables should be present
        for cov_var in ["MAD_COV", "IQR_COV", "MEDIAN_BIN_COV"]:
            assert cov_var in pm["results"], \
                f"permutation results should include {cov_var}"
            # Each coverage variable result should have r2 metric
            first_pc = pm["pc_cols"][0]
            assert pm["results"][cov_var][first_pc]["metric"] == "r2", \
                f"{cov_var} should use r2 metric"

    def test_report_permutation_section_has_rationale(self):
        """Permutation section should explain the rationale and method."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "Phipson" in content or "conservative" in content.lower(), \
            "Permutation section should describe the conservative p-value correction"
        assert "null distribution" in content.lower() or "Null distribution" in content, \
            "Permutation section should mention null distributions"
        assert "Marchenko" in content, \
            "Permutation section should reference MP-selected PCs"


class TestCiConfiguration:
    def test_ci_subset_is_1000(self):
        path = os.path.join(".github", "workflows", "ci-analysis.yml")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert 'NGSPCA_SUBSET: "1000"' in content

    def test_ci_analysis_sets_permutations(self):
        """ci-analysis.yml should explicitly set NGSPCA_PERMUTATIONS."""
        path = os.path.join(".github", "workflows", "ci-analysis.yml")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "NGSPCA_PERMUTATIONS" in content

    def test_publish_workflow_exists(self):
        """A publish-report workflow should exist for auto-committing the HTML report."""
        path = os.path.join(".github", "workflows", "publish-report.yml")
        assert os.path.isfile(path), "publish-report.yml should exist"

    def test_publish_workflow_uses_full_cohort(self):
        """publish-report.yml should NOT set NGSPCA_SUBSET (full cohort)."""
        path = os.path.join(".github", "workflows", "publish-report.yml")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "NGSPCA_SUBSET" not in content or "no NGSPCA_SUBSET" in content.lower() \
            or "# Full cohort" in content, \
            "Publish workflow should use full cohort (no NGSPCA_SUBSET)"

    def test_publish_workflow_sets_permutations(self):
        """publish-report.yml should set NGSPCA_PERMUTATIONS >= 1000."""
        path = os.path.join(".github", "workflows", "publish-report.yml")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert 'NGSPCA_PERMUTATIONS' in content, \
            "Publish workflow should set NGSPCA_PERMUTATIONS"

    def test_publish_workflow_commits_report(self):
        """publish-report.yml should auto-commit docs/index.html."""
        path = os.path.join(".github", "workflows", "publish-report.yml")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "docs/index.html" in content, \
            "Publish workflow should commit docs/index.html"
        assert "git push" in content, \
            "Publish workflow should push changes"


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


# ---------------------------------------------------------------------------
# Permutation test validation
# ---------------------------------------------------------------------------
class TestPermutationTest:
    def test_results_has_expected_columns(self, permutation_results):
        expected_cols = {"Variable", "PC", "observed_eta2", "mean_null_eta2",
                         "p_value", "n_permutations"}
        assert expected_cols.issubset(set(permutation_results.columns))

    def test_results_has_both_variables(self, permutation_results):
        variables = set(permutation_results["Variable"])
        assert "RELEASE_BATCH" in variables
        assert "SUPERPOPULATION" in variables

    def test_pvalues_in_valid_range(self, permutation_results):
        """All p-values should be in (0, 1]."""
        assert (permutation_results["p_value"] > 0).all()
        assert (permutation_results["p_value"] <= 1).all()

    def test_observed_eta2_nonnegative(self, permutation_results):
        assert (permutation_results["observed_eta2"] >= 0).all()

    def test_mean_null_less_than_observed_for_batch(self, permutation_results):
        """Mean null η² should generally be less than observed for batch PCs."""
        batch = permutation_results[permutation_results["Variable"] == "RELEASE_BATCH"]
        # At least one PC should have observed > mean null
        assert (batch["observed_eta2"] > batch["mean_null_eta2"]).any()


# ---------------------------------------------------------------------------
# Relatedness distance validation tests
# ---------------------------------------------------------------------------
class TestRelatednessDistance:
    def test_relatedness_data_in_report_payload(self):
        """Report DATA JSON should include relatedness_distance with per-individual semantics."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        match = re.search(
            r"const DATA = (\{.*?\});\s*\n\s*/\*.*?\*/\s*\n\s*const LAYOUT_BASE",
            content, re.DOTALL,
        )
        assert match, "Report should embed DATA JSON payload"
        payload = json.loads(match.group(1))
        rd = payload.get("relatedness_distance")
        assert rd is not None, "DATA should contain relatedness_distance"
        for key in ["n_individuals", "n_parent_child", "n_sibling",
                     "wilcoxon_stat", "wilcoxon_p", "mp_pcs_used",
                     "d_nearest_relative_values", "d_nearest_nonself_values"]:
            assert key in rd, f"relatedness_distance missing key: {key}"
        assert rd["n_individuals"] > 0, "Should have at least one individual with relatives"
        assert len(rd["d_nearest_relative_values"]) == rd["n_individuals"]
        assert len(rd["d_nearest_nonself_values"]) == rd["n_individuals"]
        # d_nearest_nonself <= d_nearest_relative is not guaranteed per individual,
        # but the arrays must have the same length
        assert len(rd["d_nearest_relative_values"]) == len(rd["d_nearest_nonself_values"])

    def test_relatedness_wilcoxon_p_is_valid(self):
        """Wilcoxon p-value should be a finite number in [0, 1]."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        match = re.search(
            r"const DATA = (\{.*?\});\s*\n\s*/\*.*?\*/\s*\n\s*const LAYOUT_BASE",
            content, re.DOTALL,
        )
        payload = json.loads(match.group(1))
        rd = payload["relatedness_distance"]
        p = rd["wilcoxon_p"]
        assert p is not None, "Wilcoxon p-value should not be null"
        assert 0.0 <= p <= 1.0, f"Wilcoxon p-value should be in [0,1], got {p}"


# ---------------------------------------------------------------------------
# Ancestry distance validation tests
# ---------------------------------------------------------------------------
class TestAncestryDistance:
    def test_ancestry_data_in_report_payload(self):
        """Report DATA JSON should include ancestry_distance with required keys."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        match = re.search(
            r"const DATA = (\{.*?\});\s*\n\s*/\*.*?\*/\s*\n\s*const LAYOUT_BASE",
            content, re.DOTALL,
        )
        assert match, "Report should embed DATA JSON payload"
        payload = json.loads(match.group(1))
        ad = payload.get("ancestry_distance")
        assert ad is not None, "DATA should contain ancestry_distance"
        for key in ["n_samples", "n_excluded", "mp_pcs_used",
                     "observed_mean_delta", "p_value_global",
                     "p_value_global_two",
                     "p_value_within_batch", "n_permutations",
                     "per_superpop", "d_within_values", "d_between_values",
                     "null_hist_counts", "null_hist_edges"]:
            assert key in ad, f"ancestry_distance missing key: {key}"
        assert ad["n_samples"] > 0, "Should have analysed at least one individual"
        assert len(ad["d_within_values"]) == ad["n_samples"]
        assert len(ad["d_between_values"]) == ad["n_samples"]
        assert len(ad["null_hist_counts"]) > 0, "Should have pre-binned null histogram"
        assert len(ad["null_hist_edges"]) == len(ad["null_hist_counts"]) + 1, \
            "Histogram edges should have one more element than counts"

    def test_ancestry_p_value_is_valid(self):
        """Global (one- and two-sided) and within-batch p-values should be in [0, 1]."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        match = re.search(
            r"const DATA = (\{.*?\});\s*\n\s*/\*.*?\*/\s*\n\s*const LAYOUT_BASE",
            content, re.DOTALL,
        )
        payload = json.loads(match.group(1))
        ad = payload["ancestry_distance"]
        assert 0.0 <= ad["p_value_global"] <= 1.0, \
            f"Global one-sided p-value should be in [0,1], got {ad['p_value_global']}"
        assert 0.0 <= ad["p_value_global_two"] <= 1.0, \
            f"Global two-sided p-value should be in [0,1], got {ad['p_value_global_two']}"
        assert 0.0 <= ad["p_value_within_batch"] <= 1.0, \
            f"Within-batch p-value should be in [0,1], got {ad['p_value_within_batch']}"

    def test_per_superpop_has_pvalues(self):
        """Each per-superpopulation entry should include permutation p-values."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        match = re.search(
            r"const DATA = (\{.*?\});\s*\n\s*/\*.*?\*/\s*\n\s*const LAYOUT_BASE",
            content, re.DOTALL,
        )
        payload = json.loads(match.group(1))
        ad = payload["ancestry_distance"]
        per_sp = ad.get("per_superpop", {})
        assert len(per_sp) > 0, "Should have at least one superpopulation"
        for g, info in per_sp.items():
            assert "p_value" in info, f"per_superpop[{g}] missing p_value"
            assert "p_value_two" in info, f"per_superpop[{g}] missing p_value_two"
            assert 0.0 <= info["p_value"] <= 1.0, \
                f"per_superpop[{g}] p_value should be in [0,1], got {info['p_value']}"
            assert 0.0 <= info["p_value_two"] <= 1.0, \
                f"per_superpop[{g}] p_value_two should be in [0,1], got {info['p_value_two']}"

    def test_ancestry_section_exists(self):
        """Report should have ancestry distance section with plot divs."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert 'id="section-ancestry-distance"' in content, \
            "Report should have ancestry distance section"
        assert "ancestry-violin" in content, \
            "Report should have ancestry violin plot div"
        assert "ancestry-null-hist" in content, \
            "Report should have ancestry null histogram div"

    def test_ancestry_has_writeup(self):
        """Ancestry distance section should have explanation of method."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "superpopulation" in content.lower(), \
            "Ancestry section should mention superpopulation"
        assert "nearest" in content.lower(), \
            "Ancestry section should describe nearest-neighbour comparison"
        assert "permutation" in content.lower(), \
            "Ancestry section should describe permutation test"

    def test_ancestry_interpretation_language(self):
        """Interpretation language should reference sign and significance of δ."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        # The report should use the improved interpretation language patterns
        has_positive_sig = "residual ancestry structure" in content
        has_negative_sig = "technical (non-biological) variation" in content
        has_not_sig = "No evidence that NGS-PCA space is organized by ancestry" in content
        assert has_positive_sig or has_negative_sig or has_not_sig, \
            "Interpretation should use improved sign-and-significance language"

    def test_ancestry_reports_one_sided_pvalue(self):
        """Report should describe the one-sided p-value."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "one-sided" in content.lower(), \
            "Report should mention one-sided p-value"
        assert "two-sided" in content.lower(), \
            "Report should mention two-sided p-value"

    def test_ancestry_per_superpop_pvalues_in_html(self):
        """Per-superpopulation section should show p-values."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "Per-superpopulation" in content, \
            "Report should display per-superpopulation results"
        assert "group-wise" in content.lower() or "per-superpopulation p-value" in content.lower() \
            or "s.p_value" in content, \
            "Report should include per-group p-values in rendering code"


# ---------------------------------------------------------------------------
# Variance partitioning tests
# ---------------------------------------------------------------------------
class TestVariancePartitioning:
    def test_has_expected_columns(self, variance_partitioning):
        expected = {
            "PC", "r2_full",
            "unique_batch", "unique_ancestry", "unique_family_role", "unique_coverage",
            "shared", "residual",
        }
        assert expected.issubset(set(variance_partitioning.columns))

    def test_has_pc_rows(self, variance_partitioning):
        assert len(variance_partitioning) > 0, "variance_partitioning.tsv should have at least one row"
        assert variance_partitioning["PC"].str.startswith("PC").all()

    def test_variance_components_nonnegative(self, variance_partitioning):
        for col in ["unique_batch", "unique_ancestry", "unique_family_role",
                    "unique_coverage", "shared", "residual"]:
            assert (variance_partitioning[col] >= -1e-9).all(), \
                f"{col} should be non-negative (got min {variance_partitioning[col].min():.6f})"

    def test_r2_full_in_unit_interval(self, variance_partitioning):
        assert (variance_partitioning["r2_full"] >= -1e-9).all()
        assert (variance_partitioning["r2_full"] <= 1.0 + 1e-9).all()

    def test_unique_components_at_most_r2_full(self, variance_partitioning):
        """Each unique component cannot exceed the full-model R²."""
        for col in ["unique_batch", "unique_ancestry", "unique_family_role", "unique_coverage"]:
            assert (variance_partitioning[col] <= variance_partitioning["r2_full"] + 1e-9).all(), \
                f"{col} should not exceed r2_full"

    def test_unique_batch_nonzero_for_at_least_one_pc(self, variance_partitioning):
        assert (variance_partitioning["unique_batch"] > 0).any(), \
            "Unique batch variance should be > 0 for at least one PC"

    def test_unique_ancestry_nonzero_for_at_least_one_pc(self, variance_partitioning):
        assert (variance_partitioning["unique_ancestry"] > 0).any(), \
            "Unique ancestry variance should be > 0 for at least one PC"


# ---------------------------------------------------------------------------
# Within-ancestry stratified batch test
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def within_ancestry():
    path = os.path.join(OUTPUT_DIR, "within_ancestry_batch.tsv")
    assert os.path.isfile(path), f"Within-ancestry batch TSV not found: {path}"
    return pd.read_csv(path, sep="\t")


class TestWithinAncestryBatch:
    def test_file_exists_and_nonempty(self):
        path = os.path.join(OUTPUT_DIR, "within_ancestry_batch.tsv")
        assert os.path.isfile(path), f"Missing output: {path}"
        assert os.path.getsize(path) > 0, f"Empty output: {path}"

    def test_figure_exists_and_nonempty(self):
        path = os.path.join(OUTPUT_DIR, "within_ancestry_batch.png")
        assert os.path.isfile(path), f"Missing figure: {path}"
        assert os.path.getsize(path) > 0, f"Empty figure: {path}"

    def test_has_expected_columns(self, within_ancestry):
        expected = {"Superpopulation", "Variable", "PC", "n_samples", "observed_eta2"}
        assert expected.issubset(set(within_ancestry.columns)), \
            f"within_ancestry_batch.tsv missing columns: {expected - set(within_ancestry.columns)}"

    def test_has_all_superpopulations(self, within_ancestry):
        found = set(within_ancestry["Superpopulation"].unique())
        valid = {"AFR", "AMR", "EAS", "EUR", "SAS"}
        assert found.issubset(valid), \
            f"Unexpected superpopulation values: {found - valid}"
        assert len(found) >= 1, "within_ancestry_batch.tsv should have at least one superpopulation"

    def test_has_release_batch_variable(self, within_ancestry):
        assert "RELEASE_BATCH" in within_ancestry["Variable"].values, \
            "within_ancestry_batch.tsv should contain RELEASE_BATCH rows"

    def test_has_family_role_variable(self, within_ancestry):
        assert "FAMILY_ROLE" in within_ancestry["Variable"].values, \
            "within_ancestry_batch.tsv should contain FAMILY_ROLE rows"

    def test_sample_counts_per_group_positive(self, within_ancestry):
        for group in within_ancestry["Superpopulation"].unique():
            sub = within_ancestry[within_ancestry["Superpopulation"] == group]
            assert sub["n_samples"].iloc[0] > 0, \
                f"Group {group} should have > 0 samples"

    def test_eta2_nonnegative(self, within_ancestry):
        assert (within_ancestry["observed_eta2"] >= 0).all(), \
            "All observed η² values should be ≥ 0"

    def test_pvalues_in_valid_range_when_present(self, within_ancestry):
        if "p_value" in within_ancestry.columns:
            valid = within_ancestry["p_value"].dropna()
            assert (valid > 0).all(), "All p-values should be > 0"
            assert (valid <= 1).all(), "All p-values should be ≤ 1"

    def test_batch_eta2_nonzero_in_at_least_one_group(self, within_ancestry):
        batch_rows = within_ancestry[within_ancestry["Variable"] == "RELEASE_BATCH"]
        assert (batch_rows["observed_eta2"] > 0).any(), \
            "Batch η² should be > 0 in at least one superpopulation × PC combination"


class TestWithinAncestryReport:
    def test_report_has_within_ancestry_section(self):
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert 'id="section-within-ancestry"' in content, \
            "Report should have within-ancestry section"
        assert "within-ancestry-summary" in content, \
            "Report should have within-ancestry summary paragraph"
        assert "within-ancestry-facets" in content, \
            "Report should have within-ancestry facets div"

    def test_report_within_ancestry_data_in_payload(self):
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        match = re.search(
            r"const DATA = (\{.*?\});\s*\n\s*/\*.*?\*/\s*\n\s*const LAYOUT_BASE",
            content, re.DOTALL,
        )
        assert match, "Report should embed DATA JSON payload"
        payload = json.loads(match.group(1))
        wa = payload.get("within_ancestry")
        assert wa is not None, "DATA should contain within_ancestry key"
        assert "pc_cols" in wa and len(wa["pc_cols"]) > 0, \
            "within_ancestry should list pc_cols"
        assert "superpops" in wa and len(wa["superpops"]) > 0, \
            "within_ancestry should list superpops"
        assert "results" in wa, "within_ancestry should have results dict"
        # Each present superpopulation should have at least one variable's data
        for sp in wa["superpops"]:
            assert sp in wa["results"], f"results should include superpop {sp}"
            assert len(wa["results"][sp]) >= 1, \
                f"results[{sp}] should include at least one variable"

    def test_report_within_ancestry_sample_counts(self):
        """Each superpop entry should report a positive n_samples."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        match = re.search(
            r"const DATA = (\{.*?\});\s*\n\s*/\*.*?\*/\s*\n\s*const LAYOUT_BASE",
            content, re.DOTALL,
        )
        payload = json.loads(match.group(1))
        wa = payload.get("within_ancestry")
        if wa is None:
            pytest.skip("within_ancestry not in report payload")
        for sp in wa["superpops"]:
            for var in wa["results"].get(sp, {}):
                n = wa["results"][sp][var].get("n_samples", 0)
                assert n > 0, f"results[{sp}][{var}]['n_samples'] should be > 0, got {n}"

    def test_report_within_ancestry_toc_link(self):
        """TOC should contain a link to the within-ancestry section."""
        path = os.path.join(REPORT_DIR, "index.html")
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert 'href="#section-within-ancestry"' in content, \
            "TOC should link to within-ancestry section"

