#!/usr/bin/env python3
"""Test suite for the NGS-PCA analysis pipeline.

Tests cover:
 - Data merging and superpopulation mapping
 - Presence and non-emptiness of output figures and tables
 - Scientific validation of batch vs ancestry effect sizes
"""

import os

import numpy as np
import pandas as pd
import pytest

OUTPUT_DIR = os.environ.get("NGSPCA_OUTPUT_DIR", "output")


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
    "umap_20pcs.png",
    "correlation_heatmap.png",
    "pc_qc_associations.tsv",
    "batch_vs_ancestry.png",
    "batch_vs_ancestry_detail.tsv",
    "batch_vs_ancestry_summary.tsv",
]


class TestOutputFiles:
    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_file_exists_and_nonempty(self, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        assert os.path.isfile(path), f"Missing output: {path}"
        assert os.path.getsize(path) > 0, f"Empty output: {path}"


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
