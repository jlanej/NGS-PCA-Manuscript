"""Shared statistical utilities for NGS-PCA analysis scripts."""

import os
import numpy as np
import pandas as pd


def eta_squared(categorical: pd.Series, continuous: np.ndarray) -> float:
    """Compute η² (eta-squared) for a categorical-continuous pair.

    Parameters
    ----------
    categorical : pd.Series
        Group labels (NaN values are excluded via the caller).
    continuous : np.ndarray
        Numeric values for each observation.

    Returns
    -------
    float
        η² value in [0, 1], or NaN if undefined.
    """
    groups = categorical.values
    unique = [g for g in np.unique(groups) if pd.notna(g)]
    if len(unique) < 2:
        return np.nan
    group_data = [continuous[groups == g] for g in unique]
    grand_mean = np.nanmean(continuous)
    ss_between = sum(len(g) * (np.nanmean(g) - grand_mean) ** 2 for g in group_data)
    ss_total = np.nansum((continuous - grand_mean) ** 2)
    if ss_total == 0:
        return np.nan
    return ss_between / ss_total


def _count_rows(path: str) -> int:
    """Count data rows in a tab-separated file with a header."""
    with open(path, encoding="utf-8") as fh:
        n_rows = sum(1 for _ in fh) - 1  # subtract header
    if n_rows < 1:
        raise ValueError(f"Expected at least one data row in {path}")
    return n_rows


def marchenko_pastur_pc_count(
    eigenvalues: np.ndarray,
    n_samples: int,
    n_features: int,
    min_pcs: int = 2,
    max_pcs: int | None = None,
) -> tuple[int, float]:
    """Select informative PCs using a Marchenko–Pastur upper edge cutoff."""
    if len(eigenvalues) == 0:
        return min_pcs, np.nan
    if n_samples < 1 or n_features < 1:
        raise ValueError("n_samples and n_features must both be >= 1")

    # Use the lower half of the spectrum as a robust bulk-noise estimate.
    # For len(eigenvalues) == 1, the tail is that single value.
    tail = eigenvalues[len(eigenvalues) // 2:]
    sigma2 = float(np.median(tail))

    # Aspect ratio for sample-covariance eigenvalues (bounded to (0, 1]).
    q = min(float(n_samples), float(n_features)) / max(float(n_samples), float(n_features))
    lambda_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
    n_sig = int(np.sum(eigenvalues > lambda_plus))

    n_sel = max(min_pcs, n_sig)
    if max_pcs is not None:
        n_sel = min(n_sel, max_pcs)
    n_sel = min(n_sel, len(eigenvalues))
    return n_sel, float(lambda_plus)


def marchenko_pastur_pc_count_from_data_dir(
    data_dir: str,
    eigenvalues: np.ndarray,
    min_pcs: int = 2,
    max_pcs: int | None = None,
) -> tuple[int, float]:
    """Compute Marchenko–Pastur PC count using ngsPCA bins as feature count."""
    bins_path = os.path.join(data_dir, "ngspca_output", "svd.bins.txt")
    samples_path = os.path.join(data_dir, "ngspca_output", "svd.samples.txt")
    n_features = _count_rows(bins_path)
    n_samples = _count_rows(samples_path)
    return marchenko_pastur_pc_count(
        eigenvalues=eigenvalues,
        n_samples=n_samples,
        n_features=n_features,
        min_pcs=min_pcs,
        max_pcs=max_pcs,
    )
