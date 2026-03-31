"""Shared statistical utilities for NGS-PCA analysis scripts."""

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
