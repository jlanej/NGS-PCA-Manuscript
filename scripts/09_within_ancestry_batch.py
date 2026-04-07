#!/usr/bin/env python3
"""09_within_ancestry_batch.py - Within-ancestry stratified batch effect test.

The strongest evidence that NGS-PCA captures batch — not ancestry — is to show
that **within a single superpopulation** (e.g., EUR-only samples), batch still
drives PC separation.  This script performs that test.

For each superpopulation group (AFR, AMR, EAS, EUR, SAS):
  1. Subset to samples in that group only
  2. Compute η²(RELEASE_BATCH) and η²(FAMILY_ROLE) for each of the top N PCs
  3. Run a permutation test within each subset (optional via --n-permutations)
  4. Report results in a TSV table and grouped bar chart (one facet per
     superpopulation)

Outputs:
  - within_ancestry_batch.tsv         (η² per PC per superpopulation × variable)
  - within_ancestry_batch.png         (grouped bar chart, one facet per group)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import eta_squared, marchenko_pastur_pc_count_from_data_dir


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

SUPERPOPULATIONS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
VARIABLES = ["RELEASE_BATCH", "FAMILY_ROLE"]
VARIABLE_LABELS = {
    "RELEASE_BATCH": "Batch (η²)",
    "FAMILY_ROLE": "Family Role (η²)",
}
VARIABLE_COLORS = {
    "RELEASE_BATCH": "#D95F02",
    "FAMILY_ROLE": "#7570B3",
}


def _empirical_pvalue(observed: float, null_vals: np.ndarray) -> float:
    """Phipson & Smyth (2010) conservative empirical p-value."""
    n_perm = len(null_vals)
    count = int(np.sum(null_vals >= observed))
    return (count + 1) / (n_perm + 1)


def _compute_eta2_with_permutation(
    sub: pd.DataFrame,
    pc_cols: list[str],
    variable: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Compute observed η² and (optionally) permutation p-values for *variable*.

    Returns a list of dicts, one per PC.
    """
    valid = sub[variable].notna()
    labels = sub.loc[valid, variable]
    values = sub.loc[valid, pc_cols].values
    n_valid = int(valid.sum())

    records = []
    for k, pc in enumerate(pc_cols):
        obs = eta_squared(labels, values[:, k])
        record: dict = {
            "PC": pc,
            "observed_eta2": float(obs) if not np.isnan(obs) else 0.0,
            "n_samples": n_valid,
        }

        if n_permutations > 0 and not np.isnan(obs) and labels.nunique() >= 2:
            labels_arr = labels.values.copy()
            null_vals = np.empty(n_permutations)
            shuffled = pd.Series(labels_arr.copy())
            for i in range(n_permutations):
                shuffled[:] = rng.permutation(labels_arr)
                null_vals[i] = eta_squared(shuffled, values[:, k])
            record["mean_null_eta2"] = float(null_vals.mean())
            record["p_value"] = _empirical_pvalue(record["observed_eta2"], null_vals)
            record["n_permutations"] = n_permutations
        else:
            record["mean_null_eta2"] = float("nan")
            record["p_value"] = float("nan")
            record["n_permutations"] = n_permutations

        records.append(record)
    return records


def within_ancestry_batch(
    output_dir: str,
    data_dir: str = "1000G",
    n_pcs: int = 0,
    n_permutations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run within-ancestry stratified batch test and write results + figure.

    Parameters
    ----------
    output_dir : str
        Directory containing ``merged_pcs_qc.tsv`` (pipeline outputs).
    data_dir : str
        Root data directory (for Marchenko–Pastur PC selection).
    n_pcs : int
        Maximum number of PCs to analyse (0 = Marchenko–Pastur auto).
    n_permutations : int
        Permutations per superpopulation × variable (0 = skip permutation test).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Long-format results table with columns:
        Superpopulation, Variable, PC, n_samples, observed_eta2,
        mean_null_eta2, p_value, n_permutations.
    """
    rng = np.random.default_rng(seed)

    # ---- Load data ----------------------------------------------------------
    merged_path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    df = pd.read_csv(merged_path, sep="\t")
    df["RELEASE_BATCH"] = df["RELEASE_BATCH"].astype(str)

    # ---- Determine PCs via Marchenko–Pastur ---------------------------------
    sv_path = os.path.join(data_dir, "ngspca_output", "svd.singularvalues.txt")
    eigenvalues = pd.read_csv(sv_path, sep="\t")["SINGULAR_VALUES"].values ** 2
    max_pcs = n_pcs if n_pcs > 0 else None
    mp_n_pcs, _ = marchenko_pastur_pc_count_from_data_dir(
        data_dir, eigenvalues, max_pcs=max_pcs,
    )
    pc_cols = [f"PC{i}" for i in range(1, mp_n_pcs + 1)]
    pc_cols = [c for c in pc_cols if c in df.columns]
    print(f"[09] Using {len(pc_cols)} MP-selected PCs")

    # ---- Per-superpopulation analysis ---------------------------------------
    records: list[dict] = []

    # Spawn independent RNG streams so per-superpopulation results are
    # reproducible regardless of which groups are present.
    superpops_present = [
        g for g in SUPERPOPULATIONS
        if g in df.get("SUPERPOPULATION", pd.Series(dtype=str)).unique()
    ]

    rng_streams = rng.spawn(len(superpops_present) * len(VARIABLES))
    stream_idx = 0

    for group in superpops_present:
        sub = df[df["SUPERPOPULATION"] == group].reset_index(drop=True)
        n_group = len(sub)
        print(f"[09]   {group}: {n_group} samples")

        for var in VARIABLES:
            if var not in df.columns:
                print(f"[09]     Skipping {var} (not in data)")
                stream_idx += 1
                continue

            n_unique = sub[var].dropna().nunique()
            if n_unique < 2:
                print(f"[09]     Skipping {var} in {group} (only {n_unique} unique value(s))")
                stream_idx += 1
                continue

            sub_rng = rng_streams[stream_idx]
            stream_idx += 1

            rows = _compute_eta2_with_permutation(
                sub, pc_cols, var, n_permutations, sub_rng,
            )
            for row in rows:
                row["Superpopulation"] = group
                row["Variable"] = var
                records.append(row)

    if not records:
        print("[09] No records produced — check that SUPERPOPULATION and batch columns are present")
        return pd.DataFrame()

    result = pd.DataFrame(records)
    col_order = [
        "Superpopulation", "Variable", "PC", "n_samples",
        "observed_eta2", "mean_null_eta2", "p_value", "n_permutations",
    ]
    result = result[[c for c in col_order if c in result.columns]]

    # ---- Write TSV ----------------------------------------------------------
    tsv_path = os.path.join(output_dir, "within_ancestry_batch.tsv")
    result.to_csv(tsv_path, sep="\t", index=False)
    print(f"[09] Results → {tsv_path}")

    # ---- Grouped bar chart --------------------------------------------------
    _plot_grouped_bar(result, pc_cols, superpops_present, output_dir)

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _sig_label(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _plot_grouped_bar(
    result: pd.DataFrame,
    pc_cols: list[str],
    superpops: list[str],
    output_dir: str,
) -> None:
    """Grouped bar chart — one facet (subplot) per superpopulation."""
    n_groups = len(superpops)
    if n_groups == 0:
        return

    n_pcs = len(pc_cols)
    fig, axes = plt.subplots(
        1, n_groups,
        figsize=(max(4, n_pcs * 0.6) * n_groups, 5),
        squeeze=False,
    )

    x = np.arange(n_pcs)
    width = 0.35

    for col_idx, group in enumerate(superpops):
        ax = axes[0][col_idx]
        sub_g = result[result["Superpopulation"] == group]

        for bar_idx, var in enumerate(VARIABLES):
            sub_v = sub_g[sub_g["Variable"] == var].set_index("PC")
            ys = [float(sub_v.loc[pc, "observed_eta2"]) if pc in sub_v.index else 0.0
                  for pc in pc_cols]
            pvals = [float(sub_v.loc[pc, "p_value"]) if pc in sub_v.index else float("nan")
                     for pc in pc_cols]
            offset = (bar_idx - 0.5) * width
            bars = ax.bar(
                x + offset, ys, width,
                label=VARIABLE_LABELS[var],
                color=VARIABLE_COLORS[var],
                alpha=0.85,
            )
            # Significance annotations
            y_max = max(ys) if ys else 0.0
            ann_offset = y_max * 0.04 if y_max > 0 else 0.01
            for i, (y, p) in enumerate(zip(ys, pvals)):
                lbl = _sig_label(p)
                if lbl:
                    ax.text(
                        x[i] + offset, y + ann_offset,
                        lbl, ha="center", va="bottom", fontsize=6,
                    )

        ax.set_title(group, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(pc_cols, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("η²" if col_idx == 0 else "")
        ax.set_ylim(bottom=0)
        if col_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Within-Ancestry Stratified Batch Effect (η² per PC)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out_path = os.path.join(output_dir, "within_ancestry_batch.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[09] Grouped bar chart → {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Within-ancestry stratified batch effect test",
    )
    parser.add_argument("--output-dir",
                        default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    parser.add_argument("--data-dir",
                        default=os.environ.get("NGSPCA_DATA_DIR", "1000G"))
    parser.add_argument("--n-pcs", type=int, default=0,
                        help="Number of PCs (0 = Marchenko–Pastur auto)")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Permutations per group × variable (0 = skip)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    within_ancestry_batch(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        n_pcs=args.n_pcs,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
