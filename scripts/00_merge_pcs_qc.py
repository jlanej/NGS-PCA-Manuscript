#!/usr/bin/env python3
"""00_merge_pcs_qc.py - Merge principal components with sample QC metadata.

Reads svd.pcs.txt, svd.samples.txt, and sample_qc.tsv.  Strips the
'.by1000.' suffix from PC sample IDs so they match the QC SAMPLE_ID column,
then maps the 26 1000 Genomes sub-populations to their superpopulation
(AFR / AMR / EAS / EUR / SAS).

Outputs
-------
{output_dir}/merged_pcs_qc.tsv
"""

import argparse
import os
import sys

import pandas as pd

# 1000 Genomes sub-population → superpopulation mapping
POP_TO_SUPERPOP = {
    "ACB": "AFR", "ASW": "AFR", "ESN": "AFR", "GWD": "AFR",
    "LWK": "AFR", "MSL": "AFR", "YRI": "AFR",
    "CLM": "AMR", "MXL": "AMR", "PEL": "AMR", "PUR": "AMR",
    "CDX": "EAS", "CHB": "EAS", "CHS": "EAS", "JPT": "EAS", "KHV": "EAS",
    "CEU": "EUR", "FIN": "EUR", "GBR": "EUR", "IBS": "EUR", "TSI": "EUR",
    "BEB": "SAS", "GIH": "SAS", "ITU": "SAS", "PJL": "SAS", "STU": "SAS",
}


def merge_pcs_qc(data_dir: str, output_dir: str, n_samples: int = 0) -> pd.DataFrame:
    """Load and merge PC scores with sample QC metadata.

    Parameters
    ----------
    data_dir : str
        Root data directory containing ``ngspca_output/`` and ``qc_output/``.
    output_dir : str
        Directory where merged TSV will be written.
    n_samples : int
        If >0, subsample to this many rows (for CI speed).

    Returns
    -------
    pd.DataFrame
        Merged dataframe.
    """
    pcs = pd.read_csv(os.path.join(data_dir, "ngspca_output", "svd.pcs.txt"), sep="\t")
    pcs["SAMPLE"] = pcs["SAMPLE"].str.replace(r"\.by1000\.$", "", regex=True)

    qc = pd.read_csv(os.path.join(data_dir, "qc_output", "sample_qc.tsv"), sep="\t")
    # The file's SUPERPOPULATION column actually contains family-structure
    # labels (unrel / father / mother …).  Rename it and derive the real
    # superpopulation from POPULATION.
    qc = qc.rename(columns={"SUPERPOPULATION": "FAMILY_ROLE"})
    qc["SUPERPOPULATION"] = qc["POPULATION"].map(POP_TO_SUPERPOP)

    merged = pcs.merge(qc, left_on="SAMPLE", right_on="SAMPLE_ID", how="inner")
    merged = merged.drop(columns=["SAMPLE_ID"])

    if n_samples > 0:
        merged = merged.head(n_samples)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "merged_pcs_qc.tsv")
    merged.to_csv(out_path, sep="\t", index=False)
    print(f"[00] Merged {len(merged)} samples → {out_path}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PCs with QC metadata")
    parser.add_argument("--data-dir", default=os.environ.get("NGSPCA_DATA_DIR", "1000G"))
    parser.add_argument("--output-dir", default=os.environ.get("NGSPCA_OUTPUT_DIR", "output"))
    parser.add_argument("--n-samples", type=int,
                        default=int(os.environ.get("NGSPCA_SUBSET", "0")))
    args = parser.parse_args()
    merge_pcs_qc(args.data_dir, args.output_dir, args.n_samples)


if __name__ == "__main__":
    main()
