#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_GSE138266_per_patient.py
----------------------------------
Reconstruct GSE138266 as one h5ad per donor x tissue sample from raw 10x MEX
files, so the dataset can be loaded through the patient-aware branch of
`data_io.load_dataset()`.

Expected raw filenames:
    GSM{n}_{donor}_{tissue}_GRCh38_barcodes.tsv.gz
    GSM{n}_{donor}_{tissue}_GRCh38_genes.tsv.gz
    GSM{n}_{donor}_{tissue}_GRCh38_matrix.mtx.gz

Outputs:
    data/processed/processed/GSE138266/<DONOR>_<PB|CSF>.h5ad

Notes:
- Donor prefixes MS -> MS, PST/PTC -> HC.
- PBMCs is normalised to PB in the output filenames and obs['tissue'].
- Each output file is one biological sample (donor x tissue).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread

from config.config import DATA_DIR, RAW_DIR

RAW_DIR_266 = Path(RAW_DIR) / "GSE138266_RAW"
OUT_DIR = Path(DATA_DIR) / "processed" / "GSE138266"

CONDITION_MAP: dict[str, str] = {
    "MS": "MS",
    "PST": "HC",
    "PTC": "HC",
}

MIN_GENES = 200
MAX_GENES = 6000
MAX_PCT_MT = 20


def infer_condition(donor: str) -> str:
    for prefix, cond in CONDITION_MAP.items():
        if donor.startswith(prefix):
            return cond
    return "Unknown"


def normalise_tissue(tissue: str) -> str:
    t = str(tissue)
    if t in {"PBMCs", "PBMC", "PB", "blood", "peripheral blood"}:
        return "PB"
    if t == "CSF":
        return "CSF"
    return t


def load_sample(matrix_f: Path, barcode_f: Path, gene_f: Path, donor: str, tissue: str, gsm: str) -> sc.AnnData:
    X = mmread(str(matrix_f)).T.tocsr().astype(np.float32)
    barcodes = pd.read_csv(barcode_f, header=None, sep="\t")[0].astype(str).tolist()
    genes = pd.read_csv(gene_f, header=None, sep="\t")

    var = pd.DataFrame({"gene_ids": genes[0].astype(str).tolist()}, index=genes[1].astype(str).tolist())
    if genes.shape[1] > 2:
        var["feature_types"] = genes[2].astype(str).tolist()

    tissue_norm = normalise_tissue(tissue)
    obs_names = [f"{gsm}_{b}" for b in barcodes]
    obs = pd.DataFrame(
        {
            "barcode": barcodes,
            "patient": donor,
            "donor": donor,
            "tissue": tissue_norm,
            "condition": infer_condition(donor),
            "dataset": "GSE138266",
            "gsm": gsm,
            "sample": f"{gsm}_{donor}_{tissue}",
        },
        index=obs_names,
    )

    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def qc_filter(adata: sc.AnnData, label: str) -> sc.AnnData:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    n_before = adata.n_obs
    mask = (
        (adata.obs["n_genes_by_counts"] >= MIN_GENES)
        & (adata.obs["n_genes_by_counts"] <= MAX_GENES)
        & (adata.obs["pct_counts_mt"] <= MAX_PCT_MT)
    )
    adata = adata[mask].copy()
    print(f"    QC [{label}]: {n_before:>5} â†’ {adata.n_obs:>5} cells (removed {n_before - adata.n_obs})")
    return adata


def safe_write(adata: sc.AnnData, path: Path) -> None:
    tmp = path.with_suffix(".tmp.h5ad")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(tmp)
        shutil.move(str(tmp), str(path))
        print(f"âœ“ Saved â†’ {path}")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_files = sorted(RAW_DIR_266.glob("*_matrix.mtx.gz"))
    if not matrix_files:
        raise FileNotFoundError(f"No *_matrix.mtx.gz files found in: {RAW_DIR}")

    print(f"Found {len(matrix_files)} matrix files in {RAW_DIR}\n")

    patient_rows = []
    for mf in matrix_files:
        m = re.match(r"(GSM\d+)_([^_]+)_([^_]+)_GRCh38_matrix\.mtx\.gz", mf.name)
        if not m:
            print(f"[warn] Skipping unrecognised filename: {mf.name}")
            continue

        gsm, donor, tissue = m.groups()
        tissue_norm = normalise_tissue(tissue)
        patient_key = f"{donor}_{tissue_norm}"
        label = f"{donor}/{tissue_norm}"

        bf = mf.parent / f"{gsm}_{donor}_{tissue}_GRCh38_barcodes.tsv.gz"
        gf = mf.parent / f"{gsm}_{donor}_{tissue}_GRCh38_genes.tsv.gz"
        if not bf.exists() or not gf.exists():
            print(f"[warn] Missing barcodes or genes file for {label}, skipping")
            continue

        print(f"Loading {label} â€¦")
        adata = load_sample(mf, bf, gf, donor, tissue, gsm)
        adata = qc_filter(adata, label)
        if adata.n_obs == 0:
            print(f"[warn] No cells left after QC for {label}, skipping")
            continue

        out_path = OUT_DIR / f"{patient_key}.h5ad"
        safe_write(adata, out_path)
        patient_rows.append({
            "patient_key": patient_key,
            "donor": donor,
            "tissue": tissue_norm,
            "condition": infer_condition(donor),
            "gsm": gsm,
            "cells": int(adata.n_obs),
            "file": str(out_path),
        })

    if not patient_rows:
        raise RuntimeError("No per-patient files were written.")

    manifest = pd.DataFrame(patient_rows).sort_values(["condition", "donor", "tissue"])
    manifest_path = OUT_DIR / "GSE138266_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nSaved manifest â†’ {manifest_path}")

    print("\nSuggested datasets.yaml block:\n")
    print("GSE138266:")
    print("  patients:")
    for row in manifest.itertuples(index=False):
        print(f"    {row.patient_key}:")
        print(f"      condition: {row.condition}")

    print("\nDone. Update datasets.yaml to use the patients block and remove the global h5ad entry for GSE138266.")


if __name__ == "__main__":
    main()