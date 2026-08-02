#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_io.py
----------
Data loading and gating helpers for the B-cell zinc / CXCR3 pipeline.

Public API
----------
load_cfg(yaml_path)                                      -- parse datasets.yaml
build_patient_path(dataset_name, patient_key, base_dir)  -- reconstruct .h5ad path
load_dataset(name, ds_cfg, base_dir, cfg)                -- load one dataset
load_all_datasets(cfg, base_dir)                         -- load and concatenate all
filter_conditions(adata, ds_cfg)                         -- apply keep_conditions
gate_bcells(adata, markers)                              -- gate B cells by markers
add_cxcr3_group(adata, cxcr3_gene)                      -- add cxcr3_group column
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import scanpy as sc
import yaml


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_cfg(yaml_path: str | Path) -> dict[str, Any]:
    """Load datasets.yaml and return the parsed dict."""
    with open(yaml_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def build_patient_path(
    dataset_name: str,
    patient_key: str,
    base_dir: str | Path = "data/processed",
) -> Path:
    """
    Reconstruct the .h5ad path from dataset name and patient key.
    Convention: <base_dir>/processed/<dataset_name>/<patient_key>.h5ad
    """
    return Path(base_dir) / "processed" / dataset_name / f"{patient_key}.h5ad"


def _resolve_h5ad_path(
    dataset_name: str,
    ds_cfg: dict[str, Any],
    base_dir: str | Path,
) -> Path:
    """
    Return the global .h5ad path for datasets that are not patient-based.
    Uses explicit 'h5ad' key if present, otherwise reconstructs from name.
    """
    if "h5ad" in ds_cfg:
        return Path(ds_cfg["h5ad"])
    return Path(base_dir) / dataset_name / f"{dataset_name}_processed.h5ad"


# ---------------------------------------------------------------------------
# Tissue extraction from patient key
# ---------------------------------------------------------------------------

def _tissue_from_key(patient_key: str) -> str:
    key_upper = patient_key.upper()
    if key_upper.startswith("CSF") or key_upper.endswith("_CSF"):
        return "CSF"
    if key_upper.startswith("PB") or key_upper.endswith("_PB") or "PBMC" in key_upper:
        return "PB"
    return "unknown"



def _normalise_tissue_labels(adata: ad.AnnData) -> ad.AnnData:
    """
    Harmonise tissue labels across datasets.
    PBMCs and PBMC are treated as PB.
    """
    if "tissue" not in adata.obs.columns:
        return adata

    adata.obs["tissue"] = (
        adata.obs["tissue"]
        .astype(str)
        .replace({
            "PBMCs": "PB",
            "PBMC": "PB",
            "blood": "PB",
            "peripheral blood": "PB",
        })
    )
    return adata

# ---------------------------------------------------------------------------
# Condition and cell-type filters
# ---------------------------------------------------------------------------

def filter_conditions(adata: ad.AnnData, ds_cfg: dict[str, Any]) -> ad.AnnData:
    """
    Retain only cells whose condition value is in keep_conditions.
    If keep_conditions is absent from the config, returns adata unchanged.
    """
    keep = ds_cfg.get("keep_conditions")
    if keep is None:
        return adata
    condition_col = ds_cfg.get("condition_col", "condition")
    if condition_col not in adata.obs.columns:
        print(f"  [warning] condition_col '{condition_col}' not found in .obs; skipping filter.")
        return adata
    mask = adata.obs[condition_col].isin(keep)
    return adata[mask].copy()

def gate_bcells(
    adata: ad.AnnData,
    b_markers: list[str],
    nonb_markers: list[str] | None = None,
    nonb_pct: float = 0.2,
) -> ad.AnnData:
    """
    Safer strict B-cell gate.

    1) Keep cells with any B marker expressed (count > 0).
    2) If nonb_markers is provided:
         - compute NonB-score (mean of non-B markers)
         - drop cells in the top (1 - nonb_pct) quantile of NonB-score.

    Parameters
    ----------
    nonb_pct:
      Quantile threshold for NonB-score; default 0.9 keeps the
      90% least non-B-like cells (very permissive initially).
    """
    import numpy as np
    import scipy.sparse as sp

    # --- Step 1: any B marker > 0
    present_b = [g for g in b_markers if g in adata.var_names]
    if not present_b:
        print(f"  [warning] No B markers present among {b_markers}; skipping B gate.")
        return adata

    idx_b = [adata.var_names.get_loc(g) for g in present_b]
    Xb = adata.X[:, idx_b]
    if sp.issparse(Xb):
        expr_b = np.asarray(Xb.sum(axis=1)).ravel()
    else:
        expr_b = Xb.sum(axis=1)
    keep_b = expr_b > 0

    print(f"  Step1: {keep_b.sum()} / {adata.n_obs} cells with any B marker > 0")

    adata_sub = adata[keep_b].copy()



    # --- Step 2: optional non-B filter
    if nonb_markers:
        present_nb = [g for g in nonb_markers if g in adata_sub.var_names]
        if present_nb:
            idx_nb = [adata_sub.var_names.get_loc(g) for g in present_nb]
            Xnb = adata_sub.X[:, idx_nb]
            if sp.issparse(Xnb):
                Xnb = Xnb.toarray()
            nonb_score = Xnb.mean(axis=1)
            thr = np.quantile(nonb_score, nonb_pct)
            keep_nb = nonb_score <= thr
            print(
                f"  Step2: removing {(~keep_nb).sum()} cells with high non-B score "
                f"(>= q{int(nonb_pct*100)})"
            )
            adata_sub = adata_sub[keep_nb].copy()
            adata_sub.obs["NonB_score"] = nonb_score[keep_nb]
        else:
            print(f"  [info] None of the non-B markers {nonb_markers} found; skipping Step2.")

    print(f"  Final B-gate: {adata_sub.n_obs} cells retained.")

    return adata_sub



def add_cxcr3_group(adata: ad.AnnData, cxcr3_gene: str = "CXCR3") -> ad.AnnData:
    """
    Add obs column 'cxcr3_group' with values 'CXCR3+' or 'CXCR3-'.
    """
    import numpy as np
    import scipy.sparse as sp

    if cxcr3_gene not in adata.var_names:
        print(f"  [warning] '{cxcr3_gene}' not in var_names; setting all cells to 'CXCR3-'.")
        adata.obs["cxcr3_group"] = "CXCR3-"
        return adata
    idx = adata.var_names.get_loc(cxcr3_gene)
    col = adata.X[:, idx]
    if sp.issparse(col):
        expr = np.asarray(col.todense()).ravel()
    else:
        expr = np.asarray(col).ravel()
    adata.obs["cxcr3_group"] = ["CXCR3+" if v > 0 else "CXCR3-" for v in expr]
    return adata


# ---------------------------------------------------------------------------
# Standardise .obs schema
# ---------------------------------------------------------------------------

def _standardise_obs(
    adata: ad.AnnData,
    dataset_name: str,
    ds_cfg: dict[str, Any],
    patient_key: str | None = None,
    patient_meta: dict[str, Any] | None = None,
) -> ad.AnnData:
    """
    Guarantee a common set of .obs columns across all datasets:
        dataset, patient, condition, tissue, sex
    """
    adata.obs["dataset"] = dataset_name

    condition_col = ds_cfg.get("condition_col", "condition")
    patient_col   = ds_cfg.get("patient_col",   "patient")
    tissue_col    = ds_cfg.get("tissue_col",    "tissue")
    print(patient_key)
    if patient_key is not None:
        adata.obs["patient"] = patient_key
    elif patient_col in adata.obs.columns:
        adata.obs["patient"] = adata.obs[patient_col].astype(str)
    else:
        adata.obs["patient"] = "unknown"

    if patient_meta and "condition" in patient_meta:
        adata.obs["condition"] = patient_meta["condition"]
    elif condition_col in adata.obs.columns:
        adata.obs["condition"] = adata.obs[condition_col].astype(str)
    else:
        adata.obs["condition"] = "unknown"

    # Sex is patient-level metadata from datasets.yaml only (no source .h5ad
    # column fallback -- neither dataset embeds sex in the raw object), added
    # for the sex-stratified sensitivity analysis (reviewer point #11).
    if patient_meta and "sex" in patient_meta:
        adata.obs["sex"] = patient_meta["sex"]
    else:
        adata.obs["sex"] = "unknown"

    if patient_key is not None:
        adata.obs["tissue"] = _tissue_from_key(patient_key)
    elif tissue_col in adata.obs.columns:
        adata.obs["tissue"] = adata.obs[tissue_col].astype(str)
    else:
        adata.obs["tissue"] = "unknown"

    return adata


# ---------------------------------------------------------------------------
# Dataset-level loader
# ---------------------------------------------------------------------------

def load_dataset(
    name: str,
    ds_cfg: dict[str, Any],
    base_dir: str | Path = "data/processed",
    cfg: dict[str, Any] | None = None,
    gate_b: bool = True,
    add_cxcr3: bool = True,
) -> ad.AnnData:
    """
    Load a single dataset entry from the config.

    - If 'patients' is present: loops over patient files, concatenates.
    - Otherwise: loads the single global .h5ad.
    Both paths produce the same standardised .obs schema.
    """
    markers_cfg   = (cfg or {}).get("markers", {})
    gating_cfg    = (cfg or {}).get("gating", {})

    # Base B markers from dataset-level config, extended with global gating markers
    ds_b_markers  = ds_cfg.get("bcell_markers", ["CD19", "MS4A1", "CD79A", "CD79B"])
    global_b      = gating_cfg.get("bcell_markers", [])
    bcell_markers = list(dict.fromkeys(ds_b_markers + global_b))  # dedupe, keep order

    nonb_markers  = gating_cfg.get("nonb_markers", [])

    cxcr3_gene    = markers_cfg.get("cxcr3", "CXCR3")

    patients = ds_cfg.get("patients")

    if patients:
        adatas: list[ad.AnnData] = []
        for patient_key, patient_meta in patients.items():
            path = build_patient_path(name, patient_key, base_dir)
            if not path.exists():
                print(f" [skip] {path} not found.")
                continue
            print(f"  Loading {path} ...")
            adata = sc.read_h5ad(path)

            # Ensure unique obs_names at source
            if not adata.obs_names.is_unique:
                print(f"  [fix] Making obs_names unique for {path.name}")
                adata.obs_names_make_unique()

            adata = _standardise_obs(adata, name, ds_cfg, patient_key, patient_meta)
            adatas.append(adata)

        if not adatas:
            raise FileNotFoundError(f"No patient files loaded for dataset '{name}'.")
        combined = ad.concat(adatas, label="patient_file", join="outer")
        combined = filter_conditions(combined, ds_cfg)

    else:
        # Global file mode
        path = _resolve_h5ad_path(name, ds_cfg, base_dir)
        if not path.exists():
            raise FileNotFoundError(f"Global h5ad not found: {path}")
        print(f"  Loading {path} ...")
        combined = sc.read_h5ad(path)


        if name == "GSE239626":
            print("  [diag] obs columns:", combined.obs.columns.tolist())
            print("  [diag] head of obs:")
            print(combined.obs.head())
        if not combined.obs_names.is_unique:
            print(f"  [fix] Making obs_names unique for {path.name}")
            combined.obs_names_make_unique()

        # Standardise obs first
        combined = _standardise_obs(combined, name, ds_cfg)


        # Now apply keep_conditions
        combined = filter_conditions(combined, ds_cfg)

    

    if gate_b:
        combined = gate_bcells(
            combined,
            b_markers=bcell_markers,
            nonb_markers=nonb_markers,
            nonb_pct=0.7,   #  permissive (high); tighten later (low)
        )

    if add_cxcr3:
        combined = add_cxcr3_group(combined, cxcr3_gene)

    
    print(f"  {name}: {combined.n_obs} cells retained.")
    return combined


# ---------------------------------------------------------------------------
# Multi-dataset loader
# ---------------------------------------------------------------------------

def load_all_datasets(
    cfg: dict[str, Any],
    base_dir: str | Path = "data/processed",
    gate_b: bool = True,
    add_cxcr3: bool = True,
) -> ad.AnnData:
    """
    Loop over all datasets in cfg, load each, and return one merged AnnData.
    Stable .obs columns: dataset, patient, condition, tissue, sex, cxcr3_group.
    """
    datasets_cfg = cfg.get("datasets", {})
    adatas: list[ad.AnnData] = []

    for name, ds_cfg in datasets_cfg.items():
        print(f"\n[{name}]")
        try:
            adata = load_dataset(
                name=name,
                ds_cfg=ds_cfg,
                base_dir=base_dir,
                cfg=cfg,
                gate_b=gate_b,
                add_cxcr3=add_cxcr3,
            )
            adatas.append(adata)
        except FileNotFoundError as exc:
            print(f"  [error] {exc} -- dataset skipped.")

    if not adatas:
        raise RuntimeError("No datasets were loaded successfully.")

    merged = ad.concat(adatas, label="source_dataset", join="outer")
    merged = _normalise_tissue_labels(merged)
    print(f"\nTotal cells after merge: {merged.n_obs}")
    return merged