#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
figS1_validate_scarches_integrated.py
====================================
Create Supplementary Figure S1 and optional external-atlas validation of
expression-based B-cell subtype annotations within the existing CXCR3 project
filesystem.

What this script does
---------------------
1. Loads the pooled B-cell query object from the project H5AD registry
2. Creates a first-pass manual B-cell subtype annotation from marker expression
3. Computes / refreshes a UMAP for the manual subtype panel
4. Exports a Nygen cross-check package (.h5ad + metadata .csv)
5. Optionally validates against an external reference atlas using
   scVI / scANVI + scArches-style query mapping if the reference file exists

Designed for the project config:
    from config.config import H5AD, RESULTS_DIR, FIG_DIRS

Recommended reference
---------------------
Immune Health Atlas B / plasma reference:
    human_immune_health_atlas_b-plasma.h5ad
Place it at:
    RESULTS_DIR / "reference_bcells.h5ad"

Default reference metadata keys assumed here:
    REF_LABEL_KEY = "AIFI_L3"
    REF_BATCH_KEY = "batch_id"
If your downloaded file differs, inspect ref.obs.columns and edit below.
"""



import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import numpy as np
from scipy import sparse

from config.config import RESULTS_DIR, FIG_DIRS

QUERY_H5AD = RESULTS_DIR / "merged_bcells.h5ad"
REFERENCE_H5AD = RESULTS_DIR / "reference_bcells.h5ad"
REF_LABEL_KEY = "AIFI_L3"
REF_BATCH_KEY = "batch_id"

VALIDATION_DIR = RESULTS_DIR / "scarches_validation"
NYGEN_DIR = RESULTS_DIR / "nygen_export"
SUPP_DIR = FIG_DIRS["supp"]

MANUAL_UMAP_PNG = SUPP_DIR / "figS1_manual_bcell_subtypes.png"
REF_UMAP_PNG = SUPP_DIR / "figS1_reference_predicted_subtypes.png"
SCORE_UMAP_PNG = SUPP_DIR / "figS1_prediction_score.png"

MANUAL_QUERY_OUT = VALIDATION_DIR / "query_manual_bcell_subtypes.h5ad"
REF_QUERY_OUT = VALIDATION_DIR / "query_annotated_scarches.h5ad"
CONFUSION_OUT = VALIDATION_DIR / "subtype_validation_confusion.csv"
PROBS_OUT = VALIDATION_DIR / "subtype_prediction_scores.csv"
COUNTS_OUT = VALIDATION_DIR / "manual_subtype_counts.csv"
SUMMARY_OUT = VALIDATION_DIR / "validation_summary.csv"

NYGEN_H5AD_OUT = NYGEN_DIR / "ALL_Bcells_for_nygen.h5ad"
NYGEN_META_OUT = NYGEN_DIR / "ALL_Bcells_for_nygen_metadata.csv"


def _safe_expr(df: pd.DataFrame, gene: str) -> pd.Series:
    if gene in df.columns:
        return df[gene]
    return pd.Series(0.0, index=df.index)


def annotate_bcell_subtypes(adata: sc.AnnData, outcol: str = "manual_bcell_subtype") -> sc.AnnData:
    expr = adata.to_df()
    subtype = pd.Series("Other B", index=adata.obs_names, dtype="object")

    pb_mask = (
        (_safe_expr(expr, "MZB1") > 1.0)
        | (_safe_expr(expr, "XBP1") > 1.0)
        | (_safe_expr(expr, "SDC1") > 0.5)
        | (_safe_expr(expr, "PRDM1") > 0.5)
    )

    gc_mask = (
        (_safe_expr(expr, "BCL6") > 0.8)
        | (_safe_expr(expr, "AICDA") > 0.3)
    ) & ~pb_mask

    mem_mask = (
        (_safe_expr(expr, "CD27") > 0.5)
        | (_safe_expr(expr, "TNFRSF13B") > 0.5)
        | (_safe_expr(expr, "BANK1") > 0.8)
        | (_safe_expr(expr, "AIM2") > 0.3)
    ) & ~pb_mask & ~gc_mask

    naive_mask = (
        (_safe_expr(expr, "IGHD") > 0.5)
        | (_safe_expr(expr, "TCL1A") > 0.5)
        | (_safe_expr(expr, "FCER2") > 0.5)
        | (_safe_expr(expr, "IL4R") > 0.3)
    ) & ~pb_mask & ~gc_mask & ~mem_mask

    activated_ifn_mask = (
        (_safe_expr(expr, "CXCR3") > 0.3)
        | (_safe_expr(expr, "IFIT3") > 0.5)
        | (_safe_expr(expr, "ISG15") > 0.5)
        | (_safe_expr(expr, "IFI6") > 0.5)
        | (_safe_expr(expr, "MX1") > 0.5)
    ) & ~pb_mask & ~gc_mask

    subtype[pb_mask] = "Plasmablast/ASC"
    subtype[gc_mask] = "GC-like B"
    subtype[mem_mask] = "Memory B"
    subtype[naive_mask] = "Naive B"
    subtype[activated_ifn_mask & (subtype == "Other B")] = "Activated/IFN B"

    adata.obs[outcol] = pd.Categorical(subtype)
    return adata


def ensure_umap(adata: sc.AnnData, preferred_rep: str | None = None) -> sc.AnnData:
    if "X_umap" in adata.obsm:
        return adata

    if preferred_rep and preferred_rep in adata.obsm:
        sc.pp.neighbors(adata, use_rep=preferred_rep)
    elif "X_pca" in adata.obsm:
        sc.pp.neighbors(adata, use_rep="X_pca")
    else:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.umap(adata)
    return adata


def save_umap(adata: sc.AnnData, color: str, outpath: Path, title: str) -> None:
    categorical = color in adata.obs.columns and str(adata.obs[color].dtype) == "category"
    fig = sc.pl.umap(
        adata,
        color=color,
        show=False,
        return_fig=True,
        title=title,
        # legend_loc="on data" if categorical else "right margin",
        legend_loc="right margin",
        legend_fontsize=10,
        legend_fontoutline=2,
        frameon=False,
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_for_nygen(adata: sc.AnnData) -> None:
    NYGEN_DIR.mkdir(parents=True, exist_ok=True)

    # Use raw counts if available
    if adata.raw is not None:
        adata_to_write = adata.raw.to_adata()
        # carry over relevant obs columns
        adata_to_write.obs = adata.obs.copy()
    else:
        adata_to_write = adata

    adata_to_write.write(NYGEN_H5AD_OUT)

    cols = [
        c for c in [
            "dataset", "patient", "sample", "donor", "tissue", "condition",
            "cxcr3_group", "manual_bcell_subtype", "predicted_subtype",
            "predicted_subtype_score"
        ] if c in adata_to_write.obs.columns
    ]
    adata_to_write.obs[cols].to_csv(NYGEN_META_OUT)


def run_reference_mapping(query: sc.AnnData):
    if not REFERENCE_H5AD.exists():
        warnings.warn(
            f"Reference file not found at {REFERENCE_H5AD}. "
            "Skipping external validation and keeping manual annotations only."
        )
        return None

    try:
        import scvi
    except Exception as e:
        warnings.warn(f"scvi-tools import failed ({e}). Skipping external validation.")
        return None

    ref = sc.read_h5ad(REFERENCE_H5AD)
    ref.var_names_make_unique()

    # Use raw counts
    if ref.raw is not None:
        ref = ref.raw.to_adata()

    query = query.copy()
    query.var_names_make_unique()

    common = ref.var_names.intersection(query.var_names)
    if len(common) < 500:
        warnings.warn(
            f"Too few shared genes between reference and query ({len(common)}). "
            "Skipping external validation."
        )
        return None


    ref = ref[:, common].copy()
    query = query[:, common].copy()

    # Drop cells with any NaN in X from both ref and query
    def _drop_nan_cells(adata: sc.AnnData) -> sc.AnnData:
        X = adata.X
        if sparse.issparse(X):
            X = X.toarray()
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > 0:
            print(f"Dropping {nan_mask.sum()} cells with NaNs")
            adata = adata[~nan_mask].copy()
        return adata

    ref = _drop_nan_cells(ref)
    query = _drop_nan_cells(query)

    print("NaNs in ref.X after filtering:", np.isnan(ref.X.toarray() if sparse.issparse(ref.X) else ref.X).sum())
    print("NaNs in query.X after filtering:", np.isnan(query.X.toarray() if sparse.issparse(query.X) else query.X).sum())

    scvi.model.SCVI.setup_anndata(ref, batch_key=REF_BATCH_KEY)
    vae = scvi.model.SCVI(ref)
    vae.train(max_epochs=100)

    scanvi = scvi.model.SCANVI.from_scvi_model(
        vae,
        adata=ref,
        labels_key=REF_LABEL_KEY,
        unlabeled_category="Unknown",
    )
    scanvi.train(max_epochs=20)

    # Optional: save, but DO NOT pass the path into prepare/load
    model_dir = VALIDATION_DIR / "scanvi_reference_model"
    scanvi.save(model_dir, overwrite=True)
        
    
    scvi.model.SCANVI.prepare_query_anndata(query, scanvi)
    
    Xq = query.X.toarray() if sparse.issparse(query.X) else query.X
    print("query.X min/max before SCANVI:", Xq.min(), Xq.max())
    print("NaNs in query.X before SCANVI:", np.isnan(Xq).sum())
    
    
    # Defensive: clean any NaNs created during setup
    Xq = query.X
    if sparse.issparse(Xq):
        Xq = Xq.toarray()
    nan_mask = np.isnan(Xq).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"[query] Dropping {nan_mask.sum()} cells with NaNs after prepare_query_anndata")
        query = query[~nan_mask].copy()

    # Optional: replace any remaining NaNs with zeros as an extra guard
    Xq = query.X
    if sparse.issparse(Xq):
        Xq = Xq.toarray()
    Xq = np.nan_to_num(Xq, nan=0.0, posinf=0.0, neginf=0.0)
    if sparse.issparse(query.X):
        query.X = sparse.csr_matrix(Xq)
    else:
        query.X = Xq

    # Now build query model
    query_model = scvi.model.SCANVI.load_query_data(query, scanvi)
    query_model.train(max_epochs=20, plan_kwargs={"weight_decay": 0.0})
    

    query.obs["predicted_subtype"] = query_model.predict()
    probs = query_model.predict(soft=True)

    probs_df = pd.DataFrame(probs, index=query.obs_names)
    query.obs["predicted_subtype_score"] = probs_df.max(axis=1)

    query.obsm["X_scANVI"] = query_model.get_latent_representation()
    sc.pp.neighbors(query, use_rep="X_scANVI")
    sc.tl.umap(query)

    confusion = pd.crosstab(
        query.obs["manual_bcell_subtype"],
        query.obs["predicted_subtype"],
    )

    summary = pd.DataFrame(
        {
            "n_query_cells": [query.n_obs],
            "n_reference_cells": [ref.n_obs],
            "n_shared_genes": [len(common)],
            "mean_prediction_score": [
                float(query.obs["predicted_subtype_score"].mean())
            ],
            "median_prediction_score": [
                float(query.obs["predicted_subtype_score"].median())
            ],
        }
    )

    return query, probs_df, confusion, summary


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    NYGEN_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_DIR.mkdir(parents=True, exist_ok=True)

    if not QUERY_H5AD.exists():
        raise FileNotFoundError(f"Query h5ad not found: {QUERY_H5AD}")

    # Load merged B cells
    adata = sc.read_h5ad(QUERY_H5AD)
    adata.var_names_make_unique()

    # Use raw counts if available
    if adata.raw is not None:
        adata_counts = adata.raw.to_adata()
        adata_counts.obs = adata.obs.copy()
        adata = adata_counts

    # Manual subtype annotation and UMAP on this count-based object
    annotate_bcell_subtypes(adata, outcol="manual_bcell_subtype")
    ensure_umap(adata)
    save_umap(adata, "manual_bcell_subtype", MANUAL_UMAP_PNG, "B-cell subtypes (marker-based)")
    adata.write(MANUAL_QUERY_OUT)
    adata.obs["manual_bcell_subtype"].value_counts().rename_axis("manual_bcell_subtype").to_csv(COUNTS_OUT)

    # Batch for SCVI / SCANVI
    if "batch_id" in adata.obs.columns:
        adata.obs["batch_id"] = adata.obs["batch_id"].astype(str)
    elif "dataset" in adata.obs.columns:
        adata.obs["batch_id"] = adata.obs["dataset"].astype(str)
    elif "patient" in adata.obs.columns:
        adata.obs["batch_id"] = adata.obs["patient"].astype(str)
    else:
        adata.obs["batch_id"] = "batch0"

    # Reference mapping
    mapping_result = run_reference_mapping(adata)
    
    
    
    def export_tissue_figures(mapped: sc.AnnData) -> None:
    # Expect a 'tissue' column with values like 'PB', 'CSF'
        if "tissue" not in mapped.obs.columns:
            print("No 'tissue' column in obs; skipping per-tissue PDFs.")
            return
    
        for tissue in sorted(mapped.obs["tissue"].unique()):
            sub = mapped[mapped.obs["tissue"] == tissue].copy()
            if sub.n_obs == 0:
                continue
    
            print(f"Exporting per-tissue figure for {tissue} ({sub.n_obs} cells)")
    
            fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
            # 1) Manual subtypes
            sc.pl.umap(
                sub,
                color="manual_bcell_subtype",
                ax=axes[0],
                show=False,
                title=f"{tissue} – manual subtypes",
                legend_loc="right margin",
                frameon=False,
            )
    
            # 2) Reference-transferred subtypes
            if "predicted_subtype" in sub.obs.columns:
                sc.pl.umap(
                    sub,
                    color="predicted_subtype",
                    ax=axes[1],
                    show=False,
                    title=f"{tissue} – reference subtypes",
                    legend_loc="right margin",
                    frameon=False,
                )
            else:
                axes[1].set_axis_off()
                axes[1].set_title(f"{tissue} – no reference mapping")
    
            # 3) Prediction confidence
            if "predicted_subtype_score" in sub.obs.columns:
                sc.pl.umap(
                    sub,
                    color="predicted_subtype_score",
                    ax=axes[2],
                    show=False,
                    title=f"{tissue} – prediction confidence",
                    frameon=False,
                    colorbar_loc="right",
                )
            else:
                axes[2].set_axis_off()
                axes[2].set_title(f"{tissue} – no scores")
    
            fig.tight_layout()
    
            out_pdf = SUPP_DIR / f"figS1_{tissue}_panel.pdf"
            fig.savefig(out_pdf, bbox_inches="tight")
            plt.close(fig)
            print("Saved per-tissue figure ->", out_pdf)

    

    if mapping_result is not None:
        mapped_query, probs_df, confusion, summary = mapping_result
        save_umap(mapped_query, "predicted_subtype", REF_UMAP_PNG, "Reference-transferred B-cell subtypes")
        save_umap(mapped_query, "predicted_subtype_score", SCORE_UMAP_PNG, "Reference prediction confidence")
        probs_df.to_csv(PROBS_OUT)
        confusion.to_csv(CONFUSION_OUT)
        summary.to_csv(SUMMARY_OUT, index=False)
        mapped_query.write(REF_QUERY_OUT)
        export_for_nygen(mapped_query)
        export_tissue_figures(mapped_query)
    else:
        export_for_nygen(adata)

        summary = pd.DataFrame(
            {
                "n_query_cells": [adata.n_obs],
                "n_reference_cells": [0],
                "n_shared_genes": [0],
                "mean_prediction_score": [None],
                "median_prediction_score": [None],
                "reference_mapping_status": ["skipped"],
            }
        )
        summary.to_csv(SUMMARY_OUT, index=False)

    print("Saved:")
    print(MANUAL_UMAP_PNG)
    if REF_UMAP_PNG.exists():
        print(REF_UMAP_PNG)
    if SCORE_UMAP_PNG.exists():
        print(SCORE_UMAP_PNG)
    print(COUNTS_OUT)
    print(SUMMARY_OUT)
    print(NYGEN_H5AD_OUT)
    print(NYGEN_META_OUT)


if __name__ == "__main__":
    main()
