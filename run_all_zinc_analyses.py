
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Thu May 21 15:55:12 2026

@author: remi
run_all_zinc_analyses.py
------------------------
Entry point for the B-cell zinc / CXCR3 pipeline.
Loads all datasets, merges them, and runs the zinc gene differential expression
analysis for each dataset x tissue combination.
"""

# from pathlib import Path

import scanpy as sc
import numpy as np  # put this near the top of the file with other imports
import harmonypy as hm

from config.config import DATASETS_YAML, DATA_DIR, RESULTS_DIR
from data_io import load_cfg, load_all_datasets


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SC_SEED   = 42
N_HVGS    = 2000
N_PCS     = 30
N_NEIGHBS = 15


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # 1. Load config and all datasets
    cfg   = load_cfg(DATASETS_YAML)
    adata = load_all_datasets(cfg, base_dir=DATA_DIR, gate_b=True, add_cxcr3=True)
    if not adata.obs_names.is_unique:
        print(" [fix] Making obs_names unique in merged object")
        adata.obs_names_make_unique()

    # ── Preprocessing + Harmony UMAP (single block) ───────────────────────

    # Normalise / log (on full gene set)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Preserve log-normalised expression BEFORE HVG filtering and scaling
    adata.raw = adata.copy()
    
    
    # HVGs (NO batch_key)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_HVGS,
        flavor="seurat",
    )
    
    # ------------------------------------------------------------------
    # Force-keep zinc / metallothionein genes in the HVG mask
    # ------------------------------------------------------------------
    markers_cfg       = cfg.get("markers", {})
    zinc_transporters = markers_cfg.get("zinc_transporters", [])
    metallothioneins  = markers_cfg.get("metallothioneins", [])
    zinc_genes        = zinc_transporters + metallothioneins    
    artefact          = set(markers_cfg.get("exclude_artefacts", []))
    
    zinc_genes_all = zinc_transporters + metallothioneins
    zinc_genes     = [g for g in zinc_genes_all if g not in artefact]
    
    present_zinc = [g for g in zinc_genes if g in adata.var_names]
    print(f"\nZinc genes present in merged object: {len(present_zinc)} / {len(zinc_genes)}")
        
    hvg_mask = adata.var["highly_variable"].copy()
    
    for g in present_zinc:
        idx = adata.var_names.get_loc(g)
        hvg_mask.iloc[idx] = True  # ensure zinc genes are kept
    
    adata = adata[:, hvg_mask].copy()
    # ------------------------------------------------------------------

    
    # Scale / PCA
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=N_PCS, random_state=SC_SEED)
    
    # Harmony
    X_pca = adata.obsm["X_pca"]
    meta  = adata.obs
    
    ho = hm.run_harmony(X_pca, meta, "dataset")
    print("Harmony Z_corr shape:", ho.Z_corr.shape)
    
    adata.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr)
    
    sc.pp.neighbors(
        adata,
        use_rep="X_pca_harmony",
        n_neighbors=N_NEIGHBS,
        n_pcs=N_PCS,
        random_state=SC_SEED,
    )
    sc.tl.umap(adata, random_state=SC_SEED)


    print(f"Merged object: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Datasets : {adata.obs['dataset'].unique().tolist()}")
    print(f"Conditions: {adata.obs['condition'].unique().tolist()}")
    print(f"Tissues   : {adata.obs['tissue'].unique().tolist()}")

    # 3. Save merged object
    merged_path = RESULTS_DIR / "merged_bcells.h5ad"
    adata.write_h5ad(merged_path)
    print(f"Saved merged object -> {merged_path}")

    # 4. Per-dataset zinc DE analysis (CXCR3+ vs CXCR3-)
    present_zinc = [g for g in zinc_genes if g in adata.var_names]
    print(f"\nZinc genes present in  {len(present_zinc)} / {len(zinc_genes)}")

    for ds_name in adata.obs["dataset"].unique():
        ds_mask = adata.obs["dataset"] == ds_name
        ds      = adata[ds_mask]

        for tissue in ds.obs["tissue"].unique():
            t_mask = ds.obs["tissue"] == tissue
            sub    = ds[t_mask]

            groups = sub.obs["cxcr3_group"].unique().tolist()
            if "CXCR3+" not in groups or "CXCR3-" not in groups:
                print(f"  [skip] {ds_name} / {tissue}: missing CXCR3+/- groups")
                continue

            n_pos = (sub.obs["cxcr3_group"] == "CXCR3+").sum()
            n_neg = (sub.obs["cxcr3_group"] == "CXCR3-").sum()
            print(f"  {ds_name} / {tissue}: {n_pos} CXCR3+  {n_neg} CXCR3-")

            sc.tl.rank_genes_groups(
                sub,
                groupby="cxcr3_group",
                groups=["CXCR3+"],
                reference="CXCR3-",
                method="wilcoxon",
                key_added="zinc_de",
            )

            result = sc.get.rank_genes_groups_df(sub, group="CXCR3+", key="zinc_de")
            result = result[result["names"].isin(present_zinc)].copy()
            
            if result.empty:
                print(f"    [skip] {ds_name} / {tissue}: no zinc/metallothionein genes in DE result")
                continue
            
            out_path = RESULTS_DIR / f"{ds_name}_{tissue}_zinc_results.csv"
            result.to_csv(out_path, index=False)
            print(f"    Saved -> {out_path}")


   

    print("\nDone.")


if __name__ == "__main__":
    main()
    
