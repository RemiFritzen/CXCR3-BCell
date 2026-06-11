#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:00:47 2026

@author: remi
"""

import scanpy as sc
import numpy as np
from config.config import RESULTS_DIR
from data_io import load_cfg
from config.config import DATASETS_YAML

# Load merged object
adata = sc.read_h5ad(RESULTS_DIR / "merged_bcells.h5ad")

# Load zinc/MT markers from config
cfg = load_cfg(DATASETS_YAML)
markers = cfg.get("markers", {})
zinc_genes = markers.get("zinc_transporters", [])
mt_genes   = markers.get("metallothioneins", [])
zinc_mt_genes = zinc_genes + mt_genes

# Subset to GSE133028 PB CXCR3+ B cells
pb = adata[
    (adata.obs["tissue"] == "PB") &
    (adata.obs["cxcr3_group"] == "CXCR3+")
].copy()

def gene_stats_raw(a, gene):
    if a.raw is None or gene not in a.raw.var_names:
        return None
    X = a.raw[:, [gene]].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    vals = X[:, 0]
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "pct_expr": float((vals > 0).mean() * 100),
    }

results = {}
for g in zinc_mt_genes:
    stats = gene_stats_raw(pb, g)
    if stats is not None:
        results[g] = stats

for g, s in results.items():
    print(
        f"{g:8s}  mean={s['mean']:.4f}  median={s['median']:.4f}  "
        f"%>0={s['pct_expr']:.1f}%"
    )

def subset_and_stats(tissue, cxcr3_group):
    sub = adata[
        (adata.obs["tissue"] == tissue) &
        (adata.obs["cxcr3_group"] == cxcr3_group)
    ].copy()

    print(f"\n {tissue} {cxcr3_group}: {sub.n_obs} cells")

    def gene_stats_raw(a, gene):
        if a.raw is None or gene not in a.raw.var_names:
            return None
        X = a.raw[:, [gene]].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        vals = X[:, 0]
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "pct_expr": float((vals > 0).mean() * 100),
        }

    results = {}
    for g in zinc_mt_genes:
        stats = gene_stats_raw(sub, g)
        if stats is not None:
            results[g] = stats

    for g, s in results.items():
        print(
            f"{g:8s}  mean={s['mean']:.4f}  "
            f"median={s['median']:.4f}  %>0={s['pct_expr']:.1f}%"
        )
    return results

pb_stats  = subset_and_stats( "PB",  "CXCR3+")
csf_stats = subset_and_stats( "CSF", "CXCR3+")