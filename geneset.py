#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screen zinc genes for dataset-independent markers.

- Computes per-patient, per-tissue means for zinc genes.
- Identifies genes with small baseline differences between GSE133028 and GSE138266.
- Among those, ranks genes by MS > HC median difference.
"""

import warnings
import numpy as np
import pandas as pd
import scanpy as sc

from data_io import load_cfg
from config.config import RESULTS_DIR, DATASETS_YAML
from output.figures.fig4_patient_level.Fig4 import (
    get_expression_matrix,
    load_markers,
)

# ---------------------------------------------------------------------
# Load merged_bcells and zinc genes
# ---------------------------------------------------------------------

cfg = load_cfg(DATASETS_YAML)
genes = load_markers(cfg)

adata = sc.read_h5ad(RESULTS_DIR / "merged_bcells.h5ad")
adata = adata[adata.obs["dataset"].isin({"GSE133028", "GSE138266"})].copy()

sub = adata[
    (adata.obs["tissue"].isin(["PB", "CSF"])) &
    (adata.obs["condition"].isin(["HC", "MS", "CIS", "RRMS"]))
].copy()

print(f"Subset: {sub.n_obs} cells, {sub.n_vars} genes")

# ---------------------------------------------------------------------
# Per-patient mean expression for zinc genes
# ---------------------------------------------------------------------

expr = get_expression_matrix(sub, genes)
obs = sub.obs[["patient", "tissue", "dataset", "condition"]].copy()
obs["patient"] = obs["patient"].astype(str)

df = obs.join(expr)

patient_means = (
    df.groupby(["dataset", "patient", "tissue", "condition"])[genes]
      .mean()
      .reset_index()
)

# ---------------------------------------------------------------------
# Long format
# ---------------------------------------------------------------------

long = patient_means.melt(
    id_vars=["dataset", "patient", "tissue", "condition"],
    value_vars=genes,
    var_name="gene",
    value_name="expr",
)

# ---------------------------------------------------------------------
# Dataset-level stats per gene/tissue
# ---------------------------------------------------------------------

ds_stats = (
    long.groupby(["gene", "tissue", "dataset"])["expr"]
        .agg(["mean", "std", "count"])
        .reset_index()
)

pivot = ds_stats.pivot(index=["gene", "tissue"], columns="dataset", values="mean")

# Drop genes missing in one dataset
pivot = pivot.dropna(subset=["GSE133028", "GSE138266"])

pivot["delta_mean"] = pivot["GSE133028"] - pivot["GSE138266"]
stable = pivot.reset_index()

# Filter for small dataset shift
stable = stable[np.abs(stable["delta_mean"]) < 0.05]

print(f"Stable gene/tissue combinations (|delta_mean| < 0.05): {len(stable)}")

# ---------------------------------------------------------------------
# HC vs MS difference (pooled)
# ---------------------------------------------------------------------

hm = (
    long[long["condition"].isin(["HC", "RRMS", "MS"])]
    .assign(diag_group=lambda d: d["condition"].replace({"RRMS": "MS"}))
)

hm_stats = (
    hm.groupby(["gene", "tissue", "diag_group"])["expr"]
      .median()
      .reset_index()
      .pivot(index=["gene", "tissue"], columns="diag_group", values="expr")
)

# Keep only rows with both HC and MS medians
hm_stats = hm_stats.dropna(subset=["HC", "MS"])

hm_stats["MS_minus_HC"] = hm_stats["MS"] - hm_stats["HC"]

# ---------------------------------------------------------------------
# Join on ['gene','tissue'] and rank candidates
# ---------------------------------------------------------------------

# Bring hm_stats into column form with gene/tissue as columns
hm_stats_reset = hm_stats.reset_index()[["gene", "tissue", "MS_minus_HC"]]

# Merge explicitly on gene + tissue
candidates = (
    stable.merge(
        hm_stats_reset,
        on=["gene", "tissue"],
        how="inner",
    )
    .sort_values("MS_minus_HC", ascending=False)
)

print("\nTop candidate zinc markers (dataset-stable, MS > HC):")
print(candidates.head(20))
