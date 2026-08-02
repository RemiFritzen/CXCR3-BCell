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

CHANGE (patient-level statistics):
The per-cell Wilcoxon rank-sum test (sc.tl.rank_genes_groups) below has been
replaced with `paired_pseudobulk_de`. CXCR3+ and CXCR3- cells come from the
SAME patients, so this is a within-patient (paired) comparison -- treating
every cell as an independent observation is pseudoreplication and inflates
significance. Aggregating to one value per (patient, CXCR3 group) and running
a Wilcoxon signed-rank test on those ~n_patients pairs gives an honest p-value.
See stats_utils.py for the full rationale and for the unpaired version
(pseudobulk_de) to use for MS-vs-HC type comparisons, where each patient
belongs to only one group.
"""

# from pathlib import Path

import scanpy as sc
import numpy as np  # put this near the top of the file with other imports
import pandas as pd
import harmonypy as hm
import matplotlib.pyplot as plt
import scipy.sparse as sp

from config.config import DATASETS_YAML, DATA_DIR, RESULTS_DIR
from data_io import load_cfg, load_all_datasets
from stats_utils import (
    paired_pseudobulk_de, paired_pseudobulk_signature_test, compute_gene_signs,
    resolve_pb_csf_subject_ids,
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SC_SEED   = 42
N_HVGS    = 2000
N_PCS     = 30
N_NEIGHBS = 15
CXCR3_GENE = "CXCR3"
CXCR3_THRESHOLD = 0  # matches data_io.add_cxcr3_group: CXCR3+ if raw count > this


# ---------------------------------------------------------------------------
# CXCR3+/- threshold rationale (addresses reviewer request: justify the
# binary presence/absence gate rather than asserting it without evidence)
# ---------------------------------------------------------------------------

def plot_cxcr3_threshold_rationale(adata, gene: str = CXCR3_GENE, out_dir=None) -> None:
    """
    Diagnostic figure justifying the CXCR3+/- gate used throughout this
    pipeline (data_io.add_cxcr3_group: CXCR3+ if raw count > 0).

    IMPORTANT: this must be called on RAW counts, before normalize_total/
    log1p -- add_cxcr3_group is applied to raw counts during
    load_all_datasets (i.e. before any normalization happens in this
    script), so a histogram built on normalized/scaled values would not
    actually represent the space the threshold is drawn in.

    Two panels:
      A. Distribution of raw CXCR3 UMI counts across all B cells (log
         y-axis, since surface-receptor transcripts like CXCR3 are heavily
         zero-inflated by dropout). Annotated with the %CXCR3- / %CXCR3+
         split at the actual threshold used (count > 0).
      B. Among cells with any detected CXCR3 (count > 0), the distribution
         of log1p-normalized expression -- shown to check whether there is
         evidence for a more refined threshold (e.g. a second mode) or
         whether, given how sparse detection already is, presence/absence
         is the most defensible gate the data supports.
    """
    if gene not in adata.var_names:
        print(f"  [warn] '{gene}' not in var_names -- cannot plot threshold rationale.")
        return

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    raw_expr = adata[:, gene].X
    if sp.issparse(raw_expr):
        raw_expr = raw_expr.toarray()
    raw_expr = np.asarray(raw_expr).ravel()

    n_cells = len(raw_expr)
    n_pos = int((raw_expr > CXCR3_THRESHOLD).sum())
    n_neg = n_cells - n_pos
    pct_pos = 100 * n_pos / n_cells if n_cells else float("nan")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: raw count distribution, log y-axis, capped bins for readability
    cap = 5
    counts_capped = np.minimum(raw_expr.astype(int), cap)
    bin_edges = np.arange(-0.5, cap + 1.5, 1)
    axA.hist(counts_capped, bins=bin_edges, color="#4C78A8", edgecolor="black", linewidth=0.6)
    axA.set_yscale("log")
    xticks = list(range(cap + 1))
    axA.set_xticks(xticks)
    axA.set_xticklabels([str(i) for i in range(cap)] + [f"{cap}+"])
    axA.axvline(CXCR3_THRESHOLD + 0.5, color="red", linestyle="--", linewidth=1.2,
                label=f"threshold: count > {CXCR3_THRESHOLD}")
    axA.set_xlabel(f"{gene} raw UMI count (capped at {cap}+)")
    axA.set_ylabel("Number of cells (log scale)")
    axA.set_title(
        f"{gene} detection across all B cells (n={n_cells:,})\n"
        f"CXCR3-: {n_neg:,} ({100-pct_pos:.1f}%)   |   CXCR3+: {n_pos:,} ({pct_pos:.1f}%)",
        fontsize=10,
    )
    axA.legend(frameon=False, fontsize=8, loc="upper right")

    # Panel B: among detected cells, spread of log1p-normalized expression --
    # computed independently here (light local normalization) purely to show
    # whether "detected" cells split further, NOT used anywhere else in the
    # pipeline and not a re-derivation of the actual gating threshold.
    pos_mask = raw_expr > CXCR3_THRESHOLD
    if pos_mask.sum() > 1:
        pos_counts = raw_expr[pos_mask]
        pos_log = np.log1p(pos_counts)  # local, illustrative only
        axB.hist(pos_log, bins=30, color="#E45756", edgecolor="black", linewidth=0.6, alpha=0.85)
        axB.set_xlabel(f"log1p({gene} raw count), detected cells only")
        axB.set_ylabel("Number of cells")
        axB.set_title(
            f"Spread among CXCR3+ cells (n={int(pos_mask.sum()):,})\n"
            f"(checking for a natural second breakpoint beyond presence/absence)",
            fontsize=10,
        )
    else:
        axB.text(0.5, 0.5, "Too few CXCR3+ cells to show a distribution",
                  ha="center", va="center", transform=axB.transAxes)
        axB.set_title("Spread among CXCR3+ cells", fontsize=10)

    fig.suptitle(
        f"Rationale for CXCR3+/- gate: raw count > {CXCR3_THRESHOLD} "
        f"(applied before normalization, per data_io.add_cxcr3_group)",
        fontsize=11, fontweight="bold", y=1.03,
    )
    fig.tight_layout()

    out_dir = out_dir or (RESULTS_DIR / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cxcr3_threshold_rationale.png"
    out_pdf = out_dir / "cxcr3_threshold_rationale.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"  CXCR3 threshold rationale: {n_pos:,}/{n_cells:,} cells ({pct_pos:.1f}%) "
          f"classified CXCR3+ at count > {CXCR3_THRESHOLD}")
    print(f"  Saved -> {out_png}")
    print(f"  Saved -> {out_pdf}")

    # Optional per-dataset/tissue breakdown, printed (not plotted) so
    # inconsistent detection rates across studies/tissues are visible --
    # relevant since GSE133028 and GSE138266 may differ in depth/protocol.
    for col in ("dataset", "tissue"):
        if col in adata.obs.columns:
            print(f"  CXCR3+ rate by {col}:")
            breakdown = (
                pd.DataFrame({col: adata.obs[col].values, "pos": raw_expr > CXCR3_THRESHOLD})
                .groupby(col)["pos"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "pct_CXCR3pos", "count": "n_cells"})
            )
            breakdown["pct_CXCR3pos"] = (breakdown["pct_CXCR3pos"] * 100).round(1)
            print(breakdown.to_string())


# ---------------------------------------------------------------------------
# Curated zinc/MT gene expression diagnostics (supports gene-exclusion
# decisions with checkable numbers, e.g. "MT1H excluded, not expressed")
# ---------------------------------------------------------------------------

UNEXPRESSED_PCT_THRESHOLD = 1.0  # % of cells; below this, flag as essentially undetected

def compute_zinc_gene_expression_diagnostics(
    adata,
    zinc_genes: list[str],
    out_dir=None,
) -> pd.DataFrame:
    """
    For each curated zinc/MT gene, compute expression prevalence (% of
    cells with raw count > 0) and mean raw count, both overall and
    stratified by (tissue, cxcr3_group) -- the CXCR3+ rows specifically
    match the population used throughout the DE pipeline (fig2.py/fig3.py),
    so a gene's prevalence THERE is what actually determines whether it
    carries any signal in the tests that use it.

    IMPORTANT: like plot_cxcr3_threshold_rationale, this must run on RAW
    counts (before normalize_total/log1p), since prevalence ("detected vs
    not") is most interpretable in that space -- call this before
    normalization in main(), which is where it's wired in.

    A gene showing near-zero prevalence in the CXCR3+ rows (below
    UNEXPRESSED_PCT_THRESHOLD) isn't "lowly expressed" in a meaningful
    sense -- it's essentially undetected, and no amount of relaxing a
    downstream prevalence filter changes that (the curated panel is
    force-tested regardless of any filter via `always_keep` in fig2.py/
    fig3.py -- this diagnostic exists to show WHY a gene might legitimately
    be excluded from the panel entirely, with numbers, rather than needing
    to be inferred after the fact from a degenerate all-zero DE result).

    Saves a CSV and prints a flagged-gene summary. Returns the full table.
    """
    present = [g for g in zinc_genes if g in adata.var_names]
    missing = [g for g in zinc_genes if g not in adata.var_names]
    if missing:
        print(f"  [warn] {len(missing)} curated zinc/MT gene(s) not in dataset at all: {missing}")

    X = adata[:, present].X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    obs = adata.obs
    rows = []
    for gi, gene in enumerate(present):
        vals = X[:, gi]
        rows.append({
            "gene": gene, "tissue": "ALL", "cxcr3_group": "ALL",
            "n_cells": len(vals),
            "pct_expressing": 100 * (vals > 0).mean(),
            "mean_count": vals.mean(),
        })
        for (tissue, cxcr3), idx in obs.groupby(["tissue", "cxcr3_group"], observed=True).groups.items():
            mask = obs.index.isin(idx)
            sub_vals = vals[mask]
            if len(sub_vals) == 0:
                continue
            rows.append({
                "gene": gene, "tissue": tissue, "cxcr3_group": cxcr3,
                "n_cells": len(sub_vals),
                "pct_expressing": 100 * (sub_vals > 0).mean(),
                "mean_count": sub_vals.mean(),
            })

    diag = pd.DataFrame(rows)

    out_dir = out_dir or (RESULTS_DIR / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zinc_gene_expression_diagnostics.csv"
    diag.to_csv(out_path, index=False)
    print(f"  Saved zinc/MT gene expression diagnostics -> {out_path}")

    # Flag genes essentially undetected in the CXCR3+ population specifically
    # -- this is the population the DE tests actually use, so prevalence
    # anywhere else (e.g. 'ALL' cells) doesn't tell you whether the gene
    # carries signal in the tests it's tested in.
    cxcr3_rows = diag[diag["cxcr3_group"] == "CXCR3+"]
    flagged = (
        cxcr3_rows.groupby("gene")["pct_expressing"].max()
        .loc[lambda s: s < UNEXPRESSED_PCT_THRESHOLD]
    )
    if len(flagged):
        print(f"  [FLAG] {len(flagged)} curated gene(s) essentially undetected in CXCR3+ cells "
              f"in EVERY tissue (max prevalence < {UNEXPRESSED_PCT_THRESHOLD}%) -- consider "
              f"excluding from the composite signature panel:")
        for gene, pct in flagged.items():
            print(f"    {gene}: max {pct:.2f}% of CXCR3+ cells expressing (any tissue)")
    else:
        print(f"  No curated genes fell below the {UNEXPRESSED_PCT_THRESHOLD}% "
              f"CXCR3+ prevalence flag in every tissue.")

    return diag


# ---------------------------------------------------------------------------
# Total counts / library complexity by tissue (checks whether PB vs CSF
# have a global compositional/technical difference -- motivated by
# fig3.py's competitive rank test, where ~68% of background genes showed
# negative log2FC in CSF vs PB, a pattern more consistent with a global
# shift than with ~150 independent biological effects)
# ---------------------------------------------------------------------------

def plot_total_counts_by_tissue(adata, out_dir=None) -> pd.DataFrame:
    """
    Compare total UMI counts and number of detected genes per cell between
    PB and CSF (and by dataset within each tissue), on RAW counts. A
    systematic difference here (e.g. CSF cells showing lower complexity)
    would explain a broad, non-specific negative log2FC shift across many
    genes in PB-vs-CSF comparisons, WITHOUT any real biological effect for
    most of those genes -- i.e. a normalization/technical confound rather
    than ~150 independent biology-driven changes. This is exactly the
    pattern that made fig3.py's competitive_rank_test's interpretation
    caveat necessary: it doesn't invalidate per-gene tests (each gene's
    OWN p-value already reflects its own signal relative to its own
    within-patient pairing), but it's important context for any
    ranking/pattern-level claim across many genes at once.

    Saves a CSV and a 2-panel diagnostic figure (total counts, n genes
    detected), both split by tissue.
    """
    obs = adata.obs
    X = adata.X
    if sp.issparse(X):
        total_counts = np.asarray(X.sum(axis=1)).ravel()
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        total_counts = np.asarray(X).sum(axis=1)
        n_genes = (np.asarray(X) > 0).sum(axis=1)

    df = pd.DataFrame({
        "tissue": obs["tissue"].values,
        "dataset": obs["dataset"].values if "dataset" in obs.columns else "unknown",
        "total_counts": total_counts,
        "n_genes_detected": n_genes,
    })

    print("\n  Total counts / complexity by tissue (raw counts):")
    summary = df.groupby("tissue")[["total_counts", "n_genes_detected"]].agg(["median", "mean", "std"])
    print(summary.to_string())

    print("\n  Total counts / complexity by tissue x dataset:")
    summary_ds = df.groupby(["tissue", "dataset"])[["total_counts", "n_genes_detected"]].agg(["median", "mean"])
    print(summary_ds.to_string())

    # Quick, honest flag: if PB and CSF medians differ by more than ~1.5x,
    # that's a substantial enough complexity gap to be a real confound for
    # any cross-tissue ranking/pattern analysis.
    tissue_medians = df.groupby("tissue")["total_counts"].median()
    if len(tissue_medians) >= 2:
        ratio = tissue_medians.max() / max(tissue_medians.min(), 1e-9)
        if ratio > 1.5:
            print(f"\n  [FLAG] Total-count medians differ by {ratio:.2f}x between tissues -- "
                  f"this is large enough to plausibly explain a broad, non-specific log2FC skew "
                  f"across many genes in PB-vs-CSF comparisons, independent of real biology.")
        else:
            print(f"\n  Total-count medians differ by only {ratio:.2f}x between tissues -- "
                  f"unlikely to be the main driver of any broad background log2FC skew observed "
                  f"downstream; look elsewhere (e.g. cell-type composition, ambient RNA) for that.")

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.5))
    tissues = sorted(df["tissue"].unique())
    axA.boxplot([df.loc[df["tissue"] == t, "total_counts"] for t in tissues], tick_labels=tissues, showfliers=False)
    axA.set_ylabel("Total UMI counts per cell")
    axA.set_title("Library size by tissue")
    axB.boxplot([df.loc[df["tissue"] == t, "n_genes_detected"] for t in tissues], tick_labels=tissues, showfliers=False)
    axB.set_ylabel("Genes detected per cell")
    axB.set_title("Complexity by tissue")
    fig.suptitle("Total counts / complexity: PB vs CSF (raw counts)", fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_dir = out_dir or (RESULTS_DIR / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "total_counts_by_tissue.png"
    out_pdf = out_dir / "total_counts_by_tissue.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_png}")

    df.to_csv(out_dir / "total_counts_by_tissue_percell.csv", index=False)

    return summary


def main() -> None:

    # 1. Load config and all datasets
    cfg   = load_cfg(DATASETS_YAML)
    adata = load_all_datasets(cfg, base_dir=DATA_DIR, gate_b=True, add_cxcr3=True)
    if not adata.obs_names.is_unique:
        print(" [fix] Making obs_names unique in merged object")
        adata.obs_names_make_unique()

    # CXCR3+/- gating (data_io.add_cxcr3_group) was just applied to RAW
    # counts as part of load_all_datasets, above -- i.e. before any
    # normalization happens in this script. Plot the rationale for that
    # threshold now, on the same raw values, before normalize_total/log1p
    # changes the scale (a histogram built after normalization would not
    # represent the space the actual gate was drawn in).
    print("\nPlotting CXCR3+/- threshold rationale …")
    plot_cxcr3_threshold_rationale(adata)

    # Build the curated zinc/MT gene list now (moved earlier than before)
    # so the expression diagnostics below can run on RAW counts, same
    # rationale as the CXCR3 threshold plot -- prevalence ("detected vs
    # not") is most interpretable before normalization changes the scale.
    markers_cfg       = cfg.get("markers", {})
    zinc_transporters = markers_cfg.get("zinc_transporters", [])
    metallothioneins  = markers_cfg.get("metallothioneins", [])
    artefact          = set(markers_cfg.get("exclude_artefacts", []))
    zinc_genes_all     = zinc_transporters + metallothioneins
    zinc_genes         = [g for g in zinc_genes_all if g not in artefact]

    print("\nComputing curated zinc/MT gene expression diagnostics (raw counts) …")
    compute_zinc_gene_expression_diagnostics(adata, zinc_genes)

    print("\nComputing total counts / library complexity by tissue (raw counts) …")
    plot_total_counts_by_tissue(adata)

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
    # (zinc_genes/zinc_transporters/metallothioneins already built above,
    # before normalization, for the expression diagnostic)
    # ------------------------------------------------------------------
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
    #
    # Defensive dtype fix before writing: pandas can produce nullable
    # 'string'/Arrow-backed string dtype columns (e.g. if a newer pandas
    # version is installed alongside/after other package updates), and
    # recent anndata (>=0.11) refuses to write these by default -- older
    # anndata versions can't read them back, so this is a real
    # forward/backward-compatibility hazard, not just a nuisance error.
    #
    # IMPORTANT: pandas >=3.0 introduced a NEW native 'str' dtype that is
    # DISTINCT from both the legacy 'object' dtype and the older nullable
    # 'string' extension dtype -- but it is *still implemented internally*
    # as a pandas StringArray, which is exactly the array type anndata's
    # write_nullable() rejects. This means `.astype(str)` does NOT escape
    # the problem in pandas 3.0 (it produces dtype name 'str', displayed
    # differently, but still backed by StringArray under the hood) --
    # verified directly: `idx.astype(str).array` is a `StringArray`
    # instance, while `idx.astype(object).array` is a plain
    # `NumpyExtensionArray`, which anndata accepts. `.astype(object)` is
    # therefore the only reliable fix, not `.astype(str)`.
    def _force_plain_string_dtypes(a: sc.AnnData) -> None:
        nullable_like_dtype_names = ("string", "string[python]", "string[pyarrow]", "str")
        a.obs.index = a.obs.index.astype(object)
        a.var.index = a.var.index.astype(object)
        for df in (a.obs, a.var):
            for col in df.columns:
                if str(df[col].dtype) in nullable_like_dtype_names:
                    df[col] = df[col].astype(object)
        if a.raw is not None:
            a.raw.var.index = a.raw.var.index.astype(object)
            for col in a.raw.var.columns:
                if str(a.raw.var[col].dtype) in nullable_like_dtype_names:
                    a.raw.var[col] = a.raw.var[col].astype(object)

    _force_plain_string_dtypes(adata)

    merged_path = RESULTS_DIR / "merged_bcells.h5ad"
    adata.write_h5ad(merged_path)
    print(f"Saved merged object -> {merged_path}")

    # 4. Zinc DE analysis (CXCR3+ vs CXCR3-), patient-paired.
    #
    # PRIMARY test pools BOTH datasets together (matching fig3.py's PB-vs-CSF
    # analysis, which pools first and checks dataset-confounding second --
    # previously this analysis did the opposite: it only ever computed
    # separate per-dataset tests and never a single combined one, losing
    # power for no reason once Harmony has integrated the embedding).
    #
    # IMPORTANT: Harmony (run earlier in this script) corrects the low-
    # dimensional PCA/UMAP embedding only -- it does NOT correct the actual
    # gene expression values (raw counts / log1p / pseudobulk means) used
    # here. Pooling datasets for this test does NOT automatically inherit
    # any batch correction; that's exactly why the formal between-dataset
    # check below exists, rather than assuming pooling is safe just because
    # Harmony ran upstream.
    present_zinc = [g for g in zinc_genes if g in adata.var_names]
    print(f"\nZinc genes present in  {len(present_zinc)} / {len(zinc_genes)}")

    gene_signs = compute_gene_signs(zinc_transporters, metallothioneins)
    print(f"Gene signs for composite signature: {gene_signs}")

    from scipy.stats import mannwhitneyu as _mwu

    for tissue in adata.obs["tissue"].unique():
        tissue_sub = adata[adata.obs["tissue"] == tissue]

        groups = tissue_sub.obs["cxcr3_group"].unique().tolist()
        if "CXCR3+" not in groups or "CXCR3-" not in groups:
            print(f"\n  [skip] {tissue}: missing CXCR3+/- groups")
            continue

        n_pos = (tissue_sub.obs["cxcr3_group"] == "CXCR3+").sum()
        n_neg = (tissue_sub.obs["cxcr3_group"] == "CXCR3-").sum()
        n_patients = tissue_sub.obs["patient"].nunique()
        print(f"\n=== {tissue}: {n_pos} CXCR3+ cells, {n_neg} CXCR3- cells, "
              f"{n_patients} patients (POOLED across both datasets) ===")

        # ---- PRIMARY: pooled per-gene DE -----------------------------------
        result_pooled = paired_pseudobulk_de(
            tissue_sub, groupby="cxcr3_group", group1="CXCR3-", group2="CXCR3+",
            patient_col="patient", genes=present_zinc,
        )
        if result_pooled.empty or result_pooled["pval"].isna().all():
            print(f"  [skip] {tissue}: no valid pooled paired results.")
            continue
        out_path = RESULTS_DIR / f"{tissue}_POOLED_zinc_results.csv"
        result_pooled.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")

        # ---- PRIMARY: pooled composite signature (signed + unsigned) -------
        sig_unsigned = paired_pseudobulk_signature_test(
            tissue_sub, groupby="cxcr3_group", group1="CXCR3-", group2="CXCR3+",
            signature_genes=present_zinc, patient_col="patient",
        )
        sig_signed = paired_pseudobulk_signature_test(
            tissue_sub, groupby="cxcr3_group", group1="CXCR3-", group2="CXCR3+",
            signature_genes=present_zinc, patient_col="patient", gene_signs=gene_signs,
        )
        print(f"  Signature (paired, POOLED both datasets, n={sig_unsigned.get('n_pairs')} patients): "
              f"unsigned p={sig_unsigned.get('wilcoxon_p')}, signed p={sig_signed.get('wilcoxon_p')}")
        pd.DataFrame([
            {"scope": "pooled_both_datasets", "signature": "unsigned",
             **{k: v for k, v in sig_unsigned.items() if k != "per_patient_values"}},
            {"scope": "pooled_both_datasets", "signature": "signed",
             **{k: v for k, v in sig_signed.items() if k != "per_patient_values"}},
        ]).to_csv(RESULTS_DIR / f"{tissue}_POOLED_zinc_signature.csv", index=False)

        # ---- CONFOUND CHECK: formal between-dataset comparison -------------
        # Does the MAGNITUDE of each patient's own CXCR3+/- difference itself
        # differ by dataset? Extracted from the pooled test's per-patient
        # values (not re-derived), then split by dataset and compared with an
        # unpaired Mann-Whitney U -- same logic as fig3.py's
        # check_pb_vs_csf_dataset_confounding between-dataset comparison.
        per_patient = sig_unsigned["per_patient_values"].copy()
        per_patient["diff"] = per_patient["CXCR3+"] - per_patient["CXCR3-"]
        patient_dataset = tissue_sub.obs.drop_duplicates("patient").set_index("patient")["dataset"]
        per_patient["dataset"] = patient_dataset.reindex(per_patient.index)

        ds_groups = [g["diff"].values for _, g in per_patient.groupby("dataset") if len(g) >= 3]
        ds_names_with_n = [(name, len(g)) for name, g in per_patient.groupby("dataset") if len(g) >= 3]
        if len(ds_groups) == 2:
            stat, between_ds_p = _mwu(ds_groups[0], ds_groups[1], alternative="two-sided")
            verdict = "SIGNIFICANT -- per-patient CXCR3+/- shift magnitude differs by dataset" \
                if between_ds_p < 0.05 else "not significant -- shift magnitude consistent across datasets"
            print(f"  Between-dataset comparison of per-patient CXCR3+/- differences "
                  f"({ds_names_with_n}): p={between_ds_p:.3g} ({verdict})")
        else:
            print(f"  [note] fewer than 2 datasets with >=3 patients -- cannot run formal "
                  f"between-dataset comparison; see per-dataset breakdown below instead.")

        # ---- DISEASE-STATE CHECK: pooled across datasets, per condition -----
        # Bigger n than splitting by dataset AND condition simultaneously
        # (e.g. pooling HC across both datasets rather than 3 alone / 5 alone).
        for condition in sorted(tissue_sub.obs["condition"].dropna().unique()):
            cond_sub = tissue_sub[tissue_sub.obs["condition"] == condition]
            cond_groups = cond_sub.obs["cxcr3_group"].unique().tolist()
            if "CXCR3+" not in cond_groups or "CXCR3-" not in cond_groups:
                print(f"    [{condition}, pooled] missing CXCR3+/- groups -- skipping.")
                continue
            cond_unsigned = paired_pseudobulk_signature_test(
                cond_sub, groupby="cxcr3_group", group1="CXCR3-", group2="CXCR3+",
                signature_genes=present_zinc, patient_col="patient",
            )
            cond_signed = paired_pseudobulk_signature_test(
                cond_sub, groupby="cxcr3_group", group1="CXCR3-", group2="CXCR3+",
                signature_genes=present_zinc, patient_col="patient", gene_signs=gene_signs,
            )
            print(f"    [{condition}, POOLED both datasets] n={cond_unsigned.get('n_pairs')} patients: "
                  f"unsigned p={cond_unsigned.get('wilcoxon_p')}, signed p={cond_signed.get('wilcoxon_p')}")

        # ---- SUPPORTING DETAIL: granular per-dataset breakdown --------------
        # Kept for transparency / manual inspection -- the formal
        # between-dataset test above is the one to cite for whether pooling
        # is defensible; this loop is the detail behind it.
        for ds_name in tissue_sub.obs["dataset"].unique():
            ds_sub = tissue_sub[tissue_sub.obs["dataset"] == ds_name]
            ds_groups_present = ds_sub.obs["cxcr3_group"].unique().tolist()
            if "CXCR3+" not in ds_groups_present or "CXCR3-" not in ds_groups_present:
                continue
            ds_n_patients = ds_sub.obs["patient"].nunique()
            ds_unsigned = paired_pseudobulk_signature_test(
                ds_sub, groupby="cxcr3_group", group1="CXCR3-", group2="CXCR3+",
                signature_genes=present_zinc, patient_col="patient",
            )
            ds_signed = paired_pseudobulk_signature_test(
                ds_sub, groupby="cxcr3_group", group1="CXCR3-", group2="CXCR3+",
                signature_genes=present_zinc, patient_col="patient", gene_signs=gene_signs,
            )
            print(f"    [{ds_name} only] n={ds_n_patients} patients: "
                  f"unsigned p={ds_unsigned.get('wilcoxon_p')}, signed p={ds_signed.get('wilcoxon_p')}")

    print("\nDone.")


if __name__ == "__main__":
    main()