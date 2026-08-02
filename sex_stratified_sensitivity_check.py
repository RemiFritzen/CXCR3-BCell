#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
sex_stratified_sensitivity_check.py

Sex-stratified sensitivity analysis for the CXCR3+ zinc/metallothionein
signature, restricted to MS patients only (reviewer point #11: "Expand on
how sex as a biological variable was considered in the analysis, given the
differences in zinc concentration between males and females").

WHY MS-ONLY, NOT HC: HC has only n=8 total, which would split into
roughly 3-5 per sex -- too few to test reliably (same "too few to test"
threshold used everywhere else in this pipeline, e.g. MIN_PAIRS=3 is
already a soft floor, and 3-5 per group is right at that floor with no
margin). MS has n=18-19, which splits into more usable numbers given the
dataset's ~60-70% female composition (Table S1). Restricting to MS also
directly matches Fig 2's own population -- this checks whether the
already-established MS composite signature finding is confounded by sex,
not a new independent hypothesis about HC.

WHAT THIS TESTS:
  1. The composite zinc/MT signature (signed, matching Fig 2's primary
     readout), Male vs Female, within MS CXCR3+ patients, PB and CSF
     separately.
  2. Each of Fig 2's individually-significant genes (CSF: SLC39A11,
     SLC30A9, SLC39A6, SLC39A9, SLC39A14, SLC30A1), Male vs Female, to
     check whether the disease-associated per-gene signal specifically is
     confounded by sex composition -- a null composite-level result could
     still hide a sex effect concentrated in one or two genes.

STATISTICS: unpaired (each patient is one sex), patient-level pseudobulk,
two-sided Mann-Whitney U -- the same `pseudobulk_de`/`pseudobulk_signature_test`
framework already used for the MS-vs-HC comparison (Fig 2), just with
`groupby="sex"` instead of `groupby="condition"`, and pre-filtered to
CXCR3+ MS patients only before either function runs.
"""

import scipy.sparse as sp
import numpy as np
import pandas as pd
import scanpy as sc

from config.config import RESULTS_DIR, DATASETS_YAML
from data_io import load_cfg
from stats_utils import pseudobulk_de, pseudobulk_signature_test, compute_gene_signs


def load_zinc_gene_list() -> list[str]:
    cfg = load_cfg(DATASETS_YAML)
    markers = cfg.get("markers", {})
    artefact = set(markers.get("exclude_artefacts", []))
    all_g = markers.get("zinc_transporters", []) + markers.get("metallothioneins", [])
    return [g for g in all_g if g not in artefact]


def load_zinc_gene_signs() -> dict[str, float]:
    cfg = load_cfg(DATASETS_YAML)
    markers = cfg.get("markers", {})
    artefact = set(markers.get("exclude_artefacts", []))
    transporters = [g for g in markers.get("zinc_transporters", []) if g not in artefact]
    metallothioneins = [g for g in markers.get("metallothioneins", []) if g not in artefact]
    return compute_gene_signs(transporters, metallothioneins)


# Genes individually significant in Fig 2's CXCR3+ MS-vs-HC comparison
# (panel-restricted padj < 0.05), checked here for a sex confound.
# CSF: verified significant genes. PB: Fig 2 showed 0/14 individually
# significant, so no PB gene list to check here -- only the composite.
FIG2_CSF_SIGNIFICANT_GENES = [
    "SLC39A11", "SLC30A9", "SLC39A6", "SLC39A9", "SLC39A14", "SLC30A1",
]


def run_sex_check_for_tissue(
    adata: sc.AnnData,
    tissue: str,
    zinc_genes: list[str],
    gene_signs: dict[str, float],
    genes_to_check_individually: list[str],
    out_dir,
) -> dict:
    """
    Sex-stratified (M vs F) sensitivity check for CXCR3+ MS patients in
    one tissue: composite signature test + individual gene checks for
    whatever genes were significant in Fig 2 for this tissue.
    """
    sub = adata[
        (adata.obs["tissue"] == tissue)
        & (adata.obs["condition"] == "MS")
        & (adata.obs["cxcr3_group"] == "CXCR3+")
        & (adata.obs["sex"].isin(["M", "F"]))
    ].copy()

    n_patients_m = sub.obs.loc[sub.obs["sex"] == "M", "patient"].nunique()
    n_patients_f = sub.obs.loc[sub.obs["sex"] == "F", "patient"].nunique()
    print(f"\n=== {tissue}, MS CXCR3+, sex-stratified check ===")
    print(f"  n patients: M={n_patients_m}, F={n_patients_f} ({sub.n_obs} cells total)")

    if n_patients_m < 3 or n_patients_f < 3:
        print(f"  [WARNING] fewer than 3 patients in one sex group -- result below is "
              f"exploratory only, same threshold used elsewhere in this pipeline for "
              f"'too few to test reliably'.")

    genes_present = [g for g in zinc_genes if g in sub.var_names]

    # Composite signature (signed, matching Fig 2's primary readout)
    sig_result = pseudobulk_signature_test(
        sub, groupby="sex", group1="M", group2="F",
        signature_genes=genes_present, patient_col="patient",
    )
    print(f"  Composite signature (signed), M vs F: "
          f"n_M={sig_result.get('n_patients_M')}, n_F={sig_result.get('n_patients_F')}, "
          f"p={sig_result.get('mannwhitney_p')}")

    pd.DataFrame([{k: v for k, v in sig_result.items() if k != "per_patient_values"}]).to_csv(
        out_dir / f"sex_check_{tissue}_MS_CXCR3pos_composite_signature.csv", index=False
    )

    # Individual genes that were significant in Fig 2 for this tissue
    gene_result_df = None
    genes_to_check = [g for g in genes_to_check_individually if g in genes_present]
    if genes_to_check:
        de_result = pseudobulk_de(
            sub, groupby="sex", group1="M", group2="F",
            genes=genes_to_check, patient_col="patient",
        )
        if not de_result.empty:
            print(f"  Per-gene check (Fig 2's significant genes for this tissue), M vs F:")
            print(de_result[["gene", "log2fc", "pval", "padj"]].to_string(index=False))
            gene_result_df = de_result
            de_result.to_csv(
                out_dir / f"sex_check_{tissue}_MS_CXCR3pos_fig2_genes.csv", index=False
            )
    else:
        print(f"  No Fig 2 significant genes to check individually for {tissue} "
              f"(PB had 0/14 individually significant in the original comparison).")

    return {
        "tissue": tissue, "n_patients_M": n_patients_m, "n_patients_F": n_patients_f,
        "composite_signature_result": sig_result, "gene_level_result": gene_result_df,
    }


def main() -> None:
    merged_path = RESULTS_DIR / "merged_bcells.h5ad"
    if not merged_path.exists():
        raise FileNotFoundError(f"merged_bcells.h5ad not found at {merged_path}")

    adata = sc.read_h5ad(merged_path)
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()

    if "sex" not in adata.obs.columns:
        raise KeyError(
            "Column 'sex' missing in adata.obs -- re-run the pipeline with the "
            "updated datasets.yaml (sex per patient) and patched data_io.py first."
        )

    unknown_sex = (adata.obs["sex"] == "unknown").sum()
    if unknown_sex > 0:
        print(f"[warn] {unknown_sex} cells have sex='unknown' -- these are excluded "
              f"from this check, not treated as a third group.")

    zinc_genes = load_zinc_gene_list()
    gene_signs = load_zinc_gene_signs()
    print(f"Loaded {len(zinc_genes)} curated zinc/MT genes")

    out_dir = RESULTS_DIR / "sex_stratified_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for tissue, sig_genes in [("PB", []), ("CSF", FIG2_CSF_SIGNIFICANT_GENES)]:
        results[tissue] = run_sex_check_for_tissue(
            adata, tissue=tissue, zinc_genes=zinc_genes, gene_signs=gene_signs,
            genes_to_check_individually=sig_genes, out_dir=out_dir,
        )

    print(f"\nSaved sex-stratified check outputs -> {out_dir}")
    print(
        "\nNote for the manuscript response: HC was not tested (n=8 total, "
        "~3-5 per sex after splitting -- too few to test reliably). This "
        "check is MS-only, matching Fig 2's population, and asks whether "
        "the already-established MS composite signature and per-gene "
        "findings are confounded by sex, not a new independent claim about "
        "sex effects in HC."
    )


if __name__ == "__main__":
    main()