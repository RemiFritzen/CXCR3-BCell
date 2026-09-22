#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
=========
Project-wide filesystem layout for the CXCR3 / B-cell zinc analysis pipeline.

This file is the single source of truth for all directory and file paths.
It should be imported by every other module in the package.

Dataset-specific metadata (tissue labels, condition names, marker genes, etc.)
belongs in config/datasets.yaml -- not here.

Usage
-----
    from config import BASE_DIR, H5AD, RESULTS_FILES, FIG_DIRS, DATASETS_YAML
"""

from pathlib import Path

# -- Root ---------------------------------------------------------------------
# Resolves to the directory containing this file, so the project is portable
# across machines and directory structures.
BASE_DIR = Path(__file__).resolve().parent.parent


# -- Data directories ---------------------------------------------------------
DATA_DIR      = BASE_DIR / "data"          # All input data
RAW_DIR       = DATA_DIR / "raw"           # Original count matrices (unmodified)
PROCESSED_DIR = DATA_DIR / "processed"     # QC-filtered, normalised .h5ad files

# -- Per-dataset processed subdirectories -------------------------------------
# Kept for legacy compatibility; most scripts should use H5AD dict below.
GSE133028_DIR = PROCESSED_DIR / "GSE133028"
GSE138266_DIR = PROCESSED_DIR / "GSE138266"

# -- Output directories -------------------------------------------------------
OUTPUT_DIR  = BASE_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"       # CSV result tables
QC_DIR      = OUTPUT_DIR / "qc"           # QC plots and metrics
FIGURE_DIR  = OUTPUT_DIR / "figures"      # All manuscript figures

# -- Per-figure subdirectories ------------------------------------------------
# Add new entries here when adding a new figure panel.
#
# Supplementary figures each get their OWN subdirectory under supp/, named by
# content rather than by "S-number" -- the manuscript's Figure S3/S4/S5/S7
# numbering is still being finalized (see figure-numbering audit), and a
# content-based folder name doesn't need to change if/when that numbering
# does. Whatever the final "Figure S#" label ends up being, it points to the
# same stable folder.
FIG_DIRS = {
    "fig1": FIGURE_DIR / "fig1_umap_overview",
    "fig2": FIGURE_DIR / "fig2_cxcr3_zinc",
    "fig3": FIGURE_DIR / "fig3_csf_vs_pb",
    "fig4": FIGURE_DIR / "fig4_patient_level",
    "supp": FIGURE_DIR / "supp",
    "supp_cxcr3neg_hc_vs_ms": FIGURE_DIR / "supp" / "cxcr3neg_hc_vs_ms",   # FigS4.py
    "supp_pb_vs_csf_paired":  FIGURE_DIR / "supp" / "pb_vs_csf_paired",    # FigS5.py (both CXCR3+/-)
    "supp_cxcr3_paired_pb":   FIGURE_DIR / "supp" / "cxcr3_paired_pb",     # FigS6.py
    "supp_cxcr3_paired_csf":  FIGURE_DIR / "supp" / "cxcr3_paired_csf",    # FigS7.py
}

# -- Config / metadata files --------------------------------------------------
# datasets.yaml  -- per-study metadata (obs column names, filters, markers)
# figure_mapping -- maps figure panels to the scripts that produce them
DATASETS_YAML  = BASE_DIR / "config" / "datasets.yaml"
FIGURE_MAPPING = BASE_DIR / "config" / "figure_mapping.md"

# -- Processed h5ad registry --------------------------------------------------
# Centralised lookup so scripts never hard-code .h5ad paths.
# ALL_Bcells is the pooled object written by run_all_zinc_analyses.py.
H5AD = {
    "GSE138266": PROCESSED_DIR / "GSE138266_processed.h5ad",
    "GSE133028": PROCESSED_DIR / "GSE133028_processed.h5ad",
    "GSE239626": PROCESSED_DIR / "GSE239626_processed.h5ad",
    "ALL_Bcells": PROCESSED_DIR / "ALL_Bcells_for_Nygen.h5ad",
}

# -- Results file registry ----------------------------------------------------
# Centralised lookup for CSV outputs from the zinc transporter analysis.
# ALL_datasets -- pooled across all three datasets
# ALL_patients  -- patient-level summary across all datasets
RESULTS_FILES = {
    "GSE138266":    RESULTS_DIR / "GSE138266_agnostic_zinc_results.csv",
    "GSE133028":    RESULTS_DIR / "GSE133028_agnostic_zinc_results.csv",
    "GSE239626":    RESULTS_DIR / "GSE239626_agnostic_zinc_results.csv",
    "ALL_datasets": RESULTS_DIR / "ALL_DATASETS_agnostic_zinc_results.csv",
    "ALL_patients": RESULTS_DIR / "ALL_DATASETS_by_patients_agnostic_zinc_results.csv",
}

# -- Directory creation (uncomment to auto-create on import) ------------------
# Useful when setting up a fresh clone. Commented out by default so that
# importing config does not silently create directories on shared filesystems.
#
# _dirs = [RAW_DIR, PROCESSED_DIR, GSE138266_DIR, GSE133028_DIR,
#          RESULTS_DIR, QC_DIR, *FIG_DIRS.values(),
#          BASE_DIR / "config"]
# for _d in _dirs:
#     _d.mkdir(parents=True, exist_ok=True)
