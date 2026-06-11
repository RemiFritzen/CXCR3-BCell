# CXCR3-BCell

Analysis code and supporting outputs for the study **"CXCR3⁺ B-cells adopt a zinc-programmed state in the cerebrospinal fluid of multiple sclerosis patients."** This repository contains the Python scripts, configuration files, processed outputs, and figure-generation workflow used to re-analyse public single-cell RNA-seq datasets of peripheral blood (PB) and cerebrospinal fluid (CSF) B cells from healthy controls/non-MS controls, clinically isolated syndrome (CIS), and multiple sclerosis (MS) patients, with a focus on zinc-homeostasis-related transcriptional programs in CXCR3⁺ B cells.[file:1][file:2]

## Overview

The project re-analyses integrated human single-cell datasets to test whether CXCR3⁺ B cells in demyelinating disease adopt a distinct zinc-handling transcriptional state.[file:1] The main findings reported in the manuscript are that CXCR3⁺ B cells show increased expression of zinc transporters such as **SLC39A11** in PB and CSF, increased **SLC30A9** and **SLC39A10** in CSF, reduced metallothioneins including **MT2A** and **MT1X** in CSF, and an increased composite zinc/metallothionein signature in disease compared with controls.[file:1][file:2]

The analysis is based on two public datasets referenced in the manuscript, **GSE138266** and **GSE133028**.[file:1] Downstream analyses include dataset harmonisation, B-cell filtering, CXCR3-based subsetting, differential expression testing, compartment comparisons between PB and CSF, and calculation of a per-cell zinc/metallothionein signature.[file:1][file:2]

## Repository structure

### `config/`

Repository configuration files.

- `config.py` – central analysis configuration, paths, and reusable parameters.
- `datasets.yaml` – dataset definitions and source metadata used by the pipeline.

### `data/`

Input data and dataset-specific metadata.

- `raw/` – raw inputs, GEO manifests, patient lists, and helper files used during preprocessing.
  - `10xToh5ad.py` – helper script for converting 10x-formatted input data to `.h5ad`.
  - `GSE133028_patient_list.csv`
  - `GSE138266_manifest.csv`
  - `GSE138266_patient_list.csv`
  - `GSE138266_patient_list.xlsx`
  - `pnas.2008523117.sd01.csv`
- `processed/` – processed objects and intermediate files produced during analysis. Large processed datasets are typically not suitable for version control and may be ignored in Git.

### `output/`

Generated analysis outputs.

- `figures/` – manuscript figure panels and exported graphics.
  - `fig1_umap_overview/`
  - `fig2_cxcr3_zinc/`
  - `fig3_csf_vs_pb/`
  - `fig4_patient_level/`
  - `supp/`
- `qc/` – quality-control figures and exploratory plots, including UMAPs, gene-count summaries, and zinc-signature distributions.[file:3][file:7][file:8][file:9][file:10][file:11][file:12]
- `results/` – summary tables and processed outputs, including:
  - `figure1_bar_patient_values.csv`
  - `figure1_bar_values.csv`
  - per-dataset and per-sample zinc results tables
  - `merged_bcells.h5ad`
  - `reference_bcells.h5ad`
  - `per_sample_bcell_counts.csv`
  - `supplement_per_patient_counts.csv`

### Top-level analysis scripts

- `analysis.py` – core analysis code for figure generation, zinc signature analyses, and differential expression.
- `cell_numbers.py` – utilities for counting B cells across samples, conditions, and tissues.
- `checkZnExpression.py` – exploratory analysis of zinc-related gene expression.
- `data_io.py` – shared functions for loading and saving data objects and tables.
- `geneset.py` – curated zinc/metallothionein gene-set definitions used throughout the analysis.
- `preprocess_GSE138266.py` – preprocessing script for the GSE138266 dataset.
- `run_all_zinc_analyses.py` – convenience entry point to run the main zinc-analysis workflow.

## Data sources

This repository re-analyses publicly available single-cell RNA-seq datasets described in the manuscript.[file:1] The primary datasets used are:

- **GSE138266**
- **GSE133028**

According to the manuscript and supplementary methods, analyses were restricted to harmonised B cells from PB and CSF, with non-B-cell contaminants excluded before downstream comparisons.[file:1][file:2]

## Analysis workflow

The workflow implemented in this repository follows the logic described in the manuscript and supplementary methods.[file:1][file:2]

1. Download or assemble the required public input files and sample metadata.[file:1][file:2]
2. Preprocess individual datasets into AnnData objects.
3. Harmonise datasets and retain B cells from PB and CSF for downstream analysis.[file:1][file:2]
4. Annotate or subset cells by CXCR3 status and disease group (HC/non-MS, CIS, MS).[file:1][file:2]
5. Define a curated zinc-related gene set including zinc transporters and metallothioneins while excluding genes considered biologically irrelevant or artefactual for B cells.[file:1][file:2]
6. Perform differential expression analyses between disease groups and between tissue compartments.[file:1][file:2]
7. Compute a per-cell zinc/metallothionein signature from the processed expression matrix by averaging curated zinc-related genes present in each subset.[file:2]
8. Export result tables, QC figures, and manuscript figure panels.[file:1][file:2]

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/RemiFritzen/CXCR3-BCell.git
cd CXCR3-BCell
```

### 2. Create the Python environment

An environment file is strongly recommended for reproducibility. If you add `environment.yml` or `requirements.txt`, installation can follow one of these patterns:

```bash
conda env create -f environment.yml
conda activate cxcr3-bcell
```

or

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add input data

Raw datasets are **not bundled in this repository** and should be downloaded separately from the original public sources referenced in the manuscript.[file:1] In the workflow used for this project, data were downloaded manually using **txg** and **fasterq-dump**, uploaded to the **10x Genomics** platform, and processed there with **Cell Ranger 10.0**. The resulting archive containing the filtered output data was then downloaded, and the final `.h5ad` object was created locally using the `10xToh5ad.py` script in `data/raw/`. Place raw files, manifests, downloaded filtered outputs, and patient metadata in `data/raw/` following the naming scheme already used in the repository.

### 4. Run preprocessing and analysis

Example command sequence:

```bash
python preprocess_GSE138266.py
python run_all_zinc_analyses.py
```

Depending on your local setup, individual figure or QC steps may also be run through `analysis.py`, `checkZnExpression.py`, or `cell_numbers.py`.

### 5. Find outputs

- Main figure panels: `output/figures/`
- Supplementary panels: `output/figures/supp/`
- QC outputs: `output/qc/`
- Result tables and processed objects: `output/results/`

## Methods summary

The supplementary methods state that the analysis uses Python-based single-cell workflows including **Scanpy, SciPy, NumPy, pandas, and Matplotlib**.[file:2] Differential expression was performed using a Wilcoxon rank-sum framework implemented through Scanpy, with healthy controls as the reference group for HC-versus-MS comparisons, and signature distributions were compared with two-sided Mann-Whitney tests.[file:1][file:2]

The supplementary text also notes that expression filtering before differential expression was applied separately by tissue, retaining genes expressed above a minimum proportion of cells in both groups; the current implementation uses a minimum proportion of **10% for PB** and **15% for CSF**.[file:2]

## Figures

The repository is organised around the main figure outputs used in the manuscript.[file:1]

- **Figure 1** – dataset integration and CXCR3 compartment overview; outputs are stored under `output/figures/fig1_umap_overview/`.
- **Figure 2** – zinc/metallothionein genes and signature in CXCR3⁺ B cells; outputs are stored under `output/figures/fig2_cxcr3_zinc/`.
- **Figure 3** – compartmental comparison of CXCR3⁺ B cells between CSF and PB; outputs are stored under `output/figures/fig3_csf_vs_pb/`.
- **Figure 4** – patient-level zinc signature analyses; outputs are stored under `output/figures/fig4_patient_level/`.
- **Supplementary figures** – supplementary and QC outputs are stored under `output/figures/supp/` and `output/qc/`.[file:2]

## Reproducibility notes

To make the repository easier for reviewers and readers to reproduce, it is useful to include:

- an exported conda environment file such as `environment.yml` or a `requirements.txt`,
- a short note describing the expected raw input files for each dataset,
- the curated zinc/metallothionein gene list if not already embedded clearly in `geneset.py`,
- a minimal command sequence for reproducing each main figure.

## Citation

If you use this repository, please cite the associated manuscript:

> Fritzen R. *CXCR3⁺ B-cells adopt a zinc-programmed state in the cerebrospinal fluid of multiple sclerosis patients.*[file:1]

Please also cite the original public datasets and studies from which the source data were obtained, as listed in the manuscript references.[file:1]

## Status

This repository supports an active manuscript and may continue to evolve as analyses, figures, and documentation are refined.[file:1][file:2]
