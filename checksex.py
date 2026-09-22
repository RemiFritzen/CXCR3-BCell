#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_sex_annotations.py
========================
QC check for the CXCR3 / B-cell zinc pipeline.

Validates the hand-entered `sex:` annotations in config/datasets.yaml against
X/Y marker expression in the corresponding .h5ad objects, and flags any two
objects whose expression matrices are identical.

Two failure modes this catches:

  MISMATCH   a patient's declared sex disagrees with their data -- the object
             in that slot belongs to someone else, or the YAML is wrong.
  DUPLICATE  two patient keys resolve to byte-identical expression data.

Place alongside config.py. Run with no arguments:

    python check_sex_annotations.py

Writes <RESULTS_DIR>/sex_annotation_check.csv.

In Spyder, press F5 -- there is no command line to configure. Results are
left in the module-level `results` DataFrame.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import PROCESSED_DIR, DATASETS_YAML, RESULTS_DIR  # noqa: E402
from data_io import load_cfg  # noqa: E402


# ---------------------------------------------------------------------------
# Marker panels
# ---------------------------------------------------------------------------
# XIST is expressed from the inactive X and is the female marker. The rest are
# Y-linked. Detection is scored as the fraction of cells with a non-zero value,
# not per cell -- 10x data is sparse enough that single cells are unreliable
# but a few hundred cells separate cleanly.

FEMALE_MARKERS = ["XIST", "TSIX"]
MALE_MARKERS = ["RPS4Y1", "DDX3Y", "UTY", "EIF1AY", "KDM5D", "USP9Y", "NLGN4Y"]

# Small objects cannot be called confidently. MS74594_CSF (74 cells) and
# PTC85037_CSF (73 cells) sit near this line.
MIN_CELLS_FOR_CALL = 100


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def candidate_paths(dataset: str, key: str) -> list[Path]:
    """
    data_io.build_patient_path inserts an extra 'processed' segment when
    base_dir already ends in 'processed', so the layout that actually shipped
    is not certain. Try the plausible ones and use whichever exists.
    """
    return [
        PROCESSED_DIR / dataset / f"{key}.h5ad",
        PROCESSED_DIR / "processed" / dataset / f"{key}.h5ad",
        PROCESSED_DIR / f"{key}.h5ad",
        PROCESSED_DIR.parent / dataset / f"{key}.h5ad",
    ]


def resolve(dataset: str, key: str) -> Path | None:
    for p in candidate_paths(dataset, key):
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------

def gene_index(adata: ad.AnnData) -> pd.Index:
    for col in ("gene_symbols", "gene_symbol", "feature_name", "symbol", "gene_name"):
        if col in adata.var.columns:
            return pd.Index(adata.var[col].astype(str))
    names = pd.Index(adata.var_names.astype(str))
    if names.str.startswith("ENSG").mean() > 0.5:
        print("  ! var_names look like Ensembl IDs; no symbol column found in .var")
        print("    .var columns:", list(adata.var.columns))
    return names


def counts_matrix(adata: ad.AnnData):
    """Prefer raw counts. Detection is on > 0, so log1p data works too."""
    if "counts" in adata.layers:
        return adata.layers["counts"]
    if adata.raw is not None and adata.raw.shape[1] >= adata.shape[1]:
        return adata.raw.X
    return adata.X


def detection(mat, symbols: pd.Index, markers: list[str]) -> dict[str, float]:
    out = {}
    n = mat.shape[0]
    for g in markers:
        hits = np.flatnonzero(symbols == g)
        if len(hits) == 0:
            continue
        col = mat[:, hits]
        n_pos = int((col > 0).sum()) if sp.issparse(col) else int((np.asarray(col) > 0).sum())
        out[g] = n_pos / n if n else 0.0
    return out


def call_sex(fem: dict, male: dict, n_cells: int) -> str:
    if n_cells < MIN_CELLS_FOR_CALL:
        return "too_few_cells"
    f = max(fem.values(), default=0.0)
    m = max(male.values(), default=0.0)
    n_male_genes = sum(1 for v in male.values() if v > 0.005)
    if m > 0.01 and n_male_genes >= 2 and m > f:
        return "M"
    if f > 0.02 and m < 0.005:
        return "F"
    if f > m * 3 and f > 0.01:
        return "F"
    if m > f * 3 and m > 0.005:
        return "M"
    return "ambiguous"


def matrix_hash(mat) -> str:
    h = hashlib.md5()
    if sp.issparse(mat):
        c = mat.tocsr()
        for part in (c.data, c.indices, c.indptr):
            h.update(np.ascontiguousarray(part).tobytes())
    else:
        h.update(np.ascontiguousarray(mat).tobytes())
    h.update(str(mat.shape).encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(yaml_path: Path = DATASETS_YAML) -> pd.DataFrame:
    cfg = load_cfg(yaml_path)
    rows, missing = [], []

    for dataset, ds_cfg in cfg.get("datasets", {}).items():
        patients = ds_cfg.get("patients")
        if not patients:
            print(f"[{dataset}] no per-patient block; skipped")
            continue

        print(f"\n[{dataset}] {len(patients)} configured samples")
        for key, meta in patients.items():
            path = resolve(dataset, key)
            if path is None:
                missing.append((dataset, key))
                print(f"  [MISSING] {key}")
                continue

            adata = ad.read_h5ad(path)
            symbols = gene_index(adata)
            mat = counts_matrix(adata)
            fem = detection(mat, symbols, FEMALE_MARKERS)
            male = detection(mat, symbols, MALE_MARKERS)
            called = call_sex(fem, male, mat.shape[0])
            declared = str(meta.get("sex", "?")).upper()

            rows.append({
                "dataset": dataset,
                "key": key,
                "n_cells": mat.shape[0],
                "declared_sex": declared,
                "called_sex": called,
                "agrees": called == declared if called in ("M", "F") else None,
                "declared_condition": meta.get("condition", "?"),
                "XIST_pct": round(100 * fem.get("XIST", 0.0), 2),
                "RPS4Y1_pct": round(100 * male.get("RPS4Y1", 0.0), 2),
                "DDX3Y_pct": round(100 * male.get("DDX3Y", 0.0), 2),
                "UTY_pct": round(100 * male.get("UTY", 0.0), 2),
                "max_Y_pct": round(100 * max(male.values(), default=0.0), 2),
                "matrix_md5": matrix_hash(mat),
                "path": str(path),
            })

    if not rows:
        raise RuntimeError(
            "No objects loaded. Checked layouts:\n  "
            + "\n  ".join(str(p) for p in candidate_paths("<DATASET>", "<KEY>"))
        )

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "sex_annotation_check.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)

    print("\n" + "=" * 70)
    print(f"{len(df)} objects read; written to {out}")

    bad = df[df.agrees == False]  # noqa: E712
    if len(bad):
        print(f"\n*** {len(bad)} SEX MISMATCH(ES) ***")
        print(bad[["dataset", "key", "n_cells", "declared_sex", "called_sex",
                   "XIST_pct", "max_Y_pct"]].to_string(index=False))
    else:
        print("\nAll callable samples agree with their declared sex.")

    dup = df[df.duplicated("matrix_md5", keep=False)]
    if len(dup):
        print("\n*** IDENTICAL EXPRESSION MATRICES ***")
        for h, grp in dup.groupby("matrix_md5"):
            print(f"  {h}  ({grp['n_cells'].iloc[0]} cells): {', '.join(grp['key'])}")
    else:
        print("\nNo identical matrices.")

    uncallable = df[~df.called_sex.isin(["M", "F"])]
    if len(uncallable):
        print(f"\n{len(uncallable)} not callable (inspect manually):")
        print(uncallable[["key", "n_cells", "called_sex", "XIST_pct",
                          "max_Y_pct"]].to_string(index=False))

    if missing:
        print(f"\n{len(missing)} configured sample(s) with no file on disk:")
        for d, k in missing:
            print(f"  {d}/{k}")

    print("=" * 70)
    return df


if __name__ == "__main__":
    results = run()
else:
    results = run()