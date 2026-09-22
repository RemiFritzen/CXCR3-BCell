#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
checkZnExpression.py
====================
Writes Supplementary Table 2: detection frequency and mean RAW count of every
zinc/metallothionein gene in CXCR3+ B cells, per compartment.

    output/results/table_S2_zinc_expression.csv

Replaces the previous version, which had three defects:

1. It read `adata.raw`, which in this pipeline holds log-normalised values,
   and reported their mean as the "mean count". The published Table S2 column
   is a raw-count mean, so the old output was roughly half the true value.
2. It looped only over the 14 panel genes in the config, so MT1A, MT1G, MT1H
   and MT1M -- the genes Table S2 exists to justify excluding -- never
   appeared.
3. It printed PB CXCR3+ twice (once at module level, once in the loop).

Raw counts are taken from, in order of preference:
  a) merged_bcells.h5ad layers["counts"]
  b) merged_bcells.h5ad .X, if still integer-valued
  c) a fresh load of the per-sample files through the same B-cell gate
     (data_io.load_all_datasets), which reads raw counts before any
     normalisation. Cell numbers are checked against the merged object so
     the table always describes the same cells as the figures.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from config.config import RESULTS_DIR, DATASETS_YAML
from data_io import load_cfg

# Genes excluded from the panel for negligible expression; Table S2 reports
# them so the exclusion is visible and checkable.
EXCLUDED_MT = ["MT1A", "MT1G", "MT1H", "MT1M"]
TISSUES = ["CSF", "PB"]
EXCLUSION_THRESHOLD_PCT = 1.0
REPO = Path(__file__).resolve().parent


def find_base_dir(cfg: dict) -> Path:
    """Locate the directory data_io expects as base_dir, i.e. the one for which
    <base_dir>/processed/<dataset>/<patient>.h5ad exists. Absolute paths, so it
    works whatever the working directory is (Spyder often is not the repo root)."""
    ds_name, ds_cfg = next(iter(cfg["datasets"].items()))
    patient = next(iter(ds_cfg["patients"]))
    candidates = []
    try:
        from config.config import DATA_DIR
        candidates += [Path(DATA_DIR), Path(DATA_DIR).parent]
    except ImportError:
        pass
    candidates += [REPO / "data", REPO]
    for c in candidates:
        c = c if c.is_absolute() else (REPO / c)
        if (c / "processed" / ds_name / f"{patient}.h5ad").exists():
            return c
    sys.exit("[fatal] could not find the per-sample .h5ad files; looked for "
             f"processed/{ds_name}/{patient}.h5ad under: " + ", ".join(map(str, candidates)))


def is_integer_valued(X) -> bool:
    v = X.data[:20000] if sp.issparse(X) else np.asarray(X).ravel()[:20000]
    return v.size > 0 and np.allclose(v, np.round(v))


def raw_count_source(merged: sc.AnnData, cfg: dict):
    if "counts" in merged.layers:
        return merged, merged.layers["counts"], "merged_bcells.h5ad layers['counts']"
    if is_integer_valued(merged.X):
        return merged, merged.X, "merged_bcells.h5ad .X (integer-valued)"
    print("[info] merged object holds normalised values only; re-loading raw "
          "counts from the per-sample files through the same B-cell gate ...")
    from data_io import load_all_datasets
    base_dir = find_base_dir(cfg)
    print(f"[info] per-sample files under: {base_dir}")
    raw = load_all_datasets(cfg, base_dir=base_dir)
    if not raw.obs_names.is_unique:
        raw.obs_names_make_unique()
    if not is_integer_valued(raw.X):
        sys.exit("[fatal] re-loaded object is not integer-valued either; "
                 "cannot compute raw-count means.")
    return raw, raw.X, "per-sample files via data_io.load_all_datasets"


def column(X, idx, mask):
    col = X[mask, idx]
    return col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()


def main() -> int:
    cfg = load_cfg(DATASETS_YAML)
    markers = cfg.get("markers", {})
    panel = list(markers.get("zinc_transporters", [])) + list(markers.get("metallothioneins", []))
    genes = list(dict.fromkeys(panel + EXCLUDED_MT))

    merged = sc.read_h5ad(RESULTS_DIR / "merged_bcells.h5ad")
    src_obj, X, src = raw_count_source(merged, cfg)
    print(f"[info] raw counts from: {src}")

    rows, n_cells = [], {}
    for tissue in TISSUES:
        mask = ((src_obj.obs["tissue"].astype(str) == tissue) &
                (src_obj.obs["cxcr3_group"].astype(str) == "CXCR3+")).to_numpy()
        n_cells[tissue] = int(mask.sum())

        # The table must describe the same cells as the figures.
        ref = int(((merged.obs["tissue"].astype(str) == tissue) &
                   (merged.obs["cxcr3_group"].astype(str) == "CXCR3+")).sum())
        if n_cells[tissue] != ref:
            sys.exit(f"[fatal] {tissue} CXCR3+ cells: {n_cells[tissue]} in the raw-count "
                     f"source vs {ref} in merged_bcells.h5ad - not the same cell set.")

        for g in genes:
            if g not in src_obj.var_names:
                print(f"[warn] {g} not in var_names ({tissue})")
                rows.append({"gene": g, "tissue": tissue,
                             "pct_expressing": np.nan, "mean_count": np.nan})
                continue
            v = column(X, src_obj.var_names.get_loc(g), mask)
            rows.append({"gene": g, "tissue": tissue,
                         "pct_expressing": round(100 * float((v > 0).mean()), 2),
                         "mean_count": round(float(v.mean()), 4)})

    long = pd.DataFrame(rows)
    wide = long.pivot(index="gene", columns="tissue", values=["pct_expressing", "mean_count"])
    wide.columns = [f"{t}_{m}" for m, t in wide.columns]
    wide = wide.reindex(genes)[[f"{t}_{m}" for t in TISSUES
                                for m in ("pct_expressing", "mean_count")]]

    out = RESULTS_DIR / "table_S2_zinc_expression.csv"
    wide.to_csv(out)

    print("\nSupplementary Table 2 -- CXCR3+ B cells")
    print("  " + "   ".join(f"{t} ({n_cells[t]} cells)" for t in TISSUES))
    print(wide.to_string())

    # Exclusion check: the four excluded MTs must stay below threshold in both.
    for g in EXCLUDED_MT:
        pcts = [wide.loc[g, f"{t}_pct_expressing"] for t in TISSUES]
        flag = "OK" if all(p < EXCLUSION_THRESHOLD_PCT for p in pcts if pd.notna(p)) else "ABOVE 1% - revisit exclusion"
        print(f"[exclusion] {g}: " + ", ".join(f"{t} {p}%" for t, p in zip(TISSUES, pcts)) + f"  {flag}")

    print(f"\n[done] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())