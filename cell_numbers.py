#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cell_numbers.py
===============
Regenerates the two per-patient count tables from the merged object:

  output/results/per_sample_bcell_counts.csv      one row per sample
  output/results/supplement_per_patient_counts.csv  one row per patient (Table S1)

Replaces the previous version, which had two defects:

1. It READ per_sample_bcell_counts.csv as an input, but nothing in the repo
   wrote that file -- it existed only as a committed artefact. Once deleted,
   the script could not run, and while it did run it was reporting counts
   frozen at whatever state that CSV was committed in (which is why Table S1
   disagreed with the figures). Both files are now derived from
   merged_bcells.h5ad, so they track the data.

2. extract_patient_numeric() only stripped the tissue suffix when it was a
   digit ("CSF_2" -> "2"), so GSE138266 IDs such as "MS19270_CSF" kept the
   tissue glued on and the CSF/PB rows for one patient never merged. Every
   GSE138266 patient therefore appeared twice, with NaN on one side and a
   missing Condition. Fixed in split_patient_id().

B-cell gating
-------------
The counts must describe the same cell set the figures use. If the merged
object is already gated, run without --gate (the default) and the totals will
match the figures. If it holds all cells, pass --gate to apply the two-step
gate from the Methods before counting. The script reports which situation it
is in; --report-only prints the diagnosis and writes nothing.

Usage
-----
    python cell_numbers.py --report-only     # diagnose, write nothing
    python cell_numbers.py                   # count the object as-is
    python cell_numbers.py --gate            # apply the B-cell gate first
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

try:
    from config.config import RESULTS_DIR
except Exception:                                     # runnable standalone
    RESULTS_DIR = None


# --- gate definition, per Methods ------------------------------------------
B_MARKERS = ["CD19", "MS4A1", "CD79A", "CD79B", "CD74",
             "IGHM", "IGHD", "IGKC", "IGLC2"]
NON_B_MARKERS = [
    # T / NK
    "CD3D", "CD3E", "TRAC", "TRBC1", "TRBC2", "NKG7", "GNLY", "PRF1",
    # monocyte / myeloid
    "LST1", "LILRB1", "LILRB2", "S100A8", "S100A9", "LYZ", "FCGR3A",
    "C1QA", "C1QB", "C1QC",
    # erythroid / megakaryocyte
    "HBB", "HBA1", "HBA2", "PF4", "PPBP",
]
NON_B_TOP_FRACTION = 0.30     # top 30% excluded as contaminants/doublets


# ---------------------------------------------------------------------------
def split_patient_id(pid: str) -> str:
    """Collapse a sample ID to its patient ID, dropping any tissue suffix.

        CSF_2       -> 2            (GSE133028)
        PB_9        -> 9
        MS19270_CSF -> MS19270      (GSE138266)  <-- the case the old code missed
        PTC85037_PB -> PTC85037
    """
    pid = str(pid)
    if "_" in pid:
        head, tail = pid.split("_", 1)
        if tail.isdigit():
            return tail
        if tail.upper() in ("CSF", "PB"):
            return head
        if head.upper() in ("CSF", "PB"):
            return tail
    return pid


def _dense_col(adata, gene):
    x = adata[:, gene].X
    return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()


def apply_bcell_gate(adata):
    """Two-step gate: any B marker detected, then drop the top 30% by mean
    non-B marker expression among those cells."""
    bm = [g for g in B_MARKERS if g in adata.var_names]
    nbm = [g for g in NON_B_MARKERS if g in adata.var_names]
    if not bm:
        raise RuntimeError("no B-cell markers present in var_names")
    print(f"[gate] B markers found: {len(bm)}/{len(B_MARKERS)}; "
          f"non-B markers found: {len(nbm)}/{len(NON_B_MARKERS)}")

    pos = np.zeros(adata.n_obs, dtype=bool)
    for g in bm:
        pos |= _dense_col(adata, g) > 0
    print(f"[gate] B-marker positive: {pos.sum()} / {adata.n_obs}")

    sub = adata[pos].copy()
    if nbm:
        score = np.mean([_dense_col(sub, g) for g in nbm], axis=0)
        # Rank-based, not a value threshold: many cells score exactly 0, so a
        # quantile cutoff with `score < cutoff` can land on 0 and discard every
        # cell. Dropping the highest-scoring 30% by count is what "top 30%"
        # means and is well defined under ties (stable sort keeps input order).
        n_drop = int(round(NON_B_TOP_FRACTION * sub.n_obs))
        order = np.argsort(score, kind="stable")          # ascending
        keep = np.zeros(sub.n_obs, dtype=bool)
        keep[order[: sub.n_obs - n_drop]] = True
        print(f"[gate] dropping top {NON_B_TOP_FRACTION:.0%} by non-B score "
              f"({n_drop} cells; score at the cut: "
              f"{score[order[sub.n_obs - n_drop]] if n_drop else float('nan'):.4f})")
        sub = sub[keep].copy()
    print(f"[gate] retained: {sub.n_obs} cells")
    return sub


# ---------------------------------------------------------------------------
def resolve_columns(obs):
    """Find the obs columns we need, tolerating naming variation."""
    def pick(*cands):
        for c in cands:
            if c in obs.columns:
                return c
        return None
    cols = {
        "dataset":   pick("dataset", "batch", "study"),
        "patient":   pick("patient", "sample", "sample_id", "donor", "patient_id"),
        "tissue":    pick("tissue", "compartment"),
        "condition": pick("condition", "diagnosis", "diagnosis_group", "group"),
        "cxcr3":     pick("cxcr3_group", "CXCR3_group", "cxcr3_status"),
    }
    missing = [k for k, v in cols.items() if v is None and k != "cxcr3"]
    if missing:
        raise RuntimeError(f"obs is missing required column(s): {missing}. "
                           f"Available: {list(obs.columns)}")
    return cols


def cxcr3_positive(adata, cxcr3_col):
    """Boolean CXCR3+ per cell, from the existing group column if present,
    otherwise from detectable CXCR3 (UMI > 0), per Methods."""
    if cxcr3_col is not None:
        s = adata.obs[cxcr3_col].astype(str)
        return s.str.contains(r"\+", regex=True).to_numpy()
    if "CXCR3" not in adata.var_names:
        raise RuntimeError("no CXCR3 group column and CXCR3 not in var_names")
    print("[info] no cxcr3 group column; deriving from CXCR3 UMI > 0")
    return _dense_col(adata, "CXCR3") > 0


def build_per_sample(adata, cols):
    obs = adata.obs
    df = pd.DataFrame({
        "dataset":   obs[cols["dataset"]].astype(str).to_numpy(),
        "patient":   obs[cols["patient"]].astype(str).to_numpy(),
        "tissue":    obs[cols["tissue"]].astype(str).to_numpy(),
        "condition": obs[cols["condition"]].astype(str).to_numpy(),
        "is_pos":    cxcr3_positive(adata, cols["cxcr3"]),
    })
    out = (df.groupby(["dataset", "patient", "tissue", "condition"],
                      observed=True, dropna=False)
             .agg(n_cells=("is_pos", "size"), n_CXCR3_pos=("is_pos", "sum"))
             .reset_index())
    out["n_CXCR3_neg"] = out["n_cells"] - out["n_CXCR3_pos"]
    return out[out["n_cells"] > 0].copy()


def build_per_patient(counts):
    c = counts.copy()
    c["Patient_ID"] = c["patient"].map(split_patient_id)

    def side(tis, suffix):
        return (c[c["tissue"] == tis]
                .groupby(["dataset", "Patient_ID"], observed=True)
                .agg(**{f"#cells_{suffix}":   ("n_cells", "sum"),
                        f"#CXCR3+_{suffix}":  ("n_CXCR3_pos", "sum"),
                        f"#CXCR3-_{suffix}":  ("n_CXCR3_neg", "sum")})
                .reset_index())

    # Condition taken across both compartments, so a patient present in only
    # one tissue still gets a label (the old code lost it on the PB rows).
    cond = (c.groupby(["dataset", "Patient_ID"], observed=True)["condition"]
              .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan)
              .reset_index().rename(columns={"condition": "Condition"}))

    out = (cond
           .merge(side("CSF", "CSF"), on=["dataset", "Patient_ID"], how="left")
           .merge(side("PB", "PB"),   on=["dataset", "Patient_ID"], how="left"))
    for col in out.columns:
        if col.startswith("#"):
            out[col] = out[col].astype("Int64")
    return out.sort_values(["dataset", "Patient_ID"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR) if RESULTS_DIR else None,
                    help="output/results directory holding merged_bcells.h5ad")
    ap.add_argument("--gate", action="store_true",
                    help="apply the two-step B-cell gate before counting")
    ap.add_argument("--report-only", action="store_true",
                    help="print the diagnosis and write nothing")
    args = ap.parse_args()

    if not args.results_dir:
        print("[fatal] pass --results-dir (config.config import failed)", file=sys.stderr)
        return 1
    rd = args.results_dir
    path = os.path.join(rd, "merged_bcells.h5ad")
    if not os.path.exists(path):
        print(f"[fatal] not found: {path}", file=sys.stderr)
        return 1

    adata = sc.read_h5ad(path)
    print(f"[info] merged object: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"[info] obs columns: {list(adata.obs.columns)}")

    # --- is this object gated? ---
    n_b = None
    if all(g in adata.var_names for g in ("CD3E", "MS4A1")):
        t = (_dense_col(adata, "CD3E") > 0).mean() * 100
        b = (_dense_col(adata, "MS4A1") > 0).mean() * 100
        print(f"[diag] cells detecting CD3E: {t:.1f}%   MS4A1: {b:.1f}%")
        print("[diag] a gated B-cell object should show LOW CD3E and HIGH MS4A1.")
    print(f"[diag] Figure S1 of the paper reports 38,935 B cells. This object "
          f"has {adata.n_obs}.")
    if args.report_only:
        gated = apply_bcell_gate(adata)
        print(f"[diag] applying the gate here would give {gated.n_obs} cells")
        return 0

    if args.gate:
        adata = apply_bcell_gate(adata)

    cols = resolve_columns(adata.obs)
    print(f"[info] using columns: {cols}")

    counts = build_per_sample(adata, cols)
    summary = build_per_patient(counts)

    p1 = os.path.join(rd, "per_sample_bcell_counts.csv")
    p2 = os.path.join(rd, "supplement_per_patient_counts.csv")
    counts.to_csv(p1, index=False)
    summary.to_csv(p2, index=False)

    print()
    print(f"[done] {p1}  ({len(counts)} sample rows)")
    print(f"[done] {p2}  ({len(summary)} patient rows)")
    print(f"[check] total cells counted: {int(counts['n_cells'].sum())}")
    dup = summary[summary.duplicated(["dataset", "Patient_ID"], keep=False)]
    if len(dup):
        print(f"[warn] duplicated patient rows: {sorted(dup.Patient_ID)}")
    nocond = summary[summary["Condition"].isna()]
    if len(nocond):
        print(f"[warn] patients with no Condition: {sorted(nocond.Patient_ID)}")
    onesided = summary[summary["#cells_CSF"].isna() | summary["#cells_PB"].isna()]
    if len(onesided):
        print(f"[info] patients present in one compartment only: "
              f"{sorted(onesided.Patient_ID)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
