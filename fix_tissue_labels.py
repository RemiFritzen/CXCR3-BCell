#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_tissue_labels.py
====================
Corrects the `tissue` annotation in the per-sample GSE133028 .h5ad files.

Background
----------
Six GSE133028 samples carry a `obs["tissue"]` value that contradicts their
filename (CSF_6, CSF_27 -> "PB"; CSF_32 -> "PBMC"; PB_2, PB_6, PB_7 -> "CSF"),
and two more (PB_3, PB_32) use the nonstandard label "PBMC".

The filename is correct. This was established from lineage composition:
blood-lineage transcripts (PPBP, PF4, HBB, HBA1, S100A8, S100A9, LYZ, FCN1)
were lower in the CSF-named sample than in that same patient's PB-named
sample in 16 of 16 patients (sign test p = 1.5e-5), including all six
discordant samples. The merged object already assigns tissue from the sample
ID, so no published result depends on the value being fixed here -- this
brings the per-sample metadata into line with what the analysis already used.

What it does
------------
For every *.h5ad under <PROC>/GSE133028:
  - determines the correct tissue from the sample ID (CSF_* -> CSF, else PB)
  - if the stored value already matches, leaves the file untouched
  - otherwise rewrites obs["tissue"], recording the old value and the
    evidence in .uns, and writes the file back atomically

Pandas 3 compatibility
----------------------
On pandas >= 3 string columns and indices default to StringDtype
(ArrowStringArray), which anndata refuses to write unless
`allow_write_nullable_strings` is enabled -- and enabling it produces files
that anndata < 0.11 cannot read. This script instead casts every string
index/column back to plain object dtype before writing, and verifies the
cast held (pandas re-infers StringDtype from some constructors, so the
cast is checked rather than assumed).

Usage
-----
    python fix_tissue_labels.py --dry-run     # report only, write nothing
    python fix_tissue_labels.py               # apply
    python fix_tissue_labels.py --backup      # apply, keeping <file>.bak

Run from anywhere; PROC is resolved relative to this file's location by
default, or pass --proc /path/to/data/processed.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import anndata as ad


EVIDENCE = (
    "Tissue corrected to match sample ID. Verified by lineage composition: "
    "blood-lineage markers (PPBP, PF4, HBB, HBA1, S100A8, S100A9, LYZ, FCN1) "
    "were lower in the CSF-named sample than in the paired PB-named sample "
    "in 16 of 16 patients (sign test p = 1.5e-5)."
)

VALID = {"CSF", "PB"}


# ---------------------------------------------------------------------------
# pandas 3 / anndata string handling
# ---------------------------------------------------------------------------
def _as_object_index(idx: pd.Index) -> pd.Index:
    """Return idx with plain object dtype.

    NB: pd.Index(np.asarray(idx, dtype=object)) re-infers StringDtype on
    pandas 3 and is NOT sufficient. .astype(object) holds.
    """
    if idx.dtype == object:
        return idx
    out = idx.astype(object)
    if out.dtype != object:  # belt and braces
        out = pd.Index(idx.to_numpy(dtype=object), dtype=object)
    return out


def _as_object_series(s: pd.Series) -> pd.Series:
    """Return s with plain object dtype, preserving categorical structure."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats = s.cat.categories
        if cats.dtype != object and pd.api.types.is_string_dtype(cats):
            new_cats = pd.Index(cats.to_numpy(dtype=object), dtype=object)
            s = s.cat.rename_categories(new_cats)
        return s
    if s.dtype != object and pd.api.types.is_string_dtype(s):
        return pd.Series(s.to_numpy(dtype=object), index=s.index, dtype=object)
    return s


def _plainify_frame(df: pd.DataFrame) -> None:
    """In-place: force index and all string/categorical columns to object."""
    df.index = _as_object_index(df.index)
    for col in list(df.columns):
        df[col] = _as_object_series(df[col])


def plainify(adata: ad.AnnData) -> ad.AnnData:
    """Force every string dtype in the object to plain object dtype."""
    _plainify_frame(adata.obs)
    _plainify_frame(adata.var)
    if adata.raw is not None:
        try:
            _plainify_frame(adata.raw.var)
        except Exception:
            pass
    return adata


def _string_dtype_offenders(adata: ad.AnnData) -> list[str]:
    """Names of anything still on a nullable string dtype."""
    bad = []
    for label, df in (("obs", adata.obs), ("var", adata.var)):
        if df.index.dtype != object:
            bad.append(f"{label}._index ({df.index.dtype})")
        for col in df.columns:
            dt = df[col].dtype
            if isinstance(dt, pd.CategoricalDtype):
                if dt.categories.dtype != object and \
                        pd.api.types.is_string_dtype(dt.categories):
                    bad.append(f"{label}['{col}'].categories "
                               f"({dt.categories.dtype})")
            elif dt != object and pd.api.types.is_string_dtype(dt):
                bad.append(f"{label}['{col}'] ({dt})")
    return bad


# ---------------------------------------------------------------------------
# tissue logic
# ---------------------------------------------------------------------------
def expected_tissue(sample_id: str) -> str:
    return "CSF" if sample_id.upper().startswith("CSF") else "PB"


def current_tissue(adata: ad.AnnData) -> str:
    if "tissue" not in adata.obs.columns:
        return "<missing>"
    vals = pd.unique(adata.obs["tissue"].dropna().astype(str))
    if len(vals) == 0:
        return "<empty>"
    if len(vals) > 1:
        return "|".join(sorted(map(str, vals)))
    return str(vals[0])


def write_atomic(adata: ad.AnnData, path: str, backup: bool) -> None:
    """Write to a temp file in the same directory, then swap it in."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(suffix=".h5ad.tmp", dir=d)
    os.close(fd)
    try:
        # convert_strings_to_categoricals=False: the default conversion can
        # reintroduce a nullable dtype in the new categories.
        try:
            adata.write_h5ad(tmp, convert_strings_to_categoricals=False)
        except TypeError:
            # older anndata without that keyword
            adata.write_h5ad(tmp)
        if backup and os.path.exists(path):
            bak = path + ".bak"
            if not os.path.exists(bak):
                os.replace(path, bak)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_proc = os.path.join(here, "data", "processed")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc", default=default_proc,
                    help=f"data/processed directory (default: {default_proc})")
    ap.add_argument("--dataset", default="GSE133028",
                    help="subdirectory to process (default: GSE133028)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing")
    ap.add_argument("--backup", action="store_true",
                    help="keep the original as <file>.bak")
    args = ap.parse_args()

    ddir = os.path.join(args.proc, args.dataset)
    files = sorted(glob.glob(os.path.join(ddir, "*.h5ad")))
    if not files:
        print(f"[fatal] no .h5ad files under {ddir}", file=sys.stderr)
        return 1

    print(f"[info] {len(files)} files in {ddir}")
    print(f"[info] pandas {pd.__version__}")
    if args.dry_run:
        print("[info] DRY RUN -- nothing will be written")
    print()

    changed, unchanged, failed = [], [], []

    for path in files:
        sid = os.path.splitext(os.path.basename(path))[0]
        want = expected_tissue(sid)
        try:
            a = ad.read_h5ad(path)
        except Exception as e:
            print(f"  {sid:14s} READ FAILED: {e}")
            failed.append((sid, f"read: {e}"))
            continue

        have = current_tissue(a)
        if have == want:
            print(f"  {sid:14s} ok ({have})")
            unchanged.append(sid)
            del a
            continue

        print(f"  {sid:14s} {have!r} -> {want!r}", end="")
        if args.dry_run:
            print("   [dry-run]")
            changed.append((sid, have, want))
            del a
            continue

        try:
            a.uns["tissue_original"] = have
            a.uns["tissue_corrected_on"] = pd.Timestamp.now().isoformat()
            a.uns["tissue_corrected_evidence"] = EVIDENCE
            a.obs["tissue"] = pd.Categorical([want] * a.n_obs)

            a = plainify(a)
            bad = _string_dtype_offenders(a)
            if bad:
                raise RuntimeError(
                    "nullable string dtype survived the cast: " + "; ".join(bad))

            write_atomic(a, path, backup=args.backup)
            del a

            # read back and confirm the change actually persisted
            check = ad.read_h5ad(path)
            got = current_tissue(check)
            n = check.n_obs
            del check
            if got != want:
                raise RuntimeError(f"verification failed: file now reads {got!r}")
            print(f"   done ({n} cells)")
            changed.append((sid, have, want))
        except Exception as e:
            print(f"   FAILED: {e}")
            failed.append((sid, str(e)))

    print()
    print(f"[summary] {len(changed)} changed, {len(unchanged)} already correct, "
          f"{len(failed)} failed")
    for sid, old, new in changed:
        print(f"    {sid}: {old} -> {new}")
    for sid, err in failed:
        print(f"    FAILED {sid}: {err}")

    if not args.dry_run and changed:
        print()
        print("[next] rerun qc_pipeline.py and confirm metadata_audit.csv "
              "has 0 FAIL rows, then commit the new audit.")
        print("[next] fix the tissue lookup in 10xToh5ad.py as well, or the "
              "next rebuild reintroduces this.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
