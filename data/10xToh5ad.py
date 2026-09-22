#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10xToh5ad.py
============
Converts Cell Ranger filtered_feature_bc_matrix directories to the per-sample
.h5ad files the pipeline reads, following build_patient_path's convention:

    <PROCESSED_DIR>/<dataset>/matrices/<sample>/filtered_feature_bc_matrix/
        -> <PROCESSED_DIR>/<dataset>/<sample>.h5ad

Usage
-----
    python data/10xToh5ad.py                      # convert every sample found
    python data/10xToh5ad.py --dataset GSE133028  # one dataset
    python data/10xToh5ad.py --sample CSF_10      # one sample
    python data/10xToh5ad.py --sample CSF_10 --force   # overwrite existing

WHY THE GENE-NAME CHECK
-----------------------
Sample CSF_10 was at one point written with var_names set to positional
integers ('0', '1', '2', ...) rather than gene symbols: the features table
was not read, and the conversion completed silently anyway. Nothing
downstream noticed -- the file loaded, the cell count was right, and the
sample simply contributed no zinc/MT signal, because none of the curated
genes could be matched by name. It was found only by auditing var_names
across every sample.

verify_gene_names() therefore refuses to write a file whose gene names look
positional, and reports the mitochondrial gene count (13 for a correctly
read human matrix) as a second, independent check that symbols were
actually parsed.

Cell-level metadata is NOT written here. tissue, condition, sex and patient
are set from config/datasets.yaml by data_io._standardise_obs at load time;
writing them here as well would create a second, silently divergent source
for the same fields.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import anndata as ad
import scanpy as sc

from config.config import PROCESSED_DIR

ad.settings.allow_write_nullable_strings = True

MATRIX_SUBDIR = "filtered_feature_bc_matrix"
EXPECTED_HUMAN_MT_GENES = 13


def verify_gene_names(adata: ad.AnnData, source: Path) -> None:
    """Refuse to write an object whose var_names are not gene symbols."""
    if adata.n_vars == 0:
        raise RuntimeError(f"{source}: no genes read.")

    if adata.var_names.str.fullmatch(r"\d+").all():
        raise RuntimeError(
            f"{source}: gene names are positional integers, so the features "
            f"table was not read. Refusing to write an .h5ad with no gene "
            f"identities (this is the failure that silently produced an "
            f"unusable CSF_10)."
        )

    n_mt = int(adata.var_names.str.startswith("MT-").sum())
    if n_mt != EXPECTED_HUMAN_MT_GENES:
        print(f"  [warn] {n_mt} MT- genes (expected {EXPECTED_HUMAN_MT_GENES} "
              f"for a human matrix) -- check the reference used.")
    if not adata.var_names.is_unique:
        print("  [fix] duplicate gene symbols -- making var_names unique")
        adata.var_names_make_unique()


def convert(mtx_dir: Path, out_path: Path, force: bool = False) -> bool:
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (use --force to overwrite)")
        return False

    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=False)
    verify_gene_names(adata, mtx_dir)

    n_mt = int(adata.var_names.str.startswith("MT-").sum())
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes; {n_mt} MT- genes")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(out_path)
    print(f"  [written] {out_path}")
    return True


def discover(base: Path, dataset: str | None, sample: str | None):
    """Yield (dataset, sample, matrix_dir) for every matrix directory found."""
    datasets = [base / dataset] if dataset else sorted(
        d for d in base.iterdir() if d.is_dir() and (d / "matrices").is_dir())
    for ds in datasets:
        mroot = ds / "matrices"
        if not mroot.is_dir():
            print(f"[warn] no matrices/ under {ds}")
            continue
        samples = [mroot / sample] if sample else sorted(
            s for s in mroot.iterdir() if s.is_dir())
        for s in samples:
            mtx = s / MATRIX_SUBDIR
            if mtx.is_dir():
                yield ds.name, s.name, mtx
            else:
                print(f"[warn] {s}: no {MATRIX_SUBDIR}/ -- skipped")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="convert only this dataset (e.g. GSE133028)")
    ap.add_argument("--sample", help="convert only this sample (e.g. CSF_10)")
    ap.add_argument("--force", action="store_true", help="overwrite existing .h5ad files")
    ap.add_argument("--base-dir", default=str(PROCESSED_DIR),
                    help="root holding <dataset>/matrices/ (default: PROCESSED_DIR)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not base.is_dir():
        print(f"[fatal] base dir not found: {base}", file=sys.stderr)
        return 1

    found = written = failed = 0
    for ds_name, s_name, mtx in discover(base, args.dataset, args.sample):
        found += 1
        print(f"\n[{ds_name}/{s_name}]")
        out_path = base / ds_name / f"{s_name}.h5ad"
        try:
            written += bool(convert(mtx, out_path, force=args.force))
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  [FAILED] {exc}")

    if found == 0:
        print(f"\n[fatal] no matrix directories found under {base}", file=sys.stderr)
        return 1
    print(f"\n{found} sample(s) found, {written} written, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())