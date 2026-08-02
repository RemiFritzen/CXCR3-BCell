#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_utils.py
--------------
Patient-aware statistics for the B-cell zinc / CXCR3 pipeline.

Why this file exists
---------------------
Per-cell tests (e.g. `sc.tl.rank_genes_groups` with `method="wilcoxon"`, or a
Mann-Whitney test run directly on per-cell signature values) treat every cell
as an independent observation. Cells from the same patient are correlated
(they share genotype, disease state, sampling day, technical batch, etc.),
so this is pseudoreplication: it inflates the effective sample size from
"n patients" to "n cells" and can turn a modest, patient-level effect into
an absurdly small p-value (e.g. the p = 3.34e-61 in the CXCR3- PB signature
comparison in the current draft).

Two independent fixes are provided here. Use whichever suits the analysis:

1. `pseudobulk_matrix` / `pseudobulk_de` / `pseudobulk_signature_test`
   Aggregate cells to one value per patient (mean, by default) before
   testing. This is the field-standard fix for scRNA-seq DE (same logic as
   pseudobulk + DESeq2/edgeR in R). Sample size becomes n patients, which is
   honest but low (as few as 5-8 per group here) -- expect fewer genes to
   reach significance, which is the correct, if less flattering, result.

2. `mixed_model_de`
   Fits `expression ~ group` with a random intercept per patient
   (statsmodels MixedLM), retaining all cells but modeling the
   patient-to-patient correlation explicitly, so the p-value reflects
   patient-level evidence rather than cell-level counts.

Both are patient-level correct. Pseudobulk is simpler, more standard for
reviewers, and works well with two-sided Mann-Whitney (no distributional
assumptions). Mixed models can have more power but are more fragile with
n patients as low as 2 (e.g. the CIS group) -- they can fail to converge or
give unstable estimates in that regime, so pseudobulk is the safer default
for this dataset's sample sizes; keep mixed_model_de as a sensitivity check.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Pseudobulk aggregation
# ---------------------------------------------------------------------------

def pseudobulk_matrix(
    adata,
    groupby: str,
    patient_col: str = "patient",
    genes: list[str] | None = None,
    agg: str = "mean",
    use_raw: bool = True,
) -> pd.DataFrame:
    """
    Collapse cells to one row per patient.

    Returns a DataFrame indexed by patient with one column per gene, plus:
      - '_group'   : that patient's value of `groupby` (must be 1:1 with patient)
      - 'n_cells'  : number of cells contributed by that patient

    Raises if any patient has more than one distinct `groupby` value, since
    that would make "group" undefined at the patient level.
    """
    genes = genes or adata.var_names.tolist()

    source = adata.raw if (use_raw and adata.raw is not None) else adata
    used_genes = [g for g in genes if g in source.var_names]
    missing = [g for g in genes if g not in source.var_names]
    if missing:
        print(f"  [pseudobulk] {len(missing)} gene(s) not found, skipped: {missing}")

    X = source[:, used_genes].X
    if sp.issparse(X):
        X = X.toarray()

    df = pd.DataFrame(np.asarray(X), columns=used_genes, index=adata.obs_names)
    df[patient_col] = adata.obs[patient_col].values
    df["_group"] = adata.obs[groupby].values

    n_groups_per_patient = df.groupby(patient_col)["_group"].nunique()
    bad = n_groups_per_patient[n_groups_per_patient > 1]
    if len(bad):
        raise ValueError(
            f"Patients with >1 distinct '{groupby}' value (pseudobulk undefined): "
            f"{bad.index.tolist()}"
        )

    agg_fn = "mean" if agg == "mean" else "sum"
    pb = df.groupby(patient_col)[used_genes].agg(agg_fn)
    pb["_group"] = df.groupby(patient_col)["_group"].first()
    pb["n_cells"] = df.groupby(patient_col).size()
    return pb


# ---------------------------------------------------------------------------
# Pseudobulk differential expression (per gene)
# ---------------------------------------------------------------------------

def pseudobulk_de(
    adata,
    groupby: str,
    group1: str,
    group2: str,
    patient_col: str = "patient",
    genes: list[str] | None = None,
    agg: str = "mean",
    min_patients_per_group: int = 3,
    use_raw: bool = True,
    log2fc_from_log_means: bool = True,
) -> pd.DataFrame:
    """
    Per-gene MS-vs-HC (or any two-group) DE at the patient level.

    log2fc is reported as group2 vs group1 (i.e. group1 is the reference),
    matching the convention `reference=group1` would give in
    sc.tl.rank_genes_groups(groups=[group2], reference=group1).

    log2fc_from_log_means: scanpy stores log1p (natural-log) values in
    `.raw` by default, which is what `sc.tl.rank_genes_groups` uses unless
    told otherwise. If the aggregated values here are already log-scale
    (the default, since use_raw=True pulls from `.raw`), the correct fold
    change is a DIFFERENCE of the mean log values converted to log2 units —
    NOT log2 of a ratio of means-of-logs, which is not a meaningful quantity
    (log(mean(log x)) composition). Set this to False only if `genes`/`source`
    are on a linear scale (e.g. raw counts or CPM), in which case log2fc
    reverts to log2(mean_group2 / mean_group1).

    Returns one row per gene: n per group, group means, log2fc, pval, padj.
    """
    pb = pseudobulk_matrix(
        adata, groupby, patient_col=patient_col, genes=genes, agg=agg, use_raw=use_raw
    )

    a = pb[pb["_group"] == group1]
    b = pb[pb["_group"] == group2]

    if len(a) < min_patients_per_group or len(b) < min_patients_per_group:
        print(
            f"  [warn] pseudobulk_de: only {len(a)} '{group1}' vs {len(b)} '{group2}' "
            f"patients -- results below are likely underpowered; interpret with caution."
        )

    gene_cols = [c for c in pb.columns if c not in ("_group", "n_cells")]
    rows = []
    for g in gene_cols:
        x = a[g].values
        y = b[g].values
        if len(x) == 0 or len(y) == 0:
            continue
        if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
            stat, p = np.nan, 1.0
        else:
            stat, p = mannwhitneyu(x, y, alternative="two-sided")

        if log2fc_from_log_means:
            log2fc = (y.mean() - x.mean()) / np.log(2)
        else:
            log2fc = np.log2((y.mean() + 1e-9) / (x.mean() + 1e-9))

        rows.append({
            "gene": g,
            f"n_{group1}": len(x),
            f"n_{group2}": len(y),
            f"mean_{group1}": x.mean(),
            f"mean_{group2}": y.mean(),
            "log2fc": log2fc,
            "pval": p,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["padj"] = multipletests(result["pval"].fillna(1.0), method="fdr_bh")[1]
    return result.sort_values("padj").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sign-corrected composite signature
# ---------------------------------------------------------------------------
#
# Why: the zinc/MT panel mixes genes expected to move in OPPOSITE directions
# with disease -- transporters (SLC30Axx/ZnT, SLC39Axx/ZIP) go up, buffering
# metallothioneins (MT2A, MT1X, ...) go down. A plain mean of raw expression
# across this gene set lets those opposite movements partially cancel, which
# attenuates the composite signature's separation between groups relative to
# what individual genes show, and makes "the signature went up" ambiguous
# (it could mean transporters dominated the mean while MTs quietly fell
# further). The fix is to flip the sign of the down-direction genes before
# averaging, so every gene contributes in the same direction (toward "more
# dyshomeostasis-like").
#
# Signs MUST come from prior biological role (transporter vs buffer), not
# from the DE result of the comparison being tested -- deriving sign from
# that comparison's own fold changes would be circular (the sign would be
# chosen to maximize the very separation the signature is then used to test,
# inflating apparent significance).

def compute_gene_signs(
    zinc_transporters: list[str],
    metallothioneins: list[str],
) -> dict[str, float]:
    """
    A priori sign for each gene in the zinc/MT panel: +1 for transporters
    (expected up with disease/CXCR3+ activation), -1 for metallothioneins
    (expected down). Genes in neither list are not included (caller should
    decide how to handle genes outside this scheme).
    """
    signs: dict[str, float] = {}
    for g in zinc_transporters:
        signs[g] = 1.0
    for g in metallothioneins:
        signs[g] = -1.0
    return signs


def signed_cell_signature(
    adata,
    gene_signs: dict[str, float],
    use_raw: bool = False,
) -> np.ndarray:
    """
    Per-cell composite signature: mean of sign[gene] * expression[gene]
    across the genes in gene_signs, for every cell in `adata`.

    use_raw=False (default) operates on adata.X directly, matching how the
    existing pipeline computes the per-cell signature for the violin plots
    (typically the scaled/z-scored matrix). Set True to use adata.raw
    instead (e.g. log1p values).
    """
    source = adata.raw if (use_raw and adata.raw is not None) else adata
    genes = [g for g in gene_signs if g in source.var_names]
    missing = [g for g in gene_signs if g not in source.var_names]
    if missing:
        print(f"  [signed_cell_signature] {len(missing)} gene(s) not found, skipped: {missing}")
    if not genes:
        raise ValueError("None of the requested genes are present in adata.")

    X = source[:, genes].X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    sign_vec = np.array([gene_signs[g] for g in genes])
    return (X * sign_vec[np.newaxis, :]).mean(axis=1)




def pseudobulk_signature_test(
    adata,
    groupby: str,
    group1: str,
    group2: str,
    signature_genes: list[str],
    patient_col: str = "patient",
    agg: str = "mean",
    use_raw: bool = True,
    gene_signs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Patient-level test of the composite zinc/metallothionein signature.

    This replaces per-cell Mann-Whitney on the signature (the source of the
    inflated p-values, e.g. p=3.34e-61 for PB CXCR3- MS vs HC): here n is the
    number of patients per group, not the number of cells.

    use_raw: pass False if the signature should be computed on adata.X
    directly (e.g. a scaled/z-scored matrix, as used elsewhere in this
    pipeline for the per-cell violin plots) rather than adata.raw.

    gene_signs: optional {gene: +1 or -1} from compute_gene_signs. If given,
    each gene's pseudobulk value is multiplied by its sign before averaging,
    so transporters (up) and metallothioneins (down) both contribute in the
    same direction instead of partially cancelling in a plain mean. If not
    given, falls back to a plain unsigned mean (previous behavior).
    """
    pb = pseudobulk_matrix(
        adata, groupby, patient_col=patient_col, genes=signature_genes, agg=agg, use_raw=use_raw
    )
    gene_cols = [c for c in pb.columns if c not in ("_group", "n_cells")]

    if gene_signs is not None:
        missing = [g for g in gene_cols if g not in gene_signs]
        if missing:
            print(f"  [warn] pseudobulk_signature_test: no sign specified for {missing}, "
                  f"treating as +1")
        sign_vec = np.array([gene_signs.get(g, 1.0) for g in gene_cols])
        pb["signature"] = (pb[gene_cols].values * sign_vec[np.newaxis, :]).mean(axis=1)
    else:
        pb["signature"] = pb[gene_cols].mean(axis=1)

    a = pb.loc[pb["_group"] == group1, "signature"].values
    b = pb.loc[pb["_group"] == group2, "signature"].values

    if len(a) < 2 or len(b) < 2:
        print(f"  [warn] signature test: n={len(a)} vs n={len(b)} patients -- too few to test.")
        return {
            f"n_patients_{group1}": len(a),
            f"n_patients_{group2}": len(b),
            "mannwhitney_p": np.nan,
        }

    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return {
        f"n_patients_{group1}": len(a),
        f"n_patients_{group2}": len(b),
        f"median_{group1}": float(np.median(a)),
        f"median_{group2}": float(np.median(b)),
        "mannwhitney_p": float(p),
        "per_patient_values": pb[["_group", "signature", "n_cells"]],
    }


# ---------------------------------------------------------------------------
# Paired pseudobulk (within-patient comparisons: CXCR3+/-, PB/CSF)
# ---------------------------------------------------------------------------
#
# IMPORTANT PREREQUISITE for PB-vs-CSF pairing specifically: this pipeline's
# data loader (data_io.py) builds each cell's `patient` value from a
# per-dataset "patient_key" that can ITSELF encode tissue (see
# _tissue_from_key, which detects tissue by checking whether the key starts
# or ends with "CSF"/"PB"/"PBMC"). If patient keys look like "P1_CSF" and
# "P1_PB", then adata.obs["patient"] holds a DIFFERENT literal string for the
# same person's PB sample vs their CSF sample, and pairing by `patient_col`
# directly will find zero overlap -- every pair gets silently dropped rather
# than erroring. `resolve_pb_csf_subject_ids` below checks for this before
# any paired PB/CSF test is run, rather than assuming either way.

def derive_subject_id(patient_key: str) -> str:
    """
    Best-effort strip of a tissue token (PB / CSF / PBMC, as prefix or
    suffix, matching data_io._tissue_from_key's detection logic) from a
    patient/sample key, to recover a tissue-independent subject identifier.

    This is a heuristic. It is only trustworthy after checking, via
    resolve_pb_csf_subject_ids, that it actually produces overlap between a
    person's PB and CSF samples -- do not use it standalone without that
    check.
    """
    import re
    key = patient_key
    patterns = [
        r'(?i)^pbmc[_\-]?', r'(?i)[_\-]?pbmc$',
        r'(?i)^csf[_\-]?', r'(?i)[_\-]?csf$',
        r'(?i)^pb[_\-]?', r'(?i)[_\-]?pb$',
    ]
    for pat in patterns:
        new_key = re.sub(pat, '', key)
        if new_key != key and new_key != '':
            return new_key
    return key


def resolve_pb_csf_subject_ids(
    adata,
    patient_col: str = "patient",
    tissue_col: str = "tissue",
    verbose: bool = True,
) -> pd.Series:
    """
    Determine which ID to use for pairing PB and CSF samples from the same
    subject, and print diagnostics so the result can be verified rather than
    trusted blindly.

    Strategy:
      1. Check whether `patient_col` values already overlap between PB and
         CSF (i.e. it's already tissue-independent). If any do, use it as-is
         -- this is the common case if patient keys don't encode tissue.
      2. If NOT (zero overlap), attempt to strip tissue tokens from each
         `patient_col` value (derive_subject_id) and check whether that
         produces overlap instead.
      3. If neither works, warn loudly and return the original column
         unchanged -- paired functions will then correctly report zero
         valid pairs (visible failure) rather than silently mismatching
         unrelated patients.

    Returns a pandas Series (same index as adata.obs) to pass as
    `patient_col` (after assigning it into adata.obs) to
    paired_pseudobulk_de / paired_pseudobulk_signature_test.
    """
    obs = adata.obs
    raw_ids = obs[patient_col].astype(str)
    tissues = obs[tissue_col].astype(str)

    def n_ids_with_both_tissues(ids: pd.Series) -> tuple[int, int]:
        tmp = pd.DataFrame({"id": ids.values, "tissue": tissues.values})
        per_id = tmp.groupby("id")["tissue"].apply(lambda t: set(t))
        both = per_id.apply(lambda s: {"PB", "CSF"}.issubset(s)).sum()
        return int(both), len(per_id)

    n_both_raw, n_total_raw = n_ids_with_both_tissues(raw_ids)
    if verbose:
        print(f"  [pairing check] '{patient_col}' as-is: "
              f"{n_both_raw}/{n_total_raw} distinct values have both PB and CSF cells")

    if n_both_raw > 0:
        return raw_ids

    derived_ids = raw_ids.apply(derive_subject_id)
    n_both_derived, n_total_derived = n_ids_with_both_tissues(derived_ids)
    if verbose:
        print(f"  [pairing check] '{patient_col}' does NOT pair across tissues as-is. "
              f"After stripping tissue tokens: {n_both_derived}/{n_total_derived} "
              f"derived subject IDs have both PB and CSF cells")

    if n_both_derived == 0:
        print(
            "  [WARNING] No subject IDs overlap between PB and CSF cells, even after "
            "stripping tissue tokens. Paired PB-vs-CSF tests will find ZERO valid pairs "
            "for every gene. Inspect adata.obs['" + patient_col + "'] values by hand and "
            "provide the correct tissue-independent subject ID."
        )
        return raw_ids

    if verbose:
        tmp = pd.DataFrame({"derived_subject_id": derived_ids.values, "tissue": tissues.values})
        per_subject_tissues = tmp.groupby("derived_subject_id")["tissue"].apply(lambda t: set(t))
        unpaired = per_subject_tissues[per_subject_tissues.apply(len) == 1]

        print(f"  [pairing check] {n_both_derived}/{n_total_derived} derived subject IDs have "
              f"BOTH tissues -- {len(unpaired)} do NOT:")
        if len(unpaired):
            for subj, tset in unpaired.items():
                print(f"    {subj}: only has {sorted(tset)[0]} -- no matching sample in the other tissue")
            print(f"  If a subject above is unexpectedly missing a tissue (i.e. you believe "
                  f"both a PB and CSF sample exist for them), check the earlier console output "
                  f"from load_dataset (data_io.py) for a '[skip] <path> not found' message or a "
                  f"'0 cells retained' B-cell gate result for that subject/tissue -- a listed "
                  f"file that failed to load or yielded zero B cells will silently vanish from "
                  f"pairing without appearing as an error here.")
        else:
            print(f"  All derived subject IDs pair cleanly across both tissues.")

        mapping = pd.DataFrame({
            "original": raw_ids.values, "tissue": tissues.values, "derived_subject_id": derived_ids.values,
        }).drop_duplicates().sort_values("derived_subject_id")
        print(f"  [pairing check] full derived mapping ({len(mapping)} rows) -- VERIFY this looks correct:")
        print(mapping.to_string(index=False))

    return derived_ids


def paired_pseudobulk_de(
    adata,
    groupby: str,
    group1: str,
    group2: str,
    patient_col: str = "patient",
    genes: list[str] | None = None,
    agg: str = "mean",
    use_raw: bool = True,
    min_pairs: int = 3,
    log2fc_from_log_means: bool = True,
) -> pd.DataFrame:
    """
    Paired, within-patient pseudobulk DE.

    Use this -- NOT pseudobulk_de -- when the same patient contributes cells
    to BOTH groups being compared: CXCR3+ vs CXCR3- cells from one patient,
    or PB vs CSF cells from one patient. (pseudobulk_de assumes one patient
    belongs to exactly one group, e.g. MS vs HC, and will raise an error here
    since it would be undefined which group a mixed patient "is".)

    Aggregates cells to one value per (patient, group), keeps only patients
    with both arms present, and runs a two-sided Wilcoxon signed-rank test
    per gene on the paired patient-level values. Patients missing one arm are
    dropped and counted in 'n_pairs'.

    log2fc_from_log_means: see pseudobulk_de -- True (default) is correct when
    the aggregated values are log1p (natural-log) expression, as pulled from
    `.raw` by default; set False only for linear-scale input.
    """
    from scipy.stats import wilcoxon

    genes = genes or adata.var_names.tolist()
    source = adata.raw if (use_raw and adata.raw is not None) else adata
    used_genes = [g for g in genes if g in source.var_names]

    mask = adata.obs[groupby].isin([group1, group2])
    sub_obs = adata.obs.loc[mask]
    X = source[mask.values][:, used_genes].X
    if sp.issparse(X):
        X = X.toarray()

    df = pd.DataFrame(np.asarray(X), columns=used_genes, index=sub_obs.index)
    df[patient_col] = sub_obs[patient_col].values
    df["_group"] = sub_obs[groupby].values

    agg_fn = "mean" if agg == "mean" else "sum"
    pb = df.groupby([patient_col, "_group"])[used_genes].agg(agg_fn)

    rows = []
    for g in used_genes:
        wide = pb[g].unstack("_group")
        wide = wide.dropna(subset=[c for c in [group1, group2] if c in wide.columns])
        n_pairs = len(wide)
        if n_pairs < min_pairs or group1 not in wide.columns or group2 not in wide.columns:
            rows.append({"gene": g, "n_pairs": n_pairs, "log2fc": np.nan, "pval": np.nan})
            continue
        x, y = wide[group1].values, wide[group2].values
        if np.all(x == y):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(x, y)

        if log2fc_from_log_means:
            log2fc = (y.mean() - x.mean()) / np.log(2)
        else:
            log2fc = np.log2((y.mean() + 1e-9) / (x.mean() + 1e-9))

        rows.append({
            "gene": g, "n_pairs": n_pairs,
            f"mean_{group1}": x.mean(), f"mean_{group2}": y.mean(),
            "log2fc": log2fc, "pval": p,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    valid = result["pval"].notna()
    result["padj"] = np.nan
    if valid.any():
        result.loc[valid, "padj"] = multipletests(result.loc[valid, "pval"], method="fdr_bh")[1]
    return result.sort_values("padj").reset_index(drop=True)


def paired_pseudobulk_signature_test(
    adata,
    groupby: str,
    group1: str,
    group2: str,
    signature_genes: list[str],
    patient_col: str = "patient",
    agg: str = "mean",
    use_raw: bool = True,
    min_pairs: int = 3,
    gene_signs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Paired, within-patient test of the composite zinc/metallothionein
    signature. Use for CXCR3+ vs CXCR3- or PB vs CSF (same patients in both
    arms) -- NOT for MS vs HC (use pseudobulk_signature_test there instead).

    Computes each patient's (group2 - group1) signature difference and runs a
    two-sided Wilcoxon signed-rank test against zero.

    gene_signs: optional {gene: +1 or -1} from compute_gene_signs. If given,
    each gene is sign-corrected before being averaged into the composite
    signature, so transporters and metallothioneins contribute in the same
    direction instead of partially cancelling. If not given, falls back to
    an unsigned mean (previous behavior).
    """
    from scipy.stats import wilcoxon

    genes = signature_genes
    source = adata.raw if (use_raw and adata.raw is not None) else adata
    used_genes = [g for g in genes if g in source.var_names]

    mask = adata.obs[groupby].isin([group1, group2])
    sub_obs = adata.obs.loc[mask]
    X = source[mask.values][:, used_genes].X
    if sp.issparse(X):
        X = X.toarray()

    df = pd.DataFrame(np.asarray(X), columns=used_genes, index=sub_obs.index)
    df[patient_col] = sub_obs[patient_col].values
    df["_group"] = sub_obs[groupby].values

    agg_fn = "mean" if agg == "mean" else "sum"
    # Aggregate per gene first (one value per patient x group x gene), THEN
    # combine genes into the composite signature -- this order is required
    # for sign-correction to apply per gene rather than to an already-mixed
    # per-cell average.
    pb_genes = df.groupby([patient_col, "_group"])[used_genes].agg(agg_fn)

    if gene_signs is not None:
        missing = [g for g in used_genes if g not in gene_signs]
        if missing:
            print(f"  [warn] paired_pseudobulk_signature_test: no sign specified for "
                  f"{missing}, treating as +1")
        sign_vec = np.array([gene_signs.get(g, 1.0) for g in used_genes])
        pb_signature = (pb_genes[used_genes].values * sign_vec[np.newaxis, :]).mean(axis=1)
    else:
        pb_signature = pb_genes[used_genes].mean(axis=1).values

    pb = pd.Series(pb_signature, index=pb_genes.index, name="signature")
    wide = pb.unstack("_group")
    wide = wide.dropna(subset=[c for c in [group1, group2] if c in wide.columns])

    n_pairs = len(wide)
    if n_pairs < min_pairs or group1 not in wide.columns or group2 not in wide.columns:
        print(f"  [warn] paired signature test: only {n_pairs} complete patient pairs -- too few to test.")
        return {"n_pairs": n_pairs, "wilcoxon_p": np.nan}

    x, y = wide[group1].values, wide[group2].values
    diff = y - x
    if np.all(diff == 0):
        stat, p = np.nan, 1.0
    else:
        stat, p = wilcoxon(x, y)

    return {
        "n_pairs": n_pairs,
        f"median_{group1}": float(np.median(x)),
        f"median_{group2}": float(np.median(y)),
        "median_diff": float(np.median(diff)),
        "wilcoxon_p": float(p),
        "per_patient_values": wide,
    }


# ---------------------------------------------------------------------------
# Mixed-effects alternative (retains all cells, models patient as random effect)
# ---------------------------------------------------------------------------

def mixed_model_de(
    adata,
    groupby: str,
    group1: str,
    group2: str,
    patient_col: str = "patient",
    genes: list[str] | None = None,
    use_raw: bool = True,
) -> pd.DataFrame:
    """
    Per-gene linear mixed model: expression ~ group, random intercept per
    patient. Uses all cells (unlike pseudobulk) but the random effect absorbs
    patient-to-patient correlation, so the group p-value reflects patient-level
    evidence rather than cell counts.

    Use as a sensitivity check alongside pseudobulk_de, not as a replacement --
    convergence is unreliable when a group has very few patients (e.g. CIS, n=2).
    When MixedLM fails to converge or returns a non-finite p-value, this falls
    back to OLS with cluster-robust standard errors (patient as cluster),
    which is simpler and more stable in that small-cluster regime; the
    'method' column in the output records which estimator was actually used
    for each gene.
    """
    import statsmodels.formula.api as smf

    genes = genes or adata.var_names.tolist()
    source = adata.raw if (use_raw and adata.raw is not None) else adata
    used_genes = [g for g in genes if g in source.var_names]

    mask = adata.obs[groupby].isin([group1, group2])
    sub_obs = adata.obs.loc[mask]
    sub_source = source[mask.values]

    rows = []
    for g in used_genes:
        expr = sub_source[:, g].X
        if sp.issparse(expr):
            expr = expr.toarray().ravel()
        else:
            expr = np.asarray(expr).ravel()

        df = pd.DataFrame({
            "expr": expr,
            "group": pd.Categorical(sub_obs[groupby].values, categories=[group1, group2]),
            "patient": sub_obs[patient_col].values,
        })

        coef, pval, converged, method_used = np.nan, np.nan, False, "mixedlm"
        try:
            md = smf.mixedlm("expr ~ group", df, groups=df["patient"])
            mdf = md.fit(reml=False, method="lbfgs")
            coef_names = [c for c in mdf.params.index if c.startswith("group")]
            if not coef_names:
                raise ValueError("no group coefficient fit (check factor levels)")
            coef_name = coef_names[0]
            coef, pval, converged = mdf.params[coef_name], mdf.pvalues[coef_name], mdf.converged
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] mixedlm failed for {g}: {exc}")

        # Fallback: MixedLM is prone to singular/non-converged fits with only
        # a handful of patients per group (exactly the regime here, e.g. CIS
        # n=2). Cluster-robust OLS (patient as cluster) is a simpler, more
        # stable alternative that still corrects standard errors for
        # within-patient correlation instead of assuming independent cells.
        if not np.isfinite(pval):
            try:
                ols = smf.ols("expr ~ group", df).fit(
                    cov_type="cluster", cov_kwds={"groups": df["patient"]}
                )
                coef_names = [c for c in ols.params.index if c.startswith("group")]
                coef_name = coef_names[0]
                coef, pval, converged = ols.params[coef_name], ols.pvalues[coef_name], True
                method_used = "cluster_robust_ols"
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] cluster-robust OLS fallback also failed for {g}: {exc}")

        rows.append({
            "gene": g, "coef": coef, "pval": pval,
            "converged": converged, "method": method_used,
        })

    result = pd.DataFrame(rows)
    valid = result["pval"].notna()
    result["padj"] = np.nan
    if valid.any():
        result.loc[valid, "padj"] = multipletests(result.loc[valid, "pval"], method="fdr_bh")[1]
    return result.sort_values("padj").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Competitive gene-set rank test (location + magnitude)
# ---------------------------------------------------------------------------
# Shared by fig2.py (MS vs HC) and fig3.py (PB vs CSF) -- moved here from
# fig3.py so both DE pipelines use the identical, validated implementation
# rather than maintaining two copies.

def competitive_rank_test(
    df: pd.DataFrame,
    label: str,
    min_background_genes: int = 20,
) -> dict:
    """
    Competitive gene-set rank test: are the curated zinc/MT genes, AS A SET,
    systematically shifted toward one end of the full log2FC ranking
    (compared to the other tested genes), more than would be expected from
    a random same-size gene set? Uses a two-sided (and both one-sided) Mann-
    Whitney U test comparing the zinc/MT genes' log2FC values against every
    other tested gene's log2FC values.

    WHY THIS IS DIFFERENT FROM, AND COMPLEMENTARY TO, THE PER-GENE AND
    COMPOSITE-SIGNATURE TESTS: it doesn't require any single gene to
    individually clear significance, or the panel average to move as a
    whole -- it only asks whether the PANEL'S RANK POSITION in the overall
    ranking is non-random. This can detect a real, coordinated but
    individually-weak shift that both the per-gene tests (each
    underpowered alone) and the composite-value test (vulnerable to
    direction-mixing / cancellation) can miss.

    *** CRITICAL INTERPRETATION CAVEAT, PRINTED BELOW: ***
    This is a COMPETITIVE test -- it compares the zinc panel's rank
    position AGAINST THE BACKGROUND GENE SET, not against zero. If the
    background gene set itself has a non-zero median log2FC (e.g. from a
    genuine global compositional/technical difference between the two
    conditions being compared -- common between PB and CSF scRNA-seq, where
    RNA content/complexity often differs systematically; less expected,
    but not guaranteed absent, between disease groups within the same
    tissue), a significant result means "the zinc panel deviates from the
    TYPICAL/BACKGROUND pattern," NOT "the zinc panel goes up (or down) in
    an absolute sense." Always report the background median alongside the
    test result, and phrase the conclusion relative to that background --
    e.g. "relatively preserved against a general decline" rather than
    "upregulated," if the zinc panel's own median log2FC is near zero
    while the background's is not.
    """
    from scipy.stats import mannwhitneyu

    zinc_fc = df.loc[df["is_zinc"] == True, "log2FC"].dropna().values
    bg_fc = df.loc[df["is_zinc"] == False, "log2FC"].dropna().values

    print(f"\n  === Competitive rank test: {label} ===")
    if len(zinc_fc) < 3 or len(bg_fc) < min_background_genes:
        print(f"    [skip] too few genes (zinc={len(zinc_fc)}, background={len(bg_fc)}) to test.")
        return {"label": label, "n_zinc": len(zinc_fc), "n_background": len(bg_fc),
                "zinc_median": np.nan, "background_median": np.nan,
                "pvalue_two_sided": np.nan, "pvalue_zinc_higher": np.nan, "pvalue_zinc_lower": np.nan}

    zinc_median = float(np.median(zinc_fc))
    bg_median = float(np.median(bg_fc))
    pct_bg_negative = 100 * (bg_fc < 0).mean()

    _, p_two = mannwhitneyu(zinc_fc, bg_fc, alternative="two-sided")
    _, p_higher = mannwhitneyu(zinc_fc, bg_fc, alternative="greater")
    _, p_lower = mannwhitneyu(zinc_fc, bg_fc, alternative="less")

    print(f"    n zinc genes = {len(zinc_fc)}, n background genes = {len(bg_fc)}")
    print(f"    median log2FC: zinc = {zinc_median:+.4f}, background = {bg_median:+.4f} "
          f"({pct_bg_negative:.0f}% of background genes negative)")
    print(f"    Two-sided: p={p_two:.4f}   |   zinc ranked HIGHER than background: p={p_higher:.4f}"
          f"   |   zinc ranked LOWER than background: p={p_lower:.4f}")

    if abs(bg_median) > 0.02 and p_two < 0.05:
        direction = "higher (less negative)" if zinc_median > bg_median else "lower (less positive)"
        print(f"    [INTERPRETATION] Background median is NOT near zero -- check whether this "
              f"reflects a global compositional/technical shift between the two conditions "
              f"being compared (see check_total_counts_pb_vs_csf-style diagnostics), rather than "
              f"many independent biological effects, before trusting this at face value. The "
              f"zinc panel ranks significantly {direction} than that background. Phrase this as "
              f"RELATIVE TO THE BACKGROUND PATTERN (e.g. 'relatively preserved/protected against "
              f"a general decline'), NOT as absolute up- or down-regulation, unless the zinc "
              f"panel's own median log2FC ({zinc_median:+.4f}) is itself clearly non-zero in the "
              f"same direction.")
    elif p_two < 0.05:
        print(f"    [INTERPRETATION] Background median is near zero, so this result CAN "
              f"reasonably be read as the zinc panel showing a real absolute shift, not just "
              f"relative preservation against a background artifact.")

    # ---- MAGNITUDE/DISPERSION TEST (complementary, not a replacement) -----
    # The test above asks about LOCATION/DIRECTION: is the panel shifted as
    # a whole toward one end of the ranking? It can miss a genuinely mixed-
    # direction pattern (some zinc genes up, some down) that cancels toward
    # the middle of that ranking even though the panel is clearly more (or
    # less) "responsive" to the comparison than a typical gene. This tests
    # |log2FC| instead of signed log2FC: does the zinc/MT panel show LARGER
    # (or smaller) absolute changes than background, regardless of
    # direction? A significant result here with a NON-significant location
    # test above is itself informative -- it points specifically at a
    # mixed-direction, coordinated-but-cancelling pattern, distinguishable
    # from "nothing is happening in this panel."
    zinc_abs = np.abs(zinc_fc)
    bg_abs = np.abs(bg_fc)
    zinc_abs_median = float(np.median(zinc_abs))
    bg_abs_median = float(np.median(bg_abs))
    _, p_mag_two = mannwhitneyu(zinc_abs, bg_abs, alternative="two-sided")
    _, p_mag_greater = mannwhitneyu(zinc_abs, bg_abs, alternative="greater")

    print(f"    --- Magnitude test (|log2FC|, direction-agnostic) ---")
    print(f"    median |log2FC|: zinc = {zinc_abs_median:.4f}, background = {bg_abs_median:.4f}")
    print(f"    Two-sided: p={p_mag_two:.4f}   |   zinc varies MORE than background: p={p_mag_greater:.4f}")

    if p_mag_greater < 0.05 and p_two >= 0.05:
        print(f"    [INTERPRETATION] The zinc/MT panel shows significantly LARGER changes than "
              f"background genes (in either direction), even though the DIRECTIONAL test above "
              f"was not significant. This is consistent with a mixed-direction, coordinated "
              f"pattern (some genes up, some down) that a location-only test cancels out -- "
              f"check the per-gene log2FC signs directly (not just this summary) before "
              f"concluding 'no effect' for this panel in {label}.")
    elif p_mag_greater < 0.05 and p_two < 0.05:
        print(f"    [INTERPRETATION] The zinc/MT panel is both directionally shifted AND more "
              f"variable than background -- consistent with a real, coordinated, single-direction "
              f"effect (not primarily a cancellation artifact).")
    elif p_mag_two < 0.05:
        print(f"    [INTERPRETATION] The zinc/MT panel is significantly LESS variable than "
              f"background (tighter |log2FC|) -- consistent with tighter regulatory control of "
              f"this gene set, rather than a coordinated shift. Verify this isn't itself an "
              f"artifact (e.g. background inflated by a few high-expression/technical genes) "
              f"before treating as a biological finding -- compare background |log2FC| with vs "
              f"without mitochondrial/ribosomal/housekeeping genes excluded.")

    return {
        "label": label, "n_zinc": len(zinc_fc), "n_background": len(bg_fc),
        "zinc_median": zinc_median, "background_median": bg_median,
        "pct_background_negative": pct_bg_negative,
        "pvalue_two_sided": p_two, "pvalue_zinc_higher": p_higher, "pvalue_zinc_lower": p_lower,
        "zinc_abs_median": zinc_abs_median, "background_abs_median": bg_abs_median,
        "pvalue_magnitude_two_sided": p_mag_two, "pvalue_magnitude_zinc_greater": p_mag_greater,
    }