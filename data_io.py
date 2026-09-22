#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_io.py
----------
Data loading and gating helpers for the B-cell zinc / CXCR3 pipeline.

Public API
----------
load_cfg(yaml_path)                                      -- parse datasets.yaml
build_patient_path(dataset_name, patient_key, base_dir)  -- reconstruct .h5ad path
load_dataset(name, ds_cfg, base_dir, cfg)                -- load one dataset
load_all_datasets(cfg, base_dir)                         -- load and concatenate all
filter_conditions(adata, ds_cfg)                         -- apply keep_conditions
gate_bcells(adata, markers)                              -- gate B cells by markers
_resolve_gating(cfg, ds_cfg)                             -- resolve + report gate params
_uns_safe(obj)                                           -- make .uns records h5-writable
add_cxcr3_group(adata, cxcr3_gene)                      -- add cxcr3_group column
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import scanpy as sc
import yaml


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_cfg(yaml_path: str | Path) -> dict[str, Any]:
    """Load datasets.yaml and return the parsed dict."""
    with open(yaml_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def build_patient_path(
    dataset_name: str,
    patient_key: str,
    base_dir: str | Path = "data/processed",
) -> Path:
    """
    Reconstruct the .h5ad path from dataset name and patient key.
    Convention: <base_dir>/processed/<dataset_name>/<patient_key>.h5ad
    """
    return Path(base_dir) / "processed" / dataset_name / f"{patient_key}.h5ad"


def _resolve_h5ad_path(
    dataset_name: str,
    ds_cfg: dict[str, Any],
    base_dir: str | Path,
) -> Path:
    """
    Return the global .h5ad path for datasets that are not patient-based.
    Uses explicit 'h5ad' key if present, otherwise reconstructs from name.
    """
    if "h5ad" in ds_cfg:
        return Path(ds_cfg["h5ad"])
    return Path(base_dir) / dataset_name / f"{dataset_name}_processed.h5ad"


# ---------------------------------------------------------------------------
# Tissue extraction from patient key
# ---------------------------------------------------------------------------

def _tissue_from_key(patient_key: str) -> str:
    key_upper = patient_key.upper()
    if key_upper.startswith("CSF") or key_upper.endswith("_CSF"):
        return "CSF"
    if key_upper.startswith("PB") or key_upper.endswith("_PB") or "PBMC" in key_upper:
        return "PB"
    return "unknown"



def _normalise_tissue_labels(adata: ad.AnnData) -> ad.AnnData:
    """
    Harmonise tissue labels across datasets.
    PBMCs and PBMC are treated as PB.
    """
    if "tissue" not in adata.obs.columns:
        return adata

    adata.obs["tissue"] = (
        adata.obs["tissue"]
        .astype(str)
        .replace({
            "PBMCs": "PB",
            "PBMC": "PB",
            "blood": "PB",
            "peripheral blood": "PB",
        })
    )
    return adata

# ---------------------------------------------------------------------------
# Condition and cell-type filters
# ---------------------------------------------------------------------------

def filter_conditions(adata: ad.AnnData, ds_cfg: dict[str, Any]) -> ad.AnnData:
    """
    Retain only cells whose condition value is in keep_conditions.
    If keep_conditions is absent from the config, returns adata unchanged.
    """
    keep = ds_cfg.get("keep_conditions")
    if keep is None:
        return adata
    condition_col = ds_cfg.get("condition_col", "condition")
    if condition_col not in adata.obs.columns:
        print(f"  [warning] condition_col '{condition_col}' not found in .obs; skipping filter.")
        return adata
    mask = adata.obs[condition_col].isin(keep)
    return adata[mask].copy()

def _uns_safe(obj: Any) -> Any:
    """
    Coerce a nested structure into something anndata/h5py can actually write.

    .uns is written to HDF5, which accepts scalars, strings, arrays and
    (nested) groups -- but NOT None, and not a list of dicts. The gating
    record contains both, which raised:

        TypeError: Can't implicitly convert non-string objects to strings
        Error raised while writing key '.../removal_by_stratum'

    Rules applied here:
      * None                      -> key dropped (h5py has no null)
      * dict                      -> keys stringified, values recursed
      * DataFrame / list of dicts -> compact JSON string (round-trips via
                                     json.loads / pd.DataFrame, and stays
                                     human-readable in h5dump)
      * list/tuple of scalars     -> kept as a plain list
      * numpy scalar              -> native Python scalar
    """
    import json
    import numpy as np
    import pandas as pd

    if obj is None:
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.DataFrame):
        return json.dumps(obj.to_dict("records"), default=str)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            sv = _uns_safe(v)
            if sv is None:          # drop rather than write a null
                continue
            out[str(k)] = sv
        return out
    if isinstance(obj, (list, tuple)):
        items = list(obj)
        if any(isinstance(i, (dict, list, tuple)) for i in items):
            return json.dumps(items, default=str)
        return [i.item() if isinstance(i, np.generic) else i for i in items]
    return obj


def _resolve_gating(cfg: dict[str, Any] | None, ds_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve the B-cell gating parameters, loudly.

    HISTORY / WHY THIS EXISTS
    -------------------------
    The previous implementation read its marker lists from ``cfg["gating"]``:

        gating_cfg   = (cfg or {}).get("gating", {})
        global_b     = gating_cfg.get("bcell_markers", [])
        nonb_markers = gating_cfg.get("nonb_markers", [])

    config/datasets.yaml has no top-level ``gating:`` key -- ``bcell_markers``
    and ``nonb_markers`` live under ``markers:``. So both lookups returned
    empty, with two silent consequences:

      * the B gate fell back to its four hardcoded markers rather than the
        nine listed in the Methods, and
      * ``nonb_markers == []`` made ``if nonb_markers:`` false in gate_bcells,
        so step 2 (the top-30% non-B exclusion) never executed at all.

    Neither failure raised or warned. This function looks in both places, in
    order, and prints exactly what it resolved and from where, so the gate in
    force is visible in every run log rather than inferred from the source.

    IMPORTANT -- the non-B filter is now OPT-IN. Enabling it changes which
    cells enter every downstream test, so it is not switched on merely by
    fixing the lookup path. Set ``gating.nonb_filter.enabled: true`` in
    datasets.yaml to turn it on deliberately.

    Returns a dict with keys: b_markers, nonb_markers, nonb_enabled,
    nonb_keep_frac, nonb_scope, nonb_score_mode, provenance.
    """
    cfg = cfg or {}
    gating_cfg = cfg.get("gating", {}) or {}
    markers_cfg = cfg.get("markers", {}) or {}
    prov: list[str] = []

    # --- B-cell markers: dataset -> gating -> markers -> hardcoded fallback
    ds_b = ds_cfg.get("bcell_markers")
    if ds_b:
        b_markers, src = list(ds_b), "ds_cfg.bcell_markers"
    elif gating_cfg.get("bcell_markers"):
        b_markers, src = list(gating_cfg["bcell_markers"]), "cfg.gating.bcell_markers"
    elif markers_cfg.get("bcell_markers"):
        b_markers, src = list(markers_cfg["bcell_markers"]), "cfg.markers.bcell_markers"
    else:
        b_markers = ["CD19", "MS4A1", "CD79A", "CD79B"]
        src = "HARDCODED FALLBACK"
        print("  [warning] no bcell_markers found in config; using the 4-marker "
              "fallback. This is NOT the 9-marker gate described in the Methods.")
    prov.append(f"b_markers<-{src}")

    # --- Non-B markers: gating -> markers
    if gating_cfg.get("nonb_markers"):
        nonb_markers, src = list(gating_cfg["nonb_markers"]), "cfg.gating.nonb_markers"
    elif markers_cfg.get("nonb_markers"):
        nonb_markers, src = list(markers_cfg["nonb_markers"]), "cfg.markers.nonb_markers"
    else:
        nonb_markers, src = [], "NOT FOUND"
    prov.append(f"nonb_markers<-{src}")

    # --- Filter settings (opt-in) ----------------------------------------
    nf = gating_cfg.get("nonb_filter", {}) or {}
    enabled = bool(nf.get("enabled", False))
    keep_frac = nf.get("keep_frac", nf.get("nonb_pct", 0.7))
    scope = nf.get("scope", "dataset")
    score_mode = nf.get("score_mode", "mean_raw")

    if enabled and not nonb_markers:
        raise ValueError(
            "gating.nonb_filter.enabled is true but no nonb_markers were found "
            "in the config. Refusing to silently skip the filter -- this is the "
            "exact failure mode that made the published gate ambiguous."
        )
    if not 0.0 < float(keep_frac) <= 1.0:
        raise ValueError(f"gating.nonb_filter.keep_frac must be in (0, 1]; got {keep_frac}")
    if scope not in {"dataset", "tissue", "sample"}:
        raise ValueError(f"gating.nonb_filter.scope must be dataset|tissue|sample; got {scope}")
    if score_mode not in {"mean_raw", "frac_of_total"}:
        raise ValueError(
            f"gating.nonb_filter.score_mode must be mean_raw|frac_of_total; got {score_mode}"
        )

    return {
        "b_markers": b_markers,
        "nonb_markers": nonb_markers,
        "nonb_enabled": enabled,
        "nonb_keep_frac": float(keep_frac),
        "nonb_scope": scope,
        "nonb_score_mode": score_mode,
        "provenance": "; ".join(prov),
    }


def _compute_nonb_score(adata: ad.AnnData, present_nb: list[str], mode: str):
    """
    Per-cell non-B contamination score.

    mode="mean_raw"      mean raw count across non-B markers. This is what the
                         original implementation used. Note it is computed on
                         RAW counts before normalisation, so it scales with
                         library depth: deeper cells score higher independent
                         of any actual contamination, and the filter therefore
                         partly acts as a "deep library" filter.
    mode="frac_of_total" the same sum expressed as a fraction of the cell's
                         total counts, which removes that depth confound. Not
                         the default, because switching it changes which cells
                         are dropped; use it as a sensitivity check.
    """
    import numpy as np
    import scipy.sparse as sp

    idx_nb = [adata.var_names.get_loc(g) for g in present_nb]
    Xnb = adata.X[:, idx_nb]
    if sp.issparse(Xnb):
        Xnb = Xnb.toarray()
    Xnb = np.asarray(Xnb)

    if mode == "mean_raw":
        return Xnb.mean(axis=1)

    total = adata.X.sum(axis=1)
    total = np.asarray(total).ravel().astype(float)
    total[total == 0] = np.nan
    return Xnb.sum(axis=1) / total


def gate_bcells(
    adata: ad.AnnData,
    b_markers: list[str],
    nonb_markers: list[str] | None = None,
    nonb_keep_frac: float = 0.7,
    nonb_scope: str = "dataset",
    nonb_score_mode: str = "mean_raw",
    nonb_pct: float | None = None,
) -> ad.AnnData:
    """
    Two-step B-cell gate.

    1) Keep cells with any B marker expressed (raw count > 0).
    2) If nonb_markers is provided: compute a non-B score and drop the cells
       with the highest scores, keeping the bottom `nonb_keep_frac`.

    Parameters
    ----------
    nonb_keep_frac:
        Fraction of B-marker-positive cells KEPT. 0.7 keeps the 70% least
        non-B-like and drops the top 30%. (The old parameter was named
        `nonb_pct` with the same meaning, but its docstring said "default 0.9"
        while the signature default was 0.2 and the call site passed 0.7 --
        three different numbers for one parameter. `nonb_pct` is still
        accepted as a deprecated alias.)

        This value is a QUANTILE, not a threshold: it removes exactly
        (1 - nonb_keep_frac) of the cells whether or not any of them are
        contaminated. At 0.7 that is ~30%, which is far above the expected
        10x multiplet rate (~4-8% at these loadings), so it should be
        described as a conservative purity filter and not as doublet removal.

    nonb_scope:
        Population over which the quantile is computed.
        "dataset" -- one threshold for the whole dataset, pooling PB + CSF and
                     MS + HC. This is the original behaviour. It removes
                     exactly (1 - keep_frac) of the dataset, but NOT
                     necessarily of each tissue or condition: if ambient non-B
                     signal is higher in CSF, more CSF cells are dropped, and
                     the gate becomes a compartment-biased filter sitting
                     upstream of the PB-vs-CSF comparison.
        "tissue"  -- one threshold per tissue, so PB and CSF each lose the
                     same fraction by construction.
        "sample"  -- one threshold per patient x tissue sample.

    nonb_score_mode:
        See _compute_nonb_score. "mean_raw" reproduces the original.

    Notes
    -----
    Per-stratum removal rates are printed, and the full gating record is
    stored in `adata.uns["bcell_gate"]`, so the gate in force is recoverable
    from any saved object rather than only from the source at the time.
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    if nonb_pct is not None:
        print("  [deprecated] gate_bcells(nonb_pct=...) -- use nonb_keep_frac; "
              "same meaning (fraction KEPT).")
        nonb_keep_frac = nonb_pct

    record: dict[str, Any] = {
        "b_markers_requested": list(b_markers),
        "nonb_markers_requested": list(nonb_markers or []),
        "nonb_keep_frac": float(nonb_keep_frac),
        "nonb_scope": nonb_scope,
        "nonb_score_mode": nonb_score_mode,
        "n_cells_input": int(adata.n_obs),
    }

    # --- Step 1: any B marker > 0 ----------------------------------------
    present_b = [g for g in b_markers if g in adata.var_names]
    missing_b = [g for g in b_markers if g not in adata.var_names]
    if missing_b:
        print(f"  [warning] B markers absent from var_names: {missing_b}")
    if not present_b:
        print(f"  [warning] No B markers present among {b_markers}; skipping B gate.")
        record["step1_applied"] = False
        adata.uns["bcell_gate"] = _uns_safe(record)
        return adata

    idx_b = [adata.var_names.get_loc(g) for g in present_b]
    Xb = adata.X[:, idx_b]
    if sp.issparse(Xb):
        expr_b = np.asarray(Xb.sum(axis=1)).ravel()
    else:
        expr_b = np.asarray(Xb).sum(axis=1)
    keep_b = expr_b > 0

    print(f"  Step1: {keep_b.sum()} / {adata.n_obs} cells with any B marker > 0 "
          f"({len(present_b)} markers used)")
    record.update(step1_applied=True, b_markers_used=present_b,
                  n_cells_after_step1=int(keep_b.sum()))

    adata_sub = adata[keep_b].copy()

    # --- Step 2: optional non-B filter -----------------------------------
    if not nonb_markers:
        print("  Step2: SKIPPED (no non-B markers supplied) -- all "
              f"{adata_sub.n_obs} B-marker-positive cells retained.")
        record.update(step2_applied=False, step2_skip_reason="no nonb_markers")
    elif nonb_keep_frac >= 1.0:
        print("  Step2: SKIPPED (nonb_keep_frac >= 1.0, filter disabled).")
        record.update(step2_applied=False, step2_skip_reason="keep_frac>=1")
    else:
        present_nb = [g for g in nonb_markers if g in adata_sub.var_names]
        missing_nb = [g for g in nonb_markers if g not in adata_sub.var_names]
        if missing_nb:
            print(f"  [warning] non-B markers absent from var_names: {missing_nb}")
        if not present_nb:
            print(f"  Step2: SKIPPED -- none of the non-B markers {nonb_markers} found.")
            record.update(step2_applied=False, step2_skip_reason="no nonb markers present")
        else:
            score = _compute_nonb_score(adata_sub, present_nb, nonb_score_mode)
            adata_sub.obs["NonB_score"] = score

            # Quantile scope
            if nonb_scope == "dataset":
                strata = pd.Series(["_all"] * adata_sub.n_obs, index=adata_sub.obs_names)
            elif nonb_scope == "tissue":
                strata = adata_sub.obs.get("tissue", pd.Series(
                    ["_all"] * adata_sub.n_obs, index=adata_sub.obs_names)).astype(str)
            else:  # "sample"
                pat = adata_sub.obs.get("patient", pd.Series(
                    ["_p"] * adata_sub.n_obs, index=adata_sub.obs_names)).astype(str)
                tis = adata_sub.obs.get("tissue", pd.Series(
                    ["_t"] * adata_sub.n_obs, index=adata_sub.obs_names)).astype(str)
                strata = pat.str.cat(tis, sep="|")

            keep_nb = np.ones(adata_sub.n_obs, dtype=bool)
            thresholds: dict[str, float] = {}
            for s in pd.unique(strata):
                m = (strata.values == s)
                thr = float(np.nanquantile(score[m], nonb_keep_frac))
                thresholds[str(s)] = thr
                keep_nb[m] = score[m] <= thr

            n_removed = int((~keep_nb).sum())
            print(f"  Step2: removing {n_removed} / {adata_sub.n_obs} cells "
                  f"({100 * n_removed / max(adata_sub.n_obs, 1):.1f}%) with high "
                  f"non-B score [scope={nonb_scope}, mode={nonb_score_mode}, "
                  f"keep_frac={nonb_keep_frac}]")

            # --- Per-stratum removal rates, printed every run -------------
            # This is the diagnostic a reviewer will ask for: a filter that
            # removes an even fraction from each arm cannot have produced a
            # compartment or disease effect on its own.
            audit_cols = [c for c in ("tissue", "condition") if c in adata_sub.obs.columns]
            removal_table = None
            if audit_cols:
                aud = adata_sub.obs[audit_cols].copy()
                aud["_removed"] = ~keep_nb
                removal_table = (
                    aud.groupby(audit_cols, observed=True)["_removed"]
                    .agg(n="size", n_removed="sum")
                    .assign(pct_removed=lambda d: 100 * d["n_removed"] / d["n"])
                    .reset_index()
                )
                print("    removal rate by stratum:")
                for line in removal_table.to_string(index=False).splitlines():
                    print(f"      {line}")
                spread = removal_table["pct_removed"]
                if len(spread) > 1 and (spread.max() - spread.min()) > 5.0:
                    print(f"    [warning] removal rate differs by "
                          f"{spread.max() - spread.min():.1f} percentage points "
                          f"across strata -- this filter is not arm-neutral. "
                          f"Consider nonb_scope='tissue'.")

            adata_sub = adata_sub[keep_nb].copy()
            record.update(
                step2_applied=True,
                nonb_markers_used=present_nb,
                thresholds=thresholds,
                n_removed=n_removed,
                # JSON string, not a list of dicts: see _uns_safe. Read back
                # with pd.DataFrame(json.loads(uns["removal_by_stratum_json"])).
                removal_by_stratum_json=removal_table,
            )

    record["n_cells_final"] = int(adata_sub.n_obs)
    adata_sub.uns["bcell_gate"] = _uns_safe(record)
    print(f"  Final B-gate: {adata_sub.n_obs} cells retained.")

    return adata_sub


def add_cxcr3_group(adata: ad.AnnData, cxcr3_gene: str = "CXCR3") -> ad.AnnData:
    """
    Add obs column 'cxcr3_group' with values 'CXCR3+' or 'CXCR3-'.
    """
    import numpy as np
    import scipy.sparse as sp

    if cxcr3_gene not in adata.var_names:
        print(f"  [warning] '{cxcr3_gene}' not in var_names; setting all cells to 'CXCR3-'.")
        adata.obs["cxcr3_group"] = "CXCR3-"
        return adata
    idx = adata.var_names.get_loc(cxcr3_gene)
    col = adata.X[:, idx]
    if sp.issparse(col):
        expr = np.asarray(col.todense()).ravel()
    else:
        expr = np.asarray(col).ravel()
    adata.obs["cxcr3_group"] = ["CXCR3+" if v > 0 else "CXCR3-" for v in expr]
    return adata


# ---------------------------------------------------------------------------
# Standardise .obs schema
# ---------------------------------------------------------------------------

def _standardise_obs(
    adata: ad.AnnData,
    dataset_name: str,
    ds_cfg: dict[str, Any],
    patient_key: str | None = None,
    patient_meta: dict[str, Any] | None = None,
) -> ad.AnnData:
    """
    Guarantee a common set of .obs columns across all datasets:
        dataset, patient, condition, tissue, sex
    """
    adata.obs["dataset"] = dataset_name

    condition_col = ds_cfg.get("condition_col", "condition")
    patient_col   = ds_cfg.get("patient_col",   "patient")
    tissue_col    = ds_cfg.get("tissue_col",    "tissue")
    print(patient_key)
    if patient_key is not None:
        adata.obs["patient"] = patient_key
    elif patient_col in adata.obs.columns:
        adata.obs["patient"] = adata.obs[patient_col].astype(str)
    else:
        adata.obs["patient"] = "unknown"

    if patient_meta and "condition" in patient_meta:
        adata.obs["condition"] = patient_meta["condition"]
    elif condition_col in adata.obs.columns:
        adata.obs["condition"] = adata.obs[condition_col].astype(str)
    else:
        adata.obs["condition"] = "unknown"

    # Sex is patient-level metadata from datasets.yaml only (no source .h5ad
    # column fallback -- neither dataset embeds sex in the raw object), added
    # for the sex-stratified sensitivity analysis (reviewer point #11).
    if patient_meta and "sex" in patient_meta:
        adata.obs["sex"] = patient_meta["sex"]
    else:
        adata.obs["sex"] = "unknown"

    if patient_key is not None:
        adata.obs["tissue"] = _tissue_from_key(patient_key)
    elif tissue_col in adata.obs.columns:
        adata.obs["tissue"] = adata.obs[tissue_col].astype(str)
    else:
        adata.obs["tissue"] = "unknown"

    return adata


# ---------------------------------------------------------------------------
# Dataset-level loader
# ---------------------------------------------------------------------------

def load_dataset(
    name: str,
    ds_cfg: dict[str, Any],
    base_dir: str | Path = "data/processed",
    cfg: dict[str, Any] | None = None,
    gate_b: bool = True,
    add_cxcr3: bool = True,
    gating_override: dict[str, Any] | None = None,
) -> ad.AnnData:
    """
    Load a single dataset entry from the config.

    - If 'patients' is present: loops over patient files, concatenates.
    - Otherwise: loads the single global .h5ad.
    Both paths produce the same standardised .obs schema.

    gating_override: optional dict merged over the resolved gating settings
    (keys as returned by _resolve_gating, e.g. {"nonb_keep_frac": 0.8}).
    Intended for sensitivity sweeps, so a threshold can be varied without
    editing the config or duplicating this loader.
    """
    markers_cfg   = (cfg or {}).get("markers", {})

    # Gating parameters are resolved in one place and printed, rather than
    # being assembled from lookups that fail silently. See _resolve_gating.
    gating = _resolve_gating(cfg, ds_cfg)
    if gating_override:
        gating = {**gating, **gating_override}
        print(f"  [gating] overridden in-call: {sorted(gating_override)}")

    cxcr3_gene    = markers_cfg.get("cxcr3", "CXCR3")

    patients = ds_cfg.get("patients")

    if patients:
        adatas: list[ad.AnnData] = []
        for patient_key, patient_meta in patients.items():
            path = build_patient_path(name, patient_key, base_dir)
            if not path.exists():
                print(f" [skip] {path} not found.")
                continue
            print(f"  Loading {path} ...")
            adata = sc.read_h5ad(path)

            # Ensure unique obs_names at source
            if not adata.obs_names.is_unique:
                print(f"  [fix] Making obs_names unique for {path.name}")
                adata.obs_names_make_unique()

            adata = _standardise_obs(adata, name, ds_cfg, patient_key, patient_meta)
            adatas.append(adata)

        if not adatas:
            raise FileNotFoundError(f"No patient files loaded for dataset '{name}'.")
        combined = ad.concat(adatas, label="patient_file", join="outer")
        combined = filter_conditions(combined, ds_cfg)

    else:
        # Global file mode
        path = _resolve_h5ad_path(name, ds_cfg, base_dir)
        if not path.exists():
            raise FileNotFoundError(f"Global h5ad not found: {path}")
        print(f"  Loading {path} ...")
        combined = sc.read_h5ad(path)


        if name == "GSE239626":
            print("  [diag] obs columns:", combined.obs.columns.tolist())
            print("  [diag] head of obs:")
            print(combined.obs.head())
        if not combined.obs_names.is_unique:
            print(f"  [fix] Making obs_names unique for {path.name}")
            combined.obs_names_make_unique()

        # Standardise obs first
        combined = _standardise_obs(combined, name, ds_cfg)


        # Now apply keep_conditions
        combined = filter_conditions(combined, ds_cfg)

    

    if gate_b:
        # The non-B filter is opt-in: passing nonb_markers=None when it is
        # disabled means gate_bcells reports "Step2: SKIPPED" explicitly in
        # the log, instead of the old behaviour where an empty marker list
        # skipped the step with no output at all.
        combined = gate_bcells(
            combined,
            b_markers=gating["b_markers"],
            nonb_markers=(gating["nonb_markers"] if gating["nonb_enabled"] else None),
            nonb_keep_frac=gating["nonb_keep_frac"],
            nonb_scope=gating["nonb_scope"],
            nonb_score_mode=gating["nonb_score_mode"],
        )
        if "bcell_gate" in combined.uns:
            combined.uns["bcell_gate"]["config_provenance"] = gating["provenance"]
            combined.uns["bcell_gate"]["nonb_filter_enabled"] = gating["nonb_enabled"]

    if add_cxcr3:
        combined = add_cxcr3_group(combined, cxcr3_gene)

    
    print(f"  {name}: {combined.n_obs} cells retained.")
    return combined


# ---------------------------------------------------------------------------
# Multi-dataset loader
# ---------------------------------------------------------------------------

def load_all_datasets(
    cfg: dict[str, Any],
    base_dir: str | Path = "data/processed",
    gate_b: bool = True,
    add_cxcr3: bool = True,
    gating_override: dict[str, Any] | None = None,
) -> ad.AnnData:
    """
    Loop over all datasets in cfg, load each, and return one merged AnnData.
    Stable .obs columns: dataset, patient, condition, tissue, sex, cxcr3_group.

    gating_override is forwarded to load_dataset; see _resolve_gating. The
    gate is applied PER DATASET, before merging, so any non-B quantile is
    computed within a dataset, not across the pooled object.
    """
    datasets_cfg = cfg.get("datasets", {})
    adatas: list[ad.AnnData] = []

    for name, ds_cfg in datasets_cfg.items():
        print(f"\n[{name}]")
        try:
            adata = load_dataset(
                name=name,
                ds_cfg=ds_cfg,
                base_dir=base_dir,
                cfg=cfg,
                gate_b=gate_b,
                add_cxcr3=add_cxcr3,
                gating_override=gating_override,
            )
            adatas.append(adata)
        except FileNotFoundError as exc:
            print(f"  [error] {exc} -- dataset skipped.")

    if not adatas:
        raise RuntimeError("No datasets were loaded successfully.")

    merged = ad.concat(adatas, label="source_dataset", join="outer")
    merged = _normalise_tissue_labels(merged)
    # ad.concat drops .uns; re-attach the per-dataset gating records so the
    # gate that produced a saved object is recoverable from the object.
    gate_records = {}
    for i, a in enumerate(adatas):
        rec = a.uns.get("bcell_gate")
        if rec is None:
            continue
        key = (str(a.obs["dataset"].iloc[0])
               if "dataset" in a.obs.columns and a.n_obs else str(i))
        gate_records[key] = rec
    if gate_records:
        merged.uns["bcell_gate_by_dataset"] = _uns_safe(gate_records)
    print(f"\nTotal cells after merge: {merged.n_obs}")
    return merged