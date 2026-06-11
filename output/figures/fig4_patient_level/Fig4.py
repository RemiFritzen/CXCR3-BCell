#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig4.py
Patient-level Figure 4 for CXCR3 / zinc project.

Main figure:
- A/B: zinc signature vs EDSS in CSF / PB
- C/D: HC vs CIS+MS boxplots with on-panel Mann-Whitney p-values

Supplementary figures:
- HC / CIS / MS boxplots for zinc signature in CSF / PB
- One PDF per gene, with 2 panels each: CSF vs EDSS and PB vs EDSS
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.stats import spearmanr, mannwhitneyu, kruskal

from config.config import FIG_DIRS, DATASETS_YAML, RESULTS_DIR
from data_io import load_cfg

sns.set(style="whitegrid", context="talk")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

TARGET_DATASETS = {"GSE133028", "GSE138266"}
DIAG_MAP = {"RRMS": "MS", "MS": "MS", "CIS": "CIS", "HC": "HC"}
EXCLUDE_TREATED = {"18", "26"}

PAL3 = {"HC": "#757575", "CIS": "#d95f02", "MS": "#1b9e77"}
PAL2 = {"HC": "#757575", "Disease": "#1b9e77"}


def load_markers(cfg: dict) -> list[str]:
    markers = cfg.get("markers", {})
    mt = markers.get("metallothioneins", [])
    zt = markers.get("zinc_transporters", [])
    artefacts = set(markers.get("exclude_artefacts", []))
    return [g for g in (mt + zt) if g not in artefacts]


def load_clinical_metadata() -> pd.DataFrame:
    rows = [
        ["1", 43, "F", "RRMS", 0, np.nan, "Untreated", 16],
        ["2", 33, "M", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["3", 27, "F", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["6", 44, "M", "RRMS", 2, np.nan, "Untreated", 1],
        ["7", 37, "M", "RRMS", 1.5, np.nan, "Untreated", 14],
        ["8", 32, "M", "RRMS", 1.5, np.nan, "Untreated", 4],
        ["9", 27, "F", "RRMS", 3.5, np.nan, "Untreated", 27],
        ["10", 45, "F", "RRMS", 2, np.nan, "Untreated", 3],
        ["16", 22, "M", "RRMS", 4, np.nan, "Untreated", 0],
        ["18", 29, "F", "RRMS", 2, np.nan, "Steroids", 32],
        ["22", 35, "F", "CIS", 4, np.nan, "Untreated", 1],
        ["24", 45, "F", "RRMS", 0, np.nan, "Untreated", 11],
        ["25", 42, "F", "RRMS", 1.5, np.nan, "Untreated", 4],
        ["26", 32, "F", "RRMS", 2, np.nan, "Steroids", 2],
        ["27", 37, "F", "RRMS", 1.5, np.nan, "Untreated", 166],
        ["28", 50, "F", "RRMS", 4, np.nan, "Untreated", 9],
        ["29", 54, "M", "RRMS", 2, np.nan, "Untreated", 49],
        ["31", 53, "F", "CIS", 2.5, np.nan, "Untreated", 3],
        ["32", 41, "M", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["PST95809", 43, "F", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["PST83775", 43, "M", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["PTC41540", 32, "F", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["PTC85037", 25, "F", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["PTC32190", 33, "M", "HC", np.nan, np.nan, "Untreated", np.nan],
        ["MS45044", 25, "F", "RRMS", 1.0, np.nan, "Untreated", np.nan],
        ["MS60249", 28, "M", "RRMS", 0.0, np.nan, "Untreated", np.nan],
        ["MS74594", 42, "M", "RRMS", 0.0, np.nan, "Untreated", np.nan],
        ["MS19270", 35, "F", "RRMS", 0.5, np.nan, "Untreated", np.nan],
        ["MS71658", 47, "F", "RRMS", 6.0, np.nan, "Untreated", np.nan],
        ["MS49131", 47, "F", "RRMS", 2.5, np.nan, "Untreated", np.nan],
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "patient", "Age", "Sex", "Diagnosis", "EDSS", "Latest_FU",
            "Treatment", "months_from_onset"
        ],
    )
    df["patient"] = df["patient"].astype(str)
    df["diagnosis_group"] = df["Diagnosis"].map(DIAG_MAP)
    return df


def harmonise_patient_num(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.copy()
    obs["patient"] = obs["patient"].astype(str)
    obs["dataset"] = obs["dataset"].astype(str)
    obs["patient_num"] = obs["patient"]

    mask028 = obs["dataset"] == "GSE133028"
    mask266 = obs["dataset"] == "GSE138266"

    obs.loc[mask028, "patient_num"] = (
        obs.loc[mask028, "patient"]
        .str.extract(r"(\d+)$", expand=False)
        .fillna(obs.loc[mask028, "patient"])
        .astype(str)
    )

    p266 = obs.loc[mask266, "patient"].str.split("_", n=1, expand=True)
    if p266.shape[1] == 1:
        coreid = p266[0].str.replace(r"PB|CSF", "", regex=True)
    else:
        tmp0 = p266[0].fillna("").str.replace(r"PB|CSF", "", regex=True)
        tmp1 = p266[1].fillna("").str.replace(r"PB|CSF", "", regex=True)
        coreid = tmp0.where(tmp0.ne(""), tmp1)
    obs.loc[mask266, "patient_num"] = coreid.astype(str)

    return obs


def get_expression_matrix(a: sc.AnnData, genes: list[str]) -> pd.DataFrame:
    present = [g for g in genes if g in a.var_names]
    missing = [g for g in genes if g not in a.var_names]
    if missing:
        warnings.warn(f"Missing genes skipped: {missing}")
    if not present:
        raise ValueError("None of the zinc genes are present in var_names.")
    X = a[:, present].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pd.DataFrame(X, columns=present, index=a.obs_names)


def get_patient_gene_means_from_raw(
    sub: sc.AnnData,
    gene: str,
    tissues: list[str] | None = None,
) -> pd.DataFrame:
    if sub.raw is None:
        raise ValueError("AnnData.raw is None; log-normalised layer not available.")
    if gene not in sub.raw.var_names:
        raise ValueError(f"{gene} not found in sub.raw.var_names.")

    obs = sub.obs[["patient_num", "tissue"]].copy()
    if tissues is not None:
        obs = obs[obs["tissue"].isin(tissues)].copy()
        sub_use = sub[obs.index].copy()
    else:
        sub_use = sub

    X = sub_use.raw[:, [gene]].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    df = pd.DataFrame(X, columns=[f"{gene}_raw"], index=sub_use.obs_names)
    df = obs.join(df)

    return (
        df.groupby(["patient_num", "tissue"])[f"{gene}_raw"]
        .mean()
        .reset_index()
    )


def _format_p(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def make_zinc_vs_edss_main_collapsed(out: pd.DataFrame, fig_dir) -> None:
    df = out.copy()
    df["EDSS"] = pd.to_numeric(df["EDSS"], errors="coerce")
    df = df[df["zinc_signature"].notna()].copy()
    df["diag_collapsed"] = (
        df["diagnosis_group"]
        .replace({"CIS": "Disease", "MS": "Disease"})
        .infer_objects(copy=False)
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey="row")
    fig.subplots_adjust(wspace=0.35, hspace=0.45)

    for ax, tissue, letter in zip(axes[0], ["CSF", "PB"], ["A", "B"]):
        tdf = df[(df["tissue"] == tissue) & df["EDSS"].notna()].copy()

        for grp, gdf in tdf.groupby("diagnosis_group"):
            ax.scatter(
                gdf["EDSS"], gdf["zinc_signature"], s=90,
                label=grp, color=PAL3.get(grp, "grey"), alpha=0.9,
                edgecolors="white", linewidths=1.2,
            )
            for _, r in gdf.iterrows():
                ax.text(
                    r["EDSS"], r["zinc_signature"], str(r["patient_num"]),
                    fontsize=8, ha="left", va="bottom"
                )

        rho = pval = np.nan
        if len(tdf) >= 3 and tdf["EDSS"].nunique() >= 2:
            rho, pval = spearmanr(tdf["EDSS"], tdf["zinc_signature"])
            z = np.polyfit(tdf["EDSS"], tdf["zinc_signature"], 1)
            xs = np.linspace(tdf["EDSS"].min(), tdf["EDSS"].max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), linestyle="--", color="black", alpha=0.6)

        if np.isfinite(rho):
            ax.set_title(
                f"{letter}. {tissue} B cells (ρ={rho:.2f}, p={_format_p(pval)})",
                fontsize=13, weight="bold"
            )
        else:
            ax.set_title(f"{letter}. {tissue} B cells", fontsize=13, weight="bold")
        ax.set_xlabel("EDSS", fontsize=11)

    axes[0, 0].set_ylabel("Mean zinc signature", fontsize=11)

    for ax, tissue, letter in zip(axes[1], ["CSF", "PB"], ["C", "D"]):
        bdf = df[
            (df["tissue"] == tissue) &
            (df["diag_collapsed"].isin(["HC", "Disease"]))
        ].copy()

        sns.boxplot(
            data=bdf, x="diag_collapsed", y="zinc_signature",
            order=["HC", "Disease"], hue="diag_collapsed",
            palette=PAL2, ax=ax, fliersize=0, legend=False,
        )
        sns.stripplot(
            data=bdf, x="diag_collapsed", y="zinc_signature",
            order=["HC", "Disease"], hue="diag_collapsed",
            palette=PAL2, dodge=False, ax=ax,
            linewidth=0.6, edgecolor="black", size=6, alpha=0.9, legend=False,
        )

        hc = bdf[bdf["diag_collapsed"] == "HC"]["zinc_signature"].dropna()
        dis = bdf[bdf["diag_collapsed"] == "Disease"]["zinc_signature"].dropna()

        pval = np.nan
        if len(hc) >= 1 and len(dis) >= 1:
            _, pval = mannwhitneyu(hc, dis, alternative="two-sided")

        ax.set_title(f"{letter}. {tissue} B cells", fontsize=13, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Mean zinc signature", fontsize=11)
        ax.text(
            0.5, 0.95, f"HC vs CIS+MS: p={_format_p(pval)}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, title="Diagnosis")

    fig.suptitle(
        "Figure 4. Zinc signature vs EDSS and disease status\nin PB and CSF B cells",
        fontsize=16, weight="bold", y=0.98,
    )

    png = fig_dir / "figure4_zinc_signature_EDSS_main_collapsed_pb_csf.png"
    pdf_fig = fig_dir / "figure4_zinc_signature_EDSS_main_collapsed_pb_csf.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(pdf_fig, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved main figure: {png}")
    print(f"Saved main figure: {pdf_fig}")


def make_zinc_boxplots_hc_cis_ms_supp(out: pd.DataFrame, fig_dir) -> None:
    df = out.copy()
    df = df[df["zinc_signature"].notna()].copy()
    box_df = df[df["diagnosis_group"].isin(["HC", "CIS", "MS"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(wspace=0.3)

    for ax, tissue in zip(axes, ["CSF", "PB"]):
        bdf = box_df[box_df["tissue"] == tissue].copy()

        sns.boxplot(
            data=bdf, x="diagnosis_group", y="zinc_signature",
            order=["HC", "CIS", "MS"], hue="diagnosis_group",
            palette=PAL3, ax=ax, fliersize=0, legend=False,
        )
        sns.stripplot(
            data=bdf, x="diagnosis_group", y="zinc_signature",
            order=["HC", "CIS", "MS"], hue="diagnosis_group",
            palette=PAL3, dodge=False, ax=ax,
            linewidth=0.6, edgecolor="black", size=6, alpha=0.9, legend=False,
        )

        groups = [
            bdf[bdf["diagnosis_group"] == g]["zinc_signature"].dropna()
            for g in ["HC", "CIS", "MS"]
        ]
        kw_p = np.nan
        if all(len(g) > 0 for g in groups):
            _, kw_p = kruskal(*groups)

        ax.set_title(
            f"{tissue} B cells (KW p={_format_p(kw_p)})",
            fontsize=13, weight="bold"
        )
        ax.set_xlabel("")
        ax.set_ylabel("Mean zinc signature", fontsize=11)

    fig.suptitle(
        "Supplementary: zinc signature by HC/CIS/MS diagnosis group",
        fontsize=15, weight="bold", y=1.02
    )

    png = fig_dir / "supp_zinc_signature_boxplots_HC_CIS_MS_pb_csf.png"
    pdf_fig = fig_dir / "supp_zinc_signature_boxplots_HC_CIS_MS_pb_csf.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(pdf_fig, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved supplementary boxplots: {png}")
    print(f"Saved supplementary boxplots: {pdf_fig}")


def build_gene_edss_table(
    sub: sc.AnnData,
    gene: str,
    tissue: str,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    try:
        gene_means = get_patient_gene_means_from_raw(
            sub=sub,
            gene=gene,
            tissues=[tissue],
        )
    except ValueError:
        return pd.DataFrame()

    gene_col = f"{gene}_raw"
    df = meta[meta["tissue"] == tissue].copy()
    df = df.merge(gene_means, on=["patient_num", "tissue"], how="inner")

    if gene_col not in df.columns:
        return pd.DataFrame()

    df["EDSS"] = pd.to_numeric(df["EDSS"], errors="coerce")
    df = df[df["EDSS"].notna() & df[gene_col].notna()].copy()
    return df


def _scatter_one_panel(ax, df: pd.DataFrame, gene_col: str, tissue_label: str) -> None:
    gene_name = gene_col.replace("_raw", "")

    for grp, gdf in df.groupby("diagnosis_group"):
        ax.scatter(
            gdf["EDSS"], gdf[gene_col], s=90,
            label=grp, color=PAL3.get(grp, "grey"), alpha=0.9,
            edgecolors="white", linewidths=1.2,
        )
        for _, r in gdf.iterrows():
            ax.text(
                r["EDSS"], r[gene_col], str(r["patient_num"]),
                fontsize=8, ha="left", va="bottom"
            )

    rho = pval = np.nan
    if len(df) >= 3 and df["EDSS"].nunique() >= 2:
        rho, pval = spearmanr(df["EDSS"], df[gene_col])
        z = np.polyfit(df["EDSS"], df[gene_col], 1)
        xs = np.linspace(df["EDSS"].min(), df["EDSS"].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), linestyle="--", color="black", alpha=0.6)

    title = f"{tissue_label} {gene_name}"
    if np.isfinite(rho):
        title += f" (ρ={rho:.2f}, p={_format_p(pval)})"
    title += f"\nN={len(df)}"

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("EDSS", fontsize=11)
    ax.set_ylabel(f"{tissue_label} {gene_name} (mean log-normalised expr)", fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, title="Diagnosis")


def make_gene_two_panel_pdf(
    sub: sc.AnnData,
    gene: str,
    meta: pd.DataFrame,
    fig_dir,
) -> None:
    csf_df = build_gene_edss_table(sub, gene, "CSF", meta)
    pb_df = build_gene_edss_table(sub, gene, "PB", meta)

    if csf_df.empty and pb_df.empty:
        print(f"Skipped {gene}: no EDSS-linked CSF or PB data")
        return

    gene_col = f"{gene}_raw"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.35)

    if not csf_df.empty:
        _scatter_one_panel(axes[0], csf_df, gene_col, "CSF")
    else:
        axes[0].set_title(f"CSF {gene}\nNo EDSS-linked data", fontsize=12, weight="bold")
        axes[0].set_xlabel("EDSS", fontsize=11)
        axes[0].set_ylabel(f"CSF {gene} (mean log-normalised expr)", fontsize=11)

    if not pb_df.empty:
        _scatter_one_panel(axes[1], pb_df, gene_col, "PB")
    else:
        axes[1].set_title(f"PB {gene}\nNo EDSS-linked data", fontsize=12, weight="bold")
        axes[1].set_xlabel("EDSS", fontsize=11)
        axes[1].set_ylabel(f"PB {gene} (mean log-normalised expr)", fontsize=11)

    fig.suptitle(f"{gene} vs EDSS in CSF and PB", fontsize=15, weight="bold", y=1.02)

    pdf_path = fig_dir / f"supp_{gene}_CSF_PB_vs_EDSS.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved gene PDF: {pdf_path}")


def main() -> None:
    cfg = load_cfg(DATASETS_YAML)
    genes = load_markers(cfg)

    fig_dir = FIG_DIRS["fig4"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    adata_path = RESULTS_DIR / "merged_bcells.h5ad"
    adata = sc.read_h5ad(adata_path)

    obs = adata.obs.copy()
    obs["dataset"] = obs["dataset"].astype(str)
    obs["tissue"] = obs["tissue"].astype(str)
    obs["condition"] = obs["condition"].astype(str)
    obs["cxcr3_group"] = obs["cxcr3_group"].astype(str)
    obs = harmonise_patient_num(obs)

    keep = (
        obs["dataset"].isin(TARGET_DATASETS)
        & obs["tissue"].isin(["PB", "CSF"])
        & obs["patient_num"].notna()
        & (~obs["patient_num"].isin(EXCLUDE_TREATED))
    )
    sub = adata[keep].copy()
    sub.obs = obs.loc[sub.obs_names].copy()

    expr = get_expression_matrix(sub, genes)
    df = sub.obs[["dataset", "patient_num", "tissue", "condition", "cxcr3_group"]].copy()
    df = df.join(expr)

    patient_means = (
        df.groupby(["dataset", "patient_num", "tissue"])[genes]
        .mean()
        .reset_index()
    )
    patient_means["zinc_signature"] = patient_means[genes].mean(axis=1)

    clinical = load_clinical_metadata()

    n_cells = (
        df.groupby(["patient_num", "tissue"])
        .size()
        .rename("n_cells")
        .reset_index()
    )

    out = patient_means.merge(n_cells, on=["patient_num", "tissue"], how="left")
    out = out.merge(
        clinical[["patient", "Diagnosis", "diagnosis_group", "EDSS", "months_from_onset"]],
        left_on="patient_num",
        right_on="patient",
        how="left",
    )

    out["EDSS"] = pd.to_numeric(out["EDSS"], errors="coerce")
    out["months_from_onset"] = pd.to_numeric(out["months_from_onset"], errors="coerce")
    out = out.sort_values(["tissue", "diagnosis_group", "dataset", "patient_num"])

    summary_path = fig_dir / "patient_level_zinc_edss_pb_csf.csv"
    out.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    make_zinc_vs_edss_main_collapsed(out, fig_dir)
    make_zinc_boxplots_hc_cis_ms_supp(out, fig_dir)

    meta = out[["patient_num", "diagnosis_group", "EDSS", "tissue"]].drop_duplicates()
    for gene in genes:
        make_gene_two_panel_pdf(sub=sub, gene=gene, meta=meta, fig_dir=fig_dir)


if __name__ == "__main__":
    main()
