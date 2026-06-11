#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
fig3_csf_pb_4panel.py — CXCR3+ vs CXCR3- zinc signature in HC and MS across CSF and PB

Figure 3:
- Panel A: HC, CSF, zinc/metallothionein genes, log2FC (CXCR3+ vs CXCR3-)
- Panel B: HC, PB,  zinc/metallothionein genes, log2FC (CXCR3+ vs CXCR3-)
- Panel C: MS, CSF, zinc/metallothionein genes, log2FC (CXCR3+ vs CXCR3-)
- Panel D: MS, PB,  zinc/metallothionein genes, log2FC (CXCR3+ vs CXCR3-)

Stars indicate significance based on padj:
    *   padj <= 0.05
    **  padj <= 0.01
    *** padj <= 0.001

Input:
- output/results/merged_bcells.h5ad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

from config.config import RESULTS_DIR, FIG_DIRS, DATASETS_YAML
from data_io import load_cfg

# -----------------------------------------------------------------------
# Global style (aligned with Fig1 and Fig2)
# -----------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
})


def load_zinc_gene_list() -> list[str]:
    """
    Load curated B-cell zinc + metallothionein genes from datasets.yaml.
    """
    cfg = load_cfg(DATASETS_YAML)
    markers = cfg.get("markers", {})
    zinc_transporters = markers.get("zinc_transporters", [])
    metallothioneins = markers.get("metallothioneins", [])
    artefact = set(markers.get("exclude_artefacts", []))
    zinc_genes_all = zinc_transporters + metallothioneins
    return [g for g in zinc_genes_all if g not in artefact]


def run_cxcr3pos_vs_neg(
    adata: sc.AnnData,
    tissue: str,
    condition: str,
    zinc_genes: list[str],
) -> pd.DataFrame:
    """
    Run DE for CXCR3+ vs CXCR3- within a given tissue and condition (HC or MS).

    Returns columns:
      gene, log2FC, pvalue, padj, is_zinc, tissue, condition

    log2FC is CXCR3+ vs CXCR3-.
    """
    sub = adata[
        (adata.obs["condition"] == condition)
        & (adata.obs["tissue"] == tissue)
        & adata.obs["cxcr3_group"].isin(["CXCR3+", "CXCR3-"])
    ].copy()

    if sub.n_obs == 0:
        raise ValueError(f"No {condition} {tissue} CXCR3+/- cells found.")

    groups = sub.obs["cxcr3_group"].unique().tolist()
    if not {"CXCR3+", "CXCR3-"}.issubset(set(groups)):
        raise ValueError(
            f"{tissue} {condition} subset missing CXCR3+ or CXCR3-: {groups}"
        )

    key = f"de_{tissue}_{condition}_CXCR3pos_vs_neg"
    sc.tl.rank_genes_groups(
        sub,
        groupby="cxcr3_group",
        groups=["CXCR3+"],
        reference="CXCR3-",
        method="wilcoxon",
        key_added=key,
    )

    df = sc.get.rank_genes_groups_df(
        sub,
        group="CXCR3+",
        key=key,
    )

    df = df.rename(
        columns={
            "names": "gene",
            "logfoldchanges": "log2FC",
            "pvals": "pvalue",
            "pvals_adj": "padj",
        }
    )

    df = df[np.isfinite(df["log2FC"])].copy()
    df["is_zinc"] = df["gene"].isin(zinc_genes)
    df["tissue"] = tissue
    df["condition"] = condition
    return df

def add_pseudocount_log2fc_for_plot(
    df: pd.DataFrame,
    adata: sc.AnnData,
    tissue: str,
    condition: str,
    pseudocount: float = 0.1,
) -> pd.DataFrame:
    """
    Compute a log2FC_plot using group means with a pseudocount:
        log2FC_plot = log2( (mean_CXCR3+ + pc) / (mean_CXCR3- + pc) )

    Used only for plotting; original log2FC, pvalue, padj remain unchanged.
    """
    sub = adata[
        (adata.obs["condition"] == condition)
        & (adata.obs["tissue"] == tissue)
        & adata.obs["cxcr3_group"].isin(["CXCR3+", "CXCR3-"])
    ]

    genes = df["gene"].values
    genes_in_adata = [g for g in genes if g in sub.var_names]
    if not genes_in_adata:
        return df

    cxcr3_plus = sub.obs["cxcr3_group"] == "CXCR3+"
    cxcr3_minus = sub.obs["cxcr3_group"] == "CXCR3-"

    X_plus = sub[cxcr3_plus][:, genes_in_adata].X
    X_minus = sub[cxcr3_minus][:, genes_in_adata].X
    X_plus = X_plus.toarray() if hasattr(X_plus, "toarray") else X_plus
    X_minus = X_minus.toarray() if hasattr(X_minus, "toarray") else X_minus

    mean_plus = np.asarray(X_plus.mean(axis=0)).ravel()
    mean_minus = np.asarray(X_minus.mean(axis=0)).ravel()

    stable = {}
    for g, mu_p, mu_m in zip(genes_in_adata, mean_plus, mean_minus):
        stable[g] = np.log2((mu_p + pseudocount) / (mu_m + pseudocount))

    df = df.copy()
    df["log2FC_plot"] = df["gene"].map(stable).fillna(df["log2FC"])
    return df


def select_genes_to_plot_4panel(
    dfs: list[pd.DataFrame],
    max_genes: int = 20,
    padj_thresh: float = 0.05,
) -> list[str]:
    """
    Pick a shared zinc gene set across all provided dfs
    so all panels use the same genes/order.

    Priority: significant zinc genes in any df, ranked by max absolute log2FC.
    """
    both = pd.concat(dfs, ignore_index=True)
    zinc = both[both["is_zinc"]].copy()
    if zinc.empty:
        return []

    zinc["sig"] = zinc["padj"].notna() & (zinc["padj"] <= padj_thresh)

    summary = (
        zinc.groupby("gene", as_index=False)
        .agg(
            any_sig=("sig", "max"),
            max_abs_log2FC=("log2FC", lambda x: np.max(np.abs(x))),
        )
        .sort_values(["any_sig", "max_abs_log2FC"], ascending=[False, False])
    )

    return summary["gene"].head(max_genes).tolist()


def prep_bar_df(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    out = df[df["gene"].isin(genes)].copy()
    out = out.drop_duplicates(subset=["gene"]).copy()
    out["gene"] = pd.Categorical(out["gene"], categories=genes, ordered=True)
    out = out.sort_values("gene")
    return out


def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return ""


def add_significance_stars(
    ax: plt.Axes,
    xvals: np.ndarray,
    yvals: pd.Series,
    pvals: pd.Series,
    y_offset_frac: float = 0.03,
) -> None:
    """
    Draw significance stars above/below bars based on p-values.
    """
    vals = pd.Series(yvals).astype(float)
    pv = pd.Series(pvals).astype(float)

    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.empty:
        return

    ymax = finite_vals.max()
    ymin = finite_vals.min()
    yrange = ymax - ymin
    if yrange == 0:
        yrange = max(abs(ymax), 1.0)

    offset = y_offset_frac * yrange

    for x, y, p in zip(xvals, vals, pv):
        if not np.isfinite(y):
            continue

        stars = p_to_stars(p)
        if stars == "":
            continue

        if y >= 0:
            y_text = y + offset
            va = "bottom"
        else:
            y_text = y - offset
            va = "top"

        ax.text(
            x,
            y_text,
            stars,
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
            color="black",
        )


def make_fig3_hc_ms_cxcr3pos_vs_neg_csf_pb_4panel() -> None:
    merged_path = RESULTS_DIR / "merged_bcells.h5ad"
    if not merged_path.exists():
        raise FileNotFoundError(f"merged_bcells.h5ad not found at {merged_path}")

    adata = sc.read_h5ad(merged_path)

    if not adata.obs_names.is_unique:
        print("[fix] Making obs_names unique in merged object for Fig3 4-panel")
        adata.obs_names_make_unique()

    required_cols = ["condition", "tissue", "cxcr3_group"]
    missing = [c for c in required_cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing required adata.obs columns: {missing}")

    zinc_genes = load_zinc_gene_list()

    # DE per condition/tissue
    df_csf_hc = run_cxcr3pos_vs_neg(adata, tissue="CSF", condition="HC", zinc_genes=zinc_genes)
    df_pb_hc = run_cxcr3pos_vs_neg(adata, tissue="PB", condition="HC", zinc_genes=zinc_genes)
    df_csf_ms = run_cxcr3pos_vs_neg(adata, tissue="CSF", condition="MS", zinc_genes=zinc_genes)
    df_pb_ms = run_cxcr3pos_vs_neg(adata, tissue="PB", condition="MS", zinc_genes=zinc_genes)

    df_csf_hc = add_pseudocount_log2fc_for_plot(
        df_csf_hc, adata, tissue="CSF", condition="HC", pseudocount=0.1
    )
    df_pb_hc = add_pseudocount_log2fc_for_plot(
        df_pb_hc, adata, tissue="PB", condition="HC", pseudocount=0.1
    )
    df_csf_ms = add_pseudocount_log2fc_for_plot(
        df_csf_ms, adata, tissue="CSF", condition="MS", pseudocount=0.1
    )
    df_pb_ms = add_pseudocount_log2fc_for_plot(
        df_pb_ms, adata, tissue="PB", condition="MS", pseudocount=0.1
    )


    print(f"Fig3 HC CSF: {df_csf_hc.shape[0]} genes in DE table.")
    print(f"Fig3 HC PB:  {df_pb_hc.shape[0]} genes in DE table.")
    print(f"Fig3 MS CSF: {df_csf_ms.shape[0]} genes in DE table.")
    print(f"Fig3 MS PB:  {df_pb_ms.shape[0]} genes in DE table.")

    # Shared zinc gene set across all four panels
    genes_for_plot = select_genes_to_plot_4panel(
        [df_csf_hc, df_pb_hc, df_csf_ms, df_pb_ms],
        max_genes=20,
        padj_thresh=0.05,
    )
    print("Fig3 4-panel shared zinc genes:", genes_for_plot)

    # Prepare plotting dfs
    df_bar_csf_hc = prep_bar_df(df_csf_hc, genes_for_plot)
    df_bar_pb_hc = prep_bar_df(df_pb_hc, genes_for_plot)
    df_bar_csf_ms = prep_bar_df(df_csf_ms, genes_for_plot)
    df_bar_pb_ms = prep_bar_df(df_pb_ms, genes_for_plot)

    fig3_dir = FIG_DIRS["fig3"]
    fig3_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axA, axB = axes[0]
    axC, axD = axes[1]

    x = np.arange(len(genes_for_plot))

    # Helper to get series for plotting and padj
    def get_vals(df_bar: pd.DataFrame):
        plot_df = df_bar.set_index("gene").reindex(genes_for_plot)
        col = "log2FC_plot" if "log2FC_plot" in plot_df.columns else "log2FC"
        vals = plot_df[col]
        padj = plot_df["padj"]
        return vals, padj

    vals_csf_hc, padj_csf_hc = get_vals(df_bar_csf_hc)
    vals_pb_hc, padj_pb_hc = get_vals(df_bar_pb_hc)
    vals_csf_ms, padj_csf_ms = get_vals(df_bar_csf_ms)
    vals_pb_ms, padj_pb_ms = get_vals(df_bar_pb_ms)

    # Panel A: HC CSF
    axA.bar(
        x,
        vals_csf_hc,
        width=0.7,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
    )
    axA.axhline(0, color="grey", linestyle="--", linewidth=1)
    axA.set_xticks(x)
    axA.set_xticklabels(genes_for_plot, rotation=45, ha="right")
    axA.set_ylabel("log2FC (CXCR3+ vs CXCR3-, HC)")
    axA.set_title("HC CSF CXCR3+ B cells – zinc/metallothionein genes")
    axA.text(-0.12, 1.05, "A", transform=axA.transAxes, fontsize=12, fontweight="bold")
    add_significance_stars(axA, x, vals_csf_hc, padj_csf_hc)

    # Panel B: HC PB
    axB.bar(
        x,
        vals_pb_hc,
        width=0.7,
        color="#F58518",
        edgecolor="black",
        linewidth=0.6,
    )
    axB.axhline(0, color="grey", linestyle="--", linewidth=1)
    axB.set_xticks(x)
    axB.set_xticklabels(genes_for_plot, rotation=45, ha="right")
    axB.set_ylabel("log2FC (CXCR3+ vs CXCR3-, HC)")
    axB.set_title("HC PB CXCR3+ B cells – zinc/metallothionein genes")
    axB.text(-0.12, 1.05, "B", transform=axB.transAxes, fontsize=12, fontweight="bold")
    add_significance_stars(axB, x, vals_pb_hc, padj_pb_hc)

    # Panel C: MS CSF
    axC.bar(
        x,
        vals_csf_ms,
        width=0.7,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
    )
    axC.axhline(0, color="grey", linestyle="--", linewidth=1)
    axC.set_xticks(x)
    axC.set_xticklabels(genes_for_plot, rotation=45, ha="right")
    axC.set_ylabel("log2FC (CXCR3+ vs CXCR3-, MS)")
    axC.set_title("MS CSF CXCR3+ B cells – zinc/metallothionein genes")
    axC.text(-0.12, 1.05, "C", transform=axC.transAxes, fontsize=12, fontweight="bold")
    add_significance_stars(axC, x, vals_csf_ms, padj_csf_ms)

    # Panel D: MS PB
    axD.bar(
        x,
        vals_pb_ms,
        width=0.7,
        color="#F58518",
        edgecolor="black",
        linewidth=0.6,
    )
    axD.axhline(0, color="grey", linestyle="--", linewidth=1)
    axD.set_xticks(x)
    axD.set_xticklabels(genes_for_plot, rotation=45, ha="right")
    axD.set_ylabel("log2FC (CXCR3+ vs CXCR3-, MS)")
    axD.set_title("MS PB CXCR3+ B cells – zinc/metallothionein genes")
    axD.text(-0.12, 1.05, "D", transform=axD.transAxes, fontsize=12, fontweight="bold")
    add_significance_stars(axD, x, vals_pb_ms, padj_pb_ms)

    fig.tight_layout()

    out_png = fig3_dir / "Fig3_HC_MS_CXCR3pos_vs_neg_CSF_PB_4panel_zinc_bars.png"
    out_pdf = fig3_dir / "Fig3_HC_MS_CXCR3pos_vs_neg_CSF_PB_4panel_zinc_bars.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # Export full DE tables and shared genes
    df_csf_hc.to_csv(fig3_dir / "Fig3_CSF_HC_CXCR3pos_vs_neg_full_DE.csv", index=False)
    df_pb_hc.to_csv(fig3_dir / "Fig3_PB_HC_CXCR3pos_vs_neg_full_DE.csv", index=False)
    df_csf_ms.to_csv(fig3_dir / "Fig3_CSF_MS_CXCR3pos_vs_neg_full_DE.csv", index=False)
    df_pb_ms.to_csv(fig3_dir / "Fig3_PB_MS_CXCR3pos_vs_neg_full_DE.csv", index=False)
    pd.DataFrame({"gene": genes_for_plot}).to_csv(
        fig3_dir / "Fig3_HC_MS_CXCR3pos_vs_neg_shared_zinc_genes.csv",
        index=False,
    )

    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    make_fig3_hc_ms_cxcr3pos_vs_neg_csf_pb_4panel()
