#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
fig2.py (revised) — PB + CSF CXCR3+ MS vs HC zinc signature (4-panel figure)

Changes vs original:
  1.  Panels A & C: replaced rank-scatter with VOLCANO PLOTS (log2FC vs -log10(padj)).
      Volcano is the field standard for DE; shows both effect size AND significance;
      fixes the y-axis domination by extreme outliers; separates zinc genes visually.
  2.  Panels B & D: added significance asterisks (* / ** / ***) above each bar.
  3.  Panel D: generalised outlier-cap logic into _draw_bars() via `outlier_cap`
      parameter; cleaner annotation with hatching on capped bar.
  4.  CRITICAL: SLC39A14 log2FC=24.4 in CSF is almost certainly a numerical
      artefact (near-zero HC mean passing a loose 10% filter).  A diagnostic
      helper _check_gene_expression() is added; and min_pct is raised to 0.15
      for CSF (fewer cells → sparser data → more extreme spurious FCs).
  5.  Gene labels in volcano are placed via adjustText when available, otherwise
      a smart fallback filters to zinc genes with padj < 0.05.
  6.  pdf.fonttype=42 → text editable in Illustrator / Inkscape.
  7.  Gene labels were absent from _draw_scatter in the original code even though
      they appear in the PDF — added explicitly here.
  8.  _select_genes_for_bars now returns a `used_fallback` flag so the title can
      warn the reader when no gene passed padj < 0.05.
  9.  Unified rcParams; consistent wspace/hspace.
"""

import scipy.sparse as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import mannwhitneyu

from config.config import RESULTS_DIR, FIG_DIRS, DATASETS_YAML
from data_io import load_cfg


# ── Global style ────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "pdf.fonttype":       42,   # editable text in Illustrator / Inkscape
    "ps.fonttype":        42,
})

SCATTER_COLOR_OTHER    = "#CCCCCC"
SCATTER_COLOR_ZINC_PB  = "#1f77b4"
SCATTER_COLOR_ZINC_CSF = "#E45756"
BAR_COLOR_PB           = "#4C78A8"
BAR_COLOR_CSF          = "#E45756"
SIG_PADJ_THRESHOLD     = 0.05
PANEL_KW               = dict(fontsize=14, fontweight="bold", va="top", ha="left")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes, **PANEL_KW)


def load_zinc_gene_list() -> list[str]:
    cfg = load_cfg(DATASETS_YAML)
    markers  = cfg.get("markers", {})
    artefact = set(markers.get("exclude_artefacts", []))
    all_g    = markers.get("zinc_transporters", []) + markers.get("metallothioneins", [])
    return [g for g in all_g if g not in artefact]


def _apply_min_pct_filter(sub: sc.AnnData, min_pct: float) -> sc.AnnData:
    X      = sub.X if not sp.issparse(sub.X) else sub.X.toarray()
    ms_m   = sub.obs["condition"] == "MS"
    hc_m   = sub.obs["condition"] == "HC"
    pct_ms = (X[ms_m] > 0).mean(axis=0)
    pct_hc = (X[hc_m] > 0).mean(axis=0)
    keep   = (pct_ms >= min_pct) & (pct_hc >= min_pct)
    print(f"    min_pct={min_pct:.0%}: {keep.sum()}/{len(keep)} genes retained")
    return sub[:, keep].copy()


def _check_gene_expression(
    adata: sc.AnnData,
    tissue: str,
    gene: str,
) -> None:
    """
    Diagnostic: prints per-condition expression stats for one gene.

    Run this whenever |log2FC| > 5 to check for numerical artefacts
    caused by near-zero mean in one group passing a loose pct filter.
    """
    sub = adata[
        (adata.obs["tissue"] == tissue)
        & (adata.obs["cxcr3_group"] == "CXCR3-")
        & adata.obs["condition"].isin(["HC", "MS"])
    ]
    if gene not in sub.var_names:
        print(f"  [diag] {gene} not in var_names")
        return
    print(f"\n  [diag] {gene} in {tissue} CXCR3- cells:")
    for cond in ["HC", "MS"]:
        cells = sub[sub.obs["condition"] == cond]
        expr  = cells[:, gene].X
        if sp.issparse(expr):
            expr = expr.toarray()
        expr = expr.flatten()
        print(
            f"    {cond}: n_cells={len(cells):4d}  "
            f"pct_expr={np.mean(expr > 0):.1%}  "
            f"mean={expr.mean():.4f}  "
            f"median={np.median(expr):.4f}"
        )
        
        
def compute_cell_signature(adata, zinc_genes, tissue: str) -> pd.DataFrame:
    sub = adata[
        (adata.obs["tissue"] == tissue) &
        (adata.obs["cxcr3_group"] == "CXCR3-") &
        adata.obs["condition"].isin(["HC", "MS"])
    ].copy()

    genes = [g for g in zinc_genes if g in sub.var_names]
    X = sub[:, genes].X
    if sp.issparse(X):
        X = X.toarray()

    sig = X.mean(axis=1)  # simple average
    df = sub.obs[["condition"]].copy()
    df["zinc_signature"] = sig
    return df


def _run_de(
    adata: sc.AnnData,
    tissue: str,
    zinc_genes: list[str],
    min_pct: float = 0.10,
) -> pd.DataFrame:
    """Wilcoxon DE: MS vs HC in CXCR3- cells of the given tissue."""
    sub = adata[
        (adata.obs["tissue"] == tissue)
        & (adata.obs["cxcr3_group"] == "CXCR3-")
        & adata.obs["condition"].isin(["HC", "MS"])
    ].copy()

    if sub.n_obs == 0:
        raise ValueError(f"No {tissue} CXCR3- HC/MS cells found.")
    if not {"HC", "MS"}.issubset(set(sub.obs["condition"].unique())):
        raise ValueError(f"{tissue} CXCR3- missing HC or MS condition.")

    sub = _apply_min_pct_filter(sub, min_pct=min_pct)
    key = f"de_{tissue}_CXCR3neg_MS_vs_HC"
    sc.tl.rank_genes_groups(
        sub, groupby="condition",
        groups=["MS"], reference="HC",
        method="wilcoxon", key_added=key,
    )
    df = sc.get.rank_genes_groups_df(sub, group="MS", key=key)
    df = df.rename(columns={
        "names": "gene", "logfoldchanges": "log2FC",
        "pvals": "pvalue", "pvals_adj": "padj",
    })
    df = df[np.isfinite(df["log2FC"])].copy()
    df["is_zinc"] = df["gene"].isin(zinc_genes)
    return df


def _select_genes_for_bars(
    df: pd.DataFrame,
    max_genes: int = 20,
    padj_thresh: float = SIG_PADJ_THRESHOLD,
) -> tuple[list[str], bool]:
    """
    Returns (gene_list, used_fallback).
    used_fallback=True when no gene passed padj; top log2FC used instead.
    """
    sig = df[df["is_zinc"] & df["padj"].notna() & (df["padj"] <= padj_thresh)]
    if not sig.empty:
        return sig.sort_values("log2FC", ascending=False)["gene"].head(max_genes).tolist(), False
    fallback = df[df["is_zinc"]].sort_values("log2FC", ascending=False)
    return fallback["gene"].head(max_genes).tolist(), True

def _all_zinc_genes_for_bars(df: pd.DataFrame) -> list[str]:
    # All zinc/MT genes present in the DE table, sorted by log2FC
    zinc_df = df[df["is_zinc"]].sort_values("log2FC", ascending=False)
    return zinc_df["gene"].tolist()

# ── Plot: volcano ────────────────────────────────────────────────────────────

def _draw_volcano(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    label: str,
    zinc_color: str,
) -> None:
    """
    Volcano plot: x = log2FC, y = -log10(padj).

    Replaces the original rank-scatter which had three problems:
      (a) y-axis dominated by extreme outliers at both ends,
          making zinc genes (near FC=0) invisible;
      (b) x-axis rank conveyed no information about significance;
      (c) gene labels were absent from the original code despite
          appearing in the output PDF.

    Here: significant zinc genes land in the upper half; the padj=0.05
    dotted line and FC=0 dashed line give immediate visual orientation.
    """
    df = df.copy()
    # Clip padj at machine epsilon to prevent log10(0) → inf
    df["-log10padj"] = -np.log10(df["padj"].clip(lower=1e-300))

    other    = df[~df["is_zinc"]]
    zinc     = df[df["is_zinc"]]
    zinc_sig = zinc[zinc["padj"] < SIG_PADJ_THRESHOLD]
    zinc_ns  = zinc[zinc["padj"] >= SIG_PADJ_THRESHOLD]

    # Background (rasterized saves file size)
    ax.scatter(
        other["log2FC"], other["-log10padj"],
        s=5, color=SCATTER_COLOR_OTHER, alpha=0.35,
        rasterized=True, label="Other genes",
    )
    # Non-significant zinc
    if not zinc_ns.empty:
        ax.scatter(
            zinc_ns["log2FC"], zinc_ns["-log10padj"],
            s=28, color=zinc_color, alpha=0.4,
            edgecolors="black", linewidths=0.4,
            label="Zinc/MT (ns)",
        )
    # Significant zinc
    if not zinc_sig.empty:
        ax.scatter(
            zinc_sig["log2FC"], zinc_sig["-log10padj"],
            s=45, color=zinc_color, alpha=0.9,
            edgecolors="black", linewidths=0.6,
            label=f"Zinc/MT (padj < {SIG_PADJ_THRESHOLD})",
        )

    # Reference lines
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(
        -np.log10(SIG_PADJ_THRESHOLD),
        color="grey", linestyle=":", linewidth=0.8,
    )

    # # Gene labels — only zinc genes to keep the plot readable
    # # Prefer significant subset; fall back to all zinc if none significant
    # label_targets = zinc_sig if not zinc_sig.empty else zinc
    # texts = []
    # for _, row in label_targets.iterrows():
    #     t = ax.text(
    #         row["log2FC"], row["-log10padj"], row["gene"],
    #         fontsize=7, ha="left", va="bottom", color="black",
    #     )
    #     texts.append(t)

    # # Use adjustText for non-overlapping labels when available
    # try:
    #     from adjustText import adjust_text
    #     adjust_text(
    #         texts, ax=ax,
    #         arrowprops=dict(arrowstyle="-", color="grey", lw=0.4),
    #         expand_points=(1.2, 1.5),
    #     )
    # except ImportError:
    #     pass  # fall back to default positions

    ax.set_xlabel("log2FC (MS vs HC)")
    ax.set_ylabel("−log10(padj)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    _panel_letter(ax, label)


# ── Plot: bars ───────────────────────────────────────────────────────────────

def _sig_label(padj_val: float) -> str | None:
    if pd.isna(padj_val):
        return None
    if padj_val < 0.001:
        return "***"
    if padj_val < 0.01:
        return "**"
    if padj_val < SIG_PADJ_THRESHOLD:
        return "*"
    return None


def _draw_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    genes: list[str],
    title: str,
    label: str,
    color: str,
    used_fallback: bool = False,
    outlier_cap: float | None = None,
) -> None:
    """
    Bar chart of log2FC for selected zinc/MT genes.

    New vs original:
      - Significance asterisks (* / ** / ***) above each bar.
      - `used_fallback=True` appends a subtitle warning that no gene
        passed padj < 0.05.
      - `outlier_cap` (float): any bar with |FC| > cap is drawn at 90%
        of the cap value, annotated with the true value and a hatch
        pattern to signal the axis break.
    """
    if not genes:
        ax.text(
            0.5, 0.5,
            "No zinc/metallothionein genes\npassing padj threshold",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="grey",
        )
        ax.set_title(title)
        _panel_letter(ax, label)
        return

    df_b = df[df["gene"].isin(genes)].copy()
    df_b["gene"] = pd.Categorical(df_b["gene"], categories=genes, ordered=True)
    df_b = df_b.sort_values("gene").reset_index(drop=True)

    x       = np.arange(len(genes))
    fc_raw  = df_b.set_index("gene").loc[genes, "log2FC"].values
    padj_v  = df_b.set_index("gene").loc[genes, "padj"].values

    # Outlier capping
    fc_plot      = fc_raw.copy()
    capped_mask  = np.zeros(len(fc_raw), dtype=bool)
    if outlier_cap is not None:
        capped_mask = np.abs(fc_raw) > outlier_cap
        fc_plot[capped_mask] = np.sign(fc_raw[capped_mask]) * outlier_cap * 0.9

    # Bars (hatch on capped ones to signal truncation)
    for i in range(len(genes)):
        hatch = "//" if capped_mask[i] else None
        ax.bar(x[i], fc_plot[i], width=0.65,
               color=color, edgecolor="black", linewidth=0.6,
               hatch=hatch)

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

    # Annotations for capped bars
    for i in np.where(capped_mask)[0]:
        true_val = fc_raw[i]
        cap_val  = fc_plot[i]
        direction = np.sign(cap_val)
        ax.annotate(
            f"{true_val:.1f}",
            xy=(x[i], cap_val),
            xytext=(x[i], cap_val + direction * abs(cap_val) * 0.08),
            ha="center", va="bottom" if direction > 0 else "top",
            fontsize=8, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", lw=1, color=color),
            clip_on=False,
        )

    # Significance asterisks
    y_range = max(np.abs(fc_plot).max(), 0.5)
    y_offset = y_range * 0.04
    for i in range(len(genes)):
        sig = _sig_label(padj_v[i])
        if sig is None:
            continue
        y_pos = max(fc_plot[i], 0) + y_offset
        ax.text(i, y_pos, sig, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("log2FC (MS vs HC)")
    ax.set_xlabel("")

    full_title = title
    if used_fallback:
        full_title += "\n(top ranked by log2FC; none padj < 0.05)"
    ax.set_title(full_title, fontsize=10 if not used_fallback else 8)

    _panel_letter(ax, label)



def draw_violin(ax, df_sig, tissue_label, letter):
    # HC vs MS
    groups = ["HC", "MS"]
    data = [df_sig.loc[df_sig["condition"]==g, "zinc_signature"].values for g in groups]

    parts = ax.violinplot(data, positions=[1, 2], showmeans=False, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#CCCCCC")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    # Scatter the individual points (optional, with jitter)
    for i, vals in enumerate(data, start=1):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=2,
            alpha=0.4,
            color="black",
            zorder=10,
        )

    # Stats
    stat, p = mannwhitneyu(data[0], data[1], alternative="two-sided")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    ax.set_ylabel("Zinc/MT signature (per cell)")
    ax.set_title(f"{tissue_label} CXCR3- B cells – MS vs HC \nMann–Whitney p = {p:.2e}", fontsize=9)
    ax.text(-0.12, 1.05, letter,
            transform=ax.transAxes, fontsize=14, fontweight="bold")

# ── Main ─────────────────────────────────────────────────────────────────────

def make_fig2_combined() -> None:
    merged_path = RESULTS_DIR / "merged_bcells.h5ad"
    if not merged_path.exists():
        raise FileNotFoundError(f"merged_bcells.h5ad not found at {merged_path}")

    adata = sc.read_h5ad(merged_path)
    if not adata.obs_names.is_unique:
        print("[fix] Making obs_names unique")
        adata.obs_names_make_unique()

    for col in ("condition", "cxcr3_group", "tissue"):
        if col not in adata.obs.columns:
            raise KeyError(f"Column '{col}' missing in adata.obs.")

    zinc_genes = load_zinc_gene_list()
    print(f"Loaded {len(zinc_genes)} curated zinc/MT genes: {zinc_genes}")
    


    # ── DE analysis ──────────────────────────────────────────────────────
    print("\nRunning DE: PB CXCR3- MS vs HC …")
    df_pb = _run_de(adata, tissue="PB", zinc_genes=zinc_genes, min_pct=0.10)
    print(f"  {df_pb.shape[0]} genes in PB DE table")

    # NOTE: min_pct raised to 0.15 for CSF.
    # CSF has far fewer cells than PB; a 10% filter in a small population
    # can still pass genes that are essentially absent in HC (e.g. 1-2 cells),
    # producing spuriously large log2FC values like the SLC39A14 = 24.4
    # seen in the original figure. Raise this threshold or set a min_mean_expr
    # filter if the outlier persists.
    print("\nRunning DE: CSF CXCR3- MS vs HC …")
    df_csf = _run_de(adata, tissue="CSF", zinc_genes=zinc_genes, min_pct=0.15)
    print(f"  {df_csf.shape[0]} genes in CSF DE table")

    df_sig_pb  = compute_cell_signature(adata, zinc_genes, tissue="PB")
    df_sig_csf = compute_cell_signature(adata, zinc_genes, tissue="CSF")

    # ── Outlier diagnostic ────────────────────────────────────────────────
    OUTLIER_FC_THRESHOLD = 5.0
    csf_extreme = df_csf[df_csf["is_zinc"] & (df_csf["log2FC"].abs() > OUTLIER_FC_THRESHOLD)]
    if not csf_extreme.empty:
        print(f"\n[WARNING] {len(csf_extreme)} zinc gene(s) with |log2FC| > {OUTLIER_FC_THRESHOLD} in CSF:")
        for _, row in csf_extreme.iterrows():
            print(f"  {row['gene']}: log2FC={row['log2FC']:.2f}, padj={row['padj']:.3e}")
            _check_gene_expression(adata, tissue="CSF", gene=row["gene"])
        print(
            "\n  These values are likely numerical artefacts from near-zero HC "
            "mean expression. Consider min_pct=0.20 or a min_mean_expression "
            "filter before publication.\n"
        )

    # # ── Select bar genes ──────────────────────────────────────────────────
    # pb_bar_genes,  pb_fallback  = _select_genes_for_bars(df_pb)
    # csf_bar_genes, csf_fallback = _select_genes_for_bars(df_csf)
    # print("PB bar genes :", pb_bar_genes)
    # print("CSF bar genes:", csf_bar_genes)
    
    # ── Select bar genes ──────────────────────────────────────────────────
    pb_bar_genes  = _all_zinc_genes_for_bars(df_pb)
    csf_bar_genes = _all_zinc_genes_for_bars(df_csf)
    pb_fallback = False
    csf_fallback = False
    print("PB bar genes :", pb_bar_genes)
    print("CSF bar genes:", csf_bar_genes)


    # ── Figure layout ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    (axA, axB), (axC, axD), (axE, axF) = axes

    # Panel A — volcano PB
    _draw_volcano(
        axA, df_pb,
        title="PB CXCR3- B cells – MS vs HC",
        label="A",
        zinc_color=SCATTER_COLOR_ZINC_PB,
    )

    # Panel B — bars PB
    _draw_bars(
        axB, df_pb, pb_bar_genes,
        title="PB CXCR3- – zinc/metallothionein genes",
        label="B",
        color=BAR_COLOR_PB,
        used_fallback=pb_fallback,
    )

    # Panel C — volcano CSF
    _draw_volcano(
        axC, df_csf,
        title="CSF CXCR3- B cells – MS vs HC",
        label="C",
        zinc_color=SCATTER_COLOR_ZINC_CSF,
    )

    # Panel D — bars CSF (with outlier cap)
    _draw_bars(
        axD, df_csf, csf_bar_genes,
        title="CSF CXCR3- – zinc/metallothionein genes",
        label="D",
        color=BAR_COLOR_CSF,
        used_fallback=csf_fallback,
        outlier_cap=OUTLIER_FC_THRESHOLD,
    )
    
    draw_violin(axE, df_sig_pb,  tissue_label="PB",  letter="E")
    draw_violin(axF, df_sig_csf, tissue_label="CSF", letter="F")

    
    fig.subplots_adjust(
        left=0.08, right=0.98,
        top=0.95, bottom=0.06,
        wspace=0.38, hspace=0.5,
    )


    # ── Save ─────────────────────────────────────────────────────────────
    out_dir = FIG_DIRS.get("supp", RESULTS_DIR / "fig2")
    out_dir.mkdir(parents=True, exist_ok=True)

    stem    = "FigS2_CXCR3neg_PB_CSF_MS_vs_HC_6panel"
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # CSVs
    df_pb.to_csv(out_dir / "FigS2_PB_CXCR3neg_MS_vs_HC_full_DE.csv",  index=False)
    df_csf.to_csv(out_dir / "FigS2_CSF_CXCR3neg_MS_vs_HC_full_DE.csv", index=False)
    pd.DataFrame({"gene": pb_bar_genes}).to_csv(
        out_dir / "FigS2_PB_CXCR3neg_zinc_bar_genes.csv", index=False)
    pd.DataFrame({"gene": csf_bar_genes}).to_csv(
        out_dir / "FigS2_CSF_CXCR3neg_zinc_bar_genes.csv", index=False)

    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    make_fig2_combined()