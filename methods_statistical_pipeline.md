# Methods (statistical pipeline — revised)

*Drop-in replacement/addition for the "Differential expressions and group comparisons" and "Statistics" subsections. Bracketed values [ ] need to be filled in from your actual pipeline output before submission — I've flagged each one.*

---

## CXCR3-positive cell classification

CXCR3-positive status was assigned per cell based on detection of *CXCR3* transcript in the raw (pre-normalization) count matrix: cells with a raw UMI count greater than zero for *CXCR3* were classified as CXCR3⁺, and all other cells as CXCR3⁻. This threshold was evaluated by inspecting the distribution of raw *CXCR3* counts across all B cells, which showed the pronounced zero-inflation typical of lowly-expressed surface-receptor transcripts in scRNA-seq (dropout), with no evidence of a secondary mode among cells with any detected transcript (Supplementary Figure S[X]). Given this distribution, presence/absence is the most defensible classification the data support; a more granular expression-level threshold would not be justified by the underlying count structure. CXCR3⁺ detection rates were consistent across datasets [XX.X% GSE133028, XX.X% GSE138266] and tissues [XX.X% PB, XX.X% CSF], arguing against the classification being driven by study- or compartment-specific technical variation.

## Differential expression and group comparisons

Because single cells from the same patient are not statistically independent — they share genotype, disease state, sample processing, and technical batch — differential expression and signature comparisons were performed at the patient level rather than on individual cells, to avoid pseudoreplication (treating correlated cells as independent observations, which artificially inflates the effective sample size and can produce spuriously small p-values). For each comparison, cells were first aggregated to one value per patient (mean expression across cells of the relevant subset), and statistical testing was then performed on these patient-level values (n = number of patients, not number of cells).

Two comparison structures were used, matched to the underlying data structure:

- **Unpaired comparisons** (MS vs. HC): each patient belongs to exactly one condition. Patient-level values were compared between groups using a two-sided Mann–Whitney U test.
- **Paired, within-patient comparisons** (CXCR3⁺ vs. CXCR3⁻; PB vs. CSF): the same patient contributes cells to both arms of the comparison. Patient-level values were paired by patient (or, for PB vs. CSF, by a tissue-independent subject identifier — see below), and compared using a two-sided Wilcoxon signed-rank test on the within-patient differences. This removes patient-to-patient baseline variation before testing, isolating the within-patient effect of interest.

For PB-vs-CSF comparisons specifically, patient identifiers in the underlying per-sample data can encode tissue (e.g. distinct keys for a given patient's PB and CSF samples). Pairing was therefore performed using a tissue-independent subject identifier, derived by matching sample identifiers across tissues [and, where sample-key naming did not already support this, by stripping tissue-indicating tokens from the identifier — verified by manual inspection of the resulting subject-to-sample mapping]. Only patients with both PB and CSF samples present contributed to the paired analysis; patients missing one tissue were excluded from that specific comparison and are reported in the corresponding results (n pairs).

Multiple-testing correction (Benjamini–Hochberg) was applied at two levels, appropriate to two distinct claims:

1. **Genome-wide correction**, across all genes tested in a given comparison (~[N] genes passing expression-prevalence filtering), used for volcano plots, which represent an exploratory, hypothesis-generating scan across the transcriptome.
2. **Panel-restricted correction**, applied separately within the small, pre-specified set of curated zinc transporter and metallothionein genes (n = [13] genes; Table S[X]), used for the significance annotations on the zinc/metallothionein bar charts (Figures 2A–B, 3A–B). Because this gene panel was defined a priori from established biological role rather than discovered by screening the transcriptome, correcting it against thousands of unrelated background genes (appropriate for the genome-wide volcano) is an overly conservative correction family for this specific, narrower, confirmatory claim. Both corrected values are reported in the underlying differential expression tables (Supplementary Tables S[X]) for transparency.

An expression-prevalence filter (gene detected in ≥10% of cells in at least one arm of the comparison) was applied before testing to exclude uninformative genes, with the curated zinc/metallothionein panel explicitly exempted from this filter so that every curated gene was tested and reported regardless of prevalence, even where this results in a non-significant result.

## Zinc/metallothionein composite signature

A composite zinc/metallothionein signature was calculated per cell as the mean expression across the curated gene panel. Because this panel includes genes with opposing expected direction of change — zinc transporters (import/export machinery) versus metallothioneins (zinc-buffering/sequestering proteins) — two versions of the composite score were computed:

- **Sign-corrected**: each gene's expression was multiplied by a fixed sign (+1 for transporters, −1 for metallothioneins) prior to averaging, so that genes expected to increase and genes expected to decrease with the same underlying process both contribute in the same direction to the composite score, rather than partially cancelling in a plain average. Gene signs were assigned a priori from established biological role (transporter vs. buffering protein), not derived from the differential expression results of the comparison being tested, to avoid circularity.
- **Unsigned**: the plain mean of raw/normalized expression across the panel, with no directional weighting.

The sign-corrected signature was used as the primary composite readout for the MS-vs-HC comparison (Figure 2C–D), consistent with the pre-specified hypothesis that transporter expression increases and metallothionein expression decreases with disease; the unsigned signature is reported alongside in Supplementary Figure S[X] for comparison. For the PB-vs-CSF compartmental comparison (Figure 3C–D), the unsigned signature was used as the primary readout, since no equivalent a priori directional hypothesis applies to a tissue-compartment shift (per-gene results in this comparison show a mixed-direction pattern even within the transporter category); the sign-corrected version is reported in Supplementary Figure S[X].

Patient-level signature values (mean of the per-cell composite signature, aggregated as above) were compared between groups using the same unpaired (Mann–Whitney U) or paired (Wilcoxon signed-rank) framework used for per-gene differential expression, matched to the comparison structure.

## Statistics summary

All statistical analyses were performed in Python (scanpy, scipy, statsmodels, pandas, numpy). Group comparisons of patient-level values used two-sided Mann–Whitney U tests (unpaired: MS vs. HC) or two-sided Wilcoxon signed-rank tests (paired, within-patient: CXCR3⁺ vs. CXCR3⁻, PB vs. CSF). Multiple-testing correction used the Benjamini–Hochberg procedure, applied separately at genome-wide and curated-panel scope as described above. P-values below 0.05 (corrected) were considered statistically significant. Sample sizes (number of patients/pairs contributing to each test) are reported alongside each result, and comparisons involving fewer than [3] patients per group are noted as exploratory given limited statistical power.

---

## Notes for you (not for the manuscript)

- Fill in the bracketed numbers from your actual pipeline console output (patient counts, gene counts, detection rates by dataset/tissue) — these are all printed by the current scripts (`run_all_zinc_analyses.py`, `fig2.py`, `fig3.py`).
- The subject-ID pairing sentence has an "and, where..." clause in brackets — cut that if `resolve_pb_csf_subject_ids` found the raw `patient` column already pairs cleanly in your real data (no stripping needed); keep it if it doesn't.
- Consider citing the pseudobulk approach (e.g. Squair et al. 2021, *Nat Commun*, "Confronting false discoveries in single-cell differential expression" — the canonical reference for why per-cell DE pseudoreplicates and pseudobulk is the standard fix) if your target journal expects a methodological citation here.
- I'd suggest folding the CIS n=2 / small-group caveat (reviewer point #2) into the Results text near the affected numbers, not just here in Methods — this draft doesn't do that since it's out of scope for a Methods section specifically.
