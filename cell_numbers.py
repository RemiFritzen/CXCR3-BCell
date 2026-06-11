import pandas as pd
import scanpy as sc
from config.config import RESULTS_DIR

# 1. Load merged object (for sanity checks if needed)
adata = sc.read_h5ad(RESULTS_DIR / "merged_bcells.h5ad")

# 2. Load your per-sample (patient × tissue × condition) counts
counts = pd.read_csv(RESULTS_DIR / "per_sample_bcell_counts.csv")

# 3. Keep only rows that actually have cells
counts = counts[counts["n_cells"] > 0].copy()

# 4. Derive numeric patient ID (2, 3, 6, 7, ...) from CSF_2 / PB_9 etc.
def extract_patient_numeric(pid: str) -> str:
    # CSF_2 -> 2, PB_9 -> 9, MS19270_CSF stays as is
    if "_" in pid and pid.split("_", 1)[1].isdigit():
        return pid.split("_", 1)[1]
    return pid

counts["patient_id_num"] = counts["patient"].astype(str).apply(extract_patient_numeric)

# 5. Separate CSF and PB, aggregate per patient/dataset

# CSF side
csf = (
    counts[counts["tissue"] == "CSF"]
    .groupby(["dataset", "patient_id_num"])
    .agg(
        condition=("condition", lambda x: x.mode().iloc[0]),
        n_cells_csf=("n_cells", "sum"),
        cxcr3_pos_csf=("n_CXCR3_pos", "sum"),
        cxcr3_neg_csf=("n_CXCR3_neg", "sum"),
    )
    .reset_index()
)

# PB side
pb = (
    counts[counts["tissue"] == "PB"]
    .groupby(["dataset", "patient_id_num"])
    .agg(
        n_cells_pb=("n_cells", "sum"),
        cxcr3_pos_pb=("n_CXCR3_pos", "sum"),
        cxcr3_neg_pb=("n_CXCR3_neg", "sum"),
    )
    .reset_index()
)

# 6. Combine CSF and PB per patient
summary = csf.merge(
    pb,
    on=["dataset", "patient_id_num"],
    how="outer",
)

# 7. Order columns to match your Supplementary template
summary = summary[
    [
        "dataset",
        "patient_id_num",      # Patient ID column in the table
        "condition",
        "n_cells_csf",
        "cxcr3_pos_csf",
        "cxcr3_neg_csf",
        "n_cells_pb",
        "cxcr3_pos_pb",
        "cxcr3_neg_pb",
    ]
].sort_values(["dataset", "patient_id_num"])

summary.rename(
    columns={
        "patient_id_num": "Patient_ID",
        "condition": "Condition",
        "n_cells_csf": "#cells_CSF",
        "cxcr3_pos_csf": "#CXCR3+_CSF",
        "cxcr3_neg_csf": "#CXCR3-_CSF",
        "n_cells_pb": "#cells_PB",
        "cxcr3_pos_pb": "#CXCR3+_PB",
        "cxcr3_neg_pb": "#CXCR3-_PB",
    },
    inplace=True,
)

# 8. Save for Supplementary paste
out_path = RESULTS_DIR / "supplement_per_patient_counts.csv"
summary.to_csv(out_path, index=False)
print(f"Saved -> {out_path}")
