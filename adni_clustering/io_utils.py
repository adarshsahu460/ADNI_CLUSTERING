from __future__ import annotations

import pandas as pd
from typing import List, Tuple, Dict
from pathlib import Path


BASELINE_CODES = {"bl", "sc", "v01", "baseline"}

# Candidate features by modality; we will filter to those that exist in the CSV
CANDIDATE_FEATURE_GROUPS: Dict[str, List[str]] = {
    "demographics": ["AGE", "PTGENDER", "PTEDUCAT"],
    "genetics": ["APOE4"],
    "cognition": [
        "MMSE", "ADAS13", "ADAS11", "ADASQ4",
        "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting",
        "LDELTOTAL", "DIGITSCOR", "TRABSCOR", "FAQ", "MOCA"
    ],
    "csf": ["ABETA", "TAU", "PTAU"],
    "pet": ["FDG", "AV45", "FBB", "PIB"],
    "mri_raw": [
        "Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"
    ],
}


def load_adnimerge(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    # The provided file uses dd-mm-yyyy in EXAMDATE; we parse dates later when needed
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def detect_available_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    present = {}
    for group, cols in CANDIDATE_FEATURE_GROUPS.items():
        present[group] = [c for c in cols if c in df.columns]
    return present


def filter_baseline(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer explicit baseline rows; fall back to first exam per subject if baseline missing
    viscode = df.get("VISCODE")
    if viscode is not None:
        base = df[viscode.str.lower().isin(BASELINE_CODES)].copy()
    else:
        base = df.copy()
    # Ensure one row per subject; key columns vary across ADNI phases, use RID as numeric subject id
    if "RID" in base.columns:
        base = base.sort_values(["RID", "EXAMDATE"]).groupby("RID", as_index=False).head(1)
    elif "PTID" in base.columns:
        base = base.sort_values(["PTID", "EXAMDATE"]).groupby("PTID", as_index=False).head(1)
    return base


def feature_matrix(
    df_base: pd.DataFrame,
    include_groups: List[str] | None = None,
    drop_demographics_from_latent: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    present = detect_available_features(df_base)
    if include_groups is None:
        include_groups = [
            "cognition", "csf", "pet", "mri_raw", "genetics",  # strong biological signal
            # keep demographics for post-hoc, but optionally drop from latent
            "demographics",
        ]
    cols: List[str] = []
    for g in include_groups:
        cols.extend(present.get(g, []))

    # Build working df, keep identifiers for later joins
    id_cols = [c for c in ["RID", "PTID", "DX_bl", "DX", "AGE", "PTGENDER", "EXAMDATE", "VISCODE"] if c in df_base.columns]
    work = df_base[id_cols + cols].copy()
    # Drop any duplicated columns (ADNI sometimes has both current and *_bl variants colliding)
    work = work.loc[:, ~work.columns.duplicated()]

    # Normalize categorical: PTGENDER to numeric
    if "PTGENDER" in work.columns:
        # Ensure single series after dropping duplicates; map string values to numeric
        work["PTGENDER"] = work["PTGENDER"].replace({"Male": 1, "Female": 0}).astype("float64")

    # MRI normalization: divide volumetrics by ICV when available
    if "ICV" in work.columns:
        for c in ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "WholeBrain", "Ventricles"]:
            if c in work.columns:
                work[f"{c}_norm"] = work[c] / work["ICV"]

    # Optionally drop demographics from latent features (keep for reporting)
    feature_cols = [c for c in work.columns if c not in id_cols]
    if drop_demographics_from_latent:
        for d in ["AGE", "PTGENDER", "PTEDUCAT"]:
            if d in feature_cols:
                feature_cols.remove(d)

    # Remove raw MRI if normalized exists
    for raw, norm in [(c, f"{c}_norm") for c in ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "WholeBrain", "Ventricles"]]:
        if raw in feature_cols and norm in work.columns:
            feature_cols.remove(raw)
            feature_cols.append(norm)

    # Deduplicate preserving order
    seen = set()
    ordered = []
    for c in feature_cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    X = work[ordered]
    return X, ordered
