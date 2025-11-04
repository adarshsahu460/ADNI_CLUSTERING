from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict


def cluster_summary(df_base: pd.DataFrame, labels: pd.Series, feature_cols: list[str]) -> pd.DataFrame:
    df = df_base.copy()
    df["cluster"] = labels.values

    # Drop duplicated column names to ensure df[c] retrieves a Series, not a DataFrame
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Ensure numeric dtype for key clinical descriptors that we want to average
    numeric_descriptors = ["AGE", "MMSE", "ADAS13", "ABETA", "TAU", "PTAU", "APOE4"]
    for c in numeric_descriptors:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Only summarize features that actually exist in the provided dataframe and are numeric
    agg_map = {c: ["mean", "std"] for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])}
    # Add clinical descriptors (only if numeric after coercion)
    for c in numeric_descriptors:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            agg_map[c] = ["mean", "std"]
    # Categorical summaries as value counts
    for c in ["PTGENDER", "DX_bl", "DX"]:
        if c in df.columns:
            agg_map[c] = [lambda s: s.value_counts().to_dict()]

    summary = df.groupby("cluster").agg(agg_map)
    # Flatten MultiIndex columns
    summary.columns = ["_".join([a for a in col if a]) for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    return summary


def compute_mmse_slope(adni: pd.DataFrame) -> pd.DataFrame:
    # Estimate MMSE slope per subject across months since baseline
    df = adni.copy()
    if "EXAMDATE" in df.columns:
        # Parse dd-mm-yyyy
        df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], dayfirst=True, errors="coerce")
    # Build month from VISCODE if Month not available
    if "Month" not in df.columns:
        # Approx months since first visit
        df = df.sort_values(["RID", "EXAMDATE"]).copy()
        df["Month"] = df.groupby("RID")["EXAMDATE"].transform(lambda s: ((s - s.min()).dt.days / 30.44).round())

    keep = [c for c in ["RID", "Month", "MMSE", "VISCODE"] if c in df.columns]
    df = df[keep].dropna(subset=["MMSE"]).copy()

    # Fit simple linear slope per subject
    slopes = []
    for rid, g in df.groupby("RID"):
        if g["Month"].nunique() < 2:
            continue
        x = g["Month"].values.astype(float)
        y = g["MMSE"].values.astype(float)
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        slopes.append({"RID": rid, "mmse_slope_per_month": m, "n_visits": len(g)})
    return pd.DataFrame(slopes)
