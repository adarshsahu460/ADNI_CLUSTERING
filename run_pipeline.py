from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from adni_clustering.io_utils import load_adnimerge, filter_baseline, feature_matrix
from adni_clustering.preprocess import impute_and_scale
from adni_clustering.embed import compute_embeddings
from adni_clustering.cluster import run_clustering_grid, select_best_model
from adni_clustering.posthoc import cluster_summary, compute_mmse_slope
from adni_clustering.viz import plot_embedding_scatter, barplot_cluster_means


def main():
    parser = argparse.ArgumentParser(description="ADNI Unsupervised Phenotype Discovery")
    parser.add_argument("--input", type=str, required=True, help="Path to ADNIMERGE CSV")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to write outputs")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load and baseline filter
    df = load_adnimerge(args.input)
    df_base = filter_baseline(df)

    # Build feature matrix
    X_df, feature_cols = feature_matrix(df_base, drop_demographics_from_latent=True)

    # Impute and scale
    X_imp_df, X_scaled, scaler, imputer = impute_and_scale(X_df, random_state=args.random_state)

    # Save cleaned feature matrix
    X_imp_df.to_csv(outdir / "features_imputed.csv", index=False)

    # Compute embeddings
    embeds = compute_embeddings(X_scaled, random_state=args.random_state)
    if "umap2" in embeds:
        np.savetxt(outdir / "umap2.csv", embeds["umap2"], delimiter=",")

    # Run clustering on scaled features (not just 2D embedding)
    res_df, label_store = run_clustering_grid(X_scaled, random_state=args.random_state)
    res_df.to_csv(outdir / "clustering_metrics.csv", index=False)

    best = select_best_model(res_df)
    labels = label_store[best]

    # Report and plots
    print(f"Selected model: {best}; clusters: {len(set(labels))}")

    # Attach labels to baseline df for reporting
    df_report = df_base.copy()
    df_report = df_report.reset_index(drop=True)
    df_report["cluster"] = labels
    df_report.to_csv(outdir / "cluster_assignments.csv", index=False)

    # Plot UMAP scatter colored by clusters if available
    if "umap2" in embeds:
        plot_embedding_scatter(embeds["umap2"], labels, outdir / "umap2_clusters.png", title=f"UMAP 2D colored by {best}")

    # Compute per-cluster feature deviations
    df_for_bars = pd.concat([X_imp_df.reset_index(drop=True), pd.Series(labels, name="cluster")], axis=1)
    barplot_cluster_means(df_for_bars, features=X_imp_df.columns.tolist(), out_dir=outdir / "figures", title_suffix=f"({best})")

    # Post-hoc: clinical summaries and MMSE slopes
    # Merge only non-overlapping engineered features into baseline to avoid duplicate column names
    base = df_base.reset_index(drop=True)
    feats = X_imp_df.reset_index(drop=True)
    non_overlap_cols = [c for c in feats.columns if c not in base.columns]
    df_summary = pd.concat([base, feats[non_overlap_cols]], axis=1)
    summ = cluster_summary(df_summary, pd.Series(labels), feature_cols=feature_cols)
    summ.to_csv(outdir / "cluster_summary.csv", index=False)

    mmse_slopes = compute_mmse_slope(df)
    if not mmse_slopes.empty and "RID" in df_base.columns:
        merged = df_base[["RID"]].drop_duplicates().merge(mmse_slopes, on="RID", how="left")
        merged["cluster"] = labels
        # Compare slopes by cluster
        merged.to_csv(outdir / "mmse_slopes_by_cluster.csv", index=False)


if __name__ == "__main__":
    main()
