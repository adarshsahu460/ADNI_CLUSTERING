from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


def plot_embedding_scatter(X2: np.ndarray, labels: np.ndarray, out_path: str | Path, title: str = "UMAP 2D"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    palette = sns.color_palette("viridis", n_colors=int(np.unique(labels).size + (1 if (labels == -1).any() else 0)))
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=labels.astype(int), palette=palette, s=18, alpha=0.9, edgecolor="none")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def barplot_cluster_means(df_with_labels: pd.DataFrame, features: list[str], out_dir: str | Path, top_n: int = 8, title_suffix: str = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute standardized differences vs overall mean
    means = df_with_labels.groupby("cluster")[features].mean()
    overall = df_with_labels[features].mean()
    stds = df_with_labels[features].std().replace(0, np.nan)

    diffs = (means - overall) / stds
    # For each cluster, plot top_n absolute diffs
    for c in diffs.index:
        d = diffs.loc[c].dropna()
        top = d.reindex(d.abs().sort_values(ascending=False).head(top_n).index)
        plt.figure(figsize=(8, max(3, 0.35 * len(top))))
        sns.barplot(x=top.values, y=top.index, orient="h", palette="coolwarm")
        plt.axvline(0, color="k", lw=0.8)
        plt.title(f"Cluster {c} top feature deviations {title_suffix}")
        plt.tight_layout()
        plt.savefig(out_dir / f"cluster_{c}_top_features.png", dpi=200)
        plt.close()
