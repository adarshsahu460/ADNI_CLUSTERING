from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import Dict


def compute_embeddings(X_scaled: np.ndarray, random_state: int = 42) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    # PCA to 50 components (or less if smaller)
    n_components = min(50, X_scaled.shape[1]) if X_scaled.shape[1] > 2 else X_scaled.shape[1]
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    out["pca"] = X_pca

    # UMAP 2D from PCA for smoother behavior
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1, metric="euclidean")
    X_umap = reducer.fit_transform(X_pca)
    out["umap2"] = X_umap

    # Optional t-SNE (can be slower); use PCA-50 as input
    try:
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, init="pca", learning_rate="auto")
        X_tsne = tsne.fit_transform(X_pca)
        out["tsne2"] = X_tsne
    except Exception:
        pass

    return out
