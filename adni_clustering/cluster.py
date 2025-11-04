from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Any, List, Tuple


def evaluate_labels(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    # Require at least 2 clusters and less than n_samples
    if len(set(labels)) <= 1 or len(set(labels)) >= len(labels):
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = np.nan
    try:
        db = davies_bouldin_score(X, labels)
    except Exception:
        db = np.nan
    try:
        ch = calinski_harabasz_score(X, labels)
    except Exception:
        ch = np.nan
    return {"silhouette": float(sil), "davies_bouldin": float(db), "calinski_harabasz": float(ch)}


def run_clustering_grid(X: np.ndarray, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    results: List[Dict[str, Any]] = []
    label_store: Dict[str, np.ndarray] = {}

    # KMeans k=2..6
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        lab = km.fit_predict(X)
        scores = evaluate_labels(X, lab)
        name = f"kmeans_k{k}"
        results.append({"model": name, "k": k, **scores})
        label_store[name] = lab

    # Agglomerative Ward
    for k in range(2, 7):
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab = agg.fit_predict(X)
        scores = evaluate_labels(X, lab)
        name = f"agg_ward_k{k}"
        results.append({"model": name, "k": k, **scores})
        label_store[name] = lab

    # GMM with various covariances
    for k in range(2, 7):
        for cov in ["full", "diag"]:
            gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=random_state, n_init=5)
            lab = gmm.fit_predict(X)
            scores = evaluate_labels(X, lab)
            name = f"gmm_{cov}_k{k}"
            results.append({"model": name, "k": k, **scores})
            label_store[name] = lab

    # DBSCAN over a small grid around default eps
    for eps in [0.5, 0.7, 1.0]:
        for min_samples in [5, 10]:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            lab = db.fit_predict(X)
            scores = evaluate_labels(X, lab)
            name = f"dbscan_eps{eps}_min{min_samples}"
            results.append({"model": name, "k": int(len(set(lab)) - (1 if -1 in lab else 0)), **scores})
            label_store[name] = lab

    res_df = pd.DataFrame(results)

    return res_df, label_store


def select_best_model(res_df: pd.DataFrame) -> str:
    # Primary: maximize silhouette; fallback to minimize DB if silhouette NaN
    df = res_df.copy()
    df = df.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
    return df.iloc[0]["model"]
