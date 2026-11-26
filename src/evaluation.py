"""
Evaluation Module for Spatial Clustering Models.
Computes Silhouette, Davies-Bouldin, Calinski-Harabasz, Noise Ratio, Spatial Coherence,
and Vulnerability Profiling per Cluster.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors


def evaluate_clustering(X, labels):
    """Computes standard unsupervised clustering metrics, handling HDBSCAN noise (-1)."""
    labels = np.asarray(labels)
    noise_mask = labels == -1
    noise_ratio = float(noise_mask.mean())
    valid_mask = ~noise_mask
    unique_clusters = set(labels[valid_mask])
    n_clusters = len(unique_clusters)

    metrics = {
        "n_clusters": int(n_clusters),
        "noise_ratio": round(noise_ratio, 4),
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
    }

    if n_clusters >= 2 and valid_mask.sum() > n_clusters:
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        metrics["silhouette_score"] = round(float(silhouette_score(X_valid, labels_valid)), 4)
        metrics["davies_bouldin_score"] = round(float(davies_bouldin_score(X_valid, labels_valid)), 4)
        metrics["calinski_harabasz_score"] = round(float(calinski_harabasz_score(X_valid, labels_valid)), 4)

    return metrics


def compute_spatial_coherence(df, label_col="cluster", k_neighbors=4):
    """
    Measures spatial coherence as the average fraction of k-nearest geographic neighbors
    that share the same cluster label.
    """
    if len(df) <= k_neighbors:
        return 0.0
    coords = df[["center_lat", "center_lon"]].values
    labels = df[label_col].values

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
    _, indices = nn.kneighbors(coords)

    same_fraction = []
    for i, nbrs in enumerate(indices):
        nbrs_excluding_self = nbrs[1:]
        same = (labels[nbrs_excluding_self] == labels[i]).mean()
        same_fraction.append(same)

    return round(float(np.mean(same_fraction)), 4)


def profile_clusters(df, label_col="cluster", feature_cols=None):
    """Summarizes fire indicators per cluster and assigns risk priority rank."""
    if feature_cols is None:
        feature_cols = ["hotspot_count", "fire_days", "mean_frp", "max_frp", "high_conf_count"]

    summary = (
        df.groupby(label_col)[feature_cols]
        .agg(["count", "mean", "median", "max"])
    )
    # Flatten multi-index columns cleanly
    flat_summary = pd.DataFrame(index=summary.index)
    flat_summary["grid_count"] = summary[(feature_cols[0], "count")]
    for col in feature_cols:
        flat_summary[f"{col}_mean"] = summary[(col, "mean")].round(2)
        flat_summary[f"{col}_max"] = summary[(col, "max")].round(2)

    flat_summary = flat_summary.sort_values("hotspot_count_mean", ascending=False)
    return flat_summary
