"""
Evaluation Module for Spatial Clustering Models.
Calculates Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
"""
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(X, labels):
    """Computes standard unsupervised clustering quality metrics."""
    metrics = {
        "silhouette_score": float(silhouette_score(X, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels))
    }
    return metrics
