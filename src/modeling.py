"""
Modeling Module for FIREGUARD ID.
Baseline (K-Means), Advanced (HDBSCAN), and Deep Embedded Clustering (DEC).
"""
from sklearn.cluster import KMeans

def train_kmeans(X, n_clusters=4, random_state=42):
    """Fits K-Means baseline model."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, labels
