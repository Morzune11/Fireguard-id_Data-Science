"""
Modeling Module for FIREGUARD ID.
Implements Baseline (K-Means), Advanced ML (HDBSCAN), and Deep Learning (DEC).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def train_kmeans(X, n_clusters=4, random_state=42):
    """Fits K-Means baseline model."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, labels


def sweep_kmeans(X_train, X_val=None, k_range=range(2, 9), random_state=42):
    """Evaluates K-Means across a range of k values for Elbow & Silhouette tuning."""
    results = []
    for k in k_range:
        model, train_labels = train_kmeans(X_train, n_clusters=k, random_state=random_state)
        row = {
            "k": int(k),
            "inertia": float(model.inertia_),
            "train_silhouette": float(silhouette_score(X_train, train_labels)),
            "train_dbi": float(davies_bouldin_score(X_train, train_labels)),
            "train_chi": float(calinski_harabasz_score(X_train, train_labels)),
        }
        if X_val is not None:
            val_labels = model.predict(X_val)
            if len(set(val_labels)) >= 2:
                row["val_silhouette"] = float(silhouette_score(X_val, val_labels))
                row["val_dbi"] = float(davies_bouldin_score(X_val, val_labels))
            else:
                row["val_silhouette"] = None
                row["val_dbi"] = None
        results.append(row)
    return results


def train_hdbscan(X, min_cluster_size=15, min_samples=5, cluster_selection_method="eom"):
    """Fits scikit-learn HDBSCAN density-based clustering model."""
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        store_centers="centroid",
    )
    labels = model.fit_predict(X)
    return model, labels


def predict_hdbscan_nearest(model, X_new):
    """Assigns unseen samples to the nearest fitted HDBSCAN centroid."""
    centroids = getattr(model, "centroids_", None)
    if centroids is None or len(centroids) == 0:
        return np.full(len(X_new), -1, dtype=int)
    dists = np.linalg.norm(X_new[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


class DeepEmbeddedClustering(nn.Module):
    """Autoencoder + Student's t-distribution clustering layer (Xie et al., 2016)."""

    def __init__(self, input_dim, latent_dim=4, n_clusters=4, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    def soft_assign(self, z):
        """Computes Student's t-distribution soft cluster assignment q."""
        dist_sq = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        q = self.soft_assign(z)
        return x_recon, z, q


def target_distribution(q):
    """Computes auxiliary target distribution P from soft assignments Q."""
    weight = (q ** 2) / torch.sum(q, dim=0, keepdim=True)
    return weight / torch.sum(weight, dim=1, keepdim=True)


def train_dec(
    X_train,
    n_clusters=4,
    latent_dim=4,
    pretrain_epochs=120,
    refine_epochs=80,
    lr=1e-3,
    random_state=42,
):
    """Pretrains Autoencoder, initializes centroids with K-Means, and refines DEC."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    model = DeepEmbeddedClustering(
        input_dim=X_train.shape[1],
        latent_dim=latent_dim,
        n_clusters=n_clusters,
    )

    # 1. Pretrain Autoencoder
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    model.train()
    for _ in range(pretrain_epochs):
        optimizer.zero_grad()
        x_recon, _, _ = model(X_tensor)
        loss = mse_loss(x_recon, X_tensor)
        loss.backward()
        optimizer.step()

    # 2. Initialize cluster centers via K-Means in latent space
    model.eval()
    with torch.no_grad():
        _, z_init, _ = model(X_tensor)
        z_np = z_init.numpy()
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(z_np)
    with torch.no_grad():
        model.cluster_centers.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32))

    # 3. Joint Clustering Refinement (KL Divergence + Reconstruction preservation)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.5)
    model.train()
    for _ in range(refine_epochs):
        with torch.no_grad():
            _, _, q_cur = model(X_tensor)
            p = target_distribution(q_cur)

        optimizer.zero_grad()
        x_recon, _, q = model(X_tensor)
        loss = kl_loss(torch.log(q + 1e-10), p) + 0.1 * mse_loss(x_recon, X_tensor)
        loss.backward()
        optimizer.step()

    labels, z_final, _ = predict_dec(model, X_train)
    return model, labels, z_final


def predict_dec(model, X):
    """Predicts cluster assignments and latent representations using trained DEC."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        _, z, q = model(X_tensor)
    labels = torch.argmax(q, dim=1).numpy()
    return labels, z.numpy(), q.numpy()
