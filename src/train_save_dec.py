"""
Train & Export Deep Embedded Clustering (DEC) Model Artifacts for FIREGUARD ID.
Saves PyTorch weights (models/dec_model.pt), latent space coordinates, and 2D latent plot.
"""
import os
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp")

from data_loader import load_grid_profiles
from modeling import train_dec, predict_dec
from evaluation import evaluate_clustering, profile_clusters


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    train_scaled = pd.read_csv("data/split/train_scaled.csv")
    val_scaled = pd.read_csv("data/split/val_scaled.csv")
    test_scaled = pd.read_csv("data/split/test_scaled.csv")

    scaled_cols = [
        "hotspot_count_scaled",
        "fire_days_scaled",
        "mean_frp_scaled",
        "max_frp_scaled",
        "high_conf_count_scaled",
    ]
    raw_cols = ["hotspot_count", "fire_days", "mean_frp", "max_frp", "high_conf_count"]

    X_train = train_scaled[scaled_cols].values
    X_val = val_scaled[scaled_cols].values
    X_test = test_scaled[scaled_cols].values

    print("Melatih Deep Embedded Clustering (DEC)...")
    dec_model, train_lbl, train_z = train_dec(
        X_train, n_clusters=4, latent_dim=4, pretrain_epochs=120, refine_epochs=80, random_state=42
    )
    torch.save(dec_model.state_dict(), "models/dec_model.pt")
    print("Model DEC disimpan ke: models/dec_model.pt")

    val_lbl, val_z, _ = predict_dec(dec_model, X_val)
    test_lbl, test_z, _ = predict_dec(dec_model, X_test)

    print("Train Metrics (Latent):", evaluate_clustering(train_z, train_lbl))
    print("Val Metrics   (Latent):", evaluate_clustering(val_z, val_lbl))
    print("Test Metrics  (Latent):", evaluate_clustering(test_z, test_lbl))

    # Project all 835 grids into DEC latent space
    scaler = joblib.load("data/split/scaler.joblib")
    df_all = load_grid_profiles("data/processed/riau_vulnerability_grid_01deg.csv")
    X_all_scaled = scaler.transform(df_all[raw_cols])
    all_lbl, all_z, all_q = predict_dec(dec_model, X_all_scaled)

    latent_df = pd.DataFrame({
        "grid_id": df_all["grid_id"],
        "dec_cluster": all_lbl,
        "z1": all_z[:, 0],
        "z2": all_z[:, 1],
        "z3": all_z[:, 2],
        "z4": all_z[:, 3],
        "max_prob": np.max(all_q, axis=1),
        "hotspot_count": df_all["hotspot_count"],
        "mean_frp": df_all["mean_frp"],
    })
    latent_df.to_csv("reports/dec_latent_coords.csv", index=False)

    # Plot 2D Latent Space
    plt.figure(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c", "#f1c40f", "#e67e22"]
    for cid in sorted(latent_df["dec_cluster"].unique()):
        sub = latent_df[latent_df["dec_cluster"] == cid]
        plt.scatter(sub["z1"], sub["z2"], label=f"Cluster {cid} (n={len(sub)})", alpha=0.75, s=30, color=colors[cid % len(colors)])
    plt.title("Representasi Ruang Laten DEC (z1 vs z2) — 835 Grid Riau", fontweight="bold")
    plt.xlabel("Latent Dimension 1 (z1)")
    plt.ylabel("Latent Dimension 2 (z2)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("reports/figures/dec_latent_space.png", dpi=200)
    plt.close()
    print("Visualisasi ruang laten disimpan ke: reports/figures/dec_latent_space.png")


if __name__ == "__main__":
    main()
