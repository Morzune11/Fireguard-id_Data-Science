"""
FIREGUARD ID — Clustering Modeling & Vulnerability Evaluation Pipeline
Runs K-Means (with Elbow/Silhouette sweep), HDBSCAN, and Deep Embedded Clustering (DEC),
compares metrics on Train/Val/Test (leakage-free), profiles vulnerability, and exports maps/figures.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp")

from data_loader import load_boundary, load_grid_profiles
from modeling import (
    train_kmeans,
    sweep_kmeans,
    train_hdbscan,
    predict_hdbscan_nearest,
    train_dec,
    predict_dec,
)
from evaluation import evaluate_clustering, compute_spatial_coherence, profile_clusters


def main():
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/maps", exist_ok=True)

    # 1. Load leakage-free splits
    train_scaled = pd.read_csv("data/split/train_scaled.csv")
    val_scaled = pd.read_csv("data/split/val_scaled.csv")
    test_scaled = pd.read_csv("data/split/test_scaled.csv")

    train_raw = pd.read_csv("data/split/train_raw.csv")
    val_raw = pd.read_csv("data/split/val_raw.csv")
    test_raw = pd.read_csv("data/split/test_raw.csv")

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

    print("=" * 72)
    print("FIREGUARD ID — CLUSTERING MODELING & VULNERABILITY EVALUATION")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 2. MODEL 1: K-MEANS SWEEP (k = 2..8)
    # ------------------------------------------------------------------
    print("\n[1/3] Eksperimen K-Means (k = 2 s.d. 8)...")
    sweep_df = pd.DataFrame(sweep_kmeans(X_train, X_val, k_range=range(2, 9), random_state=42))
    sweep_df.to_csv("reports/kmeans_sweep_metrics.csv", index=False)
    print(sweep_df[["k", "inertia", "train_silhouette", "val_silhouette", "train_dbi"]].to_string(index=False))

    # Plot Elbow & Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(sweep_df["k"], sweep_df["inertia"], marker="o", color="#2980b9", lw=2)
    axes[0].set_title("K-Means Elbow Method (Inertia)", fontweight="bold")
    axes[0].set_xlabel("Jumlah Klaster (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(sweep_df["k"], sweep_df["train_silhouette"], marker="o", label="Train", color="#e67e22", lw=2)
    axes[1].plot(sweep_df["k"], sweep_df["val_silhouette"], marker="s", label="Validation", color="#27ae60", lw=2)
    axes[1].set_title("K-Means Silhouette Score vs k", fontweight="bold")
    axes[1].set_xlabel("Jumlah Klaster (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("reports/figures/clustering_kmeans_elbow_silhouette.png", dpi=200)
    plt.close()

    # Best k by validation silhouette (and keep k=4 for 4-level risk baseline comparison)
    best_k = int(sweep_df.loc[sweep_df["val_silhouette"].idxmax(), "k"])
    print(f"   -> k terbaik berdasarkan Validation Silhouette: k={best_k}")

    km_best, km_train_lbl = train_kmeans(X_train, n_clusters=best_k, random_state=42)
    km_val_lbl = km_best.predict(X_val)
    km_test_lbl = km_best.predict(X_test)
    joblib.dump(km_best, "models/kmeans_best.joblib")

    # Also fit k=4 for 4-tier risk interpretability if best_k != 4
    km_4, km4_train_lbl = train_kmeans(X_train, n_clusters=4, random_state=42)
    km4_val_lbl = km_4.predict(X_val)
    km4_test_lbl = km_4.predict(X_test)
    joblib.dump(km_4, "models/kmeans_k4.joblib")

    # ------------------------------------------------------------------
    # 3. MODEL 2: HDBSCAN TUNING
    # ------------------------------------------------------------------
    print("\n[2/3] Eksperimen HDBSCAN (Density-Based Spatial Clustering)...")
    hdb_configs = [
        (10, 3), (10, 5), (15, 5), (15, 8), (20, 5), (25, 5), (30, 10)
    ]
    hdb_records = []
    best_hdb_model, best_hdb_lbl, best_hdb_score, best_hdb_cfg = None, None, -1.0, None

    for mcs, ms in hdb_configs:
        m, lbl = train_hdbscan(X_train, min_cluster_size=mcs, min_samples=ms)
        ev = evaluate_clustering(X_train, lbl)
        hdb_records.append({"min_cluster_size": mcs, "min_samples": ms, **ev})
        sil = ev["silhouette_score"] if ev["silhouette_score"] is not None else -1.0
        if ev["n_clusters"] >= 2 and ev["noise_ratio"] <= 0.35 and sil > best_hdb_score:
            best_hdb_score = sil
            best_hdb_model = m
            best_hdb_lbl = lbl
            best_hdb_cfg = (mcs, ms)

    hdb_tune_df = pd.DataFrame(hdb_records)
    hdb_tune_df.to_csv("reports/hdbscan_tuning_metrics.csv", index=False)
    print(hdb_tune_df.to_string(index=False))

    if best_hdb_model is None:
        best_hdb_cfg = (15, 5)
        best_hdb_model, best_hdb_lbl = train_hdbscan(X_train, min_cluster_size=15, min_samples=5)

    print(f"   -> Konfigurasi HDBSCAN terpilih: min_cluster_size={best_hdb_cfg[0]}, min_samples={best_hdb_cfg[1]}")
    hdb_val_lbl = predict_hdbscan_nearest(best_hdb_model, X_val)
    hdb_test_lbl = predict_hdbscan_nearest(best_hdb_model, X_test)

    # ------------------------------------------------------------------
    # 4. MODEL 3: DEEP EMBEDDED CLUSTERING (DEC)
    # ------------------------------------------------------------------
    print("\n[3/3] Eksperimen Deep Embedded Clustering (DEC - PyTorch Autoencoder)...")
    dec_model, dec_train_lbl, dec_train_z = train_dec(
        X_train, n_clusters=4, latent_dim=4, pretrain_epochs=120, refine_epochs=80, random_state=42
    )
    dec_val_lbl, dec_val_z, _ = predict_dec(dec_model, X_val)
    dec_test_lbl, dec_test_z, _ = predict_dec(dec_model, X_test)

    # ------------------------------------------------------------------
    # 5. EVALUASI & KOMPARASI MODEL (TRAIN, VAL, TEST)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("HASIL KOMPARASI MODEL CLUSTERING")
    print("=" * 72)

    comparison_rows = []
    models_eval = [
        (f"K-Means (k={best_k}, Optimal)", X_train, km_train_lbl, X_val, km_val_lbl, X_test, km_test_lbl),
        ("K-Means (k=4, Baseline)", X_train, km4_train_lbl, X_val, km4_val_lbl, X_test, km4_test_lbl),
        (f"HDBSCAN {best_hdb_cfg}", X_train, best_hdb_lbl, X_val, hdb_val_lbl, X_test, hdb_test_lbl),
        ("DEC (Latent Space, k=4)", dec_train_z, dec_train_lbl, dec_val_z, dec_val_lbl, dec_test_z, dec_test_lbl),
        ("DEC (Input Space, k=4)", X_train, dec_train_lbl, X_val, dec_val_lbl, X_test, dec_test_lbl),
    ]

    for name, Xtr, ltr, Xva, lva, Xte, lte in models_eval:
        for split_name, X_sp, l_sp, df_sp in [
            ("Train", Xtr, ltr, train_raw),
            ("Validation", Xva, lva, val_raw),
            ("Test", Xte, lte, test_raw),
        ]:
            ev = evaluate_clustering(X_sp, l_sp)
            df_tmp = df_sp.copy()
            df_tmp["cluster"] = l_sp
            coh = compute_spatial_coherence(df_tmp, label_col="cluster")
            comparison_rows.append({
                "model": name,
                "split": split_name,
                **ev,
                "spatial_coherence": coh,
            })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv("reports/clustering_comparison.csv", index=False)
    print(comp_df.to_string(index=False))

    # Plot Model Comparison on Test Set
    test_comp = comp_df[comp_df["split"] == "Test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    colors = ["#2980b9", "#3498db", "#8e44ad", "#e74c3c", "#e67e22"]

    bars1 = axes[0].barh(test_comp["model"], test_comp["silhouette_score"].fillna(0), color=colors)
    axes[0].set_title("Test Set: Silhouette Score (Lebih Tinggi = Lebih Baik)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Silhouette Score")
    axes[0].set_xlim(0, 1.05)
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)
    for b in bars1:
        w = b.get_width()
        axes[0].text(w + 0.02, b.get_y() + b.get_height()/2, f"{w:.3f}", va="center", fontsize=9)

    bars2 = axes[1].barh(test_comp["model"], test_comp["davies_bouldin_score"].fillna(0), color=colors)
    axes[1].set_title("Test Set: Davies-Bouldin Index (Lebih Rendah = Lebih Baik)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Davies-Bouldin Index")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)
    for b in bars2:
        w = b.get_width()
        axes[1].text(w + 0.02, b.get_y() + b.get_height()/2, f"{w:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("reports/figures/clustering_model_comparison.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # 6. PROFILING KERAWANAN & PRIORITISASI PADA SELURUH GRID RIAU (835 GRID)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PROFILING KERAWANAN & ZONA PRIORITAS (835 GRID RIAU)")
    print("=" * 72)

    scaler = joblib.load("data/split/scaler.joblib")
    df_all_grid = load_grid_profiles("data/processed/riau_vulnerability_grid_01deg.csv")
    X_all_scaled = scaler.transform(df_all_grid[raw_cols])

    # Assign labels from fitted K-Means (k=4), HDBSCAN, and DEC (no refitting!)
    df_all_grid["cluster_kmeans_k4"] = km_4.predict(X_all_scaled)
    df_all_grid["cluster_hdbscan"] = predict_hdbscan_nearest(best_hdb_model, X_all_scaled)
    dec_all_lbl, _, _ = predict_dec(dec_model, X_all_scaled)
    df_all_grid["cluster_dec"] = dec_all_lbl

    # Profile K-Means k=4 and DEC to assign human-readable vulnerability tiers
    prof_km4 = profile_clusters(df_all_grid, label_col="cluster_kmeans_k4", feature_cols=raw_cols)
    prof_dec = profile_clusters(df_all_grid, label_col="cluster_dec", feature_cols=raw_cols)

    print("\nProfil Klaster K-Means (k=4, Fit pada Train):")
    print(prof_km4)
    print("\nProfil Klaster DEC (k=4, Fit pada Train):")
    print(prof_dec)

    prof_km4.to_csv("reports/cluster_profile_kmeans_k4.csv")
    prof_dec.to_csv("reports/cluster_profile_dec.csv")

    # Map K-Means k=4 clusters ordered by hotspot_count_mean descending -> Priority 1..4
    risk_names = ["Sangat Tinggi (Prioritas 1)", "Tinggi (Prioritas 2)", "Sedang (Prioritas 3)", "Rendah (Prioritas 4)"]
    risk_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]

    ordered_clusters = list(prof_km4.index)
    cluster_to_risk = {cid: risk_names[i] for i, cid in enumerate(ordered_clusters)}
    cluster_to_rank = {cid: i + 1 for i, cid in enumerate(ordered_clusters)}
    cluster_to_color = {cid: risk_colors[i] for i, cid in enumerate(ordered_clusters)}

    df_all_grid["priority_zone_class"] = df_all_grid["cluster_kmeans_k4"].map(cluster_to_risk)
    df_all_grid["priority_rank"] = df_all_grid["cluster_kmeans_k4"].map(cluster_to_rank)
    df_all_grid["eval_color"] = df_all_grid["cluster_kmeans_k4"].map(cluster_to_color)

    df_all_grid.to_csv("data/processed/riau_vulnerability_clusters_evaluated.csv", index=False)
    print("\nDistribusi Zona Prioritas Akhir (835 Grid):")
    print(df_all_grid["priority_zone_class"].value_counts())

    # 7. Generate Clean Interactive Priority Map (Esri Dark Gray, Watermark-Free)
    riau_geo = load_boundary("data/processed/riau_boundary.geojson")
    grid_json = [
        {
            "id": r["grid_id"],
            "bounds": [[r["lat_min"], r["lon_min"]], [r["lat_max"], r["lon_max"]]],
            "zone": r["priority_zone_class"],
            "rank": int(r["priority_rank"]),
            "color": r["eval_color"],
            "hotspots": int(r["hotspot_count"]),
            "fire_days": int(r["fire_days"]),
            "mean_frp": float(r["mean_frp"]),
            "max_frp": float(r["max_frp"]),
            "dec_cluster": int(r["cluster_dec"]),
        }
        for _, r in df_all_grid.iterrows()
    ]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>FIREGUARD ID - Peta Evaluasi Zona Prioritas Kerawanan Riau</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: 'Segoe UI', sans-serif; }}
        #map {{ width: 100%; height: 100%; background: #111; }}
        .info-panel {{
            position: absolute; top: 20px; right: 20px;
            background: rgba(20, 20, 25, 0.92); color: #fff;
            padding: 18px 22px; border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.12);
            z-index: 1000; max-width: 330px;
        }}
        .info-panel h2 {{ margin: 0 0 6px 0; font-size: 18px; color: #ff5722; }}
        .info-panel p {{ margin: 4px 0; font-size: 12.5px; color: #ccc; }}
        .legend {{ margin-top: 12px; font-size: 12px; border-top: 1px solid #444; padding-top: 10px; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .box {{ width: 18px; height: 14px; border-radius: 3px; margin-right: 8px; display: inline-block; }}
        .leaflet-control-attribution {{ display: none !important; }}
        .leaflet-bottom.leaflet-right {{ display: none !important; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel">
        <h2>FIREGUARD ID — Evaluasi Zona</h2>
        <p><strong>Model Fit:</strong> Leakage-Free (Train Split 584 Grid)</p>
        <p><strong>Evaluasi:</strong> K-Means vs HDBSCAN vs DEC</p>
        <p><strong>Total Wilayah:</strong> {len(df_all_grid):,} grid (0.1°)</p>
        <div class="legend">
            <strong>Zona Prioritas Mitigasi:</strong>
            <div class="legend-item"><span class="box" style="background: #e74c3c;"></span> Sangat Tinggi - Prioritas 1 ({(df_all_grid['priority_rank']==1).sum()})</div>
            <div class="legend-item"><span class="box" style="background: #e67e22;"></span> Tinggi - Prioritas 2 ({(df_all_grid['priority_rank']==2).sum()})</div>
            <div class="legend-item"><span class="box" style="background: #f1c40f;"></span> Sedang - Prioritas 3 ({(df_all_grid['priority_rank']==3).sum()})</div>
            <div class="legend-item"><span class="box" style="background: #2ecc71;"></span> Rendah - Prioritas 4 ({(df_all_grid['priority_rank']==4).sum()})</div>
        </div>
    </div>
    <script>
        var map = L.map('map', {{attributionControl: false}}).setView([0.5071, 101.4478], 8);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 16, attribution: ''
        }}).addTo(map);
        L.geoJSON({json.dumps(riau_geo)}, {{
            style: {{ color: '#ffffff', weight: 1.5, fillColor: 'transparent', dashArray: '4' }}
        }}).addTo(map);
        var grids = {json.dumps(grid_json)};
        grids.forEach(function(g) {{
            var rect = L.rectangle(g.bounds, {{
                color: g.color, weight: 0.8, fillColor: g.color,
                fillOpacity: g.rank === 4 ? 0.15 : 0.68
            }}).addTo(map);
            rect.bindPopup(
                "<strong>Grid ID: " + g.id + "</strong><br>" +
                "Zona: <b style='color:" + g.color + "'>" + g.zone + "</b><br>" +
                "<hr style='margin:4px 0; border:0; border-top:1px solid #ccc;'>" +
                "Hotspot (2021-2025): " + g.hotspots + " titik<br>" +
                "Hari Api Aktif: " + g.fire_days + " hari<br>" +
                "Mean FRP: " + g.mean_frp + " MW | Max FRP: " + g.max_frp + " MW<br>" +
                "DEC Cluster ID: " + g.dec_cluster
            );
        }});
    </script>
</body>
</html>
"""
    with open("outputs/maps/peta_evaluasi_cluster_riau.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Peta evaluasi zona prioritas disimpan di: outputs/maps/peta_evaluasi_cluster_riau.html")


if __name__ == "__main__":
    main()
