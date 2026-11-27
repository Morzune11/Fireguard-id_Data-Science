"""
FIREGUARD ID — Fase 3: Spatial Prioritization Engine
Menghitung skor prioritas komposit (MCDA) yang transparan berdasarkan:
1. Riwayat & Rekurensi Kebakaran (40%)
2. Intensitas Energi Api & Reliabilitas Deteksi (25%)
3. Konsensus Multi-Model Clustering K-Means & DEC (20%)
4. Risiko Spillover Spasial Tetangga Terdekat (15%)

Output:
- data/processed/riau_spatial_prioritization.csv
- reports/top_20_priority_grids.csv
- reports/figures/spatial_prioritization_breakdown.png
- outputs/maps/peta_zona_prioritas_riau.html
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp")

from data_loader import load_boundary, load_hotspots


def minmax_norm(series):
    """Normalisasi Min-Max ke rentang [0, 1] dengan pengaman pembagian nol."""
    s_min, s_max = series.min(), series.max()
    if s_max - s_min == 0:
        return np.zeros(len(series))
    return (series - s_min) / (s_max - s_min)


def compute_recurrence_features(df_grids, df_hotspots):
    """Menghitung jumlah tahun aktif (fire recurrence) dan rasio tren terkini per grid."""
    df_h = df_hotspots.copy()
    df_h["year"] = pd.to_datetime(df_h["acq_date"]).dt.year

    active_years_list = []
    recent_hotspots_list = []

    for _, g in df_grids.iterrows():
        pts = df_h[
            (df_h["longitude"] >= g["lon_min"]) & (df_h["longitude"] < g["lon_max"]) &
            (df_h["latitude"] >= g["lat_min"]) & (df_h["latitude"] < g["lat_max"])
        ]
        active_years_list.append(int(pts["year"].nunique()))
        recent_hotspots_list.append(int((pts["year"] >= 2023).sum()))

    df_out = df_grids.copy()
    df_out["active_years"] = active_years_list
    df_out["recent_hotspots_2023_2025"] = recent_hotspots_list
    return df_out


def main():
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("outputs/maps", exist_ok=True)

    print("=" * 72)
    print("FIREGUARD ID — FASE 3: SPATIAL PRIORITIZATION")
    print("=" * 72)

    # 1. Muat dataset hasil evaluasi clustering (835 grid) & raw hotspots Riau
    df_grid = pd.read_csv("data/processed/riau_vulnerability_clusters_evaluated.csv")
    df_hotspots = load_hotspots("data/processed/riau_viirs_hotspots_2021_2025.csv")

    # 2. Hitung rekurensi tahunan (2021-2025)
    print("\n[1/4] Menghitung indikator rekurensi tahunan (active_years)...")
    df_grid = compute_recurrence_features(df_grid, df_hotspots)

    # 3. Hitung risiko spillover spasial dari 4 grid tetangga terdekat
    print("[2/4] Menghitung konteks distribusi spasial (neighbor spillover)...")
    coords = df_grid[["center_lat", "center_lon"]].values
    nn = NearestNeighbors(n_neighbors=5).fit(coords)
    _, indices = nn.kneighbors(coords)
    neighbor_mean_hotspots = [
        float(df_grid.iloc[nbrs[1:]]["hotspot_count"].mean())
        for nbrs in indices
    ]
    df_grid["neighbor_mean_hotspots"] = np.round(neighbor_mean_hotspots, 2)

    # 4. Hitung Skor Komponen (Skala 0 - 1, menggunakan log1p untuk fitur right-skewed)
    print("[3/4] Menghitung Priority Score (MCDA Transparan) & Priority Rank...")
    hist_score = (
        0.45 * minmax_norm(np.log1p(df_grid["hotspot_count"])) +
        0.35 * minmax_norm(np.log1p(df_grid["fire_days"])) +
        0.20 * minmax_norm(df_grid["active_years"])
    )
    intensity_score = (
        0.40 * minmax_norm(np.log1p(df_grid["mean_frp"])) +
        0.30 * minmax_norm(np.log1p(df_grid["max_frp"])) +
        0.30 * minmax_norm(np.log1p(df_grid["high_conf_count"]))
    )

    # Konsensus Model Clustering: konversi cluster K-Means & DEC menjadi skor ordinal [0..1]
    km_order = df_grid.groupby("cluster_kmeans_k4")["hotspot_count"].mean().rank().to_dict()
    dec_order = df_grid.groupby("cluster_dec")["hotspot_count"].mean().rank().to_dict()
    km_norm = minmax_norm(df_grid["cluster_kmeans_k4"].map(km_order))
    dec_norm = minmax_norm(df_grid["cluster_dec"].map(dec_order))
    model_consensus_score = 0.5 * km_norm + 0.5 * dec_norm

    spatial_context_score = minmax_norm(np.log1p(df_grid["neighbor_mean_hotspots"]))

    # Skor Prioritas Akhir (0 - 100)
    df_grid["priority_score"] = np.round(
        100.0 * (
            0.40 * hist_score +
            0.25 * intensity_score +
            0.20 * model_consensus_score +
            0.15 * spatial_context_score
        ),
        2,
    )

    # Ranking 1..835 (1 = paling prioritas)
    df_grid = df_grid.sort_values(
        ["priority_score", "hotspot_count", "mean_frp"], ascending=[False, False, False]
    ).reset_index(drop=True)
    df_grid["priority_rank"] = df_grid.index + 1

    # Aturan Klasifikasi Zona Prioritas Transparan
    def assign_priority_class(score):
        if score >= 65.0:
            return "Prioritas 1 - Kritis"
        elif score >= 45.0:
            return "Prioritas 2 - Tinggi"
        elif score >= 25.0:
            return "Prioritas 3 - Sedang"
        else:
            return "Prioritas 4 - Rendah"

    color_map = {
        "Prioritas 1 - Kritis": "#e74c3c",
        "Prioritas 2 - Tinggi": "#e67e22",
        "Prioritas 3 - Sedang": "#f1c40f",
        "Prioritas 4 - Rendah": "#2ecc71",
    }

    df_grid["priority_zone_class"] = df_grid["priority_score"].apply(assign_priority_class)
    df_grid["priority_color"] = df_grid["priority_zone_class"].map(color_map)

    # Simpan dataset prioritas & Top-20
    out_csv = "data/processed/riau_spatial_prioritization.csv"
    df_grid.to_csv(out_csv, index=False)

    top20_cols = [
        "priority_rank", "grid_id", "center_lat", "center_lon",
        "priority_score", "priority_zone_class", "hotspot_count",
        "fire_days", "active_years", "mean_frp", "max_frp",
        "high_conf_count", "neighbor_mean_hotspots",
    ]
    df_grid[top20_cols].head(20).to_csv("reports/top_20_priority_grids.csv", index=False)

    print("\nDistribusi Zona Prioritas (MCDA Fase 3):")
    print(df_grid["priority_zone_class"].value_counts())
    print("\nTop 10 Grid Paling Mendesak:")
    print(df_grid[top20_cols[:9]].head(10).to_string(index=False))

    # 5. Visualisasi Distribusi Priority Score & Top-15 Grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    zone_order = ["Prioritas 1 - Kritis", "Prioritas 2 - Tinggi", "Prioritas 3 - Sedang", "Prioritas 4 - Rendah"]
    counts = [int((df_grid["priority_zone_class"] == z).sum()) for z in zone_order]
    colors = [color_map[z] for z in zone_order]

    bars = axes[0].bar(zone_order, counts, color=colors, edgecolor="#333")
    axes[0].set_title("Distribusi Zona Prioritas Mitigasi (835 Grid)", fontweight="bold")
    axes[0].set_ylabel("Jumlah Grid")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 6, f"{int(b.get_height())}", ha="center", fontweight="bold")

    top15 = df_grid.head(15).iloc[::-1]
    axes[1].barh(top15["grid_id"], top15["priority_score"], color=top15["priority_color"])
    axes[1].set_title("Top 15 Grid Prioritas Tertinggi (Priority Score)", fontweight="bold")
    axes[1].set_xlabel("Priority Score (0–100)")
    axes[1].set_xlim(0, 105)
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("reports/figures/spatial_prioritization_breakdown.png", dpi=200)
    plt.close()

    # 6. Peta Interaktif Zona Prioritas (Esri Dark Gray, Watermark-Free)
    print("\n[4/4] Membuat peta interaktif Zona Prioritas Riau...")
    riau_geo = load_boundary("data/processed/riau_boundary.geojson")
    grid_json = [
        {
            "id": r["grid_id"],
            "bounds": [[r["lat_min"], r["lon_min"]], [r["lat_max"], r["lon_max"]]],
            "zone": r["priority_zone_class"],
            "rank": int(r["priority_rank"]),
            "score": float(r["priority_score"]),
            "color": r["priority_color"],
            "hotspots": int(r["hotspot_count"]),
            "fire_days": int(r["fire_days"]),
            "active_years": int(r["active_years"]),
            "mean_frp": float(r["mean_frp"]),
            "neighbors": float(r["neighbor_mean_hotspots"]),
        }
        for _, r in df_grid.iterrows()
    ]

    c1 = int((df_grid["priority_zone_class"] == "Prioritas 1 - Kritis").sum())
    c2 = int((df_grid["priority_zone_class"] == "Prioritas 2 - Tinggi").sum())
    c3 = int((df_grid["priority_zone_class"] == "Prioritas 3 - Sedang").sum())
    c4 = int((df_grid["priority_zone_class"] == "Prioritas 4 - Rendah").sum())

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>FIREGUARD ID - Peta Zona Prioritas Mitigasi Karhutla Riau</title>
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
            z-index: 1000; max-width: 340px;
        }}
        .info-panel h2 {{ margin: 0 0 6px 0; font-size: 18px; color: #ff5722; }}
        .info-panel p {{ margin: 4px 0; font-size: 12px; color: #ccc; }}
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
        <h2>FIREGUARD ID — Fase 3</h2>
        <p><strong>Spatial Prioritization (MCDA)</strong></p>
        <p>40% Riwayat + 25% Intensitas + 20% Klaster + 15% Spasial</p>
        <div class="legend">
            <strong>Klasifikasi Zona Prioritas:</strong>
            <div class="legend-item"><span class="box" style="background: #e74c3c;"></span> Prioritas 1 - Kritis ({c1} grid)</div>
            <div class="legend-item"><span class="box" style="background: #e67e22;"></span> Prioritas 2 - Tinggi ({c2} grid)</div>
            <div class="legend-item"><span class="box" style="background: #f1c40f;"></span> Prioritas 3 - Sedang ({c3} grid)</div>
            <div class="legend-item"><span class="box" style="background: #2ecc71;"></span> Prioritas 4 - Rendah ({c4} grid)</div>
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
                fillOpacity: g.zone.includes('Rendah') ? 0.15 : 0.72
            }}).addTo(map);
            rect.bindPopup(
                "<strong>Rank #" + g.rank + " — " + g.id + "</strong><br>" +
                "Zona: <b style='color:" + g.color + "'>" + g.zone + "</b><br>" +
                "Priority Score: <b>" + g.score + " / 100</b><br>" +
                "<hr style='margin:4px 0; border:0; border-top:1px solid #ccc;'>" +
                "Hotspot (2021-2025): " + g.hotspots + " titik<br>" +
                "Tahun Aktif: " + g.active_years + " / 5 tahun<br>" +
                "Hari Api: " + g.fire_days + " hari | Mean FRP: " + g.mean_frp + " MW<br>" +
                "Rata-rata Tetangga: " + g.neighbors + " hotspot"
            );
        }});
    </script>
</body>
</html>
"""
    with open("outputs/maps/peta_zona_prioritas_riau.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dataset prioritas disimpan ke: {out_csv}")
    print("Peta zona prioritas disimpan ke: outputs/maps/peta_zona_prioritas_riau.html")


if __name__ == "__main__":
    main()
