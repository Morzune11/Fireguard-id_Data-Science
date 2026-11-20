import json
import os
import numpy as np
import pandas as pd
from matplotlib.path import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("=================================================================")
print("FIREGUARD ID - Tahap 2: Spatial Grid (0.1°) & Kerawanan Karhutla")
print("=================================================================")

# 1. Muat batas wilayah Riau
with open("riau_boundary.geojson", "r", encoding="utf-8") as f:
    riau_geo = json.load(f)

geom = riau_geo["features"][0]["geometry"]
polygons = []
if geom["type"] == "Polygon":
    for ring in geom["coordinates"]:
        polygons.append(Path(ring))
elif geom["type"] == "MultiPolygon":
    for poly in geom["coordinates"]:
        for ring in poly:
            polygons.append(Path(ring))

# Bounding box Riau: Lat [-1.2, 2.6], Lon [100.0, 104.0]
grid_res = 0.1  # resolusi 0.1 derajat (~11 km)
lons = np.arange(100.0, 104.0 + grid_res, grid_res)
lats = np.arange(-1.2, 2.6 + grid_res, grid_res)

print(f"\n1. Membuat Fishnet Grid 0.1° x 0.1°...")
grid_cells = []
grid_id = 0

for i in range(len(lons) - 1):
    for j in range(len(lats) - 1):
        lon_min, lon_max = round(lons[i], 4), round(lons[i+1], 4)
        lat_min, lat_max = round(lats[j], 4), round(lats[j+1], 4)
        center_lon = round((lon_min + lon_max) / 2.0, 4)
        center_lat = round((lat_min + lat_max) / 2.0, 4)
        
        # Periksa apakah pusat grid atau sudutnya ada di dalam wilayah Riau
        test_points = [
            [center_lon, center_lat],
            [lon_min, lat_min], [lon_min, lat_max],
            [lon_max, lat_min], [lon_max, lat_max]
        ]
        
        inside = False
        for poly_path in polygons:
            if any(poly_path.contains_points(test_points)):
                inside = True
                break
                
        if inside:
            grid_cells.append({
                "grid_id": f"GRID_{grid_id:04d}",
                "lon_min": lon_min, "lon_max": lon_max,
                "lat_min": lat_min, "lat_max": lat_max,
                "center_lon": center_lon,
                "center_lat": center_lat
            })
            grid_id += 1

df_grids = pd.DataFrame(grid_cells)
print(f"   Terbentuk {len(df_grids):,} unit grid 0.1° yang mencakup daratan Riau.")

# 2. Agregasi data hotspot historis (2021-2025) ke masing-masing grid
print("\n2. Mengagregasi data hotspot VIIRS (2021-2025) ke setiap grid...")
df_hotspots = pd.read_csv("riau_viirs_hotspots_2021_2025.csv")
df_hotspots["acq_date"] = pd.to_datetime(df_hotspots["acq_date"])

# Hitung statistik per grid
features = []
for idx, grid in df_grids.iterrows():
    # Filter titik di dalam kotak grid
    pts = df_hotspots[
        (df_hotspots["longitude"] >= grid["lon_min"]) & (df_hotspots["longitude"] < grid["lon_max"]) &
        (df_hotspots["latitude"] >= grid["lat_min"]) & (df_hotspots["latitude"] < grid["lat_max"])
    ]
    
    hotspot_count = len(pts)
    fire_days = pts["acq_date"].dt.date.nunique()
    mean_frp = float(pts["frp"].mean()) if hotspot_count > 0 else 0.0
    max_frp = float(pts["frp"].max()) if hotspot_count > 0 else 0.0
    high_conf_count = len(pts[pts["confidence"] == "h"])
    
    features.append({
        "hotspot_count": hotspot_count,
        "fire_days": fire_days,
        "mean_frp": round(mean_frp, 2),
        "max_frp": round(max_frp, 2),
        "high_conf_count": high_conf_count
    })

df_features = pd.DataFrame(features)
df_grid_profiles = pd.concat([df_grids, df_features], axis=1)

# 3. Spatial Clustering Kerawanan (K-Means)
print("\n3. Menjalankan Model 1: K-Means Clustering untuk Kerawanan...")
X = df_grid_profiles[["hotspot_count", "fire_days", "mean_frp", "max_frp"]].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4 Cluster: Rendah, Sedang, Tinggi, Sangat Tinggi
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)

# Petakan cluster ke tingkatan risiko berdasarkan urutan rata-rata hotspot_count
df_grid_profiles["raw_cluster"] = raw_labels
cluster_risk_order = (
    df_grid_profiles.groupby("raw_cluster")["hotspot_count"]
    .mean()
    .sort_values()
    .index
)

risk_map = {
    cluster_risk_order[0]: "Rendah",
    cluster_risk_order[1]: "Sedang",
    cluster_risk_order[2]: "Tinggi",
    cluster_risk_order[3]: "Sangat Tinggi"
}
color_map = {
    "Rendah": "#2ecc71",       # Hijau
    "Sedang": "#f1c40f",       # Kuning
    "Tinggi": "#e67e22",       # Oranye
    "Sangat Tinggi": "#e74c3c" # Merah
}

df_grid_profiles["risk_level"] = df_grid_profiles["raw_cluster"].map(risk_map)
df_grid_profiles["color"] = df_grid_profiles["risk_level"].map(color_map)

# Simpan hasil dataset grid ke CSV
output_profile = "riau_vulnerability_grid_01deg.csv"
df_grid_profiles.to_csv(output_profile, index=False)
print(f"   Profil kerawanan grid berhasil disimpan ke: {output_profile}")

print("\nDistribusi Tingkat Kerawanan Grid:")
print(df_grid_profiles["risk_level"].value_counts())

# 4. Buat Peta Choropleth Interaktif Grid Riau
print("\n4. Membuat visualisasi peta zonasi kerawanan grid...")
grid_data_json = []
for _, row in df_grid_profiles.iterrows():
    grid_data_json.append({
        "id": row["grid_id"],
        "bounds": [
            [row["lat_min"], row["lon_min"]],
            [row["lat_max"], row["lon_max"]]
        ],
        "center": [row["center_lat"], row["center_lon"]],
        "level": row["risk_level"],
        "color": row["color"],
        "hotspots": int(row["hotspot_count"]),
        "fire_days": int(row["fire_days"]),
        "mean_frp": float(row["mean_frp"]),
        "max_frp": float(row["max_frp"])
    })

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>FIREGUARD ID - Peta Zonasi Kerawanan Karhutla Grid 0.1°</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        #map {{ width: 100%; height: 100%; background: #111; }}
        .info-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(20, 20, 25, 0.9);
            color: #fff;
            padding: 18px 22px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.1);
            z-index: 1000;
            max-width: 320px;
        }}
        .info-panel h2 {{ margin: 0 0 6px 0; font-size: 19px; color: #ff5722; }}
        .info-panel p {{ margin: 4px 0; font-size: 12.5px; color: #bbb; }}
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
        <h2>FIREGUARD ID</h2>
        <p><strong>Pilot:</strong> Provinsi Riau</p>
        <p><strong>Resolusi Analisis:</strong> Grid 0.1° (~11 x 11 km)</p>
        <p><strong>Model:</strong> K-Means Spatial Clustering</p>
        <p><strong>Total Grid:</strong> {len(df_grid_profiles):,} zona</p>
        
        <div class="legend">
            <strong>Tingkat Kerawanan Grid:</strong>
            <div class="legend-item"><span class="box" style="background: #e74c3c;"></span> Sangat Tinggi ({len(df_grid_profiles[df_grid_profiles['risk_level'] == 'Sangat Tinggi'])})</div>
            <div class="legend-item"><span class="box" style="background: #e67e22;"></span> Tinggi ({len(df_grid_profiles[df_grid_profiles['risk_level'] == 'Tinggi'])})</div>
            <div class="legend-item"><span class="box" style="background: #f1c40f;"></span> Sedang ({len(df_grid_profiles[df_grid_profiles['risk_level'] == 'Sedang'])})</div>
            <div class="legend-item"><span class="box" style="background: #2ecc71;"></span> Rendah ({len(df_grid_profiles[df_grid_profiles['risk_level'] == 'Rendah'])})</div>
        </div>
    </div>

    <script>
        var map = L.map('map', {{attributionControl: false}}).setView([0.5071, 101.4478], 8);

        // Basemap Esri Dark Gray (Bebas Watermark & Tanpa API Key)
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 16,
            attribution: ''
        }}).addTo(map);

        // Batas Luar Riau
        var riauBoundary = {json.dumps(riau_geo)};
        L.geoJSON(riauBoundary, {{
            style: {{
                color: '#ffffff',
                weight: 1.5,
                fillColor: 'transparent',
                dashArray: '4'
            }}
        }}).addTo(map);

        // Render Grid Cells
        var grids = {json.dumps(grid_data_json)};
        grids.forEach(function(g) {{
            var rect = L.rectangle(g.bounds, {{
                color: g.color,
                weight: 0.8,
                fillColor: g.color,
                fillOpacity: g.level === 'Rendah' ? 0.15 : 0.65
            }}).addTo(map);

            rect.bindPopup(
                "<strong>Grid ID: " + g.id + "</strong><br>" +
                "Tingkat Kerawanan: <b style='color:" + g.color + "'>" + g.level + "</b><br>" +
                "<hr style='margin:4px 0; border:0; border-top:1px solid #ccc;'>" +
                "Total Hotspot (2021-2025): " + g.hotspots + " titik<br>" +
                "Jumlah Hari Api: " + g.fire_days + " hari<br>" +
                "Rata-rata FRP: " + g.mean_frp + " MW<br>" +
                "Max FRP: " + g.max_frp + " MW"
            );
        }});
    </script>
</body>
</html>
"""

output_map = os.path.join("outputs", "maps", "peta_kerawanan_grid_riau.html")
with open(output_map, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"5. Peta Zonasi Grid berhasil dibuat di: {output_map}")
