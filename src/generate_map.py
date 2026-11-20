import json
import pandas as pd

# 1. Baca data titik api Riau
df = pd.read_csv("riau_viirs_hotspots_2021_2025.csv")

# 2. Baca batas Riau GeoJSON
with open("riau_boundary.geojson", "r", encoding="utf-8") as f:
    riau_geo = json.load(f)

# Siapkan data titik untuk visualisasi (ambil kolom penting untuk efisiensi render)
points_data = []
for _, row in df.iterrows():
    # Klasifikasi warna berdasarkan confidence (n: nominal, h: high, l: low)
    conf = str(row["confidence"]).lower()
    color = "red" if conf == "h" else ("orange" if conf == "n" else "gold")
    points_data.append({
        "lat": float(row["latitude"]),
        "lon": float(row["longitude"]),
        "frp": float(row["frp"]) if pd.notnull(row["frp"]) else 0.0,
        "date": str(row["acq_date"]),
        "conf": conf,
        "color": color
    })

# Format HTML dengan Leaflet.js mandiri
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>FIREGUARD ID - Peta Titik Api Provinsi Riau</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        #map {{ width: 100%; height: 100%; background: #1a1a1a; }}
        .info-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(20, 20, 25, 0.88);
            color: #fff;
            padding: 18px 22px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.1);
            z-index: 1000;
            max-width: 320px;
        }}
        .info-panel h2 {{ margin: 0 0 8px 0; font-size: 20px; color: #ff5722; }}
        .info-panel p {{ margin: 4px 0; font-size: 13px; color: #ccc; }}
        .stat-badge {{ display: inline-block; background: #333; padding: 3px 8px; border-radius: 4px; font-weight: bold; color: #00e676; }}
        .legend {{ margin-top: 12px; font-size: 12px; border-top: 1px solid #444; padding-top: 10px; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
        .dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; display: inline-block; }}
        .leaflet-control-attribution {{ display: none !important; }}
        .leaflet-bottom.leaflet-right {{ display: none !important; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-panel">
        <h2>FIREGUARD ID</h2>
        <p><strong>Pilot Wilayah:</strong> Provinsi Riau</p>
        <p><strong>Periode:</strong> 2021 – 2025 (VIIRS S-NPP / JPSS)</p>
        <p><strong>Total Titik Api:</strong> <span class="stat-badge">{len(points_data):,} Deteksi</span></p>
        
        <div class="legend">
            <strong>Tingkat Confidence:</strong>
            <div class="legend-item"><span class="dot" style="background: red;"></span> High Confidence</div>
            <div class="legend-item"><span class="dot" style="background: orange;"></span> Nominal Confidence</div>
            <div class="legend-item"><span class="dot" style="background: gold;"></span> Low Confidence</div>
        </div>
    </div>

    <script>
        var map = L.map('map', {{attributionControl: false}}).setView([0.5071, 101.4478], 8);

        // Basemap Gratis Tanpa API Key (OSM Dark via Carto CDN publik / Stadia / Esri / OSM)
        // Menggunakan tile Esri Dark Gray yang 100% bebas watermark dan tanpa API key
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 16,
            attribution: ''
        }}).addTo(map);

        // Batas Riau GeoJSON
        var riauBoundary = {json.dumps(riau_geo)};
        L.geoJSON(riauBoundary, {{
            style: {{
                color: '#00e5ff',
                weight: 2,
                fillColor: '#00e5ff',
                fillOpacity: 0.05
            }}
        }}).addTo(map);

        // Titik-titik Hotspot
        var points = {json.dumps(points_data)};
        points.forEach(function(p) {{
            L.circleMarker([p.lat, p.lon], {{
                radius: Math.min(Math.max(p.frp / 10, 3), 12),
                fillColor: p.color,
                color: p.color,
                weight: 1,
                opacity: 0.8,
                fillOpacity: 0.65
            }}).bindPopup(
                "<strong>Hotspot Deteksi VIIRS</strong><br>" +
                "Tanggal: " + p.date + "<br>" +
                "Confidence: " + p.conf + "<br>" +
                "Fire Radiative Power (FRP): " + p.frp + " MW<br>" +
                "Koordinat: " + p.lat.toFixed(4) + ", " + p.lon.toFixed(4)
            ).addTo(map);
        }});
    </script>
</body>
</html>
"""

with open("peta_titik_api_riau.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Peta interaktif berhasil digenerate di: peta_titik_api_riau.html")
