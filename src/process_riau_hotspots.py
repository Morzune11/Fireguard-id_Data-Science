import json
import glob
import os
import pandas as pd
from matplotlib.path import Path

print("1. Memuat batas wilayah resmi Provinsi Riau...")
with open("riau_boundary.geojson", "r", encoding="utf-8") as f:
    riau_geo = json.load(f)

feature = riau_geo["features"][0]
geom = feature["geometry"]

# Ekstrak semua poligon (MultiPolygon / Polygon)
polygons = []
if geom["type"] == "Polygon":
    for ring in geom["coordinates"]:
        polygons.append(Path(ring))
elif geom["type"] == "MultiPolygon":
    for poly in geom["coordinates"]:
        for ring in poly:
            polygons.append(Path(ring))

print(f"   Batas Riau berhasil dimuat ({len(polygons)} cincin/poligon kepulauan/daratan).")

# 2. Baca seluruh data VIIRS Indonesia (2021-2025)
files = sorted(glob.glob("viirs-jpss1_*_Indonesia.csv"))
print(f"\n2. Menggabungkan data VIIRS Indonesia dari {len(files)} file:")
dfs = []
for file in files:
    df_temp = pd.read_csv(file)
    print(f"   - {file}: {len(df_temp):,} records")
    dfs.append(df_temp)

df_all = pd.concat(dfs, ignore_index=True)
print(f"   Total deteksi seluruh Indonesia (2021-2025): {len(df_all):,} records")

# 3. Bounding box filtering cepat untuk efisiensi
# Bounding box Riau: Lat: ~-1.2 s.d 2.6, Lon: ~100.0 s.d 104.0
min_lon, max_lon = 100.0, 104.0
min_lat, max_lat = -1.2, 2.6

df_bbox = df_all[
    (df_all["longitude"] >= min_lon) & (df_all["longitude"] <= max_lon) &
    (df_all["latitude"] >= min_lat) & (df_all["latitude"] <= max_lat)
].copy()
print(f"\n3. Setelah filtering Bounding Box Riau: {len(df_bbox):,} kandidat titik api")

# 4. Exact Polygon Point-in-Polygon Check
print("4. Melakukan Exact Polygon Clipping (Point-in-Polygon)...")
points = df_bbox[["longitude", "latitude"]].values

inside_mask = [False] * len(points)
for poly_path in polygons:
    # Periksa apakah titik berada di dalam poligon
    mask = poly_path.contains_points(points)
    inside_mask = [a or b for a, b in zip(inside_mask, mask)]

df_riau = df_bbox[inside_mask].copy()
print(f"   Hasil Akhir: Ditemukan {len(df_riau):,} titik api persis di wilayah Provinsi Riau!")

# Simpan hasil olahan ke CSV
output_csv = "riau_viirs_hotspots_2021_2025.csv"
df_riau.to_csv(output_csv, index=False)
print(f"\n5. Data titik api Riau berhasil disimpan ke: {output_csv}")

# Preview data
print("\nRingkasan Titik Api per Tahun di Riau:")
df_riau['year'] = pd.to_datetime(df_riau['acq_date']).dt.year
print(df_riau['year'].value_counts().sort_index())
