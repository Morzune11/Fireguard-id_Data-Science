import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp")

print("="*70)
print("BAB 4: DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS (EDA)")
print("="*70)

# 1. Membaca Data
df_hotspots = pd.read_csv("riau_viirs_hotspots_2021_2025.csv")
df_grid = pd.read_csv("riau_vulnerability_grid_01deg.csv")

# 4.1 & 4.2 Informasi Dataset & Deskripsi Fitur
print("\n--- 4.1 & 4.2 Ringkasan Dataset ---")
print(f"Dataset 1 (Raw Deteksi Hotspot Riau): {df_hotspots.shape[0]:,} baris x {df_hotspots.shape[1]} kolom")
print(f"Dataset 2 (Grid Kerawanan 0.1°): {df_grid.shape[0]:,} baris x {df_grid.shape[1]} kolom")

# 4.3 Analisis Kondisi Data
print("\n--- 4.3 Analisis Kondisi Data (Quality Assessment) ---")
print("Missing values (Hotspots):", df_hotspots.isnull().sum().to_dict())
print("Missing values (Grid):", df_grid.isnull().sum().to_dict())
print("Duplikat (Hotspots):", df_hotspots.duplicated().sum())
print("Duplikat (Grid):", df_grid.duplicated().sum())

# 4.4 EDA Visualisasi & Insight
print("\n--- 4.4 Menghasilkan 3 Visualisasi EDA ---")
fig_dir = os.path.join("outputs", "figures")
os.makedirs(fig_dir, exist_ok=True)

# Visualisasi 1: Tren Titik Api Tahunan & Bulanan di Riau
df_hotspots["acq_date"] = pd.to_datetime(df_hotspots["acq_date"])
df_hotspots["year"] = df_hotspots["acq_date"].dt.year
df_hotspots["month"] = df_hotspots["acq_date"].dt.month

yearly_counts = df_hotspots["year"].value_counts().sort_index()

plt.figure(figsize=(9, 4.5))
bars = plt.bar(yearly_counts.index.astype(str), yearly_counts.values, color="#e67e22", edgecolor="#d35400", width=0.55)
plt.title("Tren Frekuensi Deteksi Titik Api di Provinsi Riau (2021 - 2025)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Tahun", fontsize=11)
plt.ylabel("Jumlah Deteksi Hotspot", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:,}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig1_path = os.path.join(fig_dir, "eda_1_tren_tahunan.png")
plt.savefig(fig1_path, dpi=200)
plt.close()
print(f"  [1] Gambar 1 disimpan: {fig1_path}")

# Visualisasi 2: Siklus Musiman Karhutla Bulanan di Riau
monthly_counts = df_hotspots["month"].value_counts().sort_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']

plt.figure(figsize=(10, 4.5))
plt.plot(month_names, [monthly_counts.get(m, 0) for m in range(1, 13)], marker='o', color="#c0392b", linewidth=2.5, markersize=7)
plt.fill_between(month_names, [monthly_counts.get(m, 0) for m in range(1, 13)], color="#e74c3c", alpha=0.25)
plt.title("Siklus Musiman Titik Api Bulanan di Provinsi Riau", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Bulan", fontsize=11)
plt.ylabel("Total Akumulasi Hotspot", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
fig2_path = os.path.join(fig_dir, "eda_2_siklus_musiman.png")
plt.savefig(fig2_path, dpi=200)
plt.close()
print(f"  [2] Gambar 2 disimpan: {fig2_path}")

# Visualisasi 3: Distribusi Fitur Kerawanan Grid (Imbalance & Skewness)
plt.figure(figsize=(11, 4.5))

plt.subplot(1, 2, 1)
plt.hist(df_grid["hotspot_count"], bins=30, color="#2980b9", edgecolor="#1b4f72", log=True)
plt.title("Distribusi Frekuensi Hotspot per Grid (Log-scale)", fontsize=11, fontweight='bold')
plt.xlabel("Jumlah Hotspot per Grid")
plt.ylabel("Frekuensi Grid (Log)")
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
risk_counts = df_grid["risk_level"].value_counts()[["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]]
colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
plt.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor="#333", width=0.6)
plt.title("Distribusi Kelas Kerawanan Grid (Imbalance)", fontsize=11, fontweight='bold')
plt.xlabel("Tingkat Kerawanan")
plt.ylabel("Jumlah Grid Unit")
plt.grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(risk_counts.values):
    plt.text(i, v + 8, str(v), ha='center', fontweight='bold')

plt.tight_layout()
fig3_path = os.path.join(fig_dir, "eda_3_distribusi_grid.png")
plt.savefig(fig3_path, dpi=200)
plt.close()
print(f"  [3] Gambar 3 disimpan: {fig3_path}")
print("\nEDA Selesai!")
