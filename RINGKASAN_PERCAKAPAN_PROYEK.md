# Ringkasan Proyek & Log Aktivitas: FIREGUARD ID

**Nama Proyek:** FIREGUARD ID (*Forest & Land Fire Intelligence, Response, and Resource Optimization System*)  
**Wilayah Pilot:** Provinsi Riau, Indonesia  
**Fokus Masalah:** Mitigasi Kebakaran Hutan dan Lahan (Karhutla)  
**Tujuan Proyek:** Mengidentifikasi profil kerawanan wilayah menggunakan pendekatan *unsupervised learning / spatial clustering* berbasis grid 0,1° dan mengoptimalkan penempatan fasilitas tanggap darurat (*resource & facility location optimization*).  
**Dukungan SDG:** SDG 13 (Climate Action) — Target 13.1  

---

## 1. Kronologi & Alur Percakapan

### Sesi 1: Penyiapan Pipeline Geospasial & Pemotongan Data
- **Permintaan Pengguna:** Meminta petunjuk konkret untuk memulai proyek FIREGUARD ID, menyiapkan data, dan memetakan titik-titik api di Provinsi Riau.
- **Tindakan & Solusi:**
  - Menelaah dokumen master proyek `FIREGUARD ID.txt`.
  - Mengaudit ketersediaan file dataset mentah NASA FIRMS VIIRS (2021–2025) dan file poligon batas wilayah GADM 4.1 (`gadm41_IDN_1.json`).
  - Mengekstrak batas resmi Provinsi Riau menjadi file `data/processed/riau_boundary.geojson`.
  - Melakukan *exact polygon clipping* menggunakan komputasi *point-in-polygon* (`matplotlib.path.Path`), menyaring 595.676 deteksi titik api seluruh Indonesia menjadi **15.510 rekaman titik api murni di daratan/pulau Provinsi Riau** (tersimpan di `data/processed/riau_viirs_hotspots_2021_2025.csv`).
  - Mengembangkan peta visualisasi titik api interaktif berbasis Leaflet di `outputs/maps/peta_titik_api_riau.html`.

---

### Sesi 2: Spatial Grid 0,1° & Baseline Clustering
- **Permintaan Pengguna:** Melanjutkan langkah berikutnya dan merapikan file kode ke dalam folder terpisah.
- **Tindakan & Solusi:**
  - Membuat folder terstruktur: `src/`, `outputs/maps/`, dan `notebooks/`.
  - Membentuk **Fishnet Grid Spasial beresolusi 0,1° × 0,1°** (~11 km × 11 km) yang menghasilkan **835 unit grid daratan Riau**.
  - Mengagregasi indikator kerawanan per grid:
    - `hotspot_count`: Akumulasi titik api (2021–2025).
    - `fire_days`: Jumlah hari aktif kebakaran.
    - `mean_frp` & `max_frp`: Rata-rata dan puncak *Fire Radiative Power* (MW).
    - `high_conf_count`: Frekuensi deteksi api berkategori *high confidence*.
  - Melakukan pemodelan *baseline* **K-Means Clustering** ($k=4$) untuk menetapkan 4 tingkatan kerawanan:
    - **Sangat Tinggi (Merah):** 3 grid
    - **Tinggi (Oranye):** 45 grid
    - **Sedang (Kuning):** 224 grid
    - **Rendah (Hijau):** 563 grid
  - Menyimpan profil dataset grid ke `data/processed/riau_vulnerability_grid_01deg.csv` dan membuat peta zonasi grid interaktif di `outputs/maps/peta_kerawanan_grid_riau.html`.

---

### Sesi 3: Penanganan Watermark Basemap
- **Permintaan Pengguna:** Melaporkan adanya watermark pada peta dan meminta menghilangkannya.
- **Identifikasi Masalah:** Watermark bertuliskan *"API KEY REQUIRED carto.com/basemaps/apikey"* muncul karena penyedia tile CartoDB memberlakukan API key.
- **Tindakan & Solusi:**
  - Mengganti *tile provider* ke **Esri World Dark Gray Canvas** (`server.arcgisonline.com`) yang 100% bebas watermark, tanpa API key, dan bertema gelap.
  - Memperbarui skrip generator dan meregenerasi kedua file peta HTML agar bersih dan rapi.

---

### Sesi 4: Verifikasi Checklist Tugas Minggu 2 (Data Preparation & Understanding)
- **Permintaan Pengguna:** Memeriksa apakah Poin 5 (Transformasi/Scaling) sudah memenuhi checklist tugas Minggu 2.
- **Evaluasi:** Poin 5 dinilai belum memenuhi kaidah bebas kebocoran (*leakage-free*) karena belum dilakukan *train/val/test split* sebelum fitting scaler.
- **Tindakan & Solusi Pengerjaan Checklist:**
  1. **Bab 4.1–4.3 (Data Understanding & Audit):** Memverifikasi 0 missing values, 0 duplikat, dan mengidentifikasi ketimpangan kelas (*imbalance*).
  2. **Bab 4.4 (EDA):** Menghasilkan 3 visualisasi di `reports/figures/`:
     - Tren tahunan titik api (`eda_1_tren_tahunan.png`): lonjakan titik api 2023–2025 dengan puncak di 2025 (4.004 deteksi).
     - Siklus musiman bulanan (`eda_2_siklus_musiman.png`): pola bimodal kemarau (Feb–Mar dan Jul–Sep).
     - Distribusi grid (`eda_3_distribusi_grid.png`): distribusi logaritmik kemunculan api per grid.
  3. **Bab 5.4 (Splitting):** Membagi 835 unit grid secara terisolasi dengan `random_state=42`:
     - **Training Set (70%):** 584 grid
     - **Validation Set (15%):** 125 grid
     - **Test Set (15%):** 126 grid (dikunci hingga Minggu 6)
  4. **Bab 5.3 (Feature Scaling Leakage-Free):** Memastikan `StandardScaler` di-**fit HANYA pada data training**, lalu men-transform validation dan test set secara terpisah. Model scaler disimpan ke `data/split/scaler.joblib`.
  5. **Bab 5.5 (Balancing):** Menetapkan bahwa metode oversampling sintetis (SMOTE) tidak diterapkan agar tidak menciptakan grid fiktif yang mendistorsi topologi spasial geografis Riau.
  6. **Bab 5.6 (Dokumentasi Apa–Mengapa–Bagaimana):** Menyusun dokumen `RINGKASAN_MINGGU_2.md` dan artifak laporan lengkap.
  7. **Notebook Modular:** Menyusun `notebooks/01_learning_problem_eda.ipynb` dan `notebooks/02_data_preparation.ipynb`.

---

### Sesi 5: Pembaruan README & Penataan Struktur Repositori
- **Permintaan Pengguna:** Memperbarui `README.md` dan menata struktur repositori sesuai standar hierarki proyek.
- **Tindakan & Solusi:**
  - Memperbarui isi `README.md` agar selaras dengan dokumen proyek master (beralih dari prediksi klasifikasi biner harian ke *unsupervised spatial clustering* K-Means, HDBSCAN, dan DEC).
  - Menyusun struktur repositori resmi:
    - `data/` (`raw/`, `processed/`, `metadata/`, `split/`)
    - `notebooks/` (`01_data_collection_and_audit.ipynb` s.d. `06_evaluation_and_comparison.ipynb`)
    - `src/` (`data_loader.py`, `preprocessing.py`, `modeling.py`, `evaluation.py`, `utils.py`, dll.)
    - `reports/` (`figures/`, `proposal/`)
    - `requirements.txt`, `.gitignore`, dan `LICENSE`.

---

## 2. Struktur Direktori Akhir Proyek

```text
FIREGUARD-ID/
├── README.md                                  # Dokumentasi utama proyek
├── RINGKASAN_MINGGU_2.md                      # Ringkasan Bab 4 & 5 (Apa-Mengapa-Bagaimana)
├── requirements.txt                           # Daftar pustaka dependensi Python
├── .gitignore                                 # Konfigurasi pengabaian file Git
├── LICENSE                                    # Lisensi open-source (MIT)
│
├── data/
│   ├── raw/                                   # Data mentah NASA FIRMS VIIRS & boundary GADM
│   ├── processed/                             # Data hasil olahan (GeoJSON batas Riau & CSV titik api)
│   ├── metadata/                              # Metadata dan register sumber data
│   └── split/                                 # Data partisi (train/val/test) & scaler.joblib
│
├── notebooks/
│   ├── 01_data_collection_and_audit.ipynb     # Notebook audit data dan pemotongan poligon
│   ├── 02_eda_and_feature_engineering.ipynb   # Notebook EDA dan rekayasa fitur
│   ├── 03_baseline_modeling.ipynb             # Notebook pemodelan K-Means
│   ├── 04_advanced_modeling.ipynb             # Notebook pemodelan HDBSCAN
│   ├── 05_deep_learning_lstm.ipynb            # Notebook pemodelan Deep Embedded Clustering
│   └── 06_evaluation_and_comparison.ipynb     # Notebook evaluasi metrik clustering
│
├── src/
│   ├── data_loader.py                         # Modul pembacaan dataset
│   ├── preprocessing.py                       # Modul clipping poligon & scaling
│   ├── modeling.py                            # Modul pelatihan model clustering
│   ├── evaluation.py                          # Modul metrik Silhouette & Davies-Bouldin
│   ├── utils.py                               # Fungsi utilitas pembantu
│   ├── generate_map.py                        # Pembuat peta sebaran titik api
│   └── spatial_grid_clustering.py             # Pembuat grid 0,1° & baseline K-Means
│
└── reports/
    ├── figures/                               # Gambar grafik hasil EDA (.png)
    └── proposal/                              # Dokumen proposal proyek
```

---

## 3. Status Milestone & Checklist

- [x] **Audit & Pengadaan Data:** Data NASA FIRMS VIIRS 2021–2025 dan batas resmi Riau siap pakai.
- [x] **Spatial Clipping:** $15.510$ titik api terisolasi di daratan Riau.
- [x] **Grid Discretization:** $835$ unit grid spasial 0,1° (~11 km).
- [x] **Interactive Mapping:** Peta titik api dan peta zonasi grid (bersih tanpa watermark).
- [x] **Data Understanding & EDA:** Analisis tren tahunan, siklus musiman, dan distribusi grid lengkap beserta insight.
- [x] **Leakage-Free Preparation:** Pemisahan dataset (Train 584, Val 125, Test 126) dengan `StandardScaler` di-fit hanya pada data training.
- [x] **Standard Repository Layout:** Seluruh folder dan modul kode tersusun rapi sesuai standar.
