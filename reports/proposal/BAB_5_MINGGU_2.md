# RINGKASAN MINGGU 2: DATA UNDERSTANDING & DATA PREPARATION
**Proyek:** FIREGUARD ID — Sistem Inteligensi & Zonasi Kerawanan Karhutla Provinsi Riau  
**Domain:** Geospatial Data Science & Unsupervised Spatial Clustering  

---

## 1. APA (What Was Done)
1. **Data Sourcing & Quality Audit (Bab 4.1 - 4.3):**
   - Mengambil 595.676 data titik api VIIRS JPSS-1 Indonesia (2021–2025).
   - Memotong (*exact polygon clipping*) ke batas administratif Provinsi Riau (GADM Level 1), menghasilkan 15.510 rekaman titik api bersih tanpa duplikat dan tanpa missing values.
2. **Spatial Grid Engineering (Bab 5.1 - 5.2):**
   - Membagi daratan Riau menjadi 835 unit analisis grid spasial berukuran 0,1° × 0,1° (~11 km × 11 km).
   - Merekayasa fitur spasial-temporal per grid: `hotspot_count`, `fire_days`, `mean_frp`, `max_frp`, dan `high_conf_count`.
3. **Exploratory Data Analysis (Bab 4.4):**
   - Menghasilkan 3 visualisasi utama: tren lonjakan tahunan (puncak di 2025 dengan 4.004 deteksi), pola musiman kemarau (Feb–Mar dan Jul–Sep), serta distribusi logaritmik kemunculan api per grid.
4. **Data Splitting & Leakage-Free Scaling (Bab 5.3 - 5.4):**
   - Membagi 835 grid secara terisolasi: 584 Train (70%), 125 Val (15%), dan 126 Test (15%) dengan `random_state=42`.
   - Melakukan standarisasi data (`StandardScaler`) yang di-fit **HANYA** pada data training.
   - Mengunci dan mengisolasi Test Set agar tidak disentuh sampai tahap evaluasi akhir (Minggu 6).

---

## 2. MENGAPA (Why It Was Done)
1. **Mengapa Perlu Exact Polygon Clipping?**
   - Data satelit mencakup seluruh Indonesia dan laut terbuka. Bounding box persegi sederhana akan memasukkan perairan Selat Malaka dan provinsi tetangga (Sumbar/Jambi/Sumut), sehingga pemotongan berbasis poligon batas resmi mutlak diperlukan agar data representatif.
2. **Mengapa Menggunakan Grid 0,1° (~11 km)?**
   - Satu titik hotspot satelit tidak mewakili satu kejadian kebakaran unik dan memiliki bias observasi. Agregasi ke grid spasial mengubah titik kejadian menjadi unit profil kerawanan wilayah jangka panjang.
3. **Mengapa Scaler Hanya Di-fit pada Data Training?**
   - Mencegah *Data Leakage* (kebocoran informasi). Jika mean dan standar deviasi dihitung dari seluruh dataset (termasuk validasi dan tes), model akan mendapat "bocoran" sebaran statistik data uji yang semestinya belum pernah dilihat di dunia nyata.
4. **Mengapa Tidak Dilakukan Oversampling Sintetis (SMOTE)?**
   - Karena proyek ini berbasis geospasial dan *unsupervised clustering*. Menambahkan titik fiktif sintetis berisiko menciptakan koordinat dan kondisi kebakaran palsu di lokasi geografis Riau yang aslinya aman.

---

## 3. BAGAIMANA (How It Was Done)
1. **Pemrosesan Spasial:**
   - Dijalankan via script `src/process_riau_hotspots.py` menggunakan `matplotlib.path.Path` untuk komputasi *point-in-polygon* yang presisi dan cepat.
2. **Splitting dan Transformasi Data:**
   - Menggunakan `sklearn.model_selection.train_test_split` untuk partisi bertingkat (70:15:15).
   - Menggunakan `sklearn.preprocessing.StandardScaler`:
     ```python
     scaler = StandardScaler()
     scaler.fit(train_df[feature_cols])  # HANYA FIT PADA TRAIN
     X_train_scaled = scaler.transform(train_df[feature_cols])
     X_val_scaled   = scaler.transform(val_df[feature_cols])
     X_test_scaled  = scaler.transform(test_df[feature_cols])
     ```
3. **Penyimpanan Output Modular:**
   - Seluruh data partisi disimpan ke folder `data/split/` (`train_raw.csv`, `val_raw.csv`, `test_raw.csv`, `train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv`, dan `scaler.joblib`).
   - Laporan visual disimpan di `outputs/figures/` dan peta interaktif bebas watermark di `outputs/maps/`.
   - Alur kerja lengkap diabadikan dalam bentuk notebook yang bisa dijalankan ulang: `notebooks/01_learning_problem_eda.ipynb` dan `notebooks/02_data_preparation.ipynb`.

---

## 4. STATUS CHECKLIST MINGGU 2
- [x] **Scaler/encoder di-fit hanya pada data training**: Terpenuhi.
- [x] **Test set disimpan dan tidak disentuh sampai Minggu 6**: Terpenuhi (`data/split/test_*.csv`).
- [x] **Jumlah sampel tiap split tercatat**: Terpenuhi (584 Train, 125 Val, 126 Test).
- [x] **Setiap langkah preprocessing punya alasan tertulis**: Terpenuhi (Bab 5.6 Apa–Mengapa–Bagaimana).
