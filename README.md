# FIREGUARD ID

Forest & Land Fire Intelligence, Response, and Resource Optimization System  
_A Geospatial Analytics, Unsupervised Learning, and Resource Optimization Decision Support Framework for Riau Province, Indonesia_

---

## 1. FORMULIR PENGAJUAN IDE

| Komponen                | Isian                                                                                                                                                                                                                                                           |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Nama / NIM**          | Ghofur Akbar M. / 244311042, Moch. Arrizal Syach Putra / 244311047                                                                                                                                                                                              |
| **Judul ide**           | **FIREGUARD ID: Forest & Land Fire Intelligence, Response, and Resource Optimization System**                                                                                                                                                                   |
| **SDG & target**        | **SDG 13 – Climate Action, Target 13.1** (Memperkuat ketahanan dan kapasitas adaptasi terhadap bencana iklim)                                                                                                                                                   |
| **Masalah nyata**       | Kerawanan Kebakaran Hutan dan Lahan (Karhutla) di Riau yang berulang, penyebaran risiko yang tidak merata, serta perlunya alokasi penempatan infrastruktur dan sumber daya tanggap darurat yang optimal dan berbasis data.                                      |
| **Tugas ML**            | **Unsupervised Learning & Spatial Clustering**: Pengelompokan wilayah berdasarkan profil kerawanan historis-lingkungan jangka panjang dan optimasi penempatan markas tanggap darurat (MCLP/MILP).                                                               |
| **Objective**           | Mengelompokkan grid wilayah menurut kemiripan profil kerawanan karhutla, menentukan zona prioritas mitigasi, serta mengoptimalkan penempatan fasilitas respons darurat.                                                                                         |
| **Metrik utama**        | **Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index**                                                                                                                                                                                             |
| **Metrik tambahan**     | Spatial Coherence, Cluster Stability, Coverage Ratio, Response Radius Efficiency                                                                                                                                                                                |
| **Dataset**             | NASA FIRMS VIIRS (S-NPP/JPSS-1), GADM 4.1 / geoBoundaries Batas Administratif Riau, Copernicus ERA5-Land, CHIRPS, MODIS Land Cover/NDVI, DEMNAS/BIG.                                                                                                            |
| **Baseline Model**      | **K-Means Clustering**                                                                                                                                                                                                                                          |
| **Advanced ML Model**   | **HDBSCAN (Density-Based Spatial Clustering)**                                                                                                                                                                                                                  |
| **Deep Learning Model** | **Deep Embedded Clustering (DEC)**                                                                                                                                                                                                                              |
| **Risiko dan etika**    | Keterbatasan resolusi satelit, ketidakseimbangan spasial (_spatial imbalance_), risiko _false sense of security_, bias observasi, serta hasil clustering yang tidak boleh dianggap langsung sebagai probabilitas prediksi tanpa interpretasi domain lingkungan. |
| **Referensi awal**      | Eaturu & Vadrevu (2025); Simulating hydrology for tropical peat fire forecasts (2023); Xie et al. (DEC - Unsupervised Deep Embedding for Clustering Analysis, ICML 2016).                                                                                       |

**Repository GitHub:** https://github.com/Ghof-AM/FIREGUARD-ID.git  
**Status Dataset:** Tahap _Data Understanding & Data Preparation (Minggu 2)_ telah selesai. Data titik api VIIRS (2021–2025) berhasil dipotong (_exact polygon clipping_) ke daratan Riau, menghasilkan $15.510$ titik api dan $835$ unit grid spasial 0,1° yang telah dipartisi (_train/val/test_) secara _leakage-free_.

---

## 2. Ringkasan Proyek

**FIREGUARD ID** merupakan sistem pendukung keputusan (_decision support system_) berbasis kecerdasan buatan, analisis geospasial, dan riset operasi yang dirancang untuk mendukung mitigasi serta penanggulangan kebakaran hutan dan lahan (Karhutla) di Provinsi Riau sebagai wilayah pilot.

Berbeda dari pendekatan klasifikasi probabilitas jangka pendek harian yang rentan noise musiman, FIREGUARD ID mengadopsi pendekatan **analisis kerawanan jangka panjang melalui unsupervised learning dan spatial clustering**. Sistem membagi daratan Riau menjadi unit analisis **grid spasial 0,1° (~11 km × 11 km)** dan mengelompokkannya berdasarkan indikator akumulasi frekuensi api, hari kebakaran (_fire days_), dan intensitas radiasi energi (_Fire Radiative Power / FRP_).

Tiga model clustering dikembangkan dan dikomparasikan:

1. **K-Means Clustering** sebagai _baseline model_.
2. **HDBSCAN** sebagai _advanced machine learning_ yang mampu menangani densitas tidak seragam dan mengisolasi noise.
3. **Deep Embedded Clustering (DEC)** sebagai pendekatan _deep learning_ berbasis autoencoder laten untuk representasi fitur kompleks.

Hasil profil zonasi kerawanan selanjutnya menjadi masukan bagi modul optimasi operasional berbasis **Maximal Covering Location Problem (MCLP) / Mixed Integer Linear Programming (MILP)** guna merekomendasikan lokasi posko/markas, kapasitas armada, alokasi personel pemadam, dan efisiensi anggaran tanggap darurat secara transparan dan terukur.

---

### BAB 1. PENDAHULUAN

#### 1.1 Latar Belakang

Provinsi Riau merupakan salah satu wilayah di Indonesia dengan ekosistem lahan gambut terluas yang sangat rentan mengalami Karhutla berulang, terutama pada periode kemarau bimodal (Februari–Maret dan Juli–September). Kebakaran ini melepaskan emisi karbon masif, merusak biodiversitas, menurunkan kualitas udara lintas batas (_transboundary haze_), dan mengganggu perekonomian.

Penanganan Karhutla selama ini kerap bersifat reaktif di lapangan. Diperlukan transformasi menuju mitigasi preventif berbasis data geospasial objektif. Pemanfaatan data penginderaan jauh seperti **NASA FIRMS VIIRS** beresolusi 375m memungkinkan pemetaan riwayat kejadian secara mikro. Dengan memadukan teknik _unsupervised spatial clustering_ dan pemodelan optimasi matematis, FIREGUARD ID hadir untuk memetakan zonasi kerawanan sekaligus memberikan rekomendasi presisi terkait penempatan infrastruktur penanggulangan bencana.

#### 1.2 Rumusan Masalah

1. Bagaimana mengidentifikasi karakteristik dan mengelompokkan wilayah kerawanan Karhutla di Provinsi Riau menggunakan _unsupervised spatial clustering_?
2. Bagaimana perbandingan performa model _baseline_ (K-Means), _advanced ML_ (HDBSCAN), dan _deep learning_ (DEC) dalam menghasilkan klaster kerawanan yang stabil dan koheren secara spasial?
3. Bagaimana merumuskan model optimasi (_facility location_) untuk menentukan lokasi markas, kapasitas armada, alokasi personel, dan anggaran operasional tanggap darurat berbasis zonasi kerawanan tersebut?

#### 1.3 Tujuan Penelitian

1. Membangun pipeline data geospasial untuk ekstraksi dan agregasi fitur Karhutla ke unit grid spasial 0,1°.
2. Mengembangkan dan mengevaluasi tiga model _unsupervised clustering_ (K-Means, HDBSCAN, dan DEC).
3. Menghasilkan peta interaktif zonasi kerawanan Karhutla Provinsi Riau.
4. Mengembangkan formulasi optimasi penempatan fasilitas tanggap darurat (MCLP/MILP) berbasis zona prioritas.
5. Menyediakan prototipe sistem pendukung keputusan yang teruji, transparan, dan dapat direproduksi (_reproducible_).

---

### BAB 2. LANDASAN DAN KETERKAITAN SDGs

#### 2.1 Landasan Penelitian

1. **Geospatial Grid Analysis:** Pemanfaatan kisi berukuran 0,1° (~11 km) menormalisasi bias observasi satelit dan mengubah deteksi titik acak menjadi unit estimasi kerawanan wilayah yang stabil.
2. **Unsupervised Learning & Spatial Clustering:** Mengakomodasi kompleksitas data lingkungan tanpa ketergantungan pada label manual subjektif. K-Means mengelompokkan partisi berbasis jarak euklidean, HDBSCAN mengidentifikasi klaster berdensitas tinggi serta noise spasial, dan DEC memproyeksikan fitur non-linear ke ruang laten untuk pemisahan klaster yang lebih optimal.
3. **Resource & Facility Location Optimization:** Mengadaptasi formulasi _Maximal Covering Location Problem (MCLP)_ untuk memaksimalkan proteksi wilayah rawan dengan batasan radius jangkauan respons armada pemadam kebakaran.

#### 2.2 Keterkaitan dengan SDGs

Mendukung **SDG 13 — Climate Action**, secara spesifik **Target 13.1**: _Memperkuat ketahanan dan kapasitas adaptasi terhadap bahaya terkait iklim dan bencana alam_. FIREGUARD ID berkontribusi langsung pada efisiensi perencanaan mitigasi dan alokasi sumber daya pemadaman berbasis bukti ilmiah (_evidence-based policy_).

---

### BAB 3. METODOLOGI PENELITIAN

#### 3.1 Wilayah Penelitian dan Sumber Data

- **Wilayah Pilot:** Provinsi Riau, Indonesia (Batas resmi GADM 4.1 Level 1).
- **NASA FIRMS VIIRS (S-NPP & NOAA-20/JPSS-1):** Deteksi hotspot historis 375m (2021–2025).
- **Copernicus ERA5-Land & CHIRPS:** Temperatur, curah hujan, komponen angin, dan kelembapan tanah (_soil moisture_).
- **MODIS (MCD12Q1 & MOD13Q1):** Tutupan lahan (_land cover_) dan indeks vegetasi (NDVI).
- **OpenStreetMap & DEMNAS:** Topografi/ketinggian dan jaringan aksesibilitas jalan raya.

#### 3.2 Learning Problem & Evaluasi

| Komponen                | Rancangan                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------- |
| **Pendekatan**          | Unsupervised Learning & Spatial Clustering                                                            |
| **Unit Data**           | Grid geografis berukuran 0,1° × 0,1° (~11 km × 11 km)                                                 |
| **Fitur Analisis**      | `hotspot_count`, `fire_days`, `mean_frp`, `max_frp`, `high_conf_count`                                |
| **Baseline Model**      | K-Means Clustering                                                                                    |
| **Advanced ML Model**   | HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)                    |
| **Deep Learning Model** | Deep Embedded Clustering (DEC) dengan Autoencoder                                                     |
| **Metrik Evaluasi**     | Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Cluster Stability, Spatial Coherence |

#### 3.3 Alur Pipeline & Sistem Pendukung Keputusan

```text
DATA SATELLITE (VIIRS) & BOUNDARY (GADM)
                   │
                   ▼
       EXACT POLYGON CLIPPING (RIAU)
                   │
                   ▼
     SPATIAL GRID ENGINEERING (0.1°)
                   │
                   ▼
 LEAKAGE-FREE SCALING & TRAIN/VAL/TEST SPLIT
                   │
                   ▼
      UNSUPERVISED CLUSTERING
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
 K-MEANS        HDBSCAN          DEC
    └──────────────┬──────────────┘
                   ▼
         CLUSTER EVALUATION
                   │
                   ▼
       SPATIAL PRIORITIZATION
                   │
                   ▼
   INFRASTRUCTURE & RESOURCE OPTIMIZATION
        (MCLP / MILP - Posko & Armada)
                   │
                   ▼
    GEOSPATIAL DECISION SUPPORT SYSTEM
```

---

## 4. Referensi Utama

1. Eaturu, A., & Vadrevu, K. P. (2025). Evaluation of machine learning and deep learning algorithms for fire prediction in Southeast Asia. _Scientific Reports_, 15, 18807.
2. Accuracy of tropical peat and non-peat fire forecasts enhanced by simulating hydrology. (2023). _Scientific Reports_. https://doi.org/10.1038/s41598-022-27075-0
3. Xie, J., Girshick, R., & Farhadi, A. (2016). Unsupervised Deep Embedding for Clustering Analysis. _Proceedings of the 33rd International Conference on Machine Learning (ICML)_, PMLR 48:478-487.
4. Church, R., & ReVelle, C. (1974). The Maximal Covering Location Problem. _Papers of the Regional Science Association_, 32, 101-118.

---

## 5. Perkembangan Proyek Saat Ini (Progress Minggu 12)

- [x] **Data Sourcing & Cleaning:** Berhasil melakukan audit dan _exact polygon clipping_ 595.676 hotspot Indonesia menjadi **15.510 rekaman titik api murni di daratan/pulau Riau** (2021–2025).
- [x] **Spatial Grid Formulation:** Membentuk **835 unit grid sel 0,1°** yang mencakup seluruh Provinsi Riau beserta rekayasa fitur akumulasi apinya.
- [x] **Visualisasi Interaktif:** Menghasilkan peta sebaran titik api (`peta_titik_api_riau.html`) dan peta choropleth kerawanan grid (`peta_kerawanan_grid_riau.html`) yang bebas watermark.
- [x] **Data Understanding & EDA:** Analisis tren temporal tahunan, siklus musiman, dan distribusi ketimpangan kelas spasial (`outputs/figures/`).
- [x] **Data Preparation (Leakage-Free):** Pemisahan dataset grid menjadi **Train (584)**, **Validation (125)**, dan **Test (126)** dengan `StandardScaler` yang di-fit murni pada set training.
- [ ] **Next Steps :** Eksperimen komparasi K-Means vs HDBSCAN vs DEC, evaluasi Silhouette/Davies-Bouldin, penentuan zona prioritas, dan pemodelan optimasi MCLP.

---

## 6. Struktur Direktori Proyek

```text
FIREGUARD-ID/
├── README.md                                  # Dokumentasi utama proyek
├── RINGKASAN_MINGGU_12.md                      # Ringkasan Bab 4 & 5 (Apa-Mengapa-Bagaimana)
├── riau_boundary.geojson                      # Batas resmi administratif Provinsi Riau
├── riau_viirs_hotspots_2021_2025.csv          # Dataset titik api Riau hasil clipping (15.510 baris)
├── riau_vulnerability_grid_01deg.csv          # Dataset profil kerawanan 835 grid 0.1°
│
├── data/
│   └── split/                                 # Data partisi siap latih (Leakage-Free)
│       ├── train_raw.csv / train_scaled.csv   # Training set (584 sampel)
│       ├── val_raw.csv / val_scaled.csv       # Validation set (125 sampel)
│       ├── test_raw.csv / test_scaled.csv     # Test set terisolasi (126 sampel)
│       └── scaler.joblib                      # StandardScaler hasil fit pada training
│
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb          # Notebook EDA & analisis kualitas data
│   └── 02_data_preparation.ipynb              # Notebook data splitting & feature scaling
│
├── outputs/
│   ├── figures/                               # Visualisasi hasil EDA (Tren tahunan, musiman, grid)
│   └── maps/                                  # Visualisasi peta interaktif (Leaflet)
│       ├── peta_titik_api_riau.html           # Peta sebaran titik api historis
│       └── peta_kerawanan_grid_riau.html      # Peta zonasi kerawanan grid 0.1°
│
└── src/
    ├── process_riau_hotspots.py               # Skrip spatial clipping data satelit
    ├── generate_map.py                        # Skrip pembuat peta titik api
    ├── spatial_grid_clustering.py             # Skrip fishnet grid 0.1° & baseline K-Means
    ├── eda_pipeline.py                        # Pipeline generate grafik EDA
    └── data_preparation_pipeline.py           # Pipeline splitting dan scaling data
```

---

## 7. Anggota Tim

- **Ghofur Akbar M.** — NIM: 244311042
- **Moch. Arrizal Syach Putra** — NIM: 244311047
- **Mata Kuliah:** Data Science / Machine Learning
- **Fokus Riset:** _Geospatial Data Science, Climate Action (SDG 13), & Disaster Emergency Optimization_
