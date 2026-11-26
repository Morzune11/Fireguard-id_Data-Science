# LAPORAN MINGGU 3 — MODEL DESIGN: BASELINE & ADVANCED
**Proyek:** FIREGUARD ID — Sistem Inteligensi & Zonasi Kerawanan Karhutla Provinsi Riau  
**Tujuan:** Melatih Model 1 (Baseline) dan Model 2 (Advanced) serta menjustifikasinya berdasarkan *hypothesis space* dan *inductive bias*, dengan evaluasi terbatas pada **Training & Validation Set** ($584$ dan $125$ grid).

---

## BAB 6. MODEL DESIGN

### 6.1 Model 1 (Baseline): K-Means Clustering

#### 6.1.1 Teori & Alasan Pemilihan Berdasarkan Karakteristik Data
**K-Means Clustering** adalah metode partisi *unsupervised* yang membagi $N$ data ke dalam $k$ klaster saling lepas dengan meminimalkan jumlah kuadrat jarak Euclidean ke masing-masing centroid (*Within-Cluster Sum of Squares / Inertia*):
$$\min_{C_1, \dots, C_k} \sum_{j=1}^{k} \sum_{x_i \in C_j} \|x_i - \mu_j\|_2^2$$

**Alasan pemilihan berdasarkan karakteristik data:**
1. **Fitur Numerik Kontinu Berdimensi Rendah:** Dataset grid $0{,}1^\circ$ Provinsi Riau memiliki 5 fitur numerik kontinu (`hotspot_count`, `fire_days`, `mean_frp`, `max_frp`, `high_conf_count`) yang sudah distandarisasi (`StandardScaler` di-*fit* hanya pada data training).
2. **Kebutuhan 4 Zona Prioritas:** Dengan $k=4$, centroid K-Means dapat langsung diurutkan berdasarkan rata-rata `hotspot_count` dan `mean_frp` menjadi 4 tingkat kerawanan (*Rendah, Sedang, Tinggi, Sangat Tinggi*).
3. **Baseline Komputasi Ringan:** Memberikan acuan performa awal yang deterministik (`random_state=42`, `n_init=10`).

#### 6.1.2 Hypothesis Space & Inductive Bias
- **Hypothesis Space:** Himpunan seluruh partisi ruang fitur $\mathbb{R}^5$ yang dibentuk oleh $k$ vektor centroid $\{\mu_1, \dots, \mu_k\}$ dengan batas keputusan *piecewise-linear* (diagram Voronoi).
- **Inductive Bias:**
  - Mengasumsikan klaster berbentuk **sferis/konveks** dengan varians relatif seragam di ruang Euclidean.
  - Memaksakan seluruh titik masuk ke salah satu klaster (*hard assignment*, tidak memodelkan *noise*). Karena data Karhutla Riau miring ke kanan (*right-skewed*), titik ekstrem dapat menggeser centroid.

#### 6.1.3–6.1.5 Hyperparameter & Hasil pada Validation Set
Hyperparameter baseline yang dicatat:
- `k=4` (kebutuhan 4 zona prioritas; juga dilakukan *sweep* $k=2 \dots 8$ sebagai pembanding), `init='k-means++'`, `n_init=10`, `random_state=42`.

Hasil evaluasi pada **Train & Validation Set** (Test Set tidak disentuh):
- **K-Means ($k=4$, Baseline):**
  - *Train:* `silhouette_score = 0.6721`, `davies_bouldin_score = 0.7784`, `calinski_harabasz_score = 311.4928`, `spatial_coherence = 0.8052`.
  - *Validation:* `silhouette_score = 0.6910`, `davies_bouldin_score = 0.5236`, `calinski_harabasz_score = 264.6037`, `spatial_coherence = 0.8840`.
- **K-Means ($k=2$, Optimal Silhouette Sweep):**
  - *Validation:* `silhouette_score = 0.8273`, `davies_bouldin_score = 1.0064`, `calinski_harabasz_score = 70.1279`, `spatial_coherence = 0.9080`.

---

### 6.2 Model 2 (Advanced): HDBSCAN

#### 6.2.1 Teori, Keunggulan, Kelemahan, & Karakteristik Data
**HDBSCAN** (*Hierarchical Density-Based Spatial Clustering of Applications with Noise*) mengelompokkan titik berdasarkan kepadatan lokal menggunakan *mutual reachability distance*, membentuk *Minimum Spanning Tree*, dan memilih klaster paling stabil dengan metode *Excess of Mass* (`eom`).

**Alasan pemilihan berdasarkan karakteristik data:**
Data grid Karhutla Riau mengalami **spatial imbalance** dan **distribusi right-skewed** yang tajam: mayoritas wilayah ($>500$ grid) berkumpul di kepadatan sangat rendah (mendekati $0$ hotspot), sedangkan wilayah episentrum tersebar dengan kepadatan bervariasi.

- **Keunggulan:** Tidak perlu menentukan jumlah klaster $k$ di awal; mampu menangkap bentuk klaster non-sferis dengan kepadatan bervariasi; otomatis mengisolasi *outlier* sebagai *noise* (`-1`).
- **Kelemahan:** Sensitif terhadap `min_cluster_size` dan `min_samples`; pada data berketimpangan ekstrem dapat menghasilkan rasio *noise* besar; implementasi `scikit-learn` tidak memiliki `predict()` bawaan sehingga titik validasi harus dipetakan ke centroid klaster terdekat.

#### 6.2.2 Hypothesis Space & Inductive Bias
- **Hypothesis Space:** Himpunan seluruh komponen terhubung berdensitas tinggi (*density-connected components*) pada graf *mutual reachability* dengan kestabilan hierarkis optimal.
- **Inductive Bias:** Mengasumsikan klaster adalah **area berdensitas tinggi yang dipisahkan area berdensitas rendah**, tidak mengasumsikan bentuk bulat/Euclidean, dan menganggap titik terisolasi sebagai *noise*.

#### 6.2.3 Pelatihan Parameter Awal & Perbandingan pada Validation Set
Parameter awal terpilih dari tuning pada Train Set: `min_cluster_size=15`, `min_samples=5`, `cluster_selection_method='eom'`.

**Perbandingan Awal Baseline vs Advanced pada Validation Set ($125$ Grid):**

| Model | Klaster (Val) | Silhouette ($\uparrow$) | Davies-Bouldin ($\downarrow$) | Calinski-Harabasz ($\uparrow$) | Spatial Coherence ($\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline: K-Means ($k=4$)** | $4$ | **$0.6910$** | **$0.5236$** | **$264.6037$** | **$0.8840$** |
| **Advanced: HDBSCAN $(15, 5)$** | $2$ | $0.1891$ | $1.0077$ | $9.9742$ | $0.6360$ |

**Analisis Singkat Validation Set:**  
Pada *Training Set*, HDBSCAN $(15, 5)$ menghasilkan klaster inti yang sangat kompak (`silhouette = 0.6255`, `dbi = 0.4263`), namun menandai **`38.53%`** sampel sebagai *noise* (`-1`). Ketika diterapkan pada *Validation Set* menggunakan *nearest-centroid assignment*, titik-titik yang seharusnya *noise* terpaksa masuk ke 2 klaster inti sehingga menurunkan skor Silhouette validasi menjadi `0.1891`. Sebaliknya, K-Means ($k=4$) lebih konsisten mempertahankan 4 zona pada Validation Set (`silhouette = 0.6910`, `spatial_coherence = 0.8840`).

---

## VERIFIKASI CHECKLIST MINGGU 3
- [x] **Alasan pemilihan model menyebut karakteristik data**: Terpenuhi (fitur kontinu terstandarisasi, *right-skewed*, *spatial imbalance*).
- [x] **Asumsi/inductive bias tiap model dijelaskan**: Terpenuhi (sferis/Euclidean pada K-Means vs *density-connected* + *noise* pada HDBSCAN).
- [x] **Hasil validasi tercatat (bukan test set)**: Terpenuhi (seluruh perbandingan di atas menggunakan Train & Validation Set).
- [x] **Notebook sesuai penamaan**: `notebooks/03_model_design_baseline_ml.ipynb` dan `notebooks/03_model_design_advanced_ml.ipynb` telah dibuat.
- [ ] **Commit ke GitHub & Asistensi Checkpoint 2**: Siap untuk Anda *commit/push* dan bawa ke sesi asistensi.
