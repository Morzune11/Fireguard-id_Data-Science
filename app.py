"""
FIREGUARD ID — Dashboard Progress Minggu 3
Model Design: Baseline (K-Means) & Advanced (HDBSCAN / DEC)
"""
import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="FIREGUARD ID — Minggu 3 Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom subtle styling
st.markdown(
    """
    <style>
        .metric-card {
            background-color: #1e2129;
            padding: 16px;
            border-radius: 10px;
            border-left: 4px solid #ff5722;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


# Sidebar
st.sidebar.title("🔥 FIREGUARD ID")
st.sidebar.caption("Forest & Land Fire Intelligence, Response, and Resource Optimization System")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigasi Dashboard",
    [
        "1. Ringkasan & Checklist Minggu 3",
        "2. Baseline: K-Means (Bab 6.1)",
        "3. Advanced: HDBSCAN (Bab 6.2)",
        "4. Deep Learning: DEC (Autoencoder)",
        "5. Peta Zonasi & Profil Kerawanan",
        "6. Fase 3: Spatial Prioritization",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Wilayah Pilot:** Provinsi Riau\n\n"
    "**Unit Analisis:** Grid Spasial 0.1° (~11 km)\n\n"
    "**Split Data:** 584 Train | 125 Val | 126 Test (Locked)"
)

# ------------------------------------------------------------------
# PAGE 1: RINGKASAN & CHECKLIST MINGGU 3
# ------------------------------------------------------------------
if page == "1. Ringkasan & Checklist Minggu 3":
    st.title("🔥 FIREGUARD ID — Progress Minggu 3: Model Design")
    st.markdown(
        "Dashboard ini menyajikan hasil pemodelan **Baseline (K-Means)** dan **Advanced (HDBSCAN & DEC)** "
        "pada **Training & Validation Set** secara *leakage-free*."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Titik Api Riau", "15.510", "VIIRS 2021–2025")
    c2.metric("Unit Grid Spasial (0.1°)", "835", "584 Train | 125 Val")
    c3.metric("Val Silhouette (K-Means k=4)", "0.6910", "DBI: 0.5236")
    c4.metric("Val Silhouette (HDBSCAN)", "0.1891", "Train Noise: 38.53%")

    st.markdown("---")
    st.subheader("Perbandingan Teori, Hypothesis Space, & Inductive Bias")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Model 1 (Baseline): K-Means")
        st.write(
            "- **Teori:** Partisi Voronoi berbasis centroid dengan meminimalkan *Within-Cluster Sum of Squares (WCSS)*.\n"
            "- **Alasan Data:** Cocok untuk 5 fitur numerik kontinu terstandarisasi berdimensi rendah ($d=5$) dan mudah dipetakan ke 4 zona prioritas.\n"
            "- **Hypothesis Space:** Partisi ruang fitur $\\mathbb{R}^5$ oleh $k$ vektor centroid.\n"
            "- **Inductive Bias:** Mengasumsikan klaster berbentuk **sferis/konveks** dengan varians seragam; seluruh titik wajib masuk klaster (tanpa *noise*)."
        )
    with col_b:
        st.markdown("### Model 2 (Advanced): HDBSCAN")
        st.write(
            "- **Teori:** *Hierarchical Density-Based Spatial Clustering* menggunakan *mutual reachability distance* dan *Excess of Mass* (`eom`).\n"
            "- **Alasan Data:** Data Karhutla Riau bersifat *right-skewed* dan mengalami *spatial imbalance* ekstrem.\n"
            "- **Hypothesis Space:** Komponen terhubung berdensitas tinggi pada graf *mutual reachability*.\n"
            "- **Inductive Bias:** Mengasumsikan klaster adalah **wilayah berdensitas tinggi** yang dipisahkan area berdensitas rendah; titik terisolasi dianggap *noise* (`-1`)."
        )

    st.markdown("---")
    st.subheader("Status Checklist Minggu 3")
    st.checkbox("Alasan pemilihan model menyebut karakteristik data", value=True, disabled=True)
    st.checkbox("Asumsi / inductive bias tiap model dijelaskan", value=True, disabled=True)
    st.checkbox("Hasil validasi tercatat (bukan test set)", value=True, disabled=True)
    st.checkbox("Notebook 03_model_design_baseline_ml.ipynb & 03_model_design_advanced_ml.ipynb dibuat", value=True, disabled=True)
    st.checkbox("Commit ke GitHub & Asistensi Checkpoint 2", value=False, disabled=True)

# ------------------------------------------------------------------
# PAGE 2: BASELINE K-MEANS
# ------------------------------------------------------------------
elif page == "2. Baseline: K-Means (Bab 6.1)":
    st.title("📊 Model 1 (Baseline): K-Means Clustering")
    sweep_df = load_csv("reports/kmeans_sweep_metrics.csv")

    st.subheader("Sweep Jumlah Klaster (k = 2 s.d. 8) — Train vs Validation")
    st.dataframe(sweep_df.style.format("{:.4f}", subset=["inertia", "train_silhouette", "train_dbi", "train_chi", "val_silhouette", "val_dbi"]), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Elbow Method (Inertia vs k)**")
        st.line_chart(sweep_df.set_index("k")["inertia"])
    with col2:
        st.markdown("**Silhouette Score (Train vs Validation)**")
        st.line_chart(sweep_df.set_index("k")[["train_silhouette", "val_silhouette"]])

    st.info(
        "**Catatan Analisis:** Secara matematis, $k=2$ meraih Silhouette Validation tertinggi (`0.8273`), "
        "sedangkan $k=4$ (`val_silhouette = 0.6910`, `val_dbi = 0.5236`) dipilih sebagai baseline operasional "
        "karena selaras dengan 4 tingkat zonasi prioritas (*Rendah, Sedang, Tinggi, Sangat Tinggi*)."
    )

# ------------------------------------------------------------------
# PAGE 3: ADVANCED HDBSCAN
# ------------------------------------------------------------------
elif page == "3. Advanced: HDBSCAN (Bab 6.2)":
    st.title("🔬 Model 2 (Advanced): HDBSCAN & Komparasi Model")
    hdb_df = load_csv("reports/hdbscan_tuning_metrics.csv")
    comp_df = load_csv("reports/clustering_comparison.csv")

    st.subheader("1. Eksperimen Tuning HDBSCAN (Training Set)")
    st.dataframe(hdb_df, use_container_width=True)

    st.subheader("2. Perbandingan Model pada Validation Set")
    val_comp = comp_df[comp_df["split"] == "Validation"].copy()
    st.dataframe(val_comp, use_container_width=True)

    if os.path.exists("reports/figures/clustering_model_comparison.png"):
        st.subheader("3. Visualisasi Perbandingan Metrik Evaluasi")
        st.image("reports/figures/clustering_model_comparison.png", use_container_width=True)

# ------------------------------------------------------------------
# PAGE 4: DEEP LEARNING DEC
# ------------------------------------------------------------------
elif page == "4. Deep Learning: DEC (Autoencoder)":
    st.title("🧠 Model 3 (Deep Learning): Deep Embedded Clustering (DEC)")
    st.markdown(
        "**Arsitektur:** PyTorch Autoencoder (`5 → 32 → 16 → 4` *Latent Space* `→ 16 → 32 → 5`) "
        "yang dikombinasikan dengan *Student's t-distribution soft assignment* dan *KL-Divergence refinement* (Xie et al., 2016)."
    )

    comp_df = load_csv("reports/clustering_comparison.csv")
    dec_comp = comp_df[comp_df["model"].str.contains("DEC")].copy()

    c1, c2, c3 = st.columns(3)
    dec_val = dec_comp[(dec_comp["model"] == "DEC (Latent Space, k=4)") & (dec_comp["split"] == "Validation")].iloc[0]
    c1.metric("Val Silhouette (Latent)", f"{dec_val['silhouette_score']:.4f}", "4 Klaster Konsisten")
    c2.metric("Val Davies-Bouldin", f"{dec_val['davies_bouldin_score']:.4f}", "Lebih rendah = lebih baik")
    c3.metric("Val Calinski-Harabasz", f"{dec_val['calinski_harabasz_score']:.2f}", "Tertinggi di Val & Test")

    st.subheader("1. Metrik Evaluasi DEC (Train, Validation, & Test)")
    st.dataframe(dec_comp, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. Representasi Ruang Laten (z1 vs z2)")
        if os.path.exists("reports/dec_latent_coords.csv"):
            latent_df = load_csv("reports/dec_latent_coords.csv")
            latent_df["dec_cluster"] = latent_df["dec_cluster"].astype(str)
            st.scatter_chart(latent_df, x="z1", y="z2", color="dec_cluster")
        elif os.path.exists("reports/figures/dec_latent_space.png"):
            st.image("reports/figures/dec_latent_space.png", use_container_width=True)

    with col2:
        st.subheader("3. Profil Kerawanan Klaster DEC (835 Grid)")
        if os.path.exists("reports/cluster_profile_dec.csv"):
            prof_dec = load_csv("reports/cluster_profile_dec.csv")
            st.dataframe(prof_dec, use_container_width=True)

# ------------------------------------------------------------------
# PAGE 5: PETA ZONASI & PROFIL KERAWANAN
# ------------------------------------------------------------------
elif page == "5. Peta Zonasi & Profil Kerawanan":
    st.title("🗺️ Peta Zonasi & Profil Kerawanan Riau (835 Grid)")
    grid_df = load_csv("data/processed/riau_vulnerability_clusters_evaluated.csv")

    zones = ["Semua"] + sorted(grid_df["priority_zone_class"].unique().tolist())
    selected_zone = st.selectbox("Filter Zona Prioritas:", zones)

    filtered_df = grid_df if selected_zone == "Semua" else grid_df[grid_df["priority_zone_class"] == selected_zone]
    st.write(f"Menampilkan **{len(filtered_df)}** dari **{len(grid_df)}** grid.")

    map_path = "outputs/maps/peta_evaluasi_cluster_riau.html"
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            map_html = f.read()
        components.html(map_html, height=560, scrolling=False)

    st.subheader("Tabel Profil Grid Kerawanan")
    cols_show = ["grid_id", "priority_zone_class", "hotspot_count", "fire_days", "mean_frp", "max_frp", "high_conf_count", "cluster_dec"]
    st.dataframe(filtered_df[cols_show].sort_values("hotspot_count", ascending=False), use_container_width=True)

# ------------------------------------------------------------------
# PAGE 6: FASE 3 - SPATIAL PRIORITIZATION
# ------------------------------------------------------------------
elif page == "6. Fase 3: Spatial Prioritization":
    st.title("🎯 Fase 3: Spatial Prioritization (MCDA)")
    st.markdown(
        "Penentuan **Priority Score (0–100)** dan **Priority Rank (1–835)** secara transparan menggunakan bobot: "
        "**40%** Riwayat & Rekurensi + **25%** Intensitas FRP & Confidence + **20%** Konsensus Klaster (K-Means & DEC) + **15%** Spillover Tetangga."
    )

    prio_df = load_csv("data/processed/riau_spatial_prioritization.csv")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prioritas 1 - Kritis", int((prio_df["priority_zone_class"] == "Prioritas 1 - Kritis").sum()), "Skor >= 65")
    c2.metric("Prioritas 2 - Tinggi", int((prio_df["priority_zone_class"] == "Prioritas 2 - Tinggi").sum()), "Skor 45–64.9")
    c3.metric("Prioritas 3 - Sedang", int((prio_df["priority_zone_class"] == "Prioritas 3 - Sedang").sum()), "Skor 25–44.9")
    c4.metric("Prioritas 4 - Rendah", int((prio_df["priority_zone_class"] == "Prioritas 4 - Rendah").sum()), "Skor < 25")

    st.subheader("1. Peta Interaktif Zona Prioritas Mitigasi Riau")
    prio_map_path = "outputs/maps/peta_zona_prioritas_riau.html"
    if os.path.exists(prio_map_path):
        with open(prio_map_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=540, scrolling=False)

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.subheader("2. Top 20 Grid Prioritas Paling Mendesak")
        top20_df = load_csv("reports/top_20_priority_grids.csv")
        st.dataframe(top20_df, use_container_width=True)
    with col_b:
        st.subheader("3. Distribusi & Breakdown Skor")
        if os.path.exists("reports/figures/spatial_prioritization_breakdown.png"):
            st.image("reports/figures/spatial_prioritization_breakdown.png", use_container_width=True)
