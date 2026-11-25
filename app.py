import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Mall Customer Segmentation App",
    page_icon="🛍️",
    layout="wide"
)

# --- 2. SIDEBAR NAVIGASI (Dengan Menu Perbandingan Ditambahkan) ---
st.sidebar.title("📌 Navigasi Menu")
menu = st.sidebar.selectbox(
    "Pilih Halaman",
    [
        "Halaman Utama",
        "Eksplorasi Data (EDA)",
        "Analisis Gender & K-Means",
        "Perbandingan Algoritma Klustering",
        "Form Simulasi Segmen",
        "Tentang Kami"
    ]
)

# --- 3. FUNGSI MEMUAT DATASET ---
@st.cache_data
def load_data():
    # Pastikan file Mall_Customers.csv berada di direktori yang sesuai (misal: dalam folder venv/)
    url = "https://raw.githubusercontent.com/AliFaridzR/Mall_Customers.csv/refs/heads/main/Mall_Customers.csv"
    df = pd.read_csv(url)

print(df.head())


# Memanggil fungsi load data
try:
    df_mall = load_data()
except Exception as e:
    df_mall = None

# ==========================================
# HALAMAN 1: HALAMAN UTAMA
# ==========================================
if menu == 'Halaman Utama':
    st.title('🛍️ Aplikasi Web Segmentasi Pelanggan Mal')
    st.markdown("Selamat datang di aplikasi web berbasis Streamlit untuk analisis dan prediksi Customer Segmentation menggunakan algoritma Unsupervised Learning (K-Means Clustering).")
    st.divider()

    st.subheader("📊 Ringkasan Dataset Mall Customers")
    if df_mall is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pelanggan", f"{df_mall.shape[0]} Orang", "Aktif")
        col2.metric("Rata-rata Usia", f"{df_mall['Age'].mean():.1f} Tahun", "± 1.2")
        col3.metric("Rata-rata Spending Score", f"{df_mall['Spending Score (1-100)'].mean():.1f}", "Skor 1-100")

        st.markdown("### 🔍 Pratinjau Data Mentah")
        st.dataframe(df_mall.head(10), use_container_width=True)
    else:
        st.warning("⚠️ File `Mall_Customers.csv` belum ditemukan di direktori. Harap letakkan file dataset pada path yang benar.")

# ==========================================
# HALAMAN 2: EKSPLORASI DATA (EDA)
# ==========================================
elif menu == 'Eksplorasi Data (EDA)':
    st.title('📈 Visualisasi & Eksplorasi Data Mall')
    st.markdown("Bagian ini mempraktikkan visualisasi grafik menggunakan Matplotlib dan **Seaborn**.")
    st.divider()

    if df_mall is not None:
        tab1, tab2, tab3 = st.tabs(["Pendapatan vs Pengeluaran", "Distribusi Usia", "Tren Data Pelanggan"])

        with tab1:
            st.subheader("Penyebaran Pendapatan Tahunan vs Skor Pengeluaran")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df_mall, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Genre', palette='deep', s=100, ax=ax)
            plt.title("Sebaran Pelanggan Berdasarkan Pendapatan & Pengeluaran")
            st.pyplot(fig)

        with tab2:
            st.subheader("Distribusi Rentang Usia Pelanggan")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.histplot(df_mall['Age'], bins=10, kde=True, color='purple', ax=ax2)
            plt.title("Distribusi Usia Pelanggan Mal")
            st.pyplot(fig2)

        with tab3:
            st.subheader("Grafik Garis / Tren Berdasarkan ID Pelanggan")
            st.line_chart(df_mall[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].head(30))
    else:
        st.error("Dataset `Mall_Customers.csv` tidak tersedia.")

# ==========================================
# HALAMAN 3: ANALISIS GENDER & K-MEANS
# ==========================================
elif menu == 'Analisis Gender & K-Means':
    st.title('🤖 Analisis Gender & Penerapan K-Means Clustering')
    st.markdown("Halaman ini membandingkan data pembelian berdasarkan jenis kelamin (*Male* vs *Female*) serta menjalankan algoritma klustering secara langsung.")
    st.divider()

    if df_mall is not None:
        st.subheader("📊 Perbandingan Jumlah & Skor Pembelian Berdasarkan Gender")
        gender_summary = df_mall.groupby('Genre').agg(
            Total_Pelanggan=('CustomerID', 'count'),
            Rata_Rata_Pendapatan=('Annual Income (k$)', 'mean'),
            Rata_Rata_Spending=('Spending Score (1-100)', 'mean'),
            Total_Spending_Score=('Spending Score (1-100)', 'sum')
        ).reset_index()

        st.dataframe(gender_summary, use_container_width=True)

        st.markdown("#### 📉 Grafik Total Skor Pengeluaran Berdasarkan Gender")
        st.bar_chart(gender_summary.set_index('Genre')['Total_Spending_Score'])

        st.divider()

        st.subheader("⚙️ Implementasi Algoritma K-Means Clustering")
        X = df_mall[['Annual Income (k$)', 'Spending Score (1-100)']]
        k_value = st.slider("Pilih Jumlah Klaster (K)", min_value=2, max_value=6, value=5)

        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        df_mall['Cluster'] = kmeans.fit_predict(X)

        fig_km, ax_km = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            data=df_mall,
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            hue='Cluster',
            palette='viridis',
            s=110,
            ax=ax_km,
            style='Genre'
        )

        centroids = kmeans.cluster_centers_
        ax_km.scatter(centroids[:, 0], centroids[:, 1], s=350, c='red', marker='X', label='Centroids')

        plt.title(f"Hasil K-Means Clustering Pelanggan Mal (K = {k_value})")
        plt.xlabel("Annual Income (k$)")
        plt.ylabel("Spending Score (1-100)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig_km)
    else:
        st.error("Dataset `Mall_Customers.csv` tidak tersedia.")

# ==========================================
# HALAMAN 4: PERBANDINGAN ALGORITMA KLUSTERING
# ==========================================
elif menu == 'Perbandingan Algoritma Klustering':
    st.title('🔬 Perbandingan Algoritma Klustering')
    st.markdown("Halaman ini menampilkan perbandingan visual dan karakteristik antara **K-Means Clustering**, **Hierarchical Clustering**, dan **DBSCAN**.")
    st.divider()

    if df_mall is not None:
        # Preprocessing data untuk algoritma lain
        X = df_mall[['Annual Income (k$)', 'Spending Score (1-100)']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 1. K-Means
        kmeans_comp = KMeans(n_clusters=5, random_state=42, n_init=10)
        df_mall['KMeans_C'] = kmeans_comp.fit_predict(X)

        # 2. Hierarchical (Agglomerative)
        hc_comp = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
        df_mall['HC_C'] = hc_comp.fit_predict(X_scaled)

        # 3. DBSCAN
        dbscan_comp = DBSCAN(eps=0.5, min_samples=5)
        df_mall['DBSCAN_C'] = dbscan_comp.fit_predict(X_scaled)

        # Menampilkan Grafik Perbandingan 3 Algoritma
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.scatterplot(data=df_mall, x='Annual Income (k$)', y='Spending Score (1-100)', hue='KMeans_C', palette='viridis', ax=axes[0], s=80)
        axes[0].set_title("1. K-Means (K=5)")
        axes[0].scatter(kmeans_comp.cluster_centers_[:, 0], kmeans_comp.cluster_centers_[:, 1], s=250, c='red', marker='X', label='Centroid')

        sns.scatterplot(data=df_mall, x='Annual Income (k$)', y='Spending Score (1-100)', hue='HC_C', palette='plasma', ax=axes[1], s=80)
        axes[1].set_title("2. Hierarchical Clustering (Ward)")

        sns.scatterplot(data=df_mall, x='Annual Income (k$)', y='Spending Score (1-100)', hue='DBSCAN_C', palette='deep', ax=axes[2], s=80)
        axes[2].set_title("3. DBSCAN (Auto / Noise Detection)")

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### 📋 Ringkasan Perbedaan Algoritma")
        st.write("""
        K-Means Clustering: Membutuhkan penentuan nilai $K$ di awal, sangat cepat, dan menggunakan titik pusat (centroid).
        Hierarchical Clustering: Membentuk struktur bertingkat (hierarki) dan sangat baik untuk melihat hubungan jarak antar kelompok data.
        DBSCAN: Berbasis kerapatan (density), mampu mendeteksi bentuk klaster bebas, serta otomatis mendeteksi data anomali (*noise*).
        """)
    else:
        st.error("Dataset `Mall_Customers.csv` tidak tersedia.")

# ==========================================
# FORM SIMULASI SEGMEN (Perbaikan Menggunakan Model K-Means Asli)
# ==========================================
elif menu == 'Form Simulasi Segmen':
    st.title('📝 Form Simulasi Prediksi Segmen Pelanggan')
    st.markdown("Masukkan karakteristik pelanggan baru. Hasil prediksi akan dikelompokkan secara otomatis menggunakan model **K-Means Clustering** yang telah dilatih dari dataset.")
    st.divider()

    # Menggunakan form agar rapi, tombol submit memproses data sekaligus
    with st.form("form_segmentasi"):
        st.subheader("Formulir Input Data Pelanggan")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            gender_input = st.selectbox("Jenis Kelamin", ["Male", "Female"])
            age_input = st.slider("Usia (Tahun)", min_value=18, max_value=70, value=30)
        with col_f2:
            income_input = st.number_input("Pendapatan Tahunan (k$)", min_value=10, max_value=150, value=50)
            spending_input = st.slider("Skor Pengeluaran / Spending Score (1-100)", min_value=1, max_value=100, value=50)

        # Tombol Submit Form
        submit_val = st.form_submit_button(label="Proses Prediksi Segmen")

    # Logika yang dijalankan setelah tombol "Proses Prediksi Segmen" ditekan
    if submit_val:
        if df_mall is not None:
            st.success("Data berhasil diproses oleh model K-Means!")

            # 1. Melatih ulang model K-Means dengan K=5 (atau jumlah cluster optimal)
            X_train = df_mall[['Annual Income (k$)', 'Spending Score (1-100)']]
            kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans_model.fit(X_train)

            # 2. Melakukan prediksi klaster untuk input baru dari user
            input_data = [[income_input, spending_input]]
            predicted_cluster = kmeans_model.predict(input_data)[0]

            # 3. Menampilkan Hasil Prediksi Klaster
            st.markdown(f"### Hasil Analisis Segmen:")
            st.info(f"Pelanggan Masuk ke dalam **Klaster {predicted_cluster}**")

            # Penjelasan deskriptif berdasarkan karakteristik klaster
            if income_input > 70 and spending_input > 60:
                deskripsi = "🌟 **Target Utama / VIP**: Pendapatan tinggi dan skor pengeluaran tinggi. Sangat potensial untuk program loyalitas."
            elif income_input > 70 and spending_input <= 40:
                deskripsi = "🛡️ **Konservatif / Hartawan Hemat**: Pendapatan tinggi namun skor pengeluaran rendah/selektif."
            elif income_input <= 40 and spending_input > 60:
                deskripsi = "🔥 **Berisiko / Boros**: Pendapatan rendah namun memiliki skor pengeluaran yang tinggi."
            elif income_input <= 40 and spending_input <= 40:
                deskripsi = "💤 **Ekonomis / Hati-hati**: Pendapatan rendah dan skor pengeluaran rendah."
            else:
                deskripsi = "⚖️ **Standar / Menengah**: Pelanggan reguler dengan pola belanja yang stabil dan moderat."

            st.write(f"**Karakteristik & Rekomendasi:** {deskripsi}")
        else:
            st.error("Dataset `Mall_Customers.csv` tidak ditemukan, sehingga model tidak dapat melakukan prediksi.")

# ==========================================
# HALAMAN 6: TENTANG KAMI
# ==========================================
elif menu == 'Tentang Kami':
    st.title('ℹ️ Tentang Pengembang')
    st.write("Aplikasi web ini disusun untuk memenuhi tugas laporan praktikum/proyek Machine Learning berbasis Streamlit.")
    st.markdown("---")
    st.caption("© 2026 - Mall Customer Segmentation App")
