import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN (Bab 2.7: Layout & Container) ---
st.set_page_config(
    page_title="Mall Customer Segmentation App",
    page_icon="🛍️",
    layout="wide"
)

# --- SIDEBAR NAVIGASI (Bab 2.7: st.sidebar) ---
st.sidebar.title("📌 Navigasi Menu")
menu = st.sidebar.selectbox("Pilih Halaman", ["Halaman Utama", "Eksplorasi Data (EDA)", "Form Simulasi Segmen", "Tentang Kami"])

# --- FUNGSI MEMUAT DATASET ---
@st.cache_data
def load_data():
    # Pastikan file Mall_Customers.csv berada dalam satu folder atau atur path-nya
    df = pd.read_csv('Mall_Customers.csv')
    return df

# Memanggil fungsi load data
try:
    df_mall = load_data()
except Exception as e:
    df_mall = None

# ==========================================
# 1. HALAMAN UTAMA (Bab 2.2: Elemen Text & Title)
# ==========================================
if menu == 'Halaman Utama':
    st.title('🛍️ Aplikasi Web Segmentasi Pelanggan Mal')
    st.markdown("Selamat datang di aplikasi web berbasis Streamlit untuk analisis dan prediksi Customer Segmentation menggunakan algoritma Unsupervised Learning (K-Means Clustering).")
    st.divider()

    # Menampilkan Subheader dan Metrik Ringkasan (Bab 2.3: st.metric)
    st.subheader("📊 Ringkasan Dataset Mall Customers")
    if df_mall is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pelanggan", f"{df_mall.shape[0]} Orang", "Aktif")
        col2.metric("Rata-rata Usia", f"{df_mall['Age'].mean():.1f} Tahun", "± 1.2")
        col3.metric("Rata-rata Spending Score", f"{df_mall['Spending Score (1-100)'].mean():.1f}", "Skor 1-100")

        st.markdown("### 🔍 Pratinjau Data Mentah")
        st.dataframe(df_mall.head(10), use_container_width=True)
    else:
        st.warning("File `Mall_Customers.csv` belum ditemukan di direktori utama. Harap unggah dataset terlebih dahulu.")

# ==========================================
# 2. EKSPLORASI DATA / EDA (Bab 2.4: Menampilkan Grafik)
# ==========================================
elif menu == 'Eksplorasi Data (EDA)':
    st.title('📈 Visualisasi & Eksplorasi Data Mall')
    st.markdown("Bagian ini mempraktikkan materi visualisasi grafik dari modul Bab 2 menggunakan **Matplotlib**, **Seaborn**, dan grafik bawaan Streamlit.")
    st.divider()

    if df_mall is not None:
        # Menggunakan Tab untuk mengorganisasi grafik (Bab 2.7: st.tabs)
        tab1, tab2, tab3 = st.tabs(["Grafik Scatter Pendapatan vs Pengeluaran", "Distribusi Usia Pelanggan", "Grafik Batang Perbandingan"])

        with tab1:
            st.subheader("Penyebaran Pendapatan Tahunan vs Skor Pengeluaran")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df_mall, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', palette='deep', s=100, ax=ax)
            plt.title("Cluster Sebaran Pelanggan Mal")
            st.pyplot(fig)

        with tab2:
            st.subheader("Distribusi Rentang Usia Pelanggan")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.histplot(df_mall['Age'], bins=10, kde=True, color='purple', ax=ax2)
            st.pyplot(fig2)

        with tab3:
            st.subheader("Grafik Garis / Tren Berdasarkan ID Pelanggan")
            st.line_chart(df_mall[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].head(30))
    else:
        st.error("Dataset `Mall_Customers.csv` tidak tersedia.")

# ==========================================
# 3. FORM SIMULASI SEGMEN (Bab 2.5 & 2.8: Input Form & Control Flow)
# ==========================================
elif menu == 'Form Simulasi Segmen':
    st.title('📝 Form Simulasi Prediksi Segmen Pelanggan')
    st.markdown("Masukkan data karakteristik pelanggan baru untuk mensimulasikan hasil pengelompokan (*clustering*).")
    st.divider()

    # Menggunakan st.form dan st.form_submit_button (Bab 2.8)
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

    # Logika setelah tombol submit ditekan (Control Flow)
    if submit_val:
        st.success("Data berhasil diproses oleh sistem!")

        # Simulasi aturan penentuan segmen sederhana (atau dihubungkan ke model K-Means .pkl nantinya)
        if income_input > 70 and spending_input > 60:
            segmen_hasil = "🌟 Target Utama (High Income, High Spending)"
            deskripsi = "Pelanggan VIP yang sering berbelanja dalam jumlah besar. Sangat potensial untuk program loyalitas."
        elif income_input > 70 and spending_input <= 40:
            segmen_hasil = "🛡️ Konservatif / Hartawan Hemat (High Income, Low Spending)"
            deskripsi = "Memiliki daya beli tinggi namun selektif dan jarang mengeluarkan uang banyak di mal."
        elif income_input <= 40 and spending_input > 60:
            segmen_hasil = "🔥 Berisiko / Boros (Low Income, High Spending)"
            deskripsi = "Pendapatan relatif rendah namun memiliki skor pengeluaran yang sangat tinggi."
        else:
            segmen_hasil = "⚖️ **Standar / Menengah (Medium Income, Medium Spending)**"
            deskripsi = "Pelanggan reguler dengan pola belanja yang stabil dan moderat."

        st.markdown(f"### Hasil Analisis Segmen:")
        st.info(f"Kategori Klaster: {segmen_hasil}")
        st.write(f"**Karakteristik:** {deskripsi}")

# ==========================================
# 4. TENTANG KAMI
# ==========================================
elif menu == 'Tentang Kami':
    st.title('ℹ️ Tentang Pengembang')
    st.write("Aplikasi ini disusun untuk memenuhi tugas laporan tutorial praktikum mata kuliah Machine Learning berbasis Web.")
    st.markdown("---")
    st.caption("© 2026 - Pengembangan Aplikasi Web Data Mining Kelompok Mal Customer Segmentation.")
