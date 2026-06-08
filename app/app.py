import streamlit as st
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="centered"
)

# Load models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

@st.cache_resource
def load_models():
    rf = joblib.load(os.path.join(MODEL_DIR, 'random_forest.pkl'))
    kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    risk_map = joblib.load(os.path.join(MODEL_DIR, 'risk_map.pkl'))
    return rf, kmeans, scaler, risk_map

rf, kmeans, scaler, risk_map = load_models()

# Header
st.markdown("""
    <h1 style='text-align:center; color:#2c3e50;'>🩺 Prediksi Risiko Diabetes</h1>
    <p style='text-align:center; color:#7f8c8d; font-size:16px;'>
        Segmentasi & Prediksi berbasis K-Means dan Random Forest<br>
        <i>Pima Indians Diabetes Dataset</i>
    </p>
    <hr>
""", unsafe_allow_html=True)

st.markdown("### 📋 Input Data Pasien")
st.markdown("Isi data kesehatan pasien di bawah ini:")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=20, value=3, step=1)
    glucose = st.number_input("Glukosa Darah (Glucose) mg/dL", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Tekanan Darah (BloodPressure) mmHg", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Ketebalan Kulit (SkinThickness) mm", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Kadar Insulin (Insulin) μU/mL", min_value=0, max_value=900, value=80)
    bmi = st.number_input("Indeks Massa Tubuh (BMI) kg/m²", min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01, format="%.3f")
    age = st.number_input("Usia (Age) tahun", min_value=1, max_value=120, value=33)

st.markdown("---")

if st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary"):
    # Prepare input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    # Clustering
    cluster_id = kmeans.predict(input_scaled)[0]
    risk_label = risk_map.get(cluster_id, "Tidak Diketahui")

    # Classification
    pred = rf.predict(input_scaled)[0]
    prob = rf.predict_proba(input_scaled)[0][1]

    st.markdown("## 📊 Hasil Analisis")

    c1, c2, c3 = st.columns(3)

    # Cluster result
    color_map = {"Risiko Rendah": "#27ae60", "Risiko Sedang": "#f39c12", "Risiko Tinggi": "#e74c3c"}
    cluster_color = color_map.get(risk_label, "#95a5a6")
    with c1:
        st.markdown(f"""
        <div style='background:{cluster_color}22; border-left: 5px solid {cluster_color};
                    padding:15px; border-radius:8px; text-align:center;'>
            <p style='color:{cluster_color}; font-size:13px; margin:0; font-weight:600;'>SEGMEN RISIKO</p>
            <h3 style='color:{cluster_color}; margin:5px 0;'>{risk_label}</h3>
            <p style='color:#666; font-size:12px; margin:0;'>Cluster {cluster_id}</p>
        </div>
        """, unsafe_allow_html=True)

    # Prediction result
    pred_color = "#e74c3c" if pred == 1 else "#27ae60"
    pred_text = "⚠️ Terindikasi Diabetes" if pred == 1 else "✅ Tidak Diabetes"
    with c2:
        st.markdown(f"""
        <div style='background:{pred_color}22; border-left: 5px solid {pred_color};
                    padding:15px; border-radius:8px; text-align:center;'>
            <p style='color:{pred_color}; font-size:13px; margin:0; font-weight:600;'>PREDIKSI</p>
            <h3 style='color:{pred_color}; margin:5px 0; font-size:16px;'>{pred_text}</h3>
        </div>
        """, unsafe_allow_html=True)

    # Probability
    prob_color = "#e74c3c" if prob > 0.5 else "#27ae60"
    with c3:
        st.markdown(f"""
        <div style='background:{prob_color}22; border-left: 5px solid {prob_color};
                    padding:15px; border-radius:8px; text-align:center;'>
            <p style='color:{prob_color}; font-size:13px; margin:0; font-weight:600;'>PROBABILITAS</p>
            <h3 style='color:{prob_color}; margin:5px 0;'>{prob*100:.1f}%</h3>
            <p style='color:#666; font-size:12px; margin:0;'>Kemungkinan diabetes</p>
        </div>
        """, unsafe_allow_html=True)

    # Progress bar
    st.markdown(f"""
    <br>
    <p style='margin:0; font-size:14px; color:#555;'>Probabilitas Diabetes</p>
    """, unsafe_allow_html=True)
    st.progress(float(prob))

    # Recommendation
    st.markdown("### 💡 Rekomendasi")
    if risk_label == "Risiko Tinggi" or pred == 1:
        st.error("Segera konsultasikan dengan dokter spesialis. Data menunjukkan risiko tinggi diabetes. Lakukan pemeriksaan lanjutan seperti tes HbA1c dan konsultasi pola makan.")
    elif risk_label == "Risiko Sedang":
        st.warning("Pertahankan pola hidup sehat. Pantau kadar glukosa secara rutin dan batasi konsumsi makanan tinggi gula. Olahraga teratur sangat dianjurkan.")
    else:
        st.success("Kondisi saat ini menunjukkan risiko rendah. Tetap jaga pola makan seimbang dan rutin berolahraga untuk mempertahankan kesehatan.")

    st.markdown("---")
    st.caption("⚠️ *Disclaimer: Hasil prediksi ini hanya bersifat indikatif dan tidak menggantikan diagnosis medis profesional.*")

# Sidebar info
with st.sidebar:
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.markdown("""
    **Dataset:** Pima Indians Diabetes  
    **Algoritma Clustering:** K-Means (K=3)  
    **Algoritma Klasifikasi:** Random Forest  
    **Akurasi Model:** 74.03%  
    **F1-Score:** 60.78%
    """)
    st.markdown("---")
    st.markdown("### 📌 Keterangan Fitur")
    st.markdown("""
    - **Glucose:** Kadar glukosa plasma 2 jam
    - **BMI:** Berat (kg) / Tinggi² (m)
    - **DPF:** Fungsi silsilah diabetes
    - **Insulin:** Kadar serum insulin
    """)
    st.markdown("---")
    st.caption("Dibuat untuk Tugas Akhir Data Mining\nSistem Informasi - Semester 4")
