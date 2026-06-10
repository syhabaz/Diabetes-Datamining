# 🩺 Segmentasi dan Prediksi Risiko Diabetes
## Menggunakan Algoritma K-Means dan Random Forest

> **Tugas Akhir Mata Kuliah Data Mining / KDD**  
> Program Studi Sistem Informasi — Semester 4

---

## 📌 Deskripsi Project

Project ini merupakan implementasi end-to-end *data mining* untuk segmentasi dan prediksi risiko diabetes menggunakan dataset Pima Indians Diabetes. Project mencakup dua pendekatan utama:

| Pendekatan | Algoritma | Tujuan |
|---|---|---|
| **Clustering** | K-Means (K=3) | Segmentasi pasien berdasarkan profil kesehatan |
| **Classification** | Random Forest | Prediksi risiko diabetes (0/1) |

---

## 📁 Struktur Folder

```
diabetes-project/
│
├── dataset/
│   └── diabetes.csv                  # Pima Indians Diabetes Dataset
│
├── notebook/
│   └── diabetes.ipynb                # Notebook analisis lengkap
│
├── app/
│   └── app.py                        # Aplikasi Streamlit
│
├── model/
│   ├── random_forest.pkl             # Model klasifikasi
│   ├── kmeans.pkl                    # Model clustering
│   ├── scaler.pkl                    # StandardScaler
│   └── risk_map.pkl                  # Mapping cluster → label risiko
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Sumber:** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Jumlah Data:** 768 baris × 9 kolom
- **Target:** `Outcome` (0 = Tidak Diabetes, 1 = Diabetes)

**Fitur:**

| Fitur | Deskripsi |
|---|---|
| Pregnancies | Jumlah kehamilan |
| Glucose | Kadar glukosa plasma (mg/dL) |
| BloodPressure | Tekanan darah diastolik (mmHg) |
| SkinThickness | Ketebalan lipatan kulit trisep (mm) |
| Insulin | Kadar insulin serum 2 jam (μU/mL) |
| BMI | Indeks Massa Tubuh (kg/m²) |
| DiabetesPedigreeFunction | Fungsi silsilah diabetes |
| Age | Usia (tahun) |
| Outcome | Label kelas (0/1) |

---

## ⚙️ Preprocessing

1. **Imputasi nilai nol** — Kolom `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` memiliki nilai 0 yang tidak valid secara medis. Diganti dengan nilai median kolom bersangkutan.
2. **Standarisasi** — Menggunakan `StandardScaler` agar semua fitur berada pada skala yang sama.
3. **Train-test split** — 80% training, 20% testing (stratified).

---

## 🔵 Clustering (K-Means)

- **Metode penentuan K:** Elbow Method → K optimal = **3**
- **Hasil segmentasi:**

| Cluster | Label Risiko | Rata-rata Glukosa | Rata-rata BMI |
|---|---|---|---|
| 1 | Risiko Rendah | ~107 | ~28.5 |
| 0 | Risiko Sedang | ~131 | ~32.8 |
| 2 | Risiko Tinggi | ~137 | ~39.2 |

---

## 🌲 Classification (Random Forest)

| Metrik | Nilai |
|---|---|
| **Accuracy** | 74.03% |
| **Precision** | 64.58% |
| **Recall** | 57.41% |
| **F1 Score** | 60.78% |

---

## 🚀 Cara Menjalankan Project

### 1. Clone atau Download Repository

```bash
git clone https://github.com/username/diabetes-project.git
cd diabetes-project
```

### 2. Buat Virtual Environment (opsional tapi disarankan)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Notebook (EDA + Training)

```bash
jupyter notebook notebook/diabetes.ipynb
```

### 5. Jalankan Aplikasi Streamlit

```bash
streamlit run app/app.py
```

Buka browser dan akses: **http://localhost:8501**

---

## 📝 Catatan

- Model `.pkl` sudah tersedia di folder `model/` sehingga aplikasi dapat langsung dijalankan **tanpa perlu training ulang**.
- Jika ingin melatih ulang model, jalankan notebook dari awal.
- Pastikan Python versi ≥ 3.9.

---

## 👤 Identitas
> **Nama:** Muhammad Mirza Shahbaz Zaydan (24051214086)
> **Mata Kuliah:** Data Mining / Knowledge Discovery in Database  
> **Program Studi:** Sistem Informasi  
> **Semester:** 4  

---

## 📚 Referensi Dataset

Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988).  
*Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*  
Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265.
