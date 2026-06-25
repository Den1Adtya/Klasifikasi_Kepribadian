# 🌲 Personality Classifier – Random Forest

Aplikasi klasifikasi kepribadian **Introvert vs Extrovert** menggunakan **Random Forest Classifier**.

---

## 📁 Struktur File

```
streamlit_app/
│
├── app.py                          # Aplikasi Streamlit (deployment)
├── EnsembleLearning_RandomForest.py # Script Python standalone (laporan)
├── requirements.txt                # Dependency Python
├── README.md                       # Panduan ini
└── personality_dataset.csv         # ← Letakkan dataset di sini
```

---

## 🚀 Cara Menjalankan

### Option A – Streamlit (Web App)

```bash
# 1. Install dependency
pip install -r requirements.txt

# 2. Jalankan aplikasi
streamlit run app.py
```

Buka browser → `http://localhost:8501`

### Option B – Script Python Biasa

```bash
python EnsembleLearning_RandomForest.py
```

---

## ☁️ Deploy ke Streamlit Cloud

1. Push semua file ke repository GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New app** → pilih repo & branch
4. Set **Main file path** → `app.py`
5. Klik **Deploy**

> ⚠️ **Catatan**: Dataset `personality_dataset.csv` harus ikut di-push ke GitHub karena Streamlit Cloud tidak menerima upload lokal secara langsung. Atau gunakan fitur upload di sidebar aplikasi.

---

## 📊 Fitur Aplikasi

| Tab | Isi |
|-----|-----|
| 📋 Data Overview | Info dataset, statistik deskriptif, missing values |
| 📊 Visualisasi EDA | Scatter plot, bar chart, pie chart, heatmap korelasi |
| 🤖 Hasil Model | Akurasi, cross-validation, feature importance |
| 📈 Evaluasi | Classification report, confusion matrix, ROC curve |
| 🔮 Prediksi Baru | Input manual → prediksi Introvert/Extrovert |

---

## 🔧 Hyperparameter yang Bisa Diubah (via Sidebar)

- `n_estimators` – jumlah pohon (50–500)
- `max_depth` – kedalaman pohon (0 = None)
- `min_samples_split` – minimum sampel untuk split
- `test_size` – ukuran data uji (10–40%)

---

## 📦 Tambahan vs Notebook Asli

| Fitur | Notebook | Script/App |
|-------|----------|------------|
| Cross-validation | ❌ | ✅ |
| Feature Importance | ❌ | ✅ |
| Correlation Heatmap | ❌ | ✅ |
| Hyperparameter via UI | ❌ | ✅ |
| Prediksi data baru | ❌ | ✅ |
| Plot disimpan otomatis | ❌ | ✅ (script) |
