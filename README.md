# 🧪 Drug Toxicity Predictor v4.0

> Sistem prediksi toksisitas senyawa kimia berbasis **Multi-Task Machine Learning** dan **Graph Neural Network (GNN)** menggunakan dataset **Tox21 (NIH)** — 12 target biologis, SMOTE + 5-Fold StratifiedKFold CV, SHAP Explainability.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Multi--Task-FF6600?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-GNN%20AttentiveFP-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-2E86AB?style=flat)
![Tox21](https://img.shields.io/badge/Dataset-Tox21%20NIH-27AE60?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📋 Daftar Isi

- [Tentang Project](#-tentang-project)
- [Fitur Utama](#-fitur-utama)
- [Dataset](#-dataset)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Struktur Folder](#-struktur-folder)
- [Instalasi](#-instalasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Hasil Evaluasi](#-hasil-evaluasi)
- [Teknologi](#-teknologi)
- [Limitasi](#-limitasi)
- [Referensi](#-referensi)

---

## 🔬 Tentang Project

Project ini merupakan **Final Project** mata kuliah Machine Learning yang membangun sistem prediksi toksisitas senyawa kimia secara *in silico* (komputasi). Sistem memprediksi apakah sebuah senyawa bersifat toksik terhadap **12 target biologis** dari dataset Tox21 NIH.

Pengguna cukup memasukkan **SMILES** senyawa, lalu sistem memberikan:
- Probabilitas toksisitas per target
- Penjelasan fitur penting (SHAP)
- Atom highlight pada struktur molekul
- Interpretasi hasil untuk orang awam

> ⚠️ **Disclaimer:** Prediksi bersifat *in silico* dan hanya untuk keperluan riset awal. Tidak menggantikan uji laboratorium atau penilaian ahli toksikologi.

---

## ✨ Fitur Utama

| Fitur | Keterangan |
|-------|-----------|
| 🎯 Multi-Task Learning | 12 target Tox21 dilatih dalam satu pipeline |
| 🧬 GNN AttentiveFP | Graph Neural Network berbasis PyTorch Geometric |
| ⚖️ SMOTE | Menangani class imbalance (rasio hingga 1:22) |
| 📊 5-Fold StratifiedKFold CV | Evaluasi robust dengan 95% Confidence Interval |
| 🎚️ Threshold Optimization | Threshold optimal per target via F1–PR Curve |
| 🔍 SHAP Explainability | Feature importance + atom highlight |
| 🌿 PubChem Integration | Nama senyawa otomatis dari SMILES via API |
| 🎨 UI Interaktif | Widget ipywidgets dengan CSS custom (Casa Chromatica) |
| 💊 Profil ADMET | Absorpsi, Distribusi, Metabolisme, Ekskresi, Toksisitas |

---

## 📦 Dataset

**Tox21 (Toxicology in the 21st Century)** — dataset dari NIH/EPA/FDA:

| Properti | Nilai |
|----------|-------|
| Sumber | NIH National Center for Advancing Translational Sciences |
| Jumlah senyawa | ~8.000 molekul unik |
| Target biologis | 12 target (NR + SR pathway) |
| Format | SMILES + label biner (0=tidak toksik, 1=toksik) |
| Download | Otomatis via script (AWS S3) |

### 12 Target Biologis

| # | Target | Deskripsi |
|---|--------|-----------|
| 1 | NR-AR | Androgen Receptor |
| 2 | NR-AR-LBD | AR Ligand Binding Domain |
| 3 | NR-AhR | Aryl Hydrocarbon Receptor |
| 4 | NR-Aromatase | Aromatase Enzyme |
| 5 | NR-ER | Estrogen Receptor Alpha |
| 6 | NR-ER-LBD | ER Ligand Binding Domain |
| 7 | NR-PPAR-gamma | PPAR-gamma Receptor |
| 8 | SR-ARE | Antioxidant Response Element |
| 9 | SR-ATAD5 | ATAD5 Genotoxicity |
| 10 | SR-HSE | Heat Shock Response |
| 11 | SR-MMP | Mitochondrial Membrane Potential |
| 12 | SR-p53 | p53 Tumor Suppressor |

---

## 🏗️ Arsitektur Sistem

```
SMILES Input
    │
    ├─── RDKit Featurization
    │       ├── Morgan Fingerprint (2048-bit, radius=2)
    │       └── RDKit Descriptors (52 → seleksi korelasi r>0.95)
    │
    ├─── Classical ML Pipeline
    │       ├── SMOTE (k_neighbors guard)
    │       ├── 5-Fold StratifiedKFold CV
    │       ├── XGBoost (scale_pos_weight per target)
    │       └── Threshold Optimization (F1-PR Curve)
    │
    ├─── GNN Pipeline (PyTorch Geometric)
    │       ├── MolGraph Featurizer (atom + edge features)
    │       ├── AttentiveFP Multi-Task (3 layers, hidden=200)
    │       ├── Masked BCE Loss (handle missing labels)
    │       └── StratifiedKFold CV (label proxy NR-AR)
    │
    └─── Output
            ├── Probabilitas per target
            ├── SHAP Feature Importance
            ├── Atom Highlight (Morgan→SHAP proxy)
            └── Interpretasi orang awam
```

### Pipeline Perbaikan (v4.0)

Versi ini memperbaiki **16 bug kritis** dari versi sebelumnya:

- ✅ Tidak ada collision variabel `dc` (deepchem) dengan descriptor loop
- ✅ `valid_idx_feat` disimpan terpisah — tidak ditimpa GNN CV loop
- ✅ `StratifiedKFold` (bukan `KFold`) untuk GNN CV
- ✅ CV GNN menyimpan AUC per-target (NR-AR) — apple-to-apple dengan XGBoost
- ✅ SMOTE k_neighbors guard: `max(1, min(5, n_pos-1))` + skip jika n_pos < 2
- ✅ Tidak double-weighting SMOTE + scale_pos_weight
- ✅ Nama file model konsisten: `mt_{task}_xgb.pkl`
- ✅ `gnn_model` sebagai alias eksplisit setelah training

---

## 📁 Struktur Folder

```
drug-toxicity-predictor/
│
├── Drug_Toxicity_Predictor_v4.ipynb   # Notebook utama
├── README.md                           # File ini
├── requirements.txt                    # Daftar dependensi
│
├── saved_models/                       # Model tersimpan (generate dari notebook)
│   ├── NR-AR_xgboost.pkl
│   ├── NR-AR_scaler.pkl
│   ├── NR-AR_desc_names.pkl
│   ├── mt_{task}_xgb.pkl              # Model per target
│   ├── mt_scaler.pkl
│   ├── mt_valid_tasks.pkl
│   ├── mt_thresholds.json
│   └── best_gnn.pt                    # GNN checkpoint
│
└── output_plots/                       # Plot yang dihasilkan
    ├── eda_distribution.png
    ├── feature_correlation.png
    ├── smote_distribution.png
    ├── cv_results.png
    ├── model_evaluation.png
    ├── multitask_analysis.png
    ├── gnn_training_curves.png
    ├── classical_vs_gnn_v4.png
    ├── shap_summary.png
    └── explainability_comparison.png
```

> 📌 **Catatan:** File `.pkl` dan `.pt` di `saved_models/` tidak diupload ke GitHub karena ukurannya besar. Generate sendiri dengan menjalankan notebook dari B1 hingga B11.

---

## ⚙️ Instalasi

### Prasyarat

- Python 3.10+
- Git
- VSCode (rekomendasi) atau Jupyter Notebook

### Langkah 1 — Clone Repository

```bash
git clone https://github.com/username/drug-toxicity-predictor.git
cd drug-toxicity-predictor
```

### Langkah 2 — Buat Virtual Environment

```bash
# Buat venv
python -m venv tox_env

# Aktivasi — Windows
tox_env\Scripts\activate

# Aktivasi — Mac/Linux
source tox_env/bin/activate
```

### Langkah 3 — Install Dependensi

```bash
pip install -r requirements.txt
```

### Langkah 4 — Install PyTorch + PyG

**CPU (Windows/Linux):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

**Mac (Apple Silicon):**
```bash
pip install torch torchvision torchaudio
pip install torch_geometric
```

### Langkah 5 — Register Jupyter Kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name=tox_env --display-name "Python (tox_env)"
```

### Verifikasi Instalasi

```bash
python -c "
import rdkit; print('rdkit      :', rdkit.__version__)
import deepchem; print('deepchem   :', deepchem.__version__)
import torch; print('torch      :', torch.__version__)
import torch_geometric; print('pyg        :', torch_geometric.__version__)
import xgboost; print('xgboost    :', xgboost.__version__)
import shap; print('shap       :', shap.__version__)
import sklearn; print('sklearn    :', sklearn.__version__)
import imblearn; print('imblearn   :', imblearn.__version__)
print('Semua library OK!')
"
```

---

## 🚀 Cara Menjalankan

### Opsi A — Sebagai Web App (Rekomendasi)

Menggunakan **Voilà** — notebook langsung jadi web app di browser, tanpa kelihatan kode:

```bash
# Install Voilà
pip install voila

# Jalankan
voila Drug_Toxicity_Predictor_v4.ipynb --theme=light
```

Browser otomatis terbuka di `http://localhost:8866`

Untuk akses dari perangkat lain di jaringan yang sama:
```bash
voila Drug_Toxicity_Predictor_v4.ipynb --port=8866 --no-browser
# Akses: http://[IP_komputer]:8866
```

### Opsi B — Di VSCode

1. Buka VSCode dari folder project: `code .`
2. Buka `Drug_Toxicity_Predictor_v4.ipynb`
3. Pilih kernel: pojok kanan atas → **Python (tox_env)**
4. Jalankan cell dari atas ke bawah (**Run All**)

### Opsi C — Jupyter Notebook

```bash
jupyter notebook Drug_Toxicity_Predictor_v4.ipynb
```

### Urutan Eksekusi Cell

| Bagian | Cell | Deskripsi | Durasi |
|--------|------|-----------|--------|
| B1 | Install | Hanya sekali, lalu restart | ~5 menit |
| B1 | Import | Wajib setiap sesi | < 1 menit |
| B2 | Load dataset | Download Tox21 (~2 MB) | < 1 menit |
| B3 | EDA | Visualisasi distribusi | < 1 menit |
| B4 | Feature engineering | Seleksi korelasi | < 2 menit |
| B5 | SMOTE + CV | 5-fold cross-validation | ~3 menit |
| B6 | Training final | XGBoost + RF | ~2 menit |
| B7 | Multi-task | 12 target XGBoost | ~5 menit |
| B8 | GNN training | AttentiveFP 30 epoch | ~15 menit |
| B9 | GNN CV | 5-fold GNN | ~20 menit |
| B10 | Perbandingan | Visualisasi | < 1 menit |
| B11 | SHAP | Explainability | ~2 menit |
| **B12** | **Widget UI** | **Sistem siap dipakai** | < 1 menit |

> ⚡ **Resume Session:** Jika kernel sudah pernah dijalankan dan model tersimpan, skip B1 Install dan jalankan dari **B1 Import → B4 → B5 → B6 → B7 → B11 → B12**.

### Cara Menggunakan Widget

1. Masukkan **SMILES** senyawa di kolom input, atau klik salah satu contoh cepat
2. Pilih **Target Biologis** dari dropdown (12 target tersedia)
3. Pilih **Model**: Classical ML (XGBoost) atau GNN (AttentiveFP)
4. Klik **🔬 Analisis Toksisitas**
5. Hasil muncul di bawah: prediksi, properti molekul, SHAP, interpretasi awam

**Contoh SMILES yang bisa dicoba:**

| Senyawa | SMILES | Prediksi NR-AR |
|---------|--------|----------------|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | Tidak Toksik |
| Caffeine | `Cn1cnc2c1c(=O)n(C)c(=O)n2C` | Tidak Toksik |
| Testosterone | `CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C` | Toksik |
| Benzene | `c1ccccc1` | Toksik |
| Ibuprofen | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` | Tidak Toksik |

---

## 📊 Hasil Evaluasi

### XGBoost Multi-Task (5-Fold StratifiedKFold CV)

| Target | AUC | F1 | MCC |
|--------|-----|----|-----|
| NR-AR | ~0.74 | ~0.42 | ~0.38 |
| NR-AhR | ~0.82 | ~0.56 | ~0.51 |
| SR-MMP | ~0.78 | ~0.61 | ~0.55 |
| SR-p53 | ~0.76 | ~0.49 | ~0.44 |

> Nilai aktual bervariasi tergantung random seed dan versi library.

### Perbandingan Classical ML vs GNN (NR-AR)

| Model | CV AUC (5-Fold) | Test AUC |
|-------|-----------------|----------|
| XGBoost (Multi-Task) | ~0.74 ± 0.03 | ~0.75 |
| GNN AttentiveFP | ~0.72 ± 0.05 | ~0.73 |

---

## 🛠️ Teknologi

| Kategori | Library | Versi |
|----------|---------|-------|
| Cheminformatics | RDKit | ≥ 2023.09 |
| ML Framework | scikit-learn | ≥ 1.3 |
| Gradient Boosting | XGBoost | ≥ 2.0 |
| GNN | PyTorch Geometric | ≥ 2.4 |
| Deep Learning | PyTorch | ≥ 2.0 |
| Molecular ML | DeepChem | ≥ 2.7 |
| Imbalanced Data | imbalanced-learn | ≥ 0.11 |
| Explainability | SHAP | ≥ 0.44 |
| UI Widget | ipywidgets | ≥ 8.0 |
| Web App | Voilà | ≥ 0.5 |
| Data | pandas, numpy | Latest |
| Visualisasi | matplotlib, seaborn | Latest |

---

## ⚠️ Limitasi

1. **Validasi eksternal** — Belum diuji pada dataset eksternal selain Tox21 challenge test set
2. **GNN Attention** — Visualisasi atom menggunakan proxy kimia, bukan attention weight GNN yang sesungguhnya
3. **Hyperparameter** — Belum dilakukan tuning sistematis (Optuna/GridSearch)
4. **Epoch GNN** — 30 epoch mungkin belum konvergen untuk semua target
5. **Validasi Lab** — Prediksi *in silico* belum divalidasi dengan uji basah di laboratorium

---

## 📚 Referensi

1. **Dataset Tox21:** Tox21 Data Challenge 2014, NIH National Center for Advancing Translational Sciences. https://tripod.nih.gov/tox21/
2. **AttentiveFP:** Xiong et al. (2020). "Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism." *Journal of Medicinal Chemistry*.
3. **SHAP:** Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
4. **RDKit:** Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org
5. **DeepChem:** Ramsundar et al. (2019). "Deep Learning for the Life Sciences." O'Reilly Media.
6. **SMOTE:** Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR*.

---

## 👤 Author

**Nama:** [Nama Kamu]  
**NIM:** [NIM Kamu]  
**Mata Kuliah:** Machine Learning  
**Institusi:** [Nama Kampus]  
**Tahun:** 2025

---

## 📄 Lisensi

Project ini dibuat untuk keperluan akademik. Dataset Tox21 merupakan milik NIH dan digunakan sesuai ketentuan penggunaan publik.

---

<div align="center">
  <sub>⚗️ Drug Toxicity Predictor v4.0 — Final Project Machine Learning</sub>
</div>
#   d r u g - t o x i c i t y - p r e d i c t o r _ f i n a l p r o j e c t _ r g  
 