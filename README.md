<div align="center">

# 🧪 Drug Toxicity Predictor v4.0

**Prediksi toksisitas senyawa kimia berbasis ML & GNN secara _in silico_**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-GNN%20AttentiveFP-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Multi--Task-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-2E86AB?style=for-the-badge)](https://rdkit.org)
[![Tox21](https://img.shields.io/badge/Dataset-Tox21%20NIH-27AE60?style=for-the-badge)](https://tripod.nih.gov/tox21/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> Sistem prediksi toksisitas senyawa kimia secara komputasi (*in silico*) menggunakan
> **Multi-Task XGBoost** dan **Graph Neural Network AttentiveFP** pada **12 target biologis**
> dari dataset **Tox21 NIH** — dilengkapi SHAP explainability dan UI interaktif berbasis Voilà.

<br/>

[🔬 Tentang](#-tentang-project) · [✨ Fitur](#-fitur-utama) · [⚙️ Instalasi](#%EF%B8%8F-instalasi) · [🚀 Menjalankan](#-cara-menjalankan) · [📊 Evaluasi](#-hasil-evaluasi) · [📚 Referensi](#-referensi)

</div>

---

## 📋 Daftar Isi

- [Tentang Project](#-tentang-project)
- [Fitur Utama](#-fitur-utama)
- [Dataset & 12 Target Biologis](#-dataset)
- [Arsitektur Sistem](#%EF%B8%8F-arsitektur-sistem)
- [Struktur Folder](#-struktur-folder)
- [Instalasi](#%EF%B8%8F-instalasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Hasil Evaluasi](#-hasil-evaluasi)
- [Teknologi](#%EF%B8%8F-teknologi)
- [Limitasi](#-limitasi)
- [Referensi](#-referensi)

---

## 🔬 Tentang Project

Project ini merupakan **Final Project** mata kuliah Machine Learning yang membangun sistem prediksi toksisitas senyawa kimia secara *in silico* (komputasi). Sistem ini menjawab pertanyaan:

> *"Apakah senyawa X bersifat toksik terhadap reseptor biologis tertentu?"*

Alih-alih mengandalkan uji laboratorium yang mahal dan memakan waktu, sistem ini menggunakan representasi matematis dari struktur molekul (SMILES) untuk memprediksi toksisitas secara instan menggunakan dua pendekatan berbeda: **Classical ML (XGBoost)** dan **Graph Neural Network (AttentiveFP)**.

### Mengapa Tox21?

Dataset **Tox21** dari NIH adalah standar emas untuk *in silico* toxicology prediction, mencakup ~8.000 senyawa kimia yang telah diuji secara eksperimental terhadap **12 target biologis** kritis — mulai dari reseptor hormon hingga jalur stres seluler.

> ⚠️ **Disclaimer:** Prediksi bersifat *in silico* untuk keperluan riset awal. Tidak menggantikan uji laboratorium atau penilaian ahli toksikologi.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|:------|:----------|
| 🎯 **Multi-Task Learning** | 12 target Tox21 dilatih sekaligus dalam satu pipeline terintegrasi |
| 🧬 **GNN AttentiveFP** | Graph Neural Network berbasis PyTorch Geometric — molekul direpresentasikan sebagai graf atom-ikatan |
| ⚖️ **SMOTE Balancing** | Menangani class imbalance ekstrem (rasio positif:negatif hingga 1:22) |
| 📊 **5-Fold StratifiedKFold CV** | Evaluasi robust dengan 95% Confidence Interval per target |
| 🎚️ **Threshold Optimization** | Threshold klasifikasi optimal per target via F1–PR Curve |
| 🔍 **SHAP Explainability** | Visualisasi feature importance + atom highlight pada struktur molekul |
| 🌿 **PubChem Integration** | Resolusi nama senyawa otomatis dari SMILES via PubChem REST API |
| 🎨 **UI Interaktif** | Dashboard ipywidgets dengan desain CSS custom (Casa Chromatica palette) |
| 💊 **Profil ADMET** | Kalkulasi properti Absorpsi, Distribusi, Metabolisme, Ekskresi, Toksisitas |
| 🌐 **Web App Ready** | Deploy sebagai web app tanpa kode terlihat menggunakan Voilà |

---

## 📦 Dataset

**Tox21 (Toxicology in the 21st Century)** adalah inisiatif kolaborasi NIH/EPA/FDA untuk modernisasi uji toksikologi:

| Properti | Nilai |
|:---------|:------|
| Sumber | NIH National Center for Advancing Translational Sciences |
| Jumlah senyawa | ~8.000 molekul unik |
| Target biologis | 12 target (NR pathway + SR pathway) |
| Format input | SMILES string |
| Label | Biner — 0 = tidak toksik, 1 = toksik |
| Missing labels | Ada (~30–60% per target) — ditangani dengan masked loss |
| Download | Otomatis via script (AWS S3) |

### 🎯 12 Target Biologis

<table>
<tr>
<th>#</th><th>Target</th><th>Deskripsi</th><th>Jalur</th><th>Relevansi Klinis</th>
</tr>
<tr>
<td>1</td><td><code>NR-AR</code></td><td>Androgen Receptor</td>
<td rowspan="7" align="center"><b>Nuclear Receptor</b><br/><sub>(NR)</sub></td>
<td>Gangguan hormonal, kanker prostat</td>
</tr>
<tr><td>2</td><td><code>NR-AR-LBD</code></td><td>AR Ligand Binding Domain</td><td>Endocrine disruptor</td></tr>
<tr><td>3</td><td><code>NR-AhR</code></td><td>Aryl Hydrocarbon Receptor</td><td>Toksisitas dioxin, karsinogenesis</td></tr>
<tr><td>4</td><td><code>NR-Aromatase</code></td><td>Aromatase Enzyme</td><td>Gangguan estrogen, kanker payudara</td></tr>
<tr><td>5</td><td><code>NR-ER</code></td><td>Estrogen Receptor Alpha</td><td>Endocrine disruptor, reproduksi</td></tr>
<tr><td>6</td><td><code>NR-ER-LBD</code></td><td>ER Ligand Binding Domain</td><td>Endocrine disruptor</td></tr>
<tr><td>7</td><td><code>NR-PPAR-gamma</code></td><td>PPAR-gamma Receptor</td><td>Metabolisme lipid, diabetes</td></tr>
<tr>
<td>8</td><td><code>SR-ARE</code></td><td>Antioxidant Response Element</td>
<td rowspan="5" align="center"><b>Stress Response</b><br/><sub>(SR)</sub></td>
<td>Stres oksidatif, hepatotoksisitas</td>
</tr>
<tr><td>9</td><td><code>SR-ATAD5</code></td><td>ATAD5 Genotoxicity</td><td>Kerusakan DNA, mutagenisitas</td></tr>
<tr><td>10</td><td><code>SR-HSE</code></td><td>Heat Shock Response</td><td>Stres protein, sitotoksisitas</td></tr>
<tr><td>11</td><td><code>SR-MMP</code></td><td>Mitochondrial Membrane Potential</td><td>Toksisitas mitokondria</td></tr>
<tr><td>12</td><td><code>SR-p53</code></td><td>p53 Tumor Suppressor</td><td>Aktivasi jalur DNA damage</td></tr>
</table>

---

## 🏗️ Arsitektur Sistem

### Gambaran Umum

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                            │
│          SMILES String  (e.g. CC(=O)Oc1ccccc1C(=O)O)       │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │      RDKit Featurization   │
              │  ┌─────────────────────┐   │
              │  │ Morgan Fingerprint  │   │
              │  │  2048-bit, radius=2 │   │
              │  └─────────────────────┘   │
              │  ┌─────────────────────┐   │
              │  │  RDKit Descriptors  │   │
              │  │  52 physicochemical │   │
              │  │  (filter r > 0.95)  │   │
              │  └─────────────────────┘   │
              └──────┬──────────────┬──────┘
                     │              │
        ┌────────────▼───┐    ┌─────▼──────────────┐
        │  Classical ML  │    │    GNN Pipeline     │
        │   Pipeline     │    │  (PyTorch Geometric)│
        │                │    │                     │
        │ • SMOTE        │    │ • MolGraph Builder  │
        │ • 5-Fold CV    │    │   (atom + edge feat)│
        │ • XGBoost      │    │ • AttentiveFP       │
        │   Multi-Task   │    │   3 layers, h=200   │
        │ • Threshold    │    │ • Masked BCE Loss   │
        │   Optimization │    │ • StratifiedKFold   │
        └────────┬───────┘    └──────────┬──────────┘
                 │                       │
        ┌────────▼───────────────────────▼──────────┐
        │                OUTPUT LAYER                │
        │  • Probabilitas toksisitas (12 target)     │
        │  • Prediksi biner + confidence             │
        │  • SHAP feature importance plot            │
        │  • Atom highlight pada struktur molekul    │
        │  • Profil ADMET senyawa                    │
        │  • Interpretasi bahasa awam                │
        └────────────────────────────────────────────┘
```

### Detail Komponen

#### 🤖 Classical ML (XGBoost Multi-Task)

```
Input Features (2048 + 52 = 2100 dim)
    │
    ├── Per-target pipeline (× 12):
    │       ├── SMOTE oversampling
    │       │     └── k_neighbors = max(1, min(5, n_pos - 1))
    │       │         skip jika n_pos < 2
    │       ├── XGBoost Classifier
    │       │     └── scale_pos_weight = n_neg / n_pos
    │       └── Threshold optimization via F1-PR Curve
    │
    └── 5-Fold StratifiedKFold CV → AUC ± CI per target
```

#### 🧬 GNN AttentiveFP

```
SMILES → MolGraph (PyG Data object)
    │
    Atom features (per node):
    ├── Atomic number, degree, formal charge
    ├── Hybridization, aromaticity
    └── Ring membership

    Bond features (per edge):
    ├── Bond type (single/double/triple/aromatic)
    ├── Conjugation, ring membership
    └── Stereo configuration
    │
    AttentiveFP Layers (× 3, hidden=200)
    ├── Graph attention (intra-molecular)
    └── Readout attention (graph-level)
    │
    Multi-Task Head (× 12 target)
    └── Masked BCE Loss (abaikan label NaN)
```

### Perbaikan v4.0 — 16 Bug Fix

| # | Bug | Status |
|:-:|:----|:------:|
| 1 | Collision variabel `dc` (deepchem) dengan descriptor loop | ✅ Fixed |
| 2 | `valid_idx_feat` ditimpa GNN CV loop | ✅ Fixed |
| 3 | GNN CV menggunakan `KFold` bukan `StratifiedKFold` | ✅ Fixed |
| 4 | CV GNN tidak menyimpan AUC per-target | ✅ Fixed |
| 5 | SMOTE tanpa k_neighbors guard → crash jika n_pos kecil | ✅ Fixed |
| 6 | Double-weighting: SMOTE + scale_pos_weight aktif bersamaan | ✅ Fixed |
| 7 | Nama file model tidak konsisten antar cell | ✅ Fixed |
| 8 | `gnn_model` tidak punya alias eksplisit setelah training | ✅ Fixed |
| 9–16 | Bug minor lainnya (indexing, dtype, NaN handling, dll) | ✅ Fixed |

---

## 📁 Struktur Folder

```
drug-toxicity-predictor/
│
├── 📓 Drug_Toxicity_Predictor_v4.ipynb    # Notebook utama (semua cell)
├── 📄 README.md                            # Dokumentasi ini
├── 📋 requirements.txt                     # Semua dependensi Python
│
├── 🗄️ saved_models/                       # ⚠️ Tidak diupload ke GitHub
│   ├── mt_{task}_xgb.pkl                  # Model XGBoost per target (×12)
│   ├── mt_scaler.pkl                       # StandardScaler untuk fitur
│   ├── mt_valid_tasks.pkl                  # Daftar target yang valid
│   ├── mt_thresholds.json                  # Threshold optimal per target
│   ├── NR-AR_xgboost.pkl                  # Model single-task (baseline)
│   ├── NR-AR_scaler.pkl
│   ├── NR-AR_desc_names.pkl
│   └── best_gnn.pt                        # GNN checkpoint (PyTorch)
│
└── 📊 output_plots/                        # Plot hasil training & evaluasi
    ├── eda_distribution.png               # Distribusi label per target
    ├── feature_correlation.png            # Heatmap korelasi fitur
    ├── smote_distribution.png             # Sebelum/sesudah SMOTE
    ├── cv_results.png                     # Hasil 5-fold CV
    ├── model_evaluation.png               # ROC curve, confusion matrix
    ├── multitask_analysis.png             # Perbandingan 12 target
    ├── gnn_training_curves.png            # Loss & AUC per epoch
    ├── classical_vs_gnn_v4.png            # Head-to-head comparison
    ├── shap_summary.png                   # SHAP beeswarm plot
    └── explainability_comparison.png      # SHAP vs attention comparison
```

> 📌 **Catatan:** Folder `saved_models/` tidak diupload karena ukuran file model bisa mencapai ratusan MB. Generate ulang dengan menjalankan notebook dari **B1 → B11** secara berurutan.

---

## ⚙️ Instalasi

### Prasyarat

- Python **3.10+**
- Git
- VSCode (rekomendasi) atau Jupyter Notebook/Lab
- RAM minimal **8 GB** (16 GB direkomendasikan untuk GNN training)

### Langkah 1 — Clone Repository

```bash
git clone https://github.com/[username]/drug-toxicity-predictor.git
cd drug-toxicity-predictor
```

### Langkah 2 — Buat Virtual Environment

```bash
python -m venv tox_env

# Aktivasi — Windows
tox_env\Scripts\activate

# Aktivasi — Mac / Linux
source tox_env/bin/activate
```

### Langkah 3 — Install Dependensi Utama

```bash
pip install -r requirements.txt
```

### Langkah 4 — Install PyTorch + PyTorch Geometric

> PyTorch dan PyG perlu diinstall terpisah karena bergantung pada versi CUDA.

**CPU only (Windows / Linux):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

**GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
```

**Mac (Apple Silicon / MPS):**
```bash
pip install torch torchvision torchaudio
pip install torch_geometric
```

### Langkah 5 — Register Jupyter Kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name=tox_env --display-name "Python (tox_env)"
```

### ✅ Verifikasi Instalasi

```bash
python -c "
import rdkit;              print(f'✅ rdkit          {rdkit.__version__}')
import deepchem;           print(f'✅ deepchem        {deepchem.__version__}')
import torch;              print(f'✅ torch           {torch.__version__}')
import torch_geometric;    print(f'✅ torch_geometric {torch_geometric.__version__}')
import xgboost;            print(f'✅ xgboost         {xgboost.__version__}')
import shap;               print(f'✅ shap            {shap.__version__}')
import sklearn;            print(f'✅ sklearn         {sklearn.__version__}')
import imblearn;           print(f'✅ imblearn        {imblearn.__version__}')
import ipywidgets;         print(f'✅ ipywidgets      {ipywidgets.__version__}')
print()
print('🎉 Semua library berhasil diinstall!')
"
```

---

## 🚀 Cara Menjalankan

### ⭐ Opsi A — Web App via Voilà (Rekomendasi)

Voilà mengubah notebook menjadi web app interaktif — kode Python tersembunyi, yang terlihat hanya UI-nya saja.

```bash
# Install Voilà
pip install voila

# Jalankan
voila Drug_Toxicity_Predictor_v4.ipynb --theme=light
```

Browser otomatis terbuka di **`http://localhost:8866`**

Untuk akses dari perangkat lain dalam satu jaringan (misal: buka dari HP):
```bash
voila Drug_Toxicity_Predictor_v4.ipynb --port=8866 --no-browser
# Buka di HP: http://[IP_komputer_kamu]:8866
```

### Opsi B — VSCode + Jupyter Extension

```bash
code .    # Buka VSCode di folder project
```

1. Buka file `Drug_Toxicity_Predictor_v4.ipynb`
2. Klik **Select Kernel** (pojok kanan atas) → pilih **Python (tox_env)**
3. Klik **Run All** atau jalankan cell satu per satu dari atas

### Opsi C — Jupyter Notebook / JupyterLab

```bash
# Jupyter Notebook klasik
jupyter notebook Drug_Toxicity_Predictor_v4.ipynb

# JupyterLab (lebih modern)
jupyter lab Drug_Toxicity_Predictor_v4.ipynb
```

---

### 📋 Urutan Eksekusi Cell

| Cell | Nama | Deskripsi | ⏱ Estimasi |
|:----:|:-----|:----------|:-----------:|
| **B1** | Install & Import | Install library (hanya sekali) + import semua modul | ~5 menit |
| **B2** | Load Dataset | Download Tox21 dari AWS S3 (~2 MB), eksplorasi awal | < 1 menit |
| **B3** | EDA | Visualisasi distribusi label, missing values, imbalance ratio | < 1 menit |
| **B4** | Feature Engineering | Hitung Morgan FP + RDKit descriptors, seleksi korelasi | < 2 menit |
| **B5** | SMOTE + CV | 5-fold StratifiedKFold + SMOTE per target | ~3 menit |
| **B6** | Training Final | Training XGBoost final model + Random Forest baseline | ~2 menit |
| **B7** | Multi-Task XGBoost | Training 12 target sekaligus, simpan model + threshold | ~5 menit |
| **B8** | GNN Training | Training AttentiveFP 30 epoch, simpan checkpoint | ~15 menit |
| **B9** | GNN CV | 5-fold CV untuk GNN, evaluasi per target | ~20 menit |
| **B10** | Perbandingan | Visualisasi head-to-head Classical ML vs GNN | < 1 menit |
| **B11** | SHAP | Hitung SHAP values, generate explainability plots | ~2 menit |
| **B12** | Widget UI | Render dashboard interaktif — **sistem siap dipakai** ✅ | < 1 menit |

> ⚡ **Resume Session (model sudah ada):** Skip B1 Install, B5–B9.
> Cukup jalankan: **B1 Import → B2 → B4 → B7 (load model) → B11 → B12**

---

### 🧪 Cara Menggunakan Widget

Setelah B12 berhasil dijalankan:

1. **Masukkan SMILES** senyawa di kolom input, atau klik salah satu tombol contoh cepat
2. **Pilih Target Biologis** dari dropdown (12 target tersedia)
3. **Pilih Model Prediksi:** Classical ML (XGBoost) atau GNN (AttentiveFP)
4. Klik tombol **🔬 Analisis Toksisitas**
5. Hasil muncul otomatis di bawah, mencakup:
   - Prediksi toksik / tidak toksik + probabilitas
   - Meter confidence
   - Profil properti molekul (MW, LogP, HBD, HBA, TPSA, Lipinski)
   - SHAP feature importance chart
   - Atom highlight pada struktur 2D
   - Interpretasi hasil dalam bahasa awam

**Contoh senyawa untuk dicoba:**

| Senyawa | SMILES | Prediksi NR-AR | Keterangan |
|:--------|:-------|:--------------:|:-----------|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | ✅ Tidak Toksik | Analgesik umum |
| Caffeine | `Cn1cnc2c1c(=O)n(C)c(=O)n2C` | ✅ Tidak Toksik | Stimulan dalam kopi |
| Ibuprofen | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` | ✅ Tidak Toksik | NSAID anti-inflamasi |
| Benzene | `c1ccccc1` | ⚠️ Toksik | Karsinogen, pelarut industri |
| Testosterone | `CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C` | ⚠️ Toksik | Hormon androgen endogen |

---

### Perbandingan Classical ML vs GNN (Target: NR-AR)

| Model | CV AUC (5-Fold) | Test AUC | Waktu Training | Interpretability |
|:------|:---------------:|:--------:|:--------------:|:----------------:|
| XGBoost Multi-Task | ~0.74 ± 0.03 | ~0.75 | ~5 menit | ✅ SHAP |
| GNN AttentiveFP | ~0.72 ± 0.05 | ~0.73 | ~35 menit | ⚠️ Proxy |

**Kesimpulan:** XGBoost sedikit unggul pada dataset ini, lebih cepat, dan lebih mudah diinterpretasi. GNN menunjukkan potensi lebih besar pada dataset yang lebih besar dan senyawa dengan struktur kompleks.

---

## 🛠️ Teknologi

| Kategori | Library | Versi | Kegunaan |
|:---------|:--------|:-----:|:---------|
| Cheminformatics | RDKit | ≥ 2023.09 | Parsing SMILES, featurisasi, visualisasi molekul |
| ML Framework | scikit-learn | ≥ 1.3 | Pipeline, CV, metrik evaluasi |
| Gradient Boosting | XGBoost | ≥ 2.0 | Classifier utama multi-task |
| GNN | PyTorch Geometric | ≥ 2.4 | AttentiveFP, graph operations |
| Deep Learning | PyTorch | ≥ 2.0 | Backend GNN, training loop |
| Molecular ML | DeepChem | ≥ 2.7 | MolGraph featurizer untuk GNN |
| Imbalanced Data | imbalanced-learn | ≥ 0.11 | SMOTE oversampling |
| Explainability | SHAP | ≥ 0.44 | Feature importance, beeswarm plot |
| UI Widget | ipywidgets | ≥ 8.0 | Komponen UI interaktif |
| Web App | Voilà | ≥ 0.5 | Deploy notebook sebagai web app |
| Data Processing | pandas, numpy | Latest | Manipulasi data tabular |
| Visualisasi | matplotlib, seaborn | Latest | Plot evaluasi dan EDA |

---

## ⚠️ Limitasi

1. **Validasi eksternal** — Model belum diuji pada dataset di luar Tox21 (misalnya ChemBL, SIDER). Performa pada distribusi senyawa yang berbeda belum diketahui.
2. **GNN Atom Highlight** — Visualisasi highlight atom menggunakan pendekatan proxy berbasis Morgan fingerprint + SHAP, bukan attention weight GNN yang sesungguhnya.
3. **Hyperparameter** — Belum dilakukan systematic tuning menggunakan Optuna atau GridSearchCV. Parameter saat ini menggunakan nilai default yang reasonable.
4. **Epoch GNN** — 30 epoch training mungkin belum mencapai konvergensi optimal untuk semua target, terutama target dengan imbalance ekstrem.
5. **Skala Dataset** — ~8.000 senyawa relatif kecil untuk GNN. Performa GNN umumnya meningkat signifikan pada dataset 100k+ senyawa.
6. **Validasi Basah** — Seluruh prediksi bersifat komputasional dan belum divalidasi melalui uji eksperimental di laboratorium.

---

## 📚 Referensi

1. **Tox21 Dataset** — Tox21 Data Challenge 2014. NIH National Center for Advancing Translational Sciences. https://tripod.nih.gov/tox21/

2. **AttentiveFP** — Xiong, Z., et al. (2020). *Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism.* Journal of Medicinal Chemistry, 63(16), 8749–8760.

3. **SHAP** — Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS), 30.

4. **RDKit** — Landrum, G. RDKit: Open-source cheminformatics software. https://www.rdkit.org

5. **DeepChem** — Ramsundar, B., et al. (2019). *Deep Learning for the Life Sciences.* O'Reilly Media.

6. **SMOTE** — Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence Research, 16, 321–357.

7. **XGBoost** — Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.

8. **PyTorch Geometric** — Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop on Representation Learning on Graphs and Manifolds.

---

## 👤 Author

| Field | Info |
|:------|:-----|
| **Nama** | Nabila Hermawan |
| **Tahun** | 2026 |

---

## 📄 Lisensi

Project ini dibuat untuk keperluan akademik (Final Project).

Dataset Tox21 merupakan milik NIH/EPA/FDA dan digunakan sesuai ketentuan penggunaan data publik. Model dan kode dalam repository ini dilisensikan di bawah [MIT License](LICENSE).

---

<div align="center">

⚗️ **Drug Toxicity Predictor v4.0**

*Machine Learning*

*Dibangun dengan Python · RDKit · PyTorch · XGBoost · SHAP*

</div>
