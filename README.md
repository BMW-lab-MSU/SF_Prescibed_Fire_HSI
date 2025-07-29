
# Attention and Edge-Aware Band Selection for Efficient Hyperspectral Classification of Burned Vegetation  

This repository provides the official implementation of the methods described in our MLSP 2025 paper:  
**"Attention and Edge-Aware Band Selection for Efficient Hyperspectral Classification of Burned Vegetation"**.  
We investigate two complementary hyperspectral band selection methods—**Spatial-Spectral Edge Preservation (SSEP)** and **Spectral-Redundancy Penalized Attention Ranking (SRPA)**—for classifying vegetation in post-burn environments using **VNIR hyperspectral imagery**.

---

## 🔬 Overview  
Effective post-burn vegetation classification is critical for ecological recovery and wildfire risk assessment.  
Hyperspectral imaging (HSI) offers rich spectral information but suffers from high dimensionality.  
This repository implements:
- **SSEP:** An unsupervised, edge-aware method aligning spectral and label-derived edges.
- **SRPA:** A supervised attention-based method with redundancy penalization for discriminative band selection.
  
Both methods are evaluated using:
- **Random Forest (RF):** Classical spectral-only classifier.
- **Lightweight 3D CNN:** Spatial-spectral deep learning model optimized for efficiency.

---

## 📂 Repository Structure  
```
SF_Prescribed_Fire_HSI/
│
├── data/                # VNIR hyperspectral cube (ENVI format) and label masks
├── preprocessing/        # Scripts for patch extraction, normalization, and labeling
├── band_selection/       # Implementations of SSEP and SRPA algorithms
├── models/               # 3D CNN architecture and training scripts
├── rf_baseline/          # Random Forest training pipeline
├── visualization/        # Band score plots and classification results
└── utils/                # Helper functions (e.g., Dice score, redundancy computation)
```

---

## 🛠 Installation  
1. Clone the repository:
   ```bash
   git clone https://github.com/BMW-lab-MSU/SF_Prescibed_Fire_HSI.git
   cd SF_Prescibed_Fire_HSI
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset  
The hyperspectral dataset was collected over a **thin-burn plot (330m × 300m)** in the **Lubrecht Experimental Forest (Montana, USA)** using a **Headwall VNIR sensor (273 bands, 400–1000 nm)**.  
- Labels include **Tree**, **Grass**, and **Soil**, manually annotated from RGB composites derived from hyperspectral bands.

> **Note:** Dataset access may require permission. Contact the maintainers for availability.

---

## 🚀 Usage  

### 1️⃣ Preprocessing & Patch Extraction  
```bash
python preprocessing/extract_patches.py --cube_path data/VNIR.hdr --labels data/labels.png --patch_size 50 --stride 25
```

### 2️⃣ Run Band Selection  
- **SSEP:**
  ```bash
  python band_selection/ssep.py --cube_path data/VNIR.hdr --labels data/labels.png --top_k 50
  ```
- **SRPA:**
  ```bash
  python band_selection/srpa.py --patch_dir data/patches --labels data/patch_labels.npy --top_k 50 --lambda_penalty 0.2
  ```

### 3️⃣ Train Models  
- **Random Forest (RF):**
  ```bash
  python rf_baseline/train_rf.py --bands_selected outputs/ssep_top50.npy
  ```
- **3D CNN:**
  ```bash
  python models/train_3dcnn.py --bands_selected outputs/srpa_top50.npy
  ```

---

## 📈 Results (Key Findings)  
- **SRPA** outperformed **SSEP** across all metrics:
  - **Best Accuracy:** 93.89% (SRPA + 3D CNN, Top-50 bands)  
  - **Best F1 Score:** 48.31% (SRPA + 3D CNN)  
- SSEP achieved competitive results in low-dimensional regimes (Top-10 bands).  
- SRPA’s attention-guided band selection better complements spatial-spectral deep learning models.

---

## 📊 Visualizations  
- Band importance plots (SSEP & SRPA)
- Classification accuracy vs. Top-k bands
- Spectral signature overlays for Tree and Grass  

---

## 🔗 Citation  
If you use this repository or its methods, please cite:
```
@inproceedings{mlsp2025_hsi_burn,
  title={Attention and Edge-Aware Band Selection for Efficient Hyperspectral Classification of Burned Vegetation},
  booktitle={IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2025},
  author={Mahmad Isaq Karankot}
}
```

---

## 🤝 Acknowledgements  
This work is supported by **NSF EPSCoR (OIA-2242802)** as part of the **SMART FIRES project** at Montana State University.  
