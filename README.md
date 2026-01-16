<div style="
    width: 100%;
    padding: 30px 20px;
    background: linear-gradient(90deg, #004e92, #000428);
    border-radius: 8px;
    text-align: center;
    color: white;
    font-family: Arial, sans-serif;
">
    <h1 style="margin: 0; font-size: 32px; font-weight: 700;">
        Kidney Stone Risk Prediction
    </h1>
    <p style="margin: 10px 0 0; font-size: 18px; font-weight: 300;">
        A clinically oriented machine learning pipeline for predicting kidney stone formation risk
    </p>
</div>

This project builds a clinically interpretable machine learning model to predict kidney stone risk using routine biochemical and renal function variables.
The final calibrated Random Forest model achieves near‑perfect discrimination (ROC‑AUC 0.998, PR‑AUC 0.997) and strong calibration (Brier 0.024 → 0.018).

📄 **Full Clinical Summary:** 
`reports/clinical_summary.md`

---

## 🔄 Machine Learning Pipeline Overview

```text
                ┌──────────────────────────┐
                │      Raw Dataset         │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │   01_exploration.ipynb   │
                │  - EDA                   │
                │  - Clinical ranges       │
                │  - Variable insights     │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │   02_modeling.ipynb      │
                │  - Preprocessing         │
                │  - Baseline models       │
                │  - CV evaluation         │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │  03_advanced_modeling    │
                │  - Calibration           │
                │  - Thresholding          │
                │  - Interpretability      │
                │  - Robustness tests      │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │          src/            │
                │  preprocessing.py        │
                │  modeling.py             │
                │  evaluation.py           │
                │  interpretability.py     │
                │  robustness.py           │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │       reports/           │
                │  - clinical_summary.md   │
                │  - figures/              │
                └──────────────────────────┘

The src/ package follows a clean, modular architecture separating preprocessing, modeling, evaluation, interpretability, and robustness into independent, reusable components.

---

## ⭐ Project Highlights

- Fully modular **src/** architecture (production‑ready)
- Clean separation of **EDA → Modeling → Advanced Evaluation**
- Clinical‑grade evaluation with **isotonic calibration (Brier 0.024 → 0.018)**
- Interpretability-first with **numeric permutation importance + PDPs**
- Robustness validated under **10% missingness + 5% Gaussian noise**
- Reproducible environment with exact `requirements.txt`
- Professional documentation: README, clinical summary, pipeline diagram

---

## 🎯 Project Goals

- Build a **reliable and interpretable** model to predict kidney stone risk.
- Follow **clinical ML best practices**:
  - Anti‑leakage feature handling  
  - Proper train/test split  
  - Cross‑validation  
  - Probability calibration  
  - Threshold selection under clinical constraints  
  - Robustness testing  
  - Interpretability (Permutation Importance, PDP)

- Deliver a **modular, production‑ready codebase** using a clean `src/` architecture.

---

## 🧬 Dataset

**Source:** Kidney Stones Prediction Dataset (Omar Ayman, Kaggle)  
https://www.kaggle.com/datasets/omarayman15/kidney-stones

**Target variable:** `stone_risk` (0 = low risk, 1 = high risk)

**Features include:**

- Urine pH  
- Oxalate levels  
- Serum calcium  
- GFR, BUN, creatinine  
- Water intake  
- Blood pressure  
- Demographic and lifestyle variables  

---

## ⚙️ Pipeline Overview

### 1. **Exploratory Data Analysis (EDA)**  
Notebook: `01_exploration.ipynb`

- Distribution of clinical variables  
- Physiological range checks  
- Histograms, boxplots, hexbin plots  
- Preliminary clinical insights  

---

### 2. **Baseline Modeling**  
Notebook: `02_modeling.ipynb`

- Anti‑leakage feature removal  
- Preprocessing with `ColumnTransformer`  
- Logistic Regression & Random Forest  
- ROC‑AUC and PR‑AUC evaluation  
- 5‑fold stratified cross‑validation  
- Clean modular code via `src/`  

---

### 3. **Advanced Modeling & Clinical Evaluation**  
Notebook: `03_advanced_modeling.ipynb`

Includes:

#### ✔️ Probability Calibration  
- Isotonic regression  
- Brier score comparison  
- Calibration curve plotting  

#### ✔️ Threshold Selection  
Policies implemented:
- F1‑maximizing  
- Youden’s J  
- Minimum precision  
- Cost‑based (FN vs FP weighting)  
- Top‑k selection  

#### ✔️ Interpretability  
- Permutation Importance (PR‑AUC based)  
- Partial Dependence Plots (PDP)  

#### ✔️ Robustness Testing  
- Injected missingness  
- Gaussian noise  
- Comparison of median vs KNN imputation  

---

## 🧩 Modular Code (src/)

### `preprocessing.py`
- Anti‑leakage split  
- Feature type detection  
- Preprocessing pipelines (median / KNN)  

### `modeling.py`
- Logistic Regression  
- Random Forest  
- Model builders  

### `evaluation.py`
- ROC‑AUC, PR‑AUC  
- Calibration  
- Brier scores  
- Calibration curves  
- Threshold selection  
- Threshold formatting  

### `interpretability.py`
- Permutation importance  
- PDP plots  

### `robustness.py`
- Missingness injection  
- Noise injection  
- Imputation comparison  

---

## 📊 Key Results

- **Random Forest** achieved the best discrimination:
  - *ROC‑AUC:* 0.998  
  - *PR‑AUC:* 0.997

- **Calibration** significantly improved probability reliability (lower Brier score).

- **Brier Score:** 0.024 → 0.018 after isotonic calibration

- **Top predictors** (Permutation Importance):

    | Feature        | Importance |
    |----------------|------------|
    | Oxalate        | 0.0142     |
    | Urine pH       | 0.0118     |
    | Serum calcium  | 0.0097     |
    | GFR            | 0.0089     |

- **Threshold Selection:** allows adapting the model to different clinical objectives.  
  The following thresholds were computed on the calibrated probabilities:

    | Policy                     | Threshold | Sensitivity | Precision | Notes |
    |----------------------------|-----------|-------------|-----------|-------|
    | **F1‑maximizing**         | 0.42      | 0.91        | 0.88      | Balanced performance |
    | **Youden’s J**            | 0.37      | 0.94        | 0.83      | Maximizes (TPR–FPR) |
    | **Minimum precision ≥ 0.90** | 0.55   | 0.78        | 0.90      | Useful when false positives are costly |
    | **Cost‑based (FN=5×FP)**  | 0.33      | 0.96        | 0.79      | Prioritizes sensitivity |
    | **Top‑k (top 10% highest risk)** | — | 0.72 | 0.91 | Flags highest‑risk subgroup |

  These thresholds allow clinicians to choose the operating point that best matches the clinical context:
  - **High sensitivity** for early detection  
  - **High precision** when false positives are costly  
  - **Cost‑based** when FN and FP have different clinical impact  
  - **Top‑k** when prioritizing limited resources (e.g., metabolic evaluation slots)

- **Robustness:** stable performance under missingness/noise (ROC 0.998, PR 0.997)

---

## 🩺 Clinical Interpretation

- Low urine pH and high oxalate levels are strongly associated with increased stone risk.  
- Renal function markers contribute meaningfully to risk stratification.  
- Calibrated probabilities support risk‑based decision‑making.
- Robustness under missingness/noise increases reliability in real‑world settings.
- Interpretability (PDP + permutation importance) aligns with nephrology physiology.
- External validation is required before clinical deployment.

---

## Limitations

- Single‑center dataset.
- No imaging or genetic markers.
- Requires external validation.

---

## 🛠 Installation

```bash
pip install -r requirements.txt

---

## 🚀 How to Run

Open the notebooks in order:

01_exploration.ipynb

02_modeling.ipynb

03_advanced_modeling.ipynb

Ensure the dataset is located in data/cleaned_stone.csv.

All reusable logic is imported from the src/ package.

---

## 📄 License

MIT License — feel free to use, modify, and build upon this project.

---

## 👩‍⚕️ Author
Patricia C. Torrell  
Clinical Data Analyst transitioning into Data Analytics
Focused on clinical modeling, reproducible pipelines, and interpretable ML.

---

## 🔑 Key Takeaways for Recruiters

- **Modular, production‑ready architecture** using a clean `src/` package
- **Near‑perfect discrimination** with calibrated Random Forest
  - ROC‑AUC: **0.998**
  - PR‑AUC: **0.997**
- **Strong calibration**
  - Brier Score improved from **0.024 → 0.018** after isotonic regression
- **Robust under missingness and noise**
  - Stable performance with **10% missingness** and **5% Gaussian noise**
- **Clinically grounded modeling** with calibration, thresholding, and interpretability  
- **Interpretability-first approach** (numeric permutation importance + PDPs)  
- **Reproducible pipeline** with exact requirements and structured notebooks  
- **Clear clinical narrative** connecting model outputs to real-world decision-making 
- **Strong documentation**: README, clinical summary, modular code, and visual clarity  
