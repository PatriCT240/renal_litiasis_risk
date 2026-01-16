# Clinical Summary — Kidney Stone Risk Prediction

## 1. Clinical Motivation
Kidney stone disease is a highly prevalent and recurrent condition, often associated with severe pain, emergency visits, and long‑term renal complications.
Early identification of individuals at elevated risk enables targeted preventive strategies, including hydration optimization, dietary counseling, and metabolic evaluation.

This project develops a machine learning model that estimates the probability of kidney stone formation using routinely collected clinical, biochemical, and lifestyle variables.
The goal is to support clinicians in risk stratification and preventive decision‑making, not to replace diagnostic imaging.

---

## 2. Key Predictors Identified by the Model

Model interpretability analyses (permutation importance + PDPs) highlight several clinically meaningful predictors:

# Top predictors (Permutation Importance, PR‑AUC–based)

- **Oxalate levels** — strongest metabolic driver of calcium oxalate stone formation  
- **Urine pH** — acidic urine markedly increases stone risk
- **Serum calcium** — associated with calcium‑based stone supersaturation
- **Renal function markers (GFR, creatinine)** — reflect underlying kidney physiology  
- **Water intake** — protective factor, consistent with hydration guidelines

These findings align with nephrology literature and known pathophysiological mechanisms.

---

## 3. Model Performance

### 3.1 Cross‑Validation (5‑fold, stratified)

| Model | CV ROC‑AUC (mean ± SD) | CV PR‑AUC (mean ± SD) |
|-------|-------------------------|-------------------------|
| **RF + Median Imputation** | **0.998 ± 0.001** | **0.996 ± 0.002** |
| **RF + KNN Imputation** | **0.996 ± 0.001** | **0.993 ± 0.003** |

### 3.2 Test Set Performance
- **ROC‑AUC:** 0.998  
- **PR‑AUC:** 0.997
- **Calibration curve:**  
Calibration performance was evaluated using a reliability diagram.  
The isotonic‑calibrated model exhibited excellent alignment between predicted and observed probabilities, with the calibration curve closely following the identity line.  
The figure is available at: `reports/figures/calibration_curve_rf.png`.  
- **Brier Score:** improved from **0.024 → 0.018** after isotonic calibration.

These values indicate **near‑perfect discrimination** and **excellent calibration**, especially important in clinical risk prediction.

---

## 4. Robustness Analysis

To evaluate stability under real‑world conditions:

### 4.1 Injected perturbations
- **10% missingness** in GFR, urine pH, water intake  
- **Gaussian noise (5% SD)** in blood pressure and serum calcium  

### 4.2 Results under perturbations

| Model | Test ROC‑AUC | Test PR‑AUC |
|-------|--------------|--------------|
| **RF + Median** | **0.998** | **0.997** |
| **RF + KNN** | **0.997** | **0.995** |

The model remained **highly stable**, demonstrating robustness to missing data and measurement variability.

---

## 5. Interpretability

Model interpretability was assessed using permutation importance, partial dependence plots (PDPs), and calibration analysis.  
These tools provide transparency and clinical insight into how the model generates predictions.

---

### 5.1 Permutation Importance (numeric results)

Permutation importance quantifies how much each feature contributes to the model’s PR‑AUC.  
The following table reports the top predictors ranked by mean importance:

| Feature        | Importance (mean) |
|----------------|-------------------|
| Oxalate        | 0.0142            |
| Urine pH       | 0.0118            |
| Serum calcium  | 0.0097            |
| GFR            | 0.0089            |
| Water intake   | 0.0074            |
| Creatinine     | 0.0068            |
| Sodium intake  | 0.0059            |

**Clinical interpretation:**
- **Oxalate** is the strongest metabolic driver of calcium oxalate stone formation.  
- **Urine pH** strongly modulates supersaturation risk, especially when < 5.5.  
- **Serum calcium** and **GFR/creatinine** reflect renal physiology and calcium handling.  
- **Water intake** acts as a protective factor by reducing urinary concentration.  

The full ranked list is shown in the permutation importance plot:  
`reports/figures/permutation_importance_rf_top15.png`

---

### 5.2 Partial Dependence Plots (PDPs)

PDPs illustrate how changes in individual predictors influence the predicted probability of kidney stone formation.

Key clinically intuitive patterns:

- **Urine pH:**  
  Risk increases sharply when urine pH falls below **5.5**, consistent with acidic environments promoting uric acid and calcium oxalate supersaturation.

- **Oxalate:**  
  Shows an **exponential risk increase**, reflecting its central role in calcium oxalate stone formation.

- **Water intake:**  
  Displays a **monotonically decreasing risk curve**, supporting hydration as a cornerstone of stone prevention.

PDP figures are available at:  
`reports/figures/pdp_*.png`

---

### 5.3 Calibration Curve

The calibration curve evaluates how well predicted probabilities match observed outcomes.

The isotonic‑calibrated model shows excellent agreement with the 45° reference line, indicating reliable probability estimates.

Calibration plot:  
`reports/figures/calibration_curve.png`

---

## 6. Clinical Interpretation

This model is designed as a **clinical decision‑support tool**, not a diagnostic test.  
Potential applications include:

- **Flagging high‑risk patients** for metabolic evaluation  
- **Guiding preventive counseling** (hydration, diet, metabolic workup)  
- **Supporting longitudinal monitoring** in recurrent stone formers  
- **Prioritizing follow‑up** in resource‑limited settings  

The model’s interpretability and robustness make it suitable for integration into clinical workflows.

---

## 7. Limitations

- Requires **external validation** in multi‑center cohorts  
- Dataset lacks imaging, genetic markers, and 24‑hour urine profiles  
- Performance may vary across demographic or metabolic subgroups  
- Not designed to replace CT/ultrasound for diagnostic confirmation  

---

## 8. Conclusion

This project delivers a **high‑performing, interpretable, and robust** machine learning model for predicting kidney stone risk using routine clinical data.  
The model demonstrates:

- **Outstanding discrimination (ROC‑AUC ~0.998)**  
- **Excellent precision in imbalanced settings (PR‑AUC ~0.997)**  
- **Strong calibration after isotonic regression**  
- **Stability under missingness and noise**  
- **Clinically meaningful predictors** consistent with nephrology literature  

With further external validation, this tool could support clinicians in **early identification, prevention, and personalized risk management** for kidney stone disease.
