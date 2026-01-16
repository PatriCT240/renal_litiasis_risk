# src/robustness.py
# ---------------------------------------------------------
# Full robustness pipeline: missingness injection, noise injection,
# validation, anti-leakage, split, imputers, CV, test evaluation.


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def run_robustness(df, target="stone_risk", random_state=42):
    # ---------------------------------------------------------
    # 1. Clone dataset
    # ---------------------------------------------------------
    df_adv = df.copy()
    rng = np.random.default_rng(random_state)

    # ---------------------------------------------------------
    # 2. Inject missingness
    # ---------------------------------------------------------
    missing_cols = [c for c in ["gfr", "urine_ph", "water_intake"] if c in df_adv.columns]

    for c in missing_cols:
        mask = rng.random(len(df_adv)) < 0.10
        df_adv.loc[mask, c] = np.nan

    # ---------------------------------------------------------
    # 3. Inject Gaussian noise
    # ---------------------------------------------------------
    def add_noise_col(s, frac_std=0.05):
        sigma = frac_std * float(np.nanstd(s.values))
        return s + rng.normal(0, sigma, size=len(s))

    noise_cols = [c for c in ["blood_pressure", "serum_calcium"] if c in df_adv.columns]

    for c in noise_cols:
        df_adv[c] = add_noise_col(df_adv[c], frac_std=0.05)

    # ---------------------------------------------------------
    # 4. Validation of perturbations
    # ---------------------------------------------------------
    print("Missingness rate after injection (~0.10 expected):")
    print(df_adv[missing_cols].isna().mean().round(3).to_dict())

    print("\nNoise injection check (std increase expected):")
    for c in noise_cols:
        base_std = float(df[c].std())
        adv_std = float(df_adv[c].std())
        print(f"{c}: base={base_std:.3f} -> adv={adv_std:.3f} (Δ={adv_std - base_std:.3f})")

    # ---------------------------------------------------------
    # 5. Anti-leakage and split
    # ---------------------------------------------------------
    leak_cols = [c for c in ["ckd_pred", "ckd_stage", "cluster", "months"] if c in df_adv.columns]

    y2 = df_adv[target].astype(int)
    X2 = df_adv.drop(columns=[target] + leak_cols)

    num2 = X2.select_dtypes(include="number").columns.tolist()
    cat2 = [c for c in X2.columns if c not in num2]

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, stratify=y2, random_state=random_state
    )

    print(f"\nX2 shape: {X2.shape} | positive rate={y2.mean():.3f}")
    print(f"Train/Test sizes: {X2_train.shape[0]}/{X2_test.shape[0]}")
    print(f"Numerical: {len(num2)} | Categorical: {len(cat2)} | Leakage removed: {leak_cols}")

    # ---------------------------------------------------------
    # 6. Two imputation pipelines
    # ---------------------------------------------------------
    prep_median = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=False)),
        ]), num2),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat2),
    ])

    prep_knn = ColumnTransformer([
        ("num", Pipeline([
            ("imp", KNNImputer(n_neighbors=5, weights="distance")),
            ("sc", StandardScaler(with_mean=False)),
        ]), num2),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat2),
    ])

    rf_median = Pipeline([
        ("prep", prep_median),
        ("model", RandomForestClassifier(n_estimators=300, random_state=random_state)),
    ])

    rf_knn = Pipeline([
        ("prep", prep_knn),
        ("model", RandomForestClassifier(n_estimators=300, random_state=random_state)),
    ])

    # ---------------------------------------------------------
    # 7. Cross-validation
    # ---------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    scoring_roc = "roc_auc"
    scoring_pr = "average_precision"

    # ---------------------------------------------------------
    # 8. Evaluate both pipelines
    # ---------------------------------------------------------
    results = {}

    for name, pipe in [("RF + Median", rf_median), ("RF + KNN", rf_knn)]:
        cv_roc = cross_val_score(pipe, X2_train, y2_train, scoring=scoring_roc, cv=cv, n_jobs=-1)
        cv_pr = cross_val_score(pipe, X2_train, y2_train, scoring=scoring_pr, cv=cv, n_jobs=-1)

        pipe.fit(X2_train, y2_train)
        p = pipe.predict_proba(X2_test)[:, 1]

        results[name] = {
            "cv_roc_mean": cv_roc.mean(),
            "cv_roc_std": cv_roc.std(),
            "cv_pr_mean": cv_pr.mean(),
            "cv_pr_std": cv_pr.std(),
            "test_roc": roc_auc_score(y2_test, p),
            "test_pr": average_precision_score(y2_test, p),
        }

    return results
