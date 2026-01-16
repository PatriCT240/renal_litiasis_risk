# scripts/modeling.py
# -------------------
# Modeling utilities for renal litiasis project

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------
# 1. Logistic regression
# ---------------------------------------
def build_logistic_regression(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            tol=1e-4
        ))
    ])

# ---------------------------------------
# 2. Random forest 
# ---------------------------------------
def build_random_forest(preprocessor, n_estimators=300, random_state=42):
    return Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        ))
    ])
