# scripts/interpretability.py
# -----------------------------
# Interpretability analysis: Permutation Importance & PDP

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import average_precision_score, make_scorer

# ---------------------------------------
# 1. Permutation Importance (PR-AUC based)
# ---------------------------------------
def permutation_importance_pr(model, X_test, y_test, n_repeats=10, random_state=42):
    """
    Computes permutation importance using PR-AUC as the scoring metric.
    Returns a sorted DataFrame with mean and std importance.
    Compatible with scikit-learn 1.7.2.
    """
    pimp = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",   # built-in scorer (predict_proba aware)
        n_repeats=n_repeats,
        random_state=random_state
    )

    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": pimp.importances_mean,
        "importance_std": pimp.importances_std
    }).sort_values("importance_mean", ascending=False)

    return imp

# ---------------------------------------
# 2. PDP (Partial Dependence Plots)
# ---------------------------------------
def plot_pdp(model, X, features):
    """
    Plots PDP for a list of features.
    """
    for f in features:
        try:
            PartialDependenceDisplay.from_estimator(model, X, [f])
            plt.title(f"PDP — {f}")
        except Exception as e:
            print(f"PDP failed for {f}: {e}")
