# scripts/evaluation.py
# -----------------------------
# Evaluating and visualization utilities for renal litiasis project

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# ---------------------------------------
# 1. Evaluate models
# ---------------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Returns ROC-AUC and PR-AUC for a fitted model.
    """
    p = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, p),
        "pr_auc": average_precision_score(y_test, p)
    }

# ---------------------------------------
# 2. Plots ROC and PR
# ---------------------------------------
def plot_roc_pr(p, y_true, title):
    """
    Plots ROC and PR curves.
    """
    fpr, tpr, _ = roc_curve(y_true, p)
    prec, rec, _ = precision_recall_curve(y_true, p)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.title(f"ROC — {title}")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")

    plt.figure()
    plt.plot(rec, prec)
    plt.title(f"PR — {title}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

# ---------------------------------------
# 3. Calibration
# ---------------------------------------
def calibrate_model(model, X_train, y_train, method="isotonic", cv=5):
    """
    Returns a calibrated version of the model using isotonic or sigmoid calibration.
    """
    cal = CalibratedClassifierCV(model, method=method, cv=cv)
    cal.fit(X_train, y_train)
    return cal

# ---------------------------------------
# 4. Brier score
# ---------------------------------------
def brier_scores(model, calibrated_model, X_test, y_test):
    """
    Computes Brier scores for uncalibrated and calibrated models.
    """
    p_uncal = model.predict_proba(X_test)[:, 1]
    p_cal = calibrated_model.predict_proba(X_test)[:, 1]

    return {
        "uncalibrated": brier_score_loss(y_test, p_uncal),
        "calibrated": brier_score_loss(y_test, p_cal)
    }

# ---------------------------------------
# 5. Calibration curve
# ---------------------------------------
def calibration_curve_plot(model, calibrated_model, X_test, y_test, bins=10):
    """
    Plots calibration curves for uncalibrated and calibrated models.
    """
    p_uncal = model.predict_proba(X_test)[:, 1]
    p_cal = calibrated_model.predict_proba(X_test)[:, 1]

    pt_u, pp_u = calibration_curve(y_test, p_uncal, n_bins=bins, strategy="quantile")
    pt_c, pp_c = calibration_curve(y_test, p_cal,   n_bins=bins, strategy="quantile")

    plt.figure()
    plt.plot(pp_u, pt_u, marker='o', label='Uncalibrated')
    plt.plot(pp_c, pt_c, marker='o', label='Calibrated')
    plt.plot([0,1],[0,1],'--', lw=1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("Calibration Curve Comparison")
    plt.legend()

# ---------------------------------------
# 6. Threshold selection
# ---------------------------------------
def choose_threshold(p, y, policy="f1_max", min_precision=0.80, cost_fn=5.0, cost_fp=1.0, k=None):
    """
    Selects an optimal threshold under different policies:
    - f1_max
    - youden
    - min_prec
    - cost
    - topk
    """
    thr = np.linspace(0.0, 1.0, 201)
    best = None

    for t in thr:
        yhat = (p >= t).astype(int)
        tp = ((yhat==1)&(y==1)).sum()
        fp = ((yhat==1)&(y==0)).sum()
        fn = ((yhat==0)&(y==1)).sum()
        tn = ((yhat==0)&(y==0)).sum()

        prec = tp/(tp+fp+1e-9)
        rec  = tp/(tp+fn+1e-9)
        spec = tn/(tn+fp+1e-9)
        f1   = 2*prec*rec/(prec+rec+1e-9)
        youden = rec + spec - 1
        cost = cost_fn*fn + cost_fp*fp

        if policy == "f1_max":
            score = f1
        elif policy == "youden":
            score = youden
        elif policy == "min_prec":
            score = rec if prec >= min_precision else -1
        elif policy == "cost":
            score = -cost
        elif policy == "topk":
            assert k is not None
            t = np.partition(p, -k)[-k]
            return {
                "policy": "topk",
                "threshold": float(t),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            }

        if (best is None) or (score > best["score"]):
            best = {
                "policy": policy,
                "threshold": float(t),
                "precision": float(prec),
                "recall": float(rec),
                "specificity": float(spec),
                "f1": float(f1),
                "youden": float(youden),
                "cost": float(cost),
                "score": float(score)
            }

    return best

# ---------------------------------------
# 7. Threshold results
# ---------------------------------------
def format_threshold_results(*results):
    """
    Pretty-print threshold selection results.
    """
    for r in results:
        print(f"{r['policy']} → t={r['threshold']:.3f} | P={r['precision']:.3f} | R={r['recall']:.3f} | F1={r['f1']:.3f}")
