# scripts/preprocessing.py
# -----------------------------
# Professional preprocessing module for renal litiasis project

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------
# 1. Split features target
# ---------------------------------------
def split_features_target(df, target="stone_risk", leakage_cols=None):
    """
    Splits dataframe into X and y, removing leakage columns.
    """
    leakage_cols = leakage_cols or []
    y = df[target].astype(int)
    X = df.drop(columns=[target] + leakage_cols)
    return X, y

# ---------------------------------------
# 2. Get feature types
# ---------------------------------------
def get_feature_types(X):
    """
    Returns numerical and categorical feature lists.
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols

# ---------------------------------------
# 3. Build preprocessor
# ---------------------------------------
def build_preprocessor(num_cols, cat_cols, strategy="median"):
    """
    Builds a preprocessing ColumnTransformer.
    strategy = 'median' or 'knn'
    """
    if strategy == "median":
        num_tr = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=False))
        ])
    elif strategy == "knn":
        num_tr = Pipeline([
            ("imp", KNNImputer(n_neighbors=5, weights="distance")),
            ("sc", StandardScaler(with_mean=False))
        ])
    else:
        raise ValueError("Unknown strategy")

    cat_tr = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_tr, num_cols),
        ("cat", cat_tr, cat_cols)
    ])
