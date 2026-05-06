import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from churnguard.data import get_feature_columns

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construit le ColumnTransformer pour le preprocessing."""
    num_cols, cat_cols = get_feature_columns(X)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    return preprocessor

def train_model(X: pd.DataFrame, y: pd.Series, model_name: str, params: dict) -> Pipeline:
    """Entraîne un pipeline sklearn et le retourne fitté."""

    if model_name == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=42, **params)
    elif model_name == "rf":
        clf = RandomForestClassifier(random_state=42, **params)
    elif model_name == "gb":
        clf = GradientBoostingClassifier(random_state=42, **params)
    else:
        raise ValueError(f"model_name doit être parmi : lr, rf, gb")

    preprocessor = build_preprocessor(X)

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", clf),
    ])

    pipeline.fit(X, y)

    return pipeline