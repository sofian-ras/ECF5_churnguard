import pandas as pd

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV brut et retourne un DataFrame."""
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Préprocesse le DataFrame et retourne (X, y) séparés."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns=["customerID"])

    y: pd.Series = (df["Churn"] == "Yes").astype(int)
    X: pd.DataFrame = df.drop(columns=["Churn"])

    return X, y

def get_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retourne (num_cols, cat_cols) — séparation pour le ColumnTransformer."""
    num_cols = [c for c in NUM_COLS if c in X.columns]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols