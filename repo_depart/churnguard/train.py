import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnguard.data import get_feature_columns, load_data, preprocess
from churnguard.evaluate import compute_metrics

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
        raise ValueError("model_name doit être parmi : lr, rf, gb")

    preprocessor = build_preprocessor(X)

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", clf),
    ])

    pipeline.fit(X, y)

    return pipeline


def run_training_cli(model_name: str, register: bool, data_path: str) -> None:
    """Exécute un entraînement simple, avec enregistrement MLflow optionnel."""
    df = load_data(data_path)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train, model_name=model_name, params={})
    metrics = compute_metrics(model, X_test, y_test)
    print("Métriques:", metrics)

    if not register:
        return

    mlflow.set_experiment("churnguard")
    with mlflow.start_run(run_name=f"cli-{model_name}") as run:
        mlflow.log_param("model", model_name)
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train.iloc[:5],
        )

        registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", "churnguard")
        MlflowClient().transition_model_version_stage(
            name="churnguard",
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Modèle enregistré en Production (version {registered.version})")


def main() -> None:
    """Point d'entrée CLI de l'entraînement."""
    parser = argparse.ArgumentParser(description="Train ChurnGuard model")
    parser.add_argument("--model", choices=["lr", "rf", "gb"], default="rf")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--data", default="data/telco_churn.csv")
    args = parser.parse_args()

    run_training_cli(model_name=args.model, register=args.register, data_path=args.data)


if __name__ == "__main__":
    main()