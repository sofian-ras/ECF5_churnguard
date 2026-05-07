import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churnguard.data import load_data, preprocess
from churnguard.evaluate import compute_metrics
from churnguard.train import run_training_cli, train_model

@pytest.fixture
def X_y():
    """Charge et préprocesse le vrai dataset pour les tests d'entraînement."""
    df = load_data("data/telco_churn.csv")
    X, y = preprocess(df)
    return X, y

def test_train_model_returns_fitted_pipeline(X_y):
    """Vérifie que train_model retourne un pipeline capable de prédire."""
    X, y = X_y

    model = train_model(X, y, model_name="rf", params={"n_estimators": 10})

    assert isinstance(model, Pipeline)
    assert len(model.predict(X)) == len(y)


# On vérifie que compute_metrics retourne exactement les 5 métriques
def test_compute_metrics_returns_expected_keys(X_y):
    """Vérifie que compute_metrics retourne bien les 5 métriques attendues."""
    X, y = X_y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train, model_name="lr", params={})
    metrics = compute_metrics(model, X_test, y_test)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}

    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0


def test_train_model_invalid_name_raises_value_error(X_y):
    """Vérifie qu'un nom de modèle invalide lève une ValueError."""
    X, y = X_y

    with pytest.raises(ValueError):
        train_model(X, y, model_name="invalid", params={})


def test_run_training_cli_without_register(monkeypatch):
    """Vérifie le chemin CLI sans enregistrement MLflow."""
    sample_df = pd.DataFrame({"a": [1, 2, 3]})
    X = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6], "f2": [0, 1, 0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1, 0, 1])

    calls: dict[str, bool] = {"train": False, "metrics": False}

    def fake_load_data(path: str) -> pd.DataFrame:
        return sample_df

    def fake_preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return X, y

    def fake_train_model(X_train, y_train, model_name: str, params: dict):
        calls["train"] = True
        return object()

    def fake_compute_metrics(model, X_test, y_test) -> dict:
        calls["metrics"] = True
        return {
            "accuracy": 0.8,
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
            "roc_auc": 0.8,
        }

    monkeypatch.setattr("churnguard.train.load_data", fake_load_data)
    monkeypatch.setattr("churnguard.train.preprocess", fake_preprocess)
    monkeypatch.setattr("churnguard.train.train_model", fake_train_model)
    monkeypatch.setattr("churnguard.train.compute_metrics", fake_compute_metrics)

    run_training_cli(model_name="rf", register=False, data_path="dummy.csv")

    assert calls["train"]
    assert calls["metrics"]