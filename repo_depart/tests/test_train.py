import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churnguard.data import load_data, preprocess
from churnguard.evaluate import compute_metrics
from churnguard.train import train_model

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