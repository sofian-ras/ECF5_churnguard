import pandas as pd
import pytest
from churnguard.data import load_data, preprocess

@pytest.fixture
def sample_csv(tmp_path):
    """Crée un petit CSV de test qui ressemble au vrai dataset Telco."""
    data = {
        "customerID": ["1111-A", "2222-B", "3333-C", "4444-D"],
        "gender": ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [12, 1, 45, 3],
        "PhoneService": ["Yes", "No", "Yes", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No", "No internet service"],
        "DeviceProtection": ["No", "No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No", "No internet service"],
        "StreamingTV": ["No", "No", "No", "No internet service"],
        "StreamingMovies": ["No", "No", "No", "No internet service"],
        "Contract": ["Month-to-month", "Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check"] * 4,
        "MonthlyCharges": [29.85, 53.85, 42.30, 20.05],
        "TotalCharges": ["358.20", " ", "1905.50", "60.15"],
        "Churn": ["No", "Yes", "No", "Yes"],
    }
    path = tmp_path / "telco_churn.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)


# test 1
def test_load_data_returns_dataframe(sample_csv):
    """Vérifie que load_data retourne bien un DataFrame non vide."""
    df = load_data(sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


# test 2
def test_load_data_has_expected_columns(sample_csv):
    """Vérifie que les 21 colonnes Telco Churn sont présentes."""
    df = load_data(sample_csv)

    expected_columns = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
    ]

    assert len(df.columns) == 21
    for col in expected_columns:
        assert col in df.columns

# Si quelqu'un renomme une colonne ou supprime une colonne par erreur, ce test le détecte immédiatement.

# test 3
def test_preprocess_returns_features_and_target(sample_csv):
    """Vérifie que preprocess retourne bien X et y séparés correctement."""
    df = load_data(sample_csv)
    X, y = preprocess(df)

    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)

# test 4
def test_preprocess_handles_missing_total_charges(sample_csv):
    """Vérifie que les espaces vides dans TotalCharges sont gérés sans crash."""
    df = load_data(sample_csv)
    X, y = preprocess(df)

    assert X["TotalCharges"].dtype == float
    assert len(X) == 3  # la ligne avec " " dans TotalCharges doit être supprimée