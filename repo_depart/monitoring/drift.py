"""Rapport de drift Evidently sur un échantillon de production simulé.

Usage:
    pip install evidently
    python monitoring/drift.py
    # => génère drift_report.html
"""

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from churnguard.data import load_data, preprocess


def run_drift_report(
    data_path: str = "data/telco_churn.csv",
    output_path: str = "drift_report.html",
    production_frac: float = 0.2,
) -> None:
    """Compare la distribution de référence vs un échantillon simulé de production."""
    df = load_data(data_path)
    X, _ = preprocess(df)

    # Simulation : 80% référence (train), 20% production
    reference = X.sample(frac=1 - production_frac, random_state=42)
    production = X.sample(frac=production_frac, random_state=0)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=production)
    report.save_html(output_path)
    print(f"Rapport de drift sauvegardé : {output_path}")


if __name__ == "__main__":
    run_drift_report()
