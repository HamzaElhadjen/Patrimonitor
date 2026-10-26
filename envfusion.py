"""Fusion expérimentale de mesures temporelles et données environnementales."""
from __future__ import annotations
import csv
from datetime import datetime
import numpy as np


def load_environment_csv(path, date_column="date", temperature_column="temperature",
                         humidity_column="humidity"):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                dt = datetime.fromisoformat(row[date_column].strip())
                rows.append({"date": dt, "temperature": float(row[temperature_column]),
                             "humidity": float(row[humidity_column])})
            except (KeyError, ValueError, TypeError):
                continue
    if not rows:
        raise ValueError("Aucune ligne valide. Colonnes attendues : date, temperature, humidity.")
    return rows


def fit_temperature_trend(dates, openings, temperatures):
    """Régression descriptive ouverture ~ température + temps; requiert assez d'observations."""
    y = np.asarray(openings, float)
    t = np.asarray(temperatures, float)
    if len(y) != len(t) or len(y) < 4:
        raise ValueError("Au moins quatre observations appariées sont nécessaires.")
    x_time = np.arange(len(y), dtype=float)
    X = np.column_stack([np.ones(len(y)), t, x_time])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    residual = y - pred
    return {"intercept": float(coef[0]), "temperature_coefficient": float(coef[1]),
            "time_trend_per_observation": float(coef[2]),
            "rmse": float(np.sqrt(np.mean(residual**2))), "n": int(len(y)),
            "causality_warning": "Association descriptive, pas une preuve de causalité."}
