# core/anomaly.py
import pandas as pd
import numpy as np

def anomaly_from_prices(hist_df: pd.DataFrame, mild=2.0, severe=3.0):
    """
    Detect anomalies via z-score on daily returns.
    Returns: (label, stats_dict)
    """
    prices = pd.to_numeric(hist_df["Close"], errors="coerce")
    rets = prices.pct_change().dropna()
    if len(rets) < 5:
        return "None", {"n": len(rets), "max_z": 0.0}

    z = (rets - rets.mean()) / (rets.std(ddof=1) + 1e-9)
    max_abs_z = float(np.abs(z).max())

    if max_abs_z >= severe:
        label = "Severe"
    elif max_abs_z >= mild:
        label = "Mild"
    else:
        label = "None"

    return label, {"n": int(len(rets)), "max_z": max_abs_z}
