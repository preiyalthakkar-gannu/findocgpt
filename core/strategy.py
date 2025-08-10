# core/strategy.py
def investment_strategy(forecast_change: float, sentiment_score: float, anomaly_level: str):
    """
    forecast_change: % change from forecast (float)
    sentiment_score: VADER compound (-1..1)
    anomaly_level: "None" | "Mild" | "Severe"
    """
    reasons = []

    # Sentiment
    if sentiment_score > 0.2:
        reasons.append("Positive sentiment")
    elif sentiment_score < -0.2:
        reasons.append("Negative sentiment")
    else:
        reasons.append("Neutral sentiment")

    # Anomaly
    if anomaly_level == "Severe":
        reasons.append("Severe anomalies detected")
    elif anomaly_level == "Mild":
        reasons.append("Mild anomalies detected")
    else:
        reasons.append("No anomalies detected")

    # Decision rules
    if forecast_change > 2 and sentiment_score > 0.1 and anomaly_level == "None":
        decision = "BUY"
    elif forecast_change < -2 or sentiment_score < -0.1 or anomaly_level == "Severe":
        decision = "SELL"
    else:
        decision = "HOLD"

    reason_text = "; ".join(reasons)
    return decision, reason_text
