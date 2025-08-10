# core/forecast.py
import yfinance as yf
import pandas as pd
import numpy as np

# ---------- CSV CLEANER ----------
def clean_price_csv(
    file,
    date_cols=("Date", "date", "timestamp", "Datetime", "datetime"),
    close_cols=("Close", "Adj Close", "close", "adj_close", "Price", "price"),
) -> pd.DataFrame:
    df = pd.read_csv(file)
    date_col = next((c for c in date_cols if c in df.columns), None)
    close_col = next((c for c in close_cols if c in df.columns), None)
    if date_col is None or close_col is None:
        raise ValueError("CSV must contain a date column (e.g., Date) and a close/price column (e.g., Close).")

    out = df[[date_col, close_col]].copy()
    out.columns = ["Date", "Close"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = (
        out.dropna()
           .drop_duplicates(subset=["Date"])
           .sort_values("Date")
           .reset_index(drop=True)
    )
    if len(out) < 10:
        raise ValueError("CSV has fewer than 10 clean rows after parsing. Add more history.")
    return out

# ---------- YFINANCE HELPERS ----------
def _extract_close_from_download(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("No price data returned (empty frame).")

    if isinstance(df.columns, pd.MultiIndex):
        close = None
        for name in ("Adj Close", "Close"):
            if name in df.columns.get_level_values(0):
                sub = df[name]
                if isinstance(sub, pd.Series):
                    close = sub
                else:
                    close = sub[ticker] if ticker in sub.columns else sub.iloc[:, 0]
                break
        if close is None:
            raise ValueError("Could not find Close/Adj Close column in yfinance data.")
        out = close.rename("Close").to_frame()
        out.index.name = "Date"
        out = out.reset_index()[["Date", "Close"]]
        return out

    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            out = df[["Adj Close"]].rename(columns={"Adj Close": "Close"})
        else:
            raise ValueError("Close column not found in yfinance data.")
    else:
        out = df[["Close"]].copy()

    out = out.rename_axis("Date").reset_index()[["Date", "Close"]]
    return out

def fetch_prices(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=True, progress=False, threads=False
    )
    out = _extract_close_from_download(df, ticker)
    out["Date"]  = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    if len(out) < 10:
        raise ValueError("Not enough history to forecast (need ≥ 10 rows).")
    return out

# ---------- FORECAST (DRIFT MODEL) ----------
def forecast_prices(hist_df: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0.")
    prices = hist_df["Close"].astype(float).to_numpy()
    rets   = pd.Series(prices).pct_change().dropna().to_numpy()
    if rets.size == 0:
        raise ValueError("Return series is empty after cleaning.")

    avg_ret    = float(np.mean(rets))
    last_price = float(prices[-1])

    steps = int(horizon_days)
    future_dates   = pd.date_range(pd.to_datetime(hist_df["Date"].iloc[-1]) + pd.Timedelta(days=1),
                                   periods=steps, freq="D")
    growth_factors = (1.0 + avg_ret) ** np.arange(1, steps + 1, dtype=float)
    preds = last_price * growth_factors

    return pd.DataFrame({"Date": future_dates, "Predicted": preds})

def growth_pct(hist_df: pd.DataFrame, fore_df: pd.DataFrame) -> float:
    last_actual = float(hist_df["Close"].iloc[-1])
    last_pred   = float(fore_df["Predicted"].iloc[-1])
    return float((last_pred - last_actual) / last_actual * 100.0)
