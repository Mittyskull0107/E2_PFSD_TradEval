import numpy as np
import pandas as pd
from .market_data import fetch_data

SUPPORTED_STRATEGIES = ["moving_average", "rsi", "momentum"]

def run_backtest(symbol: str, strategy: str) -> dict:

    if not symbol:
        return {"error": "Symbol is required"}

    strategy = strategy.lower().strip()

    if strategy not in SUPPORTED_STRATEGIES:
        return {
            "error": f"Unknown strategy '{strategy}'",
            "supported": SUPPORTED_STRATEGIES
        }

    df = fetch_data(symbol)

    if df is None or df.empty:
        return {"error": f"No data found for symbol: {symbol}"}

    # ── run the right strategy ────────────────────────────────────
    if strategy == "moving_average":
        signals = _moving_average_signals(df)
    elif strategy == "rsi":
        signals = _rsi_signals(df)
    elif strategy == "momentum":
        signals = _momentum_signals(df)

    return _compute_metrics(symbol, strategy, df, signals)


# ─────────────────────────────────────────────────────────────────
# STRATEGY SIGNAL GENERATORS
# Each returns a Series of 1 (buy), -1 (sell), 0 (hold)
# ─────────────────────────────────────────────────────────────────

def _moving_average_signals(df: pd.DataFrame) -> pd.Series:
    """Buy when 20-day MA crosses above 50-day MA, sell when it crosses below."""
    df = df.copy()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()

    signals = pd.Series(0, index=df.index)
    signals[df["ma20"] > df["ma50"]] = 1   # buy signal
    signals[df["ma20"] < df["ma50"]] = -1  # sell signal
    return signals


def _rsi_signals(df: pd.DataFrame) -> pd.Series:
    """Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)."""
    df    = df.copy()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))

    signals = pd.Series(0, index=df.index)
    signals[rsi < 30] = 1   # oversold → buy
    signals[rsi > 70] = -1  # overbought → sell
    return signals


def _momentum_signals(df: pd.DataFrame) -> pd.Series:
    """Buy if price is higher than 20 days ago, sell if lower."""
    df = df.copy()
    df["momentum"] = df["Close"] - df["Close"].shift(20)

    signals = pd.Series(0, index=df.index)
    signals[df["momentum"] > 0] = 1
    signals[df["momentum"] < 0] = -1
    return signals


# ─────────────────────────────────────────────────────────────────
# METRICS ENGINE
# ─────────────────────────────────────────────────────────────────

def _compute_metrics(
    symbol: str,
    strategy: str,
    df: pd.DataFrame,
    signals: pd.Series
) -> dict:

    df           = df.copy()
    df["signal"] = signals
    df["return"] = df["Close"].pct_change()

    # only take returns on days we have a buy signal
    df["strategy_return"] = df["return"] * df["signal"].shift(1).fillna(0)

    trades    = df[df["signal"] != 0]
    wins      = trades[df["strategy_return"] > 0]
    win_rate  = round(len(wins) / len(trades) * 100, 1) if len(trades) > 0 else 0.0

    # ── total return ──────────────────────────────────────────────
    total_return = round(
        float((1 + df["strategy_return"]).prod() - 1) * 100, 2
    )

    # ── real max drawdown ─────────────────────────────────────────
    cumulative   = (1 + df["strategy_return"]).cumprod()
    rolling_peak = cumulative.cummax()
    drawdown     = (cumulative - rolling_peak) / rolling_peak
    max_drawdown = round(float(drawdown.min()) * 100, 2)

    # ── sharpe ratio (annualised, risk-free=0) ────────────────────
    mean_r = df["strategy_return"].mean()
    std_r  = df["strategy_return"].std()
    sharpe = round(float((mean_r / std_r) * np.sqrt(252)), 2) if std_r > 0 else 0.0

    # ── volatility (annualised) ───────────────────────────────────
    volatility = round(float(std_r * np.sqrt(252)) * 100, 2)

    return {
        "symbol":        symbol,
        "strategy":      strategy,
        "total_return":  f"{total_return}%",
        "max_drawdown":  f"{max_drawdown}%",
        "sharpe_ratio":  sharpe,
        "volatility":    f"{volatility}%",
        "total_trades":  len(trades),
        "win_rate":      f"{win_rate}%",
    }
