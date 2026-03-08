import numpy as np
from .market_data import fetch_data

def run_backtest(symbol, strategy):

    df = fetch_data(symbol)

    df["returns"] = df["Close"].pct_change()

    avg_return = float(df["returns"].mean())
    volatility = float(df["returns"].std())
    max_drawdown = float(df["returns"].min())

    metrics = {
        "avg_return": avg_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown
    }

    return {
        "symbol": symbol,
        "strategy": strategy,
        "metrics": metrics
    }