import numpy as np


def calculate_metrics(returns: list) -> dict:
    """
    Calculate basic performance metrics from a list of daily returns.

    Args:
        returns: list of daily return floats e.g. [0.02, -0.01, 0.03]

    Returns:
        dict with avg_return, volatility, sharpe_ratio, max_drawdown
    """
    if not returns or len(returns) < 2:
        return {
            "error":          "Need at least 2 return values",
            "average_return": 0.0,
            "volatility":     0.0,
            "sharpe_ratio":   0.0,
            "max_drawdown":   0.0,
        }

    r = np.array(returns, dtype=float)

    avg_return = float(r.mean())
    volatility = float(r.std())
    sharpe     = round(avg_return / volatility, 4) if volatility != 0 else 0.0

    # real max drawdown — peak to trough, not just worst single day
    cumulative   = np.cumprod(1 + r)
    rolling_peak = np.maximum.accumulate(cumulative)
    drawdowns    = (cumulative - rolling_peak) / rolling_peak
    max_drawdown = float(drawdowns.min())

    return {
        "average_return": round(avg_return, 4),
        "volatility":     round(volatility, 4),
        "sharpe_ratio":   round(sharpe, 4),
        "max_drawdown":   round(max_drawdown, 4),
    }
