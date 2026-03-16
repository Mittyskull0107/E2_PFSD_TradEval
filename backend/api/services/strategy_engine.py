# strategy_engine.py
# NOTE: The real strategy/backtest logic lives in backtester.py
# This file exists only for backwards compatibility — do not add logic here

from .backtester import run_backtest

def run_strategy(symbol: str, strategy: str) -> dict:
    """Wrapper around run_backtest for backwards compatibility."""
    return run_backtest(symbol, strategy)
