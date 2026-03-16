import yfinance as yf
import pandas as pd


def fetch_data(symbol: str, period: str = "1y") -> pd.DataFrame | None:
    """
    Fetch historical OHLCV data for a stock symbol using yfinance.

    Args:
        symbol : stock ticker e.g. 'AAPL', 'TCS.NS'
        period : yfinance period string e.g. '1y', '6mo', '3mo'

    Returns:
        pd.DataFrame with columns [Open, High, Low, Close, Volume]
        or None if fetch fails
    """
    if not symbol or not isinstance(symbol, str):
        print(f"[market_data] ERROR: Invalid symbol: {symbol}")
        return None

    symbol = symbol.upper().strip()

    try:
        ticker = yf.Ticker(symbol)
        df     = ticker.history(period=period)

        if df is None or df.empty:
            print(f"[market_data] WARNING: No data returned for {symbol}")
            return None

        # flatten MultiIndex columns if present (yfinance 0.2+)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # make sure Close column exists
        if "Close" not in df.columns:
            print(f"[market_data] ERROR: No 'Close' column for {symbol}. "
                  f"Available: {list(df.columns)}")
            return None

        # drop rows where Close is NaN
        df = df.dropna(subset=["Close"])

        print(f"[market_data] Fetched {len(df)} rows for {symbol} ({period})")
        return df

    except Exception as e:
        print(f"[market_data] ERROR fetching {symbol}: {str(e)}")
        return None
