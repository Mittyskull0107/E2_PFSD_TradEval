import yfinance as yf
from textblob import TextBlob
import numpy as np

def analyze_event(symbol: str) -> dict:

    if not symbol or not isinstance(symbol, str):
        return {"error": "Invalid symbol provided"}

    symbol = symbol.upper().strip()

    try:
        ticker = yf.Ticker(symbol)

        # ── 1. HISTORICAL PRICE DATA ──────────────────────────────
        hist = ticker.history(period="3mo")

        if hist.empty:
            return {"error": f"No historical data found for {symbol}"}

        prices     = hist["Close"].tolist()
        returns    = np.diff(prices) / prices[:-1]          # daily % returns
        volatility = round(float(np.std(returns) * 100), 2) # as percentage

        # ── 2. NEWS + SENTIMENT ───────────────────────────────────
        raw_news = ticker.news or []
        analyzed_news = []

        for article in raw_news[:5]:
            title    = article.get("title", "")
            link     = article.get("link", "")
            provider = article.get("publisher", "Unknown")

            # TextBlob sentiment: polarity -1.0 (negative) to +1.0 (positive)
            sentiment_score = round(TextBlob(title).sentiment.polarity, 3)

            if sentiment_score > 0.1:
                sentiment_label = "positive"
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            analyzed_news.append({
                "title":           title,
                "link":            link,
                "publisher":       provider,
                "sentiment_score": sentiment_score,
                "sentiment":       sentiment_label,
            })

        # ── 3. OVERALL SENTIMENT SIGNAL ───────────────────────────
        if analyzed_news:
            avg_score = round(
                sum(n["sentiment_score"] for n in analyzed_news) / len(analyzed_news), 3
            )
        else:
            avg_score = 0.0

        if avg_score > 0.1:
            overall_signal = "bullish"
        elif avg_score < -0.1:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # ── 4. EARNINGS CALENDAR ──────────────────────────────────
        earnings_info = {}
        try:
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                earnings_info = {
                    col: str(cal[col].iloc[0])
                    for col in cal.columns
                    if not cal[col].isnull().all()
                }
        except Exception:
            earnings_info = {"note": "No earnings calendar available"}

        # ── 5. BEHAVIOR SUMMARY ───────────────────────────────────
        recent_return = round(
            ((prices[-1] - prices[0]) / prices[0]) * 100, 2
        )

        return {
            "symbol":           symbol,
            "volatility_pct":   volatility,
            "recent_return_pct": recent_return,
            "overall_signal":   overall_signal,
            "avg_sentiment_score": avg_score,
            "news":             analyzed_news,
            "earnings_calendar": earnings_info,
            "behavior_summary": (
                f"{symbol} shows {overall_signal} sentiment based on "
                f"{len(analyzed_news)} recent news articles. "
                f"3-month return: {recent_return}%, "
                f"volatility: {volatility}%."
            )
        }

    except Exception as e:
        return {"error": f"Analysis failed for {symbol}: {str(e)}"}
