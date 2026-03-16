import os
import requests
from datetime import datetime, timedelta
from textblob import TextBlob

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# company name mappings for better search results
SYMBOL_TO_COMPANY = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Google",
    "AMZN":  "Amazon",
    "TSLA":  "Tesla",
    "RELIANCE": "Reliance Industries",
    "TCS":   "Tata Consultancy",
    "INFY":  "Infosys",
    "HDFCBANK": "HDFC Bank",
    "WIPRO": "Wipro",
}

def _score_sentiment(text: str) -> dict:
    """Score a piece of text using TextBlob."""
    polarity = round(TextBlob(text).sentiment.polarity, 3)
    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"
    return {"score": polarity, "label": label}


def _impact_weight(article: dict) -> float:
    """
    Give higher weight to articles from reputable financial sources.
    Used to compute a weighted sentiment signal.
    """
    high_impact = [
        "reuters", "bloomberg", "wsj", "financial times",
        "economic times", "moneycontrol", "livemint", "cnbc"
    ]
    source = article.get("source", {}).get("name", "").lower()
    return 2.0 if any(h in source for h in high_impact) else 1.0


def fetch_news(symbol: str, days_back: int = 7) -> dict:
    """
    Fetch and analyse recent news for a stock symbol.

    Args:
        symbol   : stock ticker e.g. 'AAPL', 'TCS'
        days_back: how many days of news to pull (max 30 for free tier)

    Returns a dict with analyzed articles and an overall sentiment signal.
    """
    if not NEWSAPI_KEY:
        return {
            "error": (
                "NEWSAPI_KEY not set. "
                "Get a free key at https://newsapi.org and add it "
                "to your .env file as NEWSAPI_KEY=your_key_here"
            ),
            "articles":        [],
            "overall_signal":  "neutral",
            "avg_score":       0.0,
        }

    # use company name if available for better results
    query      = SYMBOL_TO_COMPANY.get(symbol.upper(), symbol)
    date_from  = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q":        f"{query} stock",
        "from":     date_from,
        "sortBy":   "relevancy",
        "language": "en",
        "pageSize": 10,
        "apiKey":   NEWSAPI_KEY,
    }

    try:
        response = requests.get(NEWSAPI_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        return {"error": "NewsAPI request timed out", "articles": []}
    except requests.exceptions.RequestException as e:
        return {"error": f"NewsAPI request failed: {str(e)}", "articles": []}

    raw_articles = data.get("articles", [])

    if not raw_articles:
        return {
            "error":           f"No news found for {symbol}",
            "articles":        [],
            "overall_signal":  "neutral",
            "avg_score":       0.0,
        }

    # ── analyse each article ──────────────────────────────────────
    analyzed   = []
    total_score  = 0.0
    total_weight = 0.0

    for article in raw_articles:
        title       = article.get("title") or ""
        description = article.get("description") or ""

        # score on title + description combined for better accuracy
        combined_text = f"{title}. {description}".strip()
        sentiment     = _score_sentiment(combined_text)
        weight        = _impact_weight(article)

        total_score  += sentiment["score"] * weight
        total_weight += weight

        analyzed.append({
            "title":       title,
            "description": description,
            "url":         article.get("url", ""),
            "source":      article.get("source", {}).get("name", "Unknown"),
            "published_at": article.get("publishedAt", ""),
            "sentiment_score": sentiment["score"],
            "sentiment":       sentiment["label"],
            "impact_weight":   weight,
        })

    # ── weighted overall signal ───────────────────────────────────
    avg_score = round(total_score / total_weight, 3) if total_weight > 0 else 0.0

    if avg_score > 0.1:
        overall_signal = "bullish"
    elif avg_score < -0.1:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    # ── count sentiment breakdown ─────────────────────────────────
    positive_count = sum(1 for a in analyzed if a["sentiment"] == "positive")
    negative_count = sum(1 for a in analyzed if a["sentiment"] == "negative")
    neutral_count  = sum(1 for a in analyzed if a["sentiment"] == "neutral")

    return {
        "symbol":         symbol.upper(),
        "query_used":     query,
        "articles_found": len(analyzed),
        "overall_signal": overall_signal,
        "avg_score":      avg_score,
        "breakdown": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral":  neutral_count,
        },
        "articles": analyzed,
    }
