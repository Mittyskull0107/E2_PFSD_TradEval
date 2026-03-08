import yfinance as yf

def analyze_event(symbol):

    ticker = yf.Ticker(symbol)

    earnings = ticker.calendar

    news = ticker.news

    return {
        "earnings_calendar": str(earnings),
        "recent_news": news[:3]
    }