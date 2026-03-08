import yfinance as yf

def fetch_data(symbol):

    data = yf.download(symbol, period="1y")

    return data