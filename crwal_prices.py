import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import os
# Download and save adjusted close prices for a list of tickers for nifty500 stocks from Yahoo Finance
url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
response.raise_for_status()

nifty500 = pd.read_csv(StringIO(response.text))

print(nifty500.head())
# Extract the list of tickers from the DataFrame
symbols = nifty500["Symbol"].dropna().tolist()

# Convert NSE symbols to Yahoo Finance format
tickers = [symbol + ".NS" for symbol in symbols]

print("Total stocks:", len(tickers))
print(tickers[:10])
# Download adjusted close prices for each ticker
all_prices = []

for i in range(0, len(tickers), 50):

    batch = tickers[i:i+50]

    data = yf.download(
        tickers=batch,
        period="3y",
        interval="1d",
        auto_adjust=False,
        group_by="column",
        threads=True,
        progress=True
    )

    if not data.empty:
        adj_close = data["Adj Close"]
        all_prices.append(adj_close)


adj_close_prices = pd.concat(all_prices, axis=1)

print(adj_close_prices.head())
adj_close_prices.to_csv("nifty500_adjusted_close.csv")
print("Adjusted close prices saved to:")
print(os.path.abspath("nifty500_adjusted_close.csv"))