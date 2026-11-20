import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import os


#reading the adjusted close prices for nifty500 stocks from the csv file
df=pd.read_csv(r'nifty500_adjusted_close.csv') #or df=pd.read_csv(r'nifty100_adjusted_close.csv') depending on which dataset you want to analyze.
df
#  Cleaning the data frame and calculating the log returns.
# Count missing values in each column
print(df.isna().sum())

# Count non-finite values in numeric columns
print(np.isfinite(df.select_dtypes(include=np.number)).sum())
# Prepare the data for analysis by the following steps:
prices = df.drop(columns=['Date']).copy()
print("Percentage of missing values in each column:")
missing_pct = prices.isna().mean() * 100
missing_summary = missing_pct.sort_values(ascending=False)
print(missing_summary.head(20))
# Remove stocks with more than 20% missing prices
prices = prices.loc[:, missing_pct <= 20]
# calc log returns on the remaining prices.
log_returns = np.log(prices / prices.shift(1))


# Check missing returns
missing_returns = log_returns.isna().sum().sort_values(ascending=False)
print(missing_returns.head(20))
# calculate the percentage of missing returns for each stock
missing_pct = (log_returns.isna().mean() * 100).sort_values(ascending=False)
print(missing_pct.head(20))
# keeping only those stocks with less than or equal to 20% missing returns
valid_stocks = missing_pct[missing_pct <= 20].index
log_returns_clean = log_returns[valid_stocks].copy()

# finding the correlation matrix to form the edges of the network graph
#min_periods=60 ensures that we only consider pairs of stocks with at least 60 overlapping observations for correlation calculation.
correlation_matrix = log_returns.corr(method='pearson', min_periods=60) 
correlation_matrix

# Create an unweighted adjacency matrix
threshold = 0.2 # You can change this as you wish.

adjacency_matrix = (
    correlation_matrix.abs() >= threshold
).astype(int)

# Remove diagonal (self-connections) without modifying the original array
adjacency_matrix = adjacency_matrix * (
    1 - np.eye(len(adjacency_matrix), dtype=int)
)

print(adjacency_matrix.head())
print("Matrix shape:", adjacency_matrix.shape)
print("Number of connections:", adjacency_matrix.values.sum() // 2)

adjacency_matrix.to_csv("unweighted_adjacency_matrix.csv") 
# or adjacency_matrix.to_csv("unweighted_adjacency_matrix_nifty100.csv") depending on which dataset you want to analyze.

