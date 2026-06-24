import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the ticker symbol for S&P 500
sp500 = yf.Ticker("^GSPC")

# Download historical data
hist = sp500.history(period="5y")  # Options: '1d', '1mo', '1y', '5y', 'max', etc.

# Print the first few rows
print(hist.head())

# Save to CSV
hist.to_csv("sp500_prices.csv")

# Optional: Plot the closing price
plt.figure(figsize=(10, 6))
plt.plot(hist.index, hist["Close"], label="S&P 500")
plt.title("S&P 500 Closing Prices (Last 5 Years)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

