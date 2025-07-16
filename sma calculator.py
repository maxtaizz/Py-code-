import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

tickerSMA = 'AAPL' #hardcoded but yeah 

data = yf.download (tickerSMA, start='2020-01-01', end='2024-11-18')
smatime = int(input("Enter sma duration : "))
ematime = int(input("Enter ema duration : "))

SMA = data['Close'].rolling(window=smatime).mean()
ema_values = data['Close'].ewm(span=ematime, adjust=False).mean()
smavalues = np.mean(SMA)

print ("The",smatime," day SMA is : ",smavalues)

plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(SMA, label=f'{smatime}-day SMA', color='orange')
plt.plot(data.index, ema_values, label=f'{ematime}-day EMA', color='green')
plt.title(f'{tickerSMA} Close Price and {smatime}-day SMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


