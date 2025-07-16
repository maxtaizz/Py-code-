import yfinance as yf
import numpy as np

def calculate_stock_beta(ticker, benchmark='^GSPC', start_date='2020-01-01', end_date='2025-01-01'):
    
    data = yf.download([ticker, benchmark], start=start_date, end=end_date)['Close']
    
    returns = np.log(data / data.shift(1))
    returns = returns.dropna()
    
    stock_returns = returns[ticker]
    benchmark_returns = returns[benchmark]
    
    covariance = np.cov(stock_returns, benchmark_returns)[0][1]
    variance = np.var(benchmark_returns)
    
    beta = covariance / variance
    return beta

if __name__ == "__main__":
    beta_aapl = calculate_stock_beta('AAPL') #asset change here
    print(f"Beta (vs benchmark): {beta_aapl:.4f}")