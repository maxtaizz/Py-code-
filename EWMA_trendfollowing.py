#need to input multiplier and fx myself keep in mind

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional

def fetch_price_series(ticker: str, start: str = None, end: str = None, interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)

    if data.empty:
        raise ValueError(f"No data for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        try:
           
            if isinstance(ticker, str) and ticker in data.columns.levels[1]:
                data = data.xs(ticker, axis=1, level=1, drop_level=True)
            else:
                
                data = data.xs(data.columns.levels[1][0], axis=1, level=1, drop_level=True)
        except Exception:
            data.columns = ['_'.join(map(str, col)).strip() for col in data.columns.values]

    if 'Adj Close' in data.columns:
        price_series = data['Adj Close'].astype(float)
    elif 'Close' in data.columns:
        price_series = data['Close'].astype(float)
    else:
        numeric_cols = data.select_dtypes(include='number').columns
        if len(numeric_cols) == 0:
            raise ValueError("No data points")
        price_series = data[numeric_cols[0]].astype(float)

    out = pd.DataFrame(index=price_series.index)
    out['price'] = price_series
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
        if col in data.columns:
            out[col] = data[col]

    out = out.dropna(subset=['price'])
    if out.empty:
        raise ValueError("No data")
    return out

def compute_buffered_ewma(df: pd.DataFrame,
                          price_col: str = "price",
                          N_short: int = 16,
                          N_long: int = 64,
                          alpha_short: Optional[float] = None,
                          alpha_long: Optional[float] = None,
                          vol_window: int = 64,
                          scale_factor: float = 4.1,
                          cap: float = 20.0,
                          capital: float = 1_000_000.0,
                          idm: float = 1.0,
                          weight: float = 1.0,
                          tau: float = 1.0,
                          multiplier: float = 1.0,
                          fx: float = 1.0,
                          buffer_fraction: float = 0.1,
                          min_sigma: float = 1e-8,
                          round_to: int = 1):
   
    df = df.copy()
    if alpha_short is None:
        alpha_short = 2.0 / (N_short + 1.0)
    if alpha_long is None:
        alpha_long = 2.0 / (N_long + 1.0)

    df['ewma_short'] = df[price_col].ewm(alpha=alpha_short, adjust=False).mean()
    df['ewma_long'] = df[price_col].ewm(alpha=alpha_long, adjust=False).mean()

    df['logret'] = np.log(df[price_col]).diff()
    df['sigma_pct'] = df['logret'].rolling(vol_window).std()
    df['sigma_pct'] = df['sigma_pct'].fillna(method='bfill').fillna(min_sigma).clip(lower=min_sigma)

    df['ewma_diff'] = df['ewma_short'] - df['ewma_long']
    df['raw_forecast'] = df['ewma_diff'] / (df[price_col] * df['sigma_pct'])

    df['scaled_forecast'] = df['raw_forecast'] * scale_factor
    df['capped_forecast'] = df['scaled_forecast'].clip(-cap, cap)

    df['N_unrounded'] = (df['capped_forecast'] * capital * idm * weight * tau) \
                        / (10.0 * multiplier * df[price_col] * fx * df['sigma_pct'])

    df['B_unrounded'] = (buffer_fraction * capital * idm * weight * tau) \
                        / (multiplier * df[price_col] * fx * df['sigma_pct'])

    def round_contracts(x):
        if round_to == 1:
            return int(np.round(x))
        else:
            return int(np.round(x / round_to) * round_to)

    df['N_target'] = df['N_unrounded'].round().astype('Int64') 
    df['lower_buffer'] = (df['N_unrounded'] - df['B_unrounded']).round().astype('Int64')
    df['upper_buffer'] = (df['N_unrounded'] + df['B_unrounded']).round().astype('Int64')

    if 'position' not in df.columns:
        df['position'] = 0

    def decide_trade(row):
        C = int(row['position'])
        L = int(row['lower_buffer'])
        U = int(row['upper_buffer'])
        if C >= L and C <= U:
            return 0  # no trade
        elif C < L:
            return int(U - C) 
        else: 
            return -int(C - L) 
    df['trade_qty'] = df.apply(decide_trade, axis=1)

    summary = {
        'last_date': df.index[-1],
        'last_price': float(df[price_col].iloc[-1]),
        'last_capped_forecast': float(df['capped_forecast'].iloc[-1]),
        'N_unrounded_last': float(df['N_unrounded'].iloc[-1]),
        'N_target_last': int(df['N_target'].iloc[-1]),
        'lower_buffer_last': int(df['lower_buffer'].iloc[-1]),
        'upper_buffer_last': int(df['upper_buffer'].iloc[-1]),
        'trade_qty_last': int(df['trade_qty'].iloc[-1]),
    }

    return df, summary

def run_for_ticker(ticker: str,
                   start: str = '2018-01-01',
                   end: str = None,
                   out_csv: Optional[str] = None,
                   **strategy_kwargs):
   
    df = fetch_price_series(ticker, start=start, end=end)
    df_signals, summary = compute_buffered_ewma(df, **strategy_kwargs)
    if out_csv:
        df_signals.to_csv(out_csv, index=True)
    print("Signal(latest):")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return df_signals, summary

# change ticker and date here
if __name__ == "__main__":
    df_signals, summary = run_for_ticker('NQ=F', start='2020-01-01', out_csv='tickersignals.csv',
                                         capital=50000,
                                         idm=1.0,
                                         weight=1.0,
                                         tau=0.2,
                                         multiplier=20.0, 
                                         fx=1.0,
                                         buffer_fraction=0.1,
                                         vol_window=64)
    print("Wrote the CSV to the code folder")
