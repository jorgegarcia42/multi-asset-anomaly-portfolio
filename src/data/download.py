import yfinance as yf
import os
import pandas as pd


def get_prices(
    tickers: set[str],
    start: str = "2018-01-01",
    end: str = "2024-01-01",
    base_out_path: str = "data/raw/",
):
    out_path = base_out_path + f"{tickers[-1]}-{tickers[0]}-{start}-{end}"
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)["Adj Close"]
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False)
        df.to_parquet(out_path)
        return df["Adj Close"]
