import yfinance as yf
import os
import pandas as pd


def get_data(
    tickers: set[str],
    start: str = "2018-01-01",
    end: str = "2024-01-01",
    base_out_path: str = "data/raw/",
):
    out_path = base_out_path + f"{hash(tickers)}-{start}-{end}"
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df = yf.download(tickers=tickers, start=start, end=end)
        df.to_parquet(out_path)
        return df
