import yfinance as yf
import os
import pandas as pd
import hashlib


def get_prices(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str = "2024-01-01",
    base_out_path: str = "data/raw/",
):
    sorted_tickers = sorted(tickers)
    ticker_string = ",".join(sorted_tickers)
    ticker_hash = hashlib.sha256(ticker_string.encode()).hexdigest()[:16]

    out_path = os.path.join(
        base_out_path,
        f"adj-close-{ticker_hash}-{start}-{end}.parquet",
    )
    if os.path.exists(out_path):
        return pd.read_parquet(out_path)["Adj Close"]
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df = yf.download(
            tickers=sorted_tickers, start=start, end=end, auto_adjust=False
        )
        df.to_parquet(out_path)
        return df["Adj Close"]
