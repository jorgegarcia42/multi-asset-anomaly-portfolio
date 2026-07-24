import pandas as pd


def get_tickers() -> list:
    sp500_df = pd.read_csv("./data/raw/constituents.csv")
    tickers = sp500_df["Symbol"].tolist()
    clean_tickers = [ticker.replace(".", "-") for ticker in tickers]
    return clean_tickers
