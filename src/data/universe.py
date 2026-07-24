import pandas as pd


def get_tickers() -> list:
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

    sp500_df = pd.read_csv(url)
    tickers = sp500_df["Symbol"].tolist()
    clean_tickers = [ticker.replace(".", "-") for ticker in tickers]
    return clean_tickers
