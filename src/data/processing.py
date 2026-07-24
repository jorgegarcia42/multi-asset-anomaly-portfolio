import pandas as pd


def get_returns_and_covariance(
    prices: pd.DataFrame, trading_days: int = 252
) -> tuple[pd.Series, pd.DataFrame]:
    returns = prices.pct_change(fill_method=None).dropna()
    expected_returns = returns.mean() * trading_days

    cov_matrix = returns.cov() * trading_days

    return (expected_returns, cov_matrix)
