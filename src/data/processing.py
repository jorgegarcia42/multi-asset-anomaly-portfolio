import pandas as pd
from sklearn.covariance import LedoitWolf


def get_returns_and_covariance(
    prices: pd.DataFrame, trading_days: int = 252
) -> tuple[pd.Series, pd.DataFrame]:
    # get expected returns and cov matrix using shrinkage method by Ledoit-Wolf

    daily_returns = prices.pct_change().dropna(how="all")

    mu = daily_returns.mean()

    lw = LedoitWolf()
    lw.fit(daily_returns.values)

    shrunk_cov = lw.covariance_

    cov_matrix = pd.DataFrame(
        shrunk_cov, index=daily_returns.columns, columns=daily_returns.columns
    )
    return mu, cov_matrix
