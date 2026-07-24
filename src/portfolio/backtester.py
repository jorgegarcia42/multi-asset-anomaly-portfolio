import numpy as np
import pandas as pd

from src.portfolio.optimizer import optimize_portfolio
from src.data.processing import get_returns_and_covariance


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    lookback_days: int = 252,
    rebalance_days: int = 21,
    max_weight: float = 0.25,
    transaction_fee: float = 0.001,
    objective_type: str = "minimum_variance",
) -> tuple[pd.Series, pd.DataFrame]:
    # run backtest with dynamic rebalancing

    daily_returns = prices.pct_change().fillna(0)

    portfolio_returns = []
    weight_history = []

    last_weights = pd.Series(0, index=prices.columns)
    for i in range(lookback_days, len(prices), rebalance_days):
        past_prices = prices.iloc[i - lookback_days : i]
        mu, cov = get_returns_and_covariance(past_prices)

        weights = optimize_portfolio(
            mu, cov, max_weight=max_weight, objective_type=objective_type
        )

        # save the weights for an specific day
        rebalance_date = prices.index[i]
        weight_history.append(weights.to_frame(name=rebalance_date))

        # turnover and costs
        turnover = np.abs(weights - last_weights).sum()
        rebalance_cost = turnover * transaction_fee

        # get the returns
        forward_returns = daily_returns.iloc[i : i + rebalance_days]
        period_returns = forward_returns.dot(weights)
        period_returns.iloc[0] -= rebalance_cost
        portfolio_returns.append(period_returns)

        last_weights = weights

    all_portolio_returns = pd.concat(portfolio_returns)
    all_weights_df = pd.concat(weight_history, axis=1).T

    return all_portolio_returns, all_weights_df
