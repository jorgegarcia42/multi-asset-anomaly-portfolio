import warnings

import numpy as np
import pandas as pd

from src.portfolio.signals import calculate_momentum_scores
from src.portfolio.optimizer import optimize_portfolio
from src.data.processing import get_returns_and_covariance


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    lookback_days: int = 252,
    rebalance_days: int = 21,
    max_weight: float = 0.25,
    transaction_fee: float = 0.001,
    objective_type: str = "minimum_variance",
    risk_aversion: float = 2.0,
) -> tuple[pd.Series, pd.DataFrame]:
    # run backtest with dynamic rebalancing

    daily_returns = prices.pct_change(fill_method=None)

    portfolio_returns = []
    weight_history = []
    retired_assets: set[str] = set()

    last_weights = pd.Series(0, index=prices.columns)
    for i in range(lookback_days, len(prices) - 1, rebalance_days):
        past_prices = prices.iloc[i - lookback_days : i]

        eligible_mask = past_prices.notna().all(axis=0)
        eligible_mask &= ~past_prices.columns.isin(retired_assets)
        eligible_columns = past_prices.columns[eligible_mask]
        eligible_prices = past_prices.loc[:, eligible_columns]

        minimum_assets = int(np.ceil(1 / max_weight))
        if len(eligible_columns) < minimum_assets:
            raise ValueError(
                f"Only {len(eligible_columns)} eligible assets on "
                f"{prices.index[i]}; at least {minimum_assets} are required"
            )

        _, cov = get_returns_and_covariance(eligible_prices)

        mu_signal = calculate_momentum_scores(
            eligible_prices,
            lookback=lookback_days,
            skip_recent=21,
        )

        eligible_weights = optimize_portfolio(
            mu_signal,
            cov,
            max_weight=max_weight,
            objective_type=objective_type,
            risk_aversion=risk_aversion,
        )

        weights = pd.Series(0.0, index=prices.columns)
        weights.loc[eligible_weights.index] = eligible_weights

        # save the weights for an specific day
        rebalance_date = prices.index[i]
        weight_history.append(weights.to_frame(name=rebalance_date))

        # turnover and costs
        turnover = np.abs(weights - last_weights).sum()
        rebalance_cost = turnover * transaction_fee

        # the signal uses prices through i - 1 and executes at close i. the new
        # weights therefore start earning returns from close i to close i + 1.
        active_weights = weights[weights.abs() > 1e-10]

        execution_prices = prices.iloc[i].loc[active_weights.index]
        if execution_prices.isna().any():
            missing = execution_prices.index[execution_prices.isna()].tolist()
            raise ValueError(
                f"Missing execution prices on {prices.index[i]} for {missing}"
            )

        forward_returns = daily_returns.iloc[i + 1 : i + rebalance_days + 1].loc[
            :, active_weights.index
        ].copy()

        liquidated_assets = []
        liquidation_costs = pd.Series(0.0, index=forward_returns.index)
        for ticker in forward_returns.columns:
            missing_positions = np.flatnonzero(
                forward_returns[ticker].isna().to_numpy()
            )
            if len(missing_positions) == 0:
                continue

            first_missing_position = int(missing_positions[0])
            first_missing_date = forward_returns.index[first_missing_position]
            latest_price = prices.loc[:first_missing_date, ticker].dropna().iloc[-1]

            # approximation: sell at the latest valid price, then hold the
            # proceeds as cash for the rest of this holding period.
            forward_returns.iloc[
                first_missing_position:,
                forward_returns.columns.get_loc(ticker),
            ] = 0.0
            liquidation_costs.iloc[first_missing_position] += (
                abs(active_weights[ticker]) * transaction_fee
            )
            liquidated_assets.append(ticker)
            warnings.warn(
                f"Liquidating {ticker} at its latest valid price "
                f"({latest_price:.2f}) before {first_missing_date.date()}",
                RuntimeWarning,
                stacklevel=2,
            )

        period_returns = forward_returns.dot(active_weights)
        period_returns -= liquidation_costs
        period_returns.iloc[0] -= rebalance_cost
        portfolio_returns.append(period_returns)

        last_weights = weights.copy()
        last_weights.loc[liquidated_assets] = 0.0
        retired_assets.update(liquidated_assets)

    all_portolio_returns = pd.concat(portfolio_returns)
    all_weights_df = pd.concat(weight_history, axis=1).T

    return all_portolio_returns, all_weights_df
