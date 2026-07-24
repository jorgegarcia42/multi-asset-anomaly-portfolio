import pandas as pd
import numpy as np


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.0):
    trading_days = 252

    # cagr
    cum_return = (1 + returns).cumprod().iloc[-1]
    years = len(returns) / trading_days
    cagr = (cum_return ** (1 / years)) - 1

    # volatility
    volatility = returns.std() * np.sqrt(trading_days)

    # sharpe ratio
    sharpe_ratio = ((returns.mean() * trading_days) - risk_free_rate) / volatility

    # maximum drawdown
    equity_curve = (1 + returns).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe_ratio,
        "MDD": max_drawdown,
    }
