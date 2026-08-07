import pandas as pd
import cvxpy as cp
import numpy as np


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.0,
    max_weight: float = 0.25,
    objective_type: str = "minimum_variance",
):
    # computes optimum weights using global minimum variance (gmv) portfolio
    # ignores expected returns and minimize risks
    n_assets = len(expected_returns)

    # weight vector to optimize
    w = cp.Variable(n_assets)

    # numeric values for the solver
    Sigma = cov_matrix.values
    mu = expected_returns.values

    Sigma = (Sigma + Sigma.T) / 2
    Sigma = cp.psd_wrap(Sigma)

    # objective function: minimum variance
    portfolio_variance = cp.quad_form(w, Sigma)
    if objective_type == "markowitz":
        porfolio_return = w.T @ mu
        objective = cp.Maximize(
            porfolio_return - (risk_aversion / 2) * portfolio_variance
        )
    elif objective_type == "minimum_variance":
        objective = cp.Minimize(portfolio_variance)
    else:
        raise ValueError("incorrect objective_type")

    # institutional constraints
    constraints = [
        cp.sum(w) == 1,  # fully invested
        w >= 0,  # long only operations
        w <= max_weight,  # no asset exceeds the limit
    ]

    # solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Portfolio optimization failed: {problem.status}")

    if w.value is None:
        raise RuntimeError("Portfolio optimizer returned no weights")

    optimal_weights = pd.Series(
        np.asarray(w.value).reshape(-1),
        index=expected_returns.index,
    )

    if not np.isfinite(optimal_weights).all():
        raise RuntimeError("Portfolio optimizer returned non-finite weights")

    return optimal_weights
