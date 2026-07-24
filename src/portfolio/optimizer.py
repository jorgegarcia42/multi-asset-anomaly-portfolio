import pandas as pd
import cvxpy as cp


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.0,
    max_weight: float = 0.25,
):
    # computes optimum weights using convex optimization (Markowitz)
    # maximizes risk-adjusted return based on institutional constraints
    n_assets = len(expected_returns)

    # weight vector to optimize
    w = cp.Variable(n_assets)

    # numeric values for the solver
    mu = expected_returns.values
    Sigma = cov_matrix.values

    # objective function: Expected Return - Risk Penalty
    portfolio_return = w.T @ mu
    portfolio_variance = cp.quad_form(w, Sigma)
    objective = cp.Maximize(portfolio_return - (risk_aversion / 2) * portfolio_variance)

    # institutional constraints
    constraints = [
        cp.sum(w) == 1,  # fully invested
        w >= 0,  # long only operations
        w <= max_weight,  # no asset exceeds the limit
    ]

    # solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    optimal_weights = pd.Series(w.value, index=expected_returns.index)
    optimal_weights = optimal_weights.round(4)
    return optimal_weights
