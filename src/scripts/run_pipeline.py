from src.portfolio.metrics import compute_metrics
from src.portfolio.backtester import run_walk_forward_backtest
from src.data.download import get_prices
from src.visualization.plotter import plot_equity_curve, plot_weight_history
import pandas as pd

if __name__ == "__main__":
    tickers = (
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "JNJ",
        "XOM",
        "JPM",
        "PG",
        "NVDA",
    )

    start_date = "2017-01-01"
    end_date = "2026-01-01"

    prices = get_prices(tickers, start_date, end_date)

    portfolio_returns, weights_history = run_walk_forward_backtest(
        prices, lookback_days=252, rebalance_days=21
    )

    portfolio_equity = (1 + portfolio_returns).cumprod()

    daily_returns = prices.pct_change().fillna(0)
    naive_returns_full = daily_returns.mean(axis=1)
    naive_returns = naive_returns_full.loc[portfolio_returns.index]

    print(f"total rebalances: {len(weights_history)}")
    print(f"total return acc: {(portfolio_equity.iloc[-1] - 1) * 100:.2f}%")
    print("last weights:")
    print(weights_history.tail())

    plot_equity_curve(portfolio_returns, naive_returns)
    plot_weight_history(weights_history)

    strategy_metrics = compute_metrics(portfolio_returns)
    naive_metrics = compute_metrics(naive_returns)
    print("metrics:")
    metrics_df = pd.DataFrame(
        {
            "Markowitz (Strategy)": strategy_metrics,
            "Equal Weight (Naive)": naive_metrics,
        }
    ).T
    metrics_df["CAGR"] = (metrics_df["CAGR"] * 100).map("{:.2f}%".format)
    metrics_df["Volatility"] = (metrics_df["Volatility"] * 100).map("{:.2f}%".format)
    metrics_df["Sharpe"] = metrics_df["Sharpe"].map("{:.2f}".format)
    metrics_df["MDD"] = (metrics_df["MDD"] * 100).map("{:.2f}%".format)

    print(metrics_df)
