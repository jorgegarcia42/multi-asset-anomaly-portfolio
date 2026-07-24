from src.portfolio.optimizer import optimize_portfolio
from src.data.download import get_prices
from src.data.processing import get_returns_and_covariance

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
    prices = get_prices(tickers)
    expected_returns, cov_matrix = get_returns_and_covariance(prices)

    print("Anual expected results:")
    print(expected_returns)
    print("Cov Matrix")
    print(cov_matrix)

    optimal_weights = optimize_portfolio(expected_returns, cov_matrix)

    print("capital allocation")
    active_positions = optimal_weights[optimal_weights > 0.001].sort_values(
        ascending=False
    )
    print(active_positions * 100)
