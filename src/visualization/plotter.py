import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(portfolio_returns: pd.DataFrame):
    equity = (1 + portfolio_returns).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(equity.index, equity, label="strategy (net)", color="blue", linewidth=2)
    plt.title("portfolio equity")
    plt.tight_layout()
    plt.show()


def plot_weight_history(weights_df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    active_weights = weights_df.loc[:, (weights_df > 0.01).any(axis=0)]
    plt.stackplot(
        active_weights.index,
        active_weights.T,
        labels=active_weights.columns,
        alpha=0.85,
    )
    plt.title("weigth history")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
