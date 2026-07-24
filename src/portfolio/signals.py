import pandas as pd


def calculate_momentum_scores(
    prices: pd.DataFrame, lookback: int = 252, skip_recent: int = 21
):
    # computes cross sectional momentum and standarizes it using z scores

    if len(prices) <= skip_recent:
        return pd.Series(0, index=prices.columns)

    # price at the start of the window
    historical_price = prices.iloc[0]

    # price before the skipped days
    recent_price = prices.iloc[-skip_recent]

    momentum_score = (recent_price / historical_price) - 1
    momentum_score = momentum_score.fillna(0).replace([float("inf"), float("-inf")], 0)

    # z score standarization
    z_scores = (momentum_score - momentum_score.mean()) / momentum_score.std()
    z_scores = z_scores.fillna(0)

    return z_scores
