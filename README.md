# multi-asset anomaly portfolio engine
a quantitative simulation engine designed to exploit market anomalies across the sp500<br>
this engine was reengineered to eliminate structural lookahead biases, survivorships biases and execution impossibilities. it features a strick walk forward testing environment

## architecture
* data ingestion: we get the sp500 tickers from the start date, ensuring the engine doesnt see the future
* signal generator: right now i implement cross-sectional momentum, applying a z-score transformation to standarize while ignoring the most recent month to bypass short term mean reversion
* optimizer: engined based on cvxpy. including shrinkage to prevent matrix collapse when N > T
* backtester: walk-forward evaluator that enforces T+1 execution, tracks assets liquidation and manages the delisted equities

## rigor & bias elimination
* system reads from a historical snapshot, elegible assets are determined strictly using the historical lookback window. if an asset disappears, the engine liquidates its position
* the strategy cannot trade at the end of the day. instead, it buys at the backtester delays credited returns by one full session
* naive benchmark: the equal weight benchmark preserves missing returns instead of zero filling or forward filling

## performance
as for now, the cross sectional momentum isnt giving good results, for the moment and with the current configuration (markowitz, max_weight=0.25, risk_aversion=1, lookback_days=252, rebalance_days=21)
| | CAGR | Volatility | Sharpe | MDD |
| :--- | :--- | :--- | :--- | :--- |
| **Strategy** | 20.45% | 36.94% | 0.69 | -49.08% |
| **Equal Weight (Naive)** | 12.23% | 20.67% | 0.66 | -40.63% |

## usage
``pip install -r requirements.txt``
<br> then ``python -m src.scripts.run_pipeline.py``

## whats next
after reducing the max_weigth to 0.1 and 0.05 and not getting good results, i believe one of the main problems may be the covariance now, since we only use 251 observations, covariance is extremely noisy, adding epsilon didnt work out. so next step will be implementing and backtesting Ledoit-Wolf shrinkage, it should produce more stable weights when implemented. for the moment, here its the
[paper](http://www.ledoit.net/Honey_2004.pdf)