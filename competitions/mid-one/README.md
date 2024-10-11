# README: Mid+One

Example notebooks are provided to help you get going with a new style of crunch at [Mid+One](  ) where you are invited to make directional predictions of time-series. 

## Goal: Detecting small deviations from the martingale property 

Your goal is to try to discern whether, in 30 data points time, a sequence of numbers will be higher or lower than where it currently is. 

![Time Series](https://github.com/microprediction/endersnotebooks/blob/main/assets/images/timeseries.png?raw=true)

However, unlike a typical forecasting task, you don't need to make a prediction for every data point. Instead, you should try to discern only the occasions when you are confident in the direction only. To be precise, your
task is to determine, for each time point, which of three conditions is true. In the language of mathematics:

$$
\begin{aligned}
\text{If} \quad E[x_{t+30} | x_t, x_{t-1}, \dots] &> x_t + \epsilon \quad &\text{then return} \quad 1 \\
\text{If} \quad E[x_{t+30} | x_t, x_{t-1}, \dots] &< x_t - \epsilon \quad &\text{then return} \quad -1 \\
\text{Otherwise} \quad  & &\text{return} \quad 0
\end{aligned}
$$

In the language of trading. If you expect the price $30$ periods from now to be ...

| Condition | Action |
| --- | --- |
| **Higher** than the current price by more than $\epsilon$ | **Buy and hold for 30 periods** (Return 1) |
| **Lower** than the current price by more than $\epsilon$ | **Sell and hold for 30 periods** (Return -1) |
| Neither| **Abstain** (Return 0) |






In this model, the "attacker" only trades when it believes the series is deviating from its expected behavior, signaling an opportunity for action.

## Key Concept: Difference from a Forecast
- A **forecast** tries to predict the next value in the sequence.
- In **mean reversion**, the model is interested in detecting when the series has deviated from its mean and is expected to revert back, signaling potential trading actions (buy, sell, hold).

The model signals when it believes there is a deviation from the martingale property and returns:
- **1**: Signal to **buy** if the value is below the mean.
- **-1**: Signal to **sell** if the value is above the mean.
- **0**: Signal to **hold** if no significant deviation is detected.

The model also considers a **trading cost**, implying that most predictions will be 0 (no action), as trading opportunities are expected to be rare.




| Notebook | Description |
| --- | --- |
| [Notebook](https://github.com/microprediction/quickstarters/blob/master/competitions/mid-one/mean_reversion/mean_reversion.ipynb) | This notebook demonstrates a mean reversion strategy that predicts whether a time series will go up or down. |
| [Notebook](https://github.com/microprediction/quickstarters/blob/master/competitions/mid-one/mean_reversion/mean_reversion.ipynb) | Implements an attacker strategy using a deviation from martingale behavior to make buy, sell, or hold decisions. |
| [Notebook](https://github.com/microprediction/quickstarters/blob/master/competitions/mid-one/mean_reversion/mean_reversion.ipynb) | Shows how to process univariate time series data streams to detect trading opportunities. |
| [Notebook](https://github.com/microprediction/quickstarters/blob/master/competitions/mid-one/mean_reversion/mean_reversion.ipynb) | Includes the basic setup, data loading, and submission process for CrunchDAO competitions. |
| [Notebook](https://github.com/microprediction/quickstarters/blob/master/competitions/mid-one/mean_reversion/mean_re



## Setup

To set up and use this notebook, follow these steps:

1. **Install dependencies:**
