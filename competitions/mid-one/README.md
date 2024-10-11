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

See the [notebook](https://github.com/crunchdao/quickstarters/blob/master/competitions/mid-one/mean_reversion/mean_reversion.ipynb) for a slightly different phrasing of the task. 

## The infer function 

Your notebook needs to have an `infer` function that can yield one `prediction` at a time, though as noted above a prediction is really a "decision".  

![Infer](https://github.com/microprediction/endersnotebooks/blob/main/assets/images/infer.png?raw=true)

You can adopt the same style as this example. Of note, the `yield` function appears twice. The first yield signals to the system that you are ready (put any time consuming initialization before that). The second yield will return the decisions. 

** The `Attacker` class **


asdf


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
