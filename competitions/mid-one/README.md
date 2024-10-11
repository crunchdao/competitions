# README: Mid+One

Example notebooks are provided to help you get going with a new style of crunch at [Mid+One](  ) where you are invited to make directional predictions of time-series. 

| Notebook | Description |
| --- | --- |
| [Notebook](https://github.com/microprediction/quickstarters/blob/master/competitions/mid-one/mean_reversion/mean_reversion.ipynb) | This notebook demonstrates a mean reversion strategy that predicts whether a time series will go up or down. |
| [Notebook](https://github.com/microprediction/quickstarters/blob/master/competitions/mid-one/mean_reversion/mean_reversion.ipynb) | Implements an attacker strategy using a deviation from martingale behavior to make buy, sell, or hold decisions. |
| [Notebook](https://github.com/microprediction/quickstarters/blob/master/competitions/mid-one/mean_reversion/mean_reversion.ipynb) | Shows how to process univariate time series data streams to detect trading opportunities. |
| [Notebook](https://github.com/microprediction/quickstarters/blob/master/competitions/mid-one/mean_reversion/mean_reversion.ipynb) | Includes the basic setup, data loading, and submission process for CrunchDAO competitions. |
| [Notebook](https://github.com/microprediction/quickstarters/blob/master/competitions/mid-one/mean_reversion/mean_re






## Overview

This notebook implements a **Mean Reversion** strategy that aims to predict whether a time series will go up or down, but only when it has a strong opinion. Instead of forecasting precise values, it focuses on detecting deviations from the expected martingale property of the time series, which is when we expect no systematic changes:

\[
E[x_{t+1} | x_t, x_{t-1}, \dots] = x_t
\]

In this model, the "attacker" only trades when it believes the series is deviating from its expected behavior, signaling an opportunity for action.

## Key Concept: Difference from a Forecast
- A **forecast** tries to predict the next value in the sequence.
- In **mean reversion**, the model is interested in detecting when the series has deviated from its mean and is expected to revert back, signaling potential trading actions (buy, sell, hold).

The model signals when it believes there is a deviation from the martingale property and returns:
- **1**: Signal to **buy** if the value is below the mean.
- **-1**: Signal to **sell** if the value is above the mean.
- **0**: Signal to **hold** if no significant deviation is detected.

The model also considers a **trading cost**, implying that most predictions will be 0 (no action), as trading opportunities are expected to be rare.

## Setup

To set up and use this notebook, follow these steps:

1. **Install dependencies:**
