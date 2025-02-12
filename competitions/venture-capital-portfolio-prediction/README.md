![Banner](./assets/banner.webp)

# Venture Capital Portfolio Prediction Rally

This bounty marks the beginning of the "Rally" series, comprising multiple iterations aimed at refining our Data-Generating Process. Each Rally features a short submission phase (up to one month) to enhance this process. The steps involved in each Rally are as follows: the release of a new dataset version, model submission, out-of-sample scoring, result analysis, and the subsequent release of the next dataset version. This iterative process will continue until the desired level of accuracy is achieved.

We release only a fraction of the data for several reasons. Firstly, some data is reserved for upcoming "Rallies" and for out-of-sample scoring. Secondly, due to the challenging size of the dataset (1.5 TB), it requires significant computing resources that may not be readily available to most community members.

# One-Month Machine Learning Bounty

You have one month to submit your best supervised machine learning model for categorizing startups likely to experience an upround (Pre-Seed to Seed, to Serie A, to Serie B, to Serie C, etc.).

## Problem Statement

We categorized startups to enable participants to develop supervised learning algorithms. Startups labeled as 1 are expected to achieve higher valuations, while those labeled as 0 are not anticipated to experience significant valuation growth.

## Scoring

To assess the AI's performance effectively, we will compute its F1 score. This metric balances precision (true positives identified by the algorithm) and recall (accounting for missed opportunities). The algorithm must accurately identify investment opportunities while minimizing false negatives and false positives for it to demonstrate effectiveness. The F1 score will provide a comprehensive view of the algorithm's accuracy and reliability.

# Links

- [Overview](https://hub.crunchdao.com/competitions/venture-capital-portfolio-prediction)
- [Rules](https://hub.crunchdao.com/competitions/venture-capital-portfolio-prediction/rules)
- [Documentation](https://docs.crunchdao.com/competitions/rallies/mid+one)
- [Notebook](https://colab.research.google.com/github/crunchdao/quickstarters/blob/master/competitions/venture-capital-portfolio-prediction/quickstarters/quickstarter_xgboost/quickstarter_xgboost.ipynb)
- [Discussion](https://discord.com/invite/veAtzsYn3M)
