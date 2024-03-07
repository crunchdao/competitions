# quant-venture-capital-resources

![xalpha](https://github.com/crunchdao/quant-venture-capital-resources/assets/39467268/7f63aafe-55c5-4092-9224-c4c9d4be6ac3)


This bounty marks the beginning of the "Rally" series, comprising multiple iterations aimed at refining our Data-Generating Process. Each Rally features a short submission phase (up to one month) to enhance this process. The steps involved in each Rally are as follows: the release of a new dataset version, model submission, out-of-sample scoring, result analysis, and the subsequent release of the next dataset version. This iterative process will continue until the desired level of accuracy is achieved.

We release only a fraction of the data for several reasons. Firstly, some data is reserved for upcoming "Rallies" and for out-of-sample scoring. Secondly, due to the challenging size of the dataset (1.5 TB), it requires significant computing resources that may not be readily available to most community members.

# One-Month Machine Learning Bounty

You have one month to submit your best supervised machine learning model for categorizing startups likely to experience an upround (Pre-Seed to Seed, to Serie A, to Serie B, to Serie C, etc.).

## Problem Statement

We categorized startups to enable participants to develop supervised learning algorithms. Startups labeled as 1 are expected to achieve higher valuations, while those labeled as 0 are not anticipated to experience significant valuation growth.

## Scoring

To assess the AI's performance effectively, we will compute its F1 score. This metric balances precision (true positives identified by the algorithm) and recall (accounting for missed opportunities). The algorithm must accurately identify investment opportunities while minimizing false negatives and false positives for it to demonstrate effectiveness. The F1 score will provide a comprehensive view of the algorithm's accuracy and reliability.
