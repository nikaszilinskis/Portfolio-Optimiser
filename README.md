# Portfolio Optimiser

## Overview

This project implements Modern Portfolio Theory (MPT) using Python to optimise a portfolio of stocks. The main objective is to find the optimal allocation of assets that maximises the Sharpe Ratio and minimises the portfolio variance. Additionally, an efficient frontier is plotted to visualise the optimal portfolios.

## Features

- Download stock data using Yahoo Finance.
- Calculate daily and annualised returns and covariances.
- Optimise portfolios using different approaches: 
  - Maximising Sharpe Ratio.
  - Minimising Variance.
  - Generating Efficient Frontier.
- Compare the optimised portfolios to an equal-weighted (1/n) portfolio.
- Visualise the results using bar charts and an efficient frontier plot.

## Requirements

- Python 3.x
- `yfinance` for downloading stock data
- `pandas` for data manipulation
- `pandas_datareader` for additional data handling
- `numpy` for numerical operations
- `scipy` for optimisation
- `matplotlib` and `seaborn` for plotting
- `plotly` for interactive plots

## Usage

Run the portfoliooptimiser.py file.
