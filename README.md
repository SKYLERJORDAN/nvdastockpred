# Stock Price Prediction Using Machine Learning

A Python-based machine learning project that predicts NVIDIA (NVDA) stock price movements using historical data and Random Forest classification.

## Overview

This project implements a machine learning model to predict whether NVIDIA's stock price will increase or decrease the next day. It uses historical price data including open, high, low, close prices, and volume, along with derived technical indicators to make predictions.

## Features

- Data fetching using yfinance API
- Technical indicator generation
- Machine learning model implementation using Random Forest
- Backtesting functionality
- Rolling window analysis
- Precision score evaluation

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- yfinance
- matplotlib

## Installation

```bash
pip install pandas numpy scikit-learn yfinance matplotlib
```

## Project Structure

The project consists of the following main components:

1. Data Collection & Preparation
   - Fetches historical NVIDIA stock data
   - Processes and cleans the data
   - Creates target variable for prediction

2. Feature Engineering
   - Calculates rolling averages
   - Generates price ratios
   - Creates trend indicators

3. Model Implementation
   - Random Forest Classifier
   - Probability threshold optimization
   - Backtesting framework

## Usage

The main workflow of the project:

```python
# Import required libraries
import pandas as pd
import yfinance as yahooFinance
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Fetch data
nvda = yahooFinance.Ticker("NVDA").history(period="max")

# Create prediction model
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Run backtesting
predictions = backtest(data, model, predictors)

# Evaluate results
precision_score(predictions["Target"], predictions["Predictions"])
```

## Model Performance

The model achieves the following performance metrics:
- Precision Score: ~57.6%
- Successfully identifies profitable trading opportunities above random chance
- Conservative prediction approach with higher confidence threshold

## Future Improvements

Potential areas for enhancement:
1. Feature engineering with additional technical indicators
2. Hyperparameter optimization
3. Implementation of different machine learning algorithms
4. Real-time prediction capabilities
5. Risk management integration
6. Portfolio optimization strategies

## License

MIT License

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss proposed changes.

## Disclaimer

This project is for educational purposes only. Trading stocks carries risk, and past performance does not guarantee future results. Always conduct your own research before making investment decisions.
