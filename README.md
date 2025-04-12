# pyfin - This project for using Python for Technical Analysis  

ToC 
- Explain Technical Indicators - MA(10), MA(50), MA(200), RSI(14), MACD(12,26,9)
- Programmatically obtain and chart these indicators using Python language
- Bash script to setup the pyfin project on Ubuntu the `pyenv virtualenv` running `Python 3.11`
- Jupyter Notebook for Stock Analysis
- Future projects to consider

---

# Explain Technical Indicators - MA(10), MA(50), MA(200), RSI(14), MACD(12,26,9)

## Technical Indicators: MA, MACD, and RSI
These three indicators are widely used in technical analysis to identify trends, predict price movements, and generate buy/sell signals.

## Moving Average (MA)
----------------------

- *Definition*: A moving average is a calculation of the average price of a security over a specified period.

- *Types*: Simple Moving Average (SMA), Exponential Moving Average (EMA)

- *MA(10)*: A short-term moving average that calculates the average price over the last 10 periods. 
            It's sensitive to recent price movements and can help identify short-term trends.

- *MA(50)*: A medium-term moving average that calculates the average price over the last 50 periods. 
            It's less sensitive to recent price movements than MA(10) and can help identify medium-term trends.

- *MA(200)*: A long-term moving average that calculates the average price over the last 200 periods. 
             It's less sensitive to recent price movements and can help identify long-term trends.

- *Uses*:
    - Identify trends: MA can help determine the direction and strength of a trend.
    - Provide support/resistance: MA can act as a support or resistance level for prices.
    - Generate signals: Crossovers between short-term and long-term MAs can generate buy/sell signals.


## Relative Strength Index (RSI)
--------------------------------

- *Definition*: RSI is a momentum oscillator that measures the magnitude of recent price changes.
- *Calculation*: RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss

- *RSI(14)*: A momentum oscillator that measures the magnitude of recent price changes over 14 periods. 
             It's used to identify overbought (RSI > 70) and oversold (RSI < 30) conditions, as well as potential trend reversals.

- *Uses*:
    - Identify overbought/oversold conditions: RSI can help identify when a security is overbought (RSI > 70) or oversold (RSI < 30).
    - Generate buy/sell signals: RSI can generate buy/sell signals based on overbought/oversold conditions or divergences.
    - Measure momentum: RSI can help measure the strength of a trend.


## Moving Average Convergence Divergence (MACD)
-----------------------------------------------

- *Definition*: MACD is a momentum indicator that calculates the difference between two EMAs.

- *MACD(12,26,9)*: A momentum indicator that calculates the difference between two EMAs.

- *Components*:
    - MACD line: Difference between 12-period EMA and 26-period EMA
    - Signal line: 9-period EMA of the MACD line

- *Uses*:
    - Identify trend reversals: MACD crossovers can signal trend reversals.
    - Generate buy/sell signals: MACD line crossing above/below the signal line can generate buy/sell signals.
    - Measure momentum: MACD can help measure the strength of a trend.

    - *Fast EMA*: 12-period EMA
    - *Slow EMA*: 26-period EMA
    - *Signal Line*: 9-period EMA of the MACD line

- *Uses*:
    - Identify trend reversals: MACD crossovers can signal trend reversals.
    - Generate buy/sell signals: MACD line crossing above/below the signal line can generate buy/sell signals.
    - Measure momentum: MACD can help measure the strength of a trend.

---

# Programmatically obtain and chart these indicators using Python

## Obtaining and Charting Technical Indicators using Python
To obtain and chart technical indicators like MA, RSI, and MACD programmatically using Python, you can use libraries like 
    `yfinance` for data retrieval, 
    `pandas` for data manipulation, and 
    `matplotlib` or `plotly` for charting.

## Required Libraries
- `yfinance` for retrieving stock data
- `pandas` for data manipulation
- `numpy` for numerical computations
- `matplotlib` or `plotly` for charting

## Example Code
Here's an example code snippet that retrieves stock data, calculates MA, RSI, and MACD, and charts them using `matplotlib`:

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Retrieve stock data
stock_data = yf.download('AAPL', start='2020-01-01', end='2022-02-26')

# Calculate MA
stock_data['MA(10)'] = stock_data['Close'].rolling(window=10).mean()
stock_data['MA(50)'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA(200)'] = stock_data['Close'].rolling(window=200).mean()

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = abs(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

stock_data['RSI(14)'] = calculate_rsi(stock_data)

# Calculate MACD
def calculate_macd(data, fast_window=12, slow_window=26, signal_window=9):
    ema_fast = data['Close'].ewm(span=fast_window, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_window, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

stock_data['MACD'], stock_data['Signal'] = calculate_macd(stock_data)

# Charting
fig, axs = plt.subplots(4, 1, figsize=(12, 12))

axs[0].plot(stock_data['Close'], label='Close Price')
axs[0].plot(stock_data['MA(10)'], label='MA(10)')
axs[0].plot(stock_data['MA(50)'], label='MA(50)')
axs[0].plot(stock_data['MA(200)'], label='MA(200)')
axs[0].legend()

axs[1].plot(stock_data['RSI(14)'], label='RSI(14)')
axs[1].axhline(y=30, color='r', linestyle='--')
axs[1].axhline(y=70, color='g', linestyle='--')
axs[1].legend()

axs[2].plot(stock_data['MACD'], label='MACD')
axs[2].plot(stock_data['Signal'], label='Signal')
axs[2].legend()

axs[3].plot(stock_data['Close'], label='Close Price')
axs[3].plot(stock_data['Close'][stock_data['MACD'] > stock_data['Signal']], 'g^', label='Buy Signal')
axs[3].plot(stock_data['Close'][stock_data['MACD'] < stock_data['Signal']], 'rv', label='Sell Signal')
axs[3].legend()

plt.show()
```
This code retrieves the historical stock data for Apple (AAPL), calculates the MA, RSI, and MACD, and charts them using `matplotlib`. 
You can customize the code to suit your specific needs and experiment with different technical indicators and charting libraries.  

---  

# Complete Bash script to setup this project for Ubuntu Desktop using `pyenv virtualenv` running `Python 3.11` 

Here's a complete bash script to set up your project on Ubuntu Desktop with Python 3.11 using pyenv and virtualenv:

```bash
#!/bin/bash

# Install dependencies
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev sqlite3 libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl \
git

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to bashrc
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Reload bashrc
source ~/.bashrc

# Install Python 3.11
pyenv install 3.11.0

# Create virtual environment
pyenv virtualenv 3.11.0 myenv

# Activate virtual environment
pyenv activate myenv

# Install required packages
pip install yfinance pandas numpy matplotlib

# Create project directory
mkdir myproject
cd myproject

# Create Python script
echo "import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Retrieve stock data
stock_data = yf.download('AAPL', start='2020-01-01', end='2022-02-26')

# Calculate MA
stock_data['MA(10)'] = stock_data['Close'].rolling(window=10).mean()
stock_data['MA(50)'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA(200)'] = stock_data['Close'].rolling(window=200).mean()

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = abs(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

stock_data['RSI(14)'] = calculate_rsi(stock_data)

# Calculate MACD
def calculate_macd(data, fast_window=12, slow_window=26, signal_window=9):
    ema_fast = data['Close'].ewm(span=fast_window, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_window, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

stock_data['MACD'], stock_data['Signal'] = calculate_macd(stock_data)

# Charting
fig, axs = plt.subplots(4, 1, figsize=(12, 12))

axs[0].plot(stock_data['Close'], label='Close Price')
axs[0].plot(stock_data['MA(10)'], label='MA(10)')
axs[0].plot(stock_data['MA(50)'], label='MA(50)')
axs[0].plot(stock_data['MA(200)'], label='MA(200)')
axs[0].legend()

axs[1].plot(stock_data['RSI(14)'], label='RSI(14)')
axs[1].axhline(y=30, color='r', linestyle='--')
axs[1].axhline(y=70, color='g', linestyle='--')
axs[1].legend()

axs[2].plot(stock_data['MACD'], label='MACD')
axs[2].plot(stock_data['Signal'], label='Signal')
axs[2].legend()

axs[3].plot(stock_data['Close'], label='Close Price')
axs[3].plot(stock_data['Close'][stock_data['MACD'] > stock_data['Signal']], 'g^', label='Buy Signal')
axs[3].plot(stock_data['Close'][stock_data['MACD'] < stock_data['Signal']], 'rv', label='Sell Signal')
axs[3].legend()

plt.show()" > stock_analysis.py

# Run the script
python stock_analysis.py
```

Save this script as `setup.sh` and run it using `bash setup.sh`. 
This script will install the required dependencies, set up pyenv and virtualenv, install Python 3.11, create a virtual environment, install

## API Keys and Subscriptions

The code snippet I provided uses the `yfinance` library to retrieve stock data from Yahoo Finance. 
This library does not require an API key or subscription for most use cases.

However, if you plan to use this code for extensive or commercial purposes, you might need to consider the following:

- *Yahoo Finance API limitations*: While `yfinance` is a convenient library, 
  it's essential to respect Yahoo Finance's terms of service and usage limits. If you exceed these limits, your IP might be blocked.
- *Alternative data providers*: If you need more robust or reliable data, you might consider using alternative data providers like Quandl, Alpha Vantage, or Intrinio. These services often require API keys or subscriptions.

To use alternative data providers, you'll need to:

1. *Sign up for an account*: Create an account with the chosen data provider.
2. *Obtain an API key*: Get an API key or token to access the data.
3. *Modify the code*: Update the code to use the new data provider's API.

Some popular data providers and their requirements are:

- *Quandl*: Offers financial and economic data. Requires an API key for extensive use.
- *Alpha Vantage*: Provides stock, foreign exchange, and cryptocurrency data. 
                   Offers free API keys with limited usage; paid plans available for more extensive use.
- *Intrinio*: Offers financial data and analytics. Requires a subscription for access to premium data.

Before choosing a data provider, review their documentation, pricing, and terms of service to ensure they meet your needs.

---

# Jupyter Notebook for Stock Analysis

Here's a Jupyter notebook that includes the code for stock analysis using `yfinance`, `pandas`, `numpy`, and `matplotlib`:

```python
# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Retrieve stock data
stock_data = yf.download('AAPL', start='2020-01-01', end='2022-02-26')

# Calculate MA
stock_data['MA(10)'] = stock_data['Close'].rolling(window=10).mean()
stock_data['MA(50)'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA(200)'] = stock_data['Close'].rolling(window=200).mean()

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = abs(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

stock_data['RSI(14)'] = calculate_rsi(stock_data)

# Calculate MACD
def calculate_macd(data, fast_window=12, slow_window=26, signal_window=9):
    ema_fast = data['Close'].ewm(span=fast_window, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_window, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

stock_data['MACD'], stock_data['Signal'] = calculate_macd(stock_data)

# Charting
fig, axs = plt.subplots(4, 1, figsize=(12, 12))

axs[0].plot(stock_data['Close'], label='Close Price')
axs[0].plot(stock_data['MA(10)'], label='MA(10)')
axs[0].plot(stock_data['MA(50)'], label='MA(50)')
axs[0].plot(stock_data['MA(200)'], label='MA(200)')
axs[0].legend()

axs[1].plot(stock_data['RSI(14)'], label='RSI(14)')
axs[1].axhline(y=30, color='r', linestyle='--')
axs[1].axhline(y=70, color='g', linestyle='--')
axs[1].legend()

axs[2].plot(stock_data['MACD'], label='MACD')
axs[2].plot(stock_data['Signal'], label='Signal')
axs[2].legend()

axs[3].plot(stock_data['Close'], label='Close Price')
axs[3].plot(stock_data['Close'][stock_data['MACD'] > stock_data['Signal']], 'g^', label='Buy Signal')
axs[3].plot(stock_data['Close'][stock_data['MACD'] < stock_data['Signal']], 'rv', label='Sell Signal')
axs[3].legend()

plt.show()
```
To use this notebook, you'll need to have Jupyter installed in your Python environment. You can install it using `pip install jupyter`.

Once installed, you can create a new notebook and paste the above code into it. Run each cell to execute the code and display the results.

*Tips:*

- Make sure to install the required libraries (`yfinance`, `pandas`, `numpy`, and `matplotlib`) in your Python environment.
- You can customize the code to analyze different stocks or use different technical indicators.
- You can also use Jupyter's interactive features, such as widgets and visualizations, to enhance your analysis.

---

# Setup Script for CPU-Based Project (without Nvidia GPU)
Here's a setup script that installs the required dependencies for a CPU-based project:

```bash
#!/bin/bash

# Install dependencies
sudo apt update
sudo apt install -y build-essential python3-dev python3-pip

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to bashrc
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Reload bashrc
source ~/.bashrc

# Install Python 3.11
pyenv install 3.11.0

# Create virtual environment
pyenv virtualenv 3.11.0 myenv

# Activate virtual environment
pyenv activate myenv

# Install required packages
pip install yfinance pandas numpy matplotlib jupyter

# Create project directory
mkdir myproject
cd myproject

# Create Jupyter notebook
jupyter notebook --generate-config

# Run Jupyter notebook
jupyter notebook
```

This script installs the required dependencies, sets up pyenv and virtualenv, installs Python 3.11, creates a virtual environment, and installs the required packages. It also creates a project directory and runs Jupyter notebook.

*Note:* Make sure to run the script in a terminal or command prompt, and follow the prompts to complete the installation.

Once the script finishes running, you can access the Jupyter notebook by navigating to `http://localhost:8888` in your web browser.


# Python Project Ideas for Financial Planning, Analysis, and Management with AI/ML

Here are some Python project ideas that can help you improve your skills in financial planning, analysis, and management, while incorporating AI/ML techniques:

### Financial Analysis and Planning
1. *Portfolio Optimization*: Use libraries like `cvxpy` or `scipy` to optimize a portfolio based on risk tolerance, returns, and constraints.
2. *Financial Statement Analysis*: Use `pandas` and `matplotlib` to analyze financial statements, calculate ratios, and visualize trends.
3. *Budgeting and Forecasting*: Create a budgeting and forecasting tool using `pandas` and `numpy` to track expenses and predict future financial performance.

### AI/ML Applications in Finance
1. *Stock Price Prediction*: Use libraries like `scikit-learn`, `TensorFlow`, or `PyTorch` to build models that predict stock prices based on historical data.
2. *Credit Risk Assessment*: Develop a credit risk assessment model using `scikit-learn` or `TensorFlow` to predict the likelihood of loan defaults.
3. *Portfolio Risk Management*: Use `scipy` and `numpy` to calculate Value-at-Risk (VaR) and Expected Shortfall (ES) for a portfolio.

### Data Visualization and Reporting
1. *Dashboard Creation*: Use libraries like `dash` or `bokeh` to create interactive dashboards for financial data visualization and reporting.
2. *Financial Data Visualization*: Use `matplotlib` and `seaborn` to create visualizations of financial data, such as stock prices, trading volumes, and economic indicators.

### Other Ideas
1. *Automated Trading*: Use libraries like `zipline` or `backtrader` to develop and backtest automated trading strategies.
2. *Financial Natural Language Processing*: Use `NLTK` or `spaCy` to analyze financial text data, such as earnings calls or financial news articles.
3. *Cryptocurrency Analysis*: Use libraries like `ccxt` or `cryptocompare` to analyze cryptocurrency data and build trading strategies.

These projects can help you develop a range of skills in financial planning, analysis, and management, while incorporating AI/ML techniques. Remember to start with simpler projects and gradually move on to more complex ones as you gain experience and confidence.
