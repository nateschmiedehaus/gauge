import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from api_retrieval import retrieve_stock_data, retrieve_option_data
from scipy.stats import norm

# Define the investment goals and risk tolerance
target_returns = 0.05
max_loss = 0.10

# Retrieve stock data from Yahoo Finance API or other APIs
ticker = "AAPL"
try:
    stock_data = retrieve_stock_data(ticker)
    returns = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1))
    volatility = returns.std() * np.sqrt(252) # Annualized standard deviation
except:
    print(f"Could not retrieve data for {ticker}. Exiting program...")
    exit()

option_type = "call"
option_data = retrieve_option_data(ticker, exp_dates_count=4, option_type=option_type)["data"]
expiration_dates = retrieve_option_data(ticker, exp_dates_count=4, option_type=option_type)["expiration"]
if isinstance(option_chain, pd.DataFrame):
    iv = option_chain.calls.impliedVolatility
    max_iv_ticker = option_chain.calls.iloc[iv.idxmax()]["contractSymbol"]
    max_iv_value = iv.max()
    print(f"Max implied volatility for {ticker} on {expiration_dates[0]}: {max_iv_value:.2%} (Ticker: {max_iv_ticker})\n")
else:
    print("No option data available for any of the selected expiration dates. Exiting program...")
    exit()

option_strike = float(max_iv_ticker.split("_")[2])
option_price = option_chain.loc[option_chain["contractSymbol"] == max_iv_ticker]["lastPrice"].values[0]

# Calculate the log returns and volatility for the stock
prices = stock_data["Close"]
returns = np.log(prices / prices.shift(1))
volatility = returns.rolling(window=21).std() * np.sqrt(252)

# Set up the grid and gauge field
x = np.linspace(prices.index[0], prices.index[-1], 100)
y = np.linspace(option_strike * 0.8, option_strike * 1.2, 100)
X, Y = np.meshgrid(x, y)


def gauge_field(x, y, num_samples=1000):
    index = np.argmin(np.abs(prices.index - pd.Timestamp(x)))
    strike = option_strike
    option_returns = np.log(option_price / prices[-1])
    returns_arr = np.array([returns[index], option_returns])
    volatility_arr = np.array([volatility[index]])
    second_deriv_arr = np.gradient(np.gradient([np.log(prices[index]), np.log(strike), np.log(option_price)]))

        # Perform Monte Carlo simulation to estimate the probability distribution of the option value
    option_values = []
    for i in range(num_samples):
        # Generate random samples of returns and volatility
        r = np.random.normal(returns_arr, scale=volatility_arr)
        # Use the ARIMA model to generate a forecast of the future return
        forecast_return = ARIMA(r[0][:index], order=(2, 0, 1)).fit().forecast(steps=1).values[0]
        # Calculate the option value at the future point in time
        option_values.append(np.exp(r[1] + forecast_return + 0.5 * (second_deriv_arr[2] * (index / 252)**2)))

    # Calculate the mean and standard deviation of the option value distribution
    option_value_mean = np.mean(option_values)
    option_value_std = np.std(option_values)

    # Determine the buy and sell prices
    buy_price = option_value_mean - 2 * option_value_std
    sell_price = option_value_mean + 2 * option_value_std

    # Calculate the profit/loss and determine if the investment goals are met
    investment = 10000
    shares = np.floor(investment / option_price)
    purchase_cost = shares * option_price
    sale_proceeds = shares * sell_price
    profit = sale_proceeds - purchase_cost
    roi = profit / investment

    # Calculate the tax liability
    tax_rate = 0.2
    if profit > 0:
        capital_gains = profit * (1 - tax_rate)
    else:
        capital_gains = profit

    # Print the results
    if roi >= target_returns and capital_gains <= max_loss * investment:
        print(f"Buy {shares} contracts of {ticker} {expiration_dates[0]} {option_type} options at {option_price:.2f} each.")
        print(f"Place a limit sell order at {sell_price:.2f} for a potential profit of {profit:.2f} ({roi:.2%}) by {expiration_dates[0]}.")
        print(f"Expected capital gains after taxes: {capital_gains:.2f}")
    else:
        print(f"Do not invest in {ticker} {expiration_dates[0]} {option_type} options at this time.")

    # Plot the volatility and option price with the gauge field
    U = np.gradient(volatility)[-1]
    V = np.gradient(np.gradient(option_values))[-1]
    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, V, color='r')
    ax.set_xlabel("Date")
    ax.set_ylabel("Option Strike Price")
    ax.set_title(f"{ticker} {expiration_dates[0]} {option_type} Option Gauge Field")
    plt.show()
