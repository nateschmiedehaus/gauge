import yfinance as yf
import requests
import finnhub
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from polygon import RESTClient

# Import API keys
from api_keys import api_keys

# Define the number of expiration dates to retrieve from the APIs
exp_dates_count = 4

# Retrieve stock data from Yahoo Finance API
def retrieve_stock_data(ticker, api="yahoo"):
    if api == "yahoo":
        return yf.download(ticker)
    elif api == "polygon":
        client = RESTClient(api_keys["polygon"]["api_key"])
        resp = client.stocks_equities_aggregates(ticker, 1, "day", "2022-03-01", "2023-03-01", unadjusted=False)
        df = pd.DataFrame(resp.results)
        df.index = pd.to_datetime(df['t'], unit='ms')
        return df[['c', 'o', 'h', 'l', 'v']]
    else:
        print(f"Unsupported API: {api}. Exiting program...")
        exit()

# Retrieve option data from Yahoo Finance, Alpha Vantage, or Finnhub APIs
# Retrieve option data from Yahoo Finance, Alpha Vantage, or Finnhub APIs
def retrieve_option_data(ticker, exp_dates_count, option_type):
    expiration_dates = yf.Ticker(ticker).options[:exp_dates_count]
    option_chains = []

    for exp_date in expiration_dates:
        try:
            option_chain = yf.Ticker(ticker).option_chain(str(exp_date))[option_type]
        except Exception as e:
            print(f"Error retrieving option data for {ticker} on {exp_date} from Yahoo Finance API: {e}")
            try:
                url = f"https://www.alphavantage.co/query?function=OPTION_CHAIN&symbol={ticker}&apikey={api_keys['alpha_vantage']}&expiry={exp_date}"
                response = requests.get(url)
                data = response.json()["optionChain"]["result"][0][option_type]
                option_chain = pd.DataFrame.from_dict(data)
            except Exception as e:
                print(f"Error retrieving option data for {ticker} on {exp_date} from Alpha Vantage API: {e}")
                try:
                    finnhub_api_key = api_keys["finnhub"]
                    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
                    res = finnhub_client.options_chain(ticker, exp_date)
                    option_chain = pd.DataFrame(res[option_type])
                except Exception as e:
                    print(f"Error retrieving option data for {ticker} on {exp_date} from Finnhub API: {e}")
                    option_chain = None

        if isinstance(option_chain, pd.DataFrame):
            option_chains.append(option_chain)

    if len(option_chains) == 0:
        print(f"No option data available for any of the selected expiration dates. Exiting program...")
        exit()

    option_data = pd.concat(option_chains)
    option_data = option_data[["strike", "lastPrice", "impliedVolatility", "delta", "gamma", "theta", "vega"]]
    option_data = option_data.rename(columns={"lastPrice": "Option Price", "impliedVolatility": "IV", "strike": "Strike", "delta": "Delta", "gamma": "Gamma", "theta": "Theta", "vega": "Vega",})

    return {"data": option_data, "expiration": expiration_dates}


