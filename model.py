
# DEPENDENCIES
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
from pprint import pprint
import csv

# LOG FUNCTION
def log(message: str) -> None:
    if False:
        pprint(message)

# SHAPIRO-WILK TEST FOR NORMALITY
def is_norm(data: pd.DataFrame, column: str) -> bool:
    result = stats.anderson(data[column], dist='norm')
    # print(f"Test Statistic: {result.statistic:.3f}")
    # print(f"Critical Values: {result.critical_values}")
    # print(f"Significance Levels: {result.significance_level}")
    for significance, critical_value in zip(result.significance_level, result.critical_values):
        decision = "Reject H₀" if result.statistic > critical_value else "Fail to Reject H₀"
        if decision == "Fail to Reject H₀":
            # print(f"At {significance}% significance level, {decision}. Data is normally distributed.")
            return True
    return False
    
# CALULATES TCDF FOR COLUMN (1-sample t-test)
def calc_probability(data: pd.DataFrame, column: str) -> tuple[float, float]:
    df = len(data[column]) - 1  # Degrees of freedom
    sample_mean = np.mean(data[column])
    sample_std = np.std(data[column], ddof=1) 
    t_statistic = (sample_mean - 0) / (sample_std / np.sqrt(len(data)))
    cumulative_probability = stats.t.cdf(t_statistic, df) # Probaility of observing a value less than or equal to the sample mean
    return cumulative_probability, 1 - cumulative_probability
    
# GETS ALL TICKERS IN THE NASDAQ STOCKS DIRECTORY
def get_all_ticker_data() -> list[str]:
    tickers = []
    for i in range(1, 4):
        directory = f'data/daily/us/nasdaq stocks/{i}'
        for filename in os.listdir(directory):
            if filename.endswith('.us.txt'):
                tickers.append(os.path.join(directory, filename))
    for i in range(1, 3):
        directory = f'data/daily/us/nyse etfs/{i}'
        for filename in os.listdir(directory):
            if filename.endswith('.us.txt'):
                tickers.append(os.path.join(directory, filename))
    for i in range(1, 3):
        directory = f'data/daily/us/nyse stocks/{i}'
        for filename in os.listdir(directory):
            if filename.endswith('.us.txt'):
                tickers.append(os.path.join(directory, filename))
    directory = f'data/daily/us/nasdaq etfs'
    for filename in os.listdir(directory):
        if filename.endswith('.us.txt'):
            tickers.append(os.path.join(directory, filename))
    directory = f'data/daily/us/nysemkt stocks'
    for filename in os.listdir(directory):
        if filename.endswith('.us.txt'):
            tickers.append(os.path.join(directory, filename))
    return tickers

# SEARCHES FOR TICKER DATA IN THE NASDAQ STOCKS DIRECTORY
def search_ticker_data(ticker: str) -> str | None:
    for i in range(1, 4):
        directory = f'data/daily/us/nasdaq stocks/{i}'
        for filename in os.listdir(directory):
            if filename.startswith(ticker) and filename.endswith('.us.txt'):
                return os.path.join(directory, filename)
    return None

# WRITES STOCK DATA TO A FILE
def write_stock_data(data: list[dict], file_path: str) -> str:
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader() 
        writer.writerows(data)
    return file_path

# LOADS STOCK DATA FROM A FILE
def load_stock_data(file_path: str, filter: int) -> pd.DataFrame | None:
    try:
        data = pd.read_csv(file_path)
        data['<PCT_CHANGE>'] = data['<CLOSE>'].pct_change()
        data['<VAL_CHANGE>'] = data['<CLOSE>'].diff()
        if len(data.dropna()) > filter:
            return data.dropna()
        else:
            log(f"No valid data found in {file_path}.")
            return None
    except Exception as e:
        log(f"Error loading data from {file_path}: {e}")
        return None
    
# CALCULATES AVERAGE GAIN ON DAYS OF GAIN
def calc_avg_gain(data: pd.DataFrame, column: str) -> float | None:
    gain_days = data[data[column] > 0]
    if not gain_days.empty:
        return gain_days[column].mean()
    else:
        return 0.0
    
# CALCULATES AVERAGE LOSS ON DAYS OF LOSS
def calc_avg_loss(data: pd.DataFrame, column: str) -> float | None:
    loss_days = data[data[column] < 0]
    if not loss_days.empty:
        return loss_days[column].mean()
    else:
        return 0.0

# PROCESSES DATA FOR ALL TICKERS AND RETURN VIABLE STOCKS TO BUY
def process_ticker_data(file_path: str) -> dict:

    data = [] # Container for viable stocks
    special = [] # Container for special stocks
    orders = [] # Container for orders to execute
    for ticker_path in get_all_ticker_data():

        # Load stock data for the ticker
        stock_data_full = load_stock_data(file_path=ticker_path, filter=365) # Full data
        if stock_data_full is None:
            continue
        stock_data_full = stock_data_full.sort_values(by='<DATE>', ascending=True) # Sort by date
        stock_data_7 = stock_data_full.tail(7) # Last week of data
        stock_data_30 = stock_data_full.tail(30) # Last month of data
        stock_data_90 = stock_data_full.tail(90) # Last 3 months of data
        stock_data_365 = stock_data_full.tail(365) # Last year of data
        # stock_data_1095 = stock_data_full.tail(1095) # Last 3 years of data
        stock_datasets = [stock_data_full, stock_data_7, stock_data_30, stock_data_90, stock_data_365]
        close_price = stock_data_full['<CLOSE>'].values[-1] # Get the last close price

        # Test stock data for normality and process if valid
        stock_object = {"ticker_path": ticker_path}
        stock_object["profit_percent"] = {}
        stock_object["norm_check"] = {}
        stock_object["avg_gain"] = {}
        stock_object["avg_loss"] = {}
        stock_object["prob_weighted_gain"] = {}
        stock_object["prob_weighted_loss"] = {}
        stock_object["prob_weighted_change"] = {}
        for stock_data in stock_datasets:
            stock_object["norm_check"][stock_data.shape[0]] = is_norm(stock_data, '<PCT_CHANGE>')
            stock_object["profit_percent"][stock_data.shape[0]] = calc_probability(stock_data, '<PCT_CHANGE>')
            stock_object["avg_gain"][stock_data.shape[0]] = calc_avg_gain(stock_data, '<VAL_CHANGE>')
            stock_object["avg_loss"][stock_data.shape[0]] = calc_avg_loss(stock_data, '<VAL_CHANGE>')
            stock_object["prob_weighted_gain"][stock_data.shape[0]] = stock_object["profit_percent"][stock_data.shape[0]][1] * stock_object["avg_gain"][stock_data.shape[0]]
            stock_object["prob_weighted_loss"][stock_data.shape[0]] = stock_object["profit_percent"][stock_data.shape[0]][0] * stock_object["avg_loss"][stock_data.shape[0]]
            stock_object["prob_weighted_change"][stock_data.shape[0]] = stock_object["prob_weighted_gain"][stock_data.shape[0]] + stock_object["prob_weighted_loss"][stock_data.shape[0]]
        data.append(stock_object)

        # Tests if all timeframes are normal
        if all(stock_object["norm_check"].values()):

            # Creates agggregate stock object
            for key in stock_object.keys():
                if key not in ['ticker_path', 'norm_check']:
                    stock_object[key] = np.mean(list(stock_object[key].values()))

            if stock_object["prob_weighted_change"] > 0:
                orders.append({
                    "ticker_path": stock_object["ticker_path"],
                    "prev_day_close_price": close_price,
                    "upper_bound": close_price + stock_object["prob_weighted_gain"],
                    "lower_bound": close_price + stock_object["prob_weighted_loss"],
                    "profit_prob": stock_object["profit_percent"],
                    "expected_yield": (stock_object["prob_weighted_change"] / close_price) * 100,
                })
            special.append(stock_object)

    # Sort special stocks by set of factors
    sorted_data = list(reversed(sorted(orders, key=lambda x: (x["profit_prob"], x["expected_yield"]))))
    return write_stock_data(data=sorted_data, file_path=file_path)

# log(data)
print(process_ticker_data("special_stocks.csv"))

