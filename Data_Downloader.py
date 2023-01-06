# Scientific Calculation
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.preprocessing import MinMaxScaler

# Financial Data
import yfinance as yf

# File Management
import os
import shutil


def stock_data(ticker: str):
    ticker = ticker
    stock_df = yf.download(ticker + '.NS',
                           start='2021-01-01',
                           end='2022-12-31',
                           progress=False)
    return stock_df.reset_index()


def money_flow_index(stock_df: pd.DataFrame, period: int = 14):
    # Fetching Data from Yahoo Finance API
    stock_df = stock_df

    # Calculate typical price
    typical_price = (stock_df['Close'] + stock_df['High'] + stock_df['Low']) / 3
    # typical_price

    # Get the period
    period = period

    # Calculate the money flow
    money_flow = typical_price * stock_df['Volume']

    # Get all the positive and negative money flows

    positive_flow = []
    negative_flow = []

    # Loop through the typical price
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i - 1]:
            positive_flow.append(money_flow[i - 1])
            negative_flow.append(0)

        elif typical_price[i] < typical_price[i - 1]:
            negative_flow.append(money_flow[i - 1])
            positive_flow.append(0)

        else:
            positive_flow.append(0)
            negative_flow.append(0)

    # Get all the positive and negative money flows within the time period

    positive_mf = []
    negative_mf = []

    for i in range(period - 1, len(positive_flow)):
        positive_mf.append(sum(positive_flow[i + 1 - period: i + 1]))

    for i in range(period - 1, len(negative_flow)):
        negative_mf.append(sum(negative_flow[i + 1 - period: i + 1]))

    # Calculate the money flow index
    MFI = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))

    # Filling NaN values to starting Period
    mfi_ext = []
    for i in range(period):
        mfi_ext.append(np.NaN)
    mfi_ext.extend(MFI)

    return mfi_ext


def log_return(stock_df: pd.DataFrame):
    # Fetching Data from Yahoo Finance API
    stock_df = stock_df

    log_rtn = np.log(stock_df['Adj Close']) - np.log(stock_df['Adj Close'].shift(1))

    return list(log_rtn)


def normalize_feature(stock_df: pd.DataFrame, col: str):
    values = stock_df[col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(values)
    scaled = scaler.transform(values)
    return list(scaled.flatten())


def data_accumilator(src_location: str, stocks_lst: list):
    parent_dir = src_location
    root_dir = os.path.join(parent_dir, 'Stocks Data')

    if os.path.exists(root_dir):
        print("Data is already present in the source location.")

    else:
        os.mkdir(root_dir)

        for ticker in stocks_lst:
            stock_df = stock_data(ticker)
            norm_stock = stock_df.copy()

            # Calculating Money-Flow Index and Logarithmic Return
            # for the underlying stock
            mfi = money_flow_index(stock_df, 14)
            returns = log_return(stock_df)

            norm_stock['MFI'] = mfi
            norm_stock['Returns'] = returns

            # Normalizing Values
            norm_close = normalize_feature(norm_stock, 'Adj Close')
            norm_mfi = normalize_feature(norm_stock, 'MFI')
            norm_returns = normalize_feature(norm_stock, 'Returns')

            norm_stock['normal_close'] = norm_close
            norm_stock['normal_mfi'] = norm_mfi
            norm_stock['normal_returns'] = norm_returns

            # Finalizing Pre-processed Data
            norm_stock = norm_stock[['Date', 'normal_close', 'normal_mfi', 'normal_returns']][14:]
            norm_stock.set_index('Date', drop=True, inplace=True)

            stock_df.to_csv(f'{ticker}.csv', index=False)
            norm_stock.to_csv(f'Normalized_{ticker}.csv', index=False)

            folder_path = os.path.join(parent_dir, ticker)
            simple_file_path = os.path.join(parent_dir, f'{ticker}.csv')
            normalized_file_path = os.path.join(parent_dir, f'Normalized_{ticker}.csv')

            os.mkdir(folder_path)
            shutil.move(simple_file_path, folder_path)
            shutil.move(normalized_file_path, folder_path)
            shutil.move(folder_path, root_dir)

        print("Directory named \"Stocks Data\" with required data is downloaded successfully at \"{}\"".format(
            src_location))


if __name__ == '__main__':
    # print_hi('PyCharm')
    nifty = pd.read_csv('nifty50.csv')
    nifty_tickers = nifty['Symbol'].sample(1)
    data_accumilator('/home/tushar_sharma/PycharmProjects/data_downloader', nifty_tickers)
