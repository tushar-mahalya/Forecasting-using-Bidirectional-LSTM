# Scientific Calculation
import pandas as pd
import numpy as np

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


def data_accumilator(src_location: str, stocks_lst: list):
    preprocessed_data_path = {}

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

            path = os.path.join(parent_dir, 'Stocks Data', ticker, f'Normalized_{ticker}.csv')
            preprocessed_data_path[ticker] = path

        print("Directory named \"Stocks Data\" with required data is downloaded successfully at \"{}\"".format(
            src_location))

        return preprocessed_data_path