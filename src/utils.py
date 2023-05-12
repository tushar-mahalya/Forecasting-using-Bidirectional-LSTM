import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_feature(stock_df: pd.DataFrame, col: str):
    values = stock_df[col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(values)
    scaled = scaler.transform(values)
    return list(scaled.flatten())


def predictor_response_split(df: pd.DataFrame, window_size: int, seq: str = 'Simple'):
    X = []
    y = []
    if seq == 'Multi':
        numpy_df = df.to_numpy()
        for i in range(len(numpy_df) - window_size):
            row = [r for r in numpy_df[i:i + window_size]]
            X.append(row)
            label = [numpy_df[i + window_size][0]]
            y.append(label)

    elif seq == 'Simple':
        numpy_df = df[['normal_close']].to_numpy()
        for i in range(len(numpy_df) - window_size):
            row = [r for r in numpy_df[i:i + window_size]]
            X.append(row)
            label = numpy_df[i + window_size]
            y.append(label)
    else:
        print('Wrong Choice of sequence.\nType either \'Simple\' or \'Multi\' in \"seq\" attribute.')

    return np.array(X), np.array(y)


def train_test_split(X: np.array, y: np.array, split_size: float):
    train_split = len(X) - int(len(X) * split_size)
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    return X_train, y_train, X_test, y_test


def train_test_data(df: pd.DataFrame, split_size: float = 0.2, window_size: int = 7, seq: str = 'Simple'):
    X, y = predictor_response_split(df, window_size, seq)
    X_train, y_train, X_test, y_test = train_test_split(X, y, split_size)

    return X_train, y_train, X_test, y_test
