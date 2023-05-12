from sklearn.preprocessing import MinMaxScaler

def normalize_feature(stock_df: pd.DataFrame, col: str):
    
    values = stock_df[col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(values)
    scaled = scaler.transform(values)
    return list(scaled.flatten())