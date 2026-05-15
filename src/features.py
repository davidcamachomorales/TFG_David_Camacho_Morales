import pandas as pd
import numpy as np

# Library for time series
# Calculate the indicator
def calculate_sma(df, window):

    data = df.copy()

    data[f'TI_SMA_{window}_Feature'] = data['close'].rolling(window=window).mean()

    return data

def calculate_ema(df, window):
    # Exponential Moving Average

    data = df.copy()

    data[f'TI_EMA_{window}_Feature'] = data['close'].ewm(span=window, adjust=False).mean()

    return data

def calculate_rsi(df, window):
    # Relative Strength Index

    data = df.copy()

    delta = data['close'].diff()
    loss = (delta.where(delta < 0, 0))
    gain = (-delta.where(delta > 0, 0))
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = abs(avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    data[f'MI_RSI_{window}_Feature'] = rsi

    return data

def calculate_macd(df, short_window, long_window, signal_window):
    # Moving Average Convergence Divergence
    
    data = df.copy()

    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
    data[f'MI_MACD_{short_window}_{long_window}_Feature'] = short_ema - long_ema
    data[f'MI_Signal_Line_{short_window}_{long_window}_Feature'] = data[f'MI_MACD_{short_window}_{long_window}_Feature'].ewm(span=signal_window, adjust=False).mean()

    return data

def calculate_bollinger_bands(df, window, num_of_std):
    # Bollinger Bands
    data = df.copy()
    data[f'VolI_BB_Middle_Band_{window}_{num_of_std}_Feature'] = data['close'].rolling(window=window).mean()
    data[f'VolI_BB_Upper_Band_{window}_{num_of_std}_Feature'] = data[f'VolI_BB_Middle_Band_{window}_{num_of_std}_Feature'] + (data['close'].rolling(window=window).std() * num_of_std)
    data[f'VolI_BB_Lower_Band_{window}_{num_of_std}_Feature'] = data[f'VolI_BB_Middle_Band_{window}_{num_of_std}_Feature'] - (data['close'].rolling(window=window).std() * num_of_std)

    return data

def calculate_stochastic_oscillator(df, window):

    data = df.copy()

    data[f'MI_Stochastic_Osclilator_percK_{window}_Feature'] = ((data['close'] - data['low'].rolling(window=window).min()) / (data['high'].rolling(window=window).max() - data['low'].rolling(window=window).min())) * 100
    data[f'MI_Stochastic_Osclilator_percD_{window}_Feature'] = data[f'MI_Stochastic_Osclilator_percK_{window}_Feature'].rolling(window=3).mean()

    return data

def calculate_atr(df, window):
    # Average True Range
    data = df.copy()
    
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    data[f'VolI_ATR_{window}_Feature'] = tr.ewm(alpha=1/window, adjust=False).mean()

    return data

def calculate_lagged_features(df, window):
    # Lagged Features

    data = df.copy()
    
    # To create a new column in pandas we DO NOT use data = [...]
    # we directly use data[col_name] = what_it_calculates
    data[f'LF_Lag_{window}_Feature'] = data['close'].shift(window)

    return data

def calculate_differences(df, period):
    # Difference between the current price and the price from period days ago
    data = df.copy()
    data[f'DF_Diff_{period}_Feature'] = data['close'].diff(periods=period)
    
    return data

def calculate_rolling_volatility(df, window):
    # Moving standard deviation of closing price log-returns
    data = df.copy()
    log_returns = np.log(data['close'] / data['close'].shift(1))
    data[f'RV_Rolling_Vol_{window}_Feature'] = log_returns.rolling(window=window).std()

    return data

def calculate_temporal_decomposition(df, period):
    # Temporal Decomposition using seasonal_decompose
    from statsmodels.tsa.seasonal import seasonal_decompose
    data = df.copy()
    
    # To avoid errors in decomposition due to NaNs created earlier
    # we use fast linear interpolation only for this mathematical step
    close_series = data['close'].interpolate(method='linear')
    # If there are initial NaNs that linear interpolation doesn't fix, we backward or forward fill
    close_series = close_series.bfill().ffill()
    
    # freq = period in old versions, now it is called period
    decomposition = seasonal_decompose(close_series, model='additive', period=period)
    
    data[f'TD_Trend_{period}_Feature'] = decomposition.trend
    data[f'TD_Seasonal_{period}_Feature'] = decomposition.seasonal
    data[f'TD_Resid_{period}_Feature'] = decomposition.resid
    
    return data

def calculate_time_delay_embedding(df, delay, dimension):
    # Time Delay Embedding
    data = df.copy()
    
    # as many delay columns as indicated by dimension
    for d in range(1, dimension + 1):
        # each step goes back delay times
        data[f'TDE_Dim{d}_Delay{delay}_Feature'] = data['close'].shift(d * delay)
        
    return data

def compute_technical_indicators(df):
    # Compute all technical indicators

    data = df.copy()
    data = calculate_sma(data, 5)
    data = calculate_sma(data, 14)
    data = calculate_sma(data, 21)

    data = calculate_ema(data, 5)
    data = calculate_ema(data, 14)
    data = calculate_ema(data, 21)

    data = calculate_rsi(data, 5)
    data = calculate_rsi(data, 14)
    data = calculate_rsi(data, 21)

    
    data = calculate_macd(data, 12, 26, 9)
    data = calculate_macd(data, 8, 17, 9)
    data = calculate_macd(data, 13, 30, 10)

    data = calculate_bollinger_bands(data, 20, 1)
    data = calculate_bollinger_bands(data, 20, 2)
    data = calculate_bollinger_bands(data, 20, 3)

    data = calculate_stochastic_oscillator(data, 5)
    data = calculate_stochastic_oscillator(data, 14)
    data = calculate_stochastic_oscillator(data, 21)

    data = calculate_atr(data, 5)
    data = calculate_atr(data, 14)
    data = calculate_atr(data, 21)

    data = calculate_lagged_features(data, 5)
    data = calculate_lagged_features(data, 14)
    data = calculate_lagged_features(data, 21)

    data = calculate_differences(data, 1)
    data = calculate_differences(data, 5)

    data = calculate_rolling_volatility(data, 5)
    data = calculate_rolling_volatility(data, 14)
    data = calculate_rolling_volatility(data, 21)

    data = calculate_temporal_decomposition(data, 5)
    data = calculate_temporal_decomposition(data, 20)
    
    data = calculate_time_delay_embedding(data, delay=1, dimension=3)
    data = calculate_time_delay_embedding(data, delay=5, dimension=3)

    return data

def OHLC_features(df):
    # Compute OHLC features

    data = df.copy()

    # log returns
    data['close_log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['open_log_returns'] = np.log(data['open'] / data['open'].shift(1))
    data['high_log_returns'] = np.log(data['high'] / data['high'].shift(1))
    data['low_log_returns'] = np.log(data['low'] / data['low'].shift(1))

    # percentage change
    data['close_perc_change'] = data['close'].pct_change()
    data['open_perc_change'] = data['open'].pct_change()
    data['high_perc_change'] = data['high'].pct_change()
    data['low_perc_change'] = data['low'].pct_change()

    # high low range
    data['high_low_range'] = data['high'] - data['low']

    # close open range
    data['close_open_range'] = data['close'] - data['open']
    
    return data