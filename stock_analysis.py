
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to load data
def load_data(directory):
    dataframes = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    if not dataframes:
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_csv(file_path, skiprows=1, header=None, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt'])
                    df['SourceFile'] = file
                    dataframes.append(df)
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    continue
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df['Open'] = pd.to_numeric(combined_df['Open'])
    combined_df['High'] = pd.to_numeric(combined_df['High'])
    combined_df['Low'] = pd.to_numeric(combined_df['Low'])
    combined_df['Close'] = pd.to_numeric(combined_df['Close'])
    combined_df['Volume'] = pd.to_numeric(combined_df['Volume'])
    combined_df['OpenInt'] = pd.to_numeric(combined_df['OpenInt'])
    combined_df_sorted = combined_df.sort_values(by=['SourceFile', 'Date'], ignore_index=True)
    combined_df_sorted.fillna(method='ffill', inplace=True)
    combined_df_sorted.fillna(method='bfill', inplace=True)
    return combined_df_sorted

# Function to calculate technical indicators
def calculate_indicators(df):
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)
    return df

# Function to compute RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to plot stock prices and indicators
# def plot_stock(df, stock_name):
#     plt.figure(figsize=(14, 7))
#     plt.subplot(2, 1, 1)
#     plt.plot(df['Date'], df['Close'], label='Close Price')
#     plt.plot(df['Date'], df['MA50'], label='50-Day MA')
#     plt.plot(df['Date'], df['MA200'], label='200-Day MA')
#     plt.title(f'{stock_name} Stock Price and Moving Averages')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()

#     plt.subplot(2, 1, 2)
#     sns.lineplot(data=df, x='Date', y='RSI')
#     plt.axhline(30, linestyle='--', alpha=0.5, color='red')
#     plt.axhline(70, linestyle='--', alpha=0.5, color='green')
#     plt.title(f'{stock_name} RSI Over Time')
#     plt.xlabel('Date')
#     plt.ylabel('RSI')
#     plt.show()

# Function to get list of available stocks
def get_stock_list(df):
    return df['SourceFile'].unique()

# def predict(df):
#     df.dropna(inplace=True)
#     df = df.drop(columns=['SourceFile'])
#     df['Date'] = pd.to_datetime(df['Date'])  # Ensure the 'Date' column is datetime type
#     df.set_index('Date', inplace=True)  # Set 'Date' as the index

#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

#     seq_length = 60  # Use 60 days of historical data to predict the next day's price

#     def create_sequences(data, seq_length):
#         sequences = []
#         for i in range(len(data) - seq_length):
#             sequences.append(data[i:i + seq_length])
#         return np.array(sequences)

#     sequences = create_sequences(scaled_data, seq_length)

#     train_size = int(len(sequences) * 0.8)
#     train_sequences = sequences[:train_size]
#     test_sequences = sequences[train_size:]

#     # Split sequences into input and target
#     X_train = train_sequences[:, :-1]
#     y_train = train_sequences[:, -1]
#     X_test = test_sequences[:, :-1]
#     y_test = test_sequences[:, -1]

#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(seq_length-1, 1)))
#     model.add(LSTM(50))
#     model.add(Dense(1))

#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#     model.fit(X_train, y_train, epochs=20, batch_size=32)
#     predicted_prices = model.predict(X_test)
#     predicted_prices = scaler.inverse_transform(predicted_prices)

#     return model, scaler, y_train, predicted_prices

def predict(df):
    df.dropna(inplace=True)
    df = df.drop(columns=['SourceFile'])
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure the 'Date' column is datetime type
    df.set_index('Date', inplace=True)  # Set 'Date' as the index

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    seq_length = 60  # Use 60 days of historical data to predict the next day's price

    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    sequences = create_sequences(scaled_data, seq_length)

    train_size = int(len(sequences) * 0.8)
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]

    # Split sequences into input and target
    X_train = train_sequences[:, :-1]
    y_train = train_sequences[:, -1]
    X_test = test_sequences[:, :-1]
    y_test = test_sequences[:, -1]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length-1, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=20, batch_size=32)
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    y_train_inverse = scaler.inverse_transform(y_train.reshape(-1, 1))

    # Extract the actual dates for plotting
    train_dates = df.index[:len(y_train)]
    test_dates = df.index[len(y_train):len(y_train) + len(predicted_prices)]

    return model, scaler, y_train_inverse, predicted_prices, train_dates, test_dates


if __name__ == "__main__":
    directory = "content/Stocks"
    combined_df = load_data(directory)
    combined_df = calculate_indicators(combined_df)

    
    # plot_stock(combined_df[combined_df['SourceFile'] == 'googl.us.txt'], 'GOOGL')
