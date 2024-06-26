#pip install streamlit
#pip install kaggle

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stock_analysis import load_data, calculate_indicators, get_stock_list, predict
import os
# from kaggle.api.kaggle_api_extended import KaggleApi

# launch = False

# def download_dataset():
#     global launch
#     # Initialize Kaggle API client and authenticate using secrets
#     api = KaggleApi()
#     api.set_config_value('username', st.secrets["kaggle"]["username"])
#     api.set_config_value('key', st.secrets["kaggle"]["key"])
#     api.authenticate()
    
#     # Define the dataset and the path where files will be downloaded
#     dataset = 'borismarjanovic/price-volume-data-for-all-us-stocks-etfs'
#     path = 'dataset'

#     # Download the dataset
#     api.dataset_download_files(dataset, path=path, unzip=True)
#     launch = True

# Load data

# if st.sidebar.button('Get Data', type="primary"):
#     download_dataset()

# if launch:
directory = "dataset/Stocks"

combined_df = load_data(directory)
combined_df = calculate_indicators(combined_df)

# Streamlit app
st.title('Stock Market Analysis')
st.sidebar.header('Select a Stock')

# Get list of stocks
stock_list = get_stock_list(combined_df)
selected_stock = st.sidebar.selectbox('Stock', stock_list)

# Filter data for selected stock
stock_df = combined_df[combined_df['SourceFile'] == selected_stock]
# def load_stock_data(stock_name, directory="dataset/Stocks"):
#     file_path = os.path.join(directory, f"{stock_name}")
#     stock_df = pd.read_csv(file_path)
#     stock_df['Date'] = pd.to_datetime(stock_df['Date'])
#     stock_df = calculate_indicators(stock_df)
#     return stock_df




# stock_df = load_stock_data(selected_stock)

# plot_stock_data(stock_df)

# Display plots
st.header(f'{selected_stock} Stock Analysis')
st.subheader('Stock Price and Moving Averages')
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(stock_df['Date'], stock_df['Close'], label='Close Price')
ax.plot(stock_df['Date'], stock_df['MA50'], label='50-Day MA')
ax.plot(stock_df['Date'], stock_df['MA200'], label='200-Day MA')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

st.subheader('RSI Over Time')
fig, ax = plt.subplots(figsize=(14, 7))
sns.lineplot(data=stock_df, x='Date', y='RSI', ax=ax)
ax.axhline(30, linestyle='--', alpha=0.5, color='red')
ax.axhline(70, linestyle='--', alpha=0.5, color='green')
ax.set_xlabel('Date')
ax.set_ylabel('RSI')
st.pyplot(fig)


st.subheader('Correlation Matrix')
fig, ax = plt.subplots(figsize=(12, 8))
corr_df = stock_df[['Close', 'MA50', 'MA200', 'RSI', 'Lag1', 'Lag2', 'Lag3']].corr()
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
st.pyplot(fig)

st.subheader('Actual Stock Price Movement')
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(stock_df['Date'], stock_df['Close'], label='Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# model, scaler, y_train, predicted_prices = predict(stock_df)
# st.subheader('Actual vs Predicted Stock Price Movement')
# fig, ax = plt.subplots(figsize=(14, 7))
# ax.plot(stock_df.index[:len(y_train)], scaler.inverse_transform(y_train.reshape(-1, 1)), label='Actual Close Prices')
# ax.plot(stock_df.index[len(y_train):len(y_train) + len(predicted_prices)], predicted_prices, label='Predicted Close Prices', linestyle='--')
# ax.set_xlabel('Date')
# ax.set_ylabel('Price')
# ax.legend()
# st.pyplot(fig)

model, scaler, y_train, predicted_prices, train_dates, test_dates = predict(stock_df)

st.subheader('Actual vs Predicted Stock Price Movement')
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(train_dates, y_train, label='Actual Close Prices')
ax.plot(test_dates, predicted_prices, label='Predicted Close Prices', linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

