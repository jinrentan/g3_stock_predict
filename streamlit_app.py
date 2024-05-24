#pip install streamlit
#pip install kaggle

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stock_analysis import load_data, calculate_indicators, plot_stock, get_stock_list

# Load data
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
