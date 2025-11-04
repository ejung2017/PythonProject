import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Stock Analysis App",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Stock Analysis App")
st.write("Enter a ticker in the sidebar and click Load data. This app shows price, returns, indicators and a simple classification demo.")

# Sidebar inputs
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL):", value="")
START_default = pd.to_datetime("2024-01-01").date()
START = st.sidebar.date_input("Start date", value=START_default)
END = st.sidebar.date_input("End date", value=date.today())
load_btn = st.sidebar.button("Load data")


if load_btn: 
    data = yf.download(ticker, start=START, end=END) 

    data.columns = data.columns.droplevel('Ticker')
    data['Daily Return'] = data['Close'].pct_change()

    data["SMA20"] = talib.SMA(data["Close"], timeperiod=20)
    data["EMA60"] = talib.EMA(data["Close"], timeperiod=60)
    data["RSI14"] = talib.RSI(data["Close"], timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    data["MACD"], data["MACD_SIGNAL"], data["MACD_HIST"] = macd, macd_signal, macd_hist
    upper, middle, lower = talib.BBANDS(data["Close"], timeperiod=20)
    data["BB_upper"], data["BB_middle"], data["BB_lower"] = upper, middle, lower
    data["ATR"] = talib.ATR(data["High"], data["Low"], data["Close"], timeperiod=14)

    # Show data summary
    st.subheader(f"{ticker} — Latest data")
    st.dataframe(data.tail(10))

    # Plots
    st.subheader("Price and Indicators")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(data['Close'], label='Close')
    if "SMA20" in data:
        ax[0].plot(data['SMA20'], label='SMA20')
    if "EMA60" in data:
        ax[0].plot(data['EMA60'], label='EMA60')
    ax[0].set_ylabel("Price")
    ax[0].legend(loc='best')

    ax[1].plot(data['Daily Return'], label='Daily Return', color='tab:orange')
    ax[1].set_ylabel("Daily Return")
    ax[1].legend(loc='best')

    fig.tight_layout()
    st.pyplot(fig)
