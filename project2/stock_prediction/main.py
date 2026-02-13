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
    page_title="📊 Stock Analysis App",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

github_url = "https://github.com/ejung2017/PythonProject/tree/main"

st.title("📊 Stock Analysis App")
st.write("Enter a ticker in the sidebar and click Load data and ARIMA Time Series Analysis. \n\nFor more information, please visit [link](%s)." % github_url)

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

    # Ensure SMA60 exists for cross strategy (use rolling fallback if not computed)
    if "SMA60" not in data:
        data["SMA60"] = data["Close"].rolling(window=60, min_periods=1).mean()

    # RSI14 fallback if needed
    if "RSI14" not in data or data["RSI14"].isna().all():
        delta = data["Close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(window=14, min_periods=14).mean()
        roll_down = down.rolling(window=14, min_periods=14).mean()
        rs = roll_up / roll_down
        data["RSI14"] = 100 - (100 / (1 + rs))

    # MACD fallback handled earlier; ensure columns exist
    if "MACD" not in data or "MACD_SIGNAL" not in data:
        ema12 = data["Close"].ewm(span=12, adjust=False).mean()
        ema26 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = ema12 - ema26
        data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()
        data["MACD_HIST"] = data["MACD"] - data["MACD_SIGNAL"]

    # Bollinger fallback ensure columns
    if not {"BB_middle", "BB_upper", "BB_lower"}.issubset(set(data.columns)):
        sma20 = data["Close"].rolling(window=20, min_periods=20).mean()
        std20 = data["Close"].rolling(window=20, min_periods=20).std()
        data["BB_middle"] = sma20
        data["BB_upper"] = sma20 + 2 * std20
        data["BB_lower"] = sma20 - 2 * std20

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

    # RSI plot
    fig_rsi, ax_rsi = plt.subplots(figsize=(10, 3))
    ax_rsi.plot(data.index, data["RSI14"], label="RSI14", color="purple")
    ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.7)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.7)
    ax_rsi.set_title(f"{ticker} RSI (14)")
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_xlabel("Date")
    ax_rsi.legend(loc="best")
    fig_rsi.tight_layout()
    st.pyplot(fig_rsi)

    # MACD plot (Close separated from MACD lines via a broken y-axis)
    # MACD + Signal + Histogram merged into one bottom panel, Close on top broken y-axis
    macd_series = pd.concat([data.get("MACD", pd.Series(dtype=float)),
                             data.get("MACD_SIGNAL", pd.Series(dtype=float))]).dropna()
    if macd_series.empty:
        macd_min, macd_max = -1.0, 1.0
    else:
        macd_min, macd_max = macd_series.min(), macd_series.max()

    close_min, close_max = data["Close"].min(), data["Close"].max()
    macd_range = macd_max - macd_min if (macd_max - macd_min) != 0 else 1.0
    close_range = close_max - close_min if (close_max - close_min) != 0 else 1.0

    macd_pad = max(macd_range * 0.1, 1e-6)
    close_pad = max(close_range * 0.02, 1e-3)

    tentative_macd_top = macd_max + macd_pad
    tentative_close_bottom = close_min - close_pad

    # Try 15$ rule if feasible, otherwise fallback to adaptive padding and enforce a small gap
    if (close_min - 2) - (macd_max + 2) > 0:
        macd_top = macd_max + 2
        close_bottom = close_min - 2
    else:
        macd_top = tentative_macd_top
        close_bottom = tentative_close_bottom
        if close_bottom <= macd_top:
            # enforce a small positive gap between macd_top and close_bottom
            close_bottom = macd_top + max(close_range * 0.02, 1.0)

    # build figure with 2 stacked rows: Close (zoomed, broken axis) and merged MACD+hist
    fig_macd = plt.figure(constrained_layout=True, figsize=(10, 5))
    gs = fig_macd.add_gridspec(2, 1, height_ratios=[1.6, 1.0])
    ax_close = fig_macd.add_subplot(gs[0, 0])
    ax_macd = fig_macd.add_subplot(gs[1, 0], sharex=ax_close)

    # top: Close (zoomed region)
    ax_close.plot(data.index, data["Close"], label="Close", color="black", alpha=0.9)
    ax_close.set_ylim(close_bottom, close_max + close_pad)
    ax_close.set_ylabel("Price")
    ax_close.legend(loc="upper left")
    plt.setp(ax_close.get_xticklabels(), visible=False)

    # bottom: merged MACD lines + histogram
    hist_vals = data.get("MACD_HIST", data.get("MACD", 0) - data.get("MACD_SIGNAL", 0))
    # bars (histogram)
    ax_macd.bar(data.index, hist_vals, color=np.where(hist_vals >= 0, "g", "r"), width=1, alpha=0.6, label="MACD Hist")
    # lines
    ax_macd.plot(data.index, data.get("MACD", pd.Series(dtype=float)), label="MACD", color="blue")
    ax_macd.plot(data.index, data.get("MACD_SIGNAL", pd.Series(dtype=float)), label="Signal", color="orange")
    ax_macd.set_ylim(macd_min - macd_pad, macd_top)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend(loc="upper left")

    # draw diagonal break markers between ax_close and ax_macd (visual break)
    d = .015  # size of diagonal lines in axes coords
    kwargs_close = dict(transform=ax_close.transAxes, color='k', clip_on=False)
    ax_close.plot((-d, +d), (-d, +d), **kwargs_close)
    ax_close.plot((1 - d, 1 + d), (-d, +d), **kwargs_close)
    kwargs_macd = dict(transform=ax_macd.transAxes, color='k', clip_on=False)
    ax_macd.plot((-d, +d), (1 - d, 1 + d), **kwargs_macd)
    ax_macd.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_macd)

    fig_macd.suptitle(f"{ticker} MACD")
    st.pyplot(fig_macd)

    # Bollinger Bands plot (overlay on price)
    fig_bb, ax_bb = plt.subplots(figsize=(10, 4))
    ax_bb.plot(data.index, data["Close"], label="Close", color="black")
    ax_bb.plot(data.index, data["BB_middle"], label="BB Middle (SMA20)", color="blue", linewidth=0.9)
    ax_bb.plot(data.index, data["BB_upper"], label="BB Upper", color="red", linestyle="--", linewidth=0.8)
    ax_bb.plot(data.index, data["BB_lower"], label="BB Lower", color="green", linestyle="--", linewidth=0.8)
    ax_bb.fill_between(data.index, data["BB_lower"], data["BB_upper"], color="gray", alpha=0.1)
    ax_bb.set_title(f"{ticker} Bollinger Bands (20)")
    ax_bb.set_xlabel("Date")
    ax_bb.set_ylabel("Price")
    ax_bb.legend(loc="best")
    fig_bb.tight_layout()
    st.pyplot(fig_bb)

    # SMA20-SMA60 cross strategy vs Buy & Hold
    # Create SMA20 if not present
    if "SMA20" not in data:
        data["SMA20"] = data["Close"].rolling(window=20, min_periods=1).mean()

    sig = pd.Series(0, index=data.index)
    sig[(data["SMA20"].shift(1) < data["SMA60"].shift(1)) & (data["SMA20"] >= data["SMA60"])] = 1
    sig[(data["SMA20"].shift(1) > data["SMA60"].shift(1)) & (data["SMA20"] <= data["SMA60"])] = -1
    position = sig.replace(to_replace=0, method="ffill").clip(-1, 1)

    ret = data["Close"].pct_change()
    strat_ret = position.shift(1) * ret
    cum_eq = (1 + strat_ret.fillna(0)).cumprod()
    buyhold = (1 + ret.fillna(0)).cumprod()

    fig_cross, ax_cross = plt.subplots(figsize=(10, 4))
    ax_cross.plot(cum_eq.index, cum_eq, label="SMA20-60 Strat")
    ax_cross.plot(buyhold.index, buyhold, label="Buy & Hold", linestyle="--")
    ax_cross.legend()
    ax_cross.set_title(f"{ticker} SMA20-SMA60 Cross Strategy vs Buy&Hold")
    ax_cross.set_xlabel("Date")
    ax_cross.set_ylabel("Equity Curve")
    fig_cross.tight_layout()
    st.pyplot(fig_cross)
