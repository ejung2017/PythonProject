import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import StringIO  
from typing import Optional, Tuple

# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="📊 ARIMA Time Series Stock Analysis App",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

github_url = "https://github.com/ejung2017/PythonProject/tree/main"

st.title("📊 Stock Analysis App")
st.write("Enter a ticker in the sidebar and click Load data and ARIMA Time Series Analysis. \n\nFor more information, please visit [link](%s)." % github_url)

# Step 1: Fetch S&P 500 ticker list from Wikipedia (reliable source)
def get_sp500_tickers() -> list[str]:
    hardcoded = [
        'A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL',
        'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET',
        'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC',
        'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF-B', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BLDR', 'BMY', 'BR', 'BRK-B',
        'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE',
        'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP',
        'COF', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT', 'CTRA',
        'CTSH', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DAY', 'DE', 'DECK', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC',
        'DOV', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMR',
        'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR',
        'F', 'FANG', 'FAST', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FICO', 'FIS', 'FITB', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV',
        'GD', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS',
        'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST',
        'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR',
        'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC',
        'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW',
        'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET',
        'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS',
        'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW',
        'NRG', 'NSC', 'NTAP', 'NTRS', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS',
        'OXY', 'PANW', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD',
        'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'REG',
        'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SEE',
        'SHW', 'SIVB', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SPY', 'SRE', 'STE', 'STT', 'STX', 'STZ',
        'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO',
        'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER',
        'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VICI', 'VLO', 'VLTO', 'VRSK', 'VRSN', 'VRTX', 'VTR',
        'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY',
        'WYNN', 'XEL', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XOM', 'XYL', 'YUM', 'ZBH',
        'ZBRA', 'ZTS'
    ]
    tickers = [t.replace('.', '-') for t in hardcoded]
    print(f"Using hardcoded list: {len(tickers)} tickers.")
    return tickers

# Step 2: Function to get P/E, Market Cap, Revenue Growth for a ticker
@st.cache_data(ttl=60 * 60)
def get_stock_info(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        pe = info.get("trailingPE")
        market_cap = info.get("marketCap")
        name = info.get("longName") or info.get("shortName") or ticker

        if pe and pe > 0 and market_cap:
            return {
                "Ticker": ticker,
                "Company": name,
                "P/E Ratio": round(pe, 5),
                "Market Cap (B)": round(market_cap / 1_000_000_000, 2)  # in billions
            }
    except Exception:
        pass
    return None

def get_pe_ratio(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe = info.get('trailingPE')  # Trailing P/E (TTM)
        if pe is not None and pe > 0:  # Filter valid positive values
            return pe
    except:
        pass
    return None

@st.cache_data(ttl=60 * 60)
def get_revenue_growth_2025_vs_2024(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Robustly extract Total Revenue for 2024 and 2025 (if present) and compute 2025 vs 2024 YoY growth.
    Returns a small DataFrame with rows for 2024 and 2025 and the numeric growth for 2025 (percent).
    """
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        if financials is None or 'Total Revenue' not in financials.index:
            # no revenue data available
            return None, None

        revenue_row = financials.loc['Total Revenue']

        # Build mapping year -> revenue (take last available value for a given year)
        year_to_revenue: dict[int, float] = {}
        for col in revenue_row.index:
            try:
                year = pd.to_datetime(col).year
            except Exception:
                # fallback: try to parse first 4 digits
                try:
                    year = int(str(col)[:4])
                except Exception:
                    continue
            val = pd.to_numeric(revenue_row[col], errors='coerce')
            if pd.isna(val):
                continue
            # Keep the value (if multiple columns map to same year, override with latest encountered)
            year_to_revenue[year] = float(val)

        # Require both years
        if 2024 not in year_to_revenue or 2025 not in year_to_revenue:
            return None, None

        rev2024 = year_to_revenue[2024]
        rev2025 = year_to_revenue[2025]

        # avoid division by zero / invalid numbers
        if rev2024 == 0 or pd.isna(rev2024) or pd.isna(rev2025):
            return None, None

        growth = (rev2025 - rev2024) / rev2024 * 100.0
        growth_rounded = round(growth, 2)

        # Build a clean DataFrame
        df = pd.DataFrame({
            "Year": [2024, 2025],
            "Revenue (B USD)": [round(rev2024 / 1e9, 2), round(rev2025 / 1e9, 2)],
            "Growth (%)": [None, growth_rounded]
        })

        return df, growth_rounded

    except Exception as e:
        print(f"Error fetching revenue for {ticker}: {e}")
        return None, None
    
@st.cache_data(ttl=60 * 60)
def get_top_growth_companies(tickers: list) -> pd.DataFrame:
    """
    From a list of tickers, return a DataFrame of companies that have valid 2025 vs 2024 revenue growth,
    sorted by descending growth.
    """
    results = []
    for t in tickers:
        df, growth = get_revenue_growth_2025_vs_2024(t)
        if growth is None or df is None:
            continue
        # safe extraction of revenues
        try:
            rev2025 = df.loc[df["Year"] == 2025, "Revenue (B USD)"].iloc[0]
            rev2024 = df.loc[df["Year"] == 2024, "Revenue (B USD)"].iloc[0]
        except Exception:
            continue

        try:
            info = yf.Ticker(t).info
            company_name = info.get("longName") or info.get("shortName") or t
        except Exception:
            company_name = t

        results.append({
            "Ticker": t,
            "Company": company_name,
            "2025 Revenue (B USD)": rev2025,
            "2024 Revenue (B USD)": rev2024,
            "Growth 2025 vs 2024 (%)": growth
        })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df
    result_df = result_df.sort_values("Growth 2025 vs 2024 (%)", ascending=False).reset_index(drop=True)
    return result_df

# Step 3: Collect P/E for all tickers (this may take ~30-60 seconds for 500 tickers)
tickers = get_sp500_tickers()
pe_data = []
for ticker in tickers:
    pe = get_pe_ratio(ticker)
    if pe:
        pe_data.append({'Ticker': ticker, 'P/E Ratio': pe})
# Progress
progress = st.progress(0)
status = st.empty()
data = []

for i, t in enumerate(tickers):
    row = get_stock_info(t)
    if row:
        data.append(row)
    if (i + 1) % 10 == 0:
        progress.progress((i + 1) / len(tickers))
        status.text(f"Fetching data... {i+1}/{len(tickers)}")

progress.progress(1.0)

df = pd.DataFrame(data)

# Step 4: Convert to DataFrame, sort, and get top 10
st.subheader("Top 10 by Highest Trailing P/E Ratio")
top_pe = df.nlargest(10, "P/E Ratio")[["Ticker", "Company", "P/E Ratio"]].reset_index(drop=True)
st.dataframe(
    top_pe.style.format({"P/E Ratio": "{:,.5f}"}),
    use_container_width=True,
    hide_index=True,
)
st.subheader("Top 10 by Largest Market Cap")
top_cap = df.nlargest(10, "Market Cap (B)")[["Ticker", "Company", "Market Cap (B)"]].reset_index(drop=True)
st.dataframe(
    top_cap.style.format({"Market Cap (B)": "{:,.2f}"}),
    use_container_width=True,
    hide_index=True,
)
st.subheader("Top 10 by Revenue Growth: 2025 vs 2024")
result_df = get_top_growth_companies(tickers)
df2 = pd.DataFrame(result_df)
top_revenue_growth = df2.nlargest(10, "Growth 2025 vs 2024 (%)")[["Ticker", "Company", "2024 Revenue (B USD)", "2025 Revenue (B USD)", "Growth 2025 vs 2024 (%)"]].reset_index(drop=True)
st.dataframe(
    top_revenue_growth.style.format({
        "2024 Revenue (B USD)": "{:,.2f}",
        "2025 Revenue (B USD)": "{:,.2f}",
        "Growth 2025 vs 2024 (%)": "{:+.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

# Sidebar inputs
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL):", value="")
START_default = pd.to_datetime("2024-01-01").date()
START = st.sidebar.date_input("Start date", value=START_default)
END = st.sidebar.date_input("End date", value=date.today())
pred_days = st.sidebar.radio("Stock price prediction after how many days?", ["1 Day", "15 Days", "1 Month"])
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
    ax_macd.legend(loc="lower left")

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

    # SMA20-SMA60 cross strategy vs Buy & Holds
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

    # ARIMA Forecast
    st.subheader(f"ARIMA Forecast of {ticker} after {pred_days}")
    series = data['Close']
    
    model = pm.auto_arima(
        series,  # Your time series data
        start_p=1, max_p=5,  # Range for p (autoregressive order)
        start_q=1, max_q=5,  # Range for q (moving average order)
        d=None, max_d=2,  # Order of differencing (auto-determined if None)
        seasonal=False,  # Set to True for SARIMA, then define P, D, Q, m
        information_criterion='aic',  # Criterion for model selection
        trace=True,  # Print model fitting progress
        error_action='ignore',  # Ignore errors during model fitting
        suppress_warnings=True,  # Suppress warnings
        stepwise=True,  # Use stepwise search for efficiency
        n_jobs=-1  # Use all available cores for parallel processing
    )

    best_p, best_d, best_q = model.order
    st.write(f"Best (p, d, q) orders: {best_p, best_d, best_q}")

    # ARIMA (p=1, d=1, q=1)
    model = ARIMA(series, order=(best_p, best_d, best_q))
    model_fit = model.fit()

    # Forecast
    if pred_days == "1 Day": 
        forecast = model_fit.forecast(steps=1)
    elif pred_days == "15 Days": 
        forecast = model_fit.forecast(steps=15)
    else: 
        forecast = model_fit.forecast(steps=30)
    st.write(f"The stock price after {pred_days} is forecasted to be ${forecast.iloc[0]:.2f}")

    # # Calculate MAE
    # mae = mean_absolute_error(series, forecast)
    # st.write(f"Mean Absolute Error (MAE): {mae:.3f}")

    # # Calculate MSE
    # mse = mean_squared_error(series, forecast)
    # st.write(f"Mean Squared Error (MSE): {mse:.3f}")

    # # Calculate RMSE
    # rmse = np.sqrt(mse)
    # st.write(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

    # # Calculate MAPE (handling potential division by zero for actual_values == 0)
    # def calculate_mape(y_true, y_pred):
    #     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # mape = calculate_mape(series, forecast)
    # st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.3f}%")