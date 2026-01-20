import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from statsmodels.tsa.arima.model import ARIMA
from dateutil.relativedelta import relativedelta
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ===== DATA PREPARATION FUNCTION =====
@st.cache_data(ttl=3600)
def prepare_stock_data(ticker: str, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Handle both MultiIndex and single-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        if data.empty:
            return None, f"No data found for ticker {ticker}"
        
        data['Daily_Return'] = data['Close'].pct_change()

        data['SMA5'] = talib.SMA(data['Close'], timeperiod=5)
        data['SMA20'] = talib.SMA(data['Close'], timeperiod=20)

        data['EMA60'] = talib.EMA(data['Close'], timeperiod=60)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data["MACD"], data["MACD_SIGNAL"], data["MACD_HIST"] = talib.MACD(data["Close"])
        data['BB_Low'], data['BB_Mid'], data['BB_High'] = talib.BBANDS(data["Close"], timeperiod=20)
        data["ATR"] = talib.ATR(data['High'], data['Close'], data['Low'], timeperiod=14)
        data["ADX"] = talib.ADX(data["High"], data["Low"], data["Close"], timeperiod=14)
        data["OBV"] = talib.OBV(data["Close"], data["Volume"])

        # Momentum Indicators
        data['STOCH_K'], data['STOCH_D'] = talib.STOCH(
            data['High'], data['Low'], data['Close'],
            fastk_period=14,
            slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )

        data['CCI_14'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['CCI_20'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=20)

        data['WILLR_14'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)

        data['MOM_10'] = talib.MOM(data['Close'], timeperiod=10)
        data['MOM_14'] = talib.MOM(data['Close'], timeperiod=14)

        data['ROC_12'] = talib.ROC(data['Close'], timeperiod=12)

        # Trend Strength Indicators
        data['ADX_14'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['PLUS_DI'] = talib.PLUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['MINUS_DI'] = talib.MINUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)

        data['AROONOSC_14'] = talib.AROONOSC(data['High'], data['Low'], timeperiod=14)

        # Volume Indicators
        data['OBV'] = talib.OBV(data['Close'], data['Volume'])
        data['MFI_14'] = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)

        data['AD'] = talib.AD(data['High'], data['Low'], data['Close'], data['Volume'])

        data['ADOSC'] = talib.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], 
                                    fastperiod=3, slowperiod=10)

        # Other
        data['SAR'] = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)
        data['NATR_14'] = talib.NATR(data['High'], data['Low'], data['Close'], timeperiod=14)

        data['nxt_ret'] = data['Daily_Return'].shift(periods=-1)
        data['prev_close'] = data['Close'].shift(periods=1)
        data['prev_open'] = data['Open'].shift(periods=1)

        # Ensure SMA60 exists for cross strategy
        if "SMA60" not in data:
            data["SMA60"] = data["Close"].rolling(window=60, min_periods=1).mean()

        # Fallbacks for RSI14, MACD, Bollinger
        if "RSI14" not in data or data["RSI14"].isna().all():
            delta = data["Close"].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            roll_up = up.rolling(window=14, min_periods=14).mean()
            roll_down = down.rolling(window=14, min_periods=14).mean()
            rs = roll_up / roll_down
            data["RSI14"] = 100 - (100 / (1 + rs))

        if "MACD" not in data or "MACD_SIGNAL" not in data:
            ema12 = data["Close"].ewm(span=12, adjust=False).mean()
            ema26 = data["Close"].ewm(span=26, adjust=False).mean()
            data["MACD"] = ema12 - ema26
            data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()
            data["MACD_HIST"] = data["MACD"] - data["MACD_SIGNAL"]

        if not {"BB_middle", "BB_upper", "BB_lower"}.issubset(set(data.columns)):
            sma20 = data["Close"].rolling(window=20, min_periods=20).mean()
            std20 = data["Close"].rolling(window=20, min_periods=20).std()
            data["BB_middle"] = sma20
            data["BB_upper"] = sma20 + 2 * std20
            data["BB_lower"] = sma20 - 2 * std20

        # Create label
        data['label'] = 0
        data.loc[data['SMA5'] > data['SMA20'], 'label'] = 1
        data.loc[data['SMA5'] <= data['SMA20'], 'label'] = -1
        data['label'] = data['label'].shift(periods=-1)

        return data, None
    
    except Exception as e:
        return None, str(e)

def profit_loss_calculation(data, end_date): 
    one_month_ago = end_date + relativedelta(months=-1)
    one_month_df = data.loc[one_month_ago:]
    first_day_price = one_month_df['Close'][0]
    last_day_price = one_month_df['Close'][-1]
    profit_loss_pct = (last_day_price / first_day_price - 1)*100
    return profit_loss_pct

@st.cache_data(ttl=36000)
def top_model_selection(data, end_date): 
    tmr_data = data.tail(1)
    data = data.dropna()

    one_month_ago = end_date + relativedelta(months=-1)
    pred_month = data.loc[one_month_ago:]
    training = data.loc[:one_month_ago]

    #feature selection 
    x_col = ['prev_close', 'prev_open', 'SMA5', 'SMA20', 'EMA60', 'RSI', 'MACD', 'MACD_SIGNAL', "MACD_HIST", "BB_Low", "BB_Mid", "BB_High", "ATR", "ADX", "OBV", 'STOCH_K', 'STOCH_D', 'CCI_14', 'CCI_20',
        'WILLR_14', 'MOM_10', 'MOM_14', 'ROC_12', 'ADX_14', 'PLUS_DI',
        'MINUS_DI', 'AROONOSC_14', 'MFI_14', 'AD', 'ADOSC', 'SAR', 'NATR_14', "Daily_Return"]
    y_col = ['label']

    # Train data selection
    X = training[x_col]
    y = training[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pred_data_x = pred_month[x_col]
    pred_data_y = pred_month[y_col]

    result_dict = {}

    ## ML Models 
    #Decision Tree
    result_dict['Decision Tree'] = {}
    clf = DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_train, y_train)
    result_dict['Decision Tree']['test'] = clf.predict(X_test)
    result_dict['Decision Tree']['train'] = clf.predict(X_train)
    result_dict['Decision Tree']['tmr'] = clf.predict(tmr_data[x_col])
    result_dict['Decision Tree']['latest_month'] = clf.predict(pred_data_x)
    result_dict['Decision Tree']['status'] = "Buy 📈" if result_dict['Decision Tree']['tmr'] == 1 else "Sell 📉"

    #KNN
    result_dict['KNN'] = {}
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    result_dict['KNN']['test'] = knn.predict(X_test)
    result_dict['KNN']['train'] = knn.predict(X_train)
    result_dict['KNN']['tmr'] = knn.predict(tmr_data[x_col])
    result_dict['KNN']['latest_month'] = knn.predict(pred_data_x)
    result_dict['KNN']['status'] = "Buy 📈" if result_dict['KNN']['tmr'] == 1 else "Sell 📉"

    #Logistic Regression
    result_dict['Logistic Regression'] = {}
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    result_dict['Logistic Regression']['test'] = log_reg.predict(X_test)
    result_dict['Logistic Regression']['train'] = log_reg.predict(X_train)
    result_dict['Logistic Regression']['tmr'] = log_reg.predict(tmr_data[x_col])
    result_dict['Logistic Regression']['latest_month'] = log_reg.predict(pred_data_x)  # Remove [x_col]
    result_dict['Logistic Regression']['status'] = "Buy 📈" if result_dict['Logistic Regression']['tmr'][0] == 1 else "Sell 📉"

    #Random Forest
    result_dict['Random Forest'] = {}
    random = RandomForestClassifier(n_estimators=30,max_depth=5)
    random.fit(X_train, y_train)
    result_dict['Random Forest']['test'] = random.predict(X_test)
    result_dict['Random Forest']['train'] = random.predict(X_train)
    result_dict['Random Forest']['tmr'] = random.predict(tmr_data[x_col])
    result_dict['Random Forest']['latest_month'] = random.predict(pred_data_x)
    result_dict['Random Forest']['status'] = "Buy 📈" if result_dict['Random Forest']['tmr'] == 1 else "Sell 📉"

    #SVM
    result_dict['SVM'] = {}
    svm = SVC()
    svm.fit(X_train, y_train)
    result_dict['SVM']['test'] = svm.predict(X_test)
    result_dict['SVM']['train'] = svm.predict(X_train)
    result_dict['SVM']['tmr'] = svm.predict(tmr_data[x_col])
    result_dict['SVM']['latest_month'] = svm.predict(pred_data_x)
    result_dict['SVM']['status'] = "Buy 📈" if result_dict['SVM']['tmr'] == 1 else "Sell 📉"

    #GaussianNB
    result_dict['GaussianNB'] = {}
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    result_dict['GaussianNB']['test'] = svm.predict(X_test)
    result_dict['GaussianNB']['train'] = svm.predict(X_train)
    result_dict['GaussianNB']['tmr'] = svm.predict(tmr_data[x_col])
    result_dict['GaussianNB']['latest_month'] = svm.predict(pred_data_x)
    result_dict['GaussianNB']['status'] = "Buy 📈" if result_dict['GaussianNB']['tmr'] == 1 else "Sell 📉"

    # Collect qualifying models and sort by test accuracy descending
    qualifying_models = []
    for m in result_dict.keys():
        test_acc = accuracy_score(result_dict[m]['test'], y_test)
        train_acc = accuracy_score(result_dict[m]['train'], y_train)
        status = result_dict[m]['status']
        if train_acc - test_acc < 0.05 and test_acc > 0.8 and train_acc > 0.8:
            qualifying_models.append((m, test_acc, status))

    # Sort by test_acc descending
    qualifying_models.sort(key=lambda x: x[1], reverse=True)

    return qualifying_models, result_dict, y_test, y_train, pred_data_y

st.set_page_config(
    page_title="📊 Stock Prediction App",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Navigation state (landing / analysis) ---
if "page" not in st.session_state:
    st.session_state.page = "landing"

st.markdown("""
    <style>
    /* Sticky top bar with blur effect */
    .custom-topbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 56px;                    /* Match Streamlit's default header height */
        background: rgba(255, 255, 255, 0.8); /* Semi-transparent white */
        backdrop-filter: blur(10px);
        border-bottom: 1px solid #eee;
        display: flex;
        justify-content: flex-end;       /* Push content to the right */
        align-items: center;
        padding: 0 20px;
        z-index: 9999;
    }
    .custom-topbar a {
        background: #f0f2f6;
        color: #111;
        padding: 8px 16px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        font-size: 14px;
        margin-left: 12px;               /* Space from Deploy button */
    }
    .custom-topbar a:hover {
        background: #e0e2e6;
    }
    /* Optional: Push main content down so it's not hidden under the bar */
    .main .block-container {
        padding-top: 70px !important;
    }
    </style>

    <div class="custom-topbar">
        <a href="?page=landing">🏠 Home</a>
    </div>
    """, unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL):", value="")
START_default = pd.to_datetime("2010-01-01").date()
START = st.sidebar.date_input("Start date", value=START_default)
END = st.sidebar.date_input("End date", value=date.today())

# show Load button (remove the Back button)
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Load Data", use_container_width=True):
        st.session_state.page = "analysis"
        # st.query_params(page="analysis")
        st.rerun()
with col2:
    if st.session_state.page == "analysis":
        if st.button("Back", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()

# --- Navigation state (landing / analysis) ---
if "page" not in st.session_state:
    st.session_state.page = "landing"

if st.session_state.page == "landing":
    github_url = "https://github.com/ejung2017/PythonProject/tree/main"
    st.title("📊 Stock Prediction App")
    st.markdown("Enter a ticker in the sidebar and click **Load Data** button for Stock Prediction Results.")
    st.image("arima_predictive_modeling/ticker.png", caption="How to search for tickers", width='content')
    st.markdown("⚠️ Please note that **Yahoo Finance** may have some issues leading to ValueError. If so, please refresh the page and try again.")

    # US, Korea (Samsung), China (Tencent), France (LVMH), Japan (Toyota)
    top_companies_ticker = ['AAPL', 'GOOGL', '005930.KS', '0700.HK', 'MC.PA', '7203.T']

    START = "2020-01-01"
    END = date.today()
    
    # Combine all data into one dictionary
    company_data_all = {}
    for ticker_symbol in top_companies_ticker:
        data, error = prepare_stock_data(ticker_symbol, START, END)
        
        if data is not None:
            # Get company name
            company_info = yf.Ticker(ticker_symbol)
            company_name = company_info.info.get('shortName', 'N/A')
            
            # Get ML models
            qualifying_models, _, _, _, _ = top_model_selection(data, END)
            
            # Get P&L
            profit_loss_pct = profit_loss_calculation(data, END)
            
            # Store all in one dictionary
            company_data_all[ticker_symbol] = {
                'name': company_name,
                'qualifying_models': qualifying_models,
                'profit_loss_pct': profit_loss_pct,
                'prediction': qualifying_models[0][2] if qualifying_models else "No qualifying models"
            }
        else:
            company_data_all[ticker_symbol] = {
                'name': 'N/A',
                'qualifying_models': [],
                'profit_loss_pct': 0,
                'prediction': "Error loading data"
            }
    
    # Result Table 
    st.divider()
    st.subheader("Top Companies Predictions")
    stocks_df = pd.DataFrame({
        "Tickers": list(company_data_all.keys()),
        "Company": [company_data_all[tick]['name'] for tick in top_companies_ticker],
        "ML Prediction": [company_data_all[tick]['prediction'] for tick in top_companies_ticker],
        "Past 1 Month P/L (%)": [f"{company_data_all[tick]['profit_loss_pct']:.2f}%" for tick in top_companies_ticker]
    })
    st.dataframe(stocks_df.set_index("Tickers"), use_container_width=True, hide_index=False)

    # Credit 
    st.divider()
    st.markdown("For more information, please visit [link](%s)." % github_url)

elif st.session_state.page == "analysis":
    st.title("📊 Stock Prediction App")
    st.header(f"Machine Learning Predictions of {ticker}")
    
    # Rest of your analysis code continues here...
    st.write("""
    This app uses various Machine Learning Models to predict whether **tomorrow**'s stock price trend will be **up (Buy 📈)** or **down (Sell 📉)**.
    """)

    with st.expander("Here's a simple, step-by-step explanation of how the prediction is made"):
        st.write("""
    1. **Define the Target Ticker (What We Want to Predict)**  
    We create a label called `label` that tells us the future trend:
    - If the 5-day Simple Moving Average (SMA5) is **above** the 20-day SMA (SMA20) → this is a classic **golden cross**, a bullish signal → label = **1 (Up/Buy)**
    - If SMA5 is **below** SMA20 → bearish signal → label = **-1 (Down/Sell)**
    - We then **shift this label forward by 1 day** so that today's data predicts **tomorrow's** trend.

    2. **Select Features (Input Signals)**  
    - The model looks various significant features among the technical indicators and others 🤫 (Don't worry, feature engineering was done in the background)
             
    3. **Train the Model**  
    - We split the historical data into training (70%) and testing (30%) sets.
    - A **Decision Tree Classifier** (with max depth = 3) is trained on the training data.
    - Why Decision Tree? After testing multiple models, it showed the **best balance of accuracy** and **lowest overfitting** (similar performance on both training and unseen test data).

    4. **Make Tomorrow's Prediction**  
    - We feed today's latest indicator values into the trained model.
    - The model outputs **1** → predicts upward trend tomorrow → **Buy 📈**
    - Or **-1** → predicts downward trend → **Sell 📉**

    **Note**: This is a directional trend prediction (up or down), not an exact price forecast. No model can guarantee future results — always combine with your own research and risk management.

    Happy investing! 🚀
    """)
        
    if not ticker:
        st.error("Please enter a ticker symbol in the sidebar.")
        st.stop()
    
    # Prepare data
    data, error = prepare_stock_data(ticker, START, END)
    
    if error:
        st.error(f"Error processing ticker {ticker}: {error}")
        st.info("Please check that the ticker symbol is correct and try again.")
        st.stop()

    st.divider()
    st.subheader("Model Recommendations")
    st.markdown("📢 This section will only show ML Predictions that have accuracy greater than 80% and generalization gap less than 5%.")

    qualifying_models, result_dict, y_test, y_train, pred_data_y = top_model_selection(data, END)
    # Display predictions in sorted order
    for m, test_acc, _ in qualifying_models:
        st.markdown(f"**{m} Prediction**: {result_dict[m]['status']}")

    with st.expander("Click to see the details"):
        st.markdown("📝 Models sorted by test accuracy (highest to lowest)")
        all_models = []
        for m in result_dict.keys():
            test_acc = accuracy_score(result_dict[m]['test'], y_test)
            train_acc = accuracy_score(result_dict[m]['train'], y_train)
            all_models.append((m, test_acc, train_acc))

        # Sort by test_acc descending
        all_models.sort(key=lambda x: x[1], reverse=True)

        qualifying_names = [q[0] for q in qualifying_models]
        for m, test_acc, train_acc in all_models:
            if m in qualifying_names:
                st.markdown(f"**{m} Accuracy:** {test_acc*100:.2f}%")
                st.markdown(f"**{m} Train Set Accuracy:** {train_acc*100:.2f}%")
            else:
                st.write(f"{m} Accuracy: {test_acc*100:.2f}%")
                st.write(f"{m} Train Set Accuracy: {train_acc*100:.2f}%")
        
    st.divider()
    st.subheader("Model Reliability")

    # Evaluation
    # with st.expander("Click to see the evaluation details"): 이걸 쓸지 말지 모르겠음 
    for m in qualifying_names: 
        # Accuracy
        pred_acc = accuracy_score(pred_data_y, result_dict[m]['latest_month'])

        # Calculate precision, recall, F1 for test set
        precision_test = precision_score(y_test, result_dict[m]['test'], average='weighted')
        recall_test = recall_score(y_test, result_dict[m]['test'], average='weighted')
        f1_test = f1_score(y_test, result_dict[m]['test'], average='weighted')
        
        # For train set
        precision_train = precision_score(y_train, result_dict[m]['train'], average='weighted')
        recall_train = recall_score(y_train, result_dict[m]['train'], average='weighted')
        f1_train = f1_score(y_train, result_dict[m]['train'], average='weighted')
        
        # For latest month prediction
        precision_pred = precision_score(pred_data_y, result_dict[m]['latest_month'], average='weighted')
        recall_pred = recall_score(pred_data_y, result_dict[m]['latest_month'], average='binary')
        f1_pred = f1_score(pred_data_y, result_dict[m]['latest_month'], average='weighted')
        
        
        st.markdown(f"**{m}**:")
        st.markdown(f"""
        - Accuracy: {pred_acc*100:.2f}%
        - Precision: {precision_pred*100:.2f}%
        - Recall: {recall_pred*100:.2f}%
        - F1 Score: {f1_pred*100:.2f}%
        """)
        
        # Confusion Matrix for latest month prediction
        cm = confusion_matrix(pred_data_y, result_dict[m]['latest_month'])
        # st.markdown(f"**Confusion Matrix**:")
        
        # Create a pandas DataFrame for better readability
        cm_df = pd.DataFrame(
            cm,
            index=[f'True {label}' for label in [-1, 1]],
            columns=[f'Pred {label}' for label in [-1, 1]]
        )

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title(f"{m} Confusion Matrix")
        ConfusionMatrixDisplay.from_predictions(result_dict[m]['latest_month'], pred_data_y, display_labels=[-1, 1], ax=ax)
        st.pyplot(fig, width=450)
        
        st.markdown("\n")

    st.divider()

    # ARIMA Forecast
    st.subheader(f"ARIMA Forecast of {ticker}")
    series = data['Close']

    best_p, best_d, best_q = 5,5,5

    # ARIMA 
    model = ARIMA(series, order=(best_p, best_d, best_q))
    model_fit = model.fit()

    # get forecast (no confidence intervals)
    try:
        forecast_res = model_fit.get_forecast(steps=5)
        forecast_mean = forecast_res.predicted_mean
    except Exception:
        # fallback to simple forecast array if get_forecast isn't available
        forecast_mean = model_fit.forecast(steps=5)

    # build future dates (5 business days after last date)
    last_date = series.index[-1]
    future_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_mean))

    # create series with the new index
    forecast_series = pd.Series(forecast_mean.values, index=future_index)

    # extended series for plotting continuity
    extended = pd.concat([series, forecast_series])

    # Plot only the latest 1 month of historical Close + next 5 business days forecast
    one_month_ago = series.index[-1] - pd.Timedelta(days=30)
    start_plot = max(one_month_ago, series.index[0])
    hist_plot = series.loc[start_plot:series.index[-1]]

    # create a combined series so the line is continuous between last historical point and first forecast point
    combined = pd.concat([hist_plot, forecast_series]).sort_index()

    fig_arima, ax_arima = plt.subplots(figsize=(10, 4))
    # draw continuous line (connects across non-trading days)
    ax_arima.plot(combined.index, combined.values, label='Close (connected)', color='black', alpha=0.6)
    # emphasize recent historical region
    ax_arima.plot(hist_plot.index, hist_plot.values, label='Close (last 1 month)', color='black', linewidth=1.5)
    # highlight forecast region
    ax_arima.plot(forecast_series.index, forecast_series.values, label='ARIMA Forecast (5 days)', color='red', linestyle='--', linewidth=1.5)

    # set x-axis to show from start_plot through last forecast date
    ax_arima.set_xlim(start_plot, future_index[-1])
    ax_arima.legend()
    ax_arima.set_title(f"{ticker} — Last 1 month Close and 5-day ARIMA Forecast")
    ax_arima.set_xlabel("Date")
    ax_arima.set_ylabel("Price")
    fig_arima.autofmt_xdate()
    fig_arima.tight_layout()
    st.pyplot(fig_arima)

    # Show the 5-day forecast values below the chart
    forecast_df = pd.DataFrame({
        "Date": forecast_series.index.date,
        "Forecast Price": forecast_series.values
    })
    forecast_df["Forecast Price"] = forecast_df["Forecast Price"].round(2)
    st.subheader("5-day ARIMA Forecast")
    st.dataframe(forecast_df.set_index("Date"), use_container_width=True, hide_index=False)