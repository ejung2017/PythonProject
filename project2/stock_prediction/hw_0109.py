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
from dateutil.relativedelta import relativedelta
from datetime import date
import datetime


import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree

from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


ticker = "AAPL"
start_date = "2020-01-01"
# end_date = date.today()
end_date = "2026-01-01"

data = yf.download(ticker, start=start_date, end=end_date)
data.columns = data.columns.droplevel(1)

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


data['Daily_Return'] = data['Close'].pct_change()
data['nxt_ret'] = data['Daily_Return'].shift(periods=-1)
data['prev_close'] = data['Close'].shift(periods=1)
data['prev_open'] = data['Open'].shift(periods=1)

data = data.dropna()

# making the label (depedent y-variable) to be golden cross (continuously going up as SMA5 is greater than SMA20)
data['label'] = 0
data.loc[data['SMA5'] > data['SMA20'], 'label'] = 1
data.loc[data['SMA5'] < data['SMA20'], 'label'] = -1
data['label'] = data['label'].shift(periods=-1) 
data = data.dropna()
x_col = ['prev_close', 'prev_open', 'SMA5', 'SMA20', 'EMA60', 'RSI', 'MACD', 'MACD_SIGNAL', "MACD_HIST", "BB_Low", "BB_Mid", "BB_High", "ATR", "ADX", "OBV", 'STOCH_K', 'STOCH_D', 'CCI_14', 'CCI_20',
       'WILLR_14', 'MOM_10', 'MOM_14', 'ROC_12', 'ADX_14', 'PLUS_DI',
       'MINUS_DI', 'AROONOSC_14', 'MFI_14', 'AD', 'ADOSC', 'SAR', 'NATR_14', "Daily_Return"]
y_col = 'label'

date_format = "%Y-%m-%d"
end_date = datetime.datetime.strptime(end_date, date_format)
one_month_ago = end_date  + relativedelta(months=-1)

# 1 month (< 30 trading days)
pred_month = data.loc[one_month_ago:]
pred_month

training = data.loc[:one_month_ago]
training

data = data.dropna()

x_col = ['SMA5', 'SMA20', 'EMA60', 'RSI', 'MACD', 'MACD_SIGNAL', "MACD_HIST", "BB_Low", "BB_Mid", "BB_High", "ATR", "ADX", "OBV", 'STOCH_K', 'STOCH_D', 'CCI_14', 'CCI_20',
    'WILLR_14', 'MOM_10', 'MOM_14', 'ROC_12', 'ADX_14', 'PLUS_DI',
    'MINUS_DI', 'AROONOSC_14', 'MFI_14', 'AD', 'ADOSC', 'SAR', 'NATR_14', "Daily_Return"]
y_col = ['label']

X = training[x_col]
y = training[y_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pred_data_x = pred_month[x_col]
pred_data_y = pred_month[y_col]

# ML Models
result_dict = {}
#Decision Tree
result_dict['Decision Tree'] = {}
clf = DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X_train, y_train)
result_dict['Decision Tree']['test'] = clf.predict(X_test)
result_dict['Decision Tree']['train'] = clf.predict(X_train)
result_dict['Decision Tree']['latest_month'] = clf.predict(pred_data_x)

#KNN
result_dict['KNN'] = {}
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
result_dict['KNN']['test'] = knn.predict(X_test)
result_dict['KNN']['train'] = knn.predict(X_train)
result_dict['KNN']['latest_month'] = knn.predict(pred_data_x)

#Logistic Regression
result_dict['Logistic Regression'] = {}
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
result_dict['Logistic Regression']['test'] = log_reg.predict(X_test)
result_dict['Logistic Regression']['train'] = log_reg.predict(X_train)
result_dict['Logistic Regression']['latest_month'] = log_reg.predict(pred_data_x[x_col])

#Random Forest
result_dict['Random Forest'] = {}
random = RandomForestClassifier(n_estimators=30,max_depth=5)
random.fit(X_train, y_train)
result_dict['Random Forest']['test'] = random.predict(X_test)
result_dict['Random Forest']['train'] = random.predict(X_train)
result_dict['Random Forest']['latest_month'] = random.predict(pred_data_x)

#SVM
result_dict['SVM'] = {}
svm = SVC()
svm.fit(X_train, y_train)
result_dict['SVM']['test'] = svm.predict(X_test)
result_dict['SVM']['train'] = svm.predict(X_train)
result_dict['SVM']['latest_month'] = svm.predict(pred_data_x)


for m in result_dict.keys(): 
    # Accuracy
    test_acc = accuracy_score(result_dict[m]['test'], y_test)
    train_acc = accuracy_score(result_dict[m]['train'], y_train)
    pred_acc = accuracy_score(result_dict[m]['latest_month'], pred_data_y)
    
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
    recall_pred = recall_score(pred_data_y, result_dict[m]['latest_month'], average='weighted')
    f1_pred = f1_score(pred_data_y, result_dict[m]['latest_month'], average='weighted')
    
    print(f"{m} Test Accuracy: {test_acc}")
    print(f"{m} Test Precision: {precision_test}")
    print(f"{m} Test Recall: {recall_test}")
    print(f"{m} Test F1 Score: {f1_test}")
    
    print(f"{m} Train Accuracy: {train_acc}")
    print(f"{m} Train Precision: {precision_train}")
    print(f"{m} Train Recall: {recall_train}")
    print(f"{m} Train F1 Score: {f1_train}")
    
    print(f"{m} Latest Month Prediction Accuracy: {pred_acc}")
    print(f"{m} Latest Month Prediction Precision: {precision_pred}")
    print(f"{m} Latest Month Prediction Recall: {recall_pred}")
    print(f"{m} Latest Month Prediction F1 Score: {f1_pred}")
    
    # Confusion Matrix for latest month prediction
    cm = confusion_matrix(pred_data_y, result_dict[m]['latest_month'])
    print(f"{m} Confusion Matrix (Latest Month Set):")
    print(cm)
    
    # Create a pandas DataFrame for better readability
    cm_df = pd.DataFrame(
        cm,
        index=[f'True {label}' for label in [-1, 1]],
        columns=[f'Pred {label}' for label in [-1, 1]]
    )
    print(cm_df)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 1])
    disp.plot()
    plt.title(f"{m} Confusion Matrix (Latest Month Set)")
    plt.show()
    
    print("\n")

