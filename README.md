# <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> PythonProject Collection

Welcome to my Python Projects Collection! This repository highlights my diverse projects, driven by my enthusiasm for Machine Learning, Natural Language Processing (NLP), and data insights, spanning data analysis, predictive modeling, web scraping, trend analysis, SQL, data visualization, and many more! 

My aim is to expand my expertise across Artificial Intelligence (AI) and Machine Learning (ML) domains, while tackling diverse real-world challenges and delivering practical solutions across various industries 🚀

## 📌 Projects

| Project  | Topic                  | Skills                                          | Description                             |
|----------|------------------------|-------------------------------------------------|-----------------------------------------|
| 1        | Sentiment Analysis     | Python, NLP, SpaCy, Web Scraping                | Analyzes market response                |
| 2        | Time-Series Analysis   | Python, Scikit-learn, TA-Lib, Seaborn, Statsmodels | Forecasts stock buy/sell signals        |
| 3        | Product Review Analysis| Python, Selenium, SpaCy, WordCloud              | Analyzes e-commerce customer feedback   |


## 🤖 [Project1: WWDC2025 Siri News Sentiment Analysis](https://github.com/ejung2017/PythonProject/tree/main/project1)

This project analyzes sentiment to address real-time customer feedback and foster strategic decisions, especially in competitive tech markets. Based on 98 news articles scraped about Apple Intelligence Siri during WWDC2025, I tried to understand public expectations and its potential business needs. 

**Key Findings**:
  - 35% showing optimism towards personalization and integration of AI-powered features.
  - 65% showing disappointment due to constant delays and incapabilities to match the users' expectations
  - Word Cloud with filtered keywords 
 <img width="790" height="427" alt="Unknown" src="https://github.com/user-attachments/assets/21ae5ffa-1d4e-401b-b7dc-d1388ab08c96" />


## 📊 [Project2: Stock Prediction Web App](https://github.com/ejung2017/PythonProject/tree/main/project2)

This project predicts stock price trends (up or down) using Machine Learning models trained on technical indicators like moving averages, RSI, MACD, and Bollinger Bands. On the landing page, it displays predictions and 3-month profit/loss percentages for six major international companies (Apple, Google, Samsung, Tencent, LVMH, Toyota). Users can enter any stock ticker in the sidebar to get detailed ML predictions on the analysis page, including model accuracy metrics, confusion matrices, and whether to buy or sell based on the model's signal. The app filters and recommends only ML models with >80% accuracy and <5% overfitting gap. It calculates potential profits/losses by simulating trades based on the model's buy/sell signals over the past 3 months.

*Streamlit URL*: https://stockanalysis-ej.streamlit.app

**Key Deliverables**:
- Interactive web app where the user inputs the ticker and the date range for stock analysis
- Estimated 3 month profit/loss returns from the top ML model


https://github.com/user-attachments/assets/47f7b22f-854e-42f0-b5dc-e0838485f6e0

## 🛍️ [Project3: Amazon Product Crawling](https://github.com/ejung2017/PythonProject/tree/main/project3)

This project compares two products based on the customer reviews and comes up with the most ideal suggestion for a customer (myself). 

**Key Findings**:
- Product A (Nicwell) is more suitable with 78% positive reviews
- Product B (COSLUS) had 52% positive reviews
- Product A had better reviews regarding durability, design, and convenience.


| Product A               |  Product B              |
|-------------------------|-------------------------|
|<img width="425" alt="Unknown-2" src="https://github.com/user-attachments/assets/d3355e37-045d-4c18-b111-291303c518cd" /> | <img width="425" alt="Unknown-3" src="https://github.com/user-attachments/assets/2251d892-2a8c-45ad-9546-c2ac1d1ae9ab" /> |

## 👩🏻‍💻 [Project4: LinkedIn Job Scraping](https://github.com/ejung2017/PythonProject/tree/main/project4)

Selenium notebook that scrapes LinkedIn job postings (Data Scientist in Hong Kong) and saves company, title, date, and job URL.

**Key Deliverables**:
- Output: `project4/output_with_links.csv` (includes URLs) and `project4/output.csv`.

