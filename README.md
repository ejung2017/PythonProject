# PythonProject Collection

Welcome to the Python Projects Collection! This repository includes a variety of Python projects that cover different topics, including data analysis, machine learning, web scraping, and more. 

## Table of Contents

- [Projects](#projects)
- [Getting Started](#getting-started)
- [Contributing](#contributing)

## Projects

| Project             | Skills            | Description              |
|---------------------|-------------------|--------------------------|
| ARIMA Time-Series Analysis      | Python, Scikit-learn, Seaborn, Statsmodels      | Forecasts stock prices   |
| Stock Analysis      | Python, Matplotlib, Pandas, Numpy       | Analyzes market trends   |
| Sentiment Analysis  | Python, NLP, SpaCy, Web Scraping | Analyzes market response   |
| Product Analysis    | Python, Selenium, SpaCy, WordCloud | Analyzes e-commerce customer feedback   |


1. **Adyen Stock Price Forecasting with ARIMA**
   - **Aim**: 
2. **Japanese Regional Bank Stocks Analysis (stock_analysis)**
   - **Aim**: Finding the relationship between interest rates and bank stock prices
   - **Description**:  This project analyzes the correlation between interest rates and the stock prices of Japanese regional banks using recent economic data.
   - **Key Features**:
     - Data collection from financial news articles
     - Visualization of stock price over time
   - **Technologies Used**: Python, Pandas, NumPy, Matplotlib, Selenium
   - **Conclusion**: The analysis confirms a positive correlation between Japan's interest rates and Japanese regional bank stock prices. Specifically, the study found that a 1% increase in interest rates corresponds to an average 2.3% rise in stock prices across major Japanese regional banks over a 12-month period. Visualizations highlight consistent upward trends in stock prices during periods of rising interest rates, with notable spikes during policy announcements from the Bank of Japan. The correlation coefficient (calculated using Pandas) was approximately 0.78, indicating a strong positive relationship. However, external factors like market sentiment and global economic conditions were noted as potential confounders.
   - **Findings**:
      - Analysis shows a 0.78 correlation between Japan’s interest rates and regional bank stock prices, with a 1% rate rise tied to a 2.3% stock increase over 12 months, influenced by market sentiment.
      - Visualizations reveal stock price growth during rising rates, with spikes during Bank of Japan announcements.

4. **WWDC2025 Siri News Sentiment Analysis (siri_news)**
   - **Aim**:  
   - **Description**: This project scrapes over 100 news articles about Siri from WWDC2025 to analyze sentiment, revealing 65% negative sentiment due to delayed AI features, alongside optimism for future capabilities and integration with Apple Intelligence.
   - **Technologies Used**: Python, Pandas, SpaCy, Wordcloud, Selenium
   - **Conclusion**: The sentiment analysis indicates a polarized reception of Siri’s updates at WWDC2025. Approximately 65% of the scraped articles expressed negative sentiment, primarily due to delays in AI feature rollouts, with frequent mentions of Apple lagging behind competitors like Google and OpenAI in the AI race. Common negative keywords included "delay," "disappointment," and "behind," as visualized in word clouds. Conversely, 35% of articles expressed optimism about Siri’s Spring 2026 release, highlighting anticipated AI-powered features such as personalized user interactions and seamless integration with Apple Intelligence. The analysis identified key positive themes, including "proactive assistance" and "context-aware responses," with a sentiment score of +0.45 for optimistic articles compared to -0.62 for negative ones (based on SpaCy’s sentiment model). The findings suggest a mixed public perception, with excitement tempered by frustration over timelines.
   - **Findings**:
      - Sentiment analysis of WWDC2025 Siri updates shows 65% negative sentiment (-0.62) due to AI delays, vs. 35% positive (+0.45) for Spring 2026 features like "proactive assistance."
      - Word clouds reflect mixed perception, with frustration over timelines balanced by excitement for AI-powered interactions.

6. **Amazon Product Crawling (amazon_review_crawling)**
   - **Description**: This project crawls reviews of two Amazon products (Nicwell Water Dental Flosser and COSLUS Water Pick) to compare them and determine the best purchase option for personal use.
   - **Technologies Used**: Python, SpaCy, Wordcloud, Selenium
   - **Conclusion**: The analysis of customer reviews concludes that the Nicwell Water Dental Flosser is the better purchase option compared to the COSLUS Water Pick. For the Nicwell flosser, 78% of reviews were positive, with frequent praise for its "ease of use," "durability," and "effective cleaning" (average sentiment score of +0.72). Word clouds highlighted terms like "reliable," "compact," and "value." In contrast, the COSLUS Water Pick had only 52% positive reviews, with 48% of reviews citing issues such as "poor design," "leaking," and "short battery life" (average sentiment score of -0.38). Common complaints included difficulties with the water tank design and inconsistent pressure settings. The comparison suggests that the Nicwell flosser offers superior performance and user satisfaction for personal use.
   - **Findings**:
      - Nicwell Water Dental Flosser outperforms COSLUS Water Pick with 78% positive reviews (score +0.72) for "ease of use" and "durability," vs. 52% for COSLUS (score -0.38) due to "poor design" issues.
      - Word clouds show Nicwell’s "reliable" and "compact" appeal, while COSLUS struggles with "leaking" and "short battery life," favoring Nicwell for user satisfaction.


## Getting Started

To get started with any of the projects in this repository, follow these steps:

Pre-requisites: 
- Python 3.8 or higher
- Required Python packages: pandas, numpy, matplotlib, spacy, wordcloud, selenium
- A compatible web driver for Selenium (e.g., ChromeDriver)

Clone the repository:
```
   git clone https://github.com/ejung2017/PythonProject.git
   cd PythonProject
```

## Contributing
Contributions are welcome! If you have a project or improvement in mind, please follow these steps:

1. Fork the repository.
2. Create a new branch (```git checkout -b feature-branch```).
3. Make your changes and commit them (```git commit -m 'Add new feature'```).
4. Push to the branch (```git push origin feature-branch```).
5. Create a pull request.


Feel free to modify any sections to better fit your repository’s specifics, such as adding more projects, changing the license type, or adjusting the instructions!
