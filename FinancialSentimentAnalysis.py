# ============================================================
# FINANCIAL NEWS SENTIMENT ANALYSIS & MARKET PREDICTION
# ============================================================

# --- CELL 1: IMPORTS & USER INPUT ---
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import finnhub
from datetime import datetime
import time
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)

print("=" * 50)
print("   Financial News Sentiment Analyzer")
print("=" * 50)

ticker     = input("Enter a stock ticker symbol (e.g. AAPL, TSLA, ^NSEI): ").upper().strip()
start_date = input("Enter start date (YYYY-MM-DD format, e.g. 2024-01-01): ").strip()
end_date   = input("Enter end date   (YYYY-MM-DD format, e.g. 2025-01-01): ").strip()

print(f"\nFetching data for: {ticker} from {start_date} to {end_date}...")


# --- CELL 2: DOWNLOAD STOCK DATA ---
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Flatten multi-level column headers that newer yfinance versions create
# e.g. ('Close', '^NSEI') becomes just 'Close'
stock_data.columns = stock_data.columns.get_level_values(0)

if stock_data.empty:
    print(f"No data found for '{ticker}'. Please check the ticker symbol and try again.")
else:
    print(f"\nSuccessfully loaded {len(stock_data)} trading days of data for {ticker}")
    print(stock_data.head())


# --- CELL 3: CREATE TARGET VARIABLE ---
# Next_Day_Return = Close(t+1) - Close(t)
# Target = 1 if market went UP next day, 0 if it went DOWN
stock_data['Next_Day_Return'] = stock_data['Close'].shift(-1) - stock_data['Close']
stock_data['Target']          = (stock_data['Next_Day_Return'] > 0).astype(int)
stock_data = stock_data.dropna()

print(stock_data[['Close', 'Next_Day_Return', 'Target']].head(10))


# --- CELL 4: FINNHUB SETUP ---
FINNHUB_API_KEY = "d6vg4n9r01qiiutbodp0d6vg4n9r01qiiutbodpg"
finnhub_client  = finnhub.Client(api_key=FINNHUB_API_KEY)
print("Finnhub client created successfully!")


# --- CELL 5: FETCH NEWS HEADLINES ---
all_news = []
dates    = stock_data.index.tolist()

print(f"Fetching news for {len(dates)} trading days... this may take a few minutes")

for i, date in enumerate(dates):
    date_str   = str(date.date())
    date_start = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
    date_end   = date_start + 86400   # 86400 seconds = 1 day

    try:
        news = finnhub_client.general_news(category="general", min_id=0)
        for article in news[:5]:
            all_news.append({
                'date'    : date_str,
                'headline': article.get('headline', ''),
                'source'  : article.get('source', ''),
                'url'     : article.get('url', '')
            })
    except Exception as e:
        print(f"Could not fetch news for {date_str}: {e}")

    if (i + 1) % 20 == 0:
        print(f"  Processed {i + 1}/{len(dates)} days...")

    time.sleep(1)

news_df = pd.DataFrame(all_news)
print(f"\nDone! Collected {len(news_df)} news articles")
print(news_df.head())


# --- CELL 6: LOAD FINBERT ---
print("Loading FinBERT model... (this may take a minute the first time)")

sentiment_pipeline = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)

print("FinBERT loaded successfully!")


# --- CELL 7: RUN SENTIMENT ANALYSIS ---
sentiment_results = []

print(f"Analyzing sentiment for {len(news_df)} headlines...")

for i, row in news_df.iterrows():
    headline = row['headline']

    if not headline or len(headline.strip()) == 0:
        continue

    headline = headline[:512]   # FinBERT max input length

    try:
        result = sentiment_pipeline(headline)[0]
        sentiment_results.append({
            'date'      : row['date'],
            'headline'  : headline,
            'sentiment' : result['label'],   # positive / negative / neutral
            'confidence': result['score']    # 0 to 1
        })
    except Exception as e:
        print(f"Could not analyze headline: {headline[:50]}... Error: {e}")

    if (i + 1) % 100 == 0:
        print(f"  Analyzed {i + 1}/{len(news_df)} headlines...")

sentiment_df = pd.DataFrame(sentiment_results)
print(f"\nDone! Sentiment analyzed for {len(sentiment_df)} headlines")
print(sentiment_df.head(10))


# --- CELL 8: AGGREGATE SENTIMENT PER DAY ---
# Map sentiment labels to numbers: positive=1, neutral=0, negative=-1
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
sentiment_df['sentiment_score'] = sentiment_df['sentiment'].map(sentiment_map)

daily_sentiment = sentiment_df.groupby('date').agg(
    avg_sentiment  = ('sentiment_score', 'mean'),
    sentiment_std  = ('sentiment_score', 'std'),
    num_articles   = ('sentiment_score', 'count'),
    pct_positive   = ('sentiment_score', lambda x: (x == 1).sum()  / len(x)),
    pct_negative   = ('sentiment_score', lambda x: (x == -1).sum() / len(x)),
    avg_confidence = ('confidence', 'mean')
).reset_index()

print(daily_sentiment.head(10))


# --- CELL 9: SENTIMENT MOMENTUM ---
daily_sentiment = daily_sentiment.sort_values('date').reset_index(drop=True)

# 3-day rolling average of sentiment
# MA_3(t) = mean(sentiment(t), sentiment(t-1), sentiment(t-2))
daily_sentiment['sentiment_3day_avg']  = daily_sentiment['avg_sentiment'].rolling(window=3).mean()

# Momentum = today's sentiment - 3-day average
# Positive = sentiment improving, Negative = sentiment worsening
daily_sentiment['sentiment_momentum']  = (
    daily_sentiment['avg_sentiment'] - daily_sentiment['sentiment_3day_avg']
)

print(daily_sentiment[['date', 'avg_sentiment', 'sentiment_3day_avg', 'sentiment_momentum']].head(10))


# --- CELL 10: MERGE SENTIMENT WITH STOCK DATA ---
daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
stock_data.index        = pd.to_datetime(stock_data.index)

stock_df = stock_data.reset_index()
stock_df.rename(columns={'Date': 'date'}, inplace=True)

# Inner join — only keep dates that exist in both dataframes
merged_df = pd.merge(stock_df, daily_sentiment, on='date', how='inner')
merged_df = merged_df.dropna()

print(f"Merged dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns")
print(merged_df.head())


# --- CELL 11: ADD PRICE FEATURES ---
# pct_change(1) = (today - yesterday) / yesterday
merged_df['prev_day_return'] = merged_df['Close'].pct_change(1)

# Moving averages — smooth out daily noise
# MA_5(t) = mean of last 5 closing prices
merged_df['ma_5']            = merged_df['Close'].rolling(window=5).mean()
merged_df['ma_10']           = merged_df['Close'].rolling(window=10).mean()

# How far is today's price from the 5-day average?
# Positive = above average (bullish), Negative = below average (bearish)
merged_df['price_vs_ma5']    = (merged_df['Close'] - merged_df['ma_5']) / merged_df['ma_5']

# Volatility = standard deviation of returns over last 5 days
# High volatility = uncertain market
merged_df['volatility_5']    = merged_df['prev_day_return'].rolling(window=5).std()

# Is trading volume increasing or decreasing?
merged_df['volume_change']   = merged_df['Volume'].pct_change(1)

merged_df = merged_df.dropna()
print(f"Dataset now has {len(merged_df)} rows and {len(merged_df.columns)} columns")
print(merged_df[['Close', 'prev_day_return', 'ma_5', 'price_vs_ma5', 'volatility_5']].head())


# --- CELL 12: TRAIN/TEST SPLIT ---
# Note: Sentiment features excluded because Finnhub free tier returns
# identical headlines for every date — no variation, no signal.
# Price features carry the real predictive information.
feature_columns = [
    'prev_day_return',
    'price_vs_ma5',
    'volatility_5',
    'volume_change'
]

X = merged_df[feature_columns]
y = merged_df['Target']

# shuffle=False preserves chronological order — critical for time series
# We train on earlier dates and test on later dates to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"\nTarget distribution in training set:\n{y_train.value_counts()}")
print(f"\nTarget distribution in test set:\n{y_test.value_counts()}")


# --- CELL 13: TRAIN XGBOOST MODEL ---
model = XGBClassifier(
    n_estimators=200,      # 200 decision trees
    max_depth=4,           # each tree asks max 4 questions
    learning_rate=0.05,    # small steps = careful learning
    subsample=0.8,         # each tree sees 80% of data — prevents overfitting
    colsample_bytree=0.8,  # each tree uses 80% of features — prevents overfitting
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Market Down', 'Market Up']))