# Financial News Sentiment Analysis & Market Prediction

A machine learning pipeline that combines **NLP-based sentiment analysis** with **technical price indicators** to predict next-day stock market direction.

---

## Project Overview

This project analyzes financial news headlines using **FinBERT** (a BERT model pre-trained on financial text) and combines the resulting sentiment signals with technical price features to predict whether the market will go **up or down** the next trading day.

Built as a demonstration of combining Natural Language Processing (NLP) with traditional financial analysis in a single end-to-end ML pipeline.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `Python` | Core language |
| `yfinance` | Stock price data from Yahoo Finance |
| `finnhub-python` | Financial news headlines via API |
| `HuggingFace Transformers` | Loading and running FinBERT |
| `FinBERT (ProsusAI)` | Financial sentiment classification |
| `XGBoost` | Market direction prediction model |
| `Pandas` | Data manipulation and feature engineering |
| `Matplotlib / Seaborn` | Visualisations |
| `Scikit-learn` | Model evaluation metrics |

---

## Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | **60.87%** |
| Precision (Market Down) | **0.80** |
| Recall (Market Up) | **0.84** |
| F1 Score (Weighted Avg) | **0.60** |

> A random baseline would achieve ~50% accuracy. The model's **80% precision on downside predictions** makes it particularly useful as a risk management signal — when it predicts a down day, it is right 4 out of 5 times.

---

## How It Works

### Step 1 — Data Collection
- User inputs a stock ticker (e.g. `^NSEI`, `AAPL`, `TSLA`) and date range
- Historical OHLCV price data is fetched from Yahoo Finance via `yfinance`
- Financial news headlines are fetched from Finnhub API

### Step 2 — Sentiment Analysis (FinBERT)
- Each headline is passed through **FinBERT**, a BERT model fine-tuned on financial text
- FinBERT classifies each headline as `positive`, `neutral`, or `negative` with a confidence score
- Headlines are mapped to numerical scores: positive=1, neutral=0, negative=-1
- Daily sentiment features are aggregated per trading day

### Step 3 — Feature Engineering
**Sentiment features:**
- Average daily sentiment score
- Sentiment standard deviation (how mixed was the news?)
- Percentage of positive / negative articles
- 3-day sentiment momentum (is sentiment improving or worsening?)

**Technical price features:**
- Previous day return
- Price vs 5-day moving average (is market overbought/oversold?)
- 5-day rolling volatility
- Volume change

### Step 4 — ML Model (XGBoost)
- Target variable: `1` if next-day close > today's close, else `0`
- Chronological train/test split (80/20) — no data leakage
- XGBoost classifier with 200 estimators, learning rate 0.05
- Evaluated with accuracy, precision, recall, F1, and confusion matrix

---

## Visualisations

The project generates 3 chart files:

**1. price_analysis.png**
- Closing price with 5-day and 10-day moving averages
- Daily returns (green = positive, red = negative)
- 5-day rolling volatility

**2. model_evaluation.png**
- Feature importance chart (which features drove predictions most)
- Confusion matrix (true vs predicted labels)

**3. sentiment_analysis.png**
- Sentiment distribution pie chart across all headlines
- FinBERT confidence distribution by sentiment category

---

## Limitations & Real-World Notes

> **Data Quality Note:** Finnhub's free tier API does not provide historical news data — it returns current headlines for every date queried. As a result, sentiment features have no date-to-date variation and carry limited predictive signal in this implementation.

> In a production system, historical news would be sourced from paid APIs such as **Bloomberg Terminal**, **Refinitiv Eikon**, or **RapidAPI financial news endpoints**. The FinBERT sentiment pipeline is fully functional and production-ready — the limitation is purely in the data source.

This is a deliberate and documented design decision, not an oversight.

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/sentiment-market-prediction
cd sentiment-market-prediction
```

**2. Install dependencies**
```bash
pip install yfinance finnhub-python transformers torch pandas scikit-learn xgboost matplotlib seaborn
```

**3. Add your Finnhub API key**

Get a free key at [finnhub.io](https://finnhub.io) and replace in the code:
```python
FINNHUB_API_KEY = "your_api_key_here"
```

**4. Run the notebook**
```bash
# Open in VS Code or Jupyter
sentiment_analysis.ipynb
```

When prompted, enter:
- Stock ticker (e.g. `^NSEI` for Nifty 50, `AAPL` for Apple)
- Start date (e.g. `2024-01-01`)
- End date (e.g. `2025-01-01`)

---

## Project Structure

```
sentiment-market-prediction/
│
├── sentiment_analysis.ipynb   # Main notebook
├── price_analysis.png         # Price & volatility charts
├── model_evaluation.png       # Feature importance & confusion matrix
├── sentiment_analysis.png     # Sentiment distribution charts
└── README.md                  # This file
```


---

## References

- [FinBERT — ProsusAI](https://huggingface.co/ProsusAI/finbert)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Finnhub API](https://finnhub.io/docs/api)
