# 🧠 TrendSense – Reddit E-commerce Insights Platform

![TrendSense Dashboard Preview](https://via.placeholder.com/1000x400.png?text=Dashboard+Preview)

**TrendSense** is a comprehensive NLP and data visualization pipeline that extracts, analyzes, and visualizes insights from e-commerce-related Reddit discussions. It helps both analysts and entrepreneurs discover real-time market sentiment, popular topics, and consumer pain points.

🚀 **Live Dashboard**: [Explore Here](trendsense-acednyffu2kwqkqmutfbc4/)

---

## 📊 Features

- **Automated Data Pipeline**
  - Collects posts and comments from 40+ e-commerce subreddits.
  - Cleans and preprocesses text using advanced NLP techniques.
- **Topic Modeling with BERTopic**
- **Sentiment Analysis with VADER**
- **Interactive Dashboard**
  - Visualizations for sentiment trends, subreddit rankings, score distribution, and topic-specific insights.

---

## 🧱 System Architecture

1. **Data Collection** – `fetcher.py`  
   Uses `praw` to fetch posts/comments from Reddit based on score and retry thresholds.

2. **Data Cleaning** – `cleaner.py`  
   Cleans raw Reddit data using `spaCy`, `BeautifulSoup`, and regex for high-quality analysis.

3. **Analysis** – `analyzer.py`  
   Applies BERTopic for topic discovery and VADER for sentiment analysis.

4. **Visualization** – `dashboard.py`  
   Streamlit-based dashboard using Plotly, WordClouds, and filterable KPIs.

---

## ⚙️ How to Run Locally

### ✅ Prerequisites

- Python 3.9+
- [Reddit API credentials](https://www.reddit.com/prefs/apps)
- Recommended: Virtual environment

### 🔧 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/HeroBala/trendsense.git
cd trendsense

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Step 1: Fetch Reddit Data
python fetcher.py

# Step 2: Clean the data
python cleaner.py

# Step 3: Analyze the data (topic modeling + sentiment)
python analyzer.py

# Step 4: Launch the Streamlit dashboard
streamlit run dashboard.py
                                                                      
