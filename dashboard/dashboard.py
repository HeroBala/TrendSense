# 📊 TrendSense Standalone Dashboard - Complete E-commerce Reddit Analysis

# --- LIBRARY IMPORTS ---
import streamlit as st                   # Streamlit: For web dashboard UI
import pandas as pd                      # Pandas: For data manipulation and analysis
import plotly.express as px              # Plotly Express: For interactive charts
from wordcloud import WordCloud          # WordCloud: For word cloud visualizations
import matplotlib.pyplot as plt          # Matplotlib: For plotting word clouds
import json                             # JSON: To load raw data from JSON files
from datetime import datetime, timedelta # Datetime: For handling dates and time ranges
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # VADER: For sentiment analysis

# ================== CONFIG ===================
st.set_page_config(page_title="💡 TrendSense Insights", layout="wide")  # Set Streamlit page configuration
st.title(":chart_with_upwards_trend: TrendSense: Reddit E-commerce Analysis Dashboard")  # Set dashboard title
st.caption("Explore Reddit discussions across subreddits with text, sentiment, and score insights")  # Set dashboard caption

# ================== SENTIMENT FUNCTION ===================
def get_sentiment_label(score):
    if score >= 0.05:              # If sentiment score is positive
        return "Positive"
    elif score <= -0.05:           # If sentiment score is negative
        return "Negative"
    else:                          # Otherwise, neutral
        return "Neutral"

# ================== DATA LOAD ===================
@st.cache_data
def load_all_data():
    cleaned = pd.read_excel("data/cleaned_data.xlsx")                  # Load cleaned data from Excel
    with open("data/ecommerce_advanced.json") as f:                    # Load raw data from JSON file
        raw = pd.DataFrame(json.load(f))
    try:
        labeled = pd.read_excel("models/labeled_data.xlsx")            # Try loading labeled data
    except:
        labeled = pd.DataFrame()                                       # If not found, create empty DataFrame

    # Ensure 'date' column exists and is in datetime format
    if "date" not in cleaned.columns or cleaned["date"].isnull().all():
        if "created_utc" in cleaned.columns:
            cleaned["date"] = pd.to_datetime(cleaned["created_utc"], unit='s', errors='coerce')
        else:
            cleaned["date"] = pd.to_datetime("today")
    else:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    cleaned["date"].fillna(pd.to_datetime("today"), inplace=True)      # Replace missing dates with today

    if "score" not in cleaned.columns:
        cleaned["score"] = 0                                          # Default score to 0

    # Ensure 'clean_text' column exists (combined title and text)
    if "clean_text" not in cleaned.columns:
        cleaned["clean_text"] = cleaned["title"].fillna("") + " " + cleaned["text"].fillna("")

    # Compute sentiment scores if missing
    if "sent_compound" not in cleaned.columns:
        analyzer = SentimentIntensityAnalyzer()
        cleaned["sent_compound"] = cleaned["clean_text"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])

    cleaned["Sentiment"] = cleaned["sent_compound"].apply(get_sentiment_label)  # Assign sentiment labels
    return cleaned, raw, labeled

cleaned_df, raw_df, labeled_df = load_all_data()  # Load all datasets

# ================== DEFAULT TOPICS ===================
subreddits = sorted(cleaned_df["subreddit"].dropna().unique())     # List all unique subreddits
sub_default = cleaned_df["subreddit"].value_counts().head(10).index.tolist()  # Top 10 subreddits

# ================== HAMBURGER FILTERS ===================
with st.expander("🔍 Filter Options", expanded=True):               # Expandable filter section
    col1, col2, col3 = st.columns(3)                               # Three columns for filter options

with col1:                                                         # Column 1: Subreddit selection
    all_selected = st.checkbox("Select All Subreddits", value=True)
    sub_selection = subreddits if all_selected else st.multiselect(":woman: Subreddits", subreddits, default=sub_default)

with col2:                                                         # Column 2: Sentiment selection
    sentiments = st.multiselect(":speech_balloon: Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])

with col3:                                                         # Column 3: Top N subreddits slider
    top_n = st.slider("Top N Subreddits", 5, 50, 20)

# ================== FILTERING ===================
filtered = cleaned_df[
    (cleaned_df["subreddit"].isin(sub_selection)) &
    (cleaned_df["Sentiment"].isin(sentiments))
]

if filtered.empty:                                                 # If no data matches filters, use most recent 200
    st.warning("No matching documents. Showing most recent 200 instead.")
    filtered = cleaned_df.sort_values("date", ascending=False).head(200)

# ================== KPIs ===================
st.markdown("## :bar_chart: Key Insights")                         # Section header for KPIs
col_a, col_b, col_c, col_d = st.columns(4)                        # Four columns for metrics
col_a.metric("Total Documents", len(filtered))                     # Show total document count
col_b.metric("Unique Subreddits", filtered["subreddit"].nunique()) # Show number of unique subreddits
col_c.metric("Average Sentiment", round(filtered["sent_compound"].mean(), 2)) # Average sentiment score
if "topic" in filtered.columns:                                    # Show unique topics if available
    col_d.metric("Unique Topics", filtered["topic"].nunique())

# ================== DARK THEME SETTINGS ===================
dark_layout = dict(
    plot_bgcolor='#121212',                # Background color for plots
    paper_bgcolor='#121212',               # Background color for paper
    font=dict(color='white'),              # Font color
    xaxis=dict(title='', showgrid=True, gridcolor='#333'),  # X-axis settings
    yaxis=dict(title='', showgrid=True, gridcolor='#333'),  # Y-axis settings
    legend_title=dict(font=dict(color='white')),            # Legend title color
    title_font=dict(color='white')                          # Title font color
)

sentiment_colors = {                       # Colors for sentiment classes
    "Positive": "#2ca02c",
    "Neutral": "#1f77b4",
    "Negative": "#ff7f0e"
}

def apply_dark_theme(fig, title=None):     # Helper to apply dark theme to Plotly figures
    if title:
        fig.update_layout(title=title)
    fig.update_layout(**dark_layout)
    return fig

# ================== TABS ===================
tabs = st.tabs([
    ":chart_with_upwards_trend: Charts",
    ":cloud: Word Cloud",
    ":page_facing_up: Documents",
    ":package: Raw JSON",
    ":brain: Topic Stats",
    ":mag: TF-IDF Terms",
    ":compass: Topic Explorer"
])

# ========= Charts =========
with tabs[0]:
    st.subheader(":calendar: Daily Sentiment Trends")
    filtered["date_only"] = filtered["date"].dt.date                     # Extract date part
    trend_df = filtered.groupby(["date_only", "Sentiment"]).size().reset_index(name="count")  # Count by date & sentiment

    if not trend_df["date_only"].empty:                                  # Calculate date range for slider
        max_date = trend_df["date_only"].max()
        min_date = max_date - timedelta(days=6)
        x_range = [min_date, max_date]
    else:
        x_range = None

    fig_trend = px.line(                                                # Line chart for sentiment trend
        trend_df,
        x="date_only",
        y="count",
        color="Sentiment",
        title="Daily Sentiment Counts",
        markers=True,
        color_discrete_map=sentiment_colors
    )
    fig_trend.update_traces(mode="lines+markers")
    fig_trend.update_xaxes(rangeslider_visible=True, range=x_range)
    fig_trend = apply_dark_theme(fig_trend)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader(":bar_chart: Subreddits by Average Score")
    top_scores = filtered.groupby("subreddit")["score"].mean().reset_index().nlargest(top_n, "score")
    fig_scores = px.bar(top_scores, x="score", y="subreddit", orientation="h", title="Top Subreddits by Avg Score")
    fig_scores = apply_dark_theme(fig_scores)
    st.plotly_chart(fig_scores, use_container_width=True)

    st.subheader(":mag: Sentiment vs Score Scatter")
    fig_scatter = px.scatter(filtered, x="score", y="sent_compound", color="Sentiment", hover_data=["title"], color_discrete_map=sentiment_colors)
    fig_scatter = apply_dark_theme(fig_scatter)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader(":bar_chart: Score Distribution")
    fig_hist = px.histogram(filtered, x="score", nbins=30, title="Distribution of Scores")
    fig_hist = apply_dark_theme(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)

    if "topic" in filtered.columns:
        st.subheader(":trophy: Top Topics by Frequency")
        top_topics = filtered["topic"].value_counts().reset_index()
        top_topics.columns = ["Topic", "Count"]
        fig_topic = px.bar(top_topics.head(15), x="Count", y="Topic", orientation="h", title="Most Frequent Topics")
        fig_topic = apply_dark_theme(fig_topic)
        st.plotly_chart(fig_topic, use_container_width=True)

# ========= Word Cloud =========
with tabs[1]:
    st.subheader(":cloud: Cleaned Text Word Cloud")
    if not filtered["clean_text"].dropna().empty:
        text = " ".join(filtered["clean_text"].dropna())                # Combine all cleaned text
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)  # Generate word cloud
        fig, ax = plt.subplots(figsize=(
