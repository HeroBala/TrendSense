# 📊 TrendSense Standalone Dashboard - Complete E-commerce Reddit Analysis

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ================== CONFIG ===================
st.set_page_config(page_title="💡 TrendSense Insights", layout="wide")
st.title(":chart_with_upwards_trend: TrendSense: Reddit E-commerce Analysis Dashboard")
st.caption("Explore Reddit discussions across subreddits with text, sentiment, and score insights")

# ================== SENTIMENT FUNCTION ===================
def get_sentiment_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# ================== DATA LOAD ===================
@st.cache_data
def load_all_data():
    cleaned = pd.read_excel("data/cleaned_data.xlsx")
    with open("data/ecommerce_full.json") as f:
        raw = pd.DataFrame(json.load(f))

    if "date" not in cleaned.columns or cleaned["date"].isnull().all():
        if "created_utc" in cleaned.columns:
            cleaned["date"] = pd.to_datetime(cleaned["created_utc"], unit='s', errors='coerce')
        else:
            cleaned["date"] = pd.to_datetime("today")
    else:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    cleaned["date"].fillna(pd.to_datetime("today"), inplace=True)

    if "score" not in cleaned.columns:
        cleaned["score"] = 0

    if "clean_text" not in cleaned.columns:
        cleaned["clean_text"] = cleaned["title"].fillna("") + " " + cleaned["text"].fillna("")

    if "sent_compound" not in cleaned.columns:
        analyzer = SentimentIntensityAnalyzer()
        cleaned["sent_compound"] = cleaned["clean_text"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])

    cleaned["Sentiment"] = cleaned["sent_compound"].apply(get_sentiment_label)
    return cleaned, raw

cleaned_df, raw_df = load_all_data()

# ================== DEFAULT TOPICS ===================
subreddits = sorted(cleaned_df["subreddit"].dropna().unique())
sub_default = cleaned_df["subreddit"].value_counts().head(10).index.tolist()

# ================== FILTER BAR ===================
st.markdown("### :weight_lifting: Filters")
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    all_selected = st.checkbox("Select All Subreddits", value=True)
    sub_selection = subreddits if all_selected else st.multiselect(":woman: Subreddits", subreddits, default=sub_default)

with col2:
    sentiments = st.multiselect(":speech_balloon: Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])

with col3:
    top_n = st.slider("Top N Subreddits", 5, 50, 20)

# ================== FILTERING ===================
filtered = cleaned_df[
    (cleaned_df["subreddit"].isin(sub_selection)) &
    (cleaned_df["Sentiment"].isin(sentiments))
]

if filtered.empty:
    st.warning("No matching documents. Showing most recent 200 instead.")
    filtered = cleaned_df.sort_values("date", ascending=False).head(200)

# ================== KPIs ===================
st.markdown("## :bar_chart: Key Insights")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Documents", len(filtered))
col_b.metric("Unique Subreddits", filtered["subreddit"].nunique())
col_c.metric("Average Sentiment", round(filtered["sent_compound"].mean(), 2))

# ================== TABS ===================
tabs = st.tabs([":chart_with_upwards_trend: Charts", ":cloud: Word Cloud", ":page_facing_up: Documents", ":package: Raw JSON", ":brain: Topic Stats"])

# ========= Charts =========
with tabs[0]:
    st.subheader(":calendar: Daily Sentiment Trends")
    filtered["date_only"] = filtered["date"].dt.date
    trend_df = filtered.groupby(["date_only", "Sentiment"]).size().reset_index(name="count")
    fig_trend = px.line(
        trend_df,
        x="date_only",
        y="count",
        color="Sentiment",
        title="Daily Sentiment Counts",
        markers=True,
    )
    fig_trend.update_layout(legend_title_text='Sentiment')
    fig_trend.update_traces(mode="lines+markers")
    fig_trend.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader(":bar_chart: Subreddits by Average Score")
    top_scores = filtered.groupby("subreddit")["score"].mean().reset_index().nlargest(top_n, "score")
    fig_scores = px.bar(top_scores, x="score", y="subreddit", orientation="h", title="Top Subreddits by Avg Score")
    st.plotly_chart(fig_scores, use_container_width=True)

    st.subheader(":mag: Sentiment vs Score Scatter")
    fig_scatter = px.scatter(filtered, x="score", y="sent_compound", color="Sentiment", hover_data=["title"])
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader(":bar_chart: Score Distribution")
    fig_hist = px.histogram(filtered, x="score", nbins=30, title="Distribution of Scores")
    st.plotly_chart(fig_hist, use_container_width=True)

# ========= Word Cloud =========
with tabs[1]:
    st.subheader(":cloud: Cleaned Text Word Cloud")
    if not filtered["clean_text"].dropna().empty:
        text = " ".join(filtered["clean_text"].dropna())
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("Not enough cleaned text for word cloud.")

# ========= Document Table =========
with tabs[2]:
    st.subheader(":page_facing_up: Reddit Documents")
    filtered_display = filtered.copy()
    filtered_display["title"] = filtered_display["title"].apply(lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else x)
    filtered_display["date"] = filtered_display["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(filtered_display[["title", "subreddit", "Sentiment", "score", "sent_compound", "date"]])
    st.download_button("Download Filtered Data", data=filtered.to_csv(index=False), file_name="filtered_documents.csv")

# ========= Raw JSON =========
with tabs[3]:
    st.subheader(":package: Raw JSON Data Sample")
    st.write(raw_df.head(5))
    st.caption(f"Total Raw Records: {len(raw_df)}")

# ========= Topic Stats =========
with tabs[4]:
    st.subheader(":brain: Subreddit Topic Statistics")
    topic_table = filtered.groupby(["subreddit", "Sentiment"]).size().unstack(fill_value=0)
    st.dataframe(topic_table)

    st.subheader(":bar_chart: Sentiment Distribution by Subreddit")
    dist_data = filtered.groupby(["subreddit", "Sentiment"]).size().reset_index(name="count")
    fig_dist = px.bar(dist_data, x="subreddit", y="count", color="Sentiment", barmode="stack")
    st.plotly_chart(fig_dist, use_container_width=True)
