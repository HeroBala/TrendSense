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
    with open("data/ecommerce_advanced.json") as f:
        raw = pd.DataFrame(json.load(f))
    try:
        labeled = pd.read_excel("models/labeled_data.xlsx")
    except:
        labeled = pd.DataFrame()

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
    return cleaned, raw, labeled

cleaned_df, raw_df, labeled_df = load_all_data()

# ================== DEFAULT TOPICS ===================
subreddits = sorted(cleaned_df["subreddit"].dropna().unique())
sub_default = cleaned_df["subreddit"].value_counts().head(10).index.tolist()
# ================== HAMBURGER FILTERS ===================
with st.expander("🔍 Filter Options", expanded=True):
    col1, col2, col3 = st.columns(3)

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
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total Documents", len(filtered))
col_b.metric("Unique Subreddits", filtered["subreddit"].nunique())
col_c.metric("Average Sentiment", round(filtered["sent_compound"].mean(), 2))
if "topic" in filtered.columns:
    col_d.metric("Unique Topics", filtered["topic"].nunique())

# ================== DARK THEME SETTINGS ===================
dark_layout = dict(
    plot_bgcolor='#121212',
    paper_bgcolor='#121212',
    font=dict(color='white'),
    xaxis=dict(title='', showgrid=True, gridcolor='#333'),
    yaxis=dict(title='', showgrid=True, gridcolor='#333'),
    legend_title=dict(font=dict(color='white')),
    title_font=dict(color='white')
)

sentiment_colors = {
    "Positive": "#2ca02c",
    "Neutral": "#1f77b4",
    "Negative": "#ff7f0e"
}

def apply_dark_theme(fig, title=None):
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
    filtered["date_only"] = filtered["date"].dt.date
    trend_df = filtered.groupby(["date_only", "Sentiment"]).size().reset_index(name="count")

    if not trend_df["date_only"].empty:
        max_date = trend_df["date_only"].max()
        min_date = max_date - timedelta(days=6)
        x_range = [min_date, max_date]
    else:
        x_range = None

    fig_trend = px.line(
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
    fig_dist = px.bar(dist_data, x="subreddit", y="count", color="Sentiment", barmode="stack", color_discrete_map=sentiment_colors)
    fig_dist = apply_dark_theme(fig_dist)
    st.plotly_chart(fig_dist, use_container_width=True)

# ========= TF-IDF Terms =========
with tabs[5]:
    st.subheader(":mag: TF-IDF Terms by Topic")
    if not labeled_df.empty and "tfidf_terms" in labeled_df.columns:
        tfidf_terms = labeled_df[["topic", "tfidf_terms"]].drop_duplicates().sort_values("topic")
        st.dataframe(tfidf_terms, use_container_width=True)
    else:
        st.warning("TF-IDF terms not available in labeled data.")

# ========= Topic Explorer =========
with tabs[6]:
    st.subheader(":compass: Topic Explorer")

    if "topic" in filtered.columns:
        topic_dist = filtered["topic"].value_counts().reset_index()
        topic_dist.columns = ["Topic", "Count"]
        fig_pie = px.pie(topic_dist, names="Topic", values="Count", title="Topic Distribution")
        fig_pie = apply_dark_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

        sent_topic = filtered.groupby(["topic", "Sentiment"]).size().reset_index(name="count")
        fig_sent_topic = px.bar(sent_topic, x="topic", y="count", color="Sentiment", barmode="stack", color_discrete_map=sentiment_colors, title="Sentiment Distribution by Topic")
        fig_sent_topic = apply_dark_theme(fig_sent_topic)
        st.plotly_chart(fig_sent_topic, use_container_width=True)

        selected_topic = st.selectbox("Select Topic to Explore", options=sorted(filtered["topic"].dropna().unique()))
        topic_docs = filtered[filtered["topic"] == selected_topic][["title", "clean_text", "Sentiment", "score"]]
        st.write(f"### Documents for Topic {selected_topic}")
        st.dataframe(topic_docs, use_container_width=True)
    else:
        st.warning("Topic information is not available in the dataset.")
