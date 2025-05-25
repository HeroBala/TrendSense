import logging
from pathlib import Path
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# ========== CONFIG ==========
DATA_PATH = Path("data/cleaned_data.xlsx")
MODEL_DIR = Path("models/bertopic_model")
SUMMARY_FILE = Path("models/topic_summary.xlsx")
LABELED_FILE = Path("models/labeled_data.xlsx")

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

# ========== LOAD CLEANED TEXT ==========
def load_cleaned_text(text_col="clean_text", title_col="clean_title") -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ File not found: {DATA_PATH}")
    
    df = pd.read_excel(DATA_PATH)
    if text_col not in df.columns or title_col not in df.columns:
        raise ValueError("❌ 'clean_text' and/or 'clean_title' columns missing.")
    
    df = df.dropna(subset=[text_col, title_col])
    df["text_for_model"] = df[title_col].fillna("") + " " + df[text_col].fillna("")
    logging.info(f"📥 Loaded {len(df)} cleaned posts.")
    return df

# ========== TRAIN BERTopic ==========
def train_topic_model() -> BERTopic:
    from hdbscan import HDBSCAN
    from umap import UMAP

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'),
        hdbscan_model=HDBSCAN(min_cluster_size=5, metric='euclidean'),
        verbose=False
    )
    return topic_model

# ========== SENTIMENT ANALYSIS ==========
def analyze_sentiment(df: pd.DataFrame, text_column="clean_text") -> pd.DataFrame:
    logging.info("🧠 Running VADER sentiment analysis...")
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    scores = df[text_column].apply(lambda x: sia.polarity_scores(str(x))).apply(pd.Series)
    scores.columns = ['sent_neg', 'sent_neu', 'sent_pos', 'sent_compound']
    df = pd.concat([df, scores], axis=1)
    logging.info("✅ Sentiment scores added.")
    return df

# ========== SAVE OUTPUT ==========
def save_outputs(model: BERTopic, df: pd.DataFrame):
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    logging.info("💾 Saving BERTopic model...")
    model.save(MODEL_DIR)

    logging.info("📝 Saving topic summary...")
    summary = model.get_topic_info()
    summary.to_excel(SUMMARY_FILE, index=False)

    logging.info("📊 Saving full labeled data...")
    df.to_excel(LABELED_FILE, index=False)

    logging.info("📁 Output saved to models/")

# ========== MAIN PIPELINE ==========
def main():
    try:
        df = load_cleaned_text()
        docs = df["text_for_model"].tolist()

        # Filter empty/short docs
        docs = [doc for doc in docs if isinstance(doc, str) and len(doc.strip()) > 10]
        if not docs:
            raise ValueError("❌ No valid documents to process. Ensure texts are not empty or too short.")

        logging.info(f"🧾 Transforming {len(docs)} documents...")

        # Topic modeling (fit + transform combined)
        topic_model = train_topic_model()
        topics, _ = topic_model.fit_transform(docs)
        df = df.iloc[:len(topics)].copy()
        df["topic"] = topics

        # Sentiment analysis
        df = analyze_sentiment(df)

        # Save everything
        save_outputs(topic_model, df)

        # Sample preview
        print("\n🔍 Sample:")
        print(df[["clean_text", "topic", "sent_compound"]].head())

        logging.info("🎉 Full NLP pipeline completed successfully.")

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()
