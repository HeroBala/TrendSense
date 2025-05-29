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
def load_cleaned_text(data_path=DATA_PATH, text_col="clean_text", title_col="clean_title") -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"❌ File not found: {data_path}")

    df = pd.read_excel(data_path)
    if text_col not in df.columns or title_col not in df.columns:
        raise ValueError("❌ 'clean_text' and/or 'clean_title' columns missing.")

    df = df.dropna(subset=[text_col, title_col])
    df["text_for_model"] = df[title_col].fillna("") + " " + df[text_col].fillna("")
    logging.info(f"📥 Loaded {len(df)} cleaned documents for modeling.")
    return df


# ========== TRAIN BERTopic ==========
def train_topic_model() -> BERTopic:
    from hdbscan import HDBSCAN
    from umap import UMAP

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'),
        hdbscan_model=HDBSCAN(min_cluster_size=3, min_samples=1, metric="euclidean", prediction_data=True),
        calculate_probabilities=True,
        verbose=True
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
def run_analysis_pipeline():
    try:
        df = load_cleaned_text()
        docs = df["text_for_model"].tolist()
        docs = [doc for doc in docs if isinstance(doc, str) and len(doc.strip()) > 3]
        logging.info(f"📄 {len(docs)} valid documents going into BERTopic...")

        if not docs:
            raise ValueError("❌ No valid documents to process. Ensure texts are not empty or too short.")

        logging.info("📊 Starting topic modeling...")
        topic_model = train_topic_model()
        topics, probs = topic_model.fit_transform(docs)

        if not topics or all(t == -1 for t in topics):
            raise ValueError("❌ No valid topics generated. Try adjusting clustering parameters or input diversity.")

        df = df.iloc[:len(topics)].copy()
        df["topic"] = topics
        df["topic_prob"] = [max(p) if p is not None else 0 for p in probs]

        df = analyze_sentiment(df)
        save_outputs(topic_model, df)

        print("\n🔍 Sample Output:")
        print(df[["clean_text", "topic", "topic_prob", "sent_compound"]].head())
        logging.info("🎉 Full NLP pipeline completed successfully.")

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    run_analysis_pipeline()
