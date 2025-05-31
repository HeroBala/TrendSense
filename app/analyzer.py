# analyzer.py

import logging
from pathlib import Path
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ========== CONFIG ==========
DATA_PATH = Path("data/cleaned_data.xlsx")
MODEL_DIR = Path("models/bertopic_model")
SUMMARY_FILE = Path("models/topic_summary.xlsx")
LABELED_FILE = Path("models/labeled_data.xlsx")
TOP_WORDS_PER_TOPIC = 10
QUERY_EXAMPLE = "shipping issues and customer service"

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

# ========== LOAD CLEANED TEXT ==========
def load_cleaned_text(data_path=DATA_PATH, text_col="clean_text", title_col="clean_title") -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"❌ File not found: {data_path}")
    df = pd.read_excel(data_path)
    if text_col not in df.columns or title_col not in df.columns:
        raise ValueError("❌ 'clean_text' and/or 'clean_title' columns missing.")
    df = df.dropna(subset=[text_col, title_col])
    df["text_for_model"] = df[title_col].fillna("") + " " + df[text_col].fillna("")
    df = df[df["text_for_model"].str.strip().astype(bool)]
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
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    scores = df[text_column].apply(lambda x: sia.polarity_scores(str(x))).apply(pd.Series)
    scores.columns = ['sent_neg', 'sent_neu', 'sent_pos', 'sent_compound']
    return pd.concat([df, scores], axis=1)

# ========== TOPIC EXPLAINABILITY ==========
def print_topic_words(model: BERTopic, top_n: int = TOP_WORDS_PER_TOPIC):
    logging.info("🔍 Top words per topic:")
    for topic in model.get_topic_freq().head(10)["Topic"]:
        if topic == -1:
            continue
        words = ", ".join([word for word, _ in model.get_topic(topic)[:top_n]])
        logging.info(f"Topic {topic}: {words}")

# ========== DOCUMENT QUERYING ==========
def query_similar_docs(model: BERTopic, docs: list, query: str, top_k: int = 3):
    embeddings = model.embedding_model.encode(docs)
    query_vec = model.embedding_model.encode([query])
    similarities = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(docs[i], round(similarities[i], 3)) for i in top_indices]

# ========== SAVE OUTPUT ==========
def save_outputs(model: BERTopic, df: pd.DataFrame):
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR)
    model.get_topic_info().to_excel(SUMMARY_FILE, index=False)
    df.to_excel(LABELED_FILE, index=False)
    logging.info("📁 Outputs saved in 'models/'")

# ========== MAIN PIPELINE ==========
def run_analysis_pipeline():
    try:
        df = load_cleaned_text()
        docs = df["text_for_model"].tolist()
        if not docs:
            raise ValueError("❌ No valid documents to process.")

        logging.info("📊 Starting topic modeling...")
        topic_model = train_topic_model()
        topics, probs = topic_model.fit_transform(docs)

        df = df.iloc[:len(topics)].copy()
        df["topic"] = topics
        df["topic_prob"] = [max(p) if p is not None else 0 for p in probs]

        df = analyze_sentiment(df)
        save_outputs(topic_model, df)
        print_topic_words(topic_model)

        # Query demonstration
        query_results = query_similar_docs(topic_model, docs, QUERY_EXAMPLE)
        print("\n🔎 Query Example Results:")
        for i, (text, score) in enumerate(query_results):
            print(f"{i+1}. ({score}) {text[:120]}...")

        logging.info("🎉 NLP analysis completed successfully.")

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    run_analysis_pipeline()
