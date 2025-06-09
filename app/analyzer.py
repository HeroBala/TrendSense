import logging
from pathlib import Path
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
def load_cleaned_text() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ File not found: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)

    for col in ["clean_title", "clean_text"]:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    df["text_for_model"] = df["clean_title"].fillna("") + " " + df["clean_text"].fillna("")
    if "clean_comments" in df.columns:
        df["text_for_model"] += " " + df["clean_comments"].fillna("")

    df = df[df["text_for_model"].str.strip().astype(bool)]
    logging.info(f"📥 Loaded {len(df)} cleaned documents for modeling.")
    return df

# ========== TRAIN BERTopic ==========
def train_topic_model() -> BERTopic:
    from hdbscan import HDBSCAN
    from umap import UMAP
    return BERTopic(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'),
        hdbscan_model=HDBSCAN(min_cluster_size=3, min_samples=1, metric="euclidean", prediction_data=True),
        calculate_probabilities=True,
        verbose=True
    )

# ========== SENTIMENT ANALYSIS ==========
def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    scores = df["clean_text"].astype(str).apply(sia.polarity_scores).apply(pd.Series)
    df = pd.concat([df, scores], axis=1)
    df["sentiment"] = df["compound"].apply(lambda s: "Positive" if s >= 0.05 else "Negative" if s <= -0.05 else "Neutral")
    return df

# ========== DOCUMENT QUERYING ==========
def query_similar_docs(model: BERTopic, docs: list, query: str, top_k: int = 3):
    model_embeddings = model.embedding_model.encode(docs, batch_size=64, show_progress_bar=False)
    query_vec = model.embedding_model.encode([query], show_progress_bar=False)
    sims = cosine_similarity(query_vec, model_embeddings)[0]
    top_idxs = np.argsort(sims)[-top_k:][::-1]
    return [(docs[i], round(sims[i], 3)) for i in top_idxs]

# ========== SAVE OUTPUT ==========
def save_outputs(model: BERTopic, df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR)
    model.get_topic_info().to_excel(SUMMARY_FILE, index=False)

    # TF-IDF per topic
    tfidf_terms = []
    for topic in sorted(df["topic"].unique()):
        topic_texts = df[df["topic"] == topic]["clean_text"].dropna().astype(str)
        if len(topic_texts) < 5:
            tfidf_terms.append({"topic": topic, "tfidf_terms": "N/A"})
            continue

        vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
        try:
            matrix = vectorizer.fit_transform(topic_texts)
            terms = vectorizer.get_feature_names_out()
            tfidf_terms.append({"topic": topic, "tfidf_terms": ", ".join(terms) if len(terms) else "N/A"})
        except:
            tfidf_terms.append({"topic": topic, "tfidf_terms": "N/A"})

    tfidf_df = pd.DataFrame(tfidf_terms)
    df = df.merge(tfidf_df, on="topic", how="left")

    output_cols = [
        "clean_title", "clean_text", "clean_comments", "topic", "topic_prob",
        "sentiment", "neg", "neu", "pos", "compound", "score", "num_comments", "tfidf_terms"
    ]
    for col in output_cols:
        if col not in df.columns:
            df[col] = ""
    df[output_cols].to_excel(LABELED_FILE, index=False)
    logging.info("✅ Labeled data with TF-IDF saved.")

# ========== MAIN PIPELINE ==========
def run_pipeline():
    try:
        df = load_cleaned_text()
        docs = df["text_for_model"].tolist()

        if not docs:
            raise ValueError("No valid documents found for modeling.")

        model = train_topic_model()
        topics, probs = model.fit_transform(docs)
        df["topic"] = topics
        df["topic_prob"] = [max(p) if p is not None else 0 for p in probs]

        df = analyze_sentiment(df)
        save_outputs(model, df)

        logging.info("🎯 Top topics:")
        for topic in model.get_topic_freq().head(5)["Topic"]:
            if topic != -1:
                terms = model.get_topic(topic)
                logging.info(f"Topic {topic}: {', '.join(w for w, _ in terms[:TOP_WORDS_PER_TOPIC])}")

        results = query_similar_docs(model, docs, QUERY_EXAMPLE)
        for i, (text, score) in enumerate(results):
            print(f"{i+1}. ({score}) {text[:100]}...")

        logging.info("🎉 Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")

# ========== ENTRY ==========
if __name__ == "__main__":
    run_pipeline()
