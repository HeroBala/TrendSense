import os
import logging
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# ========== CONFIG ==========
DATA_PATH = Path("data/cleaned_data.xlsx")
MODEL_DIR = Path("models/bertopic_model")
SUMMARY_PATH = Path("models/topic_summary.xlsx")

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

# ========== 1. LOAD CLEANED DATA ==========
def load_cleaned_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Cleaned data not found at {DATA_PATH}")
    
    df = pd.read_excel(DATA_PATH)
    if 'clean_text' not in df.columns:
        raise ValueError("❌ Column 'clean_text' not found in dataset.")
    
    df = df.dropna(subset=['clean_text'])
    texts = df['clean_text'].astype(str).tolist()

    logging.info(f"📥 Loaded {len(texts)} documents for topic modeling.")
    return texts, df

# ========== 2. TRAIN BERTopic MODEL ==========
def train_topic_model(docs: list) -> BERTopic:
    logging.info("🧠 Initializing SentenceTransformer...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    logging.info("🚀 Training BERTopic model...")
    topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
    topics, probs = topic_model.fit_transform(docs)

    logging.info(f"✅ Model trained with {len(set(topics))} topics.")
    return topic_model

# ========== 3. SAVE MODEL & TOPIC SUMMARY ==========
def save_model_and_summary(topic_model: BERTopic, df: pd.DataFrame):
    logging.info("💾 Saving BERTopic model...")
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(MODEL_DIR)

    logging.info("📝 Generating topic summary...")
    topics_info = topic_model.get_topic_info()
    topics_info.to_excel(SUMMARY_PATH, index=False)
    logging.info(f"📄 Topic summary saved to {SUMMARY_PATH}")

    # Optional: append topics to DataFrame and show a preview
    topics, _ = topic_model.transform(df['clean_text'].astype(str).tolist())
    df['topic'] = topics
    print("\n🔍 Sample with assigned topics:")
    print(df[['clean_text', 'topic']].head())

# ========== MAIN ==========
if __name__ == "__main__":
    try:
        docs, df = load_cleaned_data()
        topic_model = train_topic_model(docs)
        save_model_and_summary(topic_model, df)
        logging.info("🎉 Topic modeling pipeline completed successfully.")
    except Exception as e:
        logging.error(f"❌ Error: {e}")

