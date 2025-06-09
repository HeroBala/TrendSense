# Import the logging module for status and error messages.
import logging
# Import Path for handling filesystem paths.
from pathlib import Path
# Import pandas for data manipulation and analysis.
import pandas as pd
# Import BERTopic for topic modeling.
from bertopic import BERTopic
# Import SentenceTransformer for generating text embeddings.
from sentence_transformers import SentenceTransformer
# Import SentimentIntensityAnalyzer from NLTK for sentiment analysis.
from nltk.sentiment import SentimentIntensityAnalyzer
# Import nltk to manage NLTK resources (like datasets).
import nltk
# Import NumPy for numerical operations.
import numpy as np
# Import cosine_similarity for computing similarity between vectors.
from sklearn.metrics.pairwise import cosine_similarity

# ========== CONFIG ==========

# Path to the cleaned data Excel file.
DATA_PATH = Path("data/cleaned_data.xlsx")
# Path to save/load the BERTopic model.
MODEL_DIR = Path("models/bertopic_model")
# Path to save the summary of topics.
SUMMARY_FILE = Path("models/topic_summary.xlsx")
# Path to save the labeled data output.
LABELED_FILE = Path("models/labeled_data.xlsx")
# Number of top words to show per topic.
TOP_WORDS_PER_TOPIC = 10
# Example query string for document search.
QUERY_EXAMPLE = "shipping issues and customer service"

# ========== LOGGING ==========

# Set up logging with INFO level and custom format for timestamps and log level.
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

# ========== LOAD CLEANED TEXT ==========

# Function to load cleaned text data from Excel file.
def load_cleaned_text() -> pd.DataFrame:
    # Check if the data file exists.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ File not found: {DATA_PATH}")
    # Load the Excel file into a DataFrame.
    df = pd.read_excel(DATA_PATH)

    # Ensure required columns are present.
    for col in ["clean_title", "clean_text"]:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # Combine title and text for modeling.
    df["text_for_model"] = df["clean_title"].fillna("") + " " + df["clean_text"].fillna("")
    # If cleaned comments are available, add them to the text.
    if "clean_comments" in df.columns:
        df["text_for_model"] += " " + df["clean_comments"].fillna("")

    # Remove rows with empty combined text.
    df = df[df["text_for_model"].str.strip().astype(bool)]
    # Log the number of loaded documents.
    logging.info(f"📥 Loaded {len(df)} cleaned documents for modeling.")
    # Return the DataFrame.
    return df

# ========== TRAIN BERTopic ==========

# Function to initialize and return a BERTopic model.
def train_topic_model() -> BERTopic:
    # Import HDBSCAN for clustering within topic modeling.
    from hdbscan import HDBSCAN
    # Import UMAP for dimensionality reduction.
    from umap import UMAP
    # Return a configured BERTopic model.
    return BERTopic(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"), # Use a mini language model for embeddings.
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'), # Set UMAP params.
        hdbscan_model=HDBSCAN(min_cluster_size=3, min_samples=1, metric="euclidean", prediction_data=True), # Set HDBSCAN params.
        calculate_probabilities=True, # Calculate topic probabilities.
        verbose=True # Output progress.
    )

# ========== SENTIMENT ANALYSIS ==========

# Function to analyze sentiment of documents using NLTK's VADER.
def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # Download the VADER lexicon if needed.
    nltk.download("vader_lexicon", quiet=True)
    # Initialize the SentimentIntensityAnalyzer.
    sia = SentimentIntensityAnalyzer()
    # Compute sentiment scores for each document's text.
    scores = df["clean_text"].astype(str).apply(sia.polarity_scores).apply(pd.Series)
    # Concatenate the scores to the original DataFrame.
    df = pd.concat([df, scores], axis=1)
    # Classify sentiment as Positive, Negative, or Neutral.
    df["sentiment"] = df["compound"].apply(lambda s: "Positive" if s >= 0.05 else "Negative" if s <= -0.05 else "Neutral")
    # Return the updated DataFrame.
    return df

# ========== DOCUMENT QUERYING ==========

# Function to find documents most similar to a query using embeddings and cosine similarity.
def query_similar_docs(model: BERTopic, docs: list, query: str, top_k: int = 3):
    # Encode all documents into vectors.
    model_embeddings = model.embedding_model.encode(docs, batch_size=64, show_progress_bar=False)
    # Encode the query string into a vector.
    query_vec = model.embedding_model.encode([query], show_progress_bar=False)
    # Compute cosine similarity between the query and all documents.
    sims = cosine_similarity(query_vec, model_embeddings)[0]
    # Get indices of top_k most similar documents.
    top_idxs = np.argsort(sims)[-top_k:][::-1]
    # Return the top documents and their similarity scores.
    return [(docs[i], round(sims[i], 3)) for i in top_idxs]

# ========== SAVE OUTPUT ==========

# Function to save the model, topic summaries, and labeled data.
def save_outputs(model: BERTopic, df: pd.DataFrame):
    # Import TfidfVectorizer for extracting top terms per topic.
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Ensure the models directory exists.
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    # Save the BERTopic model.
    model.save(MODEL_DIR)
    # Save topic summary to Excel.
    model.get_topic_info().to_excel(SUMMARY_FILE, index=False)

    # TF-IDF per topic
    tfidf_terms = []
    # For each topic, extract top TF-IDF terms.
    for topic in sorted(df["topic"].unique()):
        topic_texts = df[df["topic"] == topic]["clean_text"].dropna().astype(str)
        # If not enough documents, mark as N/A.
        if len(topic_texts) < 5:
            tfidf_terms.append({"topic": topic, "tfidf_terms": "N/A"})
            continue

        # Initialize the vectorizer (top 10 terms, English stopwords).
        vectorizer = TfidfVectorizer(max_features=10, stop_words="english")
        try:
            # Fit and transform the topic's texts.
            matrix = vectorizer.fit_transform(topic_texts)
            # Get top feature names (terms).
            terms = vectorizer.get_feature_names_out()
            # Add the terms to the list.
            tfidf_terms.append({"topic": topic, "tfidf_terms": ", ".join(terms) if len(terms) else "N/A"})
        except:
            # In case of failure, mark as N/A.
            tfidf_terms.append({"topic": topic, "tfidf_terms": "N/A"})

    # Convert the list of TF-IDF terms to a DataFrame.
    tfidf_df = pd.DataFrame(tfidf_terms)
    # Merge TF-IDF terms into the main DataFrame.
    df = df.merge(tfidf_df, on="topic", how="left")

    # List of output columns to be saved.
    output_cols = [
        "clean_title", "clean_text", "clean_comments", "topic", "topic_prob",
        "sentiment", "neg", "neu", "pos", "compound", "score", "num_comments", "tfidf_terms"
    ]
    # Ensure all output columns exist in the DataFrame.
    for col in output_cols:
        if col not in df.columns:
            df[col] = ""
    # Save the labeled data to Excel.
    df[output_cols].to_excel(LABELED_FILE, index=False)
    # Log that the data was saved.
    logging.info("✅ Labeled data with TF-IDF saved.")

# ========== MAIN PIPELINE ==========

# The main function that runs the entire pipeline.
def run_pipeline():
    try:
        # Load cleaned data.
        df = load_cleaned_text()
        # Get list of texts for modeling.
        docs = df["text_for_model"].tolist()

        # If no documents found, raise error.
        if not docs:
            raise ValueError("No valid documents found for modeling.")

        # Train and fit the BERTopic model.
        model = train_topic_model()
        topics, probs = model.fit_transform(docs)
        # Assign topics and probabilities to DataFrame.
        df["topic"] = topics
        df["topic_prob"] = [max(p) if p is not None else 0 for p in probs]

        # Analyze sentiment of documents.
        df = analyze_sentiment(df)
        # Save outputs (model, summaries, labeled data).
        save_outputs(model, df)

        # Log the top topics.
        logging.info("🎯 Top topics:")
        for topic in model.get_topic_freq().head(5)["Topic"]:
            if topic != -1:
                terms = model.get_topic(topic)
                logging.info(f"Topic {topic}: {', '.join(w for w, _ in terms[:TOP_WORDS_PER_TOPIC])}")

        # Run query example and print results.
        results = query_similar_docs(model, docs, QUERY_EXAMPLE)
        for i, (text, score) in enumerate(results):
            print(f"{i+1}. ({score}) {text[:100]}...")

        # Log successful completion.
        logging.info("🎉 Pipeline completed successfully.")

    except Exception as e:
        # Log any errors that occur during the pipeline.
        logging.error(f"❌ Pipeline failed: {e}")

# ========== ENTRY ==========

# Entry point: run the pipeline if this script is called directly.
if __name__ == "__main__":
    run_pipeline()
