# ========== IMPORTS ==========
# Standard library imports
import os  # File system operations (e.g., file removal)
import re  # Regular expressions for text pattern cleaning
import logging  # Logging status and errors
import warnings  # Suppress specific warnings
from typing import List, Optional  # Type hints for better readability and static checking

# External libraries
import spacy  # Natural Language Processing toolkit
import pandas as pd  # Data loading and manipulation
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning  # HTML parsing
from tqdm import tqdm  # Progress bar for pandas operations
from spacy.cli import download as spacy_download  # Auto-download spaCy models
from langdetect import detect  # Language detection utility
import contractions  # Expand contractions (e.g., don't → do not)
from textblob import TextBlob  # For spelling correction
import argparse  # Command-line interface
import yaml  # Configuration management

# ========== CONFIGURATION LOADING ==========
# Load preprocessing configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

INPUT_FILE = config["input_file"]
OUTPUT_FILE = config["output_file"]
COLUMNS_TO_CLEAN = config["columns_to_clean"]
SPACY_MODEL = config["spacy_model"]
N_PROCESS = config["n_process"]

# ========== LOGGING SETUP ==========
# Log messages to file for debugging and traceability
logging.basicConfig(
    filename="preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========== WARNING FILTER ==========
# Ignore warnings triggered by BeautifulSoup's parser
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ========== LOAD SPACY MODEL ==========
def load_spacy_model(model_name: str):
    """Load or download the specified spaCy model."""
    try:
        logging.info(f"Loading spaCy model '{model_name}'...")
        return spacy.load(model_name)
    except OSError:
        logging.warning(f"Model '{model_name}' not found. Downloading...")
        spacy_download(model_name)
        return spacy.load(model_name)

nlp = load_spacy_model(SPACY_MODEL)
logging.info("spaCy model loaded.")

# ========== REGEX PATTERNS FOR CLEANING ==========
# Precompiled regex for speed and reuse
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMOJI_PATTERN = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+")
MARKDOWN_PATTERN = re.compile(r"(?m)^#{1,6}|\*|[-\u2022]")
SPECIAL_PATTERN = re.compile(r"[^\w\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")

# Custom domain-specific stopwords
CUSTOM_STOPWORDS = {"ecommerce", "store", "product", "buy", "sell", "amazon", "shopify"}

# ========== CLEANING FUNCTIONS ==========
def is_english(text: str) -> bool:
    """Detect if the given text is in English."""
    try:
        return detect(text) == 'en'
    except:
        return False

def correct_spelling(text: str) -> str:
    """Use TextBlob to correct spelling in the text."""
    return str(TextBlob(text).correct())

def basic_clean(text: Optional[str]) -> str:
    """Apply a series of regex and library-based cleaning steps."""
    if not isinstance(text, str):
        return ""
    text = contractions.fix(text)  # Expand contractions
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML
    text = MARKDOWN_PATTERN.sub(" ", text)  # Remove markdown
    text = URL_PATTERN.sub(" ", text)  # Remove URLs
    text = EMOJI_PATTERN.sub(" ", text)  # Remove emojis
    text = SPECIAL_PATTERN.sub(" ", text)  # Remove special characters
    text = MULTISPACE_PATTERN.sub(" ", text)  # Normalize whitespace
    return text.strip().lower()  # Lowercasing for normalization

def tokenize(text: str) -> List[str]:
    """Tokenize cleaned text using simple whitespace split."""
    return text.split()

def lemmatize_texts(texts: List[str]) -> List[str]:
    """Apply POS-filtered lemmatization using spaCy."""
    lemmatized = []
    logging.info(f"Starting spaCy pipeline on {len(texts)} texts...")
    for doc in nlp.pipe(texts, batch_size=1000, n_process=N_PROCESS):
        tokens = [
            token.lemma_ for token in doc
            if token.pos_ in {"NOUN", "VERB", "ADJ"} and  # POS filtering
               token.is_alpha and  # Keep alphabetic words only
               not token.is_stop and  # Remove stopwords
               token.ent_type_ == "" and  # Remove named entities
               len(token) > 2 and  # Filter short tokens
               token.lemma_ not in CUSTOM_STOPWORDS  # Custom domain stopwords
        ]
        lemmatized.append(" ".join(tokens))
    logging.info("Lemmatization complete.")
    return lemmatized

def clean_dataframe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Execute full cleaning pipeline for selected columns."""
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")
        logging.info(f"Cleaning column: {col}")
        tqdm.pandas(desc=f"Cleaning {col}")

        # Language filtering
        df = df[df[col].apply(lambda x: is_english(str(x)))]

        # Basic and spelling cleaning
        df[f"{col}_pre"] = df[col].progress_apply(basic_clean)
        df[f"{col}_pre"] = df[f"{col}_pre"].progress_apply(correct_spelling)

        # Tokenization
        df[f"{col}_tokens"] = df[f"{col}_pre"].apply(tokenize)

        # Lemmatization
        texts = df[f"{col}_pre"].tolist()
        df[f"clean_{col}"] = lemmatize_texts(texts)
        df[f"clean_{col}"] = df[f"clean_{col}"].str.replace(MULTISPACE_PATTERN, " ", regex=True)

        logging.info(f"Sample cleaned data from column '{col}':")
        print(df[[col, f"clean_{col}"]].head(3))
    return df

def save_excel(df: pd.DataFrame, path: str):
    """Export cleaned DataFrame to Excel format."""
    if os.path.exists(path):
        logging.warning(f"File exists and will be overwritten: {path}")
        os.remove(path)
    df.to_excel(path, index=False, engine="openpyxl")
    logging.info(f"Excel saved to: {path}")

# ========== SCRIPT EXECUTION ==========
if __name__ == "__main__":
    # Command-line argument parsing for input/output control
    parser = argparse.ArgumentParser(description="Text Preprocessing Pipeline")
    parser.add_argument("--input", type=str, help="Path to input JSON file", default=INPUT_FILE)
    parser.add_argument("--output", type=str, help="Path to output Excel file", default=OUTPUT_FILE)
    args = parser.parse_args()

    try:
        # Load input data and run cleaning pipeline
        df = pd.read_json(args.input)
        df_cleaned = clean_dataframe(df, COLUMNS_TO_CLEAN)
        save_excel(df_cleaned, args.output)
    except Exception as e:
        logging.error(f"Processing failed: {e}")
