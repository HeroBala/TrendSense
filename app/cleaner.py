import os
import re
import spacy
import pandas as pd
import logging
import warnings
from typing import List, Optional
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from tqdm import tqdm
from spacy.cli import download as spacy_download

# ========== CONFIG ==========
INPUT_FILE = "data/ecommerce_advanced.json"
OUTPUT_FILE = "data/cleaned_data.xlsx"
COLUMNS_TO_CLEAN = ["text", "title"]
SPACY_MODEL = "en_core_web_sm"
N_PROCESS = 8

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========== WARNINGS ==========
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ========== LOAD NLP MODEL ==========
def load_spacy_model(model_name: str):
    try:
        logging.info(f"⏳ Loading spaCy model '{model_name}'...")
        return spacy.load(model_name, disable=["ner", "parser"])
    except OSError:
        logging.warning(f"⚠️ Model '{model_name}' not found. Attempting download...")
        spacy_download(model_name)
        return spacy.load(model_name, disable=["ner", "parser"])

nlp = load_spacy_model(SPACY_MODEL)
logging.info("✅ spaCy model ready.")

# ========== REGEX PATTERNS ==========
URL_REGEX = r"https?://\S+|www\.\S+"
EMOJI_REGEX = "[" \
            u"\U0001F600-\U0001F64F" \
            u"\U0001F300-\U0001F5FF" \
            u"\U0001F680-\U0001F6FF" \
            u"\U0001F1E0-\U0001F1FF" \
            "]+"
MARKDOWN_REGEX = r"(?m)^#{1,6}|\*|[-•]"
SPECIAL_REGEX = r"[^\w\s]"
MULTISPACE_REGEX = r"\s+"

URL_PATTERN = re.compile(URL_REGEX)
EMOJI_PATTERN = re.compile(EMOJI_REGEX)
MARKDOWN_PATTERN = re.compile(MARKDOWN_REGEX)
SPECIAL_PATTERN = re.compile(SPECIAL_REGEX)
MULTISPACE_PATTERN = re.compile(MULTISPACE_REGEX)

# ========== CUSTOM STOPWORDS ==========
CUSTOM_STOPWORDS = {
    "ecommerce", "store", "product", "buy", "sell", "amazon", "shopify"
}

# ========== CLEANING FUNCTIONS ==========

def basic_clean(text: Optional[str]) -> str:
    """Aggressive normalization: markdown, html, urls, emojis, symbols."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = MARKDOWN_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = SPECIAL_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.strip().lower()

def lemmatize_texts(texts: List[str]) -> List[str]:
    """Lemmatize using spaCy pipeline."""
    lemmatized = []
    logging.info(f"🔀 spaCy pipe starting with {len(texts)} texts (n_process={N_PROCESS})...")
    for doc in nlp.pipe(texts, batch_size=1000, n_process=N_PROCESS):
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and token.is_alpha and len(token) > 2 and token.lemma_ not in CUSTOM_STOPWORDS
        ]
        lemmatized.append(" ".join(tokens))
    logging.info("✅ Lemmatization complete.")
    return lemmatized

# ========== MAIN PROCESSING ==========
def clean_dataframe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")

        logging.info(f"🔄 Cleaning column: {col}")
        tqdm.pandas(desc=f"🧼 Cleaning {col}")
        df[f"{col}_pre"] = df[col].progress_apply(basic_clean)

        texts = df[f"{col}_pre"].tolist()
        if all(not t for t in texts):
            logging.warning(f"⚠️ Skipping '{col}' - all texts are empty after cleaning.")
            df[f"clean_{col}"] = ""
        else:
            df[f"clean_{col}"] = lemmatize_texts(texts)
            df[f"clean_{col}"] = df[f"clean_{col}"].str.replace(MULTISPACE_REGEX, " ", regex=True)

        # Show sample
        logging.info(f"🔍 Sample '{col}' clean preview:")
        print(df[[col, f"clean_{col}"]].head(3))

    return df

def save_excel(df: pd.DataFrame, path: str):
    if os.path.exists(path):
        logging.warning(f"⚠️ File already exists: {path}. Overwriting.")
        os.remove(path)
    df.to_excel(path, index=False, engine="openpyxl")
    logging.info(f"✅ Excel saved to: {path}")

# ========== EXECUTION ==========
if __name__ == "__main__":
    try:
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"❌ File not found: {INPUT_FILE}")

        logging.info(f"📥 Loading: {INPUT_FILE}")
        df = pd.read_json(INPUT_FILE)

        df_cleaned = clean_dataframe(df, COLUMNS_TO_CLEAN)
        save_excel(df_cleaned, OUTPUT_FILE)

    except Exception as e:
        logging.error(f"❌ Processing failed: {e}")