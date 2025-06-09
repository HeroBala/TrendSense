# Import the os module for interacting with the operating system (file checks, removals)
import os
# Import re for regular expressions (pattern matching and text cleaning)
import re
# Import spacy for advanced natural language processing (NLP)
import spacy
# Import pandas as pd for data manipulation and analysis (DataFrames)
import pandas as pd
# Import logging for logging messages and information
import logging
# Import warnings to manage and filter warning messages
import warnings
# Import List and Optional from typing for type hinting
from typing import List, Optional
# Import BeautifulSoup for HTML parsing and MarkupResemblesLocatorWarning for warning control
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
# Import tqdm for progress bars in loops (progress_apply)
from tqdm import tqdm
# Import spaCy's download utility to download models if not present
from spacy.cli import download as spacy_download

# ========== CONFIG ==========
INPUT_FILE = "data/ecommerce_advanced.json"      # Path to input data file
OUTPUT_FILE = "data/cleaned_data.xlsx"           # Path to output Excel file
COLUMNS_TO_CLEAN = ["text", "title"]             # Columns in data to be cleaned
SPACY_MODEL = "en_core_web_sm"                   # spaCy model to use
N_PROCESS = 8                                    # Number of processes for spaCy pipeline

# ========== LOGGING ==========
logging.basicConfig(                             # Configure logging format and level
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========== WARNINGS ==========
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)  # Ignore specific BeautifulSoup warnings

# ========== LOAD NLP MODEL ==========
def load_spacy_model(model_name: str):
    """Load a spaCy model, downloading it if necessary."""
    try:
        logging.info(f"⏳ Loading spaCy model '{model_name}'...")           # Info log for loading model
        return spacy.load(model_name, disable=["ner", "parser"])           # Load spaCy model (disable NER/parser for speed)
    except OSError:
        logging.warning(f"⚠️ Model '{model_name}' not found. Attempting download...") # Warn if model not found
        spacy_download(model_name)                                         # Download spaCy model
        return spacy.load(model_name, disable=["ner", "parser"])           # Load downloaded model

nlp = load_spacy_model(SPACY_MODEL)            # Load the specified spaCy model
logging.info("✅ spaCy model ready.")           # Log model is ready

# ========== REGEX PATTERNS ==========
URL_REGEX = r"https?://\S+|www\.\S+"           # Matches URLs
EMOJI_REGEX = "[" \
            u"\U0001F600-\U0001F64F" \
            u"\U0001F300-\U0001F5FF" \
            u"\U0001F680-\U0001F6FF" \
            u"\U0001F1E0-\U0001F1FF" \
            "]+"                               # Matches emojis
MARKDOWN_REGEX = r"(?m)^#{1,6}|\*|[-•]"        # Matches markdown syntax
SPECIAL_REGEX = r"[^\w\s]"                     # Matches special characters (not word or whitespace)
MULTISPACE_REGEX = r"\s+"                      # Matches multiple spaces

URL_PATTERN = re.compile(URL_REGEX)             # Compiled regex for URLs
EMOJI_PATTERN = re.compile(EMOJI_REGEX)         # Compiled regex for emojis
MARKDOWN_PATTERN = re.compile(MARKDOWN_REGEX)   # Compiled regex for markdown
SPECIAL_PATTERN = re.compile(SPECIAL_REGEX)     # Compiled regex for special characters
MULTISPACE_PATTERN = re.compile(MULTISPACE_REGEX) # Compiled regex for multiple spaces

# ========== CUSTOM STOPWORDS ==========
CUSTOM_STOPWORDS = {
    "ecommerce", "store", "product", "buy", "sell", "amazon", "shopify"
}  # Set of custom words to exclude during NLP processing

# ========== CLEANING FUNCTIONS ==========

def basic_clean(text: Optional[str]) -> str:
    """Aggressive normalization: markdown, html, urls, emojis, symbols."""
    if not isinstance(text, str):                       # If the input is not a string, return empty string
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()# Remove HTML tags using BeautifulSoup
    text = MARKDOWN_PATTERN.sub(" ", text)              # Remove markdown syntax
    text = URL_PATTERN.sub(" ", text)                   # Remove URLs
    text = EMOJI_PATTERN.sub(" ", text)                 # Remove emojis
    text = SPECIAL_PATTERN.sub(" ", text)               # Remove special characters
    text = MULTISPACE_PATTERN.sub(" ", text)            # Normalize multiple spaces to single space
    return text.strip().lower()                         # Trim spaces and convert to lowercase

def lemmatize_texts(texts: List[str]) -> List[str]:
    """Lemmatize using spaCy pipeline."""
    lemmatized = []                                     # List to hold lemmatized texts
    logging.info(f"🔀 spaCy pipe starting with {len(texts)} texts (n_process={N_PROCESS})...") # Log start
    for doc in nlp.pipe(texts, batch_size=1000, n_process=N_PROCESS): # Process texts in batches, parallelized
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and token.is_alpha and len(token) > 2 and token.lemma_ not in CUSTOM_STOPWORDS
        ]                                               # Filter out stopwords, non-alphabetic, short tokens, custom stopwords
        lemmatized.append(" ".join(tokens))              # Join tokens into string
    logging.info("✅ Lemmatization complete.")           # Log completion
    return lemmatized                                   # Return list of lemmatized strings

# ========== MAIN PROCESSING ==========
def clean_dataframe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Clean and lemmatize specified columns in a DataFrame."""
    for col in columns:
        if col not in df.columns:                       # Check if required column exists
            raise KeyError(f"Missing required column: '{col}'")
        logging.info(f"🔄 Cleaning column: {col}")       # Log which column is being cleaned
        tqdm.pandas(desc=f"🧼 Cleaning {col}")           # Set up tqdm progress bar for pandas
        df[f"{col}_pre"] = df[col].progress_apply(basic_clean) # Apply basic cleaning, save intermediate
        texts = df[f"{col}_pre"].tolist()                # Get cleaned texts as list
        if all(not t for t in texts):                    # If all cleaned texts are empty
            logging.warning(f"⚠️ Skipping '{col}' - all texts are empty after cleaning.")
            df[f"clean_{col}"] = ""                      # Set cleaned col to empty
        else:
            df[f"clean_{col}"] = lemmatize_texts(texts)  # Lemmatize cleaned texts
            df[f"clean_{col}"] = df[f"clean_{col}"].str.replace(MULTISPACE_REGEX, " ", regex=True) # Normalize spaces
        # Show sample
        logging.info(f"🔍 Sample '{col}' clean preview:") # Log preview
        print(df[[col, f"clean_{col}"]].head(3))         # Print first 3 rows of original + cleaned
    return df                                            # Return cleaned DataFrame

def save_excel(df: pd.DataFrame, path: str):
    """Save DataFrame to Excel file, overwriting if exists."""
    if os.path.exists(path):                             # Check if file exists
        logging.warning(f"⚠️ File already exists: {path}. Overwriting.")
        os.remove(path)                                  # Remove existing file
    df.to_excel(path, index=False, engine="openpyxl")    # Save DataFrame to Excel
    logging.info(f"✅ Excel saved to: {path}")            # Log save

# ========== EXECUTION ==========
if __name__ == "__main__":                              # If script is run directly
    try:
        if
