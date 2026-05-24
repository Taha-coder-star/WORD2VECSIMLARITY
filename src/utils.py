"""
Shared preprocessing utilities used across all parts.
"""

import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Custom domain-irrelevant 10-K stopwords ──────────────────────────────────
CUSTOM_STOPWORDS = {
    "page", "exhibit", "pursuant", "herein", "thereto", "registrant",
    "incorporated", "commission", "fiscal", "form", "item", "section",
    "annual", "report", "filed", "filing", "whereas", "thereof",
}

STEMMER = SnowballStemmer("english")


def load_10k_files(data_dir: str) -> dict[str, str]:
    """Load all .txt 10-K files from data_dir. Returns {filename: raw_text}."""
    files = {}
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(data_dir, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                files[fname] = f.read()
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")
    return files


def preprocess(text: str, stem: bool = True) -> list[str]:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove punctuation, numbers, special characters
      3. Tokenize
      4. Remove standard + custom stopwords
      5. Stem (SnowballStemmer)
    Returns list of cleaned tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()

    all_stopwords = set(stopwords.words("english")) | CUSTOM_STOPWORDS
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]

    if stem:
        tokens = [STEMMER.stem(t) for t in tokens]

    return tokens


def tokens_to_sentence(tokens: list[str]) -> str:
    """Join token list into a single space-separated string (for Word2Vec input)."""
    return " ".join(tokens)
