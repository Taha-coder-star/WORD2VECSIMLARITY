"""
Shared preprocessing utilities used across all parts.
"""

import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords",  quiet=True)
nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)
nltk.download("wordnet",    quiet=True)
nltk.download("omw-1.4",    quiet=True)

# ── Custom domain-irrelevant 10-K stopwords ──────────────────────────────────
# Applied BEFORE and AFTER lemmatization to catch all surface forms.
CUSTOM_STOPWORDS = {
    # 10-K admin / boilerplate
    "page", "exhibit", "pursuant", "herein", "thereto", "registrant",
    "incorporated", "commission", "fiscal", "form", "table", "contents",
    "exchange", "section", "act", "annual", "issuer",
    "file", "filed", "filing", "number", "yes", "no",
    "whereas", "thereof", "therein", "hereby", "therefore",
    # Modal / legal verbs
    "may", "could", "would", "shall", "also",
    # Include family
    "include", "includes", "included", "including",
    # Provide family
    "provide", "provided", "provides",
    # Related family
    "related", "relating", "relate",
    # Generic result / statement / note
    "total", "result", "results", "statement", "statements",
    "note", "notes", "part", "certain", "respect",
    # Common filler verbs / prepositions
    "use", "used", "using", "due", "per",
    # Time / period boilerplate
    "year", "years", "july", "end", "period",
    # Quantity words that add no semantic signal
    "amount", "million", "billion",
    # Item / report noise
    "item", "items", "report",
}

LEMMATIZER = WordNetLemmatizer()


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
      1. Strip residual HTML tags
      2. Lowercase
      3. Remove punctuation, numbers, special characters
      4. Tokenize
      5. Remove standard English stopwords + custom stopwords (raw form)
      6. Lemmatize (noun POS — keeps words readable, handles plurals)
      7. Remove custom stopwords again (post-lemma base forms)
      8. Drop tokens shorter than 3 characters
    The `stem` parameter is kept for API compatibility but lemmatization
    is always applied for readability.
    """
    # 1. Strip HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # 2. Lowercase
    text = text.lower()
    # 3. Remove non-alpha
    text = re.sub(r"[^a-z\s]", " ", text)
    # 4. Tokenize
    tokens = text.split()

    all_stopwords = set(stopwords.words("english")) | CUSTOM_STOPWORDS

    # 5. Pre-lemma stopword filter + length guard
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]

    # 6. Lemmatize (noun form handles plurals; catches most financial nouns)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    # 7. Post-lemma stopword filter (catches base forms revealed by lemmatization)
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]

    return tokens


def tokens_to_sentence(tokens: list[str]) -> str:
    """Join token list into a single space-separated string (for Word2Vec input)."""
    return " ".join(tokens)
