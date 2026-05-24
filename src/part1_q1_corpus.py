"""
Part 1 – Question 1: Corpus Construction and Text Preprocessing (6 marks)

Tasks:
  1. Load all five 10-K text files into a named corpus.
  2. Preprocess: lowercase → remove punctuation/numbers/special chars →
     remove stopwords (standard + 10 custom) → stem.
  3. Vocabulary size before and after preprocessing.
  4. Top 30 most frequent terms: frequency table + word cloud.
  5. Document-Term Matrix (DTM) — report dimensions.
  6. Written interpretation of top-30 terms.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from src.utils import load_10k_files, preprocess, CUSTOM_STOPWORDS

DATA_DIR   = "data"
PLOTS_DIR  = "output/plots"
TABLES_DIR = "output/tables"


def run():
    # ── 1. Load corpus ────────────────────────────────────────────────────────
    raw_docs = load_10k_files(DATA_DIR)
    doc_names = list(raw_docs.keys())
    print(f"Loaded {len(raw_docs)} documents: {doc_names}")

    # ── 2. Vocabulary size BEFORE preprocessing ───────────────────────────────
    all_raw_tokens = []
    for text in raw_docs.values():
        all_raw_tokens.extend(text.lower().split())
    vocab_before = len(set(all_raw_tokens))
    print(f"Vocabulary size BEFORE preprocessing: {vocab_before:,}")

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    cleaned_docs = {}
    for name, text in raw_docs.items():
        cleaned_docs[name] = preprocess(text, stem=True)

    all_clean_tokens = [t for tokens in cleaned_docs.values() for t in tokens]
    vocab_after = len(set(all_clean_tokens))
    print(f"Vocabulary size AFTER  preprocessing: {vocab_after:,}")
    print(f"Reduction: {vocab_before - vocab_after:,} terms removed")

    # ── 4. Top 30 most frequent terms ─────────────────────────────────────────
    freq = Counter(all_clean_tokens)
    top30 = freq.most_common(30)

    top30_df = pd.DataFrame(top30, columns=["Term", "Frequency"])
    top30_df.index += 1
    print("\nTop 30 Terms:")
    print(top30_df.to_string())
    top30_df.to_csv(os.path.join(TABLES_DIR, "q1_top30_terms.csv"), index=True)

    # Word cloud
    wc = WordCloud(width=1200, height=600, background_color="white",
                   max_words=100, colormap="Blues")
    wc.generate_from_frequencies(dict(freq))
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud – Combined 10-K Corpus", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "q1_wordcloud.png"), dpi=150)
    plt.show()
    print("Word cloud saved.")

    # ── 5. Document-Term Matrix ────────────────────────────────────────────────
    cleaned_strings = [" ".join(tokens) for tokens in cleaned_docs.values()]
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(cleaned_strings)

    print(f"\nDTM dimensions: {dtm.shape[0]} documents × {dtm.shape[1]} unique terms")

    # ── 6. Vocab summary table ────────────────────────────────────────────────
    summary = pd.DataFrame({
        "Metric": ["Vocab before preprocessing", "Vocab after preprocessing",
                   "Terms removed", "DTM documents", "DTM unique terms"],
        "Value":  [vocab_before, vocab_after, vocab_before - vocab_after,
                   dtm.shape[0], dtm.shape[1]],
    })
    summary.to_csv(os.path.join(TABLES_DIR, "q1_vocab_summary.csv"), index=False)
    print(summary.to_string(index=False))

    # ── Report section ────────────────────────────────────────────────────────
    from src import report_builder
    report_builder.save_section("01_part1_q1_corpus.md", f"""# Part 1, Q1: Corpus Construction and Text Preprocessing

## Vocabulary Summary
[TABLE](output/tables/q1_vocab_summary.csv)

## Top 30 Most Frequent Terms
[TABLE](output/tables/q1_top30_terms.csv)

## Word Cloud
![Word Cloud](output/plots/q1_wordcloud.png)

## Interpretation
Do the top 30 terms reflect the financial and operational character of Cisco Systems, or do
generic administrative terms dominate? Discuss what the high-frequency vocabulary reveals
about management communication emphasis across the five filings.

[YOUR WRITTEN INTERPRETATION HERE]
""")
    report_builder.rebuild()

    return cleaned_docs, dtm, vectorizer


if __name__ == "__main__":
    run()
