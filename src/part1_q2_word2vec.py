"""
Part 1 – Question 2: Training the Word2Vec Model (6 marks)

Tasks:
  1. Prepare input: one list of token lists (sentences) from cleaned corpus.
  2. Train Word2Vec skip-gram (sg=1), dim=100, window=5, min_count=5, epochs=10.
  3. Extract word-vector matrix; confirm dimensions.
  4. Top 5 words by L2 norm (vector magnitude).
  5. Rationale for architecture choices.
"""

import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from src.part1_q1_corpus import run as build_corpus

MODELS_DIR = "output/models"
TABLES_DIR = "output/tables"

SEED        = 42
VECTOR_SIZE = 100
WINDOW      = 5
MIN_COUNT   = 5
EPOCHS      = 10


def run():
    # ── 1. Get cleaned corpus from Q1 ─────────────────────────────────────────
    cleaned_docs, _, _ = build_corpus()

    # Gensim expects List[List[str]]
    sentences = list(cleaned_docs.values())
    print(f"Total documents (sentences) for Word2Vec: {len(sentences)}")
    total_tokens = sum(len(s) for s in sentences)
    print(f"Total tokens across corpus: {total_tokens:,}")

    # ── 2. Train Word2Vec skip-gram ────────────────────────────────────────────
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,           # 1 = skip-gram | 0 = CBOW
        epochs=EPOCHS,
        seed=SEED,
        workers=4,
    )
    model.save(os.path.join(MODELS_DIR, "word2vec.model"))
    print(f"\nModel trained and saved. Vocabulary: {len(model.wv):,} words")

    # ── 3. Word-vector matrix ─────────────────────────────────────────────────
    wv_matrix = model.wv.vectors          # shape: (vocab_size, VECTOR_SIZE)
    words     = list(model.wv.index_to_key)
    print(f"Word-vector matrix shape: {wv_matrix.shape}")

    # ── 4. Top 5 words by L2 norm ─────────────────────────────────────────────
    norms   = np.linalg.norm(wv_matrix, axis=1)
    top5_idx = np.argsort(norms)[::-1][:5]

    top5_df = pd.DataFrame({
        "Rank":    range(1, 6),
        "Word":    [words[i] for i in top5_idx],
        "L2 Norm": [norms[i] for i in top5_idx],
    })
    print("\nTop 5 words by L2 norm:")
    print(top5_df.to_string(index=False))
    top5_df.to_csv(os.path.join(TABLES_DIR, "q2_top5_l2norm.csv"), index=False)

    # ── 5. Architecture summary ───────────────────────────────────────────────
    arch = pd.DataFrame({
        "Parameter":   ["Model type", "Vector dimensions", "Window size",
                        "Min word frequency", "Training epochs", "Seed"],
        "Value":       ["Skip-gram (sg=1)", VECTOR_SIZE, WINDOW,
                        MIN_COUNT, EPOCHS, SEED],
        "Rationale":   [
            "Skip-gram predicts context from target — better for rare/domain terms in 10-K filings",
            "100 dims balances expressiveness and overfitting on a 5-doc corpus",
            "Window=5 captures phrase-level context typical in financial sentences",
            "min_count=5 removes hapax legomena that add noise in formal filings",
            "10 epochs sufficient for convergence on a small, dense financial vocabulary",
            "Fixed seed ensures reproducibility across runs",
        ],
    })
    arch.to_csv(os.path.join(TABLES_DIR, "q2_model_architecture.csv"), index=False)
    print("\nModel architecture rationale saved.")

    # ── Report section ────────────────────────────────────────────────────────
    from src import report_builder
    report_builder.save_section("02_part1_q2_word2vec.md", f"""# Part 1, Q2: Word2Vec Model Training

## Model Architecture
[TABLE](output/tables/q2_model_architecture.csv)

## Top 5 Words by L2 Norm
[TABLE](output/tables/q2_top5_l2norm.csv)

## Interpretation
Justify the chosen model architecture (skip-gram vs CBOW), vector dimensionality, and
window size in the context of 10-K disclosures. Explain why word embeddings provide
richer contextual information than raw term-frequency counts, and how window size affects
the type of linguistic relationships captured.

[YOUR WRITTEN INTERPRETATION HERE]
""")
    report_builder.rebuild()

    return model


if __name__ == "__main__":
    run()
