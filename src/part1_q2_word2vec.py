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
    os.makedirs(MODELS_DIR, exist_ok=True)
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
    report_builder.save_section("02_part1_q2_word2vec.md", f"""# Question 2: Training the Word2Vec Model

## Model Architecture and Parameters

[TABLE](output/tables/q2_model_architecture.csv)

## Word-Vector Matrix

The extracted word-vector matrix has dimensions {wv_matrix.shape[0]} x {wv_matrix.shape[1]}.
Each row is a dense vector representation of one term in the cleaned corpus.

## Top 5 Words by L2 Norm (Vector Magnitude)

[TABLE](output/tables/q2_top5_l2norm.csv)

Among the top 5 words by L2 norm, competitor and price are semantically meaningful
business and market terms whose high vector magnitudes reflect that they appear in
diverse and discriminative contexts across the five filings. Competitor is used
consistently in competitive landscape and risk factor discussions, while price relates
to product pricing, market share dynamics, and stock valuation language. The term factor
also carries substantive meaning as it frequently co-occurs with the risk-factor
disclosures required in 10-K Section 1A filings. In contrast, inc and filer are generic
SEC administrative terms: inc appears as a suffix in formal legal company name
references, while filer is an administrative designation used on SEC cover pages and in
exhibit sections. Their high L2 norms indicate that these terms appear in sufficiently
varied textual contexts to develop large vector magnitudes, but they represent artefacts
of the 10-K filing format rather than meaningful financial vocabulary and do not
materially affect the semantic structure of the model.

## Model Rationale and Interpretation

The skip-gram architecture (sg=1) was selected over CBOW because skip-gram trains by
predicting surrounding context words from a given target word. In 10-K filings, which
contain a large number of domain-specific and relatively infrequent financial terms such
as liquidity, impairment, and derivative, skip-gram allocates more gradient updates to
rare words than CBOW. CBOW averages context vectors to predict a target, which benefits
common words but tends to underfit specialised vocabulary. Since the goal is to capture
the semantic relationships of financial terminology, skip-gram is the more appropriate
choice for this corpus.

A vector dimensionality of 100 was chosen because it provides sufficient capacity to
encode the financial vocabulary of a five-document corpus without overfitting. Higher
dimensions such as 300 are justified when training on very large corpora where many fine-
grained distinctions can be learned. With only five annual reports, a smaller dimension
avoids representing noise as meaningful structure while still allowing semantically
distinct financial concepts to occupy different regions of the vector space.

A window size of 5 means each target word is trained against the five words immediately
preceding and following it in the text. In 10-K filings, financial concepts are typically
expressed within clause-level phrases of three to seven words. A window of 5 captures
financially meaningful collocations such as net operating loss, cash and cash equivalent,
and capital expenditure without extending into long-range associations that arise from the
repetitive structure of legal and accounting boilerplate. Smaller windows of one or two
would capture only morphological neighbours, while windows larger than ten would introduce
topical noise across sentence boundaries.

Word embeddings provide richer contextual information than raw term-frequency counts
because they encode the distributional hypothesis: words used in similar contexts receive
geometrically similar vector representations. In a standard Document-Term Matrix, revenue,
sales, and income are three independent dimensions with no defined relationship. In the
Word2Vec vector space, these terms cluster together because they consistently appear
alongside similar context words. This allows downstream analyses such as cosine similarity
retrieval and k-means clustering to discover semantically coherent financial groupings
rather than surface-level counting patterns.
""")
    report_builder.rebuild()

    return model


if __name__ == "__main__":
    run()
