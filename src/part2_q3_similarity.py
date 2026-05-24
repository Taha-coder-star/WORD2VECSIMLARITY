"""
Part 2 – Question 3: Semantic Proximity Analysis (6 marks)

Tasks:
  1. Select 5 seed terms: liquidity, profitability, debt, revenue, risk.
     Use the stemmed form that exists in the model vocabulary.
  2. Retrieve top-10 nearest neighbours for each seed term (cosine similarity).
  3. Consolidated table: Seed Term | Rank | Neighbour Word | Cosine Similarity.
  4. Interpretation of neighbours for each seed term.
"""

import os
import pandas as pd
from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer

MODELS_DIR = "output/models"
TABLES_DIR = "output/tables"

STEMMER = SnowballStemmer("english")

# Seed terms — stemmed forms; adjust if not in model vocabulary
RAW_SEEDS = {
    "liquidity":    ["liquid", "cash", "liquidit"],
    "profitability":["profit", "earn", "profitabl"],
    "debt":         ["debt", "borrow", "liabil"],
    "revenue":      ["revenu", "sale", "incom"],
    "risk":         ["risk", "uncertain", "exposur"],
}


def pick_seed(model_wv, candidates: list[str]) -> str | None:
    """Return the first candidate that exists in the model vocabulary."""
    for c in candidates:
        if c in model_wv:
            return c
        stemmed = STEMMER.stem(c)
        if stemmed in model_wv:
            return stemmed
    return None


def run():
    # ── Load model ────────────────────────────────────────────────────────────
    model = Word2Vec.load(os.path.join(MODELS_DIR, "word2vec.model"))
    wv    = model.wv

    rows = []
    for dimension, candidates in RAW_SEEDS.items():
        seed = pick_seed(wv, candidates)
        if seed is None:
            print(f"WARNING: No candidate for '{dimension}' found in vocabulary. "
                  f"Tried: {candidates}")
            continue

        neighbours = wv.most_similar(seed, topn=10)
        print(f"\n[{dimension.upper()}] seed='{seed}'")
        for rank, (word, score) in enumerate(neighbours, start=1):
            print(f"  {rank:2d}. {word:<20s}  cos={score:.4f}")
            rows.append({
                "Dimension":        dimension,
                "Seed Term":        seed,
                "Rank":             rank,
                "Neighbour Word":   word,
                "Cosine Similarity": round(score, 4),
            })

    # ── Consolidated table ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "q3_semantic_neighbours.csv"), index=False)
    print(f"\nConsolidated table ({len(df)} rows) saved to output/tables/q3_semantic_neighbours.csv")

    # ── Report section ────────────────────────────────────────────────────────
    from src import report_builder
    report_builder.save_section("03_part2_q3_similarity.md", """# Part 2, Q3: Semantic Proximity Analysis

## Nearest Neighbours (Top 10 per Seed Term)
[TABLE](output/tables/q3_semantic_neighbours.csv)

## Interpretation
For each seed term, discuss whether the retrieved neighbours reflect genuine financial
meaning or reveal noise in management language. Pay particular attention to the 'risk'
neighbourhood: are the closest words predominantly negative (default, loss, impairment)
or do they include hedging/mitigation language (manage, mitigate, control)?

[YOUR WRITTEN INTERPRETATION HERE]
""")
    report_builder.rebuild()

    return df


if __name__ == "__main__":
    run()
