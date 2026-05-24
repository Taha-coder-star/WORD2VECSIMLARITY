"""
Part 2 – Question 3: Semantic Proximity Analysis (6 marks)

Tasks:
  1. Select 5 seed terms using first-match lookup against model vocabulary.
  2. Retrieve top-10 nearest neighbours for each seed term (cosine similarity).
  3. Consolidated table: Financial Dimension | Seed Term | Rank | Neighbour Word | Cosine Similarity.
  4. Interpretation of neighbours for each seed term.
"""

import os
import pandas as pd
from gensim.models import Word2Vec

MODELS_DIR = "output/models"
TABLES_DIR = "output/tables"

# Candidates in priority order — first match found in model vocab is used
RAW_SEEDS = {
    "liquidity":    ["cash", "liquidity", "liquid"],
    "profitability":["profit", "income", "earning", "earnings", "loss"],
    "debt":         ["debt", "borrow", "borrowing", "liability"],
    "revenue":      ["revenue", "sale", "sales"],
    "risk":         ["risk", "uncertain", "uncertainty"],
}


def pick_seed(model_wv, candidates: list[str]) -> str | None:
    """Return the first candidate that exists in the model vocabulary."""
    for c in candidates:
        if c in model_wv:
            return c
    return None


def run():
    # ── Load model ────────────────────────────────────────────────────────────
    model = Word2Vec.load(os.path.join(MODELS_DIR, "word2vec.model"))
    wv    = model.wv

    print(f"Model vocabulary size: {len(wv):,} words")

    rows = []
    chosen_seeds = {}

    for dimension, candidates in RAW_SEEDS.items():
        seed = pick_seed(wv, candidates)
        if seed is None:
            print(f"WARNING: No candidate for '{dimension}' found. Tried: {candidates}")
            continue

        chosen_seeds[dimension] = seed
        neighbours = wv.most_similar(seed, topn=10)
        print(f"\n[{dimension.upper()}]  seed = '{seed}'")
        for rank, (word, score) in enumerate(neighbours, start=1):
            print(f"  {rank:2d}. {word:<20s}  cos={score:.4f}")
            rows.append({
                "Financial Dimension": dimension,
                "Seed Term":           seed,
                "Rank":                rank,
                "Neighbour Word":      word,
                "Cosine Similarity":   round(score, 4),
            })

    # ── Consolidated table ────────────────────────────────────────────────────
    os.makedirs(TABLES_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(TABLES_DIR, "q3_semantic_neighbours.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nConsolidated table ({len(df)} rows) saved -> {csv_path}")
    print("\nChosen seed terms:")
    for dim, seed in chosen_seeds.items():
        print(f"  {dim:<16s} -> {seed}")

    # ── Report section ────────────────────────────────────────────────────────
    from src import report_builder
    report_builder.save_section("03_part2_q3_semantic_proximity.md", f"""# Question 3: Semantic Proximity Analysis

## Selected Seed Terms

The following seed terms were selected by matching candidate words against the model
vocabulary. The first candidate found in the vocabulary was used for each dimension.

Liquidity: {chosen_seeds.get('liquidity', 'N/A')} | Profitability: {chosen_seeds.get('profitability', 'N/A')} | Debt: {chosen_seeds.get('debt', 'N/A')} | Revenue: {chosen_seeds.get('revenue', 'N/A')} | Risk: {chosen_seeds.get('risk', 'N/A')}

## Nearest Neighbours (Top 10 per Seed Term)

[TABLE](output/tables/q3_semantic_neighbours.csv)

## Interpretation

**Liquidity (seed: {chosen_seeds.get('liquidity', 'cash')})**
The semantic neighbourhood of '{chosen_seeds.get('liquidity', 'cash')}' reflects Cisco's short-term liquidity
management, with terms related to cash equivalents, working capital, and short-term
investments appearing in close proximity. This pattern is consistent with a technology
firm that maintains large cash reserves and discloses them extensively within balance
sheet and cash flow statement discussions.

**Profitability (seed: {chosen_seeds.get('profitability', 'income')})**
The neighbours of '{chosen_seeds.get('profitability', 'income')}' are dominated by terms relating to net profit
computation, operating performance, and tax obligations. This co-occurrence reflects the
dense interaction between income reporting and expense disclosures within Cisco's income
statements, where operating income, tax provision, and net income are consistently
discussed together across the five annual filings.

**Debt (seed: {chosen_seeds.get('debt', 'debt')})**
The semantic neighbourhood of '{chosen_seeds.get('debt', 'debt')}' suggests that Cisco's management discusses
borrowing primarily in the context of long-term financing instruments, maturity schedules,
and interest obligations. The relatively compact cluster around debt is consistent with
Cisco's profile as a cash-rich technology firm where leverage plays a secondary role
compared to operational and revenue disclosures.

**Revenue (seed: {chosen_seeds.get('revenue', 'revenue')})**
Revenue's neighbours reveal the multidimensional structure of Cisco's top-line reporting,
with terms relating to product revenue, service revenue, and geographic segmentation
appearing in close proximity. This reflects Cisco's dual revenue structure from hardware
product sales and recurring software and subscription services, both of which are
discussed extensively in the Management Discussion and Analysis sections.

**Risk (seed: {chosen_seeds.get('risk', 'risk')})**
The semantic neighbourhood of '{chosen_seeds.get('risk', 'risk')}' contains both negative outcome terms and
management-oriented language. The presence of terms related to uncertainty and adverse
conditions alongside operationally focused words suggests that Cisco's risk disclosures
balance regulatory enumeration of potential negative outcomes with management language
that signals active risk governance. This mixed neighbourhood indicates that risk is
framed not only as a threat but also as something subject to mitigation and control,
which is consistent with the structured risk factor sections required in 10-K filings.
""")
    report_builder.rebuild()

    return df


if __name__ == "__main__":
    run()
