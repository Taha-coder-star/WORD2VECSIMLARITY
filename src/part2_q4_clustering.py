"""
Part 2 – Question 4: Thematic Clustering of Word Vectors (6 marks)

Tasks:
  1. Filter word-vector matrix to top 300 words by corpus frequency.
  2. Normalize each vector to unit length.
  3. K-means clustering (k=5, nstart=25 -> n_init=25 in sklearn). Report WCSS.
  4. Top 10 representative words per cluster (closest to centroid).
     Assign financial labels to each cluster.
  5. PCA 2D scatter plot coloured by cluster, labelled with centroid words.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec

from src.utils import load_10k_files, preprocess

DATA_DIR   = "data"
MODELS_DIR = "output/models"
PLOTS_DIR  = "output/plots"
TABLES_DIR = "output/tables"

SEED  = 42
K     = 5
TOP_N = 300

# Manually assigned financial labels per cluster (update after inspecting output)
CLUSTER_LABELS = {
    0: "Derivatives & Equity Instruments",
    1: "Corporate Governance & Legal",
    2: "Asset Valuation & Accounting",
    3: "Revenue & Lease Accounting",
    4: "Strategic & Operational Context",
}

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]


def run():
    # ── Load model and corpus frequency ──────────────────────────────────────
    model    = Word2Vec.load(os.path.join(MODELS_DIR, "word2vec.model"))
    wv       = model.wv

    raw_docs  = load_10k_files(DATA_DIR)
    all_tokens = [t for text in raw_docs.values() for t in preprocess(text)]
    freq       = Counter(all_tokens)

    # ── 1. Filter to top 300 words present in model vocab ────────────────────
    top300_words = [w for w, _ in freq.most_common() if w in wv][:TOP_N]
    print(f"Using {len(top300_words)} words (top {TOP_N} by frequency in vocab)")

    vectors = np.array([wv[w] for w in top300_words])

    # ── 2. Normalize to unit length ───────────────────────────────────────────
    vectors_norm = normalize(vectors, norm="l2")

    # ── 3. K-means clustering ─────────────────────────────────────────────────
    np.random.seed(SEED)
    km     = KMeans(n_clusters=K, n_init=25, random_state=SEED)
    labels = km.fit_predict(vectors_norm)
    wcss   = km.inertia_

    print(f"\nK-means WCSS (Within-Cluster Sum of Squares): {wcss:.4f}")
    print("WCSS measures total variance within clusters — lower = more compact clusters.")

    # ── 4. Top 10 words per cluster (closest to centroid) ────────────────────
    cluster_rows = []
    top3 = {}
    for c in range(K):
        mask      = labels == c
        c_vectors = vectors_norm[mask]
        c_words   = [top300_words[i] for i, m in enumerate(mask) if m]
        centroid  = km.cluster_centers_[c]
        dists     = np.linalg.norm(c_vectors - centroid, axis=1)
        top10_idx = np.argsort(dists)[:10]
        top10_words = [c_words[i] for i in top10_idx]

        label = CLUSTER_LABELS.get(c, f"Cluster {c}")
        top3[c] = top10_words[:3]

        print(f"\nCluster {c} — {label}")
        print("  Words:", ", ".join(top10_words))

        for rank, w in enumerate(top10_words, 1):
            cluster_rows.append({"Cluster": c, "Label": label,
                                  "Rank": rank, "Word": w})

    # ── Save tables ───────────────────────────────────────────────────────────
    os.makedirs(TABLES_DIR, exist_ok=True)

    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(os.path.join(TABLES_DIR, "q4_representative_words.csv"), index=False)

    summary_df = pd.DataFrame([{"K": K, "WCSS": round(wcss, 4), "n_init": 25, "seed": SEED}])
    summary_df.to_csv(os.path.join(TABLES_DIR, "q4_cluster_summary.csv"), index=False)

    # ── 5. PCA scatter plot ───────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)

    pca    = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(vectors_norm)

    fig, ax = plt.subplots(figsize=(14, 10))

    for c in range(K):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COLORS[c], alpha=0.6, s=40,
                   label=CLUSTER_LABELS.get(c, f"Cluster {c}"))

    # Label the 10 centroid-closest words per cluster
    labeled = set()
    for row in cluster_rows:
        if row["Rank"] <= 10:
            idx = top300_words.index(row["Word"])
            if idx not in labeled:
                ax.annotate(row["Word"], (coords[idx, 0], coords[idx, 1]),
                            fontsize=7, alpha=0.85)
                labeled.add(idx)

    ax.set_title("Word Vector Clusters (PCA 2D) - k=5", fontsize=14)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "q4_pca_clusters.png"), dpi=150)
    plt.show()
    print("\nPCA scatter plot saved.")

    # ── Report section ────────────────────────────────────────────────────────
    interp_lines = []
    for c in range(K):
        label = CLUSTER_LABELS.get(c, f"Cluster {c}")
        words = ", ".join(top3.get(c, []))
        interp_lines.append(
            f"The '{label}' cluster groups words such as {words}, "
            f"reflecting semantic and contextual similarity in the Word2Vec vector space "
            f"for {label.lower()} vocabulary in Cisco's annual filings."
        )
    cluster_interp = "\n\n".join(interp_lines)

    from src import report_builder
    report_builder.save_section("04_part2_q4_clustering.md", f"""# Question 4: Thematic Clustering of Word Vectors

## Clustering Summary (WCSS)
[TABLE](output/tables/q4_cluster_summary.csv)

## Representative Words per Cluster (Top 10)
[TABLE](output/tables/q4_representative_words.csv)

## PCA 2D Scatter Plot
![Figure 2: Word Vector Clusters - k=5 (PCA projection)](output/plots/q4_pca_clusters.png)

## Interpretation

K-means clustering with k=5 applied to the normalised word-vector matrix of the top 300
corpus-frequency words produced a Within-Cluster Sum of Squares (WCSS) of {wcss:.4f}.
WCSS measures the total intra-cluster variance; a lower value indicates that the cluster
members are more tightly concentrated around their respective centroids, while a higher
value suggests that the vocabulary grouped within each cluster is semantically more
dispersed.

{cluster_interp}

The five clusters only partially align with the standard financial analysis categories
of liquidity, profitability, solvency, efficiency, and market performance. Most notably,
the 'Strategic & Operational Context' cluster does not map cleanly onto any single
standard financial ratio category; its representative words — including summarized,
compared, unit, gain, collaboration, guarantee, secure, internet, and federal — span
multiple domains covering operational metrics, strategic initiatives, and regulatory
language, indicating that this cluster captures a residual grouping of contextually
similar but thematically mixed vocabulary. The remaining four clusters show clearer
thematic coherence within their respective financial domains. The PCA projection
visualises this partial overlap, with clusters showing varying degrees of separation
along the first two principal components, which together capture a limited share of
total vector-space variance due to the high-dimensional and semantically dense nature
of financial disclosure language.
""")
    report_builder.rebuild()

    return km, labels, top300_words, vectors_norm


if __name__ == "__main__":
    run()
