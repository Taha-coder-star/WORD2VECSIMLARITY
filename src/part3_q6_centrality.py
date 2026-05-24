"""
Part 3 – Question 6: Centrality Analysis and Community Detection (5 marks)

Tasks:
  1. Compute degree, betweenness (normalized), closeness (normalized) centrality.
     Present top 15 nodes by betweenness in a table.
  2. Interpret top 5 betweenness-central nodes as conceptual bridges.
  3. Louvain community detection — number of communities + modularity score.
     Top 10 highest-degree words per community + financial label.
  4. Compare community structure (Q6) with k-means clusters (Q4).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import community as community_louvain   # python-louvain

from src.part3_q5_network import run as build_network

PLOTS_DIR  = "output/plots"
TABLES_DIR = "output/tables"

# Assign labels after inspecting output (update these)
COMMUNITY_LABELS = {}   # e.g. {0: "Cash Flow & Liquidity", 1: "Production & Operations", ...}


def run():
    # ── Load network from Q5 ──────────────────────────────────────────────────
    G = build_network()

    # Work on largest connected component for centrality
    largest_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"Working on largest CC: {largest_cc.number_of_nodes()} nodes, "
          f"{largest_cc.number_of_edges()} edges")

    # ── 1. Centrality measures ────────────────────────────────────────────────
    degree_c      = nx.degree_centrality(largest_cc)
    betweenness_c = nx.betweenness_centrality(largest_cc, normalized=True, weight="weight")
    closeness_c   = nx.closeness_centrality(largest_cc, distance="weight")

    centrality_df = pd.DataFrame({
        "Word":        list(degree_c.keys()),
        "Degree":      [round(degree_c[w], 6)      for w in degree_c],
        "Betweenness": [round(betweenness_c[w], 6) for w in degree_c],
        "Closeness":   [round(closeness_c[w], 6)   for w in degree_c],
    })

    top15_between = (centrality_df
                     .sort_values("Betweenness", ascending=False)
                     .head(15)
                     .reset_index(drop=True))
    top15_between.index += 1

    print("\nTop 15 nodes by Betweenness Centrality:")
    print(top15_between.to_string())
    top15_between.to_csv(os.path.join(TABLES_DIR, "q6_top15_betweenness.csv"))
    centrality_df.to_csv(os.path.join(TABLES_DIR, "q6_all_centrality.csv"), index=False)

    # ── 2. Interpret top 5 bridge words ──────────────────────────────────────
    print("\nTop 5 betweenness-central (bridge) words:")
    for i, row in top15_between.head(5).iterrows():
        print(f"  {i}. {row['Word']:<20s}  betweenness={row['Betweenness']:.6f}")

    # ── 3. Louvain community detection ────────────────────────────────────────
    partition   = community_louvain.best_partition(largest_cc, weight="weight", random_state=42)
    modularity  = community_louvain.modularity(partition, largest_cc, weight="weight")
    n_communities = len(set(partition.values()))

    print(f"\nLouvain communities detected: {n_communities}")
    print(f"Modularity score: {modularity:.4f}")

    # Top 10 words by degree within each community
    community_rows = []
    for comm_id in sorted(set(partition.values())):
        members   = [n for n, c in partition.items() if c == comm_id]
        subG      = largest_cc.subgraph(members)
        top10     = sorted(subG.degree(), key=lambda x: x[1], reverse=True)[:10]
        label     = COMMUNITY_LABELS.get(comm_id, f"Community {comm_id}")
        print(f"\n{label} ({len(members)} words):")
        print("  Top 10:", ", ".join(w for w, _ in top10))
        for rank, (word, deg) in enumerate(top10, 1):
            community_rows.append({"Community": comm_id, "Label": label,
                                    "Rank": rank, "Word": word, "Degree": deg})

    comm_df = pd.DataFrame(community_rows)
    comm_df.to_csv(os.path.join(TABLES_DIR, "q6_community_words.csv"), index=False)

    summary_df = pd.DataFrame([{"Communities": n_communities, "Modularity": round(modularity, 4)}])
    summary_df.to_csv(os.path.join(TABLES_DIR, "q6_louvain_summary.csv"), index=False)

    # ── 4. Community-coloured network plot ───────────────────────────────────
    degree_dict = dict(largest_cc.degree())
    top80_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:80]
    subG80      = largest_cc.subgraph(top80_nodes).copy()

    cmap    = cm.get_cmap("tab20", n_communities)
    colors  = [cmap(partition.get(n, 0)) for n in subG80.nodes()]
    sizes   = [degree_dict[n] * 15 for n in subG80.nodes()]
    pos     = nx.spring_layout(subG80, seed=42, k=0.5)

    fig, ax = plt.subplots(figsize=(18, 14))
    nx.draw_networkx_nodes(subG80, pos, node_color=colors, node_size=sizes, ax=ax)
    nx.draw_networkx_edges(subG80, pos, alpha=0.3, edge_color="grey", ax=ax)
    nx.draw_networkx_labels(subG80, pos, font_size=7, ax=ax)
    ax.set_title("Co-occurrence Network – Louvain Communities (Top 80 nodes)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "q6_community_network.png"), dpi=150)
    plt.show()
    print("Community network plot saved.")

    # ── Report section ────────────────────────────────────────────────────────
    from src import report_builder
    report_builder.save_section("06_part3_q6_centrality.md", """# Part 3, Q6: Centrality Analysis and Community Detection

## Top 15 Nodes by Betweenness Centrality
[TABLE](output/tables/q6_top15_betweenness.csv)

## Louvain Community Summary
[TABLE](output/tables/q6_louvain_summary.csv)

## Community Words (Top 10 per Community)
[TABLE](output/tables/q6_community_words.csv)

## Community Network Visualisation
![Community Network](output/plots/q6_community_network.png)

## Interpretation
Interpret the role of the top 5 betweenness-central bridge words. Compare the Louvain
communities with the k-means clusters from Q4 — do both approaches produce consistent
thematic groupings, or do they reveal different facets of Cisco's textual data? Does
the community structure reflect Cisco's underlying financial strategy?

[YOUR WRITTEN INTERPRETATION HERE]
""")
    report_builder.rebuild()

    return partition, modularity


if __name__ == "__main__":
    run()
