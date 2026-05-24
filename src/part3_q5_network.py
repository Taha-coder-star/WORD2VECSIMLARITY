"""
Part 3 – Question 5: Constructing the Co-occurrence Network (6 marks)

Tasks:
  1. Build Feature Co-occurrence Matrix (FCM) with window=5, min_cooc=10.
  2. Convert FCM to an undirected weighted networkx graph.
     Remove self-loops and isolate vertices.
  3. Network statistics: nodes, edges, density, avg path length, clustering coefficient.
  4. Visualise top 80 nodes by degree (size=degree, edge width=weight).
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter, defaultdict
from sklearn.preprocessing import normalize

from src.utils import load_10k_files, preprocess

DATA_DIR   = "data"
PLOTS_DIR  = "output/plots"
TABLES_DIR = "output/tables"

WINDOW    = 5
MIN_COOC  = 10
TOP_NODES = 80


def build_cooccurrence(token_lists: list[list[str]],
                       window: int = 5,
                       min_count: int = 10) -> dict:
    """Sliding-window co-occurrence counter."""
    cooc = defaultdict(int)
    for tokens in token_lists:
        for i, w in enumerate(tokens):
            start = max(0, i - window)
            end   = min(len(tokens), i + window + 1)
            for j in range(start, end):
                if i != j:
                    pair = tuple(sorted([w, tokens[j]]))
                    cooc[pair] += 1
    return {pair: cnt for pair, cnt in cooc.items() if cnt >= min_count}


def run():
    # ── Load & preprocess corpus ──────────────────────────────────────────────
    raw_docs     = load_10k_files(DATA_DIR)
    token_lists  = [preprocess(text) for text in raw_docs.values()]
    print(f"Documents: {len(token_lists)}")

    # ── 1. Build co-occurrence matrix ─────────────────────────────────────────
    cooc = build_cooccurrence(token_lists, window=WINDOW, min_count=MIN_COOC)
    print(f"Co-occurrence pairs (min_cooc={MIN_COOC}): {len(cooc):,}")

    # ── 2. Build networkx graph ───────────────────────────────────────────────
    G = nx.Graph()
    for (w1, w2), weight in cooc.items():
        G.add_edge(w1, w2, weight=weight)

    G.remove_edges_from(nx.selfloop_edges(G))
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    print(f"Graph after cleanup — Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")

    # ── 3. Network statistics ─────────────────────────────────────────────────
    density = nx.density(G)

    # avg path length on largest connected component (full graph may be disconnected)
    largest_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    avg_path   = nx.average_shortest_path_length(largest_cc) if nx.is_connected(largest_cc) else "N/A (disconnected)"
    clustering = nx.transitivity(G)

    stats = {
        "Nodes":                        G.number_of_nodes(),
        "Edges":                        G.number_of_edges(),
        "Density":                      round(density, 6),
        "Avg Path Length (largest CC)": avg_path if isinstance(avg_path, str) else round(avg_path, 4),
        "Global Clustering Coefficient":round(clustering, 4),
    }
    stats_df = pd.DataFrame(list(stats.items()), columns=["Statistic", "Value"])
    print("\nNetwork Statistics:")
    print(stats_df.to_string(index=False))
    stats_df.to_csv(os.path.join(TABLES_DIR, "q5_network_stats.csv"), index=False)

    # ── 4. Visualise top 80 nodes by degree ───────────────────────────────────
    degree_dict = dict(G.degree())
    top80_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:TOP_NODES]
    subG = G.subgraph(top80_nodes).copy()

    pos          = nx.spring_layout(subG, seed=42, k=0.5)
    node_degrees = [degree_dict[n] for n in subG.nodes()]
    node_sizes   = [d * 15 for d in node_degrees]

    edges        = list(subG.edges(data=True))
    edge_weights = [e[2].get("weight", 1) for e in edges]
    max_w        = max(edge_weights) if edge_weights else 1
    edge_widths  = [0.5 + 3.0 * (w / max_w) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(18, 14))
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes,
                           node_color=node_degrees, cmap=cm.Blues, ax=ax)
    nx.draw_networkx_edges(subG, pos, width=edge_widths,
                           alpha=0.4, edge_color="grey", ax=ax)
    nx.draw_networkx_labels(subG, pos, font_size=7, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cm.Blues,
                               norm=plt.Normalize(vmin=min(node_degrees),
                                                  vmax=max(node_degrees)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Degree Centrality")
    ax.set_title(f"10-K Co-occurrence Network – Top {TOP_NODES} Nodes by Degree",
                 fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "q5_network.png"), dpi=150)
    plt.show()
    print("Network plot saved.")

    # ── Report section ────────────────────────────────────────────────────────
    from src import report_builder
    report_builder.save_section("05_part3_q5_network.md", """# Part 3, Q5: Co-occurrence Network Construction

## Network Statistics
[TABLE](output/tables/q5_network_stats.csv)

## Network Visualisation (Top 80 Nodes by Degree)
![Co-occurrence Network](output/plots/q5_network.png)

## Interpretation
Is the network dense or sparse? What does the global clustering coefficient indicate
about how tightly interconnected the vocabulary of Cisco's 10-K filings is? Identify
the visually apparent hub words and discuss the financial concepts they represent.

[YOUR WRITTEN INTERPRETATION HERE]
""")
    report_builder.rebuild()

    return G


if __name__ == "__main__":
    run()
