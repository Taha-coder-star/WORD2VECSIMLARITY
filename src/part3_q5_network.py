"""
Part 3 – Question 5: Constructing the Co-occurrence Network (6 marks)

Tasks:
  1. Build Feature Co-occurrence Matrix (FCM) with window=5, min_cooc=10.
  2. Convert FCM to an undirected weighted networkx graph.
     Remove self-loops and isolated nodes.
  3. Network statistics: nodes, edges, density, avg path length, clustering coefficient.
  4. Visualise top 80 nodes by degree (size=degree centrality, edge width=weight).
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter, defaultdict

from src.utils import load_10k_files, preprocess

DATA_DIR   = "data"
PLOTS_DIR  = "output/plots"
TABLES_DIR = "output/tables"

WINDOW      = 5
MIN_COOC    = 10
TOP_NODES   = 80
TOP_LABELS  = 25   # nodes to label in the plot; remaining nodes plotted unlabelled


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
    raw_docs    = load_10k_files(DATA_DIR)
    token_lists = [preprocess(text) for text in raw_docs.values()]
    print(f"Documents: {len(token_lists)}")

    # ── 1. Build co-occurrence matrix ─────────────────────────────────────────
    cooc = build_cooccurrence(token_lists, window=WINDOW, min_count=MIN_COOC)
    print(f"Co-occurrence pairs (min_cooc={MIN_COOC}): {len(cooc):,}")

    # ── 2. Build networkx graph ───────────────────────────────────────────────
    G = nx.Graph()
    for (w1, w2), weight in cooc.items():
        G.add_edge(w1, w2, weight=weight)

    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"Graph after cleanup — Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")

    # ── 3. Network statistics ─────────────────────────────────────────────────
    density = nx.density(G)

    largest_cc  = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    avg_path    = nx.average_shortest_path_length(largest_cc)
    clustering  = nx.transitivity(G)

    n_nodes      = G.number_of_nodes()
    n_edges      = G.number_of_edges()
    n_components = nx.number_connected_components(G)
    is_disconnected = n_components > 1
    lcc_size     = len(largest_cc)

    stats_rows = [
        ("Nodes",                         str(n_nodes)),
        ("Edges",                         str(n_edges)),
        ("Density",                       f"{density:.6f}"),
        ("Avg Path Length (largest CC)",  f"{avg_path:.4f}"),
        ("Global Clustering Coefficient", f"{clustering:.4f}"),
    ]
    stats_df = pd.DataFrame(stats_rows, columns=["Statistic", "Value"])
    print("\nNetwork Statistics:")
    print(stats_df.to_string(index=False))

    os.makedirs(TABLES_DIR, exist_ok=True)
    stats_df.to_csv(os.path.join(TABLES_DIR, "q5_network_summary.csv"), index=False)

    # ── Variables for dynamic section ─────────────────────────────────────────
    degree_dict = dict(G.degree())
    top_hubs    = sorted(degree_dict, key=degree_dict.get, reverse=True)[:5]
    hub_str     = ", ".join(top_hubs)

    connectivity_note = (
        f"The full graph contains {n_components} connected components; average path "
        f"length was therefore computed on the largest component ({lcc_size} nodes)."
        if is_disconnected else
        f"The full graph forms a single connected component ({lcc_size} nodes), so "
        f"average path length is computed across all nodes."
    )

    # ── 4. Visualise top 80 nodes by degree ───────────────────────────────────
    top80_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:TOP_NODES]
    subG        = G.subgraph(top80_nodes).copy()

    pos        = nx.spring_layout(subG, seed=42, k=1.5)
    dc         = nx.degree_centrality(subG)
    node_dc    = [dc[n] for n in subG.nodes()]
    node_sizes = [max(80, v * 5000) for v in node_dc]

    edges        = list(subG.edges(data=True))
    edge_weights = [e[2].get("weight", 1) for e in edges]
    max_w        = max(edge_weights) if edge_weights else 1
    edge_widths  = [0.2 + 1.2 * (w / max_w) for w in edge_weights]

    # Label only top TOP_LABELS nodes; all 80 are still drawn
    top_label_set = set(sorted(degree_dict, key=degree_dict.get, reverse=True)[:TOP_LABELS])
    labels_shown  = {n: n for n in subG.nodes() if n in top_label_set}

    fig, ax = plt.subplots(figsize=(16, 12))
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes,
                           node_color=node_dc, cmap=cm.Blues,
                           vmin=0, vmax=max(node_dc), ax=ax)
    nx.draw_networkx_edges(subG, pos, width=edge_widths,
                           alpha=0.25, edge_color="grey", ax=ax)
    nx.draw_networkx_labels(subG, pos, labels=labels_shown,
                            font_size=9, font_weight="bold", ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cm.Blues,
                               norm=plt.Normalize(vmin=0, vmax=max(node_dc)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Degree Centrality",
                 fraction=0.02, pad=0.02, shrink=0.6)
    ax.set_title(f"10-K Co-occurrence Network – Top {TOP_NODES} Nodes by Degree",
                 fontsize=14)
    ax.axis("off")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "q5_network_top80.png"),
                dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Network plot saved  (figsize=16x12, dpi=300, {TOP_NODES} nodes plotted, "
          f"{TOP_LABELS} nodes labelled).")

    # ── Report section ────────────────────────────────────────────────────────
    from src import report_builder
    report_builder.save_section("05_part3_q5_network.md", f"""# Question 5: Co-occurrence Network Construction

## Network Statistics
[TABLE](output/tables/q5_network_summary.csv)

## Network Visualisation (Top 80 Nodes by Degree)
![Figure 3: 10-K Co-occurrence Network – Top {TOP_NODES} Nodes by Degree](output/plots/q5_network_top80.png)

All {TOP_NODES} nodes are plotted. Node labels are shown for the top {TOP_LABELS} highest-degree terms only to maintain readability. Node size and colour intensity represent degree centrality; edge width represents co-occurrence weight.

## Interpretation

The co-occurrence network was constructed from Cisco's five-year 10-K corpus using a
sliding window of {WINDOW} tokens and a minimum co-occurrence threshold of {MIN_COOC},
yielding a graph of {n_nodes} nodes and {n_edges} edges. Self-loops and isolated nodes
were removed before calculating network statistics. {connectivity_note}

Nodes ({n_nodes}): The {n_nodes} nodes represent the set of lemmatised vocabulary
terms that meet the minimum co-occurrence threshold with at least one other term; this
count reflects the breadth of contextually active financial vocabulary retained after
preprocessing and threshold filtering.

Edges ({n_edges}): The {n_edges} weighted edges represent term pairs that appear
within a {WINDOW}-token window at least {MIN_COOC} times across the corpus, capturing
the density of direct contextual associations in Cisco's annual disclosure language.

Density ({density:.6f}): A network density of {density:.6f} confirms that the
co-occurrence graph is sparse — only a small fraction of all possible term pairs appear
in close proximity, which is expected in large-vocabulary financial text where most
terms are domain-specific and concentrated in particular sections of the filing.

Average Path Length ({round(avg_path, 4)}, measured on {'largest component' if is_disconnected else 'full graph'}):
An average path length of approximately {avg_path:.2f} steps indicates a small-world
property in the co-occurrence graph, meaning any two vocabulary terms are reachable
through a small number of intermediate hub terms — a structural pattern consistent with
the interconnected nature of financial disclosure language.

Global Clustering Coefficient ({round(clustering, 4)}, computed using transitivity):
A clustering coefficient of {clustering:.4f} — computed using the global transitivity
measure, which is the ratio of closed triangles to all connected triples — indicates
that terms connected to a common neighbour tend to also be connected to each other,
forming tight vocabulary triangles consistent with recurrent phrase patterns in 10-K
filings.

The five highest-degree nodes — {hub_str} — function as hub terms in Cisco's
co-occurrence vocabulary, bridging multiple financial topics and appearing across diverse
contextual positions throughout the corpus. Their high degree centrality reflects their
role as cross-domain connectors in the disclosure language, linking specialised
financial subvocabularies such as balance sheet items, income statement terminology, and
risk factor language that are each densely interconnected within their own clusters but
meet at these central hub terms.
""")
    report_builder.rebuild()

    return G


if __name__ == "__main__":
    run()
