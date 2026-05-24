# Question 5: Co-occurrence Network Construction

## Network Statistics
[TABLE](output/tables/q5_network_summary.csv)

## Network Visualisation (Top 80 Nodes by Degree)
![Figure 3: 10-K Co-occurrence Network – Top 80 Nodes by Degree](output/plots/q5_network_top80.png)

All 80 nodes are plotted. Node labels are shown for the top 25 highest-degree terms only to maintain readability. Node size and colour intensity represent degree centrality; edge width represents co-occurrence weight.

## Interpretation

The co-occurrence network was constructed from Cisco's five-year 10-K corpus using a
sliding window of 5 tokens and a minimum co-occurrence threshold of 10,
yielding a graph of 2858 nodes and 56753 edges. Self-loops and isolated nodes
were removed before calculating network statistics. The full graph forms a single connected component (2858 nodes), so average path length is computed across all nodes.

Nodes (2858): The 2858 nodes represent the set of lemmatised vocabulary
terms that meet the minimum co-occurrence threshold with at least one other term; this
count reflects the breadth of contextually active financial vocabulary retained after
preprocessing and threshold filtering.

Edges (56753): The 56753 weighted edges represent term pairs that appear
within a 5-token window at least 10 times across the corpus, capturing
the density of direct contextual associations in Cisco's annual disclosure language.

Density (0.013901): A network density of 0.013901 confirms that the
co-occurrence graph is sparse — only a small fraction of all possible term pairs appear
in close proximity, which is expected in large-vocabulary financial text where most
terms are domain-specific and concentrated in particular sections of the filing.

Average Path Length (2.5946, measured on full graph):
An average path length of approximately 2.59 steps indicates a small-world
property in the co-occurrence graph, meaning any two vocabulary terms are reachable
through a small number of intermediate hub terms — a structural pattern consistent with
the interconnected nature of financial disclosure language.

Global Clustering Coefficient (0.1848, computed using transitivity):
A clustering coefficient of 0.1848 — computed using the global transitivity
measure, which is the ratio of closed triangles to all connected triples — indicates
that terms connected to a common neighbour tend to also be connected to each other,
forming tight vocabulary triangles consistent with recurrent phrase patterns in 10-K
filings.

The five highest-degree nodes — product, customer, service, financial, market — function as hub terms in Cisco's
co-occurrence vocabulary, bridging multiple financial topics and appearing across diverse
contextual positions throughout the corpus. Their high degree centrality reflects their
role as cross-domain connectors in the disclosure language, linking specialised
financial subvocabularies such as balance sheet items, income statement terminology, and
risk factor language that are each densely interconnected within their own clusters but
meet at these central hub terms.
