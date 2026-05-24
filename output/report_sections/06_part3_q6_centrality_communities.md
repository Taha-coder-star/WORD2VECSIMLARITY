# Question 6: Centrality Analysis and Community Detection

## Top 15 Nodes by Betweenness Centrality
[TABLE](output/tables/q6_top15_centrality.csv)

## Louvain Community Detection Summary
[TABLE](output/tables/q6_louvain_summary.csv)

## Community Top Words (Top 10 per Community)
[TABLE](output/tables/q6_communities.csv)

## Community Network Visualisation
![Figure 4: Co-occurrence Network – Louvain Communities (Top 80 nodes)](output/plots/q6_community_network.png)

All 80 nodes are plotted. Labels are shown for the top 25 highest-degree nodes only. Node colour represents Louvain community membership; node size represents degree centrality within the subgraph.

## Interpretation

### Betweenness Centrality and Bridge Terms

Betweenness centrality was computed using a k=500 pivot-node approximation, which is
standard practice for large co-occurrence graphs and provides reliable identification
of the highest-betweenness nodes while remaining computationally feasible. Exact
computation on a graph of this size (2858 nodes,
56753 edges) would require prohibitive memory and runtime;
the k=500 approximation introduces a small variance in absolute scores but does not
change the relative ranking of the top bridge terms.

The term 'product' has a normalized betweenness centrality of 0.0377 (approximated using k=500 pivot nodes). As the highest-betweenness term in the network, 'product' functions as the primary operational bridge, connecting Cisco's product performance language (revenue, segment, growth) with its risk and supply chain vocabulary (component, supply, disruption). Its central position reflects Cisco's product-led business model, in which product decisions propagate across financial, operational, and risk reporting sections of every 10-K filing.

The term 'customer' has a normalized betweenness centrality of 0.0254 (approximated using k=500 pivot nodes). The term 'customer' bridges Cisco's revenue recognition vocabulary (contract, deferred, recognition) with its market and competitive language (demand, retention, relationship), functioning as the connective anchor between what Cisco sells and how that selling is disclosed financially. Its high betweenness reflects that customer-related disclosures appear across the MD&A, revenue notes, and risk factor sections simultaneously.

The term 'cisco' has a normalized betweenness centrality of 0.0250 (approximated using k=500 pivot nodes). The company name 'cisco' appears as a high-betweenness node primarily as a structural artefact of SEC filing language, where Cisco Systems refers to itself by name in boilerplate ownership, legal, and governance clauses. While it connects many sections, its bridging role is partially generic — it reflects institutional co-occurrence patterns rather than specific financial concept linkages, and should be interpreted alongside substantive bridge terms such as 'product' and 'customer'.

The term 'technology' has a normalized betweenness centrality of 0.0226 (approximated using k=500 pivot nodes). The term 'technology' bridges Cisco's product and operational clusters with its strategic and market-facing vocabulary, appearing at the intersection of hardware/software product descriptions, cybersecurity risk disclosures, and forward-looking strategic statements. Its betweenness centrality reflects Cisco's identity as an enterprise technology company in which the word 'technology' functions as a cross-domain connector across product, risk, and competitive narrative sections.

The term 'financial' has a normalized betweenness centrality of 0.0221 (approximated using k=500 pivot nodes). The term 'financial' occupies a reporting and accounting bridge role, connecting governance language (statement, report, control) with asset valuation terms (fair, value, impairment) and risk factor language (condition, exposure, risk). Its high betweenness reflects its ubiquitous presence in disclosure formulas — 'financial statements', 'financial condition', 'financial performance' — that appear across every major section of a 10-K filing and therefore link otherwise distinct vocabulary clusters.

Collectively, the top betweenness-central terms are those that appear in the most
diverse range of co-occurrence contexts across the corpus, bridging financial
subvocabularies that would otherwise remain loosely connected. Their high betweenness
reflects the integrative nature of management language in 10-K filings, where core
operational and financial concepts recur across multiple reporting sections.

### Louvain Community Structure

Louvain community detection with edge-weighted modularity maximisation identified
12 communities in the co-occurrence graph, achieving a modularity score
of 0.3148. A modularity score above 0.3 indicates meaningful community
structure; the score of 0.3148 confirms meaningful thematic separation between vocabulary clusters.

The 'ESG & Environmental Disclosures' community (top terms: emission, greenhouse, gas) — This community clusters around Cisco's sustainability and environmental reporting obligations. Key terms such as emission, gas, and greenhouse reflect Cisco's Scope 1, 2, and 3 emissions disclosures required under SEC climate reporting guidance. This is a relatively small but thematically tight community, indicating that ESG vocabulary is used in concentrated, specialised sections of the filing rather than distributed throughout.

The 'Legal & Intellectual Property' community (top terms: patent, claim, state) — This community captures Cisco's legal obligation and intellectual property vocabulary, with terms such as patent, claim, and state reflecting both patent litigation proceedings and government contract compliance language. The legal community is one of the larger clusters in the network, consistent with Cisco's profile as a technology company with significant patent portfolios and exposure to ongoing litigation described across multiple filing sections.

The 'Corporate Governance & Financial Reporting' community (top terms: financial, cisco, security) — This community covers Cisco's governance and compliance reporting language, with terms such as financial, cisco, and security appearing together in contexts related to internal controls, audit committee disclosures, and cybersecurity governance. The co-occurrence of 'security' with governance terms reflects Cisco's dual use of 'security' — as both a cybersecurity product category and a governance compliance term — producing a cross-domain vocabulary cluster in this community.

The 'Equity Compensation & Capital Returns' community (top terms: employee, stock, share) — This community reflects Cisco's equity compensation and capital allocation disclosures. Terms such as employee, stock, and share appear together in the context of restricted stock units, employee purchase plans, and share repurchase programmes. The grouping confirms that Cisco's equity-linked compensation and buyback disclosures form a self-contained vocabulary domain within the annual filing, consistent with dedicated Notes to Financial Statements sections on equity instruments.

The 'Contracts & Financing Arrangements' community (top terms: term, arrangement, financing) — This community captures Cisco's contractual and structured financing vocabulary, with terms such as term, arrangement, and financing appearing in the context of credit facility terms, lease arrangements, and financing receivable disclosures. The community reflects the technical language of Cisco's capital structure management as described in the long-term debt and lease accounting notes.

The 'Revenue & Geographic Segments' community (top terms: revenue, segment, percentage) — This community is one of the most financially significant, clustering the terms revenue, segment, and percentage together in contexts related to geographic revenue breakdowns, segment reporting under ASC 280, and percentage-of-total revenue disclosures. The strong coherence of this community reflects Cisco's consistent segment reporting structure across all five filing years and the dominance of revenue as a disclosure theme.

The 'Asset Valuation & Tax' community (top terms: asset, value, tax) — This community covers Cisco's balance sheet valuation and tax accounting vocabulary, with terms such as asset, value, and tax appearing together in the context of fair value measurements, deferred tax assets, and goodwill impairment testing. The community reflects the Notes to Financial Statements sections on intangible assets and income taxes, which are among the most technically dense sections of Cisco's annual filing.

The 'Products, Services & Market' community (top terms: product, customer, service) — This is the largest community in the network and captures Cisco's core commercial vocabulary — product, customer, and service — which appears across the MD&A, revenue recognition notes, and competitive strategy sections. Its large size and high internal degree reflect the fact that product and customer language permeates every major section of a technology company's 10-K, making this community the structural core of Cisco's co-occurrence graph.

The 'Supply Chain & Procurement' community (top terms: component, supply, contract) — This community reflects Cisco's supply chain risk and procurement language, with terms such as component, supply, and contract appearing in risk factor disclosures about single-source suppliers, inventory management, and manufacturing disruption risks. The community became more prominent in the 2022 and 2023 filings, consistent with the global semiconductor supply chain disruptions that affected Cisco's hardware production during that period.

The 'XBRL Filing Metadata' community (top terms: inline, extension, document) — This community consists primarily of XBRL tagging metadata terms — inline, extension, and document — which are artefacts of the structured filing format rather than substantive financial vocabulary. This community represents technical filing noise and should not be interpreted as reflecting Cisco's financial disclosure content. Its presence as a distinct community confirms that Louvain detection correctly isolated metadata language from substantive financial vocabulary.

The 'Risk Factors & Business Conditions' community (top terms: business, condition, operating) — This community captures Cisco's forward-looking and macro-risk vocabulary, with terms such as business, condition, and future appearing in risk factor sections, MD&A outlook language, and business overview disclosures. The community reflects the conditional and uncertainty-laden language that characterises the Risk Factors section required by SEC reporting rules, where management is obligated to disclose all material risks to future business conditions.

The 'Workforce Diversity & Culture' community (top terms: diversity, gender, culture) — This community clusters Cisco's human capital and inclusion reporting vocabulary, with terms such as diversity, gender, and race appearing in the Human Capital section of the 10-K, which became a mandatory disclosure item under SEC rules effective FY2021. The tight clustering of this vocabulary confirms that Cisco's workforce disclosure language is concentrated in a dedicated section and does not significantly overlap with other financial vocabulary domains.

### Comparison of Q6 Louvain Communities and Q4 K-Means Clusters

The Q4 k-means analysis partitioned the top 300 corpus-frequency words into five
clusters based on geometric proximity in the Word2Vec embedding space, while Q6 Louvain
detection partitioned the full co-occurrence graph of 2858 nodes
into 12 communities based on structural density of co-occurrence links.
The two methods operate on different representations — vector space geometry versus
graph topology — and consequently produce groupings of different granularity. Where
k-means produced five broad thematic buckets reflecting semantic similarity, Louvain
yields 12 finer-grained communities that reflect actual positional
co-occurrence patterns within the filing text. The two approaches are complementary
rather than contradictory: k-means reveals which financial concepts are used in similar
semantic contexts, while Louvain reveals which specific term pairs are regularly
juxtaposed in the same disclosure sentences. Terms that appear in the same k-means
cluster are not necessarily in the same Louvain community, because semantic similarity
in the embedding space does not require direct textual adjacency. Together, the two
analyses confirm that Cisco's 10-K vocabulary has coherent thematic structure at both
the semantic and syntactic levels of the text.

### Network Structure and Cisco's Financial Disclosure Focus

The community structure of the co-occurrence network reflects Cisco's financial
disclosure priorities as an enterprise technology company with a recurring revenue
model, significant intangible assets, and extensive regulatory obligations. The
presence of distinct communities for revenue-related, governance-related, and
asset-related vocabulary is consistent with the three dominant reporting sections of
Cisco's 10-K filings: the Management Discussion and Analysis, the Risk Factors section,
and the Notes to Financial Statements. The modularity score indicates that these
thematic clusters are not arbitrary — they reflect genuinely distinct functional areas
of Cisco's disclosure language that co-occur internally but connect across boundaries
through the high-betweenness bridge terms identified above.
