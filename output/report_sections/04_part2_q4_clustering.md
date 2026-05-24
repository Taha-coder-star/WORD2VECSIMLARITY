# Question 4: Thematic Clustering of Word Vectors

## Clustering Summary (WCSS)
[TABLE](output/tables/q4_cluster_summary.csv)

## Representative Words per Cluster (Top 10)
[TABLE](output/tables/q4_representative_words.csv)

## PCA 2D Scatter Plot
![Figure 2: Word Vector Clusters - k=5 (PCA projection)](output/plots/q4_pca_clusters.png)

## Interpretation

K-means clustering with k=5 applied to the normalised word-vector matrix of the top 300
corpus-frequency words produced a Within-Cluster Sum of Squares (WCSS) of 174.1161.
WCSS measures the total intra-cluster variance; a lower value indicates that the cluster
members are more tightly concentrated around their respective centroids, while a higher
value suggests that the vocabulary grouped within each cluster is semantically more
dispersed.

The 'Derivatives & Equity Instruments' cluster groups words such as recorded, contractual, repurchase, reflecting semantic and contextual similarity in the Word2Vec vector space for derivatives & equity instruments vocabulary in Cisco's annual filings.

The 'Corporate Governance & Legal' cluster groups words such as transaction, director, officer, reflecting semantic and contextual similarity in the Word2Vec vector space for corporate governance & legal vocabulary in Cisco's annual filings.

The 'Asset Valuation & Accounting' cluster groups words such as deferred, valuation, asset, reflecting semantic and contextual similarity in the Word2Vec vector space for asset valuation & accounting vocabulary in Cisco's annual filings.

The 'Revenue & Lease Accounting' cluster groups words such as account, offset, decrease, reflecting semantic and contextual similarity in the Word2Vec vector space for revenue & lease accounting vocabulary in Cisco's annual filings.

The 'Strategic & Operational Context' cluster groups words such as summarized, compared, unit, reflecting semantic and contextual similarity in the Word2Vec vector space for strategic & operational context vocabulary in Cisco's annual filings.

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
