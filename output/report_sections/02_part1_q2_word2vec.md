# Question 2: Training the Word2Vec Model

## Model Architecture and Parameters

[TABLE](output/tables/q2_model_architecture.csv)

## Word-Vector Matrix

The extracted word-vector matrix has dimensions 2799 x 100.
Each row is a dense vector representation of one term in the cleaned corpus.

## Top 5 Words by L2 Norm (Vector Magnitude)

[TABLE](output/tables/q2_top5_l2norm.csv)

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
