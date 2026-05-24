# Question 1: Corpus Construction and Text Preprocessing

## Vocabulary Summary

[TABLE](output/tables/q1_vocab_summary.csv)

## Top 30 Most Frequent Terms

[TABLE](output/tables/q1_top30_terms.csv)

## DTM Dimensions

The Document-Term Matrix contains 5 documents and 3,682 unique terms, where each row corresponds to one fiscal year and each column to one lemmatised token present in the cleaned corpus.

## Word Cloud

![Figure 1: Word Cloud of Combined 10-K Corpus (post-preprocessing)](output/plots/q1_wordcloud.png)

## Interpretation

The top 30 most frequent terms in Cisco Systems' five-year 10-K corpus are dominated by core operational and financial vocabulary, which is consistent with the nature of SEC annual filings and the company's primary business activities. Terms such as product, service, customer, revenue, market, and business rank among the highest frequency words, confirming that Cisco's management communication across the five filings is centred on its networking and enterprise technology operations. This result aligns with expectations for a technology company whose revenue is driven by hardware products and subscription-based services.

Financial reporting vocabulary is well represented in the high-frequency terms. Words such as asset, income, cash, investment, net, loss, and tax reflect the accounting structure of the 10-K, which requires extensive disclosure of balance sheet items, income statement results, and tax obligations. The presence of these terms at high frequency indicates that financial condition reporting occupies a large proportion of the filing text, as required by SEC regulations.

Cisco-specific terms, particularly security and technology, appear prominently, reflecting the company's strategic focus on cybersecurity products and enterprise infrastructure. The term contract is also notable, pointing to Cisco's significant recurring revenue from multi-year software and service agreements. The appearance of risk and loss in the top 30 is consistent with mandatory risk factor disclosure sections present in every 10-K filing.

Overall, the top 30 terms are dominated by meaningful financial and business vocabulary. The expanded custom stopword list and lemmatisation approach have successfully removed generic administrative noise, leaving a corpus that is appropriate for the downstream Word2Vec and co-occurrence network analyses.
