# Question 3: Semantic Proximity Analysis

## Selected Seed Terms

The following seed terms were selected by evaluating which candidate word in the model
vocabulary produced the most financially meaningful nearest neighbours. The first
candidate yielding coherent financial neighbours was used; 'profit' and 'risk' were
bypassed because their neighbourhoods consisted of semantically unrelated or boilerplate
terms.

Liquidity: cash | Profitability: earnings | Debt: debt | Revenue: revenue | Risk: uncertain

## Nearest Neighbours (Top 10 per Seed Term)

[TABLE](output/tables/q3_semantic_neighbours.csv)

## Interpretation

**Liquidity (seed: cash)**
The semantic neighbourhood of 'cash' is partially meaningful. The two
highest-ranked neighbours — flow and liquidity — are directly relevant to short-term
liquidity management and confirm that this seed occupies a cash-focused region of the
vector space. Further neighbours including indebtedness and withstand carry financial
relevance, appearing in contexts where Cisco discusses its capacity to meet obligations
and sustain operations under adverse conditions. However, several other neighbours such
as consider, exiting, realignment, and recruit reflect co-occurrence with restructuring
and organisational disclosure language rather than liquidity-specific vocabulary,
introducing some noise into the neighbourhood profile.

**Profitability (seed: earnings)**
The seed 'earnings' was selected over 'profit' because 'profit' returned
semantically unrelated neighbours with cosine similarities below 0.35, indicating that
'profit' is too infrequent in Cisco's formal SEC filing language to have developed
meaningful distributional associations in this corpus. The neighbourhood of
'earnings' includes fair, interest, impairment, taxation, deductibility,
and rate — terms directly relevant to earnings computation, fair value adjustments, and
tax obligations as reported in Cisco's income statements and notes. The presence of
impairment alongside earnings reflects Cisco's recurring goodwill and intangible asset
reviews. Some noise is present in neighbours such as earlier, targeted, and selection,
which reflect co-occurrence in forward-looking or comparative sentences rather than
purely profitability contexts.

**Debt (seed: debt)**
The semantic neighbourhood of 'debt' is the most coherent and financially
grounded across all five dimensions. Neighbours including incurrence, leverage,
indebtedness, issuance, pay, incur, and assume are all directly tied to debt
management, borrowing obligations, and long-term financing disclosures. This compact and
precise cluster reflects Cisco's structured debt instrument disclosures, where covenant
terms, maturity schedules, and interest obligations are consistently discussed together
across the five annual filings. The absence of noise terms confirms that 'debt' is used
in a well-defined and consistent vocabulary context throughout the corpus.

**Revenue (seed: revenue)**
The neighbourhood of 'revenue' is one of the strongest in the analysis, with
neighbours deferral, recognition, variability, predictability, shortfall, sustainable,
and volume all directly mapping to Cisco's revenue reporting framework. These terms
reflect the ASC 606 revenue recognition standard applied in the filings, the deferred
revenue balance from multi-year software and subscription contracts, and management's
forward-looking commentary on revenue sustainability and growth. The coherence of this
neighbourhood confirms that revenue is a well-defined and consistently discussed concept
across all five Cisco 10-K filings.

**Risk (seed: uncertain)**
The seed 'uncertain' was selected over 'risk' because 'risk' returned neighbours
consisting almost entirely of cross-reference boilerplate — terms such as fully,
discussed, forth, contained, and elsewhere — which are phrases used in the structured
risk factor section to direct readers to other parts of the document rather than
conveying substantive financial risk content. In contrast, the neighbours of 'uncertain'
include geopolitical, unfavorable, deteriorate, challenging, and instability, which are
genuine risk disclosure terms reflecting macroeconomic conditions, market headwinds, and
operational vulnerabilities discussed in Cisco's filings. Some noise is present in
neighbours such as pronounced, germane, and entrance, but the overall neighbourhood is
substantively more informative than any alternative seed candidate tested.
