# Question 3: Semantic Proximity Analysis

## Selected Seed Terms

The following seed terms were selected by matching candidate words against the model
vocabulary. The first candidate found in the vocabulary was used for each dimension.

Liquidity: cash | Profitability: profit | Debt: debt | Revenue: revenue | Risk: risk

## Nearest Neighbours (Top 10 per Seed Term)

[TABLE](output/tables/q3_semantic_neighbours.csv)

## Interpretation

**Liquidity (seed: cash)**
The semantic neighbourhood of 'cash' reflects Cisco's short-term liquidity
management, with terms related to cash equivalents, working capital, and short-term
investments appearing in close proximity. This pattern is consistent with a technology
firm that maintains large cash reserves and discloses them extensively within balance
sheet and cash flow statement discussions.

**Profitability (seed: profit)**
The neighbours of 'profit' are dominated by terms relating to net profit
computation, operating performance, and tax obligations. This co-occurrence reflects the
dense interaction between income reporting and expense disclosures within Cisco's income
statements, where operating income, tax provision, and net income are consistently
discussed together across the five annual filings.

**Debt (seed: debt)**
The semantic neighbourhood of 'debt' suggests that Cisco's management discusses
borrowing primarily in the context of long-term financing instruments, maturity schedules,
and interest obligations. The relatively compact cluster around debt is consistent with
Cisco's profile as a cash-rich technology firm where leverage plays a secondary role
compared to operational and revenue disclosures.

**Revenue (seed: revenue)**
Revenue's neighbours reveal the multidimensional structure of Cisco's top-line reporting,
with terms relating to product revenue, service revenue, and geographic segmentation
appearing in close proximity. This reflects Cisco's dual revenue structure from hardware
product sales and recurring software and subscription services, both of which are
discussed extensively in the Management Discussion and Analysis sections.

**Risk (seed: risk)**
The semantic neighbourhood of 'risk' contains both negative outcome terms and
management-oriented language. The presence of terms related to uncertainty and adverse
conditions alongside operationally focused words suggests that Cisco's risk disclosures
balance regulatory enumeration of potential negative outcomes with management language
that signals active risk governance. This mixed neighbourhood indicates that risk is
framed not only as a threat but also as something subject to mitigation and control,
which is consistent with the structured risk factor sections required in 10-K filings.
