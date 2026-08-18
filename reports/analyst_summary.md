# Thin-File Applicants: Executive Summary

## The question

Are we declining creditworthy applicants simply because they lack a long credit history,
rather than because they're actually higher-risk?

## Three headline numbers

**1. The split.** Of every 5 applicants, roughly 1 in 5 (22%) has little to no credit
bureau history and an incomplete credit score — we call these "thin-file." The remaining
78% are well-documented ("thick-file").

**2. The default gap.** Thin-file applicants do default more often — 9.7% of the time,
versus 7.6% for thick-file applicants. That's about 1.3x the risk. Real, but modest: it's
not the night-and-day difference that a blanket "no history, no loan" policy would imply.

**3. The data availability gap.** Here's the surprising part: outside of credit bureau
data, thin-file applicants are **just as fully documented** as everyone else. We have
complete data on their income, employment, family situation, assets owned (car, property),
and contact details — matched almost exactly to the thick-file population. 93.5% have even
applied with us before. The only real information gap is credit bureau history itself —
not a general lack of information about who these applicants are.

## What's worth investigating further

The 1.3x risk difference is real, but it's a modest gap to be making an all-or-nothing
lending decision on. Given how complete the *other* data is for this group, it's worth
testing whether a scorecard built on income, employment stability, assets, and prior
relationship with us can price thin-file risk more precisely than a blanket decline —
essentially asking whether we can replace "no history" with "here's what we do know."
That analysis is the natural next step, using the dashboard-ready dataset now available
at `data/processed/dashboard_export.csv`.

## Recommendation

Before declining thin-file applicants by default, pilot a scorecard that uses the
income, employment, and asset data we already collect on them — the modest risk gap
suggests real revenue is being left on the table.

---
*Full methodology, data quality checks, and every analytical decision behind these
numbers: [reports/segment_findings.md](segment_findings.md).*
