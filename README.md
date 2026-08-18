# creditbridge-ai

Credit risk analytics on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset, with a focus on **thin-file applicants** — borrowers who are hard to score because
they carry almost no credit-bureau history and incomplete external credit scores.

Non-technical? Start with [reports/analyst_summary.md](reports/analyst_summary.md) — a
one-page executive summary, no code or jargon.

## The thin-file question

Traditional scorecards lean on bureau depth and external scores. Applicants without them are
often declined by default, not because they are risky but because they are *unmeasured*.
This project starts by isolating that population and asking whether its default behaviour
actually differs from the well-documented ("thick-file") majority.

**Segment definition** (`src/segment.py`):

| Segment | Rule |
|---|---|
| `thin-file` | 0 or 1 records in `bureau.csv` **AND** at least one of `EXT_SOURCE_1/2/3` missing |
| `thick-file` | everything else |

## Project layout

```
creditbridge-ai/
├── src/
│   ├── config.py             # canonical paths
│   ├── load_to_duckdb.py     # CSV -> DuckDB
│   ├── clean.py              # data-quality-driven cleaning -> application_train_clean
│   ├── segment.py            # thin/thick-file flagging + segment stats
│   ├── compare_segments.py   # thin- vs thick-file comparison on cleaned data
│   └── export_dashboard.py   # cleaned + segmented data -> dashboard_export.csv
├── data/
│   ├── raw/                  # Kaggle CSVs (git-ignored)
│   └── processed/            # derived artefacts (git-ignored)
├── dashboards/
│   └── credit_risk_dashboard.pbix  # Power BI dashboard on dashboard_export.csv (git-ignored)
├── notebooks/
│   └── 01_data_quality.ipynb # full DQ pass: % missing, skew, IQR outliers, cardinality
├── tests/                    # pytest suite
├── reports/
│   ├── segment_findings.md   # cleaning decisions + segment comparison write-up
│   └── analyst_summary.md    # one-page, non-technical executive summary
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 1. Get the data

Download `application_train.csv`, `bureau.csv`, and `previous_application.csv` from
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
(requires a Kaggle account and accepting the competition rules) and place them in `data/raw/`.

**Do not commit these files.** Kaggle's competition rules permit academic / educational /
non-commercial use of the data but prohibit redistributing it — `.gitignore` excludes
`data/raw/` and every `.csv` under `data/` for exactly this reason. Never push them to a
public repo.

### 2. Build the database

```bash
python src/load_to_duckdb.py
```

Creates `data/processed/creditbridge.duckdb` with tables `application_train`, `bureau`, and
`previous_application`.

### 3. Run the data-quality pass (optional but recommended first)

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_quality.ipynb
```

Audits every feature in `application_train` — % missing, skewness, and IQR-flagged
outliers for numeric columns; cardinality and % missing for categorical ones — before any
cleaning decision is made. The findings from this pass drive every decision in step 4.

### 4. Clean the data

```bash
python src/clean.py
```

Creates `application_train_clean`: median imputation for numeric columns, an explicit
`'Missing'` category for categorical ones, IQR-detected outliers capped at the 1st/99th
percentile, `EXT_SOURCE_1/2/3` missingness preserved (not imputed) as a signal, and the
`DAYS_EMPLOYED` sentinel value resolved. Every decision, including the alternatives
considered and rejected, is written up in [reports/segment_findings.md](reports/segment_findings.md).

### 5. Segment applicants

```bash
python src/segment.py
```

Writes the `applicant_segments` table and prints the size and default rate of each segment.

### 6. Compare segments on the cleaned data

```bash
python src/compare_segments.py
```

Compares thin-file vs thick-file on income, employment length, age, application channel,
and which non-bureau features remain fully populated for thin-file applicants. Full
write-up in [reports/segment_findings.md](reports/segment_findings.md).

### 7. Export the dashboard-ready dataset

```bash
python src/export_dashboard.py
```

Writes `data/processed/dashboard_export.csv` (307,511 rows) with business-readable column
names — `file_type`, `defaulted`, `income`, `employment_years`, `age`, `family_status`,
plus five non-bureau features (`owns_car`, `owns_realty`, `income_type`,
`household_size`, `prior_applicant`) chosen from the fully-populated set identified in
[reports/segment_findings.md](reports/segment_findings.md) §4.

### 8. Power BI dashboard

Built on `data/processed/dashboard_export.csv`, saved as `dashboards/credit_risk_dashboard.pbix`.
Not tracked in git (see `.gitignore` — it embeds a full copy of the exported data in its
internal model, same reasoning as the CSV/DuckDB files). When editing the report, use
**File → Import locale = English (United States)** in Power BI Desktop's app-level
Options and Settings — the source CSV uses `.` as the decimal separator, and a mismatched
locale silently corrupts decimal columns (e.g. `employment_years`) on import.

## Dashboard Insights — What I Found

Notes on what the dashboard (`dashboards/credit_risk_dashboard.pbix`) actually shows,
written in plain terms so it still makes sense on a re-read months from now.

**What the numbers say:**

- Out of 308K applicants, 68K (22%) are "thin-file" — basically, people we don't have
  much credit history on. That's not a small group, it's more than 1 in 5 applicants.
- These thin-file people do default more often — 9.7% of the time vs 7.6% for everyone
  else. That's a real difference, but it's not huge. It's not like thin-file people are
  twice as risky or something — they're a bit riskier, not a lot.
- When I dug into the dashboard a bit more, the usual stuff still applies here too —
  people who've had their job longer, and people who are a bit older (45+), default
  less. Same pattern you'd expect for anyone, thin-file or not. So it's not like these
  people are some weird unpredictable group — the normal signals still work on them.
- Here's the interesting part: even though we don't have much bureau/credit history on
  these people, we DO have basically everything else — their income, what kind of job
  they have, family details, whether they own a car or house, and whether they've
  borrowed from us before (93.5% of them have!). So it's not that we know nothing about
  them, we just don't have the one specific thing (credit history).

**What I'd actually suggest doing:**

1. Don't auto-reject people just because they're thin-file. We have enough other info
   on them to make a real decision instead of guessing.
2. If we want to test this out, start with the safer-looking thin-file people first —
   longer job history, 45+, already applied with us before. These already look closer
   to the "safe" group in the numbers.
3. Whatever we try, actually check afterward if it worked — compare the real default
   rate of whoever we approve against that 9.7% number, to make sure we're not just
   letting in more risk than we think.

## Findings

Run on the full `application_train` population (307,511 applicants), joined against `bureau`:

| segment | applicants | share | defaults | default rate |
|---|---|---|---|---|
| thin-file | 68,367 | 22.2% | 6,654 | 9.73% |
| thick-file | 239,144 | 77.8% | 18,171 | 7.60% |
| overall | 307,511 | 100.0% | 24,825 | 8.07% |

Thin-file applicants — those with 0-1 bureau records and at least one missing `EXT_SOURCE`
score — default at **1.28x** the thick-file rate (9.73% vs 7.60%), and make up 22.2% of the
book. This is the population a conventional scorecard is effectively blind to: it isn't just
under-documented, it's measurably riskier. Reproduce with `python src/segment.py` after
building the database (step 2 below).

They also earn less (median $135,000 vs $157,500) and are slightly more often retired
(20.2% vs 17.4%) — but are essentially the **same age** as thick-file applicants, and are
**fully documented on every non-bureau dimension checked**: asset ownership, family status,
contact info, employment tenure, and 93.5% have prior application history with this same
lender. The gap is narrow and specific — bureau depth and external credit scores — not a
general data problem. Full analysis, including every cleaning decision and the
alternatives ruled out along the way, in [reports/segment_findings.md](reports/segment_findings.md).

## Tests

```bash
pytest -q
```

24 tests against synthetic in-memory DuckDB tables — segmentation boundary conditions and
cleaning decisions (median imputation, `EXT_SOURCE` missingness preservation, the
`DAYS_EMPLOYED` sentinel fix, degenerate-IQR handling, categorical fill). No downloaded
data required.
