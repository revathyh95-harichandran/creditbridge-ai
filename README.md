# creditbridge-ai

Credit risk analytics on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset, with a focus on **thin-file applicants** — borrowers who are hard to score because
they carry almost no credit-bureau history and incomplete external credit scores.

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
│   ├── config.py          # canonical paths
│   ├── load_to_duckdb.py  # CSV -> DuckDB
│   └── segment.py         # thin/thick-file flagging + segment stats
├── data/
│   ├── raw/               # Kaggle CSVs (git-ignored)
│   └── processed/         # derived artefacts (git-ignored)
├── notebooks/             # exploratory analysis
├── tests/                 # pytest suite
├── reports/               # generated output
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

### 3. Segment applicants

```bash
python src/segment.py
```

Writes the `applicant_segments` table and prints the size and default rate of each segment.

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

## Tests

```bash
pytest -q
```

The segmentation tests run against a synthetic in-memory DuckDB, so they need no downloaded data.
