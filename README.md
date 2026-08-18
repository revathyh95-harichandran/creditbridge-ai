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

## Tests

```bash
pytest -q
```

The segmentation tests run against a synthetic in-memory DuckDB, so they need no downloaded data.
