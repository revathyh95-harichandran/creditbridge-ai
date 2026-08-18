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

### 1. Authenticate with Kaggle

The dataset comes from a Kaggle *competition*, so you must have a Kaggle account and accept
the competition rules once, in the browser, before the API will serve the files:
<https://www.kaggle.com/competitions/home-credit-default-risk/rules>

Then authenticate the CLI (this opens a browser OAuth flow):

```bash
kaggle auth login
```

### 2. Download the data

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw
python -c "import zipfile; zipfile.ZipFile('data/raw/home-credit-default-risk.zip').extractall('data/raw')"
```

Expected in `data/raw/`: `application_train.csv` (~166 MB), `bureau.csv` (~170 MB), plus the
other competition tables.

### 3. Build the database

```bash
python src/load_to_duckdb.py
```

Creates `data/processed/creditbridge.duckdb` with tables `application_train` and `bureau`.

### 4. Segment applicants

```bash
python src/segment.py
```

Writes the `applicant_segments` table and prints the size and default rate of each segment.

## Tests

```bash
pytest -q
```

The segmentation tests run against a synthetic in-memory DuckDB, so they need no downloaded data.
