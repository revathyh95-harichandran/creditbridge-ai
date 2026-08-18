# Thin-file vs thick-file: data quality, cleaning decisions, and segment comparison

This report documents the full pass from raw data to a cleaned model table, and what that
cleaned data shows when thin-file and thick-file applicants are compared. It's written to
stand alone — every number here is reproducible from the commands listed at the bottom,
and every cleaning decision includes the alternatives that were considered and rejected,
not just the one that was implemented.

**Scope.** All of this operates on `application_train` (307,511 applicants) — the table
carrying `TARGET`, and the one that becomes the model matrix. `bureau` is used only to
count each applicant's credit-bureau records (needed for the thin-file definition itself).
`previous_application` is used only where explicitly noted (application channel, previous-
application coverage). Neither `bureau` nor `previous_application` is cleaned column-by-
column here — they're relational history tables that get aggregated into
`application_train`-level features at a later feature-engineering stage, not the model
table itself.

**Segment definition** (unchanged from the original segmentation, see [src/segment.py](../src/segment.py)):
an applicant is **thin-file** if they have 0–1 records in `bureau` *and* at least one of
`EXT_SOURCE_1/2/3` is missing; everyone else is **thick-file**.

---

## 1. Data quality pass

Full audit in [notebooks/01_data_quality.ipynb](../notebooks/01_data_quality.ipynb), run
before any cleaning decision was made. `application_train`'s 120 feature columns split
into 72 continuous numeric, 32 binary 0/1 flags, and 16 string/boolean categorical (flags
are numeric columns with ≤2 distinct values — skewness and IQR fences are meaningless for
a two-valued variable, so they're audited as categorical instead).

**Numeric — headline findings:**

| Feature | % missing | Skewness | Note |
|---|---|---|---|
| `EXT_SOURCE_1` | 56.4% | -0.07 | External credit score. Missingness is informative (correlates with thin-file status) — see §2.2 |
| `EXT_SOURCE_3` | 19.8% | -0.41 | Same |
| `EXT_SOURCE_2` | 0.2% | -0.79 | Same |
| `COMMONAREA_*`, `NONLIVINGAPARTMENTS_*` (building descriptors) | 55–70% | 3–16 | Populated only when a property is registered as collateral — missingness reflects loan type, not data-capture failure |
| `AMT_INCOME_TOTAL` | 0.0% | **391.56** | Extreme right skew — a small number of very high earners |
| `AMT_REQ_CREDIT_BUREAU_QRT` | 13.5% | 134.37 | Zero-inflated count, see IQR=0 note below |
| `DAYS_EMPLOYED` | 0.0% nulls | 1.66 | **Not actually 0% complete** — see §2.5 |

**Categorical — headline findings:**

| Feature | Cardinality | % missing |
|---|---|---|
| `FONDKAPREMONT_MODE` | 4 | 68.4% |
| `WALLSMATERIAL_MODE` | 7 | 50.8% |
| `HOUSETYPE_MODE` | 3 | 50.2% |
| `EMERGENCYSTATE_MODE` | 2 | 47.4% |
| `OCCUPATION_TYPE` | 18 | 31.4% |
| all other categoricals (gender, education, family status, contract type, ...) | 2–58 | 0.0–0.4% |

**Two structural findings shaped the cleaning approach directly:**

- **The IQR method breaks down on 9 columns** (`REGION_RATING_CLIENT[_W_CITY]`,
  `DEF_30/60_CNT_SOCIAL_CIRCLE`, `AMT_REQ_CREDIT_BUREAU_HOUR/DAY/WEEK/MON/QRT`) because
  their 25th and 75th percentiles are identical (`IQR = 0`) — 70–91% of applicants share
  the exact same value (a 3-level ordinal rating, or a zero-inflated count). Under the
  standard `1.5×IQR` rule, *any* different value gets flagged as an outlier, including the
  bulk of a legitimate distribution. `REGION_RATING_CLIENT` alone would "flag" 26% of the
  book as outliers this way — that's not an anomaly, it's the method failing on data it
  wasn't designed for.
- **`DAYS_EMPLOYED` has 0% nulls but is not 0% complete.** It uses `365243` (~1,000 years)
  as a sentinel for "not currently employed" (mostly pensioners) — 55,374 rows (18.0%).
  This matters more than a normal outlier because it's common enough to define the
  column's own 99th percentile (see §2.5).

---

## 2. Cleaning decisions

Implemented in [src/clean.py](../src/clean.py), which reads `application_train`, writes
a new `application_train_clean` table, and leaves the source table untouched. Every
decision below was verified against the real 307,511-row table (`python src/clean.py`
prints the exact per-column outcome), not designed in the abstract.

### 2.1 Continuous numeric missing values → median imputation

**Decision:** for every continuous numeric column except `EXT_SOURCE_1/2/3`, missing
values are filled with that column's median.

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| **Mean** | Nearly every continuous column here is skewed (`AMT_INCOME_TOTAL` skew=391.6, `AMT_ANNUITY`=1.58, `AMT_CREDIT`=1.23 — see §1). The mean of a skewed distribution is dragged toward the long tail and isn't representative of a typical applicant; the median is robust to exactly that. |
| **Mode** | Sensible for categorical data, not continuous — most continuous columns here (income, area measurements) have near-unique values, so the mode is often just one arbitrarily common value, not a central tendency. |
| **Drop rows with missing values** | Housing descriptors alone are missing on 48–70% of applicants. Dropping those rows would discard roughly half the dataset and destroy the ability to compare segments at all — the missingness is structural (no collateral property registered), not random, so dropping would also bias the remaining sample toward property owners. |
| **Model-based imputation (KNN, MICE, regression)** | Meaningfully more accurate for MAR (missing-at-random) data, but overkill and harder to audit for a first cleaning pass over 58 columns with 0–70% missingness, several of which (the housing descriptors) aren't missing at random in the first place — a model would be learning to predict values for properties that structurally don't have those values. Median imputation is transparent, reproducible in one line of SQL per column, and sets a defensible baseline; a smarter imputer is a reasonable next iteration once a model is actually being trained. |

### 2.2 EXT_SOURCE_1/2/3 → never imputed, missingness flagged instead

**Decision:** `EXT_SOURCE_1/2/3` are left exactly as they are in the raw data, NULLs
included. A companion `EXT_SOURCE_n_IS_MISSING` column (1/0) is added for each.

**Why these three are different from every other numeric column:** EXT_SOURCE scores are
external, bureau-style credit scores and the strongest predictors in this dataset. Their
absence isn't noise — it's a signal in its own right (thin-file applicants are
*defined* partly by missing EXT_SOURCE scores; see the segment definition above). Filling
that gap with a median would erase the exact information the rest of this report is
built on.

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| **Median-impute like other numeric columns** | Would silently convert "we don't have a score for this applicant" into "this applicant scored exactly average," destroying a highly predictive signal and making thin-file applicants statistically indistinguishable from average on this feature. |
| **Mean-impute** | Same problem as median, plus the skew issues from §2.1. |
| **Drop rows with any EXT_SOURCE missing** | Would remove the majority of `EXT_SOURCE_1` (56.4% missing) and, more importantly, would delete almost every thin-file applicant by construction — the population this whole analysis exists to study. |
| **Drop the EXT_SOURCE columns entirely** | Throws away the strongest signal in the dataset for the 78% of applicants (thick-file) who *do* have it, to avoid dealing with missingness in the other 22%. |
| **Impute + flag (fill with median, but also add the missingness flag)** | Closer, and defensible — a linear model needs non-null inputs, so mean/median-impute-plus-flag is a common production pattern. Rejected here specifically because tree-based models (the natural choice for tabular credit data, and the direction this project is headed given `applicant_segments` is already a `bureau`+`application_train` join) handle NULL natively and split on it directly; imputing a value that will just get overridden by the flag anyway adds a step without benefit at the cleaning stage. If a linear/logistic baseline is added later, impute-plus-flag on top of the preserved-NULL version is a one-line change. |

### 2.3 Categorical missing values → explicit "Missing" category

**Decision:** every categorical column's NULLs are filled with the literal string
`'Missing'`, not a computed value.

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| **Mode / most-frequent value** | `OCCUPATION_TYPE` is missing on 31.4% of applicants and `WALLSMATERIAL_MODE` on 50.8% — filling that much of a column with whichever category happens to be most common overstates that category's true prevalence by a wide margin and can materially shift any downstream rate computed by category. |
| **Drop rows** | Same issue as §2.1 — `FONDKAPREMONT_MODE` alone is missing on 68.4% of applicants; dropping those rows would gut the dataset. |
| **Drop the column** | Several of these columns (`OCCUPATION_TYPE` especially) are exactly the kind of feature checked for thin-file usability in §4 — dropping them pre-empts that analysis. |

An explicit `'Missing'` category costs nothing (categorical models split on it like any
other level) and — unlike mode-imputation — doesn't pretend to know something the data
doesn't say. One type-level exception: `EMERGENCYSTATE_MODE` is stored as `BOOLEAN`, so
it's cast to `VARCHAR` before the `'Missing'` fill can be applied (a boolean column can't
hold a three-way True/False/Missing value without becoming a string first).

**Binary 0/1 flag columns** (`FLAG_EMAIL`, `FLAG_DOCUMENT_3`, `REG_CITY_NOT_LIVE_CITY`,
etc.) are a related but distinct case: they were verified to have 0% missing in the source
data (checked directly, not assumed — see §1's categorical table), so no imputation
decision was needed for them at all. They pass through unchanged. `src/clean.py` asserts
this at build time and raises rather than silently guessing if that assumption is ever
violated by a data refresh.

### 2.4 Outliers → IQR detection, 1st/99th-percentile capping (not row dropping)

**Decision:** for each continuous numeric column, compute `Q1`, `Q3`, and the IQR fences
`[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`. If at least one value falls outside those fences (and the
column isn't degenerate — see §2.6), every value below the column's 1st percentile is
raised to the 1st percentile, and every value above the 99th is lowered to the 99th.
Rows are never dropped for this reason. 58 of 72 continuous columns qualified for capping;
11 had zero IQR-flagged outliers or a degenerate IQR (§2.6) and were left as median-impute-only.

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| **Z-score method** (flag `\|z\| > 3`) | Assumes an approximately normal distribution. Every monetary column here is heavily right-skewed (§1) — z-scores on a skewed distribution systematically under-flag the long tail and over-flag the short one, which is the opposite of what's needed for income/credit-amount data. |
| **Drop outlier rows** | The outliers here are real applicants (a high earner, a large loan), not data-entry errors — dropping them would bias the sample toward smaller, more typical loans and quietly change what population the model is trained on. Capping keeps every applicant in the dataset while limiting how much leverage an extreme value can exert. |
| **Cap at the IQR fences themselves** (`Q1-1.5×IQR` / `Q3+1.5×IQR`) instead of the 1st/99th percentile | The IQR fences are a *detection* threshold, not a sensible *value* to cap to — for a heavily skewed column the upper IQR fence can sit well inside the legitimate range (e.g. `AMT_INCOME_TOTAL`'s fence is 337,500 but plenty of genuine incomes sit above that), which would over-aggressively compress the tail. The 1st/99th percentile is a fixed, interpretable trim independent of the shape of the distribution, and only touches the most extreme 2% of values. |
| **No outlier handling** | `AMT_INCOME_TOTAL` has a skewness of 391.6 — a handful of extreme incomes would dominate any distance-based or gradient-based model's loss function untreated. |

### 2.5 `DAYS_EMPLOYED` sentinel (365243) — resolved before anything else runs

This is the one column-specific exception to the two general numeric rules above, and it
exists because the general rule was tried first and demonstrably failed.

**What happened:** the first version of this cleaning pass assumed §2.4's generic
percentile-capping would neutralize the `365243` sentinel as a side effect — real tenures
top out around 17,900 days, so 1,000 years looks like an obvious outlier. Running the
actual capping against the real data disproved that: **the sentinel is common enough
(18.0% of applicants) that it single-handedly sets the column's 99th percentile.** Capping
"outliers" to `p99` caps them to `365243` itself — a no-op. All 55,374 sentinel rows
survived generic capping completely untouched. This is documented with the real numbers
in [notebooks/01_data_quality.ipynb](../notebooks/01_data_quality.ipynb) (§5 finding 5),
including the corrected write-up after the bug was found.

**Decision:** before any statistics are computed, `DAYS_EMPLOYED = 365243` is converted to
`NULL` and captured separately in a `DAYS_EMPLOYED_IS_RETIRED` flag (1/0) — the same
"don't silently erase an informative signal" logic as §2.2's EXT_SOURCE handling, since
being retired/unemployed is exactly the kind of fact a median-fill would otherwise hide.
The resulting NULLs then go through the ordinary §2.1 median-impute + §2.4 IQR-capping
path, computed only from genuine tenure values (median employment length ≈ 4.5 years in
both segments post-fix; see §3).

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| **Trust generic percentile capping to handle it** | Tried first; proven not to work (above). Documented as a correction, not silently patched, because the failure mode itself is informative — it shows generic rules can't be trusted blindly on sentinel-coded data. |
| **Drop `DAYS_EMPLOYED` entirely** | Employment tenure is one of the "employment stability" features checked for thin-file completeness in §4 — dropping it removes exactly the signal that section needs. |
| **Median-impute the sentinel like EXT_SOURCE (flag but don't fill)** | Considered, but rejected for consistency with §2.2's reasoning: unlike EXT_SOURCE (an external score genuinely absent from a data source), the *true* employment tenure for a retiree is a coherent concept ("effectively very long" or "not applicable") rather than an unknown value, so treating it as ordinary missingness for a numeric column and letting the standard median-fill handle it — while preserving the fact separately via the flag — fits the existing pattern better than inventing a third treatment. |

### 2.6 Degenerate IQR (`IQR = 0`) — capping skipped, not forced

**Decision:** the 9 columns identified in §1 (`REGION_RATING_CLIENT[_W_CITY]`,
`DEF_30/60_CNT_SOCIAL_CIRCLE`, `AMT_REQ_CREDIT_BUREAU_HOUR/DAY/WEEK/MON/QRT`) are
median-imputed like any other numeric column, but **not** capped.

**Why:** when `Q1 == Q3`, the `1.5×IQR` fence collapses to a single point, so the
"outlier" flag stops measuring distance from a normal range and starts just measuring
"not equal to the single most common value." For `REGION_RATING_CLIENT` that would flag
26.2% of applicants — a three-level ordinal rating, not a continuous quantity, where the
IQR test was never a sensible tool. Capping to the 1st/99th percentile on top of that
would in some cases be harmless (e.g. `REGION_RATING_CLIENT`'s 1st/99th percentiles are
1 and 3 — the entire observed range, so capping is a no-op) but in others would silently
compress real signal (e.g. `AMT_REQ_CREDIT_BUREAU_QRT`'s 99th percentile is 2, so
applicants with 3+ recent bureau inquiries — a meaningfully different credit-shopping
pattern — would get flattened to 2).

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| **Cap anyway, ignoring the degenerate fence** | Would either be a no-op (wasted complexity) or would compress genuine tail signal in the zero-inflated count columns, as above — capping should only fire when it's actually correcting something. |
| **Switch to a pure percentile-based outlier test** (flag anything outside p1/p99, regardless of IQR) for these 9 columns specifically | More consistent in spirit, but changes the outlier-*detection* method per column, whereas the definition used throughout this report is specifically the IQR method the task calls for. Skipping capping when IQR can't detect anything meaningful is a smaller, more legible exception than swapping detection methods column-by-column. |

### 2.7 Why a new table, not an in-place update

`application_train_clean` is a new table; `application_train` is never modified. This
keeps the raw data available for exactly the kind of audit this report performs (e.g.
§2.5's completeness check in §4 deliberately reads the *raw* table, since the cleaned
table has no missing values left by construction) and means every cleaning decision can be
re-derived and re-verified against the original source at any time.

---

## 3. Thin-file vs thick-file on cleaned data

All numbers from `applicant_segments` (68,367 thin-file / 239,144 thick-file, unchanged
from the original segmentation — see [reports/README.md findings](../README.md#findings))
joined to `application_train_clean`. Reproducible via `python src/compare_segments.py`.

| Metric | Thin-file | Thick-file |
|---|---|---|
| **Applicants** | 68,367 (22.2%) | 239,144 (77.8%) |
| **Default rate** | **9.73%** | 7.60% |
| **Income — mean** | $150,422 | $170,540 |
| **Income — median** | $135,000 | $157,500 |
| **Income — IQR (p25–p75)** | $90,000 – $180,000 | $112,500 – $202,500 |
| **Employment length — mean** | 5.3 years | 6.4 years |
| **Employment length — median** | 4.5 years | 4.5 years |
| **% retired/not employed** | 20.2% | 17.4% |
| **Age — mean** | 43.5 years | 44.0 years |
| **Age — median** | 43.2 years | 43.1 years |

**Reading these together:** thin-file applicants default 1.28× more often (§ [README](../README.md#findings))
while earning noticeably less (median $135,000 vs $157,500, a ~14% gap) and being modestly
more likely to be retired (20.2% vs 17.4%). Age is essentially identical between segments —
thin-file status here isn't simply "younger applicants without a credit history yet," which
is often the assumption. Median employment tenure is identical too (4.5 years both
segments) once the `DAYS_EMPLOYED` sentinel is properly resolved (§2.5) — the mean gap
(5.3 vs 6.4 years) is driven by the tail, not a systematically shorter typical tenure.

**Application channel** (from each applicant's most recent `previous_application`, among
those with prior history at this lender):

| Channel | Thin-file | Thick-file |
|---|---|---|
| Credit and cash offices | 39.7% | 41.6% |
| Country-wide | 28.6% | 28.3% |
| Stone | 14.0% | 12.3% |
| Regional / Local | 8.1% | 7.3% |
| Contact center | 5.5% | 6.6% |
| AP+ (Cash loan) | 3.7% | 3.4% |
| Channel of corporate sales | 0.4% | 0.4% |
| Car dealer | 0.0% | 0.0% |

Channel mix is broadly similar between segments — thin-file applicants skew a few points
toward "Stone" (physical branch) and away from "Credit and cash offices," but this isn't a
large or obviously actionable difference on its own.

---

## 4. Non-bureau features that are still fully populated for thin-file applicants

By definition, thin-file applicants are missing bureau depth and EXT_SOURCE scores. The
question this section answers: **is that the only thing missing, or are thin-file
applicants generally under-documented?** Checked against the *raw* `application_train` /
`previous_application` tables (not the cleaned table, which has no missing values left by
construction — see §2.7).

| Group | Feature | Thin-file % missing | Thick-file % missing |
|---|---|---|---|
| Asset ownership | `FLAG_OWN_CAR` | 0.00% | 0.00% |
| Asset ownership | `FLAG_OWN_REALTY` | 0.00% | 0.00% |
| Employment stability | `NAME_INCOME_TYPE` | 0.00% | 0.00% |
| Employment stability | `OCCUPATION_TYPE` | 33.09% | 30.85% |
| Employment stability | `ORGANIZATION_TYPE` | 0.00% | 0.00% |
| Employment stability | `DAYS_EMPLOYED` | 0.00% | 0.00% |
| Family status | `NAME_FAMILY_STATUS` | 0.00% | 0.00% |
| Family status | `CNT_FAM_MEMBERS` | 0.00% | 0.00% |
| Family status | `CNT_CHILDREN` | 0.00% | 0.00% |
| Contact completeness | `FLAG_MOBIL` / `FLAG_EMP_PHONE` / `FLAG_WORK_PHONE` / `FLAG_CONT_MOBILE` / `FLAG_PHONE` / `FLAG_EMAIL` | 0.00% (all six) | 0.00% (all six) |

**Previous internal applications** (does this applicant have any prior application to
*this* lender, in `previous_application` — a completeness question about row presence in a
related table, structurally the same kind of check as the bureau-record count used to
define thin-file itself):

| | Thin-file | Thick-file |
|---|---|---|
| Has ≥1 previous application | 93.5% | 95.0% |

**Asset ownership rates** (not just completeness — the actual values, since they're 100%
populated and worth reading):

| | Thin-file | Thick-file |
|---|---|---|
| Owns a car | 29.5% | 35.3% |
| Owns real estate | 70.9% | 68.9% |

**Finding:** every non-bureau feature checked is essentially fully populated for
thin-file applicants — identically to thick-file — with exactly one exception
(`OCCUPATION_TYPE`, ~2 points worse for thin-file, both segments already >30% missing).
Contact information, family composition, asset ownership, income type, organization type,
and employment tenure are all captured at intake regardless of bureau history, and 93.5%
of thin-file applicants even have prior application history with this same lender. The
thin-file gap is narrow and specific — bureau depth and EXT_SOURCE scores — not a general
data-collection problem. That's the practical takeaway: a thin-file applicant isn't a
blank record. There's a full, usable feature set available to score them on; the industry
default of declining for lack of bureau history is a policy choice, not a data constraint.

---

## Reproducing these results

```bash
python src/load_to_duckdb.py       # application_train, bureau, previous_application -> DuckDB
python src/clean.py                # application_train_clean (this report's §2)
python src/segment.py              # applicant_segments (thin-file / thick-file flags)
python src/compare_segments.py     # this report's §3 and §4
pytest -q                          # 24 tests, synthetic data, no download required
```

The data-quality pass itself is [notebooks/01_data_quality.ipynb](../notebooks/01_data_quality.ipynb) —
run it (`jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_quality.ipynb`)
against a populated database to regenerate §1 from scratch.
