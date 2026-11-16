"""Small made-up applicants with the columns the pipeline needs (no Kaggle data required)."""

import numpy as np
import pandas as pd


def fake_applications(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ext1 = rng.uniform(0, 1, n)
    ext1[rng.random(n) < 0.55] = np.nan                     # mostly blank, like the real data
    ext2 = rng.uniform(0, 1, n)
    ext3 = rng.uniform(0, 1, n)
    ext3[rng.random(n) < 0.2] = np.nan
    risk = 1 / (1 + np.exp(-(-2.5 - 2.5 * (ext2 - 0.5) - 1.5 * (np.nan_to_num(ext3, nan=0.4) - 0.5))))
    employed = -rng.integers(100, 12000, n).astype(float)
    employed[rng.random(n) < 0.18] = 365243                   # "not employed" code
    car_age = np.where(rng.random(n) < 0.35, rng.integers(0, 30, n).astype(float), np.nan)
    car_age[rng.random(n) < 0.03] = 64
    credit = rng.uniform(50_000, 1_500_000, n)
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(100000, 100000 + n),
        "TARGET": (rng.random(n) < risk).astype(int),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["F", "M"], n),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married"], n),
        "FLAG_OWN_CAR": np.where(np.isnan(car_age), "N", "Y"),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
        "NAME_EDUCATION_TYPE": rng.choice(["Higher education", "Secondary / secondary special"], n),
        "ORGANIZATION_TYPE": np.where(employed == 365243, "XNA", rng.choice(["Business", "School"], n)),
        "CNT_CHILDREN": rng.integers(0, 4, n),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
        "AMT_INCOME_TOTAL": rng.lognormal(11.9, 0.5, n),
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": credit / rng.uniform(10, 30, n),
        "AMT_GOODS_PRICE": credit / rng.uniform(1.0, 1.3, n),
        "DAYS_BIRTH": -rng.integers(21 * 365, 69 * 365, n),
        "DAYS_EMPLOYED": employed,
        "DAYS_REGISTRATION": -rng.uniform(0, 20000, n),
        "DAYS_ID_PUBLISH": -rng.integers(0, 7000, n),
        "DAYS_LAST_PHONE_CHANGE": -rng.uniform(0, 4000, n),
        "OWN_CAR_AGE": car_age,
        "FLAG_MOBIL": np.ones(n, dtype=int),                    # constant -> must be dropped
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
    })
