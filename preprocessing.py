import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Fixed reference date keeps ages reproducible.
REFERENCE_DATE   = pd.Timestamp("2025-01-01")
BIRTHDATE_FORMAT = "%m/%d/%Y %I:%M %p"


SPEND_COLS = [
    "lifetime_spend_groceries", "lifetime_spend_electronics",
    "lifetime_spend_vegetables", "lifetime_spend_nonalcohol_drinks",
    "lifetime_spend_alcohol_drinks", "lifetime_spend_meat",
    "lifetime_spend_fish", "lifetime_spend_hygiene",
    "lifetime_spend_videogames", "lifetime_spend_petfood",
]

DEGREE_MAP = {"Bsc.": 1, "Msc.": 2, "Phd.": 3}

# Behavioural features only; demographics are used for profiling, not clustering.
BEHAVIOURAL_FEATURES = SPEND_COLS + [
    "percentage_of_products_bought_promotion",
    "lifetime_total_distinct_products",
    "distinct_stores_visited",
]


def _extract_degree(name: str) -> str:
    for deg in DEGREE_MAP:
        if deg in str(name):
            return deg
    return "None"


def _compute_age(birthdates) -> pd.Series:
    """Vectorised age in years; unparseable dates become NaN (imputed later)."""
    bd = pd.to_datetime(birthdates, format=BIRTHDATE_FORMAT, errors="coerce")
    return (REFERENCE_DATE - bd).dt.days / 365.25


def load_and_clean_data(info_path: str) -> pd.DataFrame:
    """Load customer_info CSV, engineer features, and impute missing values."""
    df = pd.read_csv(info_path)

    df["degree"]       = df["customer_name"].apply(_extract_degree)
    df["degree_level"] = df["degree"].map(DEGREE_MAP).fillna(0).astype(int)
    df["age"]          = _compute_age(df["customer_birthdate"])
    df["gender_male"]  = (df["customer_gender"] == "male").astype(int)
    df["total_kids"]   = df["kids_home"].fillna(0) + df["teens_home"].fillna(0)
    df["has_loyalty_card"] = df["loyalty_card_number"].notna().astype(int)
    df["total_spend"]      = df[SPEND_COLS].fillna(0).sum(axis=1)
    df["years_customer"]   = 2025 - df["year_first_transaction"]

    # Median-impute the numeric columns that feed clustering / profiling.
    impute_cols = SPEND_COLS + [
        "number_complaints", "distinct_stores_visited", "typical_hour",
        "lifetime_total_distinct_products",
        "percentage_of_products_bought_promotion", "age",
    ]
    for col in impute_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def preprocess_for_clustering(df: pd.DataFrame):
    """Scaled behavioural matrix for clustering: log1p the skewed spends, then
    standardise. Returns (X_scaled_df, scaler, feature_cols)."""
    X_raw = df[BEHAVIOURAL_FEATURES].copy()
    for col in SPEND_COLS:
        X_raw[col] = np.log1p(X_raw[col].clip(lower=0))

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return pd.DataFrame(X_scaled, columns=BEHAVIOURAL_FEATURES), scaler, BEHAVIOURAL_FEATURES
