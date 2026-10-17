"""
data_utils.py - loading, validating, cleaning and feature engineering.

Important idea (avoiding data leakage):
    Everything in this file works ROW BY ROW (e.g. "age = sale year - year built")
    or removes clearly impossible rows by fixed rules. Nothing here learns a
    statistic (like an average) from the whole dataset. Statistics such as the
    median used to fill missing values are learned later, inside the
    scikit-learn pipeline, using ONLY the training houses.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import config


class DataValidationError(Exception):
    """Raised with a friendly message when the dataset cannot be used."""


# ----------------------------------------------------------------------------
# 1. Validation
# ----------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    """Check that the dataset has all the columns the project needs."""
    if df is None or df.empty:
        raise DataValidationError("The dataset is empty.")

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            "The CSV is missing these required columns: "
            + ", ".join(missing)
            + ". Please use the same columns as data/kc_house_data.csv."
        )


# ----------------------------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------------------------
def _parse_dates(series: pd.Series) -> pd.Series:
    """Convert dates such as '20141013T000000' to real dates."""
    parsed = pd.to_datetime(series.astype(str), format="%Y%m%dT%H%M%S", errors="coerce")
    if parsed.isna().mean() > 0.5:  # a different date format? try a general parser
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def _clean_zipcode(series: pd.Series) -> pd.Series:
    """Zipcodes are labels, not numbers: store them as text like '98178'."""
    numbers = pd.to_numeric(series, errors="coerce")
    text = numbers.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    return text.astype(object)


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the raw dataset and return (clean_dataframe, report).

    The report is a dictionary that the dashboard shows to the user, so every
    cleaning step is visible and explainable.
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    report = {"rows_raw": int(len(df))}

    # Make numeric columns numeric (bad text such as "N/A" becomes missing).
    for col in config.NUMERIC_RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["zipcode"] = _clean_zipcode(df["zipcode"])

    # Step 1 - exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    # Step 2 - rows without a valid price cannot be used for learning
    before = len(df)
    df = df[df[config.TARGET].notna() & (df[config.TARGET] > 0)]
    report["invalid_price_removed"] = int(before - len(df))

    # Step 3 - rows whose sale date cannot be read
    df = df.copy()
    df["sale_date"] = _parse_dates(df["date"])
    before = len(df)
    df = df[df["sale_date"].notna()].copy()
    report["invalid_date_removed"] = int(before - len(df))
    df["sale_year"] = df["sale_date"].dt.year.astype(int)

    # Step 4 - impossible values (fixed rules, no learning involved)
    before = len(df)
    impossible = (
        (df["bedrooms"] > config.MAX_BEDROOMS)
        | (df["sqft_living"] <= 0)
        | (df["bathrooms"] < 0)
    )
    df = df[~impossible]
    report["impossible_rows_removed"] = int(before - len(df))

    if len(df) < config.MIN_ROWS:
        raise DataValidationError(
            f"Only {len(df)} usable rows remain after cleaning; at least "
            f"{config.MIN_ROWS} are needed."
        )

    # Remaining missing values are NOT filled here. They are filled inside the
    # model pipeline using training data only (see house_price_pipeline.py).
    feature_cols = [c for c in config.REQUIRED_COLUMNS if c not in ("date", config.TARGET)]
    missing_per_column = df[feature_cols].isna().sum()
    report["missing_values_left"] = int(missing_per_column.sum())
    report["missing_by_column"] = missing_per_column[missing_per_column > 0].to_dict()
    report["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), report


# ----------------------------------------------------------------------------
# 3. Feature engineering (row-by-row, so no leakage)
# ----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new, easier-to-understand columns from the existing ones."""
    out = df.copy()

    # How old the house was when it was sold (never below 0).
    out["house_age"] = (out["sale_year"] - out["yr_built"]).clip(lower=0)

    # 1 if the house was ever renovated (yr_renovated == 0 means "never").
    out["was_renovated"] = (out["yr_renovated"].fillna(0) > 0).astype(int)

    # 1 if the house has a basement.
    out["has_basement"] = (out["sqft_basement"].fillna(0) > 0).astype(int)

    # How much of the plot is covered by living space.
    ratio = out["sqft_living"] / out["sqft_lot"]
    out["living_to_lot_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
    return out


def prepare_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full preparation: validate -> clean -> engineer features."""
    validate_schema(raw)
    clean, report = clean_data(raw)
    prepared = engineer_features(clean)
    report["reference_year"] = int(prepared["sale_year"].max())
    return prepared, report


# ----------------------------------------------------------------------------
# 4. Helpers used by the dashboard
# ----------------------------------------------------------------------------
def zipcode_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Typical values for every zipcode (used to pre-fill the prediction form)."""
    stats = (
        df.groupby("zipcode")
        .agg(
            lat=("lat", "median"),
            long=("long", "median"),
            sqft_living15=("sqft_living15", "median"),
            sqft_lot15=("sqft_lot15", "median"),
            median_price=(config.TARGET, "median"),
            houses=(config.TARGET, "size"),
        )
        .reset_index()
        .sort_values("zipcode")
    )
    return stats.set_index("zipcode")


def feature_ranges(df: pd.DataFrame) -> dict:
    """Smallest and largest value of each numeric feature in the data."""
    ranges = {}
    for col in config.NUMERIC_FEATURES:
        ranges[col] = (float(df[col].min()), float(df[col].max()))
    return ranges


def validate_inputs(inputs: dict, ranges: dict, reference_year: int) -> tuple[list, list]:
    """
    Check the values typed in the prediction form.
    Returns (errors, warnings). Errors block the prediction; warnings do not.
    """
    errors, warnings_ = [], []

    if inputs["sqft_basement"] >= inputs["sqft_living"]:
        errors.append("Basement area must be smaller than the total living area.")
    if inputs["bedrooms"] < 1:
        errors.append("A house needs at least 1 bedroom.")
    if inputs["bathrooms"] <= 0:
        errors.append("A house needs at least 0.5 bathrooms.")
    if inputs["yr_built"] > reference_year:
        errors.append(f"Year built cannot be after {reference_year} (the latest sale year in the data).")
    renovated = int(inputs.get("yr_renovated", 0) or 0)
    if renovated != 0 and not (inputs["yr_built"] <= renovated <= reference_year):
        errors.append("Year renovated must be 0 (never renovated) or between the year built and "
                      f"{reference_year}.")

    # Warnings: the model is least reliable outside the range it learned from.
    for column, label in [("sqft_living", "Living area"), ("sqft_lot", "Lot size"),
                          ("bedrooms", "Number of bedrooms"), ("bathrooms", "Number of bathrooms")]:
        low, high = ranges[column]
        if not (low <= inputs[column] <= high):
            warnings_.append(f"{label} is outside the range seen in the training data "
                             f"({low:,.0f} to {high:,.0f}); the estimate may be unreliable.")
    if inputs["sqft_living"] / max(inputs["bedrooms"], 1) < 150:
        warnings_.append("Very little living area per bedroom - please double-check the values.")
    if inputs["sqft_lot"] < inputs["sqft_living"] / max(inputs["floors"], 1):
        warnings_.append("The lot is smaller than the building footprint (possible for condos or "
                         "townhouses).")
    return errors, warnings_


def build_input_row(inputs: dict, reference_year: int) -> pd.DataFrame:
    """
    Turn the values typed in the prediction form into a one-row DataFrame
    with EXACTLY the columns the models were trained on.
    """
    sqft_living = float(inputs["sqft_living"])
    sqft_lot = float(inputs["sqft_lot"])
    sqft_basement = float(inputs["sqft_basement"])
    yr_built = int(inputs["yr_built"])
    yr_renovated = int(inputs.get("yr_renovated", 0) or 0)

    row = {
        "bedrooms": float(inputs["bedrooms"]),
        "bathrooms": float(inputs["bathrooms"]),
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": float(inputs["floors"]),
        "waterfront": int(inputs["waterfront"]),
        "view": int(inputs["view"]),
        "condition": int(inputs["condition"]),
        "grade": int(inputs["grade"]),
        "sqft_above": max(sqft_living - sqft_basement, 0.0),
        "sqft_basement": sqft_basement,
        "lat": float(inputs["lat"]),
        "long": float(inputs["long"]),
        "sqft_living15": float(inputs["sqft_living15"]),
        "sqft_lot15": float(inputs["sqft_lot15"]),
        "house_age": float(max(reference_year - yr_built, 0)),
        "was_renovated": int(yr_renovated > 0),
        "has_basement": int(sqft_basement > 0),
        "living_to_lot_ratio": sqft_living / sqft_lot if sqft_lot > 0 else np.nan,
        "zipcode": str(inputs["zipcode"]),
    }
    return pd.DataFrame([row], columns=config.FEATURE_COLUMNS)
