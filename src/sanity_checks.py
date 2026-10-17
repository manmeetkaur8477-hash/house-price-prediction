"""
sanity_checks.py - automatic tests that prove the project is correct.

Run it from the project folder:
    python -m src.sanity_checks

Each check prints PASS or FAIL. These are good to show in your viva: they
prove there is no data leakage and that no metric is faked.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.data_utils import DataValidationError, build_input_row, prepare_dataset
from src.house_price_pipeline import _inner_pipeline, predict_single, train_and_compare

FAILED = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        FAILED.append(name)


def main() -> int:
    raw = pd.read_csv(config.DATA_PATH)
    data, report = prepare_dataset(raw)

    # 1. data preparation -----------------------------------------------------
    check("All model feature columns exist",
          all(c in data.columns for c in config.FEATURE_COLUMNS))
    check("Absurd 33-bedroom row was removed", data["bedrooms"].max() <= config.MAX_BEDROOMS)
    check("Zipcode is stored as text", data["zipcode"].map(type).eq(str).all())

    # 2. missing columns are reported clearly -----------------------------------
    try:
        prepare_dataset(raw.drop(columns=["sqft_living"]))
        check("Missing column raises a friendly error", False)
    except DataValidationError:
        check("Missing column raises a friendly error", True)

    # 3. train / test split and leakage ---------------------------------------
    results = train_and_compare(data, config.DEFAULT_TEST_SIZE, config.RANDOM_SEED)
    train_idx, test_idx = set(results.X_train.index), set(results.X_test.index)
    check("Train and test sets do not overlap", train_idx.isdisjoint(test_idx))
    check("Train + test = all rows", len(train_idx) + len(test_idx) == len(data))

    scaler = (_inner_pipeline(results.models[config.LINEAR])
              .named_steps["prep"].named_transformers_["num"].named_steps["scaler"])
    train_mean = results.X_train[config.NUMERIC_FEATURES].mean().values
    full_mean = data[config.NUMERIC_FEATURES].mean().values
    check("Scaler learned its mean from TRAIN data only",
          np.allclose(scaler.mean_, train_mean, equal_nan=True))
    check("...and that mean differs from the full-dataset mean",
          not np.allclose(scaler.mean_, full_mean, equal_nan=True))

    # 4. metrics are really computed (recompute them independently) -------------
    for name in config.MODEL_NAMES:
        pred = results.predictions[name]
        row = results.metrics.loc[name]
        same = (np.isclose(row["MAE ($)"], mean_absolute_error(results.y_test, pred))
                and np.isclose(row["RMSE ($)"], np.sqrt(mean_squared_error(results.y_test, pred)))
                and np.isclose(row["R2 (test)"], r2_score(results.y_test, pred)))
        check(f"Metrics recomputed independently match: {name}", bool(same))

    best = results.metrics["RMSE ($)"].idxmin()
    check("Best model is the one with the lowest test RMSE", best == results.best_model)

    # 5. reproducibility -----------------------------------------------------
    again = train_and_compare(data, config.DEFAULT_TEST_SIZE, config.RANDOM_SEED)
    check("Same seed gives identical results",
          np.allclose(results.metrics["RMSE ($)"], again.metrics["RMSE ($)"]))

    # 6. predictions for new / unusual inputs -----------------------------------
    sample_inputs = {
        "bedrooms": 3, "bathrooms": 2, "sqft_living": 1800, "sqft_lot": 6000,
        "floors": 1, "waterfront": 0, "view": 0, "condition": 3, "grade": 7,
        "sqft_basement": 0, "yr_built": 1990, "yr_renovated": 0,
        "zipcode": data["zipcode"].iloc[0], "lat": 47.5, "long": -122.2,
        "sqft_living15": 1800, "sqft_lot15": 6000,
    }
    for name in config.MODEL_NAMES:
        price = predict_single(results, name, build_input_row(sample_inputs, results.reference_year))
        check(f"Prediction is a sensible positive number: {name}",
              np.isfinite(price) and price > 0, f"(${price:,.0f})")

    unseen = dict(sample_inputs, zipcode="00000")
    price = predict_single(results, config.BOOST, build_input_row(unseen, results.reference_year))
    check("Unseen zipcode does not crash the model", np.isfinite(price))

    holes = results.X_test.head(20).copy()
    holes[config.NUMERIC_FEATURES] = holes[config.NUMERIC_FEATURES].astype(float)
    holes[["bedrooms", "sqft_living", "grade", "lat", "house_age"]] = np.nan
    holes["zipcode"] = np.nan
    pred = results.models[config.FOREST].predict(holes)
    check("Missing values in new data are handled by the pipeline", np.isfinite(pred).all())

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED: {FAILED}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
