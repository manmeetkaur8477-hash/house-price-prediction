"""
house_price_pipeline.py - the machine-learning part of the project.

Workflow implemented here:
    1. split the houses into TRAIN and TEST sets (before any learning)
    2. build a preprocessing pipeline (missing values, scaling, encoding)
    3. train 4 regression models on the TRAIN set only
    4. evaluate every model on the unseen TEST set (MAE, RMSE, R2)
    5. pick the best model and make predictions

How data leakage is avoided:
    The imputer, scaler and encoder live INSIDE a scikit-learn Pipeline. When we
    call .fit() the pipeline learns medians, means and categories from the
    training houses only. The test houses are only ever transformed, never
    learned from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.data_utils import feature_ranges, zipcode_stats


# ----------------------------------------------------------------------------
# Container that holds everything the dashboard needs
# ----------------------------------------------------------------------------
@dataclass
class TrainingResults:
    models: dict                      # model name -> fitted model
    metrics: pd.DataFrame             # one row per model
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: dict                 # model name -> test-set predictions
    best_model: str                   # lowest test RMSE
    test_size: float
    seed: int
    reference_year: int
    zip_stats: pd.DataFrame
    ranges: dict
    median_ape: dict = field(default_factory=dict)   # typical % error per model


# ----------------------------------------------------------------------------
# Step 1 - preprocessing and models
# ----------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Numbers: fill gaps with the median, then scale. Zipcode: fill, one-hot."""
    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a zipcode never seen in training will not crash
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, config.NUMERIC_FEATURES),
        ("cat", categorical_steps, config.CATEGORICAL_FEATURES),
    ])


def build_models(seed: int) -> dict:
    """The four regression models we compare (all use the same preprocessing)."""
    estimators = {
        config.LINEAR: LinearRegression(),
        config.TREE: DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=10, random_state=seed),
        config.FOREST: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        config.BOOST: GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.08, max_depth=4,
            subsample=0.8, random_state=seed),
    }
    models = {}
    for name, estimator in estimators.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        if config.USE_LOG_TARGET:
            pipe = TransformedTargetRegressor(
                regressor=pipe, func=np.log1p, inverse_func=np.expm1)
        models[name] = pipe
    return models


def _inner_pipeline(model) -> Pipeline:
    """Return the fitted Pipeline (unwrapping the log-target wrapper if used)."""
    return model.regressor_ if hasattr(model, "regressor_") else model


# ----------------------------------------------------------------------------
# Step 2 - evaluation
# ----------------------------------------------------------------------------
def predict_prices(model, X: pd.DataFrame) -> np.ndarray:
    """Predict prices in dollars (never below zero)."""
    return np.clip(model.predict(X), 0, None)


def evaluate(y_true, y_pred) -> dict:
    """MAE, RMSE and R2 (all computed on real dollar prices)."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_compare(df: pd.DataFrame, test_size: float = config.DEFAULT_TEST_SIZE,
                      seed: int = config.RANDOM_SEED) -> TrainingResults:
    """Split, train the four models, evaluate them on the test set."""
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET]

    # The split happens FIRST, before anything is learned from the data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    models, predictions, rows, median_ape = {}, {}, [], {}
    for name, blank_model in build_models(seed).items():
        model = clone(blank_model)
        start = time.perf_counter()
        model.fit(X_train, y_train)                 # learns from TRAIN only
        seconds = time.perf_counter() - start

        test_pred = predict_prices(model, X_test)
        train_pred = predict_prices(model, X_train)
        scores = evaluate(y_test, test_pred)

        models[name] = model
        predictions[name] = test_pred
        median_ape[name] = float(np.median(np.abs(y_test - test_pred) / y_test))
        rows.append({
            "Model": name,
            "MAE ($)": scores["MAE"],
            "RMSE ($)": scores["RMSE"],
            "R2 (test)": scores["R2"],
            "R2 (train)": float(r2_score(y_train, train_pred)),
            "Typical error (%)": median_ape[name] * 100,
            "Training time (s)": seconds,
        })

    metrics = pd.DataFrame(rows).set_index("Model")
    # Best model = lowest test RMSE (ties are broken by the higher test R2).
    ordered = metrics.sort_values(["RMSE ($)", "R2 (test)"], ascending=[True, False])
    best_model = ordered.index[0]

    return TrainingResults(
        models=models, metrics=metrics,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        predictions=predictions, best_model=best_model,
        test_size=test_size, seed=seed,
        reference_year=int(df["sale_year"].max()),
        zip_stats=zipcode_stats(df), ranges=feature_ranges(df),
        median_ape=median_ape,
    )


# ----------------------------------------------------------------------------
# Step 3 - using the trained models
# ----------------------------------------------------------------------------
def predict_single(results: TrainingResults, model_name: str, row: pd.DataFrame) -> float:
    """Predict the price of one house (a one-row DataFrame)."""
    return float(predict_prices(results.models[model_name], row)[0])


def build_predictions_table(results: TrainingResults, model_name: str) -> pd.DataFrame:
    """Test-set predictions as a table (used for the sample table and CSV download)."""
    table = results.X_test[
        ["zipcode", "bedrooms", "bathrooms", "sqft_living", "grade", "house_age"]
    ].copy()
    table.insert(0, "row_id", results.X_test.index)
    table["actual_price"] = results.y_test.values
    table["predicted_price"] = results.predictions[model_name]
    table["error"] = table["predicted_price"] - table["actual_price"]
    table["abs_error"] = table["error"].abs()
    table["error_pct"] = table["error"] / table["actual_price"] * 100
    for other in config.MODEL_NAMES:
        table[f"pred_{other.lower().replace(' ', '_')}"] = results.predictions[other]
    return table.reset_index(drop=True)


def get_feature_importance(results: TrainingResults, model_name: str, top_n: int = 15):
    """
    Returns (Series, kind). For tree models this is the built-in importance
    (zipcode columns are added together). For Linear Regression it is the
    size of each coefficient on standardised features.
    """
    pipe = _inner_pipeline(results.models[model_name])
    names = [n.split("__", 1)[-1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    estimator = pipe.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = pd.Series(estimator.feature_importances_, index=names)
        is_zip = values.index.str.startswith("zipcode_")
        grouped = values[~is_zip].copy()
        grouped["zipcode (location)"] = values[is_zip].sum()
        return grouped.sort_values(ascending=False).head(top_n), "importance"

    values = pd.Series(np.abs(estimator.coef_), index=names)
    return values.sort_values(ascending=False).head(top_n), "coefficient"


def summarize_results(results: TrainingResults) -> list:
    """Plain-English explanation of the results, built from the REAL metrics."""
    m = results.metrics
    best = results.best_model
    ranked = m.sort_values("RMSE ($)")
    median_price = float(results.y_test.median())
    lines = []

    lines.append(
        f"**{best}** has the lowest test-set RMSE (${m.loc[best, 'RMSE ($)']:,.0f}) "
        f"and explains {m.loc[best, 'R2 (test)'] * 100:.1f}% of the price variation "
        f"on houses it has never seen (R2 = {m.loc[best, 'R2 (test)']:.3f})."
    )
    lines.append(
        f"Its average mistake (MAE) is ${m.loc[best, 'MAE ($)']:,.0f}, which is about "
        f"{m.loc[best, 'MAE ($)'] / median_price * 100:.0f}% of the median test price "
        f"(${median_price:,.0f}). The typical (median) percentage error is "
        f"{m.loc[best, 'Typical error (%)']:.1f}%."
    )
    lines.append("Ranking from best to worst by RMSE: " + " > ".join(ranked.index) + ".")

    if best != config.LINEAR:
        change = (1 - m.loc[best, "RMSE ($)"] / m.loc[config.LINEAR, "RMSE ($)"]) * 100
        lines.append(
            f"Compared with the simple Linear Regression baseline, {best} lowers RMSE "
            f"by {change:.0f}%, so the non-linear model captures patterns "
            f"(such as location effects) that a straight-line model misses."
        )

    gaps = (m["R2 (train)"] - m["R2 (test)"]).sort_values(ascending=False)
    worst = gaps.index[0]
    lines.append(
        f"Over-fitting check: {worst} has the largest gap between train R2 "
        f"({m.loc[worst, 'R2 (train)']:.3f}) and test R2 ({m.loc[worst, 'R2 (test)']:.3f}). "
        "A big gap means the model memorised the training houses instead of learning "
        "general rules."
    )
    lines.append(
        "These numbers describe how well each model fits THIS dataset only. "
        "They do not guarantee the same accuracy on a different city or year."
    )
    return lines
