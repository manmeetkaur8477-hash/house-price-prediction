"""
app.py - the Streamlit dashboard (the file you run).

    streamlit run app.py

Layout of this file:
    1. page setup and small helper functions
    2. sidebar (dataset choice and settings)
    3. load -> clean -> train (results are cached so the app stays fast)
    4. five tabs: Overview | Data Explorer | Model Comparison |
                  Price Predictor | Sample Predictions
"""

import html
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config, visuals
from src.data_utils import (DataValidationError, build_input_row, prepare_dataset,
                            validate_inputs)
from src.house_price_pipeline import (build_predictions_table, get_feature_importance,
                                      predict_single, summarize_results, train_and_compare)
from src.ui_styles import (callout, footer, hero, inject_css, kpi, price_card, section,
                           step_card, usd)

# ----------------------------------------------------------------------------
# 1. Page setup and helpers
# ----------------------------------------------------------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
inject_css()
visuals.apply_theme()


def show_table(data, **kwargs):
    """st.dataframe that stretches to full width on old and new Streamlit versions."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:
        st.dataframe(data, use_container_width=True, **kwargs)


def download_csv(label: str, frame: pd.DataFrame, file_name: str, key: str):
    """A CSV download button (also works on old and new Streamlit versions)."""
    payload = frame.to_csv(index=False).encode("utf-8")
    try:
        st.download_button(label, payload, file_name, mime="text/csv", key=key, width="stretch")
    except Exception:
        st.download_button(label, payload, file_name, mime="text/csv", key=key,
                           use_container_width=True)


@st.cache_resource(show_spinner=False)
def chart_store() -> dict:
    """A dictionary that remembers finished charts, so reruns are instant."""
    return {}


def show_chart(key: str, builder):
    """Build a chart once, then reuse it every time the page reruns."""
    store = chart_store()
    if key not in store:
        figure = builder()
        plt.close(figure)               # stops matplotlib from piling up open figures
        store[key] = figure
    st.pyplot(store[key])


@st.cache_data(show_spinner=False)
def load_bundled_csv() -> pd.DataFrame:
    return pd.read_csv(config.DATA_PATH)


@st.cache_data(show_spinner=False)
def read_uploaded_csv(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))


@st.cache_data(show_spinner=False)
def prepare(raw: pd.DataFrame):
    return prepare_dataset(raw)


@st.cache_resource(show_spinner=False)
def train(data: pd.DataFrame, test_size: float, seed: int):
    return train_and_compare(data, test_size, seed)


# ----------------------------------------------------------------------------
# 2. Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    source = st.radio("Dataset", ["Provided King County dataset", "Upload my own CSV"])
    uploaded = None
    if source == "Upload my own CSV":
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        st.caption("The CSV needs the same columns as the provided dataset. "
                   "Download it below to see the format.")
        try:
            st.download_button("Download dataset format (CSV)",
                               config.DATA_PATH.read_bytes(), "kc_house_data.csv",
                               mime="text/csv")
        except OSError:
            st.caption("(The bundled dataset file could not be read.)")
    test_size = st.slider("Share of houses kept for testing", 0.10, 0.40,
                          config.DEFAULT_TEST_SIZE, 0.05,
                          help="These houses are hidden from the models while they learn.")
    seed = int(st.number_input("Random seed", 0, 9999, config.RANDOM_SEED, 1,
                               help="Same seed = same split = same results every time."))

# ----------------------------------------------------------------------------
# 3. Load -> clean -> train
# ----------------------------------------------------------------------------
using_upload = uploaded is not None
try:
    raw = read_uploaded_csv(uploaded.getvalue()) if using_upload else load_bundled_csv()
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

try:
    data, report = prepare(raw)
except DataValidationError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Unexpected problem while preparing the data: {error}")
    st.stop()

data_label = f"Uploaded file: {uploaded.name}" if using_upload else "King County, USA house sales"
signature = f"{data_label}|{len(data)}|{data[config.TARGET].sum():.0f}|{test_size}|{seed}"

try:
    with st.spinner("Training 4 regression models on the training houses "
                    "(about 30-60 seconds the first time, instant afterwards) ..."):
        results = train(data, test_size, seed)
except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()

with st.sidebar:
    choice = st.selectbox("Model used for predictions",
                          ["Best model (auto)"] + config.MODEL_NAMES,
                          help="'Best' = the model with the lowest RMSE on the test set.")
    st.caption(f"Best on the test set: **{results.best_model}**")
selected = results.best_model if choice.startswith("Best") else choice

date_min, date_max = data["sale_date"].min(), data["sale_date"].max()
metrics = results.metrics

# ----------------------------------------------------------------------------
# Hero banner
# ----------------------------------------------------------------------------
hero("House Price Prediction",
     "Predicting house prices from location, area and property features, and comparing "
     "four regression models on unseen test houses.",
     [data_label, f"{len(data):,} houses", "Linear | Tree | Forest | Boosting",
      "Python + scikit-learn + Streamlit"])

tab_overview, tab_eda, tab_models, tab_predict, tab_samples = st.tabs(
    ["Overview", "Data Explorer", "Model Comparison", "Price Predictor", "Sample Predictions"])

# ----------------------------------------------------------------------------
# TAB 1 - Overview
# ----------------------------------------------------------------------------
with tab_overview:
    cols = st.columns(4)
    with cols[0]:
        kpi("Houses used", f"{report['rows_clean']:,}", f"{report['rows_raw']:,} in the file")
    with cols[1]:
        kpi("Best model", results.best_model, "lowest test RMSE", config.NAVY)
    with cols[2]:
        kpi("Best test R2", f"{metrics.loc[results.best_model, 'R2 (test)']:.3f}",
            "1.0 would be perfect", config.GOLD)
    with cols[3]:
        kpi("Best test MAE", usd(metrics.loc[results.best_model, "MAE ($)"]),
            "average error per house", config.CORAL)

    if using_upload:
        callout(f"<b>Your uploaded dataset ({html.escape(uploaded.name)}) is in use.</b> "
                "The models learn only from this file, so every number below describes this "
                "dataset only.", "warn")
    else:
        callout("<b>About the data:</b> real house-sale records for King County (the Seattle area, "
                f"USA), sold between {date_min:%B %Y} and {date_max:%B %Y}, supplied with this "
                "project. Prices are in US dollars of that period. A model can only reflect the "
                "data it learned from: these predictions are <b>not</b> today's prices and not "
                "valid for other cities.", "warn")

    section("How the project works", "From raw CSV to a price estimate in six steps")
    steps = [("Load and validate", "Check the CSV has every required column"),
             ("Clean", "Remove duplicates, impossible rows, bad prices"),
             ("Create features", "House age, renovated, basement, area ratio"),
             ("Split first", f"{int((1 - test_size) * 100)}% train / "
                             f"{int(test_size * 100)}% hidden test"),
             ("Train 4 models", "Pipelines learn from the training set only"),
             ("Evaluate", "MAE, RMSE and R2 on the unseen test set")]
    for column, (title, text), number in zip(st.columns(6), steps, range(1, 7)):
        with column:
            step_card(number, title, text)

    left, right = st.columns(2)
    with left:
        section("Data cleaning report", "Every step is counted, nothing is hidden")
        cleaning = pd.DataFrame({
            "Step": ["Rows in the file", "Duplicate rows removed", "Rows with invalid price removed",
                     "Rows with unreadable date removed", "Impossible rows removed "
                     f"(e.g. more than {config.MAX_BEDROOMS} bedrooms)", "Rows used",
                     "Missing values left (filled inside the pipeline)"],
            "Count": [report["rows_raw"], report["duplicates_removed"],
                      report["invalid_price_removed"], report["invalid_date_removed"],
                      report["impossible_rows_removed"], report["rows_clean"],
                      report["missing_values_left"]]})
        show_table(cleaning, hide_index=True)
    with right:
        section("How data leakage is avoided", "Leakage = the model secretly seeing test data")
        st.markdown(
            "- The train/test split is made **before** anything is learned.\n"
            "- Missing-value medians, scaling means and zipcode categories are learned "
            "**inside the pipeline from training houses only**.\n"
            "- New features (house age, ratios) use only the same house's own values.\n"
            "- The target (price) is never used to build a feature.\n"
            "- Test houses are used only to *measure* the models, never to train them.")

    with st.expander("Features given to the models"):
        st.write("**Numeric:** " + ", ".join(config.NUMERIC_FEATURES))
        st.write("**Category:** zipcode (one-hot encoded)")
        st.caption("house_age, was_renovated, has_basement and living_to_lot_ratio are created by "
                   "the project from the original columns. The dataset has no parking column, so "
                   "parking cannot be used as an input.")

# ----------------------------------------------------------------------------
# TAB 2 - Data Explorer (EDA)
# ----------------------------------------------------------------------------
with tab_eda:
    section("Exploratory data analysis", "Understanding the data before modelling")
    cols = st.columns(4)
    with cols[0]:
        kpi("Median price", usd(data[config.TARGET].median()), "half the houses cost less")
    with cols[1]:
        kpi("Average price", usd(data[config.TARGET].mean()), "pulled up by luxury homes", config.GOLD)
    with cols[2]:
        kpi("Median living area", f"{data['sqft_living'].median():,.0f} sq ft", "", config.NAVY)
    with cols[3]:
        kpi("Zipcodes", f"{data['zipcode'].nunique()}", "different locations", config.CORAL)

    with st.expander("Preview the data and summary statistics"):
        preview_cols = [c for c in config.REQUIRED_COLUMNS if c in data.columns]
        show_table(data[preview_cols].head(10), hide_index=True)
        stats = data[["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
                      "grade", "house_age"]].describe().T
        show_table(stats.round(2))

    show_chart(f"dist|{signature}", lambda: visuals.price_distribution(data))
    c1, c2 = st.columns(2)
    with c1:
        show_chart(f"area|{signature}", lambda: visuals.price_vs_area(data))
    with c2:
        show_chart(f"grade|{signature}", lambda: visuals.price_by_grade(data))
    c1, c2 = st.columns(2)
    with c1:
        show_chart(f"map|{signature}", lambda: visuals.location_map(data))
    with c2:
        show_chart(f"zip|{signature}", lambda: visuals.top_zipcodes(data))
    show_chart(f"corr|{signature}", lambda: visuals.correlation_heatmap(data))

    correlations = (data[config.NUMERIC_FEATURES + [config.TARGET]].corr()[config.TARGET]
                    .drop(config.TARGET).abs().sort_values(ascending=False))
    top3 = ", ".join(f"<b>{name}</b> ({value:.2f})" for name, value in correlations.head(3).items())
    skew = data[config.TARGET].skew()
    callout(f"<b>What the data tells us:</b> the features most strongly related to price are {top3} "
            f"(absolute correlation, 1.0 = perfect). Prices are right-skewed (skewness {skew:.1f}): "
            "a few very expensive houses stretch the average, which is why the median is lower "
            "than the mean and why the models learn on a log scale.")

    if report["missing_by_column"]:
        section("Missing values")
        show_table(pd.DataFrame({"Column": list(report["missing_by_column"]),
                                 "Missing": list(report["missing_by_column"].values())}),
                   hide_index=True)

# ----------------------------------------------------------------------------
# TAB 3 - Model comparison
# ----------------------------------------------------------------------------
with tab_models:
    section("Model comparison", "All four models were trained on the same houses and tested on the "
                                "same unseen houses")
    styled = (metrics.style
              .format({"MAE ($)": "${:,.0f}", "RMSE ($)": "${:,.0f}", "R2 (test)": "{:.3f}",
                       "R2 (train)": "{:.3f}", "Typical error (%)": "{:.1f}%",
                       "Training time (s)": "{:.1f}"})
              .highlight_min(subset=["MAE ($)", "RMSE ($)"], color="#CDEFEA")
              .highlight_max(subset=["R2 (test)"], color="#CDEFEA"))
    show_table(styled)
    st.caption("MAE = average mistake in dollars. RMSE = like MAE but punishes big mistakes more. "
               "R2 = share of price variation explained (1.0 perfect, 0 = no better than guessing "
               "the average). Highlighted = best. All values are computed live from this run.")

    show_chart(f"cmp|{signature}", lambda: visuals.model_comparison(metrics))

    section("What do the results mean?")
    for line in summarize_results(results):
        st.markdown(f"- {line}")
    callout("The best model is chosen using the test set, as required for this project. For a "
            "stricter study, choose the model with cross-validation on the training data and use "
            "the test set only once at the very end.", "warn")

    section("Actual vs predicted price", "Points close to the dashed line are good predictions")
    show_chart(f"avp|{signature}", lambda: visuals.actual_vs_predicted_grid(
        results.y_test, results.predictions))

    section(f"Prediction errors - {selected}")
    show_chart(f"res|{signature}|{selected}", lambda: visuals.residual_plots(
        results.y_test, results.predictions[selected], selected))

    section(f"What drives the price? - {selected}")
    importance, kind = get_feature_importance(results, selected)
    c1, c2 = st.columns([3, 2])
    with c1:
        show_chart(f"imp|{signature}|{selected}",
                   lambda: visuals.feature_importance_chart(importance, kind, selected))
    with c2:
        top = importance.index[:3].tolist()
        st.markdown("**Reading this chart**")
        if kind == "importance":
            st.markdown(f"The most influential features for {selected} are **{top[0]}**, "
                        f"**{top[1]}** and **{top[2]}**. Importance shows how much each feature "
                        "was used by the model to split houses into cheaper and dearer groups. "
                        "All zipcode columns are added together as one 'location' bar.")
        else:
            st.markdown(f"The largest coefficients belong to **{top[0]}**, **{top[1]}** and "
                        f"**{top[2]}**. Larger absolute coefficient = a bigger effect on the "
                        "(log) price, with features on the same scale.")
        st.caption("Feature importance describes the model, not cause and effect.")

    with st.expander("How each model works (simple explanation)"):
        st.markdown(
            "**Linear Regression** draws the best straight-line (weighted sum) relationship between "
            "features and price. Fast and easy to explain, but cannot capture curved or "
            "interacting effects.\n\n"
            "**Decision Tree** asks a chain of yes/no questions (for example 'living area above "
            "2,000 sq ft?') and ends in a price. Easy to read but can memorise the training data.\n\n"
            "**Random Forest** trains many different trees on random parts of the data and "
            "averages their answers, which reduces over-fitting.\n\n"
            "**Gradient Boosting** builds trees one after another; each new tree focuses on the "
            "mistakes the previous trees made. Often the most accurate on table data.")

# ----------------------------------------------------------------------------
# TAB 4 - Price predictor
# ----------------------------------------------------------------------------
with tab_predict:
    section("Predict a house price", f"Using the {selected} model. Fill in the details and press "
                                     "'Predict price'.")
    callout("Estimates reflect the dataset the model learned from "
            f"({'your uploaded file' if using_upload else 'King County, USA, 2014-2015'}). "
            "They are for learning purposes and are not a professional valuation.", "warn")

    zip_stats = results.zip_stats
    zip_options = list(zip_stats.index)
    default_zip = zip_stats["houses"].idxmax()
    ref_year = results.reference_year

    zipcode = st.selectbox("Location (zipcode)", zip_options, index=zip_options.index(default_zip),
                           help="Location is one of the strongest price drivers.")
    zrow = zip_stats.loc[zipcode]
    st.caption(f"Zipcode {zipcode}: {int(zrow['houses'])} houses in the data, "
               f"median price {usd(zrow['median_price'])}.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Size**")
            sqft_living = st.number_input("Living area (sq ft)", 300, 15000, 1800, 50)
            sqft_lot = st.number_input("Lot size (sq ft)", 500, 1700000, 6000, 100)
            floors = st.select_slider("Floors", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], value=1.0)
            sqft_basement = st.number_input("Basement area (sq ft, 0 = none)", 0, 5000, 0, 50)
        with c2:
            st.markdown("**Rooms and age**")
            bedrooms = st.number_input("Bedrooms", 1, 12, 3, 1)
            bathrooms = st.number_input("Bathrooms", 0.5, 8.0, 2.0, 0.25,
                                        help="0.5 = toilet only, 0.75 = shower, 1 = full bath")
            yr_built = st.number_input("Year built", 1900, ref_year, min(1990, ref_year), 1)
            yr_renovated = st.number_input("Year renovated (0 = never)", 0, ref_year, 0, 1)
        with c3:
            st.markdown("**Quality and surroundings**")
            grade = st.slider("Construction grade (1 low - 13 luxury)", 1, 13, 7)
            condition = st.slider("Condition (1 poor - 5 very good)", 1, 5, 3)
            view = st.slider("View quality (0 none - 4 excellent)", 0, 4, 0)
            waterfront = st.checkbox("Waterfront property")

        with st.expander("Advanced: exact coordinates and neighbourhood (pre-filled for the zipcode)"):
            a1, a2 = st.columns(2)
            lat = a1.number_input("Latitude", value=float(zrow["lat"]), format="%.4f",
                                  key=f"lat_{zipcode}")
            long = a2.number_input("Longitude", value=float(zrow["long"]), format="%.4f",
                                   key=f"long_{zipcode}")
            n1, n2 = st.columns(2)
            living15 = n1.number_input("Average living area of 15 nearest houses (sq ft)",
                                       value=float(zrow["sqft_living15"]), min_value=100.0,
                                       key=f"l15_{zipcode}")
            lot15 = n2.number_input("Average lot size of 15 nearest houses (sq ft)",
                                    value=float(zrow["sqft_lot15"]), min_value=100.0,
                                    key=f"lot15_{zipcode}")
        submitted = st.form_submit_button("Predict price")

    if submitted:
        inputs = {"zipcode": zipcode, "sqft_living": sqft_living, "sqft_lot": sqft_lot,
                  "floors": floors, "sqft_basement": sqft_basement, "bedrooms": bedrooms,
                  "bathrooms": bathrooms, "yr_built": int(yr_built), "yr_renovated": int(yr_renovated),
                  "grade": grade, "condition": condition, "view": view,
                  "waterfront": int(waterfront), "lat": lat, "long": long,
                  "sqft_living15": living15, "sqft_lot15": lot15}
        errors, warnings_ = validate_inputs(inputs, results.ranges, ref_year)
        if errors:
            st.session_state.pop("prediction", None)
            for message in errors:
                st.error(message)
        else:
            try:
                row = build_input_row(inputs, ref_year)
                all_prices = {name: predict_single(results, name, row) for name in config.MODEL_NAMES}
                st.session_state["prediction"] = {
                    "signature": signature, "inputs": inputs, "prices": all_prices,
                    "model": selected, "warnings": warnings_}
            except Exception as error:
                st.session_state.pop("prediction", None)
                st.error(f"The prediction failed: {error}")

    saved = st.session_state.get("prediction")
    if saved and saved["signature"] == signature:
        model_used = saved["model"]
        price = saved["prices"][model_used]
        error_rate = results.median_ape[model_used]
        for message in saved["warnings"]:
            st.warning(message)

        left, right = st.columns([3, 2])
        with left:
            price_card(model_used, price, price * (1 - error_rate), price * (1 + error_rate),
                       error_rate * 100)
        with right:
            inp = saved["inputs"]
            kpi("Price per sq ft", f"${price / inp['sqft_living']:,.0f}", "of living area")
            st.write("")
            kpi("Median price in this zipcode", usd(results.zip_stats.loc[inp["zipcode"], "median_price"]),
                "from the dataset", config.GOLD)

        section("What do the other models say?",
                "If the models disagree strongly, treat the estimate with more caution.")
        comparison = pd.DataFrame({"Model": list(saved["prices"]),
                                   "Predicted price ($)": list(saved["prices"].values())})
        comparison["Difference from selected"] = comparison["Predicted price ($)"] - price
        show_table(comparison.style.format({"Predicted price ($)": "${:,.0f}",
                                            "Difference from selected": "${:+,.0f}"}),
                   hide_index=True)

        result_row = pd.DataFrame([{**saved["inputs"],
                                    **{f"predicted_price_{n.lower().replace(' ', '_')}": p
                                       for n, p in saved["prices"].items()},
                                    "selected_model": model_used}])
        download_csv("Download this prediction (CSV)", result_row, "my_house_prediction.csv",
                     key="download_single")

# ----------------------------------------------------------------------------
# TAB 5 - Sample predictions
# ----------------------------------------------------------------------------
with tab_samples:
    section("Sample predictions on unseen houses",
            f"{selected} predicting houses from the test set - it never saw these while learning")
    table = build_predictions_table(results, selected)

    c1, c2 = st.columns([2, 3])
    with c1:
        how = st.radio("Which houses?", ["Random sample", "Largest mistakes"], horizontal=True)
    with c2:
        count = st.slider("Number of houses", 5, 30, 12)

    if how == "Random sample":
        shown = table.sample(min(count, len(table)), random_state=seed).reset_index(drop=True)
    else:
        shown = table.sort_values("abs_error", ascending=False).head(count).reset_index(drop=True)

    display = shown[["zipcode", "bedrooms", "bathrooms", "sqft_living", "grade", "house_age",
                     "actual_price", "predicted_price", "error_pct"]].copy()
    display.index = display.index + 1
    show_table(display.style.format({"sqft_living": "{:,.0f}", "house_age": "{:.0f}",
                                     "actual_price": "${:,.0f}", "predicted_price": "${:,.0f}",
                                     "error_pct": "{:+.1f}%"}))
    show_chart(f"sample|{signature}|{selected}|{how}|{count}",
               lambda: visuals.sample_comparison(shown))

    mean_abs_pct = table["error_pct"].abs().mean()
    callout(f"Across all {len(table):,} test houses, {selected} is off by "
            f"{mean_abs_pct:.1f}% on average. Positive error = the model guessed too high.")

    section("Download results")
    d1, d2 = st.columns(2)
    with d1:
        download_csv(f"Test-set predictions - {selected} + all models (CSV)", table,
                     "prediction_results.csv", key="download_predictions")
    with d2:
        download_csv("Model metrics (CSV)", metrics.reset_index(), "model_metrics.csv",
                     key="download_metrics")

footer()
