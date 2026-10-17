# House Price Prediction Using Location, Area and Property Features with Regression Model Comparison

**Punjab College of Technical Engineering - B.Tech 3rd Year, AIML - Python Project**

| Member | Admission Roll No. | Roll No. |
|---|---|---|
| Manmeet Kaur | 2024AIML0593 | 2445863 |
| Avneet Kaur | 2024AIML1417 | 2445835 |

Submitted to: Ms. Ishatpreet Kaur

---

## 1. Overview

This project predicts the sale price of a house from its **location (zipcode, latitude, longitude)**, **area (living area, lot size, basement)** and **property features (bedrooms, bathrooms, floors, grade, condition, view, waterfront, age, renovation)**.

It trains and compares four regression models - **Linear Regression, Decision Tree, Random Forest and Gradient Boosting** - on unseen test houses and shows everything in a modern **Streamlit dashboard**.

### About the data (please read)

* The dataset is `data/kc_house_data.csv`: **21,613 real house sales from King County (Seattle area), USA, May 2014 - May 2015**, with 21 columns. It was supplied by the project team.
* Prices are in **US dollars of 2014-2015**. The models learn only from this file, so predictions are **not today's market prices** and **not valid for other cities**.
* No accuracy number in this project is typed in by hand. Every metric is computed live when you run it.
* The dataset has **no parking column**, so parking cannot be a model input.
* You can upload your own CSV, but it must have the same columns as `kc_house_data.csv`.

## 2. Features

* Modern real-estate dashboard (cards, tabs, charts, custom theme)
* Price prediction form with input validation and clear error messages
* Data cleaning, missing-value handling, one-hot encoding, scaling - all inside leak-free scikit-learn pipelines
* Four models compared with **MAE, RMSE and R2** on a held-out test set
* Charts: model comparison, actual vs predicted, error distribution, feature importance
* Exploratory data analysis: distributions, correlations, location map, top zipcodes
* "Best model" is selected automatically from the test results (or choose one manually)
* Sample predictions table + downloadable CSV of results
* Automatic sanity checks that prove there is no data leakage

## 3. Folder structure

```
house_price_prediction/
|-- app.py                      <- the Streamlit dashboard (run this)
|-- run_pipeline.py             <- trains the models in the terminal (no browser)
|-- requirements.txt            <- list of libraries to install
|-- README.md                   <- this file
|-- .streamlit/
|   `-- config.toml             <- theme colours and Streamlit settings
|-- data/
|   `-- kc_house_data.csv       <- the dataset
|-- src/
|   |-- __init__.py             <- makes "src" a Python package
|   |-- config.py               <- all settings in one place
|   |-- data_utils.py           <- load, validate, clean, create features
|   |-- house_price_pipeline.py <- preprocessing, 4 models, evaluation, prediction
|   |-- visuals.py              <- all Matplotlib / Seaborn charts
|   |-- ui_styles.py            <- CSS and HTML helpers for the dashboard look
|   `-- sanity_checks.py        <- automatic tests (leakage, metrics, predictions)
|-- outputs/                    <- results saved by run_pipeline.py
|   |-- model_metrics.csv
|   `-- test_predictions.csv
`-- docs/
    |-- RUN_GUIDE_VSCODE_WINDOWS.md
    |-- TROUBLESHOOTING.md
    |-- PROJECT_EXPLAINED.md
    `-- REPORT_AND_VIVA.md
```

## 4. Installation (short version)

Full beginner steps for VS Code on Windows are in `docs/RUN_GUIDE_VSCODE_WINDOWS.md`.

```
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Usage

```
python run_pipeline.py          # optional: train + print results in the terminal
python -m src.sanity_checks     # optional: run the automatic checks
streamlit run app.py            # start the dashboard (opens in your browser)
```

The first start trains the four models (about 30-60 seconds). After that, results are cached and the app is fast.

## 6. How it works

1. Load the CSV and check that all required columns exist.
2. Clean: remove duplicates, invalid prices/dates and impossible rows (for example a 33-bedroom house with 1,620 sq ft).
3. Create features: `house_age`, `was_renovated`, `has_basement`, `living_to_lot_ratio`.
4. **Split first** (80% train / 20% test, fixed seed 42).
5. Pipeline: fill missing values with the training median / most frequent value, standardise numbers, one-hot encode zipcode.
6. Train the four models on the training set only. They learn `log(price)`; metrics are reported in dollars.
7. Evaluate on the test set: MAE, RMSE, R2 (plus train R2 to detect over-fitting).
8. Predict a new house with the selected model.

## 7. Limitations

* One region and one year of data; prices will not match the current market.
* Models were not heavily tuned (no cross-validation or grid search) to keep the code easy to follow.
* Choosing the best model on the test set is slightly optimistic; a validation set or cross-validation would be stricter.
* Predictions are estimates, not valuations. Feature importance shows what the model uses, not what causes prices.

## 8. Credits

Dataset: King County house sales (publicly shared as "House Sales in King County, USA"). Libraries: pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit.
