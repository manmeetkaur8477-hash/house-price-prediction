# The Project Explained Simply

## 1. What does each file do?

| File | Job in one line | Explain it like this in the viva |
|---|---|---|
| `app.py` | The web dashboard | "It is the front end. It loads the data, trains the models once, and shows five tabs." |
| `run_pipeline.py` | Trains and prints results in the terminal | "Same ML code without the web page. Useful for quick testing and for saving results." |
| `requirements.txt` | List of libraries | "`pip install -r requirements.txt` installs every library the project needs." |
| `.streamlit/config.toml` | Theme colours | "Sets the light real-estate colour theme." |
| `data/kc_house_data.csv` | The dataset | "21,613 house sales in King County, USA, 2014-2015." |
| `src/config.py` | All settings | "Seed, test size, feature lists, model names, colours - one place to change them." |
| `src/data_utils.py` | Data preparation | "Checks the columns, cleans bad rows, creates house age and similar features, validates form inputs." |
| `src/house_price_pipeline.py` | The ML core | "Splits the data, builds preprocessing pipelines, trains four models, computes MAE/RMSE/R2, predicts." |
| `src/visuals.py` | All charts | "Matplotlib/Seaborn charts with one consistent colour style." |
| `src/ui_styles.py` | Dashboard styling | "CSS and small HTML pieces for the banner, cards and price box." |
| `src/sanity_checks.py` | Automatic tests | "Proves no leakage, metrics are really computed, predictions work with missing/unseen values." |
| `outputs/*.csv` | Saved results | "Created by `run_pipeline.py`; overwritten on every run." |

## 2. The dataset columns

| Column | Meaning |
|---|---|
| `id`, `date` | house sale id and sale date (date is used to compute house age) |
| `price` | **target**: sale price in US dollars |
| `bedrooms`, `bathrooms` | number of rooms (bathrooms can be 0.75, 1.5 ...) |
| `sqft_living`, `sqft_lot` | living area and lot area in square feet |
| `floors` | number of floors |
| `waterfront`, `view` | 1 if on the waterfront; view quality 0-4 |
| `condition`, `grade` | condition 1-5; construction/design quality 1-13 |
| `sqft_above`, `sqft_basement` | area above ground and in the basement |
| `yr_built`, `yr_renovated` | year built; year renovated (0 = never) |
| `zipcode`, `lat`, `long` | location |
| `sqft_living15`, `sqft_lot15` | average living / lot area of the 15 nearest neighbours |

Created by the project: `house_age`, `was_renovated`, `has_basement`, `living_to_lot_ratio`.

## 3. The libraries

| Library | What it is | Where we use it |
|---|---|---|
| **Python** | the programming language | everything |
| **pandas** | tables ("DataFrames") like Excel in code | loading CSV, cleaning, grouping |
| **NumPy** | fast maths on arrays | log/exp, square roots, medians |
| **scikit-learn** | machine-learning toolbox | split, pipelines, imputer, scaler, encoder, 4 models, metrics |
| **Matplotlib** | draws charts | all figures |
| **Seaborn** | prettier statistical charts on top of Matplotlib | histograms, box plots, heatmap |
| **Streamlit** | turns a Python script into a web app | `app.py` |

## 4. The machine-learning workflow

1. **Problem**: predict a number (price) -> this is **regression**.
2. **Collect data**: the King County CSV.
3. **Validate and clean**: required columns present; remove duplicates, invalid prices/dates and impossible rows.
4. **Feature engineering**: new useful columns (house age ...).
5. **Train/test split**: 80% to learn from, 20% hidden. Done **first** so the test set stays truly unseen.
6. **Preprocessing pipeline** (fitted on training data only):
   * missing numbers -> median of the training data
   * numbers -> standardised (mean 0, spread 1) so Linear Regression is not biased by units
   * zipcode -> one-hot encoding (one 0/1 column per zipcode)
7. **Train** four models on the training set.
8. **Evaluate** on the test set with MAE, RMSE, R2.
9. **Compare** and choose the best (lowest test RMSE).
10. **Predict** new houses with the chosen model.

### Why the "log price" trick?
Prices are right-skewed: a few mansions cost millions. Learning `log(price)` makes the relationship more regular, which especially helps Linear Regression. After predicting, `exp` turns the answer back into dollars, and all metrics are measured in dollars.

### What is data leakage and how is it avoided?
Leakage means information from the test data sneaks into training, making results look better than they really are. We avoid it by: splitting first; learning medians/means/categories inside a Pipeline that only sees the training set; building features only from a house's own values; and never using price to create a feature. `src/sanity_checks.py` verifies the scaler's mean equals the *training* mean, not the full-dataset mean.

### What do the metrics mean?
* **MAE** (Mean Absolute Error): average size of the mistake in dollars. Easy to understand.
* **RMSE** (Root Mean Squared Error): like MAE but squares errors first, so large mistakes count more. Always at least as big as MAE.
* **R2** (coefficient of determination): share of price variation explained. 1 = perfect, 0 = no better than always predicting the average, negative = worse than that.
* **Train R2 vs test R2**: a big gap means over-fitting (memorising).

## 5. How each regression model works

**Linear Regression** - Finds one weight per feature and adds them up: `price = w1*area + w2*bathrooms + ... + b`. Like drawing the best straight line through the data. *Pros:* fast, easy to explain. *Cons:* cannot learn curves or interactions (for example "area matters more in an expensive zipcode").

**Decision Tree** - Plays "20 questions": "Is the grade above 8?" -> "Is the area above 2,500?" ... each final box predicts the average price of the training houses that landed there. *Pros:* easy to draw and read, handles curves. *Cons:* a deep tree memorises the training data (over-fits) and one small change in data can change the tree. We limit `max_depth` and `min_samples_leaf` to control this.

**Random Forest** - Grows many (200) decision trees, each on a random sample of houses and a random subset of features, and **averages** their predictions. Averaging many different opinions removes much of the noise and over-fitting. *Pros:* accurate and stable. *Cons:* slower and harder to read than one tree.

**Gradient Boosting** - Builds trees **one after another**. The first tree makes a rough guess; the second tree learns to fix the first tree's mistakes; the third fixes what is still wrong, and so on (300 small trees, each adding a small correction controlled by `learning_rate`). *Pros:* often the most accurate on table data. *Cons:* more settings to tune, slower to train, can over-fit if pushed too far.

## 6. How to read the dashboard results

* If a tree-based model has a much lower RMSE than Linear Regression, location and non-linear effects matter.
* If Decision Tree has a high train R2 but lower test R2, it is over-fitting.
* If all models disagree strongly for one house, treat that estimate with caution.
* The "typical error band" is **not** a guaranteed range: it shows the percentage within which half of the test houses were predicted.

## 7. Honest limitations (say these in the viva - it shows maturity)

* Data from one county and one year: prices are 2014-2015 dollars.
* No parking field in this dataset.
* Models were not heavily tuned and no cross-validation is used.
* Choosing the best model on the test set is slightly optimistic.
* Correlation and feature importance do not prove cause and effect.
