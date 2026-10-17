"""
config.py - ONE place for all the settings of the project.

If you want to change something (the random seed, the list of features, the
colours of the charts ...) this is the first file to open.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent      # the project folder
DATA_PATH = BASE_DIR / "data" / "kc_house_data.csv"    # the dataset you provided
OUTPUT_DIR = BASE_DIR / "outputs"                      # results saved by run_pipeline.py

# ----------------------------------------------------------------------------
# Reproducibility and splitting
# ----------------------------------------------------------------------------
RANDOM_SEED = 42            # same seed -> same split -> same results every run
DEFAULT_TEST_SIZE = 0.20    # 20 % of the houses are hidden for testing
MIN_ROWS = 200              # smaller uploaded files are rejected

# ----------------------------------------------------------------------------
# Dataset columns (King County, USA house sales dataset)
# ----------------------------------------------------------------------------
TARGET = "price"            # the value we want to predict (US dollars)

# Columns that an uploaded CSV must contain (the "id" column is optional).
REQUIRED_COLUMNS = [
    "date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long",
    "sqft_living15", "sqft_lot15",
]

# Raw columns that must be numbers (everything except date and zipcode).
NUMERIC_RAW_COLUMNS = [c for c in REQUIRED_COLUMNS if c not in ("date", "zipcode")]

# Houses with more bedrooms than this are treated as data-entry mistakes
# (the dataset contains a house with 33 bedrooms and only 1,620 sq ft).
MAX_BEDROOMS = 15

# ----------------------------------------------------------------------------
# Features given to the models
# ----------------------------------------------------------------------------
# "Engineered" features are created in data_utils.engineer_features():
#   house_age, was_renovated, has_basement, living_to_lot_ratio
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement",
    "lat", "long", "sqft_living15", "sqft_lot15",
    "house_age", "was_renovated", "has_basement", "living_to_lot_ratio",
]
CATEGORICAL_FEATURES = ["zipcode"]          # location as a category
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Predict log(price) instead of price? Prices are right-skewed (a few very
# expensive houses). Learning on the log scale usually helps Linear Regression.
# All metrics are still reported in real US dollars.
USE_LOG_TARGET = True

# ----------------------------------------------------------------------------
# Model names (used as dictionary keys everywhere)
# ----------------------------------------------------------------------------
LINEAR = "Linear Regression"
TREE = "Decision Tree"
FOREST = "Random Forest"
BOOST = "Gradient Boosting"
MODEL_NAMES = [LINEAR, TREE, FOREST, BOOST]

# ----------------------------------------------------------------------------
# Colour palette (modern real-estate theme)
# ----------------------------------------------------------------------------
NAVY = "#0B2545"
TEAL = "#13A89E"
TEAL_DARK = "#0E7C86"
GOLD = "#E9A23B"
CORAL = "#E4572E"
SKY = "#8DA9C4"
INK = "#1B2A41"
MUTED = "#5C6B7A"
BACKGROUND = "#F4F7FA"
CARD = "#FFFFFF"

MODEL_COLORS = {LINEAR: SKY, TREE: GOLD, FOREST: TEAL, BOOST: NAVY}
