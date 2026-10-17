"""
run_pipeline.py - train and compare the models WITHOUT the web dashboard.

Run it from the project folder:
    python run_pipeline.py

It prints the metrics table in the terminal and saves two files in /outputs:
    model_metrics.csv     - MAE, RMSE, R2 of every model
    test_predictions.csv  - actual vs predicted price for every test house
"""

import pandas as pd

from src import config
from src.data_utils import DataValidationError, prepare_dataset
from src.house_price_pipeline import build_predictions_table, train_and_compare


def main() -> None:
    print("Loading dataset:", config.DATA_PATH)
    raw = pd.read_csv(config.DATA_PATH)

    try:
        data, report = prepare_dataset(raw)
    except DataValidationError as error:
        print("Dataset problem:", error)
        return

    print(f"Rows: {report['rows_raw']} raw -> {report['rows_clean']} after cleaning")
    print("Training 4 models (this can take 20-60 seconds) ...")
    results = train_and_compare(data, config.DEFAULT_TEST_SIZE, config.RANDOM_SEED)

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    pd.set_option("display.width", 200)
    print("\nTest-set results (models never saw these houses while learning):\n")
    print(results.metrics)
    print(f"\nBest model (lowest test RMSE): {results.best_model}")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    results.metrics.to_csv(config.OUTPUT_DIR / "model_metrics.csv")
    build_predictions_table(results, results.best_model).to_csv(
        config.OUTPUT_DIR / "test_predictions.csv", index=False)
    print(f"\nSaved results in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
