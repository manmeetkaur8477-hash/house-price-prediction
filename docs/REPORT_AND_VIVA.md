# College Report Outline, Presentation Points and Viva Questions

## A. Before you reuse your existing report (IMPORTANT)

Your existing report file follows a 14-section format and was written for the **California Housing** dataset, with **nine models**, 5-fold cross-validation, hyper-parameter tuning, MAPE and distance features. This project uses the **King County** dataset you supplied and compares the **four** required models. If you copy sections across, change these so the report matches the code you will demonstrate:

| In the old report | Change to |
|---|---|
| Dataset: California Housing (1990 Census), 20,640 rows, 10 columns | King County house sales, 21,613 rows, 21 columns (about 21,612 after cleaning) |
| Target `median_house_value` | `price` (US dollars of 2014-2015) |
| Nine models (Ridge, Lasso, KNN, HistGB ...) | Four models: Linear, Decision Tree, Random Forest, Gradient Boosting |
| 5-fold CV, tuning, MAPE | Say these are **future scope**, or add them to the code before the viva |
| Features `ocean_proximity`, distance-to-city | `zipcode`, `lat`, `long`, `house_age`, `was_renovated`, `has_basement`, `living_to_lot_ratio` |
| Appendix A `src/house_price_pipeline.py`, B `app.py`, C `src/sanity_checks.py` | Same three file names exist in this project |
| Live app / GitHub links | Replace with your own links, or remove |

Never copy numbers from the old report. Run the project and use **your own** results.

## B. Report outline (same 14 sections as your format)

**Cover page**: Punjab College of Technical Engineering; B.Tech 3rd Year - AIML; Python Project Report; title *House Price Prediction: Predicting House Prices from Location, Area and Property Features with a Comparative Study of Regression Models*; members with admission roll no. and roll no. (Manmeet Kaur 2024AIML0593 / 2445863; Avneet Kaur 2024AIML1417 / 2445835); Submitted to Ms. Ishatpreet Kaur; Tools; Dataset.

**Declaration and Individual Contribution**: declaration of own work; table of work areas (problem definition, data cleaning and EDA, feature engineering, linear model, tree models, evaluation, Streamlit app, testing, report writing) with Lead/Support for each member - fill this in honestly with what each of you really did.

**Table of Contents**

1. **Introduction** - why house prices matter; manual appraisal vs machine learning; what the project does.
2. **Problem Statement** - predict house sale price from location, area and property features; compare regression models to find the most accurate.
3. **Objectives** - clean and explore the data; engineer features; compare four models under identical conditions with MAE, RMSE, R2; avoid leakage; interpret with feature importance and residuals; build a Streamlit app.
4. **Literature Review and Existing System** - hedonic pricing (Rosen, 1974; Harrison and Rubinfeld, 1978); automated valuation models; Linear regression; decision trees (Breiman et al.); Random Forest (Breiman, 2001); Gradient Boosting (Friedman, 2001). Gap: linear models miss non-linear and location effects, so we compare model families on the same data.
5. **Dataset Description and Source** - King County (Seattle area), USA, May 2014 - May 2015, 21,613 rows; table of the 21 columns (see `PROJECT_EXPLAINED.md`); descriptive statistics table (copy from the Data Explorer tab); data-quality notes (33-bedroom outlier, `yr_renovated = 0`, right-skewed price). Verify the dataset citation (publicly shared as *House Sales in King County, USA*).
6. **Proposed Methodology** - the 10-step workflow (see `PROJECT_EXPLAINED.md` section 4) with a flow diagram.
7. **System Design and Architecture** - block diagram: CSV -> validation -> cleaning -> features -> split -> pipeline -> 4 models -> evaluation -> Streamlit dashboard; module diagram of the `src/` files.
8. **Python Concepts, Libraries and Frameworks Used** - functions, dataclasses, type hints, exceptions, pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer, models, metrics), Matplotlib, Seaborn, Streamlit, caching.
9. **Implementation and Source Code** - module by module explanation; show key snippets (pipeline, `train_and_compare`).
10. **Testing and Validation** - output of `python -m src.sanity_checks` (all PASS), input validation tests (basement larger than living area, year renovated before year built), unseen zipcode, missing values, reproducibility with the same seed.
11. **Results, Graphs, Evaluation Metrics and Output Screenshots** - metrics table, model-comparison chart, actual vs predicted, residuals, feature importance, EDA charts, screenshots of all five tabs and the prediction card. Discuss over-fitting using train vs test R2.
12. **Limitations and Future Scope** - one county/year; no parking column; no cross-validation or tuning; test-set model selection; future: Ridge/Lasso/XGBoost, cross-validation, grid search, more recent data, map visualisation, deployment on Streamlit Community Cloud.
13. **Conclusion** - state which model was best **using your own measured numbers**, and what you learned.
14. **References** - scikit-learn, pandas, Streamlit documentation; Rosen (1974); Harrison and Rubinfeld (1978); Breiman (2001); Friedman (2001); the dataset source.

**Appendices**: A - `src/house_price_pipeline.py`; B - `app.py`; C - `src/sanity_checks.py`.

## C. Presentation points (about 10 slides)

1. **Title** - project name, members, guide.
2. **Problem and motivation** - why price estimation matters; manual valuation is slow.
3. **Dataset** - King County, 21,613 houses, 21 columns; show 5 rows; real data from 2014-2015.
4. **Data cleaning and EDA** - price is skewed; strongest relationships (area, grade, location); map of prices.
5. **Features and preprocessing** - house age, renovation, basement; imputation, scaling, one-hot zipcode.
6. **Avoiding data leakage** - split first, pipeline learns only from training data.
7. **Four models explained** in one line each (straight line / questions / many trees / trees fixing mistakes).
8. **Results** - metrics table and comparison chart (your live numbers); best model; over-fitting check.
9. **Live demo** - open the dashboard, predict one house, compare all four models, download the CSV.
10. **Limitations, future scope, conclusion.**

**Demo tip:** start `streamlit run app.py` *before* your presentation so the models are already trained and cached.

## D. Likely viva questions and easy answers

**Q1. What is the aim of your project?**
To predict the price of a house from its location, area and features, and to find out which of four regression models predicts best.

**Q2. Why is this a regression problem, not classification?**
The answer is a continuous number (price in dollars), not a category.

**Q3. Which dataset did you use? Is it real?**
The King County (Seattle area, USA) house-sales dataset: 21,613 real sales from May 2014 to May 2015. So our predictions describe that place and time, not today's market.

**Q4. What cleaning did you do?**
Checked the required columns, removed duplicate rows, invalid prices and dates and impossible rows (for example a house with 33 bedrooms but 1,620 sq ft). Remaining missing values are filled inside the pipeline.

**Q5. How do you handle missing values?**
A `SimpleImputer` fills numbers with the median and zipcode with the most frequent value. It is fitted on the training data only.

**Q6. Why median and not mean?**
The median is not distorted by very expensive or very large houses.

**Q7. How did you encode categorical data?**
Zipcode is a label, not a quantity, so it is one-hot encoded: one 0/1 column for each zipcode. `handle_unknown="ignore"` stops a new zipcode from crashing the app.

**Q8. Why scale the features?**
So that features measured in different units (square feet vs number of floors) are comparable. It matters for Linear Regression; trees do not need it but are not harmed.

**Q9. What is data leakage? How did you avoid it?**
Leakage is when test information influences training and makes the score unrealistically good. We split before anything is learned and keep all learning steps inside a scikit-learn Pipeline fitted on training data. `sanity_checks.py` proves it.

**Q10. Why train-test split? Why 80/20?**
To measure performance on houses the model has never seen. 80/20 is a common default that leaves enough data for both training and testing; it can be changed in the sidebar.

**Q11. What is `random_state` / the seed?**
It fixes the random choices (which houses go to the test set, how trees are grown) so that anyone gets the same results.

**Q12. Why did you predict log(price)?**
Prices are right-skewed. On a log scale the relationship is more regular, which helps especially Linear Regression. Predictions are converted back with `exp`, and metrics are in dollars.

**Q13. Explain MAE, RMSE and R2.**
MAE is the average mistake in dollars. RMSE squares the mistakes first, so large errors weigh more. R2 is the share of price variation the model explains (1 is perfect).

**Q14. Why is RMSE bigger than MAE?**
Squaring gives big errors extra weight; RMSE is always greater than or equal to MAE, and the gap shows how many large errors exist.

**Q15. How does Linear Regression work and what is its weakness?**
It learns a weight for every feature and adds them up (a straight-line relationship). It cannot capture curves or interactions such as location-dependent effects.

**Q16. How does a Decision Tree work?**
It asks yes/no questions on features to split houses into groups and predicts each group's average price. It is easy to read but over-fits if allowed to grow too deep.

**Q17. Why is Random Forest usually better than one tree?**
It averages many trees trained on random parts of the data, which reduces variance and over-fitting.

**Q18. How is Gradient Boosting different from Random Forest?**
Random Forest grows trees independently and averages them. Gradient Boosting grows trees one after another, each correcting the previous trees' mistakes.

**Q19. What is over-fitting? How did you check?**
Memorising training data and failing on new data. We compare train R2 with test R2; a large gap signals over-fitting. We also limit tree depth and use averaging.

**Q20. How did you choose the best model?**
The one with the lowest RMSE on the test set (ties broken by higher R2). The sidebar also lets you choose manually. A stricter method would use cross-validation, which we list as future work.

**Q21. What is a Pipeline and why use it?**
A chain of steps (impute -> scale/encode -> model) that runs as one object. It prevents leakage and guarantees training and prediction use identical preprocessing.

**Q22. What does feature importance mean?**
For tree models, how much each feature was used to split the data; larger means more influential. It shows what the model relies on, not what causes price changes.

**Q23. Why does the form not have a parking field?**
The dataset has no parking column, and a model cannot use information it was not trained on. Adding parking needs a dataset that contains it.

**Q24. How accurate is your model in real life?**
We can only say how accurate it is on the unseen test houses of this dataset (see the metrics table). For another city or a later year the accuracy would be different and probably worse.

**Q25. What is Streamlit? What does caching do here?**
Streamlit turns Python code into a web app. Caching (`st.cache_resource`) stores the trained models and charts so the app does not retrain on every click.

**Q26. How do you validate user input?**
The form limits ranges, and `validate_inputs` blocks impossible combinations (basement larger than the house, renovation before construction) and warns when values are outside the range seen in training.

**Q27. What would you improve with more time?**
Cross-validation and hyper-parameter search, more models (Ridge, Lasso, XGBoost), newer data, maps, and deployment online.

**Q28. Can the app handle a house in a zipcode it has never seen?**
The pipeline will not crash (unknown zipcodes become all-zero), but the estimate then relies only on the other features, so it is less reliable.
