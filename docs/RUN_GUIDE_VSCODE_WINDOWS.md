# Run the Project in VS Code on Windows - Step by Step

No experience needed. Follow the steps in order. Do not skip any.

## Part A - One-time setup of your computer

### Step 1 - Install Python
1. Open https://www.python.org/downloads/ and download **Python 3.11 or 3.12** (3.10 and newer work).
2. Run the installer. **VERY IMPORTANT: tick the box "Add python.exe to PATH"** at the bottom of the first screen.
3. Click **Install Now**, then **Close**.
4. Check it: press the Windows key, type `cmd`, press Enter, then type:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`. (If you see "not recognized", see TROUBLESHOOTING.md, problem 1.)

### Step 2 - Install VS Code
1. Download from https://code.visualstudio.com/ and install with the default options.

### Step 3 - Install the Python extension in VS Code
1. Open VS Code.
2. Click the **Extensions** icon on the left (four squares) or press `Ctrl+Shift+X`.
3. Search **Python**, choose the one by **Microsoft**, click **Install**.

## Part B - Open the project

### Step 4 - Put the project folder somewhere simple
Copy the folder `house_price_prediction` to a short path with no special characters, for example:
```
C:\Projects\house_price_prediction
```
(If you received a .zip file, right-click it -> **Extract All** first.)

### Step 5 - Open the folder in VS Code
1. In VS Code: **File -> Open Folder...**
2. Select `house_price_prediction` and click **Select Folder**.
3. If VS Code asks "Do you trust the authors?", click **Yes, I trust the authors**.
4. You should see `app.py`, `src`, `data` ... in the left panel. **You must open the folder that contains `app.py`**, not its parent.

### Step 6 - Open the terminal inside VS Code
Press `` Ctrl+` `` (the key above Tab), or use **Terminal -> New Terminal**. The prompt should end with `house_price_prediction>`.

## Part C - Install the libraries (once)

### Step 7 - Create a virtual environment
A virtual environment is a private box of libraries for this project only.
```
python -m venv venv
```
Wait until the prompt returns (10-30 seconds). A new folder `venv` appears.

### Step 8 - Activate it
If your terminal is **PowerShell** (the default) type:
```
venv\Scripts\Activate.ps1
```
If you see a red error about "running scripts is disabled", run this once and try again:
```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
(Type `Y` if asked.) **Or** simply use the Command Prompt style instead:
```
venv\Scripts\activate.bat
```
When it works, the line starts with `(venv)`.

### Step 9 - Tell VS Code to use this environment
1. Press `Ctrl+Shift+P`, type **Python: Select Interpreter**, press Enter.
2. Choose the one that shows `.\venv\Scripts\python.exe`.

### Step 10 - Install everything
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This downloads pandas, NumPy, scikit-learn, Matplotlib, Seaborn and Streamlit (a few minutes, needs internet). Wait for "Successfully installed ...".

## Part D - Run it

### Step 11 - (Recommended) Test the machine-learning part first
```
python run_pipeline.py
```
After 30-60 seconds you will see a table with MAE, RMSE and R2 for the four models and the name of the best model. If this works, everything is installed correctly.

### Step 12 - (Optional) Run the automatic checks
```
python -m src.sanity_checks
```
You should see many lines starting with `[PASS]` and finally `All checks passed.`

### Step 13 - Start the dashboard
```
streamlit run app.py
```
* Your browser opens at **http://localhost:8501**. If it does not, open it yourself.
* If Streamlit asks for your email the very first time, just press **Enter** to skip.
* The first load shows "Training 4 regression models ..." for 30-60 seconds. This is normal.
* If Windows Firewall asks for permission, choose **Allow access**.

### Step 14 - Use the dashboard
1. **Overview** - dataset facts, cleaning report, how the project works.
2. **Data Explorer** - graphs about prices, locations and correlations.
3. **Model Comparison** - metrics table, comparison chart, actual-vs-predicted, feature importance.
4. **Price Predictor** - choose a zipcode, fill in the form, press **Predict price**. Download the result as CSV.
5. **Sample Predictions** - real test houses, actual vs predicted, and the downloadable results file.
6. Sidebar: upload your own CSV, change the test size or random seed, and choose which model predicts.

### Step 15 - Stop and restart
* To stop: click in the terminal and press `Ctrl+C`.
* Next time, you only need: open the folder -> open the terminal -> `venv\Scripts\activate` -> `streamlit run app.py`.

## Part E - Making small changes (safe ideas)

| I want to... | Open this file | Change this |
|---|---|---|
| change the colours | `src/config.py` | `NAVY`, `TEAL`, `GOLD` ... |
| change the random seed | `src/config.py` | `RANDOM_SEED = 42` |
| change the test share | `src/config.py` | `DEFAULT_TEST_SIZE = 0.20` |
| turn the log-price trick off | `src/config.py` | `USE_LOG_TARGET = False` |
| make the forest bigger | `src/house_price_pipeline.py` | `n_estimators=200` in `build_models` |
| change the title | `app.py` | the text inside `hero(...)` |

After editing, save with `Ctrl+S`; Streamlit shows "Source file changed" - click **Rerun** (or **Always rerun**).
