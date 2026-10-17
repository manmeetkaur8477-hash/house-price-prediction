# Troubleshooting Guide

Find the error message you see, then follow the fix.

| # | What you see | Why it happens | Fix |
|---|---|---|---|
| 1 | `'python' is not recognized as an internal or external command` | Python was installed without "Add to PATH" | Re-run the Python installer -> **Modify/Repair** -> tick **Add Python to environment variables**. Or reinstall and tick *Add python.exe to PATH*. Close and reopen VS Code. Try `py --version` too; if `py` works, use `py` in place of `python`. |
| 2 | `'pip' is not recognized` | Same as above | Use `python -m pip install -r requirements.txt` |
| 3 | `venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell security policy | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use `venv\Scripts\activate.bat` in Command Prompt (Terminal dropdown -> Command Prompt). |
| 4 | `ModuleNotFoundError: No module named 'streamlit'` (or pandas, sklearn, seaborn ...) | Libraries not installed, or the venv is not active | Check the prompt starts with `(venv)`. Then run `pip install -r requirements.txt`. In VS Code, `Ctrl+Shift+P` -> *Python: Select Interpreter* -> pick the `venv` one. |
| 5 | `ModuleNotFoundError: No module named 'src'` | You are in the wrong folder | The terminal must be inside the folder that contains `app.py`. Use `cd C:\Projects\house_price_prediction`. Run the checks as `python -m src.sanity_checks` (with `-m`), not `python src/sanity_checks.py`. |
| 6 | `'streamlit' is not recognized` | venv not active | Activate the venv, or run `python -m streamlit run app.py`. |
| 7 | `FileNotFoundError ... kc_house_data.csv` | Dataset missing/moved or wrong folder | Make sure `data\kc_house_data.csv` exists. Do not rename the `data` folder or the file. |
| 8 | `TypeError: ... got an unexpected keyword argument 'sparse_output'` | Old scikit-learn (before 1.2) | `pip install --upgrade scikit-learn` (the requirements file asks for 1.2 or newer). |
| 9 | Red box: "The CSV is missing these required columns: ..." | Uploaded CSV has different column names | Use the same columns as `data/kc_house_data.csv` (the sidebar has a *Download dataset format* button). |
| 10 | "Could not read the CSV file" / `UnicodeDecodeError` | File is not a normal comma-separated UTF-8 CSV | In Excel use *Save As -> CSV UTF-8*. Make sure it is a `.csv`, not `.xlsx`. |
| 11 | "Only N usable rows remain after cleaning" | Too small a file or many bad rows | Use at least 200 valid rows with real prices and dates. |
| 12 | Page stays on "Training 4 regression models ..." for a long time | First run on a slow computer (Random Forest and Gradient Boosting take the time) | Wait 1-3 minutes. Later runs are instant. To speed up: in `src/house_price_pipeline.py` lower `n_estimators` (200 -> 100, 300 -> 150). |
| 13 | `Port 8501 is already in use` | Another Streamlit is still running | Close the old terminal (or press `Ctrl+C`), or run `streamlit run app.py --server.port 8502`. |
| 14 | Browser does not open automatically | Normal on some systems | Copy the **Local URL** (`http://localhost:8501`) from the terminal into Chrome/Edge. |
| 15 | pip install fails with SSL / connection / timeout errors | Internet, proxy or college network restriction | Try mobile hotspot, then run `pip install -r requirements.txt` again. For a proxy: `pip install -r requirements.txt --proxy http://user:pass@host:port`. |
| 16 | pip tries to compile and fails (long red text about wheels / build tools) | Brand-new Python version without ready-made packages | Install **Python 3.11 or 3.12**, delete the `venv` folder, and repeat Steps 7-10. |
| 17 | Yellow squiggles "Import ... could not be resolved" in VS Code | VS Code uses another Python | `Ctrl+Shift+P` -> *Python: Select Interpreter* -> choose `.\venv\Scripts\python.exe`. The code still runs from the terminal. |
| 18 | Text looks plain / fonts differ | The dashboard loads Google Fonts from the internet | Works offline too, just with default fonts. |
| 19 | Charts or numbers look old after changing the CSV or settings | Cached results | Press `C` in the browser page -> *Clear cache*, or stop and restart Streamlit. |
| 20 | Prediction says "Basement area must be smaller..." or similar | Input validation doing its job | Fix the highlighted value. Yellow warnings only mean the value is outside the training data range. |
| 21 | Download button does nothing / downloads empty | Browser blocked download | Allow downloads for `localhost` in your browser. |
| 22 | `MemoryError` or computer freezes | Very large uploaded CSV | Use a smaller file, or lower `n_estimators` as in #12. |
| 23 | A Windows Firewall popup appears | Streamlit starts a local web server | Click **Allow access**. It is only on your own computer. |

## Still stuck?

1. Copy the **last 10 lines** of the red error message.
2. Note which step you were on and your Python version (`python --version`).
3. Run `python run_pipeline.py` - if it works but the dashboard does not, the problem is only in Streamlit or the browser; if it fails, the problem is in the installation or the data folder.
