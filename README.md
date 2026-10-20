# BMW i3 Battery State-of-Charge (SOC) Research Reproduction

This beginner-friendly Python project recreates the main result plots from:

**Sulaiman, M. H. & Mustaffa, Z. (2024). _State of charge estimation for electric vehicles using random forest._ Green Energy and Intelligent Transportation, 3, 100177.**

The code uses the provided 60-trip training CSV and 10-trip testing CSV.

## Figures and outputs

Running `main.py` creates the following inside `outputs/`:

- `figure2_rf_configuration_comparison.png` — Random Forest metrics for 25, 50, 75 and 100 trees across five repeated runs.
- `figure3_random_forest_results.png` — observed vs predicted SOC and RF error.
- `figure4_elm_results.png` — observed vs predicted SOC and ELM error.
- `figure5_rf_vs_elm_soc.png` — actual SOC vs RF and ELM predictions.
- `figure6_error_comparison.png` — RF and ELM prediction errors.
- CSV files with per-run metrics, averaged metrics, model metrics and predictions.

## Important reproducibility note

These are research-style reproductions using the supplied CSV files, not guaranteed bit-for-bit reproductions of the paper's original plots. The paper reports a five-fold evaluation and five repeated runs, but the exact random seeds, all preprocessing choices and complete ELM implementation are not fully specified in the paper. In this project, Figure 2 uses five repeated random train/validation splits of a 30,000-row training sample to keep the experiment practical on a student laptop. The final 25-tree Random Forest is fitted on the full training file. The ELM is an explicitly implemented random-hidden-layer / ridge-output baseline, fitted on a reproducible 60,000-row sample.

Metric units are SOC percentage points. The error is defined as `actual SOC - predicted SOC`. Small differences from the paper are expected.

## 1. Install prerequisites

Install Python 3.10 or newer and Git. Install Visual Studio Code and the Microsoft Python extension.

## 2. Create a virtual environment

Open this project folder in VS Code. Choose **Terminal > New Terminal**, then run:

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then run the installation commands again.

### Windows Command Prompt

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add the dataset files

Copy both CSV files into the `data/` folder, and make sure their names are exactly:

- `Final_60trips_Training_BMW_Data.csv`
- `Final_10trips_Testing_BMW_Data.csv`

The uploaded originals may include `(1)` in the filenames. Rename the copies as shown above.

## 4. Run the program

In the VS Code terminal, make sure the virtual environment is active, then run:

```bash
python main.py
```

The script may take several minutes because the training CSV contains nearly one million rows. CPU and RAM matter more than GPU for this Random Forest implementation. Do not stop the terminal while it is running.

## 5. View results

Open the `outputs/` folder in VS Code. Double-click a PNG to view it. Open the CSV files in VS Code or Excel to inspect numeric results.

## 6. Publish to GitHub

Create a new, empty repository on GitHub. Do not initialize it with a README if you plan to push this local project as-is.

In the VS Code terminal:

```bash
git init
git add main.py requirements.txt README.md .gitignore
git add data/*.csv
git add outputs/.gitkeep
git commit -m "Add BMW SOC estimation and research plots"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the URL with your own repository URL. If Git asks you to authenticate, follow the browser sign-in prompt.

For a cleaner repository, you can choose not to commit the large CSV files and instead document how to obtain them. Confirm your course's data-sharing rules before publishing datasets.

## Method overview

1. Load and validate the two CSV files.
2. Use battery voltage, current, battery temperature and ambient temperature as input features.
3. Use manufacturer SOC (`SoC [%]`) as the target.
4. Compare RF tree counts of 25, 50, 75 and 100.
5. Fit a final 25-tree Random Forest on all training rows.
6. Fit an ELM baseline on a reproducible subset.
7. Predict SOC on the separate test file.
8. Calculate RMSE, MAE, maximum absolute error and error standard deviation.
9. Export figures and CSVs.

## Metric definitions

- **RMSE:** square root of mean squared prediction error.
- **MAE:** mean absolute prediction error.
- **MAX_ERROR:** maximum absolute prediction error.
- **STD_DEV:** standard deviation of the signed error.

The source paper's table may label maximum error and standard deviation differently. Here, both are calculated directly from the test residuals; the code documents the definitions explicitly.
