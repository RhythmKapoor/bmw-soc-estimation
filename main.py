"""Paper-oriented reproduction of Sulaiman & Mustaffa (2024), BMW i3 SOC estimation.
Version 2: transparent metrics, fixed-seed final figures, and explicit CV diagnostics.

Uses the supplied 60-trip training and 10-trip test CSV files. It compares RF
models with 25/50/75/100 trees over five random seeds, runs 5-fold shuffled
K-fold CV on the training rows, evaluates an ELM baseline, and writes figures,
metrics, predictions, and a Table-2-style summary to ./outputs.

Important: the paper does not publish every implementation detail (e.g. exact
random seeds and all ELM settings). This is a transparent reproduction attempt,
not a claim of bit-for-bit identity with the authors' MATLAB results.
"""
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = DATA_DIR / "Final_60trips_Training_BMW_Data.csv"
TEST_FILE = DATA_DIR / "Final_10trips_Testing_BMW_Data.csv"
FEATURES = [
    "Battery Voltage [V]", "Battery Current [A]",
    "Battery Temperature [°C]", "Ambient Temperature [°C]",
]
TARGET = "SoC [%]"
TREE_COUNTS = [25, 50, 75, 100]
N_RUNS = 5
N_FOLDS = 5
BASE_SEED = 0
ELM_HIDDEN = 100

plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.alpha": 0.3})


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Put the CSV inside {DATA_DIR} with this exact filename: {path.name}"
        )
    # latin1 safely handles the degree symbol in the supplied CSV headers.
    frame = pd.read_csv(path, encoding="latin1")
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}\nFound: {frame.columns.tolist()}")
    frame = frame[required].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return frame


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAX_ERROR": float(np.max(np.abs(err))),
        "STD_DEV": float(np.std(err, ddof=0)),
    }


def make_rf(n_trees, seed):
    # These RF settings are the settings used in the supplied companion notebook.
    return RandomForestRegressor(
        n_estimators=int(n_trees), max_features=2, min_samples_leaf=5,
        n_jobs=-1, random_state=int(seed),
    )


class ELM:
    """Simple ELM baseline: scaled inputs, sigmoid hidden layer, least-squares output."""
    def __init__(self, n_hidden=100, seed=0):
        self.n_hidden = n_hidden
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        Xs = (X - self.mean_) / self.std_
        rng = np.random.default_rng(self.seed)
        self.W_ = rng.uniform(-1, 1, size=(Xs.shape[1], self.n_hidden))
        self.b_ = rng.uniform(-1, 1, size=self.n_hidden)
        H = self._sigmoid(Xs @ self.W_ + self.b_)
        self.beta_ = np.linalg.lstsq(H, y, rcond=None)[0]
        return self

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        return self._sigmoid(Xs @ self.W_ + self.b_) @ self.beta_


def run_cv(X, y, n_trees, seed):
    """Five-fold shuffled K-fold CV on training rows, as in the supplied notebook."""
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        model = make_rf(n_trees, seed + fold)
        start = time.perf_counter()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        elapsed = time.perf_counter() - start
        fold_rows.append({"Trees": n_trees, "Seed": seed, "Fold": fold,
                          **metrics(y[val_idx], pred), "Fit_Predict_Time_s": elapsed})
        print(f"  CV {n_trees:>3} trees | fold {fold}/{N_FOLDS} | RMSE={fold_rows[-1]['RMSE']:.4f}")
    return fold_rows


def plot_figure2(run_df):
    cols = ["RMSE", "MAE", "MAX_ERROR", "STD_DEV"]
    colors = ["#1f4e9c", "#d95f2b", "#8c8c8c", "#e8a825", "#2f6fb5"]
    fig, ax = plt.subplots(figsize=(12, 6))
    width, x, ticks, labels, centers = 0.14, 0.0, [], [], []
    for trees in TREE_COUNTS:
        subset = run_df[run_df["Trees"] == trees].sort_values("Run")
        start = x
        for col in cols:
            for ri, val in enumerate(subset[col].to_numpy()):
                ax.bar(x + ri * width, val, width, color=colors[ri],
                       label=f"Run {ri + 1}" if trees == TREE_COUNTS[0] and col == cols[0] else None)
            ticks.append(x + width * (len(subset) - 1) / 2)
            labels.append(col.replace("_", " "))
            x += width * len(subset) + 0.12
        centers.append((start + x - 0.12) / 2)
        x += 0.8
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    for center, trees in zip(centers, TREE_COUNTS):
        ax.text(center, -0.28, f"{trees} trees", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10)
    ymax = max(30, int(np.ceil(run_df[cols].to_numpy().max() / 5) * 5))
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Error (SOC percentage points)")
    ax.set_title("Figure 2. RF performance metrics for 25, 50, 75 and 100 trees (five runs)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.38), frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure2_rf_configuration_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def set_instance_axis(ax, n):
    ax.set_xlim(0, n - 1)
    ticks = np.linspace(0, n - 1, 7).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:,}" for v in ticks])


def plot_tracking_figure(actual, pred, title, filename):
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    err = pred - actual
    idx = np.arange(len(actual))
    peak = int(np.argmax(np.abs(err)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(idx, actual, color="#1f4e9c", lw=1.5, label="Observed SOC")
    ax1.plot(idx, pred, color="#b03a2e", lw=0.9, label="Predicted SOC")
    ax1.set_ylabel("SOC (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title(title)
    ax1.legend(loc="best")
    ax2.plot(idx, err, color="#1f4e9c", lw=0.8, label="Predicted − actual")
    ax2.axhline(0, color="black", lw=0.7)
    ax2.plot(peak, err[peak], "ko", ms=4)
    ax2.annotate(f"Max |error|={abs(err[peak]):.2f}\ninstance {peak + 1:,}",
                 (peak, err[peak]), xytext=(12, 12), textcoords="offset points",
                 arrowprops={"arrowstyle": "->"}, fontsize=8)
    ax2.set_ylabel("Error (percentage points)")
    ax2.set_xlabel("Test observation index")
    ax2.legend(loc="best")
    set_instance_axis(ax2, len(actual))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading the supplied BMW i3 datasets...")
    train = load_data(TRAIN_FILE)
    test = load_data(TEST_FILE)
    Xtr, ytr = train[FEATURES].to_numpy(), train[TARGET].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(), test[TARGET].to_numpy()
    print(f"Training instances: {len(train):,}; testing instances: {len(test):,}")
    print(f"Missing values after cleaning — train: {int(train.isna().sum().sum())}; test: {int(test.isna().sum().sum())}")
    if len(train) != 945027 or len(test) != 118974:
        print("NOTE: row counts differ from Table 1 of the paper; confirm the exact source CSV/version.")

    run_rows, cv_rows, predictions = [], [], {}
    # Five runs for every tree count. CV is calculated once per tree count (seed 0),
    # and every run's held-out test metrics are retained, including best and average.
    for trees in TREE_COUNTS:
        cv_seed = BASE_SEED
        print(f"\n5-fold CV for {trees} trees (training data only)...")
        cv_rows.extend(run_cv(Xtr, ytr, trees, cv_seed))
        for run in range(N_RUNS):
            seed = BASE_SEED + run
            print(f"Test run {run + 1}/{N_RUNS}: RF {trees} trees, seed={seed}")
            model = make_rf(trees, seed)
            start = time.perf_counter()
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            elapsed = time.perf_counter() - start
            row = {**metrics(yte, pred), "Model": f"RF ({trees} trees)",
                   "Trees": trees, "Run": run + 1, "Seed": seed,
                   "CV_RMSE_mean": float(np.mean([r["RMSE"] for r in cv_rows if r["Trees"] == trees])),
                   "Computation_Time_s": elapsed}
            run_rows.append(row)
            predictions[(f"RF ({trees} trees)", run + 1)] = pred
            print(f"  RMSE={row['RMSE']:.4f} | MAE={row['MAE']:.4f} | MAX={row['MAX_ERROR']:.4f} | time={elapsed:.2f}s")

    # Five ELM baseline runs.
    elm_rows = []
    for run in range(N_RUNS):
        seed = BASE_SEED + run
        print(f"\nELM test run {run + 1}/{N_RUNS}, seed={seed}")
        model = ELM(ELM_HIDDEN, seed=seed)
        start = time.perf_counter()
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        elapsed = time.perf_counter() - start
        row = {**metrics(yte, pred), "Model": "ELM", "Trees": np.nan,
               "Run": run + 1, "Seed": seed, "CV_RMSE_mean": np.nan,
               "Computation_Time_s": elapsed}
        elm_rows.append(row)
        predictions[("ELM", run + 1)] = pred
        print(f"  RMSE={row['RMSE']:.4f} | MAE={row['MAE']:.4f} | MAX={row['MAX_ERROR']:.4f} | time={elapsed:.2f}s")

    run_df = pd.DataFrame(run_rows)
    elm_df = pd.DataFrame(elm_rows)
    all_runs = pd.concat([run_df, elm_df], ignore_index=True)
    cv_df = pd.DataFrame(cv_rows)
    all_runs.to_csv(OUTPUT_DIR / "all_test_run_metrics.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "five_fold_cv_metrics.csv", index=False)
    plot_figure2(run_df)

    # Table 2 style: best observed run and average over five runs for each model.
    model_order = [f"RF ({n} trees)" for n in TREE_COUNTS] + ["ELM"]
    summary_rows = []
    for name in model_order:
        subset = all_runs[all_runs["Model"] == name]
        for statistic, row in [("Best run (lowest test RMSE)", subset.loc[subset["RMSE"].idxmin()]),
                                ("Average over five runs", None)]:
            if row is not None:
                summary_rows.append({"Model": name, "Summary": statistic,
                                     **{c: row[c] for c in ["RMSE", "MAE", "MAX_ERROR", "STD_DEV", "Computation_Time_s", "Run"]}})
            else:
                summary_rows.append({"Model": name, "Summary": statistic,
                                     **{c: subset[c].mean() for c in ["RMSE", "MAE", "MAX_ERROR", "STD_DEV", "Computation_Time_s"]},
                                     "Run": np.nan})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "table2_paper_style_summary.csv", index=False)
    print("\nTable-2-style summary (test results across five runs):")
    print(summary_df.round(4).to_string(index=False))

    # For tracking figures, use the predeclared seed-0 run rather than choosing
    # whichever test-set run has the best score. This avoids test-set selection bias.
    # The paper-style summary still reports both best-of-five and five-run average.
    rf25_run = 1  # BASE_SEED + run 0; run numbering starts at 1
    best_rf_run = rf25_run
    best_rf_pred = predictions[("RF (25 trees)", rf25_run)]
    elm_run = 1
    best_elm_run = elm_run
    best_elm_pred = predictions[("ELM", elm_run)]
    plot_tracking_figure(yte, best_rf_pred,
                         f"Figure 3. RF SOC estimation on test data (25 trees; predeclared seed-0 run, run {best_rf_run})",
                         "figure3_random_forest_results.png")
    plot_tracking_figure(yte, best_elm_pred,
                         f"Figure 4. ELM SOC estimation on test data (predeclared seed-0 run, run {best_elm_run})",
                         "figure4_elm_results.png")

    idx = np.arange(len(yte))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(idx, yte, color="#1f4e9c", lw=1.8, label="Actual SOC")
    ax.plot(idx, best_rf_pred, color="#d95f02", lw=0.8, label="Predicted by RF (25 trees)")
    ax.plot(idx, best_elm_pred, color="#2e8b57", lw=0.8, label="Predicted by ELM")
    set_instance_axis(ax, len(yte)); ax.set_ylim(0, 100)
    ax.set_xlabel("Test observation index"); ax.set_ylabel("SOC (%)")
    ax.set_title("Figure 5. Comparison of RF and ELM SOC estimation")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure5_rf_vs_elm_soc.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(idx, best_rf_pred - yte, color="#1f4e9c", lw=0.9, label="Error by RF (25 trees)")
    ax.plot(idx, best_elm_pred - yte, color="#d95f02", lw=0.9, label="Error by ELM")
    ax.axhline(0, color="black", lw=0.7); set_instance_axis(ax, len(yte))
    ax.set_xlabel("Test observation index"); ax.set_ylabel("Predicted − actual SOC (percentage points)")
    ax.set_title("Figure 6. Error comparison between RF and ELM")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure6_error_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    pred_df = pd.DataFrame({"Test observation": np.arange(1, len(yte) + 1), "Actual SOC (%)": yte,
                            "RF 25-tree prediction (%)": best_rf_pred,
                            "RF error (predicted - actual)": best_rf_pred - yte,
                            "ELM prediction (%)": best_elm_pred,
                            "ELM error (predicted - actual)": best_elm_pred - yte})
    pred_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print("\nFinished. Outputs saved in:", OUTPUT_DIR)
    print("Paper's reported best RF (25 trees): RMSE 5.9028, MAE 4.4321, MAX 24.2175, STD_DEV 5.8999.")
    print("Compare those reported values with measured values in table2_paper_style_summary.csv. The best-of-five row is descriptive, not an unbiased final estimate.")
    print("Note: exact equality is not guaranteed because the paper does not publish all implementation details/seeds.")


if __name__ == "__main__":
    main()
