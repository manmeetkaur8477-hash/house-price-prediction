"""
visuals.py - all charts (Matplotlib + Seaborn) in one file.

Every function RETURNS a figure; app.py decides where to show it.
All charts share the same colours (taken from config.py) so the dashboard
looks consistent.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from src import config

PRICE_CMAP = LinearSegmentedColormap.from_list(
    "price", [config.SKY, config.TEAL, config.GOLD, config.CORAL])


def apply_theme() -> None:
    """Set one clean look for every chart."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D5DDE5",
        "axes.labelcolor": config.INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlecolor": config.NAVY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#E8EDF2",
        "text.color": config.INK,
        "xtick.color": config.MUTED,
        "ytick.color": config.MUTED,
        "font.size": 10,
    })


def _money(value, _pos=None) -> str:
    """Axis label formatter: 1250000 -> $1.2M, 450000 -> $450K."""
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    return f"${value / 1e3:.0f}K"


MONEY_FORMAT = FuncFormatter(_money)


# ----------------------------------------------------------------------------
# Exploratory data analysis charts
# ----------------------------------------------------------------------------
def price_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.histplot(df[config.TARGET], bins=60, color=config.TEAL, ax=axes[0])
    axes[0].xaxis.set_major_formatter(MONEY_FORMAT)
    axes[0].set_title("Price distribution (skewed to the right)")
    axes[0].set_xlabel("Sale price")

    sns.histplot(np.log10(df[config.TARGET]), bins=60, color=config.NAVY, ax=axes[1])
    axes[1].set_title("Same prices on a log10 scale")
    axes[1].set_xlabel("log10(price)")
    fig.tight_layout()
    return fig


def price_vs_area(df: pd.DataFrame, max_points: int = 5000):
    sample = df.sample(min(max_points, len(df)), random_state=config.RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(sample["sqft_living"], sample[config.TARGET], s=10, alpha=0.35,
               color=config.TEAL_DARK, edgecolor="none")
    ax.yaxis.set_major_formatter(MONEY_FORMAT)
    ax.set_xlabel("Living area (sq ft)")
    ax.set_ylabel("Sale price")
    ax.set_title("Bigger houses cost more")
    fig.tight_layout()
    return fig


def price_by_grade(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    sns.boxplot(data=df, x="grade", y=config.TARGET, color=config.SKY,
                showfliers=False, linecolor=config.NAVY, ax=ax)
    ax.yaxis.set_major_formatter(MONEY_FORMAT)
    ax.set_xlabel("Construction grade (1-13)")
    ax.set_ylabel("Sale price")
    ax.set_title("Price by construction grade")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame):
    columns = [config.TARGET, "sqft_living", "grade", "sqft_above", "sqft_living15",
               "bathrooms", "bedrooms", "view", "sqft_basement", "lat", "floors",
               "waterfront", "condition", "house_age", "sqft_lot"]
    columns = [c for c in columns if c in df.columns]
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size": 7}, cmap="GnBu",
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation between features and price")
    fig.tight_layout()
    return fig


def location_map(df: pd.DataFrame, max_points: int = 8000):
    sample = df.sample(min(max_points, len(df)), random_state=config.RANDOM_SEED)
    low, high = sample[config.TARGET].quantile([0.05, 0.95])
    fig, ax = plt.subplots(figsize=(6, 4.8))
    points = ax.scatter(sample["long"], sample["lat"], c=sample[config.TARGET],
                        cmap=PRICE_CMAP, vmin=low, vmax=high, s=7, alpha=0.7,
                        edgecolor="none")
    bar = fig.colorbar(points, ax=ax, shrink=0.85)
    bar.ax.yaxis.set_major_formatter(MONEY_FORMAT)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Where the expensive houses are")
    fig.tight_layout()
    return fig


def top_zipcodes(df: pd.DataFrame, top_n: int = 12):
    medians = (df.groupby("zipcode")[config.TARGET].median()
               .sort_values(ascending=False).head(top_n).iloc[::-1])
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.barh(medians.index, medians.values, color=config.NAVY)
    ax.xaxis.set_major_formatter(MONEY_FORMAT)
    ax.set_xlabel("Median sale price")
    ax.set_title(f"Top {top_n} zipcodes by median price")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Model evaluation charts
# ----------------------------------------------------------------------------
def model_comparison(metrics: pd.DataFrame):
    """Three bar charts: MAE and RMSE (lower is better), R2 (higher is better)."""
    panels = [("MAE ($)", "MAE - average error (lower is better)", True),
              ("RMSE ($)", "RMSE - penalises big errors (lower is better)", True),
              ("R2 (test)", "R2 score (higher is better)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (column, title, is_money) in zip(axes, panels):
        values = metrics[column]
        best = values.idxmin() if is_money else values.idxmax()
        colors = [config.TEAL if name == best else config.SKY for name in values.index]
        bars = ax.bar(range(len(values)), values.values, color=colors)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in values.index], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
        if is_money:
            ax.yaxis.set_major_formatter(MONEY_FORMAT)
        for bar, value in zip(bars, values.values):
            label = f"${value / 1e3:,.0f}K" if is_money else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                    ha="center", va="bottom", fontsize=8, color=config.INK)
    fig.suptitle("Model comparison on the unseen test set (green = best)",
                 fontweight="bold", color=config.NAVY)
    fig.tight_layout()
    return fig


def _draw_actual_vs_predicted(ax, y_true, y_pred, name, color):
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, color=color, edgecolor="none")
    limit = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([0, limit], [0, limit], color=config.CORAL, linewidth=1.5, linestyle="--")
    ax.xaxis.set_major_formatter(MONEY_FORMAT)
    ax.yaxis.set_major_formatter(MONEY_FORMAT)
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title(name)


def actual_vs_predicted(y_true, y_pred, name: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    _draw_actual_vs_predicted(ax, y_true, y_pred, f"{name}: actual vs predicted",
                              config.MODEL_COLORS.get(name, config.TEAL))
    fig.tight_layout()
    return fig


def actual_vs_predicted_grid(y_true, predictions: dict):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, name in zip(axes.ravel(), config.MODEL_NAMES):
        _draw_actual_vs_predicted(ax, y_true, predictions[name], name,
                                  config.MODEL_COLORS[name])
    fig.suptitle("Actual vs predicted price (dashed line = perfect prediction)",
                 fontweight="bold", color=config.NAVY)
    fig.tight_layout()
    return fig


def residual_plots(y_true, y_pred, name: str):
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.histplot(residuals, bins=60, color=config.TEAL, ax=axes[0])
    axes[0].axvline(0, color=config.CORAL, linestyle="--")
    axes[0].xaxis.set_major_formatter(MONEY_FORMAT)
    axes[0].set_xlabel("Prediction error (predicted - actual)")
    axes[0].set_title(f"{name}: error distribution")

    axes[1].scatter(y_pred, residuals, s=8, alpha=0.3, color=config.NAVY, edgecolor="none")
    axes[1].axhline(0, color=config.CORAL, linestyle="--")
    axes[1].xaxis.set_major_formatter(MONEY_FORMAT)
    axes[1].yaxis.set_major_formatter(MONEY_FORMAT)
    axes[1].set_xlabel("Predicted price")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Errors vs predicted price")
    fig.tight_layout()
    return fig


def sample_comparison(table: pd.DataFrame):
    """Side-by-side bars of actual and predicted price for a few test houses."""
    positions = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(positions - 0.2, table["actual_price"], width=0.4, color=config.SKY, label="Actual")
    ax.bar(positions + 0.2, table["predicted_price"], width=0.4, color=config.TEAL, label="Predicted")
    ax.yaxis.set_major_formatter(MONEY_FORMAT)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i + 1) for i in positions], fontsize=8)
    ax.set_xlabel("House number in the table above")
    ax.set_ylabel("Price")
    ax.set_title("Actual vs predicted price for the sampled test houses")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def feature_importance_chart(values: pd.Series, kind: str, name: str):
    ordered = values.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(ordered.index, ordered.values, color=config.MODEL_COLORS.get(name, config.TEAL))
    ax.grid(axis="y", visible=False)
    if kind == "importance":
        ax.set_xlabel("Importance (share of the model's decisions)")
        ax.set_title(f"{name}: most important features")
    else:
        ax.set_xlabel("Absolute coefficient (features are standardised)")
        ax.set_title(f"{name}: largest coefficients")
    fig.tight_layout()
    return fig
