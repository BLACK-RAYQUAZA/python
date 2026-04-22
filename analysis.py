"""
CSV Data Analysis — College Project
====================================
Uses: pandas, matplotlib, seaborn, numpy
Run :  python analysis.py
       python analysis.py my_data.csv      ← to load a specific file
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 0. Setup ────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "#0f0f13",
                     "axes.facecolor": "#17171f", "axes.edgecolor": "#2a2a38",
                     "axes.labelcolor": "#e2e2e8", "xtick.color": "#6b6b80",
                     "ytick.color": "#6b6b80", "text.color": "#e2e2e8",
                     "grid.color": "#2a2a38", "grid.linewidth": 0.6})

ACCENT   = "#00d4aa"
ACCENT2  = "#7c6af7"
PALETTE  = [ACCENT, ACCENT2, "#f59e0b", "#f43f5e", "#38bdf8", "#a3e635"]


# ── 1. Load CSV ──────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"\n✔  Loaded '{path}'  →  {df.shape[0]} rows × {df.shape[1]} columns\n")
    return df


# ── 2. Basic EDA ─────────────────────────────────────────────────────────────
def basic_eda(df: pd.DataFrame) -> None:
    """Print basic exploratory data analysis to the console."""
    print("=" * 60)
    print("  DATASET INFO")
    print("=" * 60)
    df.info()

    print("\n" + "=" * 60)
    print("  DESCRIPTIVE STATISTICS  (df.describe())")
    print("=" * 60)
    print(df.describe(include="all").T.to_string())

    print("\n" + "=" * 60)
    print("  MISSING VALUES  (df.isnull().sum())")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing": missing, "pct %": missing_pct})
    print(summary[summary["missing"] > 0].to_string() or "  No missing values ✓")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        print("\n" + "=" * 60)
        print("  AVERAGES  (df[col].mean())")
        print("=" * 60)
        for col in num_cols:
            print(f"  {col:<30} mean = {df[col].mean():.4f}   "
                  f"median = {df[col].median():.4f}   "
                  f"std = {df[col].std():.4f}")


# ── 3. Bar Chart ─────────────────────────────────────────────────────────────
def plot_bar(df: pd.DataFrame, col: str, ax: plt.Axes) -> None:
    """Bar chart: distribution histogram of a numeric column."""
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        ax.set_visible(False)
        return
    ax.hist(df[col].dropna(), bins=12, color=ACCENT, edgecolor="#0b0c10",
            linewidth=0.4, alpha=0.9)
    mean_val = df[col].mean()
    ax.axvline(mean_val, color=ACCENT2, linewidth=1.8, linestyle="--",
               label=f"Mean = {mean_val:.2f}")
    ax.set_title(f"Distribution — {col}", fontweight="bold", pad=10)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)


# ── 4. Scatter Plot ──────────────────────────────────────────────────────────
def plot_scatter(df: pd.DataFrame, col_x: str, col_y: str, ax: plt.Axes) -> None:
    """Scatter plot with regression line between two numeric columns."""
    if col_x not in df.columns or col_y not in df.columns:
        ax.set_visible(False)
        return
    clean = df[[col_x, col_y]].dropna()
    ax.scatter(clean[col_x], clean[col_y], color=ACCENT, alpha=0.65,
               edgecolors="#0b0c10", linewidths=0.4, s=40, zorder=3)

    # regression line
    m, b = np.polyfit(clean[col_x], clean[col_y], 1)
    xs = np.linspace(clean[col_x].min(), clean[col_x].max(), 100)
    ax.plot(xs, m * xs + b, color=ACCENT2, linewidth=1.8,
            linestyle="--", label=f"r = {clean[col_x].corr(clean[col_y]):.3f}")

    ax.set_title(f"Scatter — {col_x} vs {col_y}", fontweight="bold", pad=10)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.legend(fontsize=8)


# ── 5. Heatmap ───────────────────────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Correlation matrix heatmap for all numeric columns."""
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        ax.set_visible(False)
        return
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle only
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, linewidths=0.5, linecolor="#0b0c10",
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix Heatmap", fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=8)


# ── 6. Value-Counts Bar Chart (categorical) ──────────────────────────────────
def plot_value_counts(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Bar chart of the top-10 values in the first categorical column."""
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    if not cat_cols:
        ax.set_visible(False)
        return
    col = cat_cols[0]
    vc = df[col].value_counts().head(10)
    bars = ax.bar(range(len(vc)), vc.values,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(vc))],
                  edgecolor="#0b0c10", linewidth=0.4)
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc.index, rotation=35, ha="right", fontsize=8)
    ax.set_title(f"Value Counts — {col}", fontweight="bold", pad=10)
    ax.set_ylabel("Count")
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontsize=7, color="#e2e2e8")


# ── 7. Line Trend ─────────────────────────────────────────────────────────────
def plot_trend(df: pd.DataFrame, col: str, ax: plt.Axes) -> None:
    """Line chart showing the trend of a numeric column across row index."""
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        ax.set_visible(False)
        return
    series = df[col].dropna().reset_index(drop=True)
    ax.plot(series.index, series.values, color=ACCENT, linewidth=2, zorder=3)
    ax.fill_between(series.index, series.values, alpha=0.15, color=ACCENT)
    rolling = series.rolling(window=max(1, len(series)//10)).mean()
    ax.plot(rolling.index, rolling.values, color=ACCENT2, linewidth=1.5,
            linestyle="--", label="Rolling mean")
    ax.set_title(f"Trend — {col}", fontweight="bold", pad=10)
    ax.set_xlabel("Row index")
    ax.set_ylabel(col)
    ax.legend(fontsize=8)


# ── 8. Build & Save Figure ────────────────────────────────────────────────────
def build_figure(df: pd.DataFrame, output_path: str = "analysis_output.png") -> None:
    """Compose all charts into one figure and save as PNG."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    col1 = num_cols[0] if num_cols else None
    col2 = num_cols[1] if len(num_cols) > 1 else col1

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("CSV Data Analysis Dashboard", fontsize=16,
                 fontweight="bold", color="#e2e2e8", y=1.01)
    fig.patch.set_facecolor("#0b0c10")
    plt.subplots_adjust(hspace=0.45, wspace=0.32)

    if col1:
        plot_bar(df, col1, axes[0, 0])
        plot_scatter(df, col1, col2 or col1, axes[0, 1])
        plot_trend(df, col1, axes[1, 0])
    else:
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            ax.set_visible(False)

    plot_heatmap(df, axes[0, 2])
    plot_value_counts(df, axes[1, 1])

    # Summary text box
    ax_info = axes[1, 2]
    ax_info.axis("off")
    summary_lines = [
        f"Rows:     {len(df)}",
        f"Columns:  {len(df.columns)}",
        f"Numeric:  {len(df.select_dtypes(include='number').columns)}",
        f"Categ.:   {len(df.select_dtypes(exclude='number').columns)}",
        f"Missing:  {df.isnull().sum().sum()}",
    ]
    if col1:
        summary_lines += [
            "",
            f"{col1}",
            f"  mean   = {df[col1].mean():.4f}",
            f"  median = {df[col1].median():.4f}",
            f"  std    = {df[col1].std():.4f}",
            f"  min    = {df[col1].min():.4f}",
            f"  max    = {df[col1].max():.4f}",
        ]
    ax_info.text(0.05, 0.95, "\n".join(summary_lines),
                 transform=ax_info.transAxes, va="top", ha="left",
                 fontsize=9, color="#e2e2e8", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#17171f",
                           edgecolor="#2a2a38", linewidth=1))
    ax_info.set_title("Dataset Summary", fontweight="bold", pad=10)

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"\n✔  Figure saved → {output_path}")
    plt.show()


# ── 9. Entry point ────────────────────────────────────────────────────────────
def main():
    # Accept CSV path as CLI argument, otherwise prompt
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter path to CSV file (default: data.csv): ").strip() or "data.csv"

    df = load_data(path)
    basic_eda(df)
    build_figure(df, output_path="analysis_output.png")
    print("\nDone! Open 'analysis_output.png' to see all charts.")


if __name__ == "__main__":
    main()
