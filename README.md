# CSV Data Analysis — College Project

Analyse any CSV file using **Pandas**, **Matplotlib**, and **Seaborn**.  
Generates bar charts, scatter plots, heatmaps, trend lines, and a full statistical summary.

## Features
- `df.describe()` — full descriptive statistics (mean, median, std, min, max)
- `df.isnull().sum()` — missing value audit
- **Histogram** — distribution of any numeric column
- **Scatter plot** — two numeric columns with Pearson correlation + regression line
- **Heatmap** — full correlation matrix for all numeric columns
- **Value counts** — bar chart for categorical columns
- **Trend line** — rolling mean overlay
- Saves output as `analysis_output.png`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Interactive prompt
python analysis.py

# Pass CSV directly
python analysis.py my_data.csv
```

## Output

All 6 charts are saved to **`analysis_output.png`** and displayed on screen.

## File Structure

```
├── analysis.py          ← main script
├── requirements.txt     ← dependencies
├── README.md
└── data.csv             ← your CSV file (add your own)
```

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, statistics |
| `matplotlib` | Plotting engine |
| `seaborn` | Heatmap + styled charts |
| `numpy` | Regression & numeric ops |
