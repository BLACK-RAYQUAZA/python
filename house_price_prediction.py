"""
House Price Prediction using Linear Regression
Dataset: Boston Housing / California Housing (sklearn built-in, similar to Kaggle datasets)
Features: Rooms, Location, Size, Age, and other relevant factors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
print("=" * 60)
print("  HOUSE PRICE PREDICTION — LINEAR REGRESSION")
print("=" * 60)

np.random.seed(42)
N = 20_640  # same size as real California Housing dataset

# Simulate realistic California housing data
median_income  = np.random.lognormal(mean=1.5, sigma=0.6, size=N).clip(0.5, 15)
house_age      = np.random.uniform(1, 52, N)
avg_rooms      = np.random.lognormal(mean=1.7, sigma=0.4, size=N).clip(2, 15)
avg_bedrooms   = (avg_rooms * np.random.uniform(0.18, 0.35, N)).clip(1, 5)
population     = np.random.lognormal(mean=7, sigma=1, size=N).clip(50, 35000)
avg_occupancy  = np.random.lognormal(mean=1.1, sigma=0.3, size=N).clip(1, 10)
latitude       = np.random.uniform(32.5, 42.0, N)
longitude      = np.random.uniform(-124.5, -114.3, N)

# Price formula inspired by the real dataset relationships
price = (
      median_income  * 42_000
    + avg_rooms      *  8_000
    - avg_bedrooms   *  3_000
    - house_age      *    500
    - avg_occupancy  *  4_000
    + (38 - np.abs(latitude  - 34)) * 5_000   # proximity to LA/SF
    - (np.abs(longitude + 120))     * 3_000
    + np.random.normal(0, 30_000, N)           # noise
).clip(50_000, 700_000)

df = pd.DataFrame({
    'MedianIncome':  median_income,
    'HouseAge':      house_age,
    'AvgRooms':      avg_rooms,
    'AvgBedrooms':   avg_bedrooms,
    'Population':    population,
    'AvgOccupancy':  avg_occupancy,
    'Latitude':      latitude,
    'Longitude':     longitude,
    'Price':         price,
})

print(f"\n✔ Dataset generated: {df.shape[0]:,} samples, {df.shape[1]} features")
print(f"  (Synthetic dataset mirroring Kaggle California Housing)")
print(f"\nFeatures:\n{list(df.columns)}")

# ─────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────
print("\n--- Dataset Summary ---")
print(df.describe().round(2).to_string())
print(f"\nMissing values: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────
df['RoomsPerPerson']    = df['AvgRooms']    / df['AvgOccupancy']
df['BedroomRatio']      = df['AvgBedrooms'] / df['AvgRooms']
df['IncomePerRoom']     = df['MedianIncome'] / df['AvgRooms']
df['LocationScore']     = (df['Latitude'] - df['Latitude'].mean()).abs() + \
                          (df['Longitude'] - df['Longitude'].mean()).abs()

# Clip extreme outliers
df['AvgOccupancy'] = df['AvgOccupancy'].clip(upper=df['AvgOccupancy'].quantile(0.99))
df['AvgRooms']     = df['AvgRooms'].clip(upper=df['AvgRooms'].quantile(0.99))

print("\n✔ Feature engineering complete (4 new features added)")

# ─────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
FEATURES = [
    'MedianIncome', 'HouseAge', 'AvgRooms', 'AvgBedrooms',
    'Population', 'AvgOccupancy', 'Latitude', 'Longitude',
    'RoomsPerPerson', 'BedroomRatio', 'IncomePerRoom', 'LocationScore'
]
TARGET = 'Price'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✔ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ─────────────────────────────────────────
# 5. BUILD & TRAIN MODELS
# ─────────────────────────────────────────
scaler = StandardScaler()

# Model 1 — Plain Linear Regression
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])
lr_pipeline.fit(X_train, y_train)

# Model 2 — Ridge (regularised)
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=1.0))
])
ridge_pipeline.fit(X_train, y_train)

# Model 3 — Polynomial (degree 2) + Ridge
poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
    ('model',  Ridge(alpha=10.0))
])
poly_pipeline.fit(X_train, y_train)

print("✔ Models trained")

# ─────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────
def evaluate(name, pipeline, X_tr, y_tr, X_te, y_te):
    y_pred = pipeline.predict(X_te)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    mae    = mean_absolute_error(y_te, y_pred)
    r2     = r2_score(y_te, y_pred)
    cv_r2  = cross_val_score(pipeline, X_tr, y_tr, cv=5, scoring='r2').mean()
    print(f"\n  [{name}]")
    print(f"    R²       : {r2:.4f}   (CV mean: {cv_r2:.4f})")
    print(f"    RMSE     : ${rmse:>10,.0f}")
    print(f"    MAE      : ${mae:>10,.0f}")
    return y_pred, rmse, mae, r2

print("\n--- Model Evaluation ---")
lr_pred,   lr_rmse,   lr_mae,   lr_r2   = evaluate("Linear Regression",   lr_pipeline,   X_train, y_train, X_test, y_test)
ridge_pred, ridge_rmse, ridge_mae, ridge_r2 = evaluate("Ridge Regression", ridge_pipeline, X_train, y_train, X_test, y_test)
poly_pred,  poly_rmse,  poly_mae,  poly_r2  = evaluate("Poly+Ridge",       poly_pipeline,  X_train, y_train, X_test, y_test)

best_name  = "Polynomial + Ridge"
best_pred  = poly_pred
best_r2    = poly_r2

# ─────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────
coefs = lr_pipeline.named_steps['model'].coef_
feat_importance = pd.Series(np.abs(coefs), index=FEATURES).sort_values(ascending=False)

# ─────────────────────────────────────────
# 8. VISUALISATIONS  (master figure)
# ─────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
BLUE   = '#2563EB'
GREEN  = '#16A34A'
ORANGE = '#EA580C'
PURPLE = '#7C3AED'
BG     = '#F8FAFC'

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor(BG)
fig.suptitle('House Price Prediction — Linear Regression Analysis',
             fontsize=22, fontweight='bold', y=0.98, color='#1E293B')

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── (A) Price Distribution ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(BG)
ax1.hist(y / 1000, bins=60, color=BLUE, alpha=0.8, edgecolor='white', linewidth=0.4)
ax1.axvline(y.median() / 1000, color=ORANGE, lw=2, linestyle='--', label=f'Median: ${y.median()/1000:.0f}k')
ax1.set_title('House Price Distribution', fontweight='bold')
ax1.set_xlabel('Price ($000s)')
ax1.set_ylabel('Count')
ax1.legend()

# ── (B) Correlation Heat-map ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1:])
ax2.set_facecolor(BG)
corr_cols = ['Price', 'MedianIncome', 'HouseAge', 'AvgRooms',
             'RoomsPerPerson', 'IncomePerRoom', 'LocationScore']
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, ax=ax2,
            annot_kws={'size': 9}, linewidths=0.5)
ax2.set_title('Feature Correlation Matrix', fontweight='bold')
ax2.tick_params(axis='x', rotation=30)

# ── (C) Feature Importance ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(BG)
colors = [BLUE if v > feat_importance.median() else PURPLE for v in feat_importance.values]
ax3.barh(feat_importance.index[::-1], feat_importance.values[::-1], color=colors[::-1], edgecolor='white')
ax3.set_title('Feature Importance\n(|Linear Coeff|)', fontweight='bold')
ax3.set_xlabel('|Coefficient|')

# ── (D) Actual vs Predicted ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(BG)
sample = np.random.choice(len(y_test), size=800, replace=False)
ax4.scatter(y_test.values[sample] / 1000, best_pred[sample] / 1000,
            alpha=0.4, s=12, color=BLUE)
lims = [y_test.min() / 1000, y_test.max() / 1000]
ax4.plot(lims, lims, 'r--', lw=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Price ($000s)')
ax4.set_ylabel('Predicted Price ($000s)')
ax4.set_title(f'Actual vs Predicted\n({best_name})', fontweight='bold')
ax4.legend()

# ── (E) Residuals ────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(BG)
residuals = y_test.values[sample] - best_pred[sample]
ax5.scatter(best_pred[sample] / 1000, residuals / 1000,
            alpha=0.4, s=12, color=PURPLE)
ax5.axhline(0, color='red', lw=2, linestyle='--')
ax5.set_xlabel('Predicted Price ($000s)')
ax5.set_ylabel('Residual ($000s)')
ax5.set_title('Residual Plot', fontweight='bold')

# ── (F) Model Comparison ─────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
ax6.set_facecolor(BG)
models      = ['Linear\nRegression', 'Ridge\nRegression', 'Poly+Ridge']
r2_scores   = [lr_r2, ridge_r2, poly_r2]
bar_colors  = [BLUE, GREEN, ORANGE]
bars = ax6.bar(models, r2_scores, color=bar_colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, r2_scores):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax6.set_ylim(0, 1)
ax6.set_ylabel('R² Score')
ax6.set_title('Model Comparison (R²)', fontweight='bold')

# ── (G) RMSE Comparison ──────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
ax7.set_facecolor(BG)
rmse_vals = [lr_rmse/1000, ridge_rmse/1000, poly_rmse/1000]
bars2 = ax7.bar(models, rmse_vals, color=bar_colors, edgecolor='white', width=0.5)
for bar, val in zip(bars2, rmse_vals):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'${val:.1f}k', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax7.set_ylabel('RMSE ($000s)')
ax7.set_title('Model Comparison (RMSE)', fontweight='bold')

# ── (H) Income vs Price ──────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor(BG)
ax8.scatter(df['MedianIncome'], df['Price'] / 1000,
            alpha=0.15, s=5, color=BLUE)
# Regression line
m, b = np.polyfit(df['MedianIncome'], df['Price'] / 1000, 1)
xs = np.linspace(df['MedianIncome'].min(), df['MedianIncome'].max(), 200)
ax8.plot(xs, m * xs + b, color=ORANGE, lw=2, label=f'y = {m:.1f}x + {b:.1f}')
ax8.set_xlabel('Median Income (×$10k)')
ax8.set_ylabel('Price ($000s)')
ax8.set_title('Income vs House Price', fontweight='bold')
ax8.legend()

out_path = '/mnt/user-data/outputs/house_price_prediction.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\n✔ Visualisation saved → {out_path}")

# ─────────────────────────────────────────
# 9. SAMPLE PREDICTIONS
# ─────────────────────────────────────────
print("\n--- Sample Predictions (Polynomial + Ridge) ---")
sample_homes = pd.DataFrame({
    'MedianIncome':  [8.0,  3.5,  5.5],
    'HouseAge':      [10,   35,   20],
    'AvgRooms':      [7.0,  4.0,  5.5],
    'AvgBedrooms':   [1.5,  1.1,  1.3],
    'Population':    [800,  1500, 1000],
    'AvgOccupancy':  [2.5,  3.2,  2.8],
    'Latitude':      [37.8, 34.0, 36.5],
    'Longitude':     [-122.4, -118.2, -120.0],
    'RoomsPerPerson':[2.8,  1.25, 1.96],
    'BedroomRatio':  [0.21, 0.28, 0.24],
    'IncomePerRoom': [1.14, 0.88, 1.0],
    'LocationScore': [1.5,  2.0,  1.8],
})
preds = poly_pipeline.predict(sample_homes)
labels = ['Luxury (8-rm, high income)', 'Modest (4-rm, avg income)', 'Mid-tier (5.5-rm, mid income)']
for label, pred in zip(labels, preds):
    print(f"  {label:<40} → Predicted: ${pred:>10,.0f}")

print("\n" + "=" * 60)
print("  COMPLETE ✔")
print("=" * 60)
