"""
Example: PRISM on California Housing
=====================================
Reproduces the primary empirical application from the paper.

Expected output (approximate):
  - 8 variables selected, 6 with non-linear transformations
  - Base R² ≈ 67.1% (training), 65.8% (test)
  - OLS R² ≈ 60.6% (training), 57.6% (test)
  - Improvement over OLS: +8.2 pp on test set
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# If running from the repo root (not pip-installed), uncomment:
# import sys; sys.path.insert(0, '..')

from prism import PRISMRegressor

# ------------------------------------------------------------------
# 1.  Load data
# ------------------------------------------------------------------
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

print(f"Full dataset: n={len(X)}, p={X.shape[1]}")
print(f"Features: {list(X.columns)}\n")

# ------------------------------------------------------------------
# 2.  Train / test split (80/20, seed=42 — same as paper)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 3.  Fit PRISM (m=10 for publication-quality attribution)
# ------------------------------------------------------------------
model = PRISMRegressor(m=10)
model.fit(X_train, y_train, include_interactions=True, verbose=True)

# ------------------------------------------------------------------
# 4.  Evaluate on test set
# ------------------------------------------------------------------
test_r2 = model.score(X_test, y_test)
print(f"\nTest R² : {test_r2:.4f}  ({test_r2*100:.1f}%)")
print(f"Train R²: {model.r2_:.4f}  ({model.r2_*100:.1f}%)")

# ------------------------------------------------------------------
# 5.  Compare to OLS
# ------------------------------------------------------------------
from sklearn.linear_model import LinearRegression

ols = LinearRegression().fit(X_train, y_train)
ols_test = ols.score(X_test, y_test)
print(f"\nOLS test R²: {ols_test:.4f}  ({ols_test*100:.1f}%)")
print(f"PRISM improvement: +{(test_r2 - ols_test)*100:.1f} pp")

# ------------------------------------------------------------------
# 6.  Variance attribution
# ------------------------------------------------------------------
print("\nVariance Attribution Table:")
print(model.get_variance_attribution().to_string(index=False))

# ------------------------------------------------------------------
# 7.  Diagnostic plots
# ------------------------------------------------------------------
fig = model.plot_results(X_train, y_train, figsize=(16, 12))
fig.savefig('california_housing_prism.png', dpi=150, bbox_inches='tight')
print("\nPlots saved to california_housing_prism.png")
