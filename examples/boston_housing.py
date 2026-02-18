"""
Example: PRISM on Boston Housing
=================================
Reproduces the third empirical application from the paper.

Boston Housing (n=506, p=13) is a classic linear regression benchmark.
PRISM discovered meaningful non-linearities achieving 74.8% test R²
versus 66.9% for OLS (+7.9 pp) using half the parameters (7 vs 14).

Note: This dataset contains a variable (B) derived from racial
demographics.  We include it for benchmark comparability only.
PRISM's F-test stopping rule excludes it from the selected model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# If running from the repo root (not pip-installed), uncomment:
# import sys; sys.path.insert(0, '..')

from prism import PRISMRegressor

# ------------------------------------------------------------------
# 1.  Load data  (OpenML or CSV fallback)
# ------------------------------------------------------------------
try:
    from sklearn.datasets import fetch_openml
    boston = fetch_openml(name='boston', version=1, as_frame=True,
                         parser='auto')
    X = boston.data
    y = boston.target.astype(float)
    print("Loaded Boston Housing from OpenML.\n")
except Exception:
    print("Could not load from OpenML.  Trying CSV fallback...")
    try:
        url = (
            "https://raw.githubusercontent.com/selva86/datasets/"
            "master/BostonHousing.csv"
        )
        df = pd.read_csv(url)
        y = df.pop('medv')
        X = df
        print("Loaded Boston Housing from GitHub.\n")
    except Exception:
        raise RuntimeError(
            "Could not load Boston Housing. Please download it manually."
        )

print(f"Dataset: n={len(X)}, p={X.shape[1]}")
print(f"Features: {list(X.columns)}\n")

# ------------------------------------------------------------------
# 2.  Train / test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 3.  Fit PRISM
# ------------------------------------------------------------------
model = PRISMRegressor(m=10)
model.fit(X_train, y_train, include_interactions=True, verbose=True)

# ------------------------------------------------------------------
# 4.  Evaluate
# ------------------------------------------------------------------
test_r2 = model.score(X_test, y_test)
print(f"\nTest R² : {test_r2:.4f}  ({test_r2*100:.1f}%)")

from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(X_train, y_train)
ols_test = ols.score(X_test, y_test)
print(f"OLS R²  : {ols_test:.4f}  ({ols_test*100:.1f}%)")
print(f"Improvement: +{(test_r2 - ols_test)*100:.1f} pp")

# ------------------------------------------------------------------
# 5.  Variance attribution
# ------------------------------------------------------------------
print("\nVariance Attribution:")
print(model.get_variance_attribution().to_string(index=False))
