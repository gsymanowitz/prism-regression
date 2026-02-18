"""
Example: PRISM with dirty / messy data
========================================
Demonstrates PRISM's automatic data cleaning capabilities.

PRISM handles:
  - Missing values (NaN) in predictors → median imputation
  - Missing values in the response → rows dropped
  - Infinite values → replaced or dropped
  - Non-numeric columns → dropped
  - Zero-variance columns → dropped

All cleaning actions are reported transparently.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# If running from the repo root (not pip-installed), uncomment:
# import sys; sys.path.insert(0, '..')

from prism import PRISMRegressor

# ------------------------------------------------------------------
# 1.  Load clean data, then make it dirty
# ------------------------------------------------------------------
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

print(f"Original dataset: n={len(X)}, p={X.shape[1]}")
print()

# Introduce various data quality issues
X_dirty = X.copy()
y_dirty = y.copy()

# a) Missing values in predictors (500 NaNs in MedInc, 200 in AveRooms)
rng = np.random.RandomState(42)
X_dirty.loc[rng.choice(len(X_dirty), 500, replace=False), 'MedInc'] = np.nan
X_dirty.loc[rng.choice(len(X_dirty), 200, replace=False), 'AveRooms'] = np.nan

# b) Missing values in response (100 NaNs)
y_dirty.iloc[rng.choice(len(y_dirty), 100, replace=False)] = np.nan

# c) A non-numeric column
X_dirty['CityName'] = 'Unknown'

# d) A zero-variance column
X_dirty['Constant'] = 42.0

# e) A few infinite values
X_dirty.iloc[0, 3] = np.inf
X_dirty.iloc[1, 3] = -np.inf

print(f"Dirty dataset: n={len(X_dirty)}, p={X_dirty.shape[1]}")
print(f"  NaN in predictors: {X_dirty.isna().sum().sum()}")
print(f"  NaN in response:   {y_dirty.isna().sum()}")
print(f"  Inf in predictors: {np.isinf(X_dirty.select_dtypes(np.number).values).sum()}")
print(f"  Non-numeric cols:  {len(X_dirty.select_dtypes(exclude=np.number).columns)}")
print(f"  Zero-var cols:     1 (Constant)")
print()

# ------------------------------------------------------------------
# 2.  Fit PRISM — it handles all of the above automatically
# ------------------------------------------------------------------
model = PRISMRegressor(m=10)
model.fit(X_dirty, y_dirty, include_interactions=False, verbose=True)

# ------------------------------------------------------------------
# 3.  Confirm it still works
# ------------------------------------------------------------------
print(f"\nFinal R²: {model.r2_*100:.1f}%")
print(f"Variables selected: {model.n_selected_}")
print(f"Features: {model.selected_features_}")
