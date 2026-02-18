"""
Example: PRISM on Concrete Compressive Strength
=================================================
Reproduces the second empirical application from the paper.

This dataset (Yeh, 1998) predicts concrete strength from mix-design
variables.  PRISM achieved its strongest result here: +18.7 pp over OLS.

The algorithm discovers physically interpretable transformations:
  - Logarithmic curing age (well-known curing curve)
  - Square root blast furnace slag (diminishing returns)
  - Logarithmic superplasticizer (saturation effect)

Note: The dataset is fetched from the UCI ML Repository.
      If the download fails, you can get it manually from:
      https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# If running from the repo root (not pip-installed), uncomment:
# import sys; sys.path.insert(0, '..')

from prism import PRISMRegressor

# ------------------------------------------------------------------
# 1.  Load data
# ------------------------------------------------------------------
# Try to load from UCI via openpyxl / direct URL
try:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "concrete/compressive/Concrete_Data.xls"
    )
    df = pd.read_excel(url)
    print("Loaded Concrete dataset from UCI.\n")
except Exception:
    print("Could not download from UCI.  Trying alternative...")
    try:
        url2 = (
            "https://raw.githubusercontent.com/stedy/Machine-Learning-"
            "with-R-datasets/master/concrete.csv"
        )
        df = pd.read_csv(url2)
        print("Loaded Concrete dataset from GitHub mirror.\n")
    except Exception:
        raise RuntimeError(
            "Could not load the Concrete dataset.  Please download it "
            "manually from UCI and place it in this directory as "
            "'concrete.csv'."
        )

# Standardise column names
df.columns = [
    'Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water',
    'Superplasticizer', 'CoarseAggregate', 'FineAggregate',
    'Age', 'Strength',
]

X = df.drop(columns=['Strength'])
y = df['Strength']

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
