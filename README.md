# PRISM: Progressive Refinement with Interpretable Sequential Modeling

A Python library implementing the PRISM regression methodology for automatic non-linear transformation discovery with exact sequential variance decomposition.

**Paper:** *"PRISM: Progressive Refinement with Interpretable Sequential Modeling — A Novel Regression Methodology for Automatic Transformation Discovery and Variance Attribution"* by Gavin Symanowitz (2026).

---

## What is PRISM?

PRISM is a regression algorithm that solves a common problem: real-world data often has non-linear relationships, but the most interpretable models (like OLS) assume everything is linear. PRISM automatically discovers the best non-linear transformation for each predictor while telling you exactly how much each variable contributes to the model.

**Key capabilities:**

- **Automatic transformation discovery** — tests 7 parametric types (linear, log, sqrt, square, cubic, inverse, exponential) per variable
- **Near-exact sequential variance decomposition** — with m≥10, the sum of incremental R² contributions equals the final R² (multivariate refinement < 0.1%)
- **Full interpretability** — explicit parametric forms like "inverse occupancy" or "logarithmic age", not black-box curves
- **Built-in interaction testing** — Phase 2 tests all two-way interactions using BIC with k×5 penalties
- **Robust data handling** — automatically handles missing values, non-numeric columns, and zero-variance features

## Installation

```bash
pip install prism-regression
```

Or install from source:

```bash
git clone https://github.com/gsymanowitz/prism-regression.git
cd prism-regression
pip install -e .
```

## Quick Start

```python
from prism import PRISMRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit PRISM
model = PRISMRegressor(m=10)
model.fit(X_train, y_train)

# Evaluate
print(f"Test R²: {model.score(X_test, y_test):.3f}")

# See variance attribution
print(model.get_variance_attribution())
```

## How It Works

PRISM operates in two phases:

### Phase 1: Base Model Construction

1. **Step 1 — Transformation Screening:** For each predictor, test all 7 transformations against the original response. Rank variables by univariate R².
2. **Step 2 — Sequential Selection:** Iteratively add the variable+transformation with the highest F-statistic on current residuals. Critically, transformations are *retested* at each round because the optimal transform on residuals differs from the optimal transform on the original response. After each addition, run `m` coordinate descent iterations to properly attribute joint effects.
3. **Step 3 — Final Convergence:** Run coordinate descent until R² stabilises (|ΔR²| < 10⁻⁸).

### Phase 2: Interaction Enhancement

4. **Step 4 — Interaction Testing:** Test all two-way interactions between selected variables using BIC with a k×5 complexity penalty. Only interactions that overcome the penalty threshold are retained.

## The `m` Parameter

The `m` parameter controls variance attribution quality:

| m | MR (unattributed) | Use case |
|---|---|---|
| 0 | ~5% | Quick exploration |
| 2 | ~1.5% | Preliminary analysis |
| **10** | **< 0.1%** | **Recommended default** |
| 100 | ~0% | Theoretical verification |

**Recommendation:** Use `m=10` for publication and production. It achieves near-exact sequential decomposition with modest computational overhead.

## Empirical Results

PRISM has been validated on three contrasting datasets:

| Dataset | n | PRISM R² | OLS R² | Improvement |
|---|---|---|---|---|
| California Housing | 20,640 | 65.8% | 57.6% | +8.2 pp |
| Concrete Strength | 1,030 | 81.5% | 62.8% | +18.7 pp |
| Boston Housing | 506 | 74.8% | 66.9% | +7.9 pp |

PRISM achieves 74–92% of Random Forest performance while maintaining complete interpretability and exact sequential variance decomposition.

On Concrete Compressive Strength, PRISM discovered physically interpretable transformations — logarithmic curing age and square root blast furnace slag — that align with established materials science, demonstrating genuine transformation discovery without domain expertise.

## API Reference

### `PRISMRegressor`

```python
PRISMRegressor(
    m=10,                      # Coord descent iterations per variable
    alpha=0.05,                # F-test significance level
    interaction_penalty=5.0,   # BIC multiplier for interactions
    max_iterations=100,        # Max Step 3 iterations
    convergence_tolerance=1e-8 # R² change threshold
)
```

**Methods:**

| Method | Description |
|---|---|
| `.fit(X, y)` | Fit the model. Handles dirty data automatically. |
| `.predict(X)` | Predict on new data. |
| `.score(X, y)` | Return R² on test data. |
| `.get_variance_attribution()` | Return the sequential R² decomposition table. |
| `.plot_results(X, y)` | Diagnostic plots (transformation curves + predicted vs actual). |

**Attributes (after fitting):**

| Attribute | Description |
|---|---|
| `.selected_features_` | List of selected variable names |
| `.transform_dict_` | Dict mapping variable → transformation type |
| `.coefficients_` | Dict mapping variable → coefficient |
| `.incremental_r2_` | List of incremental R² contributions |
| `.mr_` | Multivariate refinement (ideally < 0.001) |
| `.r2_` | Final model R² |
| `.n_selected_` | Number of variables selected |
| `.runtime_` | Fitting time in seconds |

### `fit_prism` (convenience function)

```python
from prism import fit_prism

model = fit_prism(X, y, m=10, include_interactions=True, verbose=True)
```

## Dirty Data Handling

PRISM automatically handles common data quality issues:

- **Missing values in predictors** → imputed with column medians
- **Missing values in response** → rows dropped
- **Infinite values** → replaced or dropped
- **Non-numeric columns** → dropped with a warning
- **Zero-variance columns** → dropped with a warning
- **All actions are reported** so you know exactly what happened

```python
# Works even with messy data
import numpy as np

X_dirty = X.copy()
X_dirty.iloc[0:50, 2] = np.nan         # Missing values
X_dirty['Notes'] = 'some text'          # Non-numeric column

model = PRISMRegressor(m=10)
model.fit(X_dirty, y)  # Cleans automatically, reports what it did
```

## Transformation Library

PRISM tests these 7 transformations for each variable:

| Transform | Formula | Captures |
|---|---|---|
| Linear | x | Proportional effects |
| Logarithmic | log(x + 1) | Diminishing returns |
| Square Root | √\|x\| | Moderate diminishing returns |
| Square | x² | Accelerating effects |
| Cubic | x³ | S-shaped relationships |
| Inverse | 1/(x + 1) | Asymptotic decay |
| Exponential | exp(clip(x/σ, −10, 10)) | Explosive growth |

## Scikit-learn Compatibility

`PRISMRegressor` follows the scikit-learn estimator interface (`BaseEstimator`, `RegressorMixin`), so it works with sklearn utilities:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(PRISMRegressor(m=10, verbose=False),
                         X, y, cv=5, scoring='r2')
print(f"CV R²: {scores.mean():.3f} ± {scores.std():.3f}")
```

> **Note:** Cross-validation with `verbose=False` suppresses output for each fold.

## Citation

If you use PRISM in your research, please cite:

```bibtex
@article{symanowitz2026prism,
  title   = {PRISM: Progressive Refinement with Interpretable Sequential
             Modeling -- A Novel Regression Methodology for Automatic
             Transformation Discovery and Variance Attribution},
  author  = {Symanowitz, Gavin},
  year    = {2026},
  journal = {[Manuscript under review]}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
