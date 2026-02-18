"""
Tests for PRISM regression.

Run with:  python -m pytest tests/ -v
Or:        python tests/test_prism.py
"""

import numpy as np
import pandas as pd
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prism import PRISMRegressor, apply_transform


def test_linear_equivalence():
    """
    Theorem 2: When restricted to linear-only transformations,
    PRISM should recover standard OLS exactly.

    We verify this by fitting PRISM on a simple linear dataset and
    comparing to sklearn's LinearRegression.
    """
    from sklearn.linear_model import LinearRegression

    rng = np.random.RandomState(42)
    n, p = 200, 5
    X = pd.DataFrame(rng.randn(n, p), columns=[f"X{i}" for i in range(p)])
    beta = np.array([3, -2, 1.5, 0.5, -1])
    y = pd.Series(X.values @ beta + 5 + rng.randn(n) * 0.5, name='y')

    # PRISM with m=100 (should converge fully)
    model = PRISMRegressor(m=100)
    model.fit(X, y, include_interactions=False, verbose=False)

    # OLS
    ols = LinearRegression().fit(X, y)
    ols_r2 = ols.score(X, y)

    # R² should be nearly identical (within floating point)
    assert abs(model.r2_ - ols_r2) < 0.01, (
        f"PRISM R² ({model.r2_:.6f}) and OLS R² ({ols_r2:.6f}) "
        f"differ by more than 0.01"
    )
    print(f"  PASS: Linear equivalence (PRISM R²={model.r2_:.6f}, "
          f"OLS R²={ols_r2:.6f})")


def test_transformations():
    """Verify all 7 transformations produce finite output."""
    x = np.array([-2, -1, 0, 0.5, 1, 5, 100], dtype=float)

    for tf in ['Linear', 'Logarithmic', 'Sqrt', 'Square', 'Cubic',
               'Inverse', 'Exponential']:
        result = apply_transform(x, tf)
        assert np.all(np.isfinite(result)), (
            f"Transform '{tf}' produced non-finite values: {result}"
        )
    print("  PASS: All transformations produce finite output")


def test_dirty_data_handling():
    """PRISM should handle NaN, non-numeric columns, and zero-variance."""
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame({
        'A': rng.randn(n),
        'B': rng.randn(n),
        'C': rng.randn(n),
        'Text': ['hello'] * n,     # non-numeric
        'Const': [42.0] * n,       # zero variance
    })
    X.iloc[0:10, 0] = np.nan       # NaN in A
    y = pd.Series(2 * X['A'].fillna(0) + 3 * X['B'] + rng.randn(n) * 0.2)
    y.iloc[95:100] = np.nan         # NaN in response

    model = PRISMRegressor(m=2)
    model.fit(X, y, include_interactions=False, verbose=False)

    # Should have selected from A, B, C only
    assert all(f in ['A', 'B', 'C'] for f in model.selected_features_), (
        f"Unexpected features selected: {model.selected_features_}"
    )
    assert model.r2_ > 0.5, f"R² too low: {model.r2_}"
    print(f"  PASS: Dirty data handled (R²={model.r2_:.3f}, "
          f"features={model.selected_features_})")


def test_predict_on_new_data():
    """Predict should work on unseen data."""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({'A': rng.randn(n), 'B': rng.randn(n)})
    y = pd.Series(2 * X['A'] + 3 * X['B'] + rng.randn(n) * 0.3)

    model = PRISMRegressor(m=5)
    model.fit(X, y, include_interactions=False, verbose=False)

    X_new = pd.DataFrame({'A': rng.randn(50), 'B': rng.randn(50)})
    y_pred = model.predict(X_new)

    assert len(y_pred) == 50
    assert np.all(np.isfinite(y_pred))
    print(f"  PASS: Predict on new data (50 predictions, all finite)")


def test_variance_attribution_sums():
    """Incremental R² + MR should equal final R² (approximately)."""
    rng = np.random.RandomState(42)
    n = 500
    X = pd.DataFrame({
        'A': rng.randn(n),
        'B': rng.randn(n),
        'C': rng.randn(n),
    })
    y = pd.Series(3 * X['A'] ** 2 + 2 * np.log(X['B'].abs() + 1)
                   + X['C'] + rng.randn(n) * 0.5)

    model = PRISMRegressor(m=10)
    model.fit(X, y, include_interactions=False, verbose=False)

    total_incr = sum(model.incremental_r2_) + model.mr_
    diff = abs(total_incr - model.r2_)

    assert diff < 0.02, (
        f"Attribution sum ({total_incr:.6f}) and final R² ({model.r2_:.6f}) "
        f"differ by {diff:.6f}"
    )
    print(f"  PASS: Attribution sums correctly "
          f"(sum={total_incr:.4f}, R²={model.r2_:.4f}, diff={diff:.6f})")


def test_interaction_testing():
    """Phase 2 should run without errors and either select or reject."""
    rng = np.random.RandomState(42)
    n = 300
    X = pd.DataFrame({'A': rng.randn(n), 'B': rng.randn(n)})
    y = pd.Series(X['A'] + X['B'] + 0.5 * X['A'] * X['B']
                   + rng.randn(n) * 0.3)

    model = PRISMRegressor(m=5)
    model.fit(X, y, include_interactions=True, verbose=False)

    # Should have run interaction testing
    assert model.step4_results_ is not None
    assert model.step4_results_['interactions_tested'] >= 1
    print(f"  PASS: Interaction testing ran "
          f"(tested={model.step4_results_['interactions_tested']}, "
          f"selected={len(model.step4_results_['interactions_selected'])})")


def test_score_method():
    """The .score() method should return a valid R²."""
    rng = np.random.RandomState(42)
    n = 200
    X_train = pd.DataFrame({'A': rng.randn(n), 'B': rng.randn(n)})
    y_train = pd.Series(2 * X_train['A'] + rng.randn(n) * 0.5)

    X_test = pd.DataFrame({'A': rng.randn(50), 'B': rng.randn(50)})
    y_test = pd.Series(2 * X_test['A'] + rng.randn(50) * 0.5)

    model = PRISMRegressor(m=5)
    model.fit(X_train, y_train, include_interactions=False, verbose=False)

    r2 = model.score(X_test, y_test)
    assert -1 < r2 < 1.1, f"R² out of range: {r2}"
    print(f"  PASS: score() returns valid R² ({r2:.3f})")


if __name__ == '__main__':
    print("=" * 60)
    print("PRISM Regression — Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_transformations,
        test_dirty_data_handling,
        test_predict_on_new_data,
        test_linear_equivalence,
        test_variance_attribution_sums,
        test_interaction_testing,
        test_score_method,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed "
          f"out of {len(tests)} tests")
    print("=" * 60)
