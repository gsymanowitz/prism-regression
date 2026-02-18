"""
PRISM: Progressive Refinement with Interpretable Sequential Modeling

A regression methodology that automatically discovers optimal non-linear
transformations while providing exact sequential variance decomposition
through explicit parametric forms.

Author: Gavin Symanowitz
Paper:  "PRISM: Progressive Refinement with Interpretable Sequential Modeling"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

TRANSFORM_TYPES = [
    'Linear', 'Logarithmic', 'Sqrt', 'Square',
    'Cubic', 'Inverse', 'Exponential',
]


def apply_transform(x, transform_type):
    """
    Apply one of the seven PRISM transformations to a numeric array.

    Parameters
    ----------
    x : array-like
        Input values (1-D).
    transform_type : str
        One of: 'Linear', 'Logarithmic', 'Sqrt', 'Square',
        'Cubic', 'Inverse', 'Exponential'.

    Returns
    -------
    np.ndarray
        Transformed values with any non-finite entries replaced by 0.
    """
    x = np.asarray(x, dtype=np.float64).ravel()

    if transform_type == 'Linear':
        result = x
    elif transform_type == 'Logarithmic':
        result = np.log(x + 1)                       # NOT log(|x|+1)
    elif transform_type == 'Sqrt':
        result = np.sqrt(np.abs(x))
    elif transform_type == 'Square':
        result = x ** 2
    elif transform_type == 'Cubic':
        result = x ** 3
    elif transform_type == 'Inverse':
        result = 1.0 / (x + 1)
    elif transform_type == 'Exponential':
        sigma = np.std(x) + 1                         # NOT +1e-10
        clipped = np.clip(x / sigma, -10, 10)
        result = np.exp(clipped)
    else:
        raise ValueError(f"Unknown transform: {transform_type}")

    # Replace any non-finite values produced by extreme inputs
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


# ---------------------------------------------------------------------------
# Data validation / cleaning
# ---------------------------------------------------------------------------

def _validate_and_clean(X, y, verbose=False):
    """
    Validate inputs, handle missing data, and convert to pandas objects.

    Cleaning steps
    --------------
    1.  Convert X / y to DataFrame / Series if they aren't already.
    2.  Drop columns that are entirely NaN or have zero variance.
    3.  Drop rows where y is NaN.
    4.  For remaining NaN values in X, impute with column medians
        (median is robust to outliers and skew).
    5.  Drop any columns that are non-numeric.
    6.  Warn the user about every cleaning action taken.

    Returns
    -------
    X_clean : pd.DataFrame
    y_clean : pd.Series
    """
    # --- Convert types ---------------------------------------------------
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    elif not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")
    elif not isinstance(y, pd.Series):
        y = pd.Series(y, name="y")

    X = X.copy()
    y = y.copy()

    # Align indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    cleaning_actions = []

    # --- Drop non-numeric columns ----------------------------------------
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        cleaning_actions.append(
            f"Dropped {len(non_numeric)} non-numeric column(s): {non_numeric}"
        )
        X = X.drop(columns=non_numeric)

    if X.shape[1] == 0:
        raise ValueError("No numeric predictor columns remain after cleaning.")

    # --- Drop columns that are all NaN -----------------------------------
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        cleaning_actions.append(
            f"Dropped {len(all_nan_cols)} all-NaN column(s): {all_nan_cols}"
        )
        X = X.drop(columns=all_nan_cols)

    # --- Drop rows where y is NaN ----------------------------------------
    y_nan_mask = y.isna()
    if y_nan_mask.any():
        n_drop = y_nan_mask.sum()
        cleaning_actions.append(
            f"Dropped {n_drop} row(s) where the response variable was NaN"
        )
        keep = ~y_nan_mask
        X = X.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)

    # --- Drop infinite values in y ---------------------------------------
    y_inf_mask = ~np.isfinite(y)
    if y_inf_mask.any():
        n_drop = y_inf_mask.sum()
        cleaning_actions.append(
            f"Dropped {n_drop} row(s) where the response variable was infinite"
        )
        keep = ~y_inf_mask
        X = X.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)

    # --- Impute NaN in X with column medians -----------------------------
    nan_counts = X.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        details = ", ".join(
            f"{c}({n})" for c, n in cols_with_nan.items()
        )
        cleaning_actions.append(
            f"Imputed NaN with column medians in {len(cols_with_nan)} "
            f"column(s): {details}"
        )
        X = X.fillna(X.median())

    # --- Replace infinities in X with column max/min ---------------------
    inf_mask = ~np.isfinite(X.values)
    if inf_mask.any():
        n_inf = inf_mask.sum()
        cleaning_actions.append(
            f"Replaced {n_inf} infinite value(s) in predictors with "
            f"column max/min"
        )
        for col in X.columns:
            col_vals = X[col]
            finite_vals = col_vals[np.isfinite(col_vals)]
            if len(finite_vals) == 0:
                X[col] = 0.0
            else:
                X[col] = col_vals.replace(
                    [np.inf], finite_vals.max()
                ).replace([-np.inf], finite_vals.min())

    # --- Drop zero-variance columns --------------------------------------
    zero_var_cols = X.columns[X.std() == 0].tolist()
    if zero_var_cols:
        cleaning_actions.append(
            f"Dropped {len(zero_var_cols)} zero-variance column(s): "
            f"{zero_var_cols}"
        )
        X = X.drop(columns=zero_var_cols)

    if X.shape[1] == 0:
        raise ValueError("No predictor columns remain after cleaning.")

    if len(X) < 10:
        raise ValueError(
            f"Only {len(X)} observations remain after cleaning. "
            f"PRISM requires at least ~10 rows."
        )

    # --- Report -----------------------------------------------------------
    if verbose and cleaning_actions:
        print("DATA CLEANING")
        print("-" * 60)
        for action in cleaning_actions:
            print(f"  * {action}")
        print(f"  Final dataset: n={len(X)}, p={X.shape[1]}")
        print()

    return X, y


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PRISMRegressor(BaseEstimator, RegressorMixin):
    """
    PRISM regression: automatic transformation discovery with
    exact sequential variance decomposition.

    Parameters
    ----------
    m : int, default=10
        Coordinate descent iterations after each variable addition in
        Step 2.  Higher values reduce multivariate refinement (MR).
        m=10 gives MR < 0.1 %% (publication quality).
    alpha : float, default=0.05
        Significance level for the F-test stopping rule.
    interaction_penalty : float, default=5.0
        BIC complexity multiplier for interaction terms (k x penalty).
    max_iterations : int, default=100
        Maximum iterations in Step 3 (final convergence).
    convergence_tolerance : float, default=1e-8
        R-squared change threshold for convergence.
    """

    def __init__(
        self,
        m=10,
        alpha=0.05,
        interaction_penalty=5.0,
        max_iterations=100,
        convergence_tolerance=1e-8,
    ):
        self.m = m
        self.alpha = alpha
        self.interaction_penalty = interaction_penalty
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance

        # --- attributes set during fit ---
        self.step1_results_ = None
        self.step2_results_ = None
        self.step3_results_ = None
        self.step4_results_ = None

        self.selected_features_ = None
        self.transform_dict_ = None
        self.coefficients_ = None
        self.intercepts_ = None
        self.interactions_ = None

        self.n_selected_ = None
        self.incremental_r2_ = None
        self.mr_ = None
        self.r2_ = None
        self.runtime_ = None

        self._y_mean = None          # needed for R² calc in score()
        self._is_fitted = False

    # ---- public interface ------------------------------------------------

    def fit(self, X, y, include_interactions=True, verbose=True):
        """
        Fit PRISM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.  Can be a DataFrame, numpy array, or list.
            Missing values and non-numeric columns are handled
            automatically.
        y : array-like of shape (n_samples,)
            Response variable.
        include_interactions : bool, default=True
            Whether to run Phase 2 (interaction testing).
        verbose : bool, default=True
            Print progress.

        Returns
        -------
        self
        """
        t0 = time.time()

        # Clean / validate ------------------------------------------------
        X, y = _validate_and_clean(X, y, verbose=verbose)
        self._y_mean = y.mean()
        n, p = X.shape

        if verbose:
            print("=" * 70)
            print("PRISM REGRESSION ANALYSIS")
            print("=" * 70)
            print(f"  Dataset : n={n}, p={p}")
            print(f"  m={self.m}  alpha={self.alpha}  "
                  f"interaction_penalty={self.interaction_penalty}")
            print()

        # Phase 1, Step 1 -------------------------------------------------
        if verbose:
            print("STEP 1: TRANSFORMATION SCREENING")
            print("-" * 70)
        self.step1_results_ = self._step1_screening(X, y, verbose)

        # Phase 1, Step 2 -------------------------------------------------
        if verbose:
            print("\nSTEP 2: SEQUENTIAL SELECTION (m={})".format(self.m))
            print("-" * 70)
        self.step2_results_ = self._step2_sequential(X, y, verbose)

        # Phase 1, Step 3 -------------------------------------------------
        if verbose:
            print("\nSTEP 3: FINAL CONVERGENCE")
            print("-" * 70)
        self.step3_results_ = self._step3_convergence(X, y, verbose)

        # Phase 2, Step 4 -------------------------------------------------
        if include_interactions:
            if verbose:
                print("\nSTEP 4: INTERACTION TESTING (BIC k×{})".format(
                    self.interaction_penalty))
                print("-" * 70)
            self.step4_results_ = self._step4_interactions(X, y, verbose)
        else:
            self.step4_results_ = {
                'interactions_tested': 0,
                'interactions_selected': [],
                'interaction_results': pd.DataFrame(),
            }

        # Compile ----------------------------------------------------------
        self._compile(X, y, include_interactions, verbose)
        self.runtime_ = time.time() - t0

        if verbose:
            self._print_summary()

        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Predict response for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Must contain columns with the same names as the training data
            (when a DataFrame) or the same number of columns (when an array).

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = self._coerce_X(X)
        return self._predict_internal(X)

    def score(self, X, y):
        """Return R² on (X, y)."""
        self._check_fitted()
        X = self._coerce_X(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self._predict_internal(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def get_variance_attribution(self):
        """Return the variance attribution table as a DataFrame."""
        self._check_fitted()
        return self._attribution_df.copy()

    def plot_results(self, X, y, figsize=(15, 10)):
        """
        Create diagnostic plots: one subplot per selected variable
        (scatter + fitted curve) plus a predicted-vs-actual plot.
        """
        self._check_fitted()
        X = self._coerce_X(X)
        y = np.asarray(y, dtype=np.float64).ravel()

        feats = self.selected_features_
        n_plots = len(feats) + 1
        ncols = 3
        nrows = (n_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()

        for idx, feat in enumerate(feats):
            ax = axes[idx]
            xv = X[feat].values
            xt = apply_transform(xv, self.transform_dict_[feat])
            coef = self.coefficients_[feat]
            intc = self.intercepts_[feat]
            y_fit = coef * xt + intc

            order = np.argsort(xv)
            ax.scatter(xv, y, alpha=0.25, s=8)
            ax.plot(xv[order], y_fit[order], 'r-', lw=2)
            ax.set_title(f"{feat} ({self.transform_dict_[feat]})")
            ax.set_xlabel(feat)
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.3)

        # Predicted vs actual
        ax = axes[len(feats)]
        y_pred = self._predict_internal(X)
        ax.scatter(y, y_pred, alpha=0.25, s=8)
        lo = min(y.min(), y_pred.min())
        hi = max(y.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Predicted vs Actual (R²={self.r2_:.4f})")
        ax.grid(True, alpha=0.3)

        for idx in range(len(feats) + 1, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig

    # ---- Step implementations --------------------------------------------

    def _step1_screening(self, X, y, verbose):
        """Test all transforms for each variable on original y."""
        rows = []
        for feat in X.columns:
            best_r2, best_tf = -1.0, None
            for tf in TRANSFORM_TYPES:
                xt = apply_transform(X[feat].values, tf).reshape(-1, 1)
                # Guard against constant columns after transform
                if np.std(xt) == 0:
                    continue
                mdl = LinearRegression().fit(xt, y)
                r2 = mdl.score(xt, y)
                if r2 > best_r2:
                    best_r2, best_tf = r2, tf
            rows.append({'Feature': feat, 'Transform': best_tf, 'R2': best_r2})

        df = pd.DataFrame(rows).sort_values('R2', ascending=False)

        if verbose:
            print()
            for _, row in df.iterrows():
                print(f"  {row['Feature']:20s}  {row['Transform']:14s}  "
                      f"R²={row['R2']:.4f}  ({row['R2']*100:.1f}%)")
            print()

        return {
            'results_df': df,
            'feature_order': df['Feature'].tolist(),
        }

    def _step2_sequential(self, X, y, verbose):
        """Sequential forward selection with transformation retesting."""
        n = len(y)
        feature_order = self.step1_results_['feature_order']

        selected = []
        transforms = {}
        coefs = {}
        intercepts = {}
        rounds = []
        current_r2 = 0.0

        if verbose:
            print()

        for rnd, feat in enumerate(feature_order, 1):
            # Current predictions / residuals
            if selected:
                preds = self._calc_preds(X, selected, transforms, coefs,
                                         intercepts)
                residuals = y.values - preds
                ss_res = np.sum((y.values - preds) ** 2)
                ss_tot = np.sum((y.values - y.mean()) ** 2)
                current_r2 = 1.0 - ss_res / ss_tot
            else:
                residuals = y.values - y.mean()
                current_r2 = 0.0

            r2_before = current_r2

            # Test every transform on residuals
            best_f, best_tf = -1.0, None
            for tf in TRANSFORM_TYPES:
                xt = apply_transform(X[feat].values, tf).reshape(-1, 1)
                if np.std(xt) == 0:
                    continue
                mdl = LinearRegression().fit(xt, residuals)
                ss_r = np.sum((residuals - mdl.predict(xt)) ** 2)
                ss_t = np.sum((residuals - residuals.mean()) ** 2)
                if ss_t == 0:
                    continue
                r2_resid = 1.0 - ss_r / ss_t
                r2_inc = r2_resid * (1.0 - current_r2)
                denom = 1.0 - current_r2 - r2_inc
                if denom > 0:
                    f_stat = (r2_inc / 1.0) / (
                        denom / max(n - len(selected) - 2, 1)
                    )
                else:
                    f_stat = 0.0
                if f_stat > best_f:
                    best_f, best_tf = f_stat, tf

            # Stopping rule
            f_crit = stats.f.ppf(
                1 - self.alpha, 1, max(n - len(selected) - 2, 1)
            )
            if best_f < f_crit:
                if verbose:
                    print(f"  Round {rnd}: {feat} ({best_tf})  "
                          f"F={best_f:.1f} < {f_crit:.1f} -> STOP")
                break

            # Add variable
            selected.append(feat)
            transforms[feat] = best_tf
            xt = apply_transform(X[feat].values, best_tf).reshape(-1, 1)
            mdl = LinearRegression().fit(xt, residuals)
            coefs[feat] = mdl.coef_[0]
            intercepts[feat] = mdl.intercept_

            # Coordinate descent (m iterations)
            self._coord_descent(X, y.values, selected, transforms,
                                coefs, intercepts, self.m)

            # R² after m iterations
            preds = self._calc_preds(X, selected, transforms, coefs,
                                     intercepts)
            ss_res = np.sum((y.values - preds) ** 2)
            ss_tot = np.sum((y.values - y.mean()) ** 2)
            new_r2 = 1.0 - ss_res / ss_tot
            r2_gain = new_r2 - r2_before

            rounds.append({
                'Round': rnd, 'Feature': feat, 'Transform': best_tf,
                'F-statistic': best_f, 'R2_gain': r2_gain,
                'Cumulative_R2': new_r2,
            })

            if verbose:
                print(f"  Round {rnd}: {feat:20s} ({best_tf:12s})  "
                      f"ΔR²={r2_gain*100:6.2f}%  "
                      f"Cum={new_r2*100:6.2f}%  F={best_f:.0f}")

        step2_r2 = new_r2 if selected else 0.0

        return {
            'selected_features': selected,
            'transform_dict': transforms,
            'coefficients': dict(coefs),
            'intercepts': dict(intercepts),
            'round_results': pd.DataFrame(rounds),
            'final_r2': step2_r2,
        }

    def _step3_convergence(self, X, y, verbose):
        """Coordinate descent until R² stabilises."""
        selected = self.step2_results_['selected_features']
        transforms = self.step2_results_['transform_dict']
        coefs = dict(self.step2_results_['coefficients'])
        intercepts = dict(self.step2_results_['intercepts'])

        step2_coefs = dict(coefs)

        preds = self._calc_preds(X, selected, transforms, coefs, intercepts)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        r2_prev = 1.0 - np.sum((y.values - preds) ** 2) / ss_tot

        iters = 0
        for it in range(self.max_iterations):
            self._coord_descent(X, y.values, selected, transforms,
                                coefs, intercepts, 1)
            preds = self._calc_preds(X, selected, transforms, coefs,
                                     intercepts)
            r2_now = 1.0 - np.sum((y.values - preds) ** 2) / ss_tot
            if abs(r2_now - r2_prev) < self.convergence_tolerance:
                iters = it + 1
                break
            r2_prev = r2_now
        else:
            iters = self.max_iterations

        final_r2 = r2_now
        mr = final_r2 - self.step2_results_['final_r2']

        # Stability
        stab = []
        for f in selected:
            old = step2_coefs[f]
            new = coefs[f]
            pct = abs((new - old) / old) * 100 if old != 0 else 0.0
            stab.append({'Feature': f, 'Step2': old, 'Step3': new,
                         'Change_pct': pct})

        if verbose:
            print(f"\n  Step 2 R² : {self.step2_results_['final_r2']*100:.4f}%")
            print(f"  Final R²  : {final_r2*100:.4f}%")
            print(f"  MR        : {mr*100:.4f}%")
            print(f"  Iterations: {iters}")
            if stab:
                avg = np.mean([s['Change_pct'] for s in stab])
                mx = max(s['Change_pct'] for s in stab)
                print(f"  Coef stability: avg {avg:.1f}%, max {mx:.1f}%")
            print()

        return {
            'final_r2': final_r2,
            'mr': mr,
            'convergence_iterations': iters,
            'coefficients': coefs,
            'intercepts': intercepts,
            'coefficient_stability': pd.DataFrame(stab),
        }

    def _step4_interactions(self, X, y, verbose):
        """Test two-way interactions with BIC k×penalty."""
        selected = self.step2_results_['selected_features']
        transforms = self.step2_results_['transform_dict']
        base_coefs = self.step3_results_['coefficients']
        base_ints = self.step3_results_['intercepts']

        n = len(y)
        preds_base = self._calc_preds(X, selected, transforms,
                                      base_coefs, base_ints)
        resid_base = y.values - preds_base
        rss_base = np.sum(resid_base ** 2)
        k_base = len(selected) * 2
        bic_base = n * np.log(rss_base / n) + k_base * np.log(n)

        results = []
        chosen = []
        tested = 0

        for i, fj in enumerate(selected):
            for fk in selected[i + 1:]:
                tested += 1
                xj = apply_transform(X[fj].values, transforms[fj])
                xk = apply_transform(X[fk].values, transforms[fk])
                z = xj * xk

                best_bic, best_tf, best_dbic = np.inf, None, 0
                best_r2g, best_c, best_i = 0, None, None

                for tf in TRANSFORM_TYPES:
                    zt = apply_transform(z, tf).reshape(-1, 1)
                    if np.std(zt) == 0:
                        continue
                    mdl = LinearRegression().fit(zt, resid_base)
                    preds_new = preds_base + mdl.predict(zt).ravel()
                    rss_new = np.sum((y.values - preds_new) ** 2)
                    ss_tot = np.sum((y.values - y.mean()) ** 2)
                    r2_new = 1.0 - rss_new / ss_tot
                    r2_base = 1.0 - rss_base / ss_tot
                    r2g = r2_new - r2_base
                    k_new = k_base + self.interaction_penalty * 2
                    bic_new = n * np.log(rss_new / n) + k_new * np.log(n)
                    dbic = bic_base - bic_new

                    if bic_new < best_bic:
                        best_bic = bic_new
                        best_tf = tf
                        best_dbic = dbic
                        best_r2g = r2g
                        best_c = mdl.coef_[0]
                        best_i = mdl.intercept_

                results.append({
                    'Interaction': f"{fj} × {fk}",
                    'Transform': best_tf,
                    'R2_gain': best_r2g,
                    'Delta_BIC': best_dbic,
                    'Selected': best_dbic > 0,
                })
                if best_dbic > 0:
                    chosen.append({
                        'feature_j': fj, 'feature_k': fk,
                        'transform': best_tf,
                        'coefficient': best_c, 'intercept': best_i,
                        'r2_gain': best_r2g, 'delta_bic': best_dbic,
                    })

        if verbose:
            print(f"\n  Tested: {tested}   Selected: {len(chosen)}")
            if chosen:
                for c in chosen:
                    print(f"    {c['feature_j']} × {c['feature_k']} "
                          f"({c['transform']})  "
                          f"ΔR²={c['r2_gain']*100:.2f}%  "
                          f"ΔBIC={c['delta_bic']:.1f}")
            else:
                print("    None passed BIC threshold")
            print()

        return {
            'interactions_tested': tested,
            'interactions_selected': chosen,
            'interaction_results': pd.DataFrame(results),
        }

    # ---- internal helpers ------------------------------------------------

    def _coord_descent(self, X, y_arr, selected, transforms, coefs,
                       intercepts, n_iters):
        """Run *n_iters* rounds of coordinate descent in-place."""
        for _ in range(n_iters):
            for feat in selected:
                preds_other = np.zeros(len(y_arr))
                for other in selected:
                    if other != feat:
                        xt = apply_transform(X[other].values,
                                             transforms[other])
                        preds_other += coefs[other] * xt + intercepts[other]
                resid = y_arr - preds_other
                xt = apply_transform(X[feat].values,
                                     transforms[feat]).reshape(-1, 1)
                mdl = LinearRegression().fit(xt, resid)
                coefs[feat] = mdl.coef_[0]
                intercepts[feat] = mdl.intercept_

    @staticmethod
    def _calc_preds(X, selected, transforms, coefs, intercepts):
        preds = np.zeros(len(X))
        for feat in selected:
            xt = apply_transform(X[feat].values, transforms[feat])
            preds += coefs[feat] * xt + intercepts[feat]
        return preds

    def _predict_internal(self, X):
        preds = self._calc_preds(
            X, self.selected_features_, self.transform_dict_,
            self.coefficients_, self.intercepts_,
        )
        if self.interactions_:
            for inter in self.interactions_:
                xj = apply_transform(
                    X[inter['feature_j']].values,
                    self.transform_dict_[inter['feature_j']],
                )
                xk = apply_transform(
                    X[inter['feature_k']].values,
                    self.transform_dict_[inter['feature_k']],
                )
                z = apply_transform(xj * xk, inter['transform'])
                preds += inter['coefficient'] * z + inter['intercept']
        return preds

    def _compile(self, X, y, include_interactions, verbose):
        selected = self.step2_results_['selected_features']
        transforms = self.step2_results_['transform_dict']

        self.selected_features_ = list(selected)
        self.transform_dict_ = dict(transforms)
        self.coefficients_ = dict(self.step3_results_['coefficients'])
        self.intercepts_ = dict(self.step3_results_['intercepts'])
        self.interactions_ = (
            self.step4_results_['interactions_selected']
            if include_interactions else []
        )
        self.n_selected_ = len(selected)
        self.mr_ = self.step3_results_['mr']

        # Incremental R²
        rr = self.step2_results_['round_results']
        self.incremental_r2_ = (
            rr['R2_gain'].tolist() if len(rr) > 0 else []
        )

        # Final R²
        preds = self._predict_internal(X)
        ss_res = np.sum((y.values - preds) ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        self.r2_ = 1.0 - ss_res / ss_tot

        # OLS baseline (linear, same features)
        X_lin = X[selected]
        ols = LinearRegression().fit(X_lin, y)
        self._ols_r2 = ols.score(X_lin, y)

        # Attribution table
        rows = []
        for _, row in rr.iterrows():
            rows.append({
                'Variable': row['Feature'],
                'Transform': row['Transform'],
                'Incremental_R2': row['R2_gain'],
                'Cumulative_R2': row['Cumulative_R2'],
            })
        rows.append({
            'Variable': 'Multivariate Refinement',
            'Transform': '-',
            'Incremental_R2': self.mr_,
            'Cumulative_R2': self.step3_results_['final_r2'],
        })
        if self.interactions_:
            cum = self.step3_results_['final_r2']
            for inter in self.interactions_:
                cum += inter['r2_gain']
                rows.append({
                    'Variable': f"{inter['feature_j']} × {inter['feature_k']}",
                    'Transform': inter['transform'],
                    'Incremental_R2': inter['r2_gain'],
                    'Cumulative_R2': cum,
                })
        self._attribution_df = pd.DataFrame(rows)

    def _print_summary(self):
        print()
        print("=" * 70)
        print("FINAL MODEL SUMMARY")
        print("=" * 70)

        print("\nVariance Attribution:")
        print("-" * 70)
        df = self._attribution_df.copy()
        df['Incr %'] = df['Incremental_R2'].apply(lambda v: f"{v*100:.2f}%")
        df['Cum %'] = df['Cumulative_R2'].apply(lambda v: f"{v*100:.2f}%")
        for _, row in df.iterrows():
            print(f"  {row['Variable']:30s}  {row['Transform']:14s}  "
                  f"{row['Incr %']:>8s}  {row['Cum %']:>8s}")

        print(f"\nModel Comparison:")
        print(f"  OLS R² (linear only)   : {self._ols_r2*100:.2f}%")
        print(f"  PRISM R² (final)       : {self.r2_*100:.2f}%")
        print(f"  Improvement over OLS   : "
              f"{(self.r2_ - self._ols_r2)*100:.2f} pp")

        print(f"\n  Variables selected     : {self.n_selected_}")
        n_int = len(self.interactions_) if self.interactions_ else 0
        print(f"  Interactions selected  : {n_int}")
        print(f"  MR                     : {self.mr_*100:.4f}%")
        print(f"  Runtime                : {self.runtime_:.2f}s")
        print("=" * 70)

    def _coerce_X(self, X):
        """Ensure X is a DataFrame with the right columns."""
        if isinstance(X, np.ndarray):
            if self.selected_features_ and X.shape[1] >= len(
                    self.selected_features_):
                # Attempt to match by position if no column names
                cols = [f"X{i}" for i in range(X.shape[1])]
                X = pd.DataFrame(X, columns=cols)
            else:
                X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "Model has not been fitted. Call .fit(X, y) first."
            )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def fit_prism(X, y, m=10, include_interactions=True, verbose=True):
    """
    One-liner convenience function.

    Parameters
    ----------
    X : array-like
        Predictors.
    y : array-like
        Response.
    m : int
        Coordinate descent iterations.
    include_interactions : bool
        Run Phase 2?
    verbose : bool
        Print progress?

    Returns
    -------
    PRISMRegressor
        Fitted model.
    """
    mdl = PRISMRegressor(m=m)
    mdl.fit(X, y, include_interactions=include_interactions, verbose=verbose)
    return mdl
