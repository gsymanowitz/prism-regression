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
    interaction_recompete : bool, default=True
        If True, re-evaluate all remaining interaction candidates after
        each addition (optimal but slower, O(p⁴) worst case).  If False,
        walk down the initial screening ranking and re-test each against
        updated residuals (faster, O(p²) worst case).  Set to False for
        large p where interaction testing is computationally expensive.
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
        interaction_recompete=True,
        max_iterations=100,
        convergence_tolerance=1e-8,
    ):
        self.m = m
        self.alpha = alpha
        self.interaction_penalty = interaction_penalty
        self.interaction_recompete = interaction_recompete
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

        self._is_fitted = True

        if verbose:
            self._print_summary()
            self.plot_results(X, y)
            self.plot_prism_chart()
            plt.show()

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
        Create diagnostic plots in two figures:
          Figure 1: Partial residual plots for each selected variable
          Figure 2: Predicted vs actual y values

        Partial residuals for variable j:
            e_j = y - ŷ_{-j}  (prediction from all OTHER variables)
        Plotted against x_j with the fitted component overlaid.
        """
        self._check_fitted()
        X = self._coerce_X(X)
        y = np.asarray(y, dtype=np.float64).ravel()

        feats = self.selected_features_
        ncols = 3
        nrows = (len(feats) + ncols - 1) // ncols

        # --- Figure 1: Partial residual plots ---
        fig1, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()

        y_pred_full = self._predict_internal(X)

        for idx, feat in enumerate(feats):
            ax = axes[idx]
            xv = X[feat].values
            xt = apply_transform(xv, self.transform_dict_[feat])
            coef = self.coefficients_[feat]
            intc = self.intercepts_[feat]
            component_j = coef * xt + intc

            partial_resid = y - (y_pred_full - component_j)

            order = np.argsort(xv)
            ax.scatter(xv, partial_resid, alpha=0.15, s=6,
                       color='steelblue', label='Partial residuals')
            ax.plot(xv[order], component_j[order], 'r-', lw=2,
                    label=f'{self.transform_dict_[feat]}')
            r2_contrib = (self.incremental_r2_[idx] * 100
                          if idx < len(self.incremental_r2_) else 0)
            ax.set_title(f"{feat} ({self.transform_dict_[feat]}, "
                         f"ΔR²={r2_contrib:.1f}%)")
            ax.set_xlabel(feat)
            ax.set_ylabel("Partial residual")
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

        for idx in range(len(feats), len(axes)):
            axes[idx].set_visible(False)

        fig1.suptitle("Partial Residual Plots", fontsize=14, y=1.01)
        fig1.tight_layout()

        # --- Figure 2: Predicted vs Actual ---
        fig2, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.scatter(y, y_pred_full, alpha=0.15, s=6, color='steelblue')
        lo = min(y.min(), y_pred_full.min())
        hi = max(y.max(), y_pred_full.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect fit')
        ax.set_xlabel("Actual y")
        ax.set_ylabel("Predicted y")
        ax.set_title(f"Predicted vs Actual (R²={self.r2_:.4f})")
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        fig2.tight_layout()

        return fig1, fig2



    def plot_prism_chart(self, figsize=(18, 11), ols_baseline=True,
                         title="PRISM Chart",
                         subtitle="Sequential Variance Decomposition "
                                  "and Best-fit Variable Transforms",
                         dataset_name=None, logo_path=None):
        """
        Generate the signature PRISM waterfall chart showing sequential
        variance attribution with transformation mini-curves.

        Variables are displayed in decreasing order of incremental R²
        contribution (largest first).

        Parameters
        ----------
        figsize : tuple, default=(18, 11)
        ols_baseline : bool, default=True
        title : str
        subtitle : str
        dataset_name : str or None
            Shown on a separate line below the subtitle.
        logo_path : str or None
            Path to a logo image (PNG) to display in the top-left corner.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from matplotlib.patches import FancyBboxPatch
        import math

        self._check_fitted()

        # --- Collect data from fitted model ---
        feats = list(self.selected_features_)
        n_sel = len(feats)
        inc_r2_raw = [v * 100 for v in self.incremental_r2_[:n_sel]]
        transforms_raw = [self.transform_dict_[f] for f in feats]
        total_r2 = self.r2_ * 100
        ols_val = self._ols_r2 * 100 if hasattr(self, '_ols_r2') else None

        # Sort variables by decreasing incremental R²
        order = sorted(range(n_sel), key=lambda k: inc_r2_raw[k],
                        reverse=True)
        feats_sorted = [feats[k] for k in order]
        inc_r2 = [inc_r2_raw[k] for k in order]
        transforms = [transforms_raw[k] for k in order]

        variables = feats_sorted + ['TOTAL']
        incremental = inc_r2 + [total_r2]
        all_transforms = transforms + ['']
        cumulative = list(np.cumsum([0.0] + inc_r2))
        num_vars = len(variables)

        # Prism spectrum colours + grey for total
        rainbow = [
            '#c084fc', '#818cf8', '#38bdf8', '#22d3ee',
            '#34d399', '#facc15', '#fb923c', '#f87171',
            '#e879f9', '#a78bfa', '#67e8f9', '#a3e635',
        ]
        colors = [rainbow[i % len(rainbow)] for i in range(n_sel)]
        colors.append('#6b7280')

        y_axis_max = math.ceil(total_r2 / 10) * 10

        # --- Figure ---
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor('white')

        # Logo (top-left)
        if logo_path is not None:
            try:
                from PIL import Image
                logo = Image.open(logo_path)
                logo_ax = fig.add_axes([0.02, 0.84, 0.12, 0.12],
                                       anchor='NW', zorder=1)
                logo_ax.imshow(logo)
                logo_ax.axis('off')
            except Exception:
                pass  # silently skip if logo can't be loaded

        # Title & subtitle & dataset name (each on own line)
        fig.text(0.5, 0.98, title,
                 ha='center', va='top', fontsize=36,
                 fontweight='bold', color='#2d3748')
        fig.text(0.5, 0.93, subtitle,
                 ha='center', va='top', fontsize=17, color='#718096')
        if dataset_name:
            fig.text(0.5, 0.90, dataset_name,
                     ha='center', va='top', fontsize=17, color='#718096')

        ax = fig.add_axes([0.10, 0.10, 0.85, 0.73])

        bar_width = 0.9

        # --- Waterfall bars ---
        for i in range(num_vars):
            bl = i
            if i < n_sel:
                height = inc_r2[i]
                bottom = cumulative[i]
                rect = FancyBboxPatch(
                    (bl, bottom), bar_width, height,
                    boxstyle='round,pad=0.05',
                    facecolor=colors[i], edgecolor='none',
                    alpha=0.92, zorder=2)
                ax.add_patch(rect)
                ax.text(bl + bar_width / 2, bottom + height + 1.5,
                        f'+{height:.1f}%',
                        ha='center', va='bottom', fontsize=13,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor='white',
                                  edgecolor='#e2e8f0', alpha=0.95))
                if i < n_sel - 1:
                    ax.plot([bl + bar_width, i + 1],
                            [cumulative[i + 1], cumulative[i + 1]],
                            'k--', lw=1.8, alpha=0.35, zorder=1)
            else:
                # TOTAL bar
                rect = FancyBboxPatch(
                    (bl, 0), bar_width, total_r2,
                    boxstyle='round,pad=0.05',
                    facecolor=colors[i], edgecolor='none',
                    alpha=0.92, zorder=2)
                ax.add_patch(rect)
                ax.text(bl + bar_width / 2, total_r2 + 1.5,
                        f'{total_r2:.1f}%',
                        ha='center', va='bottom', fontsize=14,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor='#d1d5db',
                                  edgecolor='#9ca3af', alpha=0.95))
                if ols_baseline and ols_val is not None:
                    ax.plot([bl, bl + bar_width],
                            [ols_val, ols_val],
                            'r-', lw=4, zorder=10)
                    ax.text(bl - 0.05, ols_val,
                            f'OLS: {ols_val:.1f}%',
                            ha='right', va='center', fontsize=10,
                            fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.4',
                                      facecolor='#ef4444',
                                      edgecolor='none', alpha=0.95))

        # --- Variable blocks below chart (data coordinates) ---
        block_height_data = 30
        block_bottom_data = -33

        for i, (var, trans, contrib) in enumerate(
                zip(variables, all_transforms, incremental)):
            bl = i
            if i < n_sel:
                block = FancyBboxPatch(
                    (bl, block_bottom_data), bar_width, block_height_data,
                    boxstyle='round,pad=0.05',
                    facecolor=colors[i], edgecolor='none',
                    alpha=0.92, zorder=1)
                ax.add_patch(block)

                ax.text(bl + bar_width / 2,
                        block_bottom_data + block_height_data * 0.92,
                        var, ha='center', va='top', fontsize=11,
                        fontweight='bold', color='white', zorder=3)
                ax.text(bl + bar_width / 2,
                        block_bottom_data + block_height_data * 0.75,
                        f'({trans})', ha='center', va='top', fontsize=11,
                        color='white', alpha=0.95, zorder=3)
                ax.text(bl + bar_width / 2,
                        block_bottom_data + block_height_data * 0.58,
                        f'+{contrib:.1f}%', ha='center', va='top',
                        fontsize=10.5, fontweight='bold', color='white',
                        zorder=3,
                        bbox=dict(boxstyle='round,pad=0.35',
                                  facecolor='black', alpha=0.18,
                                  edgecolor='none'))

                # Transformation curve in data coords
                cx = np.linspace(0, 1, 100)
                if trans == 'Linear':
                    cy = cx
                elif trans == 'Inverse':
                    cy = 1 / (cx * 10 + 1)
                    cy = (cy - cy.min()) / (cy.max() - cy.min())
                elif trans == 'Cubic':
                    cy = cx ** 3
                elif trans == 'Sqrt':
                    cy = np.sqrt(cx)
                elif trans == 'Square':
                    cy = cx ** 2
                elif trans in ('Log', 'Logarithmic'):
                    cy = np.log(cx * 10 + 1)
                    cy = (cy - cy.min()) / (cy.max() - cy.min())
                elif trans == 'Exponential':
                    cy = np.exp(cx)
                    cy = (cy - cy.min()) / (cy.max() - cy.min())
                else:
                    cy = cx

                cx_s = bl + 0.15 * bar_width + cx * 0.7 * bar_width
                cy_s = (block_bottom_data + 0.12 * block_height_data
                        + cy * 0.35 * block_height_data)
                ax.plot(cx_s, cy_s, 'w-', lw=2.5, alpha=0.95, zorder=3)

            else:
                # TOTAL block
                block = FancyBboxPatch(
                    (bl, block_bottom_data), bar_width, block_height_data,
                    boxstyle='round,pad=0.05',
                    facecolor=colors[i], edgecolor='none',
                    alpha=0.92, zorder=1)
                ax.add_patch(block)
                ax.text(bl + bar_width / 2,
                        block_bottom_data + block_height_data * 0.92,
                        'TOTAL\nEXPLAINED\nVARIANCE',
                        ha='center', va='top', fontsize=8.5,
                        fontweight='bold', color='white',
                        linespacing=1.3, zorder=3)
                ax.text(bl + bar_width / 2,
                        block_bottom_data + block_height_data * 0.58,
                        f'{contrib:.1f}%', ha='center', va='top',
                        fontsize=12, fontweight='bold', color='white',
                        zorder=3,
                        bbox=dict(boxstyle='round,pad=0.35',
                                  facecolor='black', alpha=0.22,
                                  edgecolor='none'))

        # --- Info text ---
        info_y = block_bottom_data - 4
        lines = [
            'How to read this chart:',
            'Each colored bar shows a variable\'s incremental '
            'contribution to R\u00b2.',
            'The blocks below display each variable\'s transformation '
            'type and contribution percentage, with a curve showing '
            'that transformation\'s shape.',
        ]
        for idx, line in enumerate(lines):
            ax.text(0, info_y - idx * 2.5, line,
                    ha='left', va='top', fontsize=11,
                    color='#4a5568', zorder=3)

        # --- Axis formatting ---
        ax.set_xlim(-0.5, num_vars + 0.5)
        ax.set_ylim(info_y - 10, y_axis_max + 10)
        ax.set_ylabel('Cumulative R\u00b2 (%)', fontsize=15,
                       fontweight='600', color='#4a5568')
        ax.set_xticks([])
        y_ticks = list(range(0, y_axis_max + 1, 10))
        ax.set_yticks(y_ticks)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e0')
        ax.spines['left'].set_bounds(0, y_axis_max)
        for yt in range(10, y_axis_max + 1, 10):
            ax.axhline(y=yt, color='#e2e8f0', ls='-', lw=0.5,
                        alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', labelsize=12, colors='#4a5568')

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
        """Sequential forward selection with full re-competition.

        At each round, ALL remaining variables compete — each is tested
        with all 7 transformations on the current residuals, and the
        variable+transformation pair with the highest F-statistic wins.
        Step 1 ordering is used only for initial screening/ranking,
        not to fix the selection order.
        """
        n = len(y)
        all_features = list(self.step1_results_['feature_order'])

        selected = []
        remaining = set(all_features)
        transforms = {}
        coefs = {}
        intercepts = {}
        rounds = []
        current_r2 = 0.0

        if verbose:
            print()

        for rnd in range(1, len(all_features) + 1):
            # Current predictions / residuals
            if selected:
                preds = self._calc_preds(X, selected, transforms, coefs,
                                         intercepts)
                residuals = y.values - preds
                ss_res = np.sum((y.values - preds) ** 2)
                ss_tot = np.sum((y.values - y.mean()) ** 2)
                current_r2 = 1.0 - ss_res / ss_tot
            else:
                residuals = y.values
                current_r2 = 0.0

            r2_before = current_r2

            # Evaluate ALL remaining variables × ALL transforms
            best_f, best_feat, best_tf = -1.0, None, None
            for feat in remaining:
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
                        best_f, best_feat, best_tf = f_stat, feat, tf

            if best_feat is None:
                break

            # Stopping rule
            f_crit = stats.f.ppf(
                1 - self.alpha, 1, max(n - len(selected) - 2, 1)
            )
            if best_f < f_crit:
                if verbose:
                    print(f"  Round {rnd}: {best_feat:20s} ({best_tf:12s})  "
                          f"F={best_f:.1f} < {f_crit:.1f}  Reject -> STOP")
                break

            # Add winning variable
            selected.append(best_feat)
            remaining.discard(best_feat)
            transforms[best_feat] = best_tf
            xt = apply_transform(
                X[best_feat].values, best_tf).reshape(-1, 1)
            mdl = LinearRegression().fit(xt, residuals)
            coefs[best_feat] = mdl.coef_[0]
            intercepts[best_feat] = mdl.intercept_

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
                'Round': rnd, 'Feature': best_feat,
                'Transform': best_tf,
                'F-statistic': best_f, 'R2_gain': r2_gain,
                'Cumulative_R2': new_r2,
            })

            if verbose:
                print(f"  Round {rnd}: {best_feat:20s} ({best_tf:12s})  "
                      f"ΔR²={r2_gain*100:6.2f}%  "
                      f"Cum={new_r2*100:6.2f}%  F={best_f:.0f}  Add")

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
        """Greedy sequential interaction testing with BIC k×penalty.

        Procedure (mirrors Phase 1 forward selection logic):
        1. Screen all p(p-1)/2 interactions against base residuals
           using RAW variable products with 7 transformations each.
        2. Rank by ΔBIC (best first).
        3. Add the top-ranked interaction if ΔBIC > 0, then run
           coordinate descent on the full model (base + interaction).
        4. Re-evaluate the next candidate against UPDATED residuals.
        5. Stop at the first candidate that fails the BIC test.
        """
        selected = list(self.step2_results_['selected_features'])
        transforms = dict(self.step2_results_['transform_dict'])
        coefs = dict(self.step3_results_['coefficients'])
        intercepts = dict(self.step3_results_['intercepts'])

        n = len(y)
        y_arr = y.values
        ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)

        # --- Phase A: Screen all interactions against base residuals -------
        preds_base = self._calc_preds(X, selected, transforms,
                                      coefs, intercepts)
        resid_base = y_arr - preds_base
        rss_base = np.sum(resid_base ** 2)
        r2_base = 1.0 - rss_base / ss_tot
        k_base = len(selected)
        bic_base = n * np.log(rss_base / n) + k_base * np.log(n)

        screening = []
        tested = 0

        for i, fj in enumerate(selected):
            for fk in selected[i + 1:]:
                tested += 1
                z_raw = X[fj].values * X[fk].values

                best_bic, best_tf, best_dbic = np.inf, None, 0
                best_r2g, best_c, best_i = 0, None, None

                for tf in TRANSFORM_TYPES:
                    zt = apply_transform(z_raw, tf).reshape(-1, 1)
                    if np.std(zt) == 0:
                        continue
                    mdl = LinearRegression().fit(zt, resid_base)
                    preds_new = preds_base + mdl.predict(zt).ravel()
                    rss_new = np.sum((y_arr - preds_new) ** 2)
                    r2_new = 1.0 - rss_new / ss_tot
                    r2g = r2_new - r2_base

                    k_new = k_base + 1
                    bic_new = (n * np.log(rss_new / n)
                               + (k_new * self.interaction_penalty)
                               * np.log(n))
                    dbic = bic_base - bic_new

                    if bic_new < best_bic:
                        best_bic = bic_new
                        best_tf = tf
                        best_dbic = dbic
                        best_r2g = r2g
                        best_c = mdl.coef_[0]
                        best_i = mdl.intercept_

                screening.append({
                    'feature_j': fj, 'feature_k': fk,
                    'transform': best_tf,
                    'coefficient': best_c, 'intercept': best_i,
                    'r2_gain': best_r2g, 'delta_bic': best_dbic,
                })

        # Rank by ΔBIC (largest improvement first)
        screening.sort(key=lambda x: x['delta_bic'], reverse=True)

        # --- Phase B: Greedy sequential addition ----------------------------
        # Two modes controlled by interaction_recompete:
        #   True  = full re-competition (re-evaluate all remaining after each
        #           addition; optimal but O(p⁴) worst case)
        #   False = fixed-rank (walk down screening list, re-test each against
        #           updated residuals; faster, O(p²) worst case)
        chosen = []
        current_interactions = []

        if self.interaction_recompete:
            # --- Full re-competition mode ---
            remaining = [(s['feature_j'], s['feature_k'])
                         for s in screening if s['delta_bic'] > 0]

            while remaining:
                preds_cur = self._calc_preds(X, selected, transforms,
                                             coefs, intercepts)
                for inter in current_interactions:
                    z_raw = (X[inter['feature_j']].values
                             * X[inter['feature_k']].values)
                    zt = apply_transform(z_raw, inter['transform'])
                    preds_cur += (inter['coefficient'] * zt
                                  + inter['intercept'])

                resid_cur = y_arr - preds_cur
                rss_cur = np.sum(resid_cur ** 2)
                r2_cur = 1.0 - rss_cur / ss_tot
                k_cur = k_base + len(current_interactions)
                bic_cur = (n * np.log(rss_cur / n)
                           + (k_cur * self.interaction_penalty)
                           * np.log(n))

                best_overall = None
                for fj, fk in remaining:
                    z_raw = X[fj].values * X[fk].values
                    best_bic, best_tf, best_dbic = np.inf, None, 0
                    best_r2g, best_c, best_i = 0, None, None

                    for tf in TRANSFORM_TYPES:
                        zt = apply_transform(z_raw, tf).reshape(-1, 1)
                        if np.std(zt) == 0:
                            continue
                        mdl = LinearRegression().fit(zt, resid_cur)
                        preds_new = preds_cur + mdl.predict(zt).ravel()
                        rss_new = np.sum((y_arr - preds_new) ** 2)
                        r2_new = 1.0 - rss_new / ss_tot
                        r2g = r2_new - r2_cur
                        k_new = k_cur + 1
                        bic_new = (n * np.log(rss_new / n)
                                   + (k_new * self.interaction_penalty)
                                   * np.log(n))
                        dbic = bic_cur - bic_new
                        if bic_new < best_bic:
                            best_bic = bic_new
                            best_tf = tf
                            best_dbic = dbic
                            best_r2g = r2g
                            best_c = mdl.coef_[0]
                            best_i = mdl.intercept_

                    if (best_overall is None
                            or best_dbic > best_overall['delta_bic']):
                        best_overall = {
                            'feature_j': fj, 'feature_k': fk,
                            'transform': best_tf,
                            'coefficient': best_c, 'intercept': best_i,
                            'r2_gain': best_r2g, 'delta_bic': best_dbic,
                        }

                if best_overall is None or best_overall['delta_bic'] <= 0:
                    break

                current_interactions.append(best_overall)
                chosen.append(best_overall)
                remaining = [
                    (fj, fk) for fj, fk in remaining
                    if not (fj == best_overall['feature_j']
                            and fk == best_overall['feature_k'])]

                self._coord_descent_with_interactions(
                    X, y_arr, selected, transforms, coefs, intercepts,
                    current_interactions, self.m,
                )

        else:
            # --- Fixed-rank mode ---
            for candidate in screening:
                if candidate['delta_bic'] <= 0:
                    break

                fj = candidate['feature_j']
                fk = candidate['feature_k']

                preds_cur = self._calc_preds(X, selected, transforms,
                                             coefs, intercepts)
                for inter in current_interactions:
                    z_raw = (X[inter['feature_j']].values
                             * X[inter['feature_k']].values)
                    zt = apply_transform(z_raw, inter['transform'])
                    preds_cur += (inter['coefficient'] * zt
                                  + inter['intercept'])

                resid_cur = y_arr - preds_cur
                rss_cur = np.sum(resid_cur ** 2)
                r2_cur = 1.0 - rss_cur / ss_tot
                k_cur = k_base + len(current_interactions)
                bic_cur = (n * np.log(rss_cur / n)
                           + (k_cur * self.interaction_penalty)
                           * np.log(n))

                z_raw = X[fj].values * X[fk].values
                best_bic, best_tf, best_dbic = np.inf, None, 0
                best_r2g, best_c, best_i = 0, None, None

                for tf in TRANSFORM_TYPES:
                    zt = apply_transform(z_raw, tf).reshape(-1, 1)
                    if np.std(zt) == 0:
                        continue
                    mdl = LinearRegression().fit(zt, resid_cur)
                    preds_new = preds_cur + mdl.predict(zt).ravel()
                    rss_new = np.sum((y_arr - preds_new) ** 2)
                    r2_new = 1.0 - rss_new / ss_tot
                    r2g = r2_new - r2_cur
                    k_new = k_cur + 1
                    bic_new = (n * np.log(rss_new / n)
                               + (k_new * self.interaction_penalty)
                               * np.log(n))
                    dbic = bic_cur - bic_new
                    if bic_new < best_bic:
                        best_bic = bic_new
                        best_tf = tf
                        best_dbic = dbic
                        best_r2g = r2g
                        best_c = mdl.coef_[0]
                        best_i = mdl.intercept_

                if best_dbic <= 0:
                    break

                inter_info = {
                    'feature_j': fj, 'feature_k': fk,
                    'transform': best_tf,
                    'coefficient': best_c, 'intercept': best_i,
                    'r2_gain': best_r2g, 'delta_bic': best_dbic,
                }
                current_interactions.append(inter_info)
                chosen.append(inter_info)

                self._coord_descent_with_interactions(
                    X, y_arr, selected, transforms, coefs, intercepts,
                    current_interactions, self.m,
                )

        # Build results table for reporting
        results = []
        for s in screening:
            results.append({
                'Interaction': f"{s['feature_j']} × {s['feature_k']}",
                'Transform': s['transform'],
                'R2_gain': s['r2_gain'],
                'Delta_BIC': s['delta_bic'],
                'Selected': any(
                    c['feature_j'] == s['feature_j']
                    and c['feature_k'] == s['feature_k']
                    for c in chosen),
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

    def _coord_descent_with_interactions(self, X, y_arr, selected,
                                          transforms, coefs, intercepts,
                                          interactions, n_iters):
        """Coordinate descent over base variables AND interaction terms.
        Interaction terms use RAW variable products (matching paper)."""
        for _ in range(n_iters):
            # Update base variable coefficients
            for feat in selected:
                preds_other = np.zeros(len(y_arr))
                for other in selected:
                    if other != feat:
                        xt = apply_transform(X[other].values,
                                             transforms[other])
                        preds_other += coefs[other] * xt + intercepts[other]
                for inter in interactions:
                    z_raw = (X[inter['feature_j']].values
                             * X[inter['feature_k']].values)
                    zt = apply_transform(z_raw, inter['transform'])
                    preds_other += inter['coefficient'] * zt + inter['intercept']
                resid = y_arr - preds_other
                xt = apply_transform(X[feat].values,
                                     transforms[feat]).reshape(-1, 1)
                mdl = LinearRegression().fit(xt, resid)
                coefs[feat] = mdl.coef_[0]
                intercepts[feat] = mdl.intercept_

            # Update interaction coefficients
            for inter in interactions:
                preds_other = np.zeros(len(y_arr))
                for feat in selected:
                    xt = apply_transform(X[feat].values, transforms[feat])
                    preds_other += coefs[feat] * xt + intercepts[feat]
                for other_inter in interactions:
                    if other_inter is not inter:
                        z_raw = (X[other_inter['feature_j']].values
                                 * X[other_inter['feature_k']].values)
                        zt = apply_transform(z_raw, other_inter['transform'])
                        preds_other += (other_inter['coefficient'] * zt
                                        + other_inter['intercept'])
                resid = y_arr - preds_other
                z_raw = (X[inter['feature_j']].values
                         * X[inter['feature_k']].values)
                zt = apply_transform(z_raw,
                                     inter['transform']).reshape(-1, 1)
                mdl = LinearRegression().fit(zt, resid)
                inter['coefficient'] = mdl.coef_[0]
                inter['intercept'] = mdl.intercept_

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
                # Interactions use RAW variable products (matching paper)
                z_raw = (X[inter['feature_j']].values
                         * X[inter['feature_k']].values)
                z = apply_transform(z_raw, inter['transform'])
                preds += inter['coefficient'] * z + inter['intercept']
        return preds

    def _compile(self, X, y, include_interactions, verbose):
        selected = self.step2_results_['selected_features']
        transforms = self.step2_results_['transform_dict']

        # Start from step3 converged coefficients
        coefs = dict(self.step3_results_['coefficients'])
        intercepts = dict(self.step3_results_['intercepts'])

        self.selected_features_ = list(selected)
        self.transform_dict_ = dict(transforms)
        self.interactions_ = (
            self.step4_results_['interactions_selected']
            if include_interactions else []
        )

        # If interactions were added, run final convergence on full model
        if self.interactions_:
            self._coord_descent_with_interactions(
                X, y.values, selected, transforms, coefs, intercepts,
                self.interactions_, self.m * 2,
            )
            # Converge until stable
            ss_tot = np.sum((y.values - y.mean()) ** 2)
            for _ in range(self.max_iterations):
                preds_before = self._calc_preds(
                    X, selected, transforms, coefs, intercepts)
                for inter in self.interactions_:
                    z_raw = (X[inter['feature_j']].values
                             * X[inter['feature_k']].values)
                    zt = apply_transform(z_raw, inter['transform'])
                    preds_before += inter['coefficient'] * zt + inter['intercept']
                r2_before = 1.0 - np.sum((y.values - preds_before) ** 2) / ss_tot

                self._coord_descent_with_interactions(
                    X, y.values, selected, transforms, coefs, intercepts,
                    self.interactions_, 1,
                )

                preds_after = self._calc_preds(
                    X, selected, transforms, coefs, intercepts)
                for inter in self.interactions_:
                    z_raw = (X[inter['feature_j']].values
                             * X[inter['feature_k']].values)
                    zt = apply_transform(z_raw, inter['transform'])
                    preds_after += inter['coefficient'] * zt + inter['intercept']
                r2_after = 1.0 - np.sum((y.values - preds_after) ** 2) / ss_tot

                if abs(r2_after - r2_before) < self.convergence_tolerance:
                    break

        self.coefficients_ = coefs
        self.intercepts_ = intercepts
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

        # Variance attribution sorted by contribution (descending)
        print("\nVariance Attribution (sorted by contribution):")
        print("-" * 70)
        df = self._attribution_df.copy()
        # Separate MR row from variable rows
        mr_rows = df[df['Variable'] == 'Multivariate Refinement']
        var_rows = df[df['Variable'] != 'Multivariate Refinement']
        var_rows = var_rows.sort_values('Incremental_R2', ascending=False)

        for _, row in var_rows.iterrows():
            print(f"  {row['Variable']:30s}  {row['Transform']:14s}  "
                  f"{row['Incremental_R2']*100:>8.2f}%")
        for _, row in mr_rows.iterrows():
            print(f"  {row['Variable']:30s}  {row['Transform']:14s}  "
                  f"{row['Incremental_R2']*100:>8.4f}%")

        # Final model parameters and functional forms
        print(f"\nFinal Model Parameters:")
        print("-" * 70)
        total_intercept = sum(self.intercepts_[f]
                              for f in self.selected_features_)
        print(f"  {'Variable':20s}  {'Transform':14s}  {'Coefficient':>12s}  "
              f"{'Intercept':>12s}")
        for feat in self.selected_features_:
            tf = self.transform_dict_[feat]
            coef = self.coefficients_[feat]
            intc = self.intercepts_[feat]
            print(f"  {feat:20s}  {tf:14s}  {coef:>12.6f}  {intc:>12.6f}")
        if self.interactions_:
            for inter in self.interactions_:
                label = f"{inter['feature_j']}×{inter['feature_k']}"
                print(f"  {label:20s}  {inter['transform']:14s}  "
                      f"{inter['coefficient']:>12.6f}  "
                      f"{inter['intercept']:>12.6f}")
                total_intercept += inter['intercept']
        print(f"  {'':20s}  {'':14s}  {'':>12s}  {'----------':>12s}")
        print(f"  {'Global intercept':20s}  {'(sum of αᵢ)':14s}  "
              f"{'':>12s}  {total_intercept:>12.6f}")

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
