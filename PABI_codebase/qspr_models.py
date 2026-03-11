#!/usr/bin/env python3
"""
qspr_models.py - QSPR Regression Models for PABI Study

Implements:
    1. Univariate linear regression (PABI vs. each property)
    2. Multivariate linear regression (MLR) with stepwise selection
    3. Cross-validation (5-fold and LOO)
    4. External test set validation
    5. Descriptor comparison (PABI vs. MR, TPSA, n_AR, J, MW)

Author: [Your Name]
License: MIT
"""

import numpy as np
import pandas as pd
from itertools import combinations
import warnings
import os
import json

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. Using numpy-based implementations.")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Numpy-based linear regression (fallback when sklearn is unavailable)
# ---------------------------------------------------------------------------

def _ols_fit(X, y):
    """Ordinary least squares via normal equations."""
    X_b = np.column_stack([np.ones(len(X)), X]) if X.ndim == 1 else \
          np.column_stack([np.ones(X.shape[0]), X])
    theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
    return theta[0], theta[1:]  # intercept, coefficients


def _ols_predict(X, intercept, coefficients):
    """Predict using OLS parameters."""
    if X.ndim == 1:
        return intercept + coefficients[0] * X
    return intercept + X @ coefficients


def _r_squared(y_true, y_pred):
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def _rmse(y_true, y_pred):
    """Root mean square error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def _mae(y_true, y_pred):
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


# ---------------------------------------------------------------------------
# Univariate Linear Regression
# ---------------------------------------------------------------------------

def univariate_regression(x, y, confidence=0.95):
    """
    Perform univariate linear regression with full statistics.

    Parameters
    ----------
    x : array-like
        Predictor variable (e.g., PABI values).
    y : array-like
        Response variable (e.g., melting point).
    confidence : float
        Confidence level for intervals (default 0.95).

    Returns
    -------
    dict
        Regression statistics including R^2, Q^2_LOO, slope, intercept,
        slope_CI, intercept_CI, RMSE, MAE, F-statistic, p-value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    # Fit
    intercept, coeffs = _ols_fit(x, y)
    slope = coeffs[0]
    y_pred = intercept + slope * x

    # R-squared
    r2 = _r_squared(y, y_pred)

    # LOO cross-validation Q^2
    q2_loo = _loo_cross_validation(x.reshape(-1, 1), y)

    # Standard errors
    residuals = y - y_pred
    mse = np.sum(residuals ** 2) / (n - 2)
    se_slope = np.sqrt(mse / np.sum((x - np.mean(x)) ** 2))
    se_intercept = np.sqrt(mse * (1/n + np.mean(x)**2 / np.sum((x - np.mean(x))**2)))

    # t-distribution critical value
    if HAS_SCIPY:
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 2)
    else:
        t_crit = 1.96  # Approximation for large n

    slope_ci = (slope - t_crit * se_slope, slope + t_crit * se_slope)
    intercept_ci = (intercept - t_crit * se_intercept, intercept + t_crit * se_intercept)

    # F-statistic
    f_stat = (r2 / 1) / ((1 - r2) / (n - 2)) if r2 < 1 else np.inf
    if HAS_SCIPY:
        p_value = 1 - stats.f.cdf(f_stat, 1, n - 2)
    else:
        p_value = 0.0 if f_stat > 10 else None

    # RMSE, MAE
    rmse = _rmse(y, y_pred)
    mae_val = _mae(y, y_pred)

    return {
        'n': n,
        'r_squared': r2,
        'q2_loo': q2_loo,
        'slope': slope,
        'slope_se': se_slope,
        'slope_ci': slope_ci,
        'intercept': intercept,
        'intercept_se': se_intercept,
        'intercept_ci': intercept_ci,
        'rmse': rmse,
        'mae': mae_val,
        'f_statistic': f_stat,
        'p_value': p_value,
        'residuals': residuals,
        'y_pred': y_pred,
    }


def _loo_cross_validation(X, y):
    """
    Leave-one-out cross-validation R-squared.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Predictor matrix.
    y : np.ndarray of shape (n,)
        Response variable.

    Returns
    -------
    float
        Q^2_LOO value.
    """
    n = len(y)
    press = 0.0  # Predicted residual sum of squares

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        y_test = y[i]

        intercept, coeffs = _ols_fit(X_train, y_train)
        y_pred = _ols_predict(X_test, intercept, coeffs)
        press += (y_test - y_pred[0]) ** 2

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - press / ss_tot if ss_tot > 0 else 0


def kfold_cross_validation(X, y, k=5, random_state=42):
    """
    K-fold cross-validation.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
    y : np.ndarray
    k : int
        Number of folds (default 5).
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Q^2_kfold, fold_r2_values, std.
    """
    n = len(y)
    indices = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    fold_size = n // k
    fold_r2s = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        test_idx = indices[start:end]
        train_idx = np.setdiff1d(indices, test_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        intercept, coeffs = _ols_fit(X_train, y_train)
        y_pred = _ols_predict(X_test, intercept, coeffs)
        r2 = _r_squared(y_test, y_pred)
        fold_r2s.append(r2)

    return {
        'q2_kfold': np.mean(fold_r2s),
        'fold_r2s': fold_r2s,
        'std': np.std(fold_r2s),
    }


# ---------------------------------------------------------------------------
# Multivariate Linear Regression with Stepwise Selection
# ---------------------------------------------------------------------------

def compute_vif(X):
    """
    Compute Variance Inflation Factors for each predictor.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Predictor matrix.

    Returns
    -------
    np.ndarray of shape (p,)
        VIF values for each predictor.
    """
    p = X.shape[1]
    vif = np.zeros(p)

    for i in range(p):
        mask = np.ones(p, dtype=bool)
        mask[i] = False
        X_other = X[:, mask]
        y_i = X[:, i]

        intercept, coeffs = _ols_fit(X_other, y_i)
        y_pred = _ols_predict(X_other, intercept, coeffs)
        r2 = _r_squared(y_i, y_pred)

        vif[i] = 1 / (1 - r2) if r2 < 1 else np.inf

    return vif


def forward_stepwise_selection(X, y, feature_names, max_features=None,
                                vif_threshold=5.0, significance=0.05):
    """
    Forward stepwise selection with VIF monitoring.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Full predictor matrix.
    y : np.ndarray
        Response variable.
    feature_names : list of str
        Names of features (columns of X).
    max_features : int, optional
        Maximum number of features to select.
    vif_threshold : float
        Maximum allowable VIF (default 5.0).
    significance : float
        p-value threshold for inclusion (default 0.05).

    Returns
    -------
    dict
        Selected features, model statistics, VIF values.
    """
    p = X.shape[1]
    if max_features is None:
        max_features = p

    selected = []
    remaining = list(range(p))
    selection_history = []

    for step in range(max_features):
        best_r2 = -np.inf
        best_feature = None

        for feat in remaining:
            candidate = selected + [feat]
            X_cand = X[:, candidate]

            intercept, coeffs = _ols_fit(X_cand, y)
            y_pred = _ols_predict(X_cand, intercept, coeffs)
            r2 = _r_squared(y, y_pred)

            # Check VIF
            if len(candidate) > 1:
                vifs = compute_vif(X_cand)
                if np.any(vifs > vif_threshold):
                    continue

            if r2 > best_r2:
                best_r2 = r2
                best_feature = feat

        if best_feature is None:
            break

        # Check if improvement is significant
        if selected:
            X_current = X[:, selected]
            intercept_c, coeffs_c = _ols_fit(X_current, y)
            y_pred_c = _ols_predict(X_current, intercept_c, coeffs_c)
            r2_current = _r_squared(y, y_pred_c)
            delta_r2 = best_r2 - r2_current

            if delta_r2 < 0.01:  # Minimal improvement threshold
                break

        selected.append(best_feature)
        remaining.remove(best_feature)

        selection_history.append({
            'step': step + 1,
            'feature': feature_names[best_feature],
            'r2': best_r2,
        })

    # Final model statistics
    X_final = X[:, selected]
    intercept, coeffs = _ols_fit(X_final, y)
    y_pred = _ols_predict(X_final, intercept, coeffs)
    r2 = _r_squared(y, y_pred)

    # VIF for final model
    vifs = compute_vif(X_final) if len(selected) > 1 else np.array([1.0])

    # Cross-validation
    cv_results = kfold_cross_validation(X_final, y, k=5)
    q2_loo = _loo_cross_validation(X_final, y)

    # Residuals
    residuals = y - y_pred
    n = len(y)
    p_model = len(selected) + 1  # +1 for intercept
    mse = np.sum(residuals ** 2) / (n - p_model)

    # Standard errors of coefficients
    X_b = np.column_stack([np.ones(n), X_final])
    try:
        cov_matrix = mse * np.linalg.inv(X_b.T @ X_b)
        se = np.sqrt(np.diag(cov_matrix))
        se_intercept = se[0]
        se_coeffs = se[1:]
    except np.linalg.LinAlgError:
        se_intercept = np.nan
        se_coeffs = np.full(len(selected), np.nan)

    # t-values and p-values
    t_intercept = intercept / se_intercept if se_intercept > 0 else np.inf
    t_coeffs = coeffs / se_coeffs

    if HAS_SCIPY:
        p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), n - p_model))
        p_coeffs = 2 * (1 - stats.t.cdf(np.abs(t_coeffs), n - p_model))
        t_crit = stats.t.ppf(0.975, n - p_model)
    else:
        p_intercept = 0.0
        p_coeffs = np.zeros(len(selected))
        t_crit = 1.96

    # Confidence intervals
    ci_intercept = (intercept - t_crit * se_intercept,
                    intercept + t_crit * se_intercept)
    ci_coeffs = [(c - t_crit * se, c + t_crit * se)
                 for c, se in zip(coeffs, se_coeffs)]

    # F-statistic
    f_stat = (r2 / len(selected)) / ((1 - r2) / (n - p_model)) if r2 < 1 else np.inf
    if HAS_SCIPY:
        f_pvalue = 1 - stats.f.cdf(f_stat, len(selected), n - p_model)
    else:
        f_pvalue = 0.0

    rmse = _rmse(y, y_pred)
    s = np.sqrt(mse)  # standard error of estimation

    return {
        'selected_features': [feature_names[i] for i in selected],
        'selected_indices': selected,
        'intercept': intercept,
        'coefficients': coeffs.tolist(),
        'se_intercept': se_intercept,
        'se_coefficients': se_coeffs.tolist(),
        't_intercept': t_intercept,
        't_coefficients': t_coeffs.tolist(),
        'p_intercept': p_intercept,
        'p_coefficients': p_coeffs.tolist() if isinstance(p_coeffs, np.ndarray) else p_coeffs,
        'ci_intercept': ci_intercept,
        'ci_coefficients': ci_coeffs,
        'vif': dict(zip([feature_names[i] for i in selected], vifs.tolist())),
        'r_squared': r2,
        'q2_5fold': cv_results['q2_kfold'],
        'q2_loo': q2_loo,
        'rmse': rmse,
        'mae': _mae(y, y_pred),
        'f_statistic': f_stat,
        'f_pvalue': f_pvalue,
        's': s,
        'n': n,
        'p': len(selected),
        'residuals': residuals.tolist(),
        'y_pred': y_pred.tolist(),
        'selection_history': selection_history,
    }


# ---------------------------------------------------------------------------
# External Validation
# ---------------------------------------------------------------------------

def external_validation(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Perform external validation on test set.

    Parameters
    ----------
    X_train, y_train : arrays
        Training data.
    X_test, y_test : arrays
        Test data.

    Returns
    -------
    dict
        External validation statistics.
    """
    intercept, coeffs = _ols_fit(X_train, y_train)
    y_pred_train = _ols_predict(X_train, intercept, coeffs)
    y_pred_test = _ols_predict(X_test, intercept, coeffs)

    r2_train = _r_squared(y_train, y_pred_train)
    r2_ext = _r_squared(y_test, y_pred_test)
    rmse_ext = _rmse(y_test, y_pred_test)
    mae_ext = _mae(y_test, y_pred_test)

    # Pred vs Obs slope and intercept for test set
    slope_po, intercept_po = np.polyfit(y_test, y_pred_test, 1)
    se_slope_po = 0  # simplified

    return {
        'r2_train': r2_train,
        'r2_ext': r2_ext,
        'rmse_ext': rmse_ext,
        'mae_ext': mae_ext,
        'slope_pred_vs_obs': slope_po,
        'intercept_pred_vs_obs': intercept_po,
        'y_pred_test': y_pred_test.tolist() if isinstance(y_pred_test, np.ndarray) else y_pred_test,
        'n_train': len(y_train),
        'n_test': len(y_test),
    }


# ---------------------------------------------------------------------------
# Descriptor Comparison
# ---------------------------------------------------------------------------

def compare_descriptors(df, target_properties, descriptor_columns):
    """
    Compare univariate R^2 values for PABI vs. established descriptors.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with descriptor and property columns.
    target_properties : list of str
        Column names for target properties (e.g., ['mp', 'bp', 'logp_exp']).
    descriptor_columns : list of str
        Column names for descriptors to compare.

    Returns
    -------
    pd.DataFrame
        Comparison table with R^2 values.
    """
    results = []

    for desc in descriptor_columns:
        row = {'descriptor': desc}
        r2_values = []

        for prop in target_properties:
            mask = df[desc].notna() & df[prop].notna()
            x = df.loc[mask, desc].values
            y = df.loc[mask, prop].values

            if len(x) < 3:
                row[prop] = np.nan
                continue

            intercept, coeffs = _ols_fit(x, y)
            y_pred = _ols_predict(x, intercept, coeffs)
            r2 = _r_squared(y, y_pred)
            row[prop] = round(r2, 3)
            r2_values.append(r2)

        row['average_r2'] = round(np.mean(r2_values), 3) if r2_values else np.nan
        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Full QSPR Analysis Pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(df, alpha=0.75, output_dir='results'):
    """
    Run the complete QSPR analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns: pabi, mp, bp, logp_exp, mr, tpsa, balaban_j,
        mw, n_hbd, n_hba, n_rotb, n_atoms, set.
    alpha : float
        Scaling exponent used.
    output_dir : str
        Directory to save results.

    Returns
    -------
    dict
        Complete analysis results.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'alpha': alpha,
        'univariate': {},
        'multivariate': {},
        'external_validation': {},
        'descriptor_comparison': None,
    }

    # Split data
    train = df[df['set'] == 'T'].copy()
    test = df[df['set'] == 'E'].copy()

    target_properties = {
        'mp': 'Melting Point (C)',
        'bp': 'Boiling Point (C)',
        'logp_exp': 'LogP',
    }

    # -----------------------------------------------------------------------
    # 1. Univariate Regression (PABI vs. each property)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("1. UNIVARIATE LINEAR REGRESSION")
    print("=" * 60)

    for prop_col, prop_name in target_properties.items():
        mask = train['pabi'].notna() & train[prop_col].notna()
        x = train.loc[mask, 'pabi'].values
        y = train.loc[mask, prop_col].values

        res = univariate_regression(x, y)
        results['univariate'][prop_col] = {
            k: v for k, v in res.items()
            if k not in ('residuals', 'y_pred')
        }

        print(f"\n  {prop_name}:")
        print(f"    R^2 = {res['r_squared']:.3f}, Q^2_LOO = {res['q2_loo']:.3f}")
        print(f"    Slope = {res['slope']:.1f} (CI: {res['slope_ci'][0]:.1f}, {res['slope_ci'][1]:.1f})")
        print(f"    Intercept = {res['intercept']:.1f}")
        print(f"    RMSE = {res['rmse']:.1f}, MAE = {res['mae']:.1f}")
        print(f"    F = {res['f_statistic']:.1f}, p = {res['p_value']:.2e}")

    # -----------------------------------------------------------------------
    # 2. Multivariate Regression (Stepwise selection)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. MULTIVARIATE LINEAR REGRESSION")
    print("=" * 60)

    # Available descriptors for MLR
    descriptor_cols = ['pabi', 'mr', 'tpsa', 'mw', 'n_hbd', 'n_hba', 'n_rotb', 'n_atoms']
    available_cols = [c for c in descriptor_cols if c in train.columns]

    for prop_col, prop_name in target_properties.items():
        mask = train[available_cols + [prop_col]].notna().all(axis=1)
        X = train.loc[mask, available_cols].values
        y = train.loc[mask, prop_col].values

        mlr = forward_stepwise_selection(X, y, available_cols, max_features=4)
        results['multivariate'][prop_col] = {
            k: v for k, v in mlr.items()
            if k not in ('residuals', 'y_pred')
        }

        print(f"\n  {prop_name}:")
        print(f"    Selected: {', '.join(mlr['selected_features'])}")
        print(f"    R^2 = {mlr['r_squared']:.3f}, Q^2_5fold = {mlr['q2_5fold']:.3f}")
        print(f"    RMSE = {mlr['rmse']:.1f}, s = {mlr['s']:.1f}")
        print(f"    F = {mlr['f_statistic']:.1f}")
        print(f"    VIF: {mlr['vif']}")

        # Build equation string
        eq = f"    {prop_col.upper()} = {mlr['intercept']:.2f}"
        for feat, coeff in zip(mlr['selected_features'], mlr['coefficients']):
            sign = "+" if coeff >= 0 else ""
            eq += f" {sign}{coeff:.3f}*{feat}"
        print(eq)

    # -----------------------------------------------------------------------
    # 3. External Validation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. EXTERNAL VALIDATION")
    print("=" * 60)

    for prop_col, prop_name in target_properties.items():
        mlr = results['multivariate'][prop_col]
        selected = mlr['selected_features']
        sel_cols = [c for c in selected if c in available_cols]

        mask_train = train[sel_cols + [prop_col]].notna().all(axis=1)
        mask_test = test[sel_cols + [prop_col]].notna().all(axis=1)

        X_train = train.loc[mask_train, sel_cols].values
        y_train = train.loc[mask_train, prop_col].values
        X_test = test.loc[mask_test, sel_cols].values
        y_test = test.loc[mask_test, prop_col].values

        if len(X_test) > 0:
            ext = external_validation(X_train, y_train, X_test, y_test)
            results['external_validation'][prop_col] = {
                k: v for k, v in ext.items()
                if k != 'y_pred_test'
            }

            print(f"\n  {prop_name}:")
            print(f"    R^2_ext = {ext['r2_ext']:.3f}")
            print(f"    RMSE_ext = {ext['rmse_ext']:.1f}")
            print(f"    MAE_ext = {ext['mae_ext']:.1f}")
            print(f"    Slope (pred vs obs) = {ext['slope_pred_vs_obs']:.3f}")
        else:
            print(f"\n  {prop_name}: No test data available.")

    # -----------------------------------------------------------------------
    # 4. Descriptor Comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. DESCRIPTOR COMPARISON")
    print("=" * 60)

    comp_descriptors = ['pabi', 'mr', 'tpsa', 'mw', 'n_ar']
    comp_cols = [c for c in comp_descriptors if c in train.columns]
    prop_cols = [c for c in target_properties.keys() if c in train.columns]

    comparison = compare_descriptors(train, prop_cols, comp_cols)
    results['descriptor_comparison'] = comparison.to_dict('records')

    print(comparison.to_string(index=False))

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    results_serializable = json.loads(
        json.dumps(results, default=convert)
    )

    with open(os.path.join(output_dir, 'qspr_results.json'), 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {output_dir}/qspr_results.json")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='QSPR modeling for PABI')
    parser.add_argument('--data', type=str, default='data/dataset_250.csv',
                        help='Input dataset CSV')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='PABI scaling exponent')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    results = run_full_analysis(df, alpha=args.alpha, output_dir=args.output_dir)
