#!/usr/bin/env python3
"""
diagnostics.py - Regression Diagnostic Tests for PABI QSPR Models

Implements:
    1. Shapiro-Wilk test for residual normality
    2. Breusch-Pagan test for homoscedasticity
    3. Durbin-Watson statistic for autocorrelation
    4. Cook's distance for influential observations
    5. Leverage (hat) values
    6. Standardized residuals
    7. Williams plot data preparation

Author: [Your Name]
License: MIT
"""

import numpy as np
import pandas as pd
import warnings

try:
    from scipy import stats
    from scipy.stats import shapiro, chi2
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available. Some diagnostic tests will be limited.")


def compute_hat_matrix(X):
    """
    Compute the hat (leverage) matrix H = X(X'X)^{-1}X'.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Design matrix WITHOUT intercept column (will be added).

    Returns
    -------
    np.ndarray of shape (n, n)
        Hat matrix.
    """
    n = X.shape[0]
    X_b = np.column_stack([np.ones(n), X])
    try:
        H = X_b @ np.linalg.inv(X_b.T @ X_b) @ X_b.T
    except np.linalg.LinAlgError:
        H = X_b @ np.linalg.pinv(X_b.T @ X_b) @ X_b.T
    return H


def leverage_values(X):
    """
    Compute leverage (hat) values h_ii for each observation.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Predictor matrix (without intercept).

    Returns
    -------
    np.ndarray of shape (n,)
        Leverage values.
    """
    H = compute_hat_matrix(X)
    return np.diag(H)


def standardized_residuals(residuals, X):
    """
    Compute internally standardized residuals.

    r_i = e_i / (s * sqrt(1 - h_ii))

    Parameters
    ----------
    residuals : np.ndarray of shape (n,)
        Raw residuals (y - y_hat).
    X : np.ndarray of shape (n, p)
        Predictor matrix (without intercept).

    Returns
    -------
    np.ndarray of shape (n,)
        Standardized residuals.
    """
    n = len(residuals)
    p = X.shape[1] + 1  # +1 for intercept
    h = leverage_values(X)
    mse = np.sum(residuals ** 2) / (n - p)
    s = np.sqrt(mse)

    std_res = np.zeros(n)
    for i in range(n):
        denom = s * np.sqrt(max(1 - h[i], 1e-10))
        std_res[i] = residuals[i] / denom

    return std_res


def studentized_residuals(residuals, X):
    """
    Compute externally studentized (deleted) residuals.

    t_i = e_i / (s_{(i)} * sqrt(1 - h_ii))

    where s_{(i)} is computed without observation i.

    Parameters
    ----------
    residuals : np.ndarray
    X : np.ndarray

    Returns
    -------
    np.ndarray
        Studentized residuals.
    """
    n = len(residuals)
    p = X.shape[1] + 1
    h = leverage_values(X)
    mse = np.sum(residuals ** 2) / (n - p)

    stud_res = np.zeros(n)
    for i in range(n):
        # Leave-one-out MSE
        mse_i = (mse * (n - p) - residuals[i] ** 2 / (1 - h[i])) / (n - p - 1)
        mse_i = max(mse_i, 1e-10)
        s_i = np.sqrt(mse_i)
        denom = s_i * np.sqrt(max(1 - h[i], 1e-10))
        stud_res[i] = residuals[i] / denom

    return stud_res


def cooks_distance(residuals, X):
    """
    Compute Cook's distance for each observation.

    D_i = (e_i^2 / (p * MSE)) * (h_ii / (1 - h_ii)^2)

    Parameters
    ----------
    residuals : np.ndarray
    X : np.ndarray

    Returns
    -------
    np.ndarray
        Cook's distance values.
    """
    n = len(residuals)
    p = X.shape[1] + 1
    h = leverage_values(X)
    mse = np.sum(residuals ** 2) / (n - p)

    cooks_d = np.zeros(n)
    for i in range(n):
        if (1 - h[i]) > 1e-10:
            cooks_d[i] = (residuals[i] ** 2 / (p * mse)) * \
                         (h[i] / (1 - h[i]) ** 2)
    return cooks_d


def shapiro_wilk_test(residuals):
    """
    Shapiro-Wilk test for normality of residuals.

    H0: Residuals are normally distributed.
    Reject H0 if p < 0.05.

    Parameters
    ----------
    residuals : np.ndarray

    Returns
    -------
    dict
        W statistic and p-value.
    """
    if not HAS_SCIPY:
        warnings.warn("scipy not available for Shapiro-Wilk test")
        return {'W': None, 'p_value': None, 'normal': None}

    # Shapiro-Wilk has a sample size limit
    if len(residuals) > 5000:
        residuals = np.random.choice(residuals, 5000, replace=False)

    W, p = shapiro(residuals)
    return {
        'W': W,
        'p_value': p,
        'normal': p > 0.05,
    }


def breusch_pagan_test(residuals, X):
    """
    Breusch-Pagan test for homoscedasticity.

    H0: Variance of residuals is constant (homoscedastic).
    Reject H0 if p < 0.10.

    Parameters
    ----------
    residuals : np.ndarray
    X : np.ndarray

    Returns
    -------
    dict
        Chi-squared statistic and p-value.
    """
    n = len(residuals)
    p = X.shape[1]

    # Auxiliary regression: e^2 on X
    e_sq = residuals ** 2
    X_b = np.column_stack([np.ones(n), X])

    try:
        theta = np.linalg.lstsq(X_b, e_sq, rcond=None)[0]
        e_sq_pred = X_b @ theta
        ss_reg = np.sum((e_sq_pred - np.mean(e_sq)) ** 2)
        ss_tot = np.sum((e_sq - np.mean(e_sq)) ** 2)

        if ss_tot > 0:
            r2_aux = ss_reg / ss_tot
        else:
            r2_aux = 0

        bp_stat = n * r2_aux

        if HAS_SCIPY:
            p_value = 1 - chi2.cdf(bp_stat, p)
        else:
            p_value = None

        return {
            'chi2': bp_stat,
            'p_value': p_value,
            'homoscedastic': p_value > 0.10 if p_value is not None else None,
        }
    except Exception:
        return {'chi2': None, 'p_value': None, 'homoscedastic': None}


def durbin_watson(residuals):
    """
    Durbin-Watson statistic for autocorrelation of residuals.

    d approx 2: no autocorrelation
    d < 2: positive autocorrelation
    d > 2: negative autocorrelation

    Acceptable range: 1.5 to 2.5 (conservative: 1.8 to 2.2).

    Parameters
    ----------
    residuals : np.ndarray

    Returns
    -------
    dict
        d statistic and interpretation.
    """
    diff = np.diff(residuals)
    d = np.sum(diff ** 2) / np.sum(residuals ** 2)

    if 1.8 <= d <= 2.2:
        interpretation = 'No significant autocorrelation'
    elif d < 1.8:
        interpretation = 'Possible positive autocorrelation'
    else:
        interpretation = 'Possible negative autocorrelation'

    return {
        'd': d,
        'interpretation': interpretation,
        'acceptable': 1.5 <= d <= 2.5,
    }


def run_all_diagnostics(residuals, X, y_fitted, model_name='Model'):
    """
    Run complete suite of regression diagnostics.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    X : np.ndarray
        Predictor matrix (without intercept).
    y_fitted : np.ndarray
        Fitted values.
    model_name : str
        Name for reporting.

    Returns
    -------
    dict
        Complete diagnostics results.
    """
    n = len(residuals)
    p = X.shape[1] + 1

    print(f"\n  Diagnostics for {model_name}:")
    print(f"  {'='*50}")

    # Leverage
    h = leverage_values(X)
    h_threshold = 3 * p / n

    # Standardized residuals
    std_res = standardized_residuals(residuals, X)

    # Cook's distance
    cooks_d = cooks_distance(residuals, X)
    cooks_threshold = 4 / n

    # Shapiro-Wilk
    sw = shapiro_wilk_test(residuals)
    print(f"  Shapiro-Wilk: W = {sw['W']:.3f}, p = {sw['p_value']:.3f}" if sw['W'] else
          "  Shapiro-Wilk: Not available")

    # Breusch-Pagan
    bp = breusch_pagan_test(residuals, X)
    if bp['chi2'] is not None:
        print(f"  Breusch-Pagan: chi2 = {bp['chi2']:.2f}, p = {bp['p_value']:.3f}")

    # Durbin-Watson
    dw = durbin_watson(residuals)
    print(f"  Durbin-Watson: d = {dw['d']:.2f} ({dw['interpretation']})")

    # Cook's distance
    max_cook = np.max(cooks_d)
    n_influential = np.sum(cooks_d > cooks_threshold)
    print(f"  Cook's D max = {max_cook:.4f} (threshold = {cooks_threshold:.4f})")
    print(f"  Influential observations: {n_influential}")

    # Leverage
    n_high_leverage = np.sum(h > h_threshold)
    print(f"  High leverage points (h > {h_threshold:.4f}): {n_high_leverage}")

    # Outliers (|std residual| > 3)
    n_outliers = np.sum(np.abs(std_res) > 3)
    print(f"  Outliers (|r_i| > 3): {n_outliers}")

    results = {
        'model_name': model_name,
        'n': n,
        'p': p - 1,
        'shapiro_wilk': {
            'W': float(sw['W']) if sw['W'] is not None else None,
            'p_value': float(sw['p_value']) if sw['p_value'] is not None else None,
        },
        'breusch_pagan': {
            'chi2': float(bp['chi2']) if bp['chi2'] is not None else None,
            'p_value': float(bp['p_value']) if bp['p_value'] is not None else None,
        },
        'durbin_watson': {
            'd': float(dw['d']),
            'acceptable': dw['acceptable'],
        },
        'cooks_distance': {
            'max': float(max_cook),
            'threshold': float(cooks_threshold),
            'n_influential': int(n_influential),
        },
        'leverage': {
            'h_values': h.tolist(),
            'threshold': float(h_threshold),
            'n_high_leverage': int(n_high_leverage),
        },
        'standardized_residuals': std_res.tolist(),
        'cooks_d_values': cooks_d.tolist(),
        'y_fitted': y_fitted.tolist() if isinstance(y_fitted, np.ndarray) else y_fitted,
    }

    return results


def qq_plot_data(residuals):
    """
    Prepare data for Q-Q (quantile-quantile) plot.

    Parameters
    ----------
    residuals : np.ndarray

    Returns
    -------
    dict
        theoretical_quantiles, sample_quantiles.
    """
    n = len(residuals)
    sorted_res = np.sort(residuals)

    # Standardize
    mean_r = np.mean(sorted_res)
    std_r = np.std(sorted_res, ddof=1)
    if std_r > 0:
        standardized = (sorted_res - mean_r) / std_r
    else:
        standardized = sorted_res

    # Theoretical quantiles
    if HAS_SCIPY:
        theoretical = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    else:
        # Approximation using inverse normal CDF
        p = np.arange(1, n + 1) / (n + 1)
        theoretical = np.sqrt(2) * _inv_erf(2 * p - 1)

    return {
        'theoretical_quantiles': theoretical.tolist(),
        'sample_quantiles': standardized.tolist(),
    }


def _inv_erf(x):
    """Approximate inverse error function for Q-Q plots without scipy."""
    a = 0.147
    ln_part = np.log(1 - x ** 2)
    inner = (2 / (np.pi * a) + ln_part / 2) ** 2 - ln_part / a
    return np.sign(x) * np.sqrt(np.sqrt(inner) - 2 / (np.pi * a) - ln_part / 2)


# ---------------------------------------------------------------------------
# Williams Plot (Applicability Domain)
# ---------------------------------------------------------------------------

def williams_plot_data(X_train, X_test, residuals_train, residuals_test):
    """
    Prepare data for Williams plot (standardized residuals vs leverage).

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, p)
    X_test : np.ndarray of shape (n_test, p)
    residuals_train : np.ndarray
    residuals_test : np.ndarray

    Returns
    -------
    dict
        leverage_train, leverage_test, std_res_train, std_res_test,
        h_threshold, residual_threshold.
    """
    n_train = X_train.shape[0]
    p = X_train.shape[1] + 1

    # Compute leverage for training set
    h_train = leverage_values(X_train)

    # Compute leverage for test set using training set hat matrix
    X_b_train = np.column_stack([np.ones(n_train), X_train])
    try:
        XtX_inv = np.linalg.inv(X_b_train.T @ X_b_train)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_b_train.T @ X_b_train)

    X_b_test = np.column_stack([np.ones(len(X_test)), X_test])
    h_test = np.array([x @ XtX_inv @ x for x in X_b_test])

    # Standardized residuals
    std_res_train = standardized_residuals(residuals_train, X_train)

    mse_train = np.sum(residuals_train ** 2) / (n_train - p)
    s = np.sqrt(mse_train)
    std_res_test = residuals_test / s if s > 0 else residuals_test

    # Thresholds
    h_threshold = 3 * p / n_train
    residual_threshold = 3.0

    return {
        'leverage_train': h_train.tolist(),
        'leverage_test': h_test.tolist(),
        'std_res_train': std_res_train.tolist(),
        'std_res_test': std_res_test.tolist(),
        'h_threshold': h_threshold,
        'residual_threshold': residual_threshold,
        'n_outside_ad': int(
            np.sum((np.abs(std_res_train) > residual_threshold) |
                   (h_train > h_threshold)) +
            np.sum((np.abs(std_res_test) > residual_threshold) |
                   (h_test > h_threshold))
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Demo with synthetic data
    np.random.seed(42)
    n = 200
    p = 3
    X = np.random.randn(n, p)
    true_beta = np.array([2.0, -1.5, 0.5])
    y = 10 + X @ true_beta + np.random.randn(n) * 2

    # Fit
    X_b = np.column_stack([np.ones(n), X])
    theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
    y_pred = X_b @ theta
    residuals = y - y_pred

    results = run_all_diagnostics(residuals, X, y_pred, model_name='Demo Model')
    print("\nDiagnostics completed successfully.")
