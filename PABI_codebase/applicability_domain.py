#!/usr/bin/env python3
"""
applicability_domain.py - Applicability Domain Analysis for PABI QSPR Models

Implements Williams plot analysis to define the chemical space
where PABI-based QSPR models produce reliable predictions.

The applicability domain is defined by:
    1. Leverage threshold: h* = 3p/n (where p = model parameters, n = training size)
    2. Standardized residual threshold: |r_i| < 3

Compounds within both thresholds are within the AD.

Author: [Your Name]
License: MIT
"""

import numpy as np
import pandas as pd
import os

from diagnostics import (
    leverage_values,
    standardized_residuals,
    compute_hat_matrix,
    williams_plot_data,
)


def define_applicability_domain(X_train, y_train, X_test=None, y_test=None,
                                 y_pred_train=None, y_pred_test=None):
    """
    Define the applicability domain for a QSPR model.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, p)
        Training set predictors.
    y_train : np.ndarray
        Training set observed values.
    X_test : np.ndarray, optional
        Test set predictors.
    y_test : np.ndarray, optional
        Test set observed values.
    y_pred_train : np.ndarray, optional
        Training set predicted values. Computed if not provided.
    y_pred_test : np.ndarray, optional
        Test set predicted values.

    Returns
    -------
    dict
        AD analysis results.
    """
    n_train = X_train.shape[0]
    p = X_train.shape[1] + 1  # +1 for intercept

    # Compute predictions if not provided
    if y_pred_train is None:
        X_b = np.column_stack([np.ones(n_train), X_train])
        theta = np.linalg.lstsq(X_b, y_train, rcond=None)[0]
        y_pred_train = X_b @ theta

    residuals_train = y_train - y_pred_train

    # Leverage for training set
    h_train = leverage_values(X_train)
    h_threshold = 3 * p / n_train

    # Standardized residuals
    std_res_train = standardized_residuals(residuals_train, X_train)

    # AD membership for training set
    in_ad_train = (np.abs(std_res_train) < 3) & (h_train < h_threshold)

    results = {
        'n_train': n_train,
        'p': p - 1,
        'h_threshold': h_threshold,
        'residual_threshold': 3.0,
        'train': {
            'leverage': h_train,
            'std_residuals': std_res_train,
            'in_ad': in_ad_train,
            'n_in_ad': int(np.sum(in_ad_train)),
            'n_outside_ad': int(np.sum(~in_ad_train)),
            'pct_in_ad': float(np.mean(in_ad_train) * 100),
        },
    }

    # Test set analysis
    if X_test is not None and y_pred_test is not None:
        n_test = X_test.shape[0]
        residuals_test = y_test - y_pred_test if y_test is not None else np.zeros(n_test)

        # Leverage for test set (using training hat matrix)
        X_b_train = np.column_stack([np.ones(n_train), X_train])
        try:
            XtX_inv = np.linalg.inv(X_b_train.T @ X_b_train)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X_b_train.T @ X_b_train)

        X_b_test = np.column_stack([np.ones(n_test), X_test])
        h_test = np.array([x @ XtX_inv @ x for x in X_b_test])

        # Standardized residuals for test set
        mse = np.sum(residuals_train ** 2) / (n_train - p)
        s = np.sqrt(mse)
        std_res_test = residuals_test / s if s > 0 else residuals_test

        in_ad_test = (np.abs(std_res_test) < 3) & (h_test < h_threshold)

        results['test'] = {
            'leverage': h_test,
            'std_residuals': std_res_test,
            'in_ad': in_ad_test,
            'n_in_ad': int(np.sum(in_ad_test)),
            'n_outside_ad': int(np.sum(~in_ad_test)),
            'pct_in_ad': float(np.mean(in_ad_test) * 100),
        }

    return results


def predict_with_ad_check(X_new, X_train, y_train, model_intercept, model_coeffs):
    """
    Make predictions and check if new compounds are within the AD.

    Parameters
    ----------
    X_new : np.ndarray
        New compound descriptors.
    X_train : np.ndarray
        Training set descriptors.
    y_train : np.ndarray
        Training set response.
    model_intercept : float
        Model intercept.
    model_coeffs : np.ndarray
        Model coefficients.

    Returns
    -------
    dict
        Predictions with AD flags and reliability metrics.
    """
    n_train = X_train.shape[0]
    p = X_train.shape[1] + 1
    h_threshold = 3 * p / n_train

    # Predictions
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    y_pred = model_intercept + X_new @ model_coeffs

    # Leverage
    X_b_train = np.column_stack([np.ones(n_train), X_train])
    try:
        XtX_inv = np.linalg.inv(X_b_train.T @ X_b_train)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_b_train.T @ X_b_train)

    X_b_new = np.column_stack([np.ones(len(X_new)), X_new])
    h_new = np.array([x @ XtX_inv @ x for x in X_b_new])

    # Prediction confidence based on leverage
    in_ad = h_new < h_threshold

    # Prediction intervals
    y_pred_train = model_intercept + X_train @ model_coeffs
    residuals = y_train - y_pred_train
    mse = np.sum(residuals ** 2) / (n_train - p)
    se = np.sqrt(mse)

    pi_width = 1.96 * se * np.sqrt(1 + h_new)  # Approximation

    return {
        'predictions': y_pred.flatten(),
        'leverage': h_new,
        'in_ad': in_ad,
        'prediction_interval_lower': (y_pred.flatten() - pi_width),
        'prediction_interval_upper': (y_pred.flatten() + pi_width),
        'h_threshold': h_threshold,
        'reliability': np.where(in_ad, 'High', 'Low/Outside AD'),
    }


def print_ad_summary(ad_results, model_name='Model'):
    """Print a formatted summary of applicability domain analysis."""
    print(f"\n  Applicability Domain Summary: {model_name}")
    print(f"  {'='*50}")
    print(f"  Leverage threshold (h*): {ad_results['h_threshold']:.4f}")
    print(f"  Residual threshold: +/- {ad_results['residual_threshold']}")

    train = ad_results['train']
    print(f"\n  Training set:")
    print(f"    Total compounds: {ad_results['n_train']}")
    print(f"    Within AD: {train['n_in_ad']} ({train['pct_in_ad']:.1f}%)")
    print(f"    Outside AD: {train['n_outside_ad']}")

    if 'test' in ad_results:
        test = ad_results['test']
        print(f"\n  Test set:")
        print(f"    Total compounds: {test['n_in_ad'] + test['n_outside_ad']}")
        print(f"    Within AD: {test['n_in_ad']} ({test['pct_in_ad']:.1f}%)")
        print(f"    Outside AD: {test['n_outside_ad']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Demo with synthetic data
    np.random.seed(42)
    n_train, n_test, p = 200, 50, 3

    X_train = np.random.randn(n_train, p)
    beta = np.array([2.0, -1.5, 0.5])
    y_train = 10 + X_train @ beta + np.random.randn(n_train) * 2

    X_test = np.random.randn(n_test, p)
    y_test = 10 + X_test @ beta + np.random.randn(n_test) * 2

    # Fit model
    X_b = np.column_stack([np.ones(n_train), X_train])
    theta = np.linalg.lstsq(X_b, y_train, rcond=None)[0]
    y_pred_train = X_b @ theta
    y_pred_test = theta[0] + X_test @ theta[1:]

    ad = define_applicability_domain(
        X_train, y_train, X_test, y_test, y_pred_train, y_pred_test)
    print_ad_summary(ad, 'Demo Model')
