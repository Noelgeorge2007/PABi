#!/usr/bin/env python3
"""
main.py - Complete PABI QSPR Analysis Pipeline

Orchestrates the full analysis workflow:
    1. Dataset preparation (compute PABI for 250 compounds)
    2. Univariate and multivariate QSPR modeling
    3. Cross-validation and external validation
    4. Regression diagnostics
    5. Applicability domain analysis
    6. Drug-like molecule analysis
    7. Alpha optimization
    8. Figure generation

Usage:
    python main.py                    # Run full pipeline
    python main.py --step dataset     # Run specific step
    python main.py --alpha 0.75       # Specify alpha
    python main.py --no-figures       # Skip figure generation

Author: [Your Name]
License: MIT
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import json
import warnings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ALPHA = 0.75
DATA_DIR = 'data'
RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
RANDOM_SEED = 42


def setup_directories():
    """Create output directories."""
    for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Dataset Preparation
# ---------------------------------------------------------------------------

def step_dataset(alpha=DEFAULT_ALPHA):
    """Prepare the compound dataset with PABI values."""
    print("\n" + "=" * 70)
    print("STEP 1: DATASET PREPARATION")
    print("=" * 70)

    from prepare_dataset import build_dataset, compute_descriptor_statistics

    df = build_dataset(alpha=alpha, output_dir=DATA_DIR)

    print("\nDescriptor Statistics:")
    stats = compute_descriptor_statistics(df)
    print(stats.to_string())

    return df


# ---------------------------------------------------------------------------
# Step 2: QSPR Modeling
# ---------------------------------------------------------------------------

def step_qspr(df, alpha=DEFAULT_ALPHA):
    """Run QSPR regression analysis."""
    print("\n" + "=" * 70)
    print("STEP 2: QSPR MODELING")
    print("=" * 70)

    from qspr_models import run_full_analysis

    results = run_full_analysis(df, alpha=alpha, output_dir=RESULTS_DIR)
    return results


# ---------------------------------------------------------------------------
# Step 3: Regression Diagnostics
# ---------------------------------------------------------------------------

def step_diagnostics(df, qspr_results):
    """Run regression diagnostic tests."""
    print("\n" + "=" * 70)
    print("STEP 3: REGRESSION DIAGNOSTICS")
    print("=" * 70)

    from diagnostics import run_all_diagnostics
    from qspr_models import _ols_fit, _ols_predict

    train = df[df['set'] == 'T'].copy()
    diagnostic_results = {}

    target_properties = {'mp': 'Melting Point', 'bp': 'Boiling Point', 'logp_exp': 'LogP'}

    for prop_col, prop_name in target_properties.items():
        if prop_col in qspr_results.get('multivariate', {}):
            mlr = qspr_results['multivariate'][prop_col]
            selected = mlr['selected_features']
            sel_cols = [c for c in selected if c in train.columns]

            mask = train[sel_cols + [prop_col]].notna().all(axis=1)
            X = train.loc[mask, sel_cols].values
            y = train.loc[mask, prop_col].values

            intercept, coeffs = _ols_fit(X, y)
            y_pred = _ols_predict(X, intercept, coeffs)
            residuals = y - y_pred

            diag = run_all_diagnostics(residuals, X, y_pred, model_name=prop_name)
            diagnostic_results[prop_col] = diag

    # Save
    # Convert numpy arrays for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    serializable = json.loads(json.dumps(diagnostic_results, default=convert))
    with open(os.path.join(RESULTS_DIR, 'diagnostics.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    return diagnostic_results


# ---------------------------------------------------------------------------
# Step 4: Applicability Domain
# ---------------------------------------------------------------------------

def step_applicability_domain(df, qspr_results):
    """Analyze applicability domain for all models."""
    print("\n" + "=" * 70)
    print("STEP 4: APPLICABILITY DOMAIN ANALYSIS")
    print("=" * 70)

    from applicability_domain import define_applicability_domain, print_ad_summary
    from qspr_models import _ols_fit, _ols_predict

    train = df[df['set'] == 'T'].copy()
    test = df[df['set'] == 'E'].copy()

    ad_results = {}
    target_properties = {'mp': 'Melting Point', 'bp': 'Boiling Point', 'logp_exp': 'LogP'}

    for prop_col, prop_name in target_properties.items():
        if prop_col in qspr_results.get('multivariate', {}):
            mlr = qspr_results['multivariate'][prop_col]
            selected = mlr['selected_features']
            sel_cols = [c for c in selected if c in train.columns]

            mask_train = train[sel_cols + [prop_col]].notna().all(axis=1)
            mask_test = test[sel_cols + [prop_col]].notna().all(axis=1)

            X_train = train.loc[mask_train, sel_cols].values
            y_train = train.loc[mask_train, prop_col].values
            X_test = test.loc[mask_test, sel_cols].values
            y_test = test.loc[mask_test, prop_col].values

            intercept, coeffs = _ols_fit(X_train, y_train)
            y_pred_train = _ols_predict(X_train, intercept, coeffs)
            y_pred_test = _ols_predict(X_test, intercept, coeffs)

            ad = define_applicability_domain(
                X_train, y_train, X_test, y_test,
                y_pred_train, y_pred_test)
            print_ad_summary(ad, prop_name)
            ad_results[prop_col] = ad

    return ad_results


# ---------------------------------------------------------------------------
# Step 5: Drug Analysis
# ---------------------------------------------------------------------------

def step_drug_analysis(alpha=DEFAULT_ALPHA):
    """Analyze PABI for FDA-approved drugs."""
    print("\n" + "=" * 70)
    print("STEP 5: FDA DRUG ANALYSIS")
    print("=" * 70)

    from drug_analysis import analyze_fda_drugs
    return analyze_fda_drugs(alpha=alpha, output_dir=RESULTS_DIR)


# ---------------------------------------------------------------------------
# Step 6: Alpha Optimization
# ---------------------------------------------------------------------------

def step_alpha_optimization(df):
    """Optimize the scaling exponent alpha."""
    print("\n" + "=" * 70)
    print("STEP 6: ALPHA OPTIMIZATION")
    print("=" * 70)

    from pabi import optimize_alpha

    train = df[df['set'] == 'T'].copy()
    smiles_list = train['smiles'].tolist()

    properties = {}
    for col in ['mp', 'bp', 'logp_exp']:
        if col in train.columns:
            properties[col] = train[col].tolist()

    result = optimize_alpha(
        smiles_list, properties,
        alpha_range=(0.50, 1.00), alpha_step=0.05)

    print(f"\n  Best alpha: {result['best_alpha']:.2f}")
    print(f"  Best average R^2: {result['best_r2']:.4f}")
    print(f"\n  Alpha scan:")
    for a, r2 in zip(result['all_alphas'], result['all_r2s']):
        marker = " <-- optimal" if a == result['best_alpha'] else ""
        print(f"    alpha = {a:.2f}: avg R^2 = {r2:.4f}{marker}")

    return result


# ---------------------------------------------------------------------------
# Step 7: Figure Generation
# ---------------------------------------------------------------------------

def step_figures(df, qspr_results=None):
    """Generate all publication figures."""
    print("\n" + "=" * 70)
    print("STEP 7: FIGURE GENERATION")
    print("=" * 70)

    from figures import generate_all_figures
    generate_all_figures(df, qspr_results or {}, output_dir=FIGURES_DIR)


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(alpha=DEFAULT_ALPHA, skip_figures=False):
    """Run the complete analysis pipeline."""
    start_time = time.time()

    print("=" * 70)
    print("PABI (Polar-Aromatic Balance Index) - QSPR Analysis Pipeline")
    print("=" * 70)
    print(f"Alpha = {alpha}")
    print(f"Random seed = {RANDOM_SEED}")

    np.random.seed(RANDOM_SEED)
    setup_directories()

    # Step 1: Dataset
    df = step_dataset(alpha=alpha)

    # Step 2: QSPR
    qspr_results = step_qspr(df, alpha=alpha)

    # Step 3: Diagnostics
    diag_results = step_diagnostics(df, qspr_results)

    # Step 4: Applicability Domain
    ad_results = step_applicability_domain(df, qspr_results)

    # Step 5: Drug Analysis
    try:
        drug_results = step_drug_analysis(alpha=alpha)
    except Exception as e:
        warnings.warn(f"Drug analysis skipped: {e}")
        drug_results = None

    # Step 6: Alpha Optimization
    try:
        alpha_results = step_alpha_optimization(df)
    except Exception as e:
        warnings.warn(f"Alpha optimization skipped: {e}")
        alpha_results = None

    # Step 7: Figures
    if not skip_figures:
        try:
            step_figures(df, qspr_results)
        except Exception as e:
            warnings.warn(f"Figure generation failed: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE ({elapsed:.1f} seconds)")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Data:    {DATA_DIR}/")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"  Figures: {FIGURES_DIR}/")

    return {
        'dataset': df,
        'qspr': qspr_results,
        'diagnostics': diag_results,
        'applicability_domain': ad_results,
        'drug_analysis': drug_results,
        'alpha_optimization': alpha_results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PABI QSPR Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  dataset        Prepare compound dataset
  qspr           Run QSPR regression models
  diagnostics    Regression diagnostics
  ad             Applicability domain analysis
  drugs          FDA drug analysis
  alpha          Alpha optimization
  figures        Generate figures
  all            Run full pipeline (default)
        """
    )
    parser.add_argument('--step', type=str, default='all',
                        choices=['dataset', 'qspr', 'diagnostics', 'ad',
                                 'drugs', 'alpha', 'figures', 'all'],
                        help='Which step to run (default: all)')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help=f'Scaling exponent alpha (default: {DEFAULT_ALPHA})')
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to existing dataset CSV')

    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    setup_directories()

    if args.step == 'all':
        run_full_pipeline(alpha=args.alpha, skip_figures=args.no_figures)

    elif args.step == 'dataset':
        step_dataset(alpha=args.alpha)

    elif args.step == 'qspr':
        data_path = args.data or os.path.join(DATA_DIR, 'dataset_250.csv')
        df = pd.read_csv(data_path)
        step_qspr(df, alpha=args.alpha)

    elif args.step == 'diagnostics':
        data_path = args.data or os.path.join(DATA_DIR, 'dataset_250.csv')
        df = pd.read_csv(data_path)
        qspr_results = json.load(open(os.path.join(RESULTS_DIR, 'qspr_results.json')))
        step_diagnostics(df, qspr_results)

    elif args.step == 'drugs':
        step_drug_analysis(alpha=args.alpha)

    elif args.step == 'alpha':
        data_path = args.data or os.path.join(DATA_DIR, 'dataset_250.csv')
        df = pd.read_csv(data_path)
        step_alpha_optimization(df)

    elif args.step == 'figures':
        data_path = args.data or os.path.join(DATA_DIR, 'dataset_250.csv')
        df = pd.read_csv(data_path)
        step_figures(df)
