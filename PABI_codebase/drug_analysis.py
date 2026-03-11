#!/usr/bin/env python3
"""
drug_analysis.py - PABI Analysis for Drug-like Molecules

Implements:
    1. PABI distribution analysis for FDA-approved drugs
    2. Stratification by administration route (oral, injectable, CNS)
    3. Molecular docking correlation analysis
    4. Comparison with Lipinski's Rule of Five

Author: [Your Name]
License: MIT
"""

import numpy as np
import pandas as pd
import warnings
import os

try:
    from pabi import PABICalculator
    HAS_PABI = True
except ImportError:
    HAS_PABI = False


# ---------------------------------------------------------------------------
# FDA-approved drug database (representative sample)
# ---------------------------------------------------------------------------

FDA_DRUGS = [
    # Oral drugs
    {'name': 'Aspirin', 'smiles': 'CC(=O)Oc1ccccc1C(=O)O', 'route': 'oral'},
    {'name': 'Ibuprofen', 'smiles': 'CC(C)Cc1ccc(C(C)C(=O)O)cc1', 'route': 'oral'},
    {'name': 'Metformin', 'smiles': 'CN(C)C(=N)NC(=N)N', 'route': 'oral'},
    {'name': 'Atorvastatin', 'smiles': 'CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(c2ccc(F)cc2)c(c1c1ccccc1)C(=O)Nc1ccccc1', 'route': 'oral'},
    {'name': 'Amlodipine', 'smiles': 'CCOC(=O)C1=C(COCCN)NC(C)=C(C1c1ccccc1Cl)C(=O)OC', 'route': 'oral'},
    {'name': 'Omeprazole', 'smiles': 'COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1', 'route': 'oral'},
    {'name': 'Losartan', 'smiles': 'CCCCc1nc(Cl)c(n1Cc1ccc(-c2ccccc2C=O)cc1)CO', 'route': 'oral'},
    {'name': 'Metoprolol', 'smiles': 'COCCc1ccc(OCC(O)CNC(C)C)cc1', 'route': 'oral'},
    {'name': 'Naproxen', 'smiles': 'COc1ccc2cc(C(C)C(=O)O)ccc2c1', 'route': 'oral'},
    {'name': 'Warfarin', 'smiles': 'CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O', 'route': 'oral'},
    {'name': 'Diclofenac', 'smiles': 'OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl', 'route': 'oral'},
    {'name': 'Paracetamol', 'smiles': 'CC(=O)Nc1ccc(O)cc1', 'route': 'oral'},
    {'name': 'Ciprofloxacin', 'smiles': 'O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O', 'route': 'oral'},
    {'name': 'Furosemide', 'smiles': 'NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl', 'route': 'oral'},
    {'name': 'Carbamazepine', 'smiles': 'NC(=O)N1c2ccccc2C=Cc2ccccc21', 'route': 'oral'},

    # Injectable drugs
    {'name': 'Propofol', 'smiles': 'CC(C)c1cccc(C(C)C)c1O', 'route': 'injectable'},
    {'name': 'Ketorolac', 'smiles': 'OC(=O)C1CCc2n1c1ccccc1c2=O', 'route': 'injectable'},
    {'name': 'Ondansetron', 'smiles': 'Cn1c2c(c(=O)c3ccccc31)CC(CC2)n1ccnc1C', 'route': 'injectable'},
    {'name': 'Dexamethasone', 'smiles': 'C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO', 'route': 'injectable'},
    {'name': 'Vancomycin', 'smiles': 'CC1O[C@H](OC2=CC=C3C=C2)C(O)C(O)C1NC(=O)C', 'route': 'injectable'},
    {'name': 'Midazolam', 'smiles': 'Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2', 'route': 'injectable'},
    {'name': 'Fentanyl', 'smiles': 'CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1', 'route': 'injectable'},
    {'name': 'Indomethacin', 'smiles': 'COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1', 'route': 'injectable'},

    # CNS agents
    {'name': 'Diazepam', 'smiles': 'CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21', 'route': 'cns'},
    {'name': 'Fluoxetine', 'smiles': 'CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1', 'route': 'cns'},
    {'name': 'Sertraline', 'smiles': 'CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21', 'route': 'cns'},
    {'name': 'Risperidone', 'smiles': 'Cc1nc2n(c1C=O)CCc1ccc(F)cc1-2', 'route': 'cns'},
    {'name': 'Caffeine', 'smiles': 'Cn1c(=O)c2c(ncn2C)n(C)c1=O', 'route': 'cns'},
    {'name': 'Phenytoin', 'smiles': 'O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1', 'route': 'cns'},
    {'name': 'Zolpidem', 'smiles': 'Cc1ccc(-c2nc3ccc(C)cn3c2CC(=O)N(C)C)cc1', 'route': 'cns'},
    {'name': 'Lamotrigine', 'smiles': 'Nc1nnc(-c2cccc(Cl)c2Cl)c(N)n1', 'route': 'cns'},
]


def analyze_fda_drugs(alpha=0.75, output_dir='results'):
    """
    Analyze PABI distribution for FDA-approved drugs.

    Parameters
    ----------
    alpha : float
        PABI scaling exponent.
    output_dir : str
        Output directory.

    Returns
    -------
    dict
        Analysis results by administration route.
    """
    if not HAS_PABI:
        raise RuntimeError("pabi module required")

    calc = PABICalculator(alpha=alpha)
    records = []

    for drug in FDA_DRUGS:
        try:
            result = calc.from_smiles(drug['smiles'])
            record = {
                'name': drug['name'],
                'smiles': drug['smiles'],
                'route': drug['route'],
                'pabi': result['pabi'],
                'mw': result['mw'],
                'logp': result['logp'],
                'tpsa': result['tpsa'],
                'n_hbd': result['n_hbd'],
                'n_hba': result['n_hba'],
                'n_ar': result['n_ar'],
            }
            records.append(record)
        except Exception as e:
            warnings.warn(f"Skipping {drug['name']}: {e}")

    df = pd.DataFrame(records)

    # Analysis by route
    results = {}
    for route in ['oral', 'injectable', 'cns']:
        route_data = df[df['route'] == route]
        if len(route_data) > 0:
            results[route] = {
                'n': len(route_data),
                'pabi_mean': route_data['pabi'].mean(),
                'pabi_std': route_data['pabi'].std(),
                'pabi_min': route_data['pabi'].min(),
                'pabi_max': route_data['pabi'].max(),
                'pabi_median': route_data['pabi'].median(),
                'mw_mean': route_data['mw'].mean(),
                'logp_mean': route_data['logp'].mean(),
                'pabi_values': route_data['pabi'].tolist(),
            }

    # Lipinski compliance
    df['ro5_violations'] = 0
    df.loc[df['mw'] > 500, 'ro5_violations'] += 1
    df.loc[df['logp'] > 5, 'ro5_violations'] += 1
    df.loc[df['n_hbd'] > 5, 'ro5_violations'] += 1
    df.loc[df['n_hba'] > 10, 'ro5_violations'] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("FDA DRUG PABI ANALYSIS")
    print("=" * 60)

    for route, data in results.items():
        print(f"\n  {route.upper()} (n = {data['n']}):")
        print(f"    PABI: {data['pabi_mean']:.3f} +/- {data['pabi_std']:.3f}")
        print(f"    Range: [{data['pabi_min']:.3f}, {data['pabi_max']:.3f}]")
        print(f"    MW mean: {data['mw_mean']:.1f}")
        print(f"    LogP mean: {data['logp_mean']:.2f}")

    # Lipinski vs PABI correlation
    print(f"\n  Lipinski Rule of Five compliance:")
    print(f"    0 violations: {(df['ro5_violations'] == 0).sum()}")
    print(f"    1 violation:  {(df['ro5_violations'] == 1).sum()}")
    print(f"    2+ violations: {(df['ro5_violations'] >= 2).sum()}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'fda_drug_analysis.csv'), index=False)

    return {
        'df': df,
        'route_analysis': results,
    }


def docking_correlation_analysis(pabi_values, binding_affinities):
    """
    Analyze correlation between PABI and molecular docking binding affinities.

    Parameters
    ----------
    pabi_values : array-like
        PABI values for docked compounds.
    binding_affinities : array-like
        Binding affinity values (kcal/mol) from docking.

    Returns
    -------
    dict
        Correlation statistics.
    """
    pabi = np.asarray(pabi_values)
    dg = np.asarray(binding_affinities)
    n = len(pabi)

    # Linear fit
    slope, intercept = np.polyfit(pabi, dg, 1)
    dg_pred = slope * pabi + intercept

    ss_res = np.sum((dg - dg_pred) ** 2)
    ss_tot = np.sum((dg - np.mean(dg)) ** 2)
    r2 = 1 - ss_res / ss_tot

    try:
        from scipy import stats
        _, p_value = stats.pearsonr(pabi, dg)
    except ImportError:
        p_value = None

    print(f"\n  Docking Correlation Analysis:")
    print(f"    n = {n}")
    print(f"    dG = {slope:.2f} * PABI + ({intercept:.2f})")
    print(f"    R^2 = {r2:.3f}")
    if p_value:
        print(f"    p-value = {p_value:.2e}")

    return {
        'n': n,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r2,
        'p_value': p_value,
        'equation': f'dG = {slope:.2f} * PABI + {intercept:.2f}',
    }


def pabi_drug_likeness_score(pabi_value, route='oral'):
    """
    Compute a PABI-based drug-likeness score for a given administration route.

    Parameters
    ----------
    pabi_value : float
        Computed PABI value.
    route : str
        Administration route: 'oral', 'injectable', or 'cns'.

    Returns
    -------
    dict
        Score and interpretation.
    """
    ranges = {
        'oral': (0.35, 0.55),
        'injectable': (0.45, 0.65),
        'cns': (0.25, 0.45),
    }

    if route not in ranges:
        raise ValueError(f"Unknown route: {route}. Use 'oral', 'injectable', or 'cns'.")

    low, high = ranges[route]
    center = (low + high) / 2
    half_width = (high - low) / 2

    if low <= pabi_value <= high:
        # Within optimal range: score based on distance from center
        distance = abs(pabi_value - center) / half_width
        score = 1.0 - 0.3 * distance  # Range: 0.7 to 1.0
        interpretation = 'Optimal PABI range'
    elif pabi_value < low:
        distance = (low - pabi_value) / half_width
        score = max(0.0, 0.7 - 0.5 * distance)
        interpretation = 'Below optimal PABI (insufficient polar-aromatic balance)'
    else:
        distance = (pabi_value - high) / half_width
        score = max(0.0, 0.7 - 0.5 * distance)
        interpretation = 'Above optimal PABI (excess polar-aromatic character)'

    return {
        'pabi': pabi_value,
        'route': route,
        'optimal_range': (low, high),
        'score': round(score, 3),
        'interpretation': interpretation,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    results = analyze_fda_drugs(alpha=0.75, output_dir='results')

    # Demo: drug-likeness scoring
    print("\n" + "=" * 60)
    print("PABI DRUG-LIKENESS SCORING")
    print("=" * 60)

    test_pabi_values = [0.30, 0.40, 0.50, 0.60, 0.70]
    for pabi_val in test_pabi_values:
        for route in ['oral', 'injectable', 'cns']:
            score = pabi_drug_likeness_score(pabi_val, route)
            print(f"  PABI={pabi_val:.2f}, Route={route}: "
                  f"Score={score['score']:.3f} ({score['interpretation']})")

    # Demo: docking correlation
    np.random.seed(42)
    n_dock = 30
    pabi_dock = np.random.uniform(0.3, 0.8, n_dock)
    dg_dock = -2.34 * pabi_dock - 4.78 + np.random.normal(0, 0.5, n_dock)
    docking_correlation_analysis(pabi_dock, dg_dock)
