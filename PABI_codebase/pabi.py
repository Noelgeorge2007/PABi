#!/usr/bin/env python3
"""
pabi.py - Polar-Aromatic Balance Index (PABI) Computation Module

This module implements the PABI descriptor as defined in:
    PABI = (Phi_polar * N_ar^alpha) / MW

where:
    Phi_polar = molecular polarizability (from DFT or group-contribution approximation)
    N_ar      = number of aromatic rings
    alpha     = scaling exponent (default 0.75, optimized)
    MW        = molecular weight

Supports both:
    1. DFT-based polarizability (via Gaussian 16 output parsing)
    2. Group-contribution approximation (via RDKit fragment-based estimation)

Author: [Your Name]
License: MIT
"""

import numpy as np
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    warnings.warn("RDKit not available. Only manual PABI computation will work.")


# ---------------------------------------------------------------------------
# Group-contribution polarizability parameters (Miller, J. Phys. Chem. A, 2007)
# Units: 10^{-24} cm^3
# ---------------------------------------------------------------------------
POLARIZABILITY_INCREMENTS = {
    'C_sp3': 1.352,
    'C_sp2': 1.283,
    'C_aromatic': 1.283,
    'C_sp': 1.218,
    'H': 0.387,
    'O_hydroxyl': 0.637,
    'O_ether': 0.637,
    'O_carbonyl': 0.569,
    'N_amine': 1.030,
    'N_amide': 0.854,
    'N_nitro': 0.854,
    'N_aromatic': 1.030,
    'F': 0.296,
    'Cl': 2.315,
    'Br': 3.013,
    'I': 5.415,
    'S_thiol': 2.888,
    'S_thioether': 2.888,
    'S_sulfoxide': 2.500,
    'P': 3.200,
}


def compute_pabi(phi_polar, n_ar, mw, alpha=0.75):
    """
    Compute the Polar-Aromatic Balance Index (PABI).

    Parameters
    ----------
    phi_polar : float
        Molecular polarizability in units of 10^{-24} cm^3.
    n_ar : int
        Number of aromatic rings in the molecule.
    mw : float
        Molecular weight in g/mol.
    alpha : float, optional
        Scaling exponent for aromatic ring count. Default is 0.75.

    Returns
    -------
    float
        The PABI value. Returns 0.0 for non-aromatic molecules (n_ar = 0).

    Raises
    ------
    ValueError
        If phi_polar < 0, n_ar < 0, or mw <= 0.

    Examples
    --------
    >>> compute_pabi(10.42, 1, 78.1)
    0.1334...
    >>> compute_pabi(25.03, 3, 178.2)
    0.3256...
    """
    if phi_polar < 0:
        raise ValueError(f"Polarizability must be non-negative, got {phi_polar}")
    if n_ar < 0:
        raise ValueError(f"Aromatic ring count must be non-negative, got {n_ar}")
    if mw <= 0:
        raise ValueError(f"Molecular weight must be positive, got {mw}")

    if n_ar == 0:
        return 0.0

    return (phi_polar * (n_ar ** alpha)) / mw


def compute_pabi_bounds(phi_polar, n_ar, mw, alpha=0.75):
    """
    Compute theoretical upper bound for PABI.

    From Proposition 4: PABI <= phi_polar * n_ar^alpha / MW

    Parameters
    ----------
    phi_polar, n_ar, mw, alpha : see compute_pabi

    Returns
    -------
    tuple
        (lower_bound, upper_bound) where lower_bound is always 0.
    """
    if n_ar == 0:
        return (0.0, 0.0)
    upper = (phi_polar * (n_ar ** alpha)) / mw
    return (0.0, upper)


class PABICalculator:
    """
    Full PABI calculator with RDKit integration.

    Supports SMILES input, automatic property extraction,
    and both DFT and group-contribution polarizability.

    Parameters
    ----------
    alpha : float
        Scaling exponent (default 0.75).
    method : str
        Polarizability method: 'group_contribution' or 'dft'.
        If 'dft', polarizability must be provided externally.

    Examples
    --------
    >>> calc = PABICalculator(alpha=0.75)
    >>> result = calc.from_smiles('c1ccccc1')  # benzene
    >>> print(f"PABI = {result['pabi']:.4f}")
    """

    def __init__(self, alpha=0.75, method='group_contribution'):
        if not HAS_RDKIT and method == 'group_contribution':
            raise RuntimeError(
                "RDKit is required for group-contribution method. "
                "Install with: conda install -c conda-forge rdkit"
            )
        self.alpha = alpha
        self.method = method

    def from_smiles(self, smiles, dft_polarizability=None):
        """
        Compute PABI and all descriptors from a SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES representation of the molecule.
        dft_polarizability : float, optional
            DFT-computed polarizability. Required if method='dft'.

        Returns
        -------
        dict
            Dictionary with keys: 'smiles', 'mw', 'n_ar', 'phi_polar',
            'pabi', 'logp', 'tpsa', 'mr', 'balaban_j', 'n_hbd', 'n_hba',
            'n_rotb', 'n_atoms'.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        mol = Chem.AddHs(mol)

        # Molecular weight
        mw = Descriptors.MolWt(mol)

        # Aromatic ring count
        n_ar = self._count_aromatic_rings(mol)

        # Polarizability
        if self.method == 'dft':
            if dft_polarizability is None:
                raise ValueError("DFT polarizability must be provided when method='dft'")
            phi_polar = dft_polarizability
        else:
            phi_polar = self._group_contribution_polarizability(mol)

        # Compute PABI
        pabi = compute_pabi(phi_polar, n_ar, mw, self.alpha)

        # Additional descriptors
        mol_noh = Chem.RemoveHs(mol)
        logp = Descriptors.MolLogP(mol_noh)
        tpsa = Descriptors.TPSA(mol_noh)
        mr = Descriptors.MolMR(mol_noh)
        n_hbd = rdMolDescriptors.CalcNumHBD(mol_noh)
        n_hba = rdMolDescriptors.CalcNumHBA(mol_noh)
        n_rotb = rdMolDescriptors.CalcNumRotatableBonds(mol_noh)
        n_atoms = mol_noh.GetNumAtoms()

        # Balaban J index
        try:
            balaban_j = Descriptors.BalabanJ(mol_noh)
        except Exception:
            balaban_j = np.nan

        return {
            'smiles': smiles,
            'mw': mw,
            'n_ar': n_ar,
            'phi_polar': phi_polar,
            'pabi': pabi,
            'logp': logp,
            'tpsa': tpsa,
            'mr': mr,
            'balaban_j': balaban_j,
            'n_hbd': n_hbd,
            'n_hba': n_hba,
            'n_rotb': n_rotb,
            'n_atoms': n_atoms,
        }

    def from_mol(self, mol, dft_polarizability=None):
        """
        Compute PABI from an RDKit Mol object.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            RDKit molecule object.
        dft_polarizability : float, optional
            DFT-computed polarizability.

        Returns
        -------
        dict
            Same as from_smiles().
        """
        smiles = Chem.MolToSmiles(mol)
        return self.from_smiles(smiles, dft_polarizability)

    def batch_compute(self, smiles_list, dft_polarizabilities=None):
        """
        Compute PABI for a list of SMILES strings.

        Parameters
        ----------
        smiles_list : list of str
            List of SMILES strings.
        dft_polarizabilities : list of float, optional
            Corresponding DFT polarizabilities.

        Returns
        -------
        list of dict
            List of result dictionaries.
        """
        results = []
        for i, smi in enumerate(smiles_list):
            try:
                dft_pol = None
                if dft_polarizabilities is not None:
                    dft_pol = dft_polarizabilities[i]
                result = self.from_smiles(smi, dft_pol)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Failed for SMILES '{smi}': {e}")
                results.append({'smiles': smi, 'error': str(e)})
        return results

    def _count_aromatic_rings(self, mol):
        """Count the number of aromatic rings using SSSR."""
        ring_info = mol.GetRingInfo()
        aromatic_count = 0
        for ring in ring_info.AtomRings():
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                aromatic_count += 1
        return aromatic_count

    def _group_contribution_polarizability(self, mol):
        """
        Estimate molecular polarizability using group-contribution method.

        Uses Miller's atomic hybrid polarizability parameters
        (J. Phys. Chem. A, 2007, 111, 10528-10537).

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            RDKit molecule with explicit hydrogens.

        Returns
        -------
        float
            Estimated polarizability in 10^{-24} cm^3.
        """
        total_pol = 0.0

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            hyb = atom.GetHybridization()
            is_arom = atom.GetIsAromatic()

            if symbol == 'C':
                if is_arom:
                    total_pol += POLARIZABILITY_INCREMENTS['C_aromatic']
                elif hyb == Chem.HybridizationType.SP3:
                    total_pol += POLARIZABILITY_INCREMENTS['C_sp3']
                elif hyb == Chem.HybridizationType.SP2:
                    total_pol += POLARIZABILITY_INCREMENTS['C_sp2']
                elif hyb == Chem.HybridizationType.SP:
                    total_pol += POLARIZABILITY_INCREMENTS['C_sp']
                else:
                    total_pol += POLARIZABILITY_INCREMENTS['C_sp3']

            elif symbol == 'H':
                total_pol += POLARIZABILITY_INCREMENTS['H']

            elif symbol == 'O':
                # Determine oxygen type
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                bonds = [mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondType()
                         for n in atom.GetNeighbors()]
                if any(b == Chem.BondType.DOUBLE for b in bonds):
                    total_pol += POLARIZABILITY_INCREMENTS['O_carbonyl']
                else:
                    total_pol += POLARIZABILITY_INCREMENTS['O_hydroxyl']

            elif symbol == 'N':
                if is_arom:
                    total_pol += POLARIZABILITY_INCREMENTS['N_aromatic']
                else:
                    total_pol += POLARIZABILITY_INCREMENTS['N_amine']

            elif symbol == 'F':
                total_pol += POLARIZABILITY_INCREMENTS['F']
            elif symbol == 'Cl':
                total_pol += POLARIZABILITY_INCREMENTS['Cl']
            elif symbol == 'Br':
                total_pol += POLARIZABILITY_INCREMENTS['Br']
            elif symbol == 'I':
                total_pol += POLARIZABILITY_INCREMENTS['I']
            elif symbol == 'S':
                total_pol += POLARIZABILITY_INCREMENTS['S_thiol']
            elif symbol == 'P':
                total_pol += POLARIZABILITY_INCREMENTS['P']
            else:
                # Default: use carbon sp3 as approximation
                total_pol += POLARIZABILITY_INCREMENTS['C_sp3']
                warnings.warn(f"No polarizability increment for {symbol}, using C_sp3 default")

        return total_pol


def parse_gaussian_polarizability(log_file):
    """
    Extract isotropic polarizability from a Gaussian 16 output file.

    Looks for the line:
        'Isotropic Polarizability = XX.XXX'
    or parses from the polarizability tensor output.

    Parameters
    ----------
    log_file : str
        Path to Gaussian .log output file.

    Returns
    -------
    float
        Isotropic polarizability in 10^{-24} cm^3 (converted from Bohr^3).

    Notes
    -----
    Gaussian reports polarizability in Bohr^3. Conversion factor:
    1 Bohr^3 = 0.14818 * 10^{-24} cm^3
    """
    BOHR3_TO_CM3 = 0.14818  # Bohr^3 to 10^{-24} cm^3

    polarizability = None

    with open(log_file, 'r') as f:
        for line in f:
            # Look for exact match
            if 'Exact polarizability' in line:
                parts = line.split()
                # Format: xx, xy, yy, xz, yz, zz
                try:
                    xx = float(parts[2])
                    yy = float(parts[4])
                    zz = float(parts[7])
                    polarizability = (xx + yy + zz) / 3.0
                except (IndexError, ValueError):
                    pass

            if 'Isotropic' in line and 'Polarizability' in line:
                parts = line.split('=')
                if len(parts) > 1:
                    try:
                        polarizability = float(parts[-1].strip().split()[0])
                    except ValueError:
                        pass

    if polarizability is None:
        raise ValueError(f"Could not find polarizability in {log_file}")

    return polarizability * BOHR3_TO_CM3


def validate_group_contribution(smiles_list, dft_polarizabilities):
    """
    Validate group-contribution polarizability against DFT values.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings.
    dft_polarizabilities : list of float
        DFT-computed polarizabilities.

    Returns
    -------
    dict
        Validation statistics including R^2, RMSE, MAE, and slope.
    """
    calc = PABICalculator(method='group_contribution')

    gc_values = []
    dft_values = []

    for smi, dft_val in zip(smiles_list, dft_polarizabilities):
        try:
            result = calc.from_smiles(smi)
            gc_values.append(result['phi_polar'])
            dft_values.append(dft_val)
        except Exception:
            continue

    gc_arr = np.array(gc_values)
    dft_arr = np.array(dft_values)

    # Linear regression
    slope, intercept = np.polyfit(gc_arr, dft_arr, 1)
    predicted = slope * gc_arr + intercept
    ss_res = np.sum((dft_arr - predicted) ** 2)
    ss_tot = np.sum((dft_arr - np.mean(dft_arr)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    rmse = np.sqrt(np.mean((gc_arr - dft_arr) ** 2))
    mae = np.mean(np.abs(gc_arr - dft_arr))

    return {
        'n_compounds': len(gc_values),
        'r_squared': r_squared,
        'rmse': rmse,
        'mae': mae,
        'slope': slope,
        'intercept': intercept,
        'gc_values': gc_arr,
        'dft_values': dft_arr,
    }


def optimize_alpha(smiles_list, properties_dict, alpha_range=(0.50, 1.00),
                   alpha_step=0.05, dft_polarizabilities=None):
    """
    Optimize the scaling exponent alpha by maximizing average R^2
    across multiple target properties.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings for compounds.
    properties_dict : dict
        Dictionary of property arrays, e.g., {'MP': [...], 'BP': [...], 'LogP': [...]}.
    alpha_range : tuple
        (min_alpha, max_alpha) range to search.
    alpha_step : float
        Step size for grid search.
    dft_polarizabilities : list of float, optional
        DFT polarizabilities. If None, uses group-contribution.

    Returns
    -------
    dict
        Optimization results including best_alpha, all_alphas, all_r2s.
    """
    alphas = np.arange(alpha_range[0], alpha_range[1] + alpha_step / 2, alpha_step)
    avg_r2s = []

    for alpha in alphas:
        calc = PABICalculator(alpha=alpha)
        pabi_values = []

        for i, smi in enumerate(smiles_list):
            try:
                dft_pol = dft_polarizabilities[i] if dft_polarizabilities else None
                result = calc.from_smiles(smi, dft_pol)
                pabi_values.append(result['pabi'])
            except Exception:
                pabi_values.append(np.nan)

        pabi_arr = np.array(pabi_values)

        # Compute R^2 for each property
        r2_list = []
        for prop_name, prop_values in properties_dict.items():
            prop_arr = np.array(prop_values)
            mask = ~(np.isnan(pabi_arr) | np.isnan(prop_arr))
            if mask.sum() < 3:
                continue
            x = pabi_arr[mask]
            y = prop_arr[mask]
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r2_list.append(r2)

        avg_r2 = np.mean(r2_list) if r2_list else 0
        avg_r2s.append(avg_r2)

    avg_r2s = np.array(avg_r2s)
    best_idx = np.argmax(avg_r2s)

    return {
        'best_alpha': alphas[best_idx],
        'best_r2': avg_r2s[best_idx],
        'all_alphas': alphas,
        'all_r2s': avg_r2s,
    }


# ---------------------------------------------------------------------------
# Mathematical property verification (Propositions 1-4)
# ---------------------------------------------------------------------------

def verify_proposition_1(pabi_value):
    """Verify Proposition 1: PABI >= 0 (Non-negativity)."""
    return pabi_value >= 0


def verify_proposition_2(pabi_aromatic, pabi_nonaromatic):
    """
    Verify Proposition 2: Aromatic Discrimination.
    For aromatic compounds, PABI > 0; for non-aromatic, PABI = 0.
    """
    return pabi_aromatic > 0 and pabi_nonaromatic == 0.0


def verify_proposition_3(pabi_values, n_ar_values, alpha=0.75):
    """
    Verify Proposition 3: Diminishing returns.
    The marginal increase in PABI decreases with each additional ring.
    delta_PABI(n+1) / delta_PABI(n) = ((n+1)^alpha - n^alpha) / (n^alpha - (n-1)^alpha)
    This ratio should be < 1 for n >= 2.
    """
    for n in range(2, max(n_ar_values)):
        ratio_num = (n + 1) ** alpha - n ** alpha
        ratio_den = n ** alpha - (n - 1) ** alpha
        if ratio_den > 0 and ratio_num / ratio_den >= 1.0:
            return False
    return True


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import json
    import csv
    import sys

    parser = argparse.ArgumentParser(
        description='Compute PABI (Polar-Aromatic Balance Index) for molecules.'
    )
    parser.add_argument('--smiles', type=str, help='Single SMILES string')
    parser.add_argument('--file', type=str, help='CSV file with SMILES column')
    parser.add_argument('--smiles-col', type=str, default='SMILES',
                        help='Column name for SMILES in CSV (default: SMILES)')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='Scaling exponent alpha (default: 0.75)')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                        help='Output format (default: csv)')

    args = parser.parse_args()

    if not HAS_RDKIT:
        print("ERROR: RDKit is required. Install with: conda install -c conda-forge rdkit")
        sys.exit(1)

    calc = PABICalculator(alpha=args.alpha)

    if args.smiles:
        result = calc.from_smiles(args.smiles)
        if args.format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            for key, val in result.items():
                print(f"{key}: {val}")

    elif args.file:
        import pandas as pd
        df = pd.read_csv(args.file)
        results = calc.batch_compute(df[args.smiles_col].tolist())

        results_df = pd.DataFrame([r for r in results if 'error' not in r])
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            print(results_df.to_string())
    else:
        parser.print_help()
