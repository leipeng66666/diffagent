"""
Common statistical utilities module
Provides shared statistical functions for both projects: Kruskal-Wallis, Spearman, Arrhenius fitting, etc.

All methods are traditional statistical methods. No machine learning.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict
import json

# ============================================================
# Zeolite -> topology type mapping and classification
# ============================================================

ZEOLITE_TOPOLOGY_MAP = {
    # MFI family (ZSM-5, silicalite-1, TS-1)
    'silicalite-1': 'MFI', 'silicalite': 'MFI', 'silicalite-1 (B)': 'MFI',
    'ZSM-5': 'MFI', 'H-ZSM-5': 'MFI', 'HZSM-5': 'MFI', 'HZSM5': 'MFI',
    'HZSM5 extrudates': 'MFI', 'ZSM-5 (silicalite-1)': 'MFI',
    'ZSM-5 (AA)': 'MFI', 'ZSM-5 (BA)': 'MFI', 'ZSM-5 (CA)': 'MFI',
    'ZSM-5 (DA)': 'MFI', 'ZSM-5 (DA*)': 'MFI',
    'MFI': 'MFI', 'TS-1': 'MFI',
    'BAS/Cu+-HZSM-5': 'MFI',

    # FAU family (X, Y, USY, LSX)
    '13X': 'FAU', 'NaX': 'FAU', 'NaY': 'FAU', 'Zeolite X': 'FAU',
    'Zeolite Y': 'FAU', 'USY': 'FAU', 'HY': 'FAU',
    'Li-LSX': 'FAU', 'DAY': 'FAU', 'faujasite': 'FAU',

    # LTA family (3A, 4A, 5A, NaA)
    '3A': 'LTA', '4A': 'LTA', '5A': 'LTA', 'NaA': 'LTA',
    'NaA-ZA1': 'LTA', 'NaA-ZA2': 'LTA', 'NaA-ZA3': 'LTA',
    'NaA-ZA4': 'LTA', 'NaA-ZA5': 'LTA', '4-A zeolite': 'LTA',
    'Davison 5A (C-521)': 'LTA',

    # DDR family
    'DD3R': 'DDR', 'DDR II': 'DDR',

    # BEA family
    'BEA': 'BEA', 'β-zeolite': 'BEA', 'Fe-beta': 'BEA',
    'H-beta': 'BEA', 'Beta': 'BEA',

    # MOR family
    'MOR': 'MOR', 'mordenite': 'MOR',

    # CHA family
    'SAPO-34': 'CHA', 'SSZ-13': 'CHA', 'CHA': 'CHA',

    # HEU family
    'clinoptilolite': 'HEU', 'zeolitic tuff': 'HEU',

    # AFI family
    'AlPO4-5': 'AFI',

    # Other
    'ITQ-39': 'ITT',
    'CON': 'CON',
    'STT': 'STT',
    'RHO': 'RHO',
    'Pt/NaY': 'FAU',
}


def classify_topology(zeolite_name: str) -> str:
    """Infer topology type from zeolite name"""
    if pd.isna(zeolite_name) or str(zeolite_name).strip() == '':
        return 'Unknown'

    name = str(zeolite_name).strip()

    # Exact match
    if name in ZEOLITE_TOPOLOGY_MAP:
        return ZEOLITE_TOPOLOGY_MAP[name]

    # Fuzzy match
    name_lower = name.lower()
    if any(kw in name_lower for kw in ['zsm-5', 'zsm5', 'silicalite', 'mfi']):
        return 'MFI'
    if any(kw in name_lower for kw in ['13x', 'nax', 'nay', 'lsx', 'faujasite',
                                         'zeolite x', 'zeolite y', 'usy']):
        return 'FAU'
    if any(kw in name_lower for kw in ['3a', '4a', '5a', 'naa', 'lta']):
        return 'LTA'
    if any(kw in name_lower for kw in ['dd3r', 'ddr']):
        return 'DDR'
    if 'beta' in name_lower or 'bea' in name_lower:
        return 'BEA'
    if 'mor' in name_lower or 'mordenite' in name_lower:
        return 'MOR'
    if 'sapo-34' in name_lower or 'ssz-13' in name_lower or 'cha' in name_lower:
        return 'CHA'
    if 'clinoptilolite' in name_lower or 'heu' in name_lower:
        return 'HEU'
    if 'alpo' in name_lower:
        return 'AFI'
    if 'itq-39' in name_lower:
        return 'ITT'
    if 'con' in name_lower:
        return 'CON'

    return 'Other'


def classify_method_type(method: str) -> str:
    """Classify experimental method as simulation or experiment"""
    if pd.isna(method):
        return 'unknown'
    method_lower = str(method).lower()
    sim_keywords = ['molecular dynamics', 'md ', 'simulation', 'monte carlo',
                    'calculation', 'dft', 'knudsen', 'correlation',
                    'theoretical', 'empirical', 'kinetic theory', 'millington-quirk',
                    'wilke-chang', 'renkin', 'vignes']
    if any(kw in method_lower for kw in sim_keywords):
        return 'simulation'
    return 'experiment'


def classify_si_al_range(si_al):
    """Classify Si/Al ratio into physically meaningful bins"""
    if pd.isna(si_al):
        return 'Unknown'
    val = float(si_al)
    if val <= 0:
        return 'Unknown'
    if val < 1.5:
        return 'Si/Al<1.5 (X-type low-Si)'
    elif val < 5:
        return 'Si/Al 1.5-5'
    elif val < 20:
        return 'Si/Al 5-20'
    elif val < 100:
        return 'Si/Al 20-100'
    elif val < 500:
        return 'Si/Al 100-500'
    else:
        return 'Si/Al>500 (high-Si/all-Si)'


# ============================================================
# Data preparation
# ============================================================

def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    """Read CSV, convert numeric columns, and add classification columns"""
    df = pd.read_csv(csv_path)

    # Numeric columns
    df['D_value'] = pd.to_numeric(df['diffusion_coefficient_value'], errors='coerce')
    df['logD'] = np.log10(df['D_value'].clip(lower=1e-30))
    df['T_value'] = pd.to_numeric(df['temperature_value'], errors='coerce')
    df['loading'] = pd.to_numeric(df['loading_value'], errors='coerce')
    df['si_al_ratio_num'] = pd.to_numeric(df['si_al_ratio'], errors='coerce')
    df['concentration_num'] = pd.to_numeric(df['concentration_value'], errors='coerce')
    df['adsorption_num'] = pd.to_numeric(df['adsorption_loading_value'], errors='coerce')

    # Categorical columns
    # std_zeolite_name provides standardized naming: TOPOLOGY-SUFFIX (e.g., MFI-H, FAU-NaY)
    # Rule: -H suffix means the zeolite is in proton form, equivalent to the base topology
    #        (MFI-H -> MFI, DDR-H -> DDR, BEA-H -> BEA, etc.)
    #        Other suffixes (FAU-NaY, LTA-K, etc.) are kept as-is since they denote distinct materials
    if 'std_zeolite_name' in df.columns:
        raw_group = df['std_zeolite_name'].fillna(df['zeolite_name'])
        # Strip -H suffix: MFI-H -> MFI, DDR-H -> DDR, BEA-H -> BEA, etc.
        df['zeolite_group'] = raw_group.apply(
            lambda x: str(x)[:-2] if str(x).endswith('-H') else str(x))
        # Extract topology from zeolite_group (part before first '-')
        df['topology'] = df['zeolite_group'].apply(
            lambda x: str(x).split('-')[0] if pd.notna(x) else 'Unknown')
    else:
        df['zeolite_group'] = df['zeolite_name']
        df['topology'] = df['zeolite_name'].apply(classify_topology)
    df['method_type'] = df['method_type'].fillna('unknown') if 'method_type' in df.columns else df['experimental_method'].apply(classify_method_type)
    # Experimental method category (MD/PFG NMR/QENS/ZLC/Uptake/Membrane etc.)
    df['method_category'] = df['method_category'].fillna('Other') if 'method_category' in df.columns else 'Unknown'
    df['ion_group'] = df['modified_ion'].fillna('(none)').replace('', '(none)')
    df['si_al_range'] = df['si_al_ratio_num'].apply(classify_si_al_range)
    # Topology x Si/Al joint grouping (core analysis dimension)
    df['topo_sial'] = df['topology'] + ' | ' + df['si_al_range']

    # Drop raw zeolite_name to prevent LLM from seeing ambiguous names;
    # zeolite_group (from std_zeolite_name) is the canonical identifier
    if 'zeolite_name' in df.columns:
        df = df.drop(columns=['zeolite_name'])

    return df


# ============================================================
# Group statistics
# ============================================================

def group_stats(df, group_col, value_col='D_value', min_count=1):
    """
    Calculate descriptive statistics grouped by a column.
    Returns DataFrame sorted by count descending.
    """
    valid = df[[group_col, value_col]].dropna().copy()
    valid = valid[valid[group_col] != '']

    agg_df = valid.groupby(group_col)[value_col].agg([
        ('count', 'count'),
        ('mean_D', 'mean'),
        ('median_D', 'median'),
        ('std_D', 'std'),
        ('min_D', 'min'),
        ('max_D', 'max'),
    ]).reset_index()

    # Add log10 statistics (more intuitive)
    log_vals = np.log10(valid[value_col].clip(lower=1e-30))
    valid_copy = valid.copy()
    valid_copy['logD'] = log_vals
    log_agg = valid_copy.groupby(group_col)['logD'].agg([
        ('mean_logD', 'mean'),
        ('median_logD', 'median'),
        ('std_logD', 'std'),
    ]).reset_index()

    result = agg_df.merge(log_agg, on=group_col, how='left')
    result['D_range_orders'] = np.log10(
        result['max_D'].clip(lower=1e-30)
    ) - np.log10(result['min_D'].clip(lower=1e-30))

    result = result[result['count'] >= min_count]
    result = result.sort_values('count', ascending=False)
    return result


# ============================================================
# Statistical tests
# ============================================================

def kruskal_test(df, group_col, value_col='logD', min_group_size=3):
    """
    Kruskal-Wallis H test (non-parametric multi-group comparison).
    Tests whether logD differs significantly between groups.

    Returns:
        dict with H_statistic, p_value, significant, n_groups, total_n, group_stats
    """
    valid = df[[group_col, value_col]].dropna()
    # Filter out groups with too few samples
    group_counts = valid.groupby(group_col).size()
    valid_groups = group_counts[group_counts >= min_group_size].index
    valid = valid[valid[group_col].isin(valid_groups)]

    if valid[group_col].nunique() < 2:
        return {'error': f'Insufficient groups (need >=2, got {valid[group_col].nunique()})'}

    groups = [grp[value_col].values for _, grp in valid.groupby(group_col)]

    h_stat, p_value = stats.kruskal(*groups)

    # Effect size eta2 = H/(N-1) approximation
    n_total = len(valid)
    eta_sq = h_stat / (n_total - 1) if n_total > 1 else 0

    # Per-group statistics
    grp_stats = valid.groupby(group_col)[value_col].agg([
        ('n', 'count'), ('mean', 'mean'), ('median', 'median'), ('std', 'std')
    ]).round(4).reset_index()

    return {
        'H_statistic': round(h_stat, 2),
        'p_value': round(p_value, 6),
        'significant': p_value < 0.05,
        'highly_significant': p_value < 0.001,
        'n_groups': len(groups),
        'total_n': n_total,
        'eta_squared': round(eta_sq, 4),  # Effect size: ~0.01 small, ~0.06 medium, ~0.14 large
        'group_details': json.loads(grp_stats.to_json(orient='records', double_precision=4)),
    }


def mannwhitney_test(df, group_col, value_col='logD', group1=None, group2=None):
    """
    Mann-Whitney U test (two-group comparison).
    If group1/group2 are not specified, performs all pairwise comparisons.
    Returns dict with U_statistic, p_value, effect_size (r), significant.

    Returns:
        dict with U_statistic, p_value, effect_size (r), significant
    """
    valid = df[[group_col, value_col]].dropna()
    groups_present = valid[group_col].unique()

    if group1 is not None and group2 is not None:
        # Specified two-group comparison
        a = valid[valid[group_col] == group1][value_col].values
        b = valid[valid[group_col] == group2][value_col].values
        if len(a) < 3 or len(b) < 3:
            return None
        u_stat, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
        n_total = len(a) + len(b)
        r = u_stat / (len(a) * len(b))  # Simplified effect size
        return {
            'group1': str(group1), 'group2': str(group2),
            'n1': len(a), 'n2': len(b),
            'U_statistic': round(u_stat, 2),
            'p_value': round(p_value, 6),
            'effect_size_r': round(r, 3),
            'significant': p_value < 0.05,
            'mean_logD_1': round(np.mean(a), 3),
            'mean_logD_2': round(np.mean(b), 3),
        }

    # All pairwise comparisons
    results = []
    for i, g1 in enumerate(groups_present):
        for g2 in groups_present[i+1:]:
            res = mannwhitney_test(df, group_col, value_col, g1, g2)
            if res:
                results.append(res)
    # Sort by effect size
    results.sort(key=lambda x: abs(1 - x['effect_size_r']), reverse=True)
    return results


# ============================================================
# Correlation analysis
# ============================================================

def spearman_correlation(df, x_col, y_col='logD', by_group=None, min_n=5):
    """
    Spearman rank correlation coefficient.
    If by_group is None, returns a single dict {r, p_value, n, significant}.
    If by_group is specified, returns a list of per-group results.

    Args:
        df: DataFrame
        x_col: x-axis variable
        y_col: y-axis variable (default logD)
        by_group: If specified, compute correlation separately for each group
        min_n: Minimum sample size required

    Returns:
        If by_group is None: {'r': float, 'p_value': float, 'n': int, 'significant': bool}
        If by_group is specified: [{group:, r:, p_value:, n:, significant:}, ...]
    """
    if by_group is None:
        valid = df[[x_col, y_col]].dropna()
        if len(valid) < min_n:
            return {'error': f'Insufficient samples (n={len(valid)} < {min_n})'}
        r, p = stats.spearmanr(valid[x_col], valid[y_col])
        return {
            'r': round(r, 4), 'p_value': round(p, 6),
            'n': len(valid), 'significant': p < 0.05,
        }

    # Grouped computation
    results = []
    for grp_name, grp_df in df.groupby(by_group):
        valid = grp_df[[x_col, y_col]].dropna()
        if len(valid) >= min_n:
            r, p = stats.spearmanr(valid[x_col], valid[y_col])
            results.append({
                'group': str(grp_name),
                'r': round(r, 4), 'p_value': round(p, 6),
                'n': len(valid), 'significant': p < 0.05,
            })
    results.sort(key=lambda x: abs(x['r']), reverse=True)
    return results


# ============================================================
# Arrhenius fitting
# ============================================================

def arrhenius_fit(df, by_group=None, min_points=5, T_col='T_value', D_col='D_value'):
    """
    Arrhenius fit: D = D0 * exp(-Ea / (R * T))
    Linearized: ln(D) = ln(D0) - (Ea/R) * (1/T)

    Args:
        df: DataFrame
        by_group: Grouping column (fit each group separately)
        min_points: Minimum number of data points required
        T_col: Temperature column name
        D_col: Diffusion coefficient column name

    Returns:
        dict with Ea (kJ/mol), D0, R_squared, n_points
    """
    R_gas = 8.314  # J/(mol·K)

    def _fit_one(data):
        """Fit one group of data"""
        valid = data[[T_col, D_col]].dropna().copy()
        valid = valid[valid[T_col] > 0]
        valid = valid[valid[D_col] > 0]

        if len(valid) < min_points:
            return {'error': f'Insufficient samples (n={len(valid)} < {min_points})'}

        x = 1.0 / valid[T_col].values  # 1/T
        y = np.log(valid[D_col].values)  # ln(D)

        # Check if all x values are identical (cannot fit regression)
        if np.std(x) < 1e-15:
            return {'error': 'All temperature values are identical, cannot fit Arrhenius'}

        # Linear regression y = a + b*x, where b = -Ea/R, a = ln(D0)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        Ea = -slope * R_gas / 1000  # kJ/mol
        D0 = np.exp(intercept)

        return {
            'Ea_kJ_per_mol': round(Ea, 2),
            'ln_D0': round(intercept, 2),
            'D0': float(f"{D0:.4e}"),
            'R_squared': round(r_value ** 2, 4),
            'p_value': round(p_value, 6),
            'significant': p_value < 0.05,
            'n_points': len(valid),
            'T_range': f"{valid[T_col].min():.0f} ~ {valid[T_col].max():.0f} K",
            'D_range': f"{valid[D_col].min():.4e} ~ {valid[D_col].max():.4e}",
        }

    if by_group is None:
        return _fit_one(df)

    # Grouped fitting
    results = {}
    for grp_name, grp_df in df.groupby(by_group):
        res = _fit_one(grp_df)
        if 'error' not in res:
            results[str(grp_name)] = res
    return results


# ============================================================
# Top N selection
# ============================================================

def get_top_n(df, col, n=20):
    """Return top N categories by sample count"""
    counts = df[col].value_counts()
    return list(counts.head(n).index)


# ============================================================
# LLM prompt table formatting
# ============================================================

def format_table(data, columns, max_rows=30):
    """
    Convert DataFrame to Markdown table string for LLM prompt building
    """
    if isinstance(data, pd.DataFrame):
        df = data[columns].head(max_rows).copy()
    else:
        df = data

    # Convert all columns to strings
    df = df.astype(str)
    header = '| ' + ' | '.join(columns) + ' |'
    sep = '|' + '|'.join(['---' for _ in columns]) + '|'
    rows = []
    for _, row in df.iterrows():
        rows.append('| ' + ' | '.join(row[col] for col in columns) + ' |')

    return '\n'.join([header, sep] + rows)


def format_stat_block(stats_dict, title):
    """Format a statistics dict as an LLM-friendly text block"""
    lines = [f'### {title}', '']
    if isinstance(stats_dict, dict):
        for k, v in stats_dict.items():
            if isinstance(v, float):
                lines.append(f'- {k}: {v:.4f}')
            else:
                lines.append(f'- {k}: {v}')
    return '\n'.join(lines)
