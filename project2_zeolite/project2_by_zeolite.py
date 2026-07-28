"""
Project 2: Lock top 20 zeolites -> automated analysis -> LLM interpretation -> generate report

Usage:
    python project2_by_zeolite.py              # Full auto: stats + LLM analysis + report
    python project2_by_zeolite.py --no-llm     # Stats only + generate prompts (no LLM calls)
    python project2_by_zeolite.py --zeolite ZSM-5,silicalite-1  # Analyze specified zeolites only

Output:
    output/final_report.md              <- Final analysis report (for direct reading)
    output/stats/*.json                 <- Statistical results per zeolite
    output/figures/*.png                <- Charts
    output/llm_responses/*.md           <- LLM interpretation per zeolite
"""

import pandas as pd
import numpy as np
import json
import sys
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import utils

warnings.filterwarnings('ignore')

# ============================================================
# Path configuration
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "consolidated_results3_clean.csv"
OUTPUT_DIR = BASE_DIR / "output"
STATS_DIR = OUTPUT_DIR / "stats"
FIGURES_DIR = OUTPUT_DIR / "figures"
LLM_DIR = OUTPUT_DIR / "llm_responses"
TABLES_DIR = OUTPUT_DIR / "tables"
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "project2_system.txt"

TOP_N = 20

# ============================================================
# Kinetic diameter lookup table
# ============================================================
KINETIC_DIAMETER = {
    'He': 2.6, 'H2': 2.89, 'H2O': 2.65, 'water': 2.65,
    'CO2': 3.3, 'CO': 3.76, 'N2': 3.64, 'O2': 3.46,
    'Ar': 3.4, 'CH4': 3.8, 'methane': 3.8,
    'C2H4': 4.16, 'ethylene': 4.16,
    'C2H6': 4.44, 'ethane': 4.44,
    'C3H6': 4.68, 'propene': 4.68, 'propylene': 4.68,
    'C3H8': 4.3, 'propane': 4.3,
    'n-C4H10': 4.3, 'n-butane': 4.3, 'n-C4': 4.3,
    'n-C5H12': 4.3, 'n-pentane': 4.3,
    'n-C6H14': 4.3, 'n-hexane': 4.3,
    'neopentane': 6.2,
    'benzene': 5.85, 'toluene': 5.85,
    'p-xylene': 5.85, 'm-xylene': 6.8, 'o-xylene': 6.8,
    'ethylbenzene': 6.0, 'cyclohexane': 6.0,
    'methanol': 3.8, 'ethanol': 4.3,
    'NH3': 2.9, 'SO2': 3.6,
    'SF6': 5.5, 'CF4': 4.7,
    'i-C4H10': 5.0, 'isobutane': 5.0,
}


def get_kinetic_diameter(guest_name):
    if pd.isna(guest_name) or guest_name == '':
        return None
    name = str(guest_name).strip()
    if name in KINETIC_DIAMETER:
        return KINETIC_DIAMETER[name]
    name_lower = name.lower()
    for key, val in KINETIC_DIAMETER.items():
        if key.lower() == name_lower:
            return val
    return None if '/' in name else None


# ============================================================
# LLM calls
# ============================================================

def load_system_prompt():
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding='utf-8')
    return "You are an expert in zeolite diffusion."


def call_llm(system_prompt, user_prompt, label=""):
    try:
        from llm_config import get_llm_client
        client = get_llm_client()
        print(f"  [LLM] Calling LLM for {label} ...")
        response = client.chat(system_prompt, user_prompt)
        n_chars = len(response)
        print(f"  [OK] LLM response length: {n_chars} chars")
        if n_chars < 500:
            print(f"  [WARN] Response is very short ({n_chars} chars) — may be truncated or API issue")
            return None
        last_char = response.strip()[-1] if response.strip() else ''
        if last_char not in '.。!！?？)）]】}」』"\'`*_~':
            print(f"  [WARN] Response may be truncated — does not end with sentence-ending punctuation")
            print(f"  [WARN] Consider increasing MAX_TOKENS in llm_config.py")
        return response
    except Exception as e:
        print(f"  [ERR] LLM call failed: {e}")
        return None


# ============================================================
# Statistical analysis
# ============================================================

def analyze_one_zeolite(df_zeo, zeo_name):
    result = {'zeolite_name': zeo_name, 'n_total': len(df_zeo)}
    topo = df_zeo['topology'].iloc[0] if len(df_zeo) > 0 else 'Unknown'
    result['topology'] = topo
    print(f"\n{'='*50}")
    print(f"  [Stats] {zeo_name} [{topo}] (n={len(df_zeo)})")

    # 1. Guest molecules
    guest_stats = utils.group_stats(df_zeo, 'guest_molecule', min_count=2)
    guest_top = guest_stats.head(20).copy()
    guest_top['kinetic_diameter_A'] = guest_top['guest_molecule'].apply(get_kinetic_diameter)
    guest_cols = ['guest_molecule', 'count', 'mean_logD', 'median_logD', 'std_logD', 'D_range_orders', 'kinetic_diameter_A']
    result['by_guest'] = json.loads(guest_top[guest_cols].to_json(orient='records', double_precision=4))
    result['kw_guest'] = utils.kruskal_test(df_zeo, 'guest_molecule', min_group_size=3)

    # 2. Molecular size vs D
    size_data = df_zeo.copy()
    size_data['kd'] = size_data['guest_molecule'].apply(get_kinetic_diameter)
    size_valid = size_data[['kd', 'logD']].dropna()
    if len(size_valid) >= 5:
        result['kinetic_diameter_correlation'] = utils.spearman_correlation(size_data, 'kd', 'logD')

    # 2.5 Experimental method effects
    method_stats = utils.group_stats(df_zeo, 'method_category', min_count=2)
    method_cols = ['method_category', 'count', 'mean_logD', 'median_logD', 'std_logD']
    result['by_method'] = json.loads(
        method_stats[method_cols].to_json(orient='records', double_precision=4))
    result['kw_method'] = utils.kruskal_test(df_zeo, 'method_category', min_group_size=3)

    # 3. Si/Al ratio
    si_al_data = df_zeo[['si_al_ratio_num', 'logD']].dropna()
    result['si_al_ratio'] = {
        'n_available': len(si_al_data),
        'correlation': utils.spearman_correlation(df_zeo, 'si_al_ratio_num', 'logD')
        if len(si_al_data) >= 5 else None,
        'range': f"{si_al_data['si_al_ratio_num'].min():.1f} ~ {si_al_data['si_al_ratio_num'].max():.1f}"
        if len(si_al_data) > 0 else 'N/A',
    }
    # 3.5 Si/Al range breakdown (same zeolite, different Si/Al = different diffusion)
    sial_range_stats = utils.group_stats(df_zeo, 'si_al_range', min_count=2)
    sial_range_cols = ['si_al_range', 'count', 'mean_logD', 'median_logD', 'std_logD']
    result['by_si_al_range'] = json.loads(
        sial_range_stats[sial_range_cols].to_json(orient='records', double_precision=4))
    # Si/Al effect on individual guest molecules (grouped Spearman)
    result['si_al_by_guest'] = utils.spearman_correlation(
        df_zeo, 'si_al_ratio_num', 'logD', by_group='guest_molecule', min_n=5)
    # 3.6 method x Si/Al cross-check: detect confounding
    sial_method = df_zeo.dropna(subset=['si_al_ratio_num']).copy()
    if len(sial_method) >= 5:
        sial_method['sial_method'] = sial_method['si_al_range'] + ' | ' + sial_method['method_category']
        sm_stats = utils.group_stats(sial_method, 'sial_method', min_count=3)
        sm_cols = ['sial_method', 'count', 'mean_logD', 'median_logD', 'std_logD']
        result['by_sial_method'] = json.loads(
            sm_stats.head(20)[sm_cols].to_json(orient='records', double_precision=4))

    # 4. Exchangeable cations
    ion_stats = utils.group_stats(df_zeo, 'ion_group', min_count=2)
    ion_cols = ['ion_group', 'count', 'mean_logD', 'median_logD', 'std_logD']
    result['by_ion'] = json.loads(ion_stats[ion_cols].to_json(orient='records', double_precision=4))
    mw = utils.mannwhitney_test(df_zeo, 'ion_group')
    if isinstance(mw, list):
        result['mw_ion_pairs'] = mw[:10]

    # 5. Temperature
    result['arrhenius_overall'] = utils.arrhenius_fit(df_zeo, min_points=5)
    arr_by_guest = utils.arrhenius_fit(df_zeo, by_group='guest_molecule', min_points=5)
    result['arrhenius_by_guest'] = {
        k: v for k, v in arr_by_guest.items()
        if isinstance(v, dict) and 'error' not in v
    }

    # 6. Concentration
    result['loading_correlation'] = utils.spearman_correlation(df_zeo, 'loading', 'logD')
    result['concentration_correlation'] = utils.spearman_correlation(df_zeo, 'concentration_num', 'logD')

    return result


# ============================================================
# Visualization
# ============================================================

def generate_zeolite_plots(df_zeo, zeo_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    safe_name = zeo_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')

    # (a) Guest molecule boxplot
    guest_counts = df_zeo['guest_molecule'].value_counts()
    top_guests = guest_counts.head(12).index.tolist()
    plot_data = df_zeo[df_zeo['guest_molecule'].isin(top_guests)].copy()
    if len(plot_data) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_data['guest_molecule'] = pd.Categorical(plot_data['guest_molecule'], categories=top_guests, ordered=True)
        plot_data_sorted = plot_data.sort_values('guest_molecule')
        bp = plot_data_sorted.boxplot(column='logD', by='guest_molecule', ax=ax,
                                       patch_artist=True, showfliers=True,
                                       flierprops=dict(markersize=2, alpha=0.3))
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_guests)))
        for patch, c in zip(bp.patches, colors):
            patch.set_facecolor(c)
        ax.set_title(f'Diffusion in {zeo_name}'); ax.set_xlabel('Guest'); ax.set_ylabel('log10(D)')
        plt.suptitle(''); plt.xticks(rotation=35, ha='right', fontsize=9)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f'{safe_name}_guests.png', dpi=150, bbox_inches='tight')
        plt.close()

    # (b) Molecular size vs D
    size_data = df_zeo.copy()
    size_data['kd'] = size_data['guest_molecule'].apply(get_kinetic_diameter)
    size_valid = size_data[['kd', 'logD']].dropna()
    if len(size_valid) >= 5:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(size_valid['kd'], size_valid['logD'], alpha=0.5, s=30, c='steelblue')
        ax.set_xlabel('Kinetic Diameter (Å)'); ax.set_ylabel('log₁₀(D)')
        ax.set_title(f'Molecular Size vs D: {zeo_name}')
        r = size_valid[['kd', 'logD']].corr(method='spearman').iloc[0, 1]
        ax.text(0.05, 0.05, f'Spearman r = {r:.3f} (n={len(size_valid)})',
                transform=ax.transAxes, fontsize=10, va='bottom')
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f'{safe_name}_size_vs_D.png', dpi=150, bbox_inches='tight')
        plt.close()

    # (c) Arrhenius
    temp_data = df_zeo[['T_value', 'D_value']].dropna()
    temp_data = temp_data[(temp_data['T_value'] > 0) & (temp_data['D_value'] > 0)]
    if len(temp_data) >= 5:
        fig, ax = plt.subplots(figsize=(7, 5))
        inv_T = 1000 / temp_data['T_value'].values
        logD = np.log10(temp_data['D_value'].values)
        ax.scatter(inv_T, logD, alpha=0.5, s=20, c='coral')
        try:
            from scipy.stats import linregress
            x_fit = np.linspace(inv_T.min(), inv_T.max(), 100)
            valid_fit = temp_data.copy()
            valid_fit['inv_T'] = 1.0 / valid_fit['T_value']
            valid_fit['lnD'] = np.log(valid_fit['D_value'])
            slope, intercept, r_val, _, _ = linregress(valid_fit['inv_T'].values, valid_fit['lnD'].values)
            y_fit = intercept + slope / (x_fit * 1000)
            ax.plot(x_fit, y_fit / 2.303, 'r--', linewidth=2,
                    label=f'Ea = {-slope * 8.314 / 1000:.1f} kJ/mol, R² = {r_val**2:.3f}')
            ax.legend()
        except Exception:
            pass
        ax.set_xlabel('1000 / T [K⁻¹]'); ax.set_ylabel('log₁₀(D)')
        ax.set_title(f'Arrhenius Plot: {zeo_name}')
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f'{safe_name}_arrhenius.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"    Charts saved: {safe_name}_*.png")


# ============================================================
# LLM Prompt Building
# ============================================================

def build_llm_prompt(zeo_name, topo, stats_result):
    s = stats_result
    lines = [f"# Zeolite: {zeo_name} [{topo}]", f"Sample size: n = {s['n_total']}", ""]

    lines.append("## 1. Diffusion Coefficients of Different Guest Molecules in This Zeolite")
    if s.get('by_guest'):
        lines.append("| Guest Molecule | n | mean_logD | median_logD | Kinetic Diameter (A) |")
        lines.append("|---------------|-----|-----------|-------------|---------------------|")
        for row in s['by_guest'][:20]:
            kd = row.get('kinetic_diameter_A', '')
            kd_str = f"{kd:.1f}" if kd and kd != 'None' else '-'
            lines.append(f"| {row['guest_molecule']} | {row['count']} | {row['mean_logD']} | "
                        f"{row['median_logD']} | {kd_str} |")
        lines.append("")
    if s.get('kw_guest') and 'error' not in s['kw_guest']:
        kw = s['kw_guest']
        lines.append(f"Kruskal-Wallis: H={kw['H_statistic']}, p={kw['p_value']}, significant={'Yes' if kw['significant'] else 'No'}")
        lines.append("")

    lines.append("## 2. Kinetic Diameter vs D")
    kd_corr = s.get('kinetic_diameter_correlation')
    if kd_corr and 'error' not in kd_corr:
        lines.append(f"Spearman r = {kd_corr['r']}, p = {kd_corr['p_value']}, n = {kd_corr['n']}, significant={'Yes' if kd_corr['significant'] else 'No'}")
        lines.append("")

    # 2.5 Experimental method
    if s.get('by_method'):
        lines.append("## 2.5 Effect of Experimental Method (critical: different methods = different spatiotemporal scales)")
        lines.append("| Method | n | mean_logD | median_logD | std_logD |")
        lines.append("|--------|-----|-----------|-------------|----------|")
        for row in s['by_method'][:12]:
            lines.append(f"| {row['method_category']} | {row['count']} | {row['mean_logD']} | "
                        f"{row['median_logD']} | {row['std_logD']} |")
        lines.append("")
        if s.get('kw_method') and 'error' not in s['kw_method']:
            kw = s['kw_method']
            lines.append(f"Kruskal-Wallis: H={kw['H_statistic']}, p={kw['p_value']}, "
                        f"significant={'Yes' if kw['significant'] else 'No'}, eta2={kw['eta_squared']}")
            if kw['significant']:
                lines.append("**WARNING: Systematic bias between methods! Method confounding must be considered when interpreting guest selectivity.**")
        lines.append("")

    lines.append("## 3. Effect of Si/Al Ratio (critical: zeolite+Si/Al = complete material definition)")
    si = s.get('si_al_ratio', {})
    lines.append(f"Si/Al data available: n = {si.get('n_available', 0)}, range: {si.get('range', 'N/A')}")
    if si.get('correlation') and 'error' not in si['correlation']:
        c = si['correlation']
        lines.append(f"Overall Spearman r = {c['r']}, p = {c['p_value']}, significant={'Yes' if c['significant'] else 'No'}")
    lines.append("")

    # CRITICAL: method x Si/Al cross-check
    if s.get('by_sial_method') and len(s['by_sial_method']) > 1:
        lines.append("### CRITICAL: Method x Si/Al Confounding Check")
        lines.append("Verify that Si/Al correlation is not an artifact of method clustering:")
        lines.append("| Si/Al Range + Method | n | mean_logD | median_logD | std_logD |")
        lines.append("|---------------------|-----|-----------|-------------|----------|")
        for row in s['by_sial_method'][:20]:
            lines.append(f"| {row['sial_method']} | {row['count']} | {row['mean_logD']} | "
                        f"{row['median_logD']} | {row['std_logD']} |")
        lines.append("")
        lines.append("**If one Si/Al range is dominated by MD (fast self-diffusion) while another is dominated by Uptake (slow transport diffusion), the Si/Al correlation is SPURIOUS — an artifact of method bias, not material chemistry.**")
        lines.append("")

    # Si/Al range breakdown
    if s.get('by_si_al_range') and len(s['by_si_al_range']) > 1:
        lines.append("### Diffusion Coefficients in Different Si/Al Ranges")
        lines.append("| Si/Al Range | n | mean_logD | median_logD | std_logD |")
        lines.append("|------------|-----|-----------|-------------|----------|")
        for row in s['by_si_al_range']:
            if row['si_al_range'] != 'Unknown':
                lines.append(f"| {row['si_al_range']} | {row['count']} | {row['mean_logD']} | "
                           f"{row['median_logD']} | {row['std_logD']} |")
        lines.append("")

    # Si/Al effect on individual guests
    if s.get('si_al_by_guest') and isinstance(s['si_al_by_guest'], list):
        sig_guests = [x for x in s['si_al_by_guest'] if x['significant']]
        if sig_guests:
            lines.append("### Guest Molecules SENSITIVE to Si/Al Ratio (Si/Al significantly affects their diffusion)")
            lines.append("| Guest Molecule | Spearman r | p | n | Effect Direction |")
            lines.append("|---------------|-----------|-----|---|-----------------|")
            for x in sig_guests[:10]:
                direction = 'Higher Si/Al -> D increases' if x['r'] > 0 else 'Higher Si/Al -> D decreases'
                lines.append(f"| {x['group']} | {x['r']} | {x['p_value']} | {x['n']} | {direction} |")
            lines.append("")

        insig_guests = [x for x in s['si_al_by_guest'] if not x['significant']]
        if insig_guests:
            lines.append("### Guest Molecules INSENSITIVE to Si/Al Ratio")
            guest_names = [x['group'] for x in insig_guests[:8]]
            lines.append(f"{', '.join(guest_names)}")
            lines.append("")

    lines.append("")

    lines.append("## 4. Effect of Exchangeable Cations")
    if s.get('by_ion') and len(s['by_ion']) > 1:
        lines.append("| Cation | n | mean_logD | median_logD |")
        lines.append("|--------|---|-----------|-------------|")
        for row in s['by_ion']:
            lines.append(f"| {row['ion_group']} | {row['count']} | {row['mean_logD']} | {row['median_logD']} |")
        lines.append("")

    lines.append("## 5. Temperature Effect")
    arr = s.get('arrhenius_overall', {})
    if arr and 'error' not in arr:
        lines.append(f"Ea = {arr['Ea_kJ_per_mol']} kJ/mol, R_squared = {arr['R_squared']}, n = {arr['n_points']}")
        lines.append("")
    if s.get('arrhenius_by_guest'):
        lines.append("Activation energy for different guest molecules:")
        lines.append("| Guest Molecule | Ea (kJ/mol) | R_squared | n |")
        lines.append("|---------------|------------|-----------|-----|")
        for g, a in s['arrhenius_by_guest'].items():
            if isinstance(a, dict):
                lines.append(f"| {g} | {a.get('Ea_kJ_per_mol','?')} | {a.get('R_squared','?')} | {a.get('n_points','?')} |")
        lines.append("")

    lines.append("## 6. Effect of Concentration/Loading")
    for label, key in [('loading', 'loading_correlation'), ('concentration', 'concentration_correlation')]:
        corr = s.get(key)
        if corr and 'error' not in corr and corr.get('n', 0) >= 5:
            lines.append(f"{label}: r={corr['r']}, p={corr['p_value']}, n={corr['n']}, significant={'Yes' if corr['significant'] else 'No'}")

    lines.append("")
    lines.append("---")
    lines.append("Please analyze from the following perspectives:")
    lines.append("1. What dominates this zeolite's diffusion selectivity? Molecular size, Si/Al ratio, or exchangeable cations?")
    lines.append("2. How does Si/Al ratio regulate diffusion in this zeolite? (Note: is the Si/Al effect consistent across different guest molecules?)")
    lines.append("3. **Which experimental methods contribute data for this zeolite?** Is there systematic bias between methods (MD/PFG NMR/QENS/Uptake, etc.)?")
    lines.append("   If significant bias exists, be cautious in interpreting diffusion trends — observed differences may partially arise from method differences rather than material properties.")
    lines.append("4. Which guest molecules are Si/Al-sensitive and which are Si/Al-insensitive? What mechanism does this reveal?")
    lines.append("5. How temperature-sensitive is this zeolite? What does the activation energy distribution indicate?")
    lines.append("6. What unique diffusion behavior does this zeolite exhibit compared to other zeolites?")
    return '\n'.join(lines)


def build_cross_prompt(all_results, zeo_list):
    lines = ["# Cross-Zeolite Comprehensive Comparison", f"Total: {len(zeo_list)} zeolites", ""]
    lines.append("## Key Metrics Summary by Zeolite")
    lines.append("| Zeolite | Topology | n | Si/Al Range | D_spread | Ea(kJ/mol) | Size_Selectivity_r | Si/Al_corr_r |")
    lines.append("|---------|----------|---|------------|----------|------------|-------------------|-------------|")
    for z in zeo_list:
        s = all_results.get(z, {})
        arr = s.get('arrhenius_overall', {}) or {}
        guest_vals = [r['mean_logD'] for r in (s.get('by_guest') or [])]
        spread = f"{max(guest_vals)-min(guest_vals):.1f}" if len(guest_vals) >= 2 else 'N/A'
        ea = f"{arr.get('Ea_kJ_per_mol','?')}" if arr and 'error' not in arr else 'N/A'
        kd = s.get('kinetic_diameter_correlation') or {}
        size_r = f"{kd.get('r','?')}" if 'error' not in kd else 'N/A'
        si = s.get('si_al_ratio',{}).get('correlation',{}) or {}
        si_r = f"{si.get('r','?')}" if 'error' not in si else 'N/A'
        si_range = s.get('si_al_ratio',{}).get('range','?')
        lines.append(f"| {z} | {s.get('topology','?')} | {s.get('n_total','?')} | {si_range} | {spread} | {ea} | {size_r} | {si_r} |")
    lines.append("")
    lines.append("---")
    lines.append("Please analyze: 1) Common patterns 2) Differences across topology types 3) Universality of size selectivity 4) General impact of Si/Al 5) Implications for zeolite screening/design")
    return '\n'.join(lines)


# ============================================================
# Final Report Compilation
# ============================================================

def compile_final_report(zeo_list, all_results, llm_responses, cross_response):
    lines = [
        f"# Diffusion Coefficient Analysis Report (Project 2: By Zeolite)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Zeolites analyzed: {len(zeo_list)}", ""
    ]
    lines.append("## Table of Contents")
    for i, z in enumerate(zeo_list, 1):
        lines.append(f"{i}. [{z}](#{z.replace(' ','-').replace('(','').replace(')','').lower()})")
    lines.append(f"{len(zeo_list)+1}. [Cross-Zeolite Comparison](#cross-zeolite-comparison)")
    lines.append("")

    for zeo_name in zeo_list:
        s = all_results.get(zeo_name, {})
        lines.append(f"## {zeo_name} [{s.get('topology', '?')}]")
        lines.append(f"Sample size: {s.get('n_total', 'N/A')} records")
        arr = s.get('arrhenius_overall', {}) or {}
        if arr and 'error' not in arr:
            lines.append(f"Activation energy: {arr.get('Ea_kJ_per_mol', 'N/A')} kJ/mol")
        lines.append("")

        if s.get('by_guest') and len(s['by_guest']) >= 3:
            top3 = sorted(s['by_guest'], key=lambda x: x['mean_logD'], reverse=True)[:3]
            bot3 = sorted(s['by_guest'], key=lambda x: x['mean_logD'])[:3]
            lines.append(f"**Fastest diffusing molecules**: {', '.join(r['guest_molecule'] for r in top3)}")
            lines.append(f"**Slowest diffusing molecules**: {', '.join(r['guest_molecule'] for r in bot3)}")
            lines.append("")

        response = llm_responses.get(zeo_name)
        if response:
            lines.append("### LLM Analysis")
            lines.append(response)
        lines.append("\n---\n")

    lines.append("## Cross-Zeolite Comparison")
    if cross_response:
        lines.append(cross_response)
    lines.append("")

    return '\n'.join(lines)


# ============================================================
# Main workflow
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Project 2: Lock zeolites, analyze factors influencing diffusion coefficients')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM calls')
    parser.add_argument('--zeolite', type=str, default='', help='Specify zeolites to analyze, comma-separated')
    args = parser.parse_args()

    print("=" * 60)
    print("  Project 2: Zeolites -> Automated Analysis -> LLM Interpretation")
    print("=" * 60)

    use_llm = not args.no_llm
    if not use_llm:
        print("  [!] LLM calls disabled (--no-llm)")

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = utils.prepare_dataframe(DATA_PATH)
    print(f"  Total rows: {len(df)}, unique zeolites: {df['zeolite_group'].nunique()}")

    # 2. Select target zeolites
    print(f"\n[2/5] Selecting target zeolites...")
    if args.zeolite:
        top_zeolites = [z.strip() for z in args.zeolite.split(',')]
    else:
        top_zeolites = utils.get_top_n(df, 'zeolite_group', n=TOP_N)
    top_zeolites = [z for z in top_zeolites if pd.notna(z) and str(z).strip() != '']
    print(f"  Selected {len(top_zeolites)} zeolites:")
    for i, z in enumerate(top_zeolites, 1):
        topo = str(z).split('-')[0] if '-' in str(z) else str(z)
        print(f"    {i:2d}. {z:35s} [{topo:5s}] n={(df['zeolite_group']==z).sum()}")

    # 3. Stats + LLM
    print(f"\n[3/5] Statistical analysis + LLM interpretation...")
    system_prompt = load_system_prompt()
    all_results = {}
    llm_responses = {}

    for zeo_name in top_zeolites:
        df_zeo = df[df['zeolite_group'] == zeo_name].copy()
        topo = df_zeo['topology'].iloc[0] if len(df_zeo) > 0 else 'Unknown'

        stats = analyze_one_zeolite(df_zeo, zeo_name)
        all_results[zeo_name] = stats

        safe_name = zeo_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        with open(STATS_DIR / f'{safe_name}.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

        # Export CSV tables
        TABLES_DIR.mkdir(exist_ok=True)
        for table_key, csv_name in [
            ('by_guest', f'{safe_name}_by_guest.csv'),
            ('by_method', f'{safe_name}_by_method.csv'),
            ('by_si_al_range', f'{safe_name}_by_si_al_range.csv'),
            ('by_sial_method', f'{safe_name}_by_sial_method.csv'),
            ('by_ion', f'{safe_name}_by_ion.csv'),
        ]:
            if stats.get(table_key):
                pd.DataFrame(stats[table_key]).to_csv(TABLES_DIR / csv_name, index=False, encoding='utf-8-sig')
        if stats.get('arrhenius_by_guest'):
            arr_rows = [{'guest': k, **v} for k, v in stats['arrhenius_by_guest'].items() if isinstance(v, dict)]
            if arr_rows:
                pd.DataFrame(arr_rows).to_csv(TABLES_DIR / f'{safe_name}_arrhenius.csv', index=False, encoding='utf-8-sig')

        try:
            generate_zeolite_plots(df_zeo, zeo_name)
        except Exception as e:
            print(f"    Chart generation failed: {e}")

        prompt = build_llm_prompt(zeo_name, topo, stats)
        with open(LLM_DIR / f'{safe_name}_prompt.md', 'w', encoding='utf-8') as f:
            f.write(prompt)

        if use_llm:
            response = call_llm(system_prompt, prompt, label=zeo_name)
            if response:
                llm_responses[zeo_name] = response
                with open(LLM_DIR / f'{safe_name}_llm_response.md', 'w', encoding='utf-8') as f:
                    f.write(response)

    # 4. Cross-zeolite synthesis
    print(f"\n[4/5] Cross-zeolite comparison...")
    cross_response = None
    cross_prompt = build_cross_prompt(all_results, top_zeolites)

    with open(OUTPUT_DIR / 'cross_zeolite_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'n_zeolites': len(top_zeolites),
            'zeolite_list': top_zeolites,
            'comparison': [
                {'zeolite_name': z, 'topology': s.get('topology'), 'n': s.get('n_total'),
                 'Ea': (s.get('arrhenius_overall') or {}).get('Ea_kJ_per_mol')}
                for z, s in all_results.items()
            ]
        }, f, ensure_ascii=False, indent=2)

    # Export cross-zeolite comparison table
    cross_rows = []
    for z in top_zeolites:
        s = all_results.get(z, {})
        arr = s.get('arrhenius_overall', {}) or {}
        kd = s.get('kinetic_diameter_correlation') or {}
        si = (s.get('si_al_ratio') or {}).get('correlation', {}) or {}
        guest_vals = [r['mean_logD'] for r in (s.get('by_guest') or [])]
        cross_rows.append({
            'zeolite_name': z,
            'topology': s.get('topology', '?'),
            'n_samples': s.get('n_total'),
            'D_spread_log_orders': round(max(guest_vals)-min(guest_vals), 2) if len(guest_vals)>=2 else None,
            'n_guest_types': len(s.get('by_guest') or []),
            'Ea_kJ_per_mol': arr.get('Ea_kJ_per_mol') if arr and 'error' not in arr else None,
            'size_selectivity_spearman_r': kd.get('r') if kd and 'error' not in kd else None,
            'si_al_spearman_r': si.get('r') if si and 'error' not in si else None,
        })
    pd.DataFrame(cross_rows).to_csv(TABLES_DIR / 'cross_zeolite_summary.csv', index=False, encoding='utf-8-sig')

    if use_llm:
        cross_sys = system_prompt + "\n\nThis is a cross-zeolite synthesis task."
        cross_response = call_llm(cross_sys, cross_prompt, label="Cross-zeolite")
        if cross_response:
            with open(LLM_DIR / 'cross_zeolite_llm_response.md', 'w', encoding='utf-8') as f:
                f.write(cross_response)

    # 5. Generate final report
    print(f"\n[5/5] Generating final report...")
    report = compile_final_report(top_zeolites, all_results, llm_responses, cross_response)
    report_path = OUTPUT_DIR / 'final_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"  Project 2 Complete!")
    print(f"  [Report] Final report: {report_path}")
    print(f"  [Stats]  JSON stats: {STATS_DIR}/")
    print(f"  [Tables] CSV tables: {TABLES_DIR}/")
    print(f"  [Plots]  Charts: {FIGURES_DIR}/")
    print(f"  [LLM]    LLM responses: {LLM_DIR}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
