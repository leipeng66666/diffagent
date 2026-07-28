"""
Project 1: Lock top 20 guest molecules -> automated analysis -> LLM interpretation -> generate report

Usage:
    python project1_by_guest.py              # Full auto: stats + LLM analysis + report
    python project1_by_guest.py --no-llm     # Stats only + generate prompts (no LLM calls)
    python project1_by_guest.py --guest CH4,CO2,N2  # Analyze specified molecules only

Output:
    output/final_report.md              <- Final analysis report (for direct reading)
    output/stats/*.json                 <- Statistical results per molecule
    output/figures/*.png                <- Charts
    output/llm_responses/*.md           <- LLM interpretation per molecule
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
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "project1_system.txt"

TOP_N = 20


# ============================================================
# LLM calls
# ============================================================

def load_system_prompt():
    """Load the system prompt from file"""
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding='utf-8')
    return "You are an expert in zeolite diffusion."


def call_llm(system_prompt, user_prompt, label=""):
    """Call LLM and return response text"""
    try:
        from llm_config import get_llm_client
        client = get_llm_client()
        print(f"  [LLM] Calling LLM for {label} ...")
        response = client.chat(system_prompt, user_prompt)
        n_chars = len(response)
        print(f"  [OK] LLM response length: {n_chars} chars")
        # Warn if response appears truncated (ends mid-sentence without proper ending)
        if n_chars < 500:
            print(f"  [WARN] Response is very short ({n_chars} chars) — may be truncated or API issue")
            return None
        # Check if response ends mid-sentence (no proper sentence-ending punctuation)
        last_char = response.strip()[-1] if response.strip() else ''
        if last_char not in '.。!！?？)）]】}」』"\'`*_~':
            print(f"  [WARN] Response may be truncated — does not end with sentence-ending punctuation (last char: '{last_char}')")
            print(f"  [WARN] Consider increasing MAX_TOKENS in llm_config.py")
        return response
    except Exception as e:
        print(f"  [ERR] LLM call failed: {e}")
        return None


# ============================================================
# Statistical analysis
# ============================================================

def analyze_one_guest(df_guest, guest_name):
    """Perform 7-dimension statistical analysis for one guest molecule"""
    result = {'guest_name': guest_name, 'n_total': len(df_guest)}
    print(f"\n{'='*50}")
    print(f"  [Stats] {guest_name} (n={len(df_guest)})")

    # 1. Zeolite effect (using std_zeolite_name as primary identifier)
    zeo_stats = utils.group_stats(df_guest, 'zeolite_group', min_count=2)
    zeo_cols = ['zeolite_group', 'count', 'mean_logD', 'median_logD', 'std_logD', 'D_range_orders']
    result['by_zeolite'] = json.loads(
        zeo_stats.head(15)[zeo_cols].to_json(orient='records', double_precision=4))
    result['kw_zeolite'] = utils.kruskal_test(df_guest, 'zeolite_group', min_group_size=3)

    # 2. Topology type
    topo_stats = utils.group_stats(df_guest, 'topology', min_count=2)
    topo_cols = ['topology', 'count', 'mean_logD', 'median_logD', 'std_logD']
    result['by_topology'] = json.loads(
        topo_stats[topo_cols].to_json(orient='records', double_precision=4))
    result['kw_topology'] = utils.kruskal_test(df_guest, 'topology', min_group_size=3)

    # 2.5 Topology x Si/Al range cross-analysis (critical: zeolite+Si/Al = complete material definition)
    toposial_stats = utils.group_stats(df_guest, 'topo_sial', min_count=2)
    toposial_cols = ['topo_sial', 'count', 'mean_logD', 'median_logD', 'std_logD']
    result['by_topo_sial'] = json.loads(
        toposial_stats.head(20)[toposial_cols].to_json(orient='records', double_precision=4))
    result['kw_topo_sial'] = utils.kruskal_test(df_guest, 'topo_sial', min_group_size=3)

    # 2.6 Experimental method effects (different methods probe different spatiotemporal scales)
    method_stats = utils.group_stats(df_guest, 'method_category', min_count=2)
    method_cols = ['method_category', 'count', 'mean_logD', 'median_logD', 'std_logD']
    result['by_method'] = json.loads(
        method_stats[method_cols].to_json(orient='records', double_precision=4))
    result['kw_method'] = utils.kruskal_test(df_guest, 'method_category', min_group_size=3)

    # 3. Si/Al ratio
    si_al_data = df_guest[['si_al_ratio_num', 'logD']].dropna()
    result['si_al_ratio'] = {
        'n_available': len(si_al_data),
        'correlation': utils.spearman_correlation(df_guest, 'si_al_ratio_num', 'logD')
        if len(si_al_data) >= 5 else None,
        'range': f"{si_al_data['si_al_ratio_num'].min():.1f} ~ {si_al_data['si_al_ratio_num'].max():.1f}"
        if len(si_al_data) > 0 else 'N/A',
    }
    # 3.5 method x Si/Al cross-check: reveal confounding between method type and Si/Al range
    sial_method = df_guest.dropna(subset=['si_al_ratio_num']).copy()
    if len(sial_method) >= 5:
        sial_method['sial_method'] = sial_method['si_al_range'] + ' | ' + sial_method['method_category']
        sm_stats = utils.group_stats(sial_method, 'sial_method', min_count=3)
        sm_cols = ['sial_method', 'count', 'mean_logD', 'median_logD', 'std_logD']
        result['by_sial_method'] = json.loads(
            sm_stats.head(20)[sm_cols].to_json(orient='records', double_precision=4))

    # 4. Exchangeable cations
    ion_stats = utils.group_stats(df_guest, 'ion_group', min_count=3)
    ion_cols = ['ion_group', 'count', 'mean_logD', 'median_logD', 'std_logD']
    result['by_ion'] = json.loads(
        ion_stats[ion_cols].to_json(orient='records', double_precision=4))
    mw = utils.mannwhitney_test(df_guest, 'ion_group')
    if isinstance(mw, list):
        result['mw_ion_pairs'] = mw[:10]

    # 5. Temperature
    result['arrhenius_overall'] = utils.arrhenius_fit(df_guest, min_points=5)
    arr_by_zeo = utils.arrhenius_fit(df_guest, by_group='zeolite_group', min_points=5)
    result['arrhenius_by_zeolite'] = {
        k: v for k, v in arr_by_zeo.items()
        if isinstance(v, dict) and 'error' not in v
    }

    # 6. Concentration/loading
    result['loading_correlation'] = utils.spearman_correlation(df_guest, 'loading', 'logD')
    result['concentration_correlation'] = utils.spearman_correlation(df_guest, 'concentration_num', 'logD')
    result['adsorption_correlation'] = utils.spearman_correlation(df_guest, 'adsorption_num', 'logD')

    return result


# ============================================================
# Visualization
# ============================================================

def generate_guest_plots(df_guest, guest_name):
    """Generate charts for one guest molecule"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    safe_name = guest_name.replace('/', '_').replace('\\', '_')

    # (a) Zeolite logD boxplot (using std_zeolite_name)
    zeo_counts = df_guest['zeolite_group'].value_counts()
    top_zeo = zeo_counts.head(12).index.tolist()
    plot_data = df_guest[df_guest['zeolite_group'].isin(top_zeo)].copy()
    if len(plot_data) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_data['zeolite_group'] = pd.Categorical(
            plot_data['zeolite_group'], categories=top_zeo, ordered=True)
        plot_data_sorted = plot_data.sort_values('zeolite_group')
        bp = plot_data_sorted.boxplot(column='logD', by='zeolite_group', ax=ax,
                                       patch_artist=True, showfliers=True,
                                       flierprops=dict(markersize=2, alpha=0.3))
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_zeo)))
        for patch, c in zip(bp.patches, colors):
            patch.set_facecolor(c)
        ax.set_title(f'Diffusion of {guest_name} in Different Zeolites')
        ax.set_xlabel('Zeolite'); ax.set_ylabel('log10(D)')
        plt.suptitle(''); plt.xticks(rotation=35, ha='right', fontsize=8)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f'{safe_name}_zeolites.png', dpi=150, bbox_inches='tight')
        plt.close()

    # (b) Arrhenius plot
    temp_data = df_guest[['T_value', 'D_value']].dropna()
    temp_data = temp_data[(temp_data['T_value'] > 0) & (temp_data['D_value'] > 0)]
    if len(temp_data) >= 5:
        fig, ax = plt.subplots(figsize=(8, 5))
        inv_T = 1000 / temp_data['T_value'].values
        logD = np.log10(temp_data['D_value'].values)
        ax.scatter(inv_T, logD, alpha=0.5, s=20, c='steelblue')
        try:
            from scipy.stats import linregress
            x_fit = np.linspace(inv_T.min(), inv_T.max(), 100)
            valid_fit = temp_data.copy()
            valid_fit['inv_T'] = 1.0 / valid_fit['T_value']
            valid_fit['lnD'] = np.log(valid_fit['D_value'])
            slope, intercept, r_val, _, _ = linregress(valid_fit['inv_T'].values, valid_fit['lnD'].values)
            y_fit = intercept + slope / (x_fit * 1000)
            ax.plot(x_fit, y_fit / 2.303, 'r--', linewidth=2,
                    label=f'Ea = {-slope * 8.314 / 1000:.1f} kJ/mol, R2 = {r_val**2:.3f}')
            ax.legend()
        except Exception:
            pass
        ax.set_xlabel('1000 / T [K^-1]'); ax.set_ylabel('log10(D)')
        ax.set_title(f'Arrhenius Plot: {guest_name}')
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f'{safe_name}_arrhenius.png', dpi=150, bbox_inches='tight')
        plt.close()

    # (c) Si/Al scatter plot
    si_al_data = df_guest[['si_al_ratio_num', 'logD']].dropna()
    if len(si_al_data) >= 5:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(si_al_data['si_al_ratio_num'], si_al_data['logD'], alpha=0.5, s=20, c='darkgreen')
        ax.set_xlabel('Si/Al Ratio'); ax.set_ylabel('log10(D)')
        ax.set_title(f'Si/Al vs D: {guest_name}')
        r = si_al_data[['si_al_ratio_num', 'logD']].corr(method='spearman').iloc[0, 1]
        ax.text(0.05, 0.95, f'Spearman r = {r:.3f} (n={len(si_al_data)})',
                transform=ax.transAxes, fontsize=10, va='top')
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f'{safe_name}_si_al.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"    Charts saved: {safe_name}_*.png")


# ============================================================
# LLM Prompt Building
# ============================================================

def build_llm_prompt(guest_name, stats_result):
    """Build structured prompt from statistical results"""
    s = stats_result
    lines = [f"# Guest Molecule: {guest_name}", f"Sample size: n = {s['n_total']}", ""]

    lines.append("## 1. Diffusion Coefficients in Different Zeolites")
    if s.get('by_zeolite'):
        lines.append("| Zeolite (std) | n | mean_logD | median_logD | std_logD | D_range(orders) |")
        lines.append("|--------------|-----|-----------|-------------|----------|------------------|")
        for row in s['by_zeolite'][:15]:
            lines.append(f"| {row['zeolite_group']} | {row['count']} | {row['mean_logD']} | "
                        f"{row['median_logD']} | {row['std_logD']} | {row['D_range_orders']} |")
        lines.append("")
    if s.get('kw_zeolite') and 'error' not in s['kw_zeolite']:
        kw = s['kw_zeolite']
        lines.append(f"Kruskal-Wallis: H={kw['H_statistic']}, p={kw['p_value']}, "
                    f"significant={'Yes' if kw['significant'] else 'No'}, eta2={kw['eta_squared']}")
        lines.append("")

    lines.append("## 2. Effect of Topology Type")
    if s.get('by_topology'):
        lines.append("| Topology | n | mean_logD | median_logD | std_logD |")
        lines.append("|----------|---|-----------|-------------|----------|")
        for row in s['by_topology']:
            lines.append(f"| {row['topology']} | {row['count']} | {row['mean_logD']} | "
                        f"{row['median_logD']} | {row['std_logD']} |")
        lines.append("")

    # 2.5 Topology x Si/Al cross-analysis (critical: same topology, different Si/Al = different behavior)
    if s.get('by_topo_sial'):
        lines.append("## 2.5 Topology x Si/Al Range Cross-Analysis")
        lines.append("| Topology+Si/Al | n | mean_logD | median_logD | std_logD |")
        lines.append("|---------------|-----|-----------|-------------|----------|")
        for row in s['by_topo_sial'][:20]:
            lines.append(f"| {row['topo_sial']} | {row['count']} | {row['mean_logD']} | "
                        f"{row['median_logD']} | {row['std_logD']} |")
        lines.append("")
        if s.get('kw_topo_sial') and 'error' not in s['kw_topo_sial']:
            kw = s['kw_topo_sial']
            lines.append(f"Kruskal-Wallis: H={kw['H_statistic']}, p={kw['p_value']}, "
                        f"significant={'Yes' if kw['significant'] else 'No'}, eta2={kw['eta_squared']}")
        lines.append("")

    # 2.6 Experimental method effects
    if s.get('by_method'):
        lines.append("## 2.6 Effect of Experimental Method (critical: different methods probe different spatiotemporal scales)")
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
                lines.append("**WARNING: Systematic bias exists between experimental methods! Method confounding must be considered when interpreting other dimensions.**")
        lines.append("")

    lines.append("## 3. Effect of Si/Al Ratio")
    si = s.get('si_al_ratio', {})
    lines.append(f"Valid samples: n={si.get('n_available', 0)}, range: {si.get('range', 'N/A')}")
    if si.get('correlation') and 'error' not in si['correlation']:
        c = si['correlation']
        lines.append(f"Spearman r={c['r']}, p={c['p_value']}, significant={'Yes' if c['significant'] else 'No'}")
    lines.append("")

    # CRITICAL: method x Si/Al cross-check to detect confounding
    if s.get('by_sial_method') and len(s['by_sial_method']) > 1:
        lines.append("### CRITICAL: Method x Si/Al Confounding Check")
        lines.append("If different Si/Al ranges are dominated by DIFFERENT experimental methods, the Si/Al correlation may be SPURIOUS — driven by method bias, not material chemistry.")
        lines.append("| Si/Al Range + Method | n | mean_logD | median_logD | std_logD |")
        lines.append("|---------------------|-----|-----------|-------------|----------|")
        for row in s['by_sial_method'][:20]:
            lines.append(f"| {row['sial_method']} | {row['count']} | {row['mean_logD']} | "
                        f"{row['median_logD']} | {row['std_logD']} |")
        lines.append("")
        lines.append("**Before attributing D differences to Si/Al, verify that the same method types are represented across Si/Al ranges. If one Si/Al range is dominated by MD (fast self-diffusion) while another is dominated by Uptake (slow transport diffusion), the apparent Si/Al effect is an artifact.**")
        lines.append("")

    lines.append("## 4. Effect of Exchangeable Cations")
    if s.get('by_ion') and len(s['by_ion']) > 1:
        lines.append("| Cation | n | mean_logD | median_logD | std_logD |")
        lines.append("|--------|---|-----------|-------------|----------|")
        for row in s['by_ion']:
            lines.append(f"| {row['ion_group']} | {row['count']} | {row['mean_logD']} | "
                        f"{row['median_logD']} | {row['std_logD']} |")
        lines.append("")
        if s.get('mw_ion_pairs'):
            lines.append("Ion pairs with largest difference:")
            for pair in s['mw_ion_pairs'][:5]:
                lines.append(f"  {pair['group1']} vs {pair['group2']}: "
                           f"delta_mean_logD={pair['mean_logD_1'] - pair['mean_logD_2']:.2f}, "
                           f"p={pair['p_value']}")
            lines.append("")

    lines.append("## 5. Temperature Effect (Arrhenius)")
    arr = s.get('arrhenius_overall', {})
    if arr and 'error' not in arr:
        lines.append(f"Activation energy Ea = {arr['Ea_kJ_per_mol']} kJ/mol, R_squared = {arr['R_squared']}, "
                    f"n = {arr['n_points']}, T range: {arr['T_range']}")
        lines.append("")
    if s.get('arrhenius_by_zeolite'):
        lines.append("Activation energy in different zeolites:")
        lines.append("| Zeolite | Ea (kJ/mol) | R_squared | n |")
        lines.append("|---------|-------------|-----------|-----|")
        for zeo, a in s['arrhenius_by_zeolite'].items():
            if isinstance(a, dict):
                lines.append(f"| {zeo} | {a.get('Ea_kJ_per_mol','?')} | {a.get('R_squared','?')} | {a.get('n_points','?')} |")
        lines.append("")

    lines.append("## 6. Effect of Concentration/Loading")
    for label, key in [('Loading', 'loading_correlation'),
                        ('Concentration', 'concentration_correlation'),
                        ('Adsorption', 'adsorption_correlation')]:
        corr = s.get(key)
        if corr and 'error' not in corr and corr.get('n', 0) >= 5:
            lines.append(f"{label}: r={corr['r']}, p={corr['p_value']}, n={corr['n']}, "
                        f"significant={'Yes' if corr['significant'] else 'No'}")

    lines.append("")
    lines.append("---")
    lines.append("Please analyze from the following perspectives:")
    lines.append("1. Rank the main influencing factors (from most to least important)")
    lines.append("2. How do topology and Si/Al ratio COMBINE to influence diffusion? (Note: the same topology at different Si/Al can differ by several orders of magnitude in D)")
    lines.append("3. **Is there systematic bias between different experimental methods?** MD simulation, PFG NMR, QENS, Uptake, etc. probe diffusion at DIFFERENT spatiotemporal scales —")
    lines.append("   self-diffusion vs. transport diffusion, microscopic vs. macroscopic. Could observed differences partially arise from method differences rather than material properties?")
    lines.append("4. What role do exchangeable cations and temperature play? What does the activation energy indicate?")
    lines.append("5. Does concentration/loading significantly affect diffusion?")
    lines.append("6. Are there any noteworthy anomalies or counter-intuitive findings?")
    return '\n'.join(lines)


def build_cross_prompt(all_results, guest_list):
    """Build cross-molecule comparison prompt"""
    lines = ["# Cross-Molecule Comprehensive Comparison", f"Total: {len(guest_list)} guest molecules", ""]

    lines.append("## Key Metrics Summary by Molecule")
    lines.append("| Molecule | n | logD_spread(orders) | Ea(kJ/mol) | Si/Al_corr_r | n_topologies |")
    lines.append("|----------|---|--------------------|------------|-------------|-------------|")
    for guest_name in guest_list:
        s = all_results.get(guest_name, {})
        arr = s.get('arrhenius_overall', {}) or {}
        si = s.get('si_al_ratio', {}) or {}
        corr = si.get('correlation', {}) or {}
        zeo_vals = [r['mean_logD'] for r in (s.get('by_zeolite') or [])]
        spread = f"{max(zeo_vals)-min(zeo_vals):.1f}" if len(zeo_vals) >= 2 else 'N/A'
        ea = f"{arr.get('Ea_kJ_per_mol','?')}" if arr and 'error' not in arr else 'N/A'
        sr = f"{corr.get('r','?')}" if corr and 'error' not in corr else 'N/A'
        n_topo = len(s.get('by_topology') or [])
        lines.append(f"| {guest_name} | {s.get('n_total','?')} | {spread} | {ea} | {sr} | {n_topo} |")

    lines.append("")
    lines.append("---")
    lines.append("Please provide a comprehensive cross-molecule analysis:")
    lines.append("1. What are the common patterns across the 20 molecules? Is the dominant role of topology+Si/Al ratio universal?")
    lines.append("2. Which molecules are most and least affected by Si/Al ratio? Why?")
    lines.append("3. What is the relationship between activation energy (Ea) and molecular size/polarity?")
    lines.append("4. Which molecules exhibit unique behavior? Any counter-intuitive findings?")
    lines.append("5. Implications for zeolite design and separation process optimization")
    return '\n'.join(lines)


# ============================================================
# Final Report Compilation
# ============================================================

def compile_final_report(guest_list, all_results, llm_responses, cross_response):
    """Compile statistical results + LLM interpretations into a final report"""
    lines = []
    lines.append(f"# Diffusion Coefficient Analysis Report (Project 1: By Guest Molecule)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Molecules analyzed: {len(guest_list)}")
    lines.append("")

    lines.append("## Table of Contents")
    for i, g in enumerate(guest_list, 1):
        lines.append(f"{i}. [{g}](#{g.replace(' ', '-').lower()})")
    lines.append(f"{len(guest_list)+1}. [Cross-Molecule Comparison](#cross-molecule-comparison)")
    lines.append("")

    # Per-molecule analysis
    for guest_name in guest_list:
        lines.append(f"## {guest_name}")
        lines.append("")

        # Statistical summary
        s = all_results.get(guest_name, {})
        lines.append(f"**Sample size**: {s.get('n_total', 'N/A')} records")
        arr = s.get('arrhenius_overall', {}) or {}
        if arr and 'error' not in arr:
            lines.append(f"**Activation energy**: {arr.get('Ea_kJ_per_mol', 'N/A')} kJ/mol")
        lines.append("")

        # Key statistics
        if s.get('by_zeolite') and len(s['by_zeolite']) >= 3:
            top3 = sorted(s['by_zeolite'], key=lambda x: x['mean_logD'], reverse=True)[:3]
            bot3 = sorted(s['by_zeolite'], key=lambda x: x['mean_logD'])[:3]
            lines.append(f"**Zeolites with highest D**: {', '.join(r['zeolite_group'] for r in top3)}")
            lines.append(f"**Zeolites with lowest D**: {', '.join(r['zeolite_group'] for r in bot3)}")
            lines.append("")

        # LLM interpretation
        response = llm_responses.get(guest_name)
        if response:
            lines.append("### LLM Analysis")
            lines.append(response)
        else:
            lines.append("*(LLM not run)*")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Cross-molecule synthesis
    lines.append("## Cross-Molecule Comparison")
    lines.append("")
    if cross_response:
        lines.append(cross_response)
    else:
        lines.append("*(LLM not run)*")
    lines.append("")

    return '\n'.join(lines)


# ============================================================
# Main workflow
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Project 1: Lock guest molecules, analyze factors influencing diffusion coefficients')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM calls, only generate stats and prompts')
    parser.add_argument('--guest', type=str, default='', help='Specify guest molecules to analyze, comma-separated (default: top 20)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Project 1: Guest Molecules -> Automated Analysis -> LLM Interpretation")
    print("=" * 60)

    use_llm = not args.no_llm
    if not use_llm:
        print("  [!] LLM calls disabled (--no-llm)")

    # ---- 1. Load data ----
    print("\n[1/5] Loading data...")
    df = utils.prepare_dataframe(DATA_PATH)
    print(f"  Total rows: {len(df)}, unique guest molecules: {df['guest_molecule'].nunique()}")

    # ---- 2. Select target molecules ----
    print(f"\n[2/5] Selecting target molecules...")
    if args.guest:
        top_guests = [g.strip() for g in args.guest.split(',')]
    else:
        top_guests = utils.get_top_n(df, 'guest_molecule', n=TOP_N)
    top_guests = [g for g in top_guests if pd.notna(g) and str(g).strip() != '']
    print(f"  Selected {len(top_guests)} molecules:")
    for i, g in enumerate(top_guests, 1):
        print(f"    {i:2d}. {g:20s} (n={(df['guest_molecule']==g).sum()})")

    # ---- 3. Stats + LLM analysis ----
    print(f"\n[3/5] Statistical analysis + LLM interpretation...")
    system_prompt = load_system_prompt()
    all_results = {}
    llm_responses = {}

    for guest_name in top_guests:
        df_guest = df[df['guest_molecule'] == guest_name].copy()

        # Statistics
        stats = analyze_one_guest(df_guest, guest_name)
        all_results[guest_name] = stats

        # Save stats JSON
        safe_name = guest_name.replace('/', '_').replace('\\', '_')
        with open(STATS_DIR / f'{safe_name}.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

        # Export CSV tables
        TABLES_DIR.mkdir(exist_ok=True)
        for table_key, csv_name in [
            ('by_zeolite', f'{safe_name}_by_zeolite.csv'),
            ('by_topology', f'{safe_name}_by_topology.csv'),
            ('by_topo_sial', f'{safe_name}_by_topo_sial.csv'),
            ('by_method', f'{safe_name}_by_method.csv'),
            ('by_sial_method', f'{safe_name}_by_sial_method.csv'),
            ('by_ion', f'{safe_name}_by_ion.csv'),
        ]:
            if stats.get(table_key):
                pd.DataFrame(stats[table_key]).to_csv(TABLES_DIR / csv_name, index=False, encoding='utf-8-sig')
        # Arrhenius table
        if stats.get('arrhenius_by_zeolite'):
            arr_rows = [{'zeolite': k, **v} for k, v in stats['arrhenius_by_zeolite'].items() if isinstance(v, dict)]
            if arr_rows:
                pd.DataFrame(arr_rows).to_csv(TABLES_DIR / f'{safe_name}_arrhenius.csv', index=False, encoding='utf-8-sig')

        # Charts
        try:
            generate_guest_plots(df_guest, guest_name)
        except Exception as e:
            print(f"    Chart generation failed: {e}")

        # LLM analysis
        prompt = build_llm_prompt(guest_name, stats)
        with open(LLM_DIR / f'{safe_name}_prompt.md', 'w', encoding='utf-8') as f:
            f.write(prompt)

        if use_llm:
            response = call_llm(system_prompt, prompt, label=guest_name)
            if response:
                llm_responses[guest_name] = response
                with open(LLM_DIR / f'{safe_name}_llm_response.md', 'w', encoding='utf-8') as f:
                    f.write(response)

    # ---- 4. Cross-molecule synthesis ----
    print(f"\n[4/5] Cross-molecule comparison...")
    cross_response = None
    cross_prompt = build_cross_prompt(all_results, top_guests)

    with open(OUTPUT_DIR / 'cross_molecule_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'n_molecules': len(top_guests),
            'molecule_list': top_guests,
            'molecule_comparison': [
                {
                    'guest_name': g,
                    'n': s.get('n_total'),
                    'Ea': (s.get('arrhenius_overall') or {}).get('Ea_kJ_per_mol') if 'error' not in (s.get('arrhenius_overall') or {}) else None,
                }
                for g, s in all_results.items()
            ]
        }, f, ensure_ascii=False, indent=2)

    # Export cross-molecule comparison table
    cross_rows = []
    for guest_name in top_guests:
        s = all_results.get(guest_name, {})
        arr = s.get('arrhenius_overall', {}) or {}
        si = s.get('si_al_ratio', {}).get('correlation', {}) or {}
        zeo_vals = [r['mean_logD'] for r in (s.get('by_zeolite') or [])]
        cross_rows.append({
            'guest_molecule': guest_name,
            'n_samples': s.get('n_total'),
            'logD_spread': round(max(zeo_vals)-min(zeo_vals), 2) if len(zeo_vals)>=2 else None,
            'Ea_kJ_per_mol': arr.get('Ea_kJ_per_mol') if arr and 'error' not in arr else None,
            'Ea_R2': arr.get('R_squared') if arr and 'error' not in arr else None,
            'si_al_spearman_r': si.get('r') if si and 'error' not in si else None,
            'n_topologies': len(s.get('by_topology') or []),
        })
    pd.DataFrame(cross_rows).to_csv(TABLES_DIR / 'cross_molecule_summary.csv', index=False, encoding='utf-8-sig')

    if use_llm:
        cross_sys = system_prompt + "\n\nThis is a cross-molecule synthesis task. Please synthesize and compare data across all molecules."
        cross_response = call_llm(cross_sys, cross_prompt, label="Cross-molecule")
        if cross_response:
            with open(LLM_DIR / 'cross_molecule_llm_response.md', 'w', encoding='utf-8') as f:
                f.write(cross_response)

    # ---- 5. Generate final report ----
    print(f"\n[5/5] Generating final report...")
    report = compile_final_report(top_guests, all_results, llm_responses, cross_response)
    report_path = OUTPUT_DIR / 'final_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"  Project 1 Complete!")
    print(f"  [Report] Final report: {report_path}")
    print(f"  [Stats]  JSON stats: {STATS_DIR}/")
    print(f"  [Tables] CSV tables: {TABLES_DIR}/")
    print(f"  [Plots]  Charts: {FIGURES_DIR}/")
    print(f"  [LLM]    LLM responses: {LLM_DIR}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
