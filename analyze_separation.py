import sys
sys.path.insert(0, r'c:\Users\Administrator\Desktop\ai agent')
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv(r'c:\Users\Administrator\Desktop\ai agent\uploads\classified_diffusion_data.csv')

# Filter for methane and CO2
ch4_keywords = ['methane', 'CH4', 'ch4']
co2_keywords = ['carbon dioxide', 'CO2', 'co2']

def is_ch4(s):
    if pd.isna(s): return False
    s = str(s).lower()
    return any(k.lower() in s for k in ch4_keywords)

def is_co2(s):
    if pd.isna(s): return False
    s = str(s).lower()
    return any(k.lower() in s for k in co2_keywords)

df_ch4 = df[df['material_formula'].apply(is_ch4)].copy()
df_co2 = df[df['material_formula'].apply(is_co2)].copy()

# Find zeolites that have both CH4 and CO2 data
ch4_zeolites = set(df_ch4['zeolite_formula'].dropna().unique())
co2_zeolites = set(df_co2['zeolite_formula'].dropna().unique())
common_zeolites = ch4_zeolites & co2_zeolites

# For each common zeolite, find pairs at matching temperatures
results = []

for zeolite in common_zeolites:
    if not zeolite or str(zeolite).strip() == '':
        continue
    z_ch4 = df_ch4[df_ch4['zeolite_formula'] == zeolite][['temperature_formula', 'converted_value']]
    z_co2 = df_co2[df_co2['zeolite_formula'] == zeolite][['temperature_formula', 'converted_value']]

    for _, ch4_row in z_ch4.iterrows():
        for _, co2_row in z_co2.iterrows():
            if ch4_row['temperature_formula'] == co2_row['temperature_formula']:
                try:
                    ch4_val = float(ch4_row['converted_value'])
                    co2_val = float(co2_row['converted_value'])
                except:
                    continue
                if ch4_val > 0 and co2_val > 0:
                    ratio = max(ch4_val, co2_val) / min(ch4_val, co2_val)
                    log_ratio = np.log10(ratio)
                    faster = 'CO2' if co2_val > ch4_val else 'CH4'
                    results.append({
                        'zeolite': zeolite,
                        'temperature': ch4_row['temperature_formula'],
                        'ch4_D': ch4_val,
                        'co2_D': co2_val,
                        'ratio': ratio,
                        'log10_ratio': log_ratio,
                        'faster_gas': faster
                    })

results_df = pd.DataFrame(results)

# Deduplicate: keep best pair per (zeolite, temperature)
best_df = (
    results_df
    .sort_values('log10_ratio', ascending=False)
    .drop_duplicates(subset=['zeolite', 'temperature'], keep='first')
    .sort_values('log10_ratio', ascending=False)
    .reset_index(drop=True)
)

# Best unique pair per zeolite (across all temperatures)
best_per_zeolite = (
    best_df
    .drop_duplicates(subset=['zeolite'], keep='first')
    .reset_index(drop=True)
)

SEP = '=' * 100
DSEP = '-' * 100

print(SEP)
print('  CH4 / CO2 DIFFUSIVITY SEPARATION ANALYSIS IN ZEOLITES')
print('  Metric: log10(D_max / D_min) -- higher value = greater separation potential')
print(SEP)

print()
print('DATASET SUMMARY')
print(DSep := '-' * 50)
print(f'  CH4 entries                          : {len(df_ch4)}')
print(f'  CO2 entries                          : {len(df_co2)}')
print(f'  Zeolites with data for BOTH gases    : {len(common_zeolites)}')
print(f'  Matched temperature pairs (raw)      : {len(results_df)}')
print(f'  Unique (zeolite, temperature) pairs  : {len(best_df)}')

HDR = f'  {"Rank":<5} {"Zeolite":<15} {"Temp":<8} {"D(CH4) [m2/s]":<18} {"D(CO2) [m2/s]":<18} {"Ratio":<16} {"Log10(ratio)":<14} {"Faster gas"}'
print()
print(SEP)
print('  TOP 10 ZEOLITES  (best temperature pair, one entry per zeolite)')
print(SEP)
print(HDR)
print(DSep)
for i, row in best_per_zeolite.head(10).iterrows():
    print(f'  {i+1:<5} {row["zeolite"]:<15} {row["temperature"]:<8} '
          f'{row["ch4_D"]:<18.3e} {row["co2_D"]:<18.3e} '
          f'{row["ratio"]:<16.2e} {row["log10_ratio"]:<14.2f} {row["faster_gas"]}')

print()
print(SEP)
print('  ALL UNIQUE (ZEOLITE, TEMPERATURE) PAIRS WITH LOG10(RATIO) >= 2 ORDERS OF MAGNITUDE')
print(SEP)
large = best_df[best_df['log10_ratio'] >= 2.0].reset_index(drop=True)
print(f'  Total pairs meeting threshold: {len(large)}')
print()
print(HDR)
print(DSep)
for i, row in large.iterrows():
    print(f'  {i+1:<5} {row["zeolite"]:<15} {row["temperature"]:<8} '
          f'{row["ch4_D"]:<18.3e} {row["co2_D"]:<18.3e} '
          f'{row["ratio"]:<16.2e} {row["log10_ratio"]:<14.2f} {row["faster_gas"]}')

print()
print(SEP)
print('  CONCLUSION')
print(SEP)
best = best_per_zeolite.iloc[0]
print(f'  Best zeolite for CH4/CO2 separation: {best["zeolite"]}')
print(f'  Best temperature                    : {best["temperature"]}')
print(f'  D(CH4)                              : {best["ch4_D"]:.3e} m2/s')
print(f'  D(CO2)                              : {best["co2_D"]:.3e} m2/s')
print(f'  Diffusivity ratio (D_max/D_min)     : {best["ratio"]:.3e}')
print(f'  Log10(ratio)                        : {best["log10_ratio"]:.2f} orders of magnitude')
print(f'  Faster diffusing gas                : {best["faster_gas"]}')
print()
print('  Ranking of ALL zeolites by best log10(ratio):')
for i, row in best_per_zeolite.iterrows():
    print(f'    {i+1:>2}. {row["zeolite"]:<15} at {row["temperature"]}:  '
          f'log10(ratio) = {row["log10_ratio"]:.2f}  '
          f'(CH4={row["ch4_D"]:.3e}, CO2={row["co2_D"]:.3e}, faster={row["faster_gas"]})')
print(SEP)
