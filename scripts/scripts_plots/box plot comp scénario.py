import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === PARAMÈTRES ===
base_path = 'C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies'
scenarios = {
    'Scénario_1': 'SA_TS_20%_imp_fin',
    'Scénario_2': 'SA_TS_75%_imp_fin'
}
num_runs = 52
file_name = 'gwp_breakdown.txt'
column_name = 'GWP_constr'
key_column = 'Name'

# === CATÉGORIES ===
groupes = {
    "production": ["PV", "WIND_ONSHORE", "WIND_OFFSHORE", "NUCLEAR", "CCGT", "CCGT_AMMONIA", "COAL_US", "COAL_IGCC", "HYDRO_RIVER", "GEOTHERMAL", "H2_ELECTROLYSIS"],
    "stockage": ["BATT_LI", "PHS", "TS_DHN_DAILY", "H2_STORAGE", "TS_DHN_SEASONAL", "TS_HIGH_TEMP", "CO2_STORAGE"],
}

# === CONVERSION TXT -> CSV ===
def convert_to_csv(run_folder, base_path, num_runs, file_name):
    for run in range(num_runs):
        txt_path = os.path.join(base_path, run_folder, f'Run_{run}', 'output', file_name)
        csv_path = txt_path.replace('.txt', '.csv')
        if not os.path.exists(txt_path):
            continue
        try:
            df = pd.read_csv(txt_path, sep=None, engine='python', skiprows=[1])
            df.columns = df.columns.str.strip()
            df.to_csv(csv_path, index=False)
        except Exception:
            continue

# === EXTRACTION POURCENTAGES ===
all_percent_data = []

for scenario_label, run_folder in scenarios.items():
    convert_to_csv(run_folder, base_path, num_runs, file_name)

    for run in range(num_runs):
        csv_path = os.path.join(base_path, run_folder, f'Run_{run}', 'output', file_name.replace('.txt', '.csv'))
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
        except Exception:
            continue
        if column_name not in df.columns or key_column not in df.columns:
            continue

        df = df[[key_column, column_name]].dropna()
        total_gwp = df[column_name].sum()
        if total_gwp == 0:
            continue

        for category, tech_list in groupes.items():
            cat_sum = df[df[key_column].isin(tech_list)][column_name].sum()
            pourcent = 100 * cat_sum / total_gwp
            all_percent_data.append({
                'Scenario': scenario_label,
                'Category': category,
                'Percentage': pourcent
            })

# === CRÉATION DU BOXPLOT ===
df_percent = pd.DataFrame(all_percent_data)

plt.figure(figsize=(12, 6))
ax = sns.boxplot(
    data=df_percent,
    x='Scenario',
    y='Percentage',
    hue='Category',
    palette='Set2',           # Palette douce et lisible
    showfliers=True,          # Affiche les points extrêmes (moustaches complètes)
    width=0.5
)

sns.despine(top=True, right=True)

# Annotation des moyennes
grouped = df_percent.groupby(['Scenario', 'Category'])['Percentage'].mean().reset_index()
scenarios_list = list(df_percent['Scenario'].unique())
categories_list = list(groupes.keys())

for idx, row in grouped.iterrows():
    scenario = row['Scenario']
    category = row['Category']
    mean_val = row['Percentage']
    scen_idx = scenarios_list.index(scenario)
    cat_offset = (categories_list.index(category) - 0.5) * 0.25  # centrage
    xloc = scen_idx + cat_offset
    ax.text(xloc, mean_val + 1.5, f'{mean_val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

# Titres et légendes
plt.title("Distribution des parts (%) du GWP de construction par catégorie", fontsize=13, fontweight='bold')
plt.ylabel("Part en % du GWP de construction", fontsize=11)
plt.xlabel("Scénario", fontsize=11)
plt.legend(title="Catégorie", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
