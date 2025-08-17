# -*- coding: utf-8 -*-
"""
Script de seuil d’apparition et de domination pour une ou plusieurs technologies de stockage
"""

import pandas as pd
import subprocess
from pathlib import Path
import yaml
from io import StringIO

# === CONFIGURATION ===
base_path = Path("C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py")
technos_csv_path = base_path / "Data/2050/Technologies.csv"
assets_path = base_path / "case_studies"
yaml_path = base_path / "scripts/config_ref.yaml"
script_path = base_path / "scripts/run_energyscope.py"
log_path = base_path / "scripts/log_app_disparition.csv"

# === PARAMÈTRES UTILISATEUR ===
techno_targets = ["TS_HIGH_TEMP"]

delta = -0.5                              # 🔁 Pas de décrément
gwp_min = 0                                 # 🔁 GWP min
colonne = "gwp_constr"                      # 🔁 Colonne modifiée

# === Liste des technologies de stockage ===
stockage_technos = [
    "BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS",
    "TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS",
    "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2",
    "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL",
    "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP",
    "GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE",
    "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"
]

def read_assets(run_name):
    path = assets_path / run_name / 'output' / 'assets.txt'
    if not path.exists():
        print(f"❌ Fichier non trouvé : {path}")
        return None
    with open(path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split('\t')
    df = pd.read_csv(StringIO(''.join(lines[2:])), sep='\t', header=None)
    df.columns = header
    df.set_index('TECHNOLOGIES', inplace=True)
    return df

def run_simulation(run_name):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['case_study'] = run_name
    config['cs_path'] = 'case_studies'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    subprocess.run(["python", str(script_path)], shell=True)

def update_gwp(df, techno_list, value):
    for t in techno_list:
        mask = df['Technologies param'].str.strip() == t
        df.loc[mask, colonne] = value
    return df

# === Initialisation
tech_df_base = pd.read_csv(technos_csv_path, sep=';')
initial_gwp = {}
for t in techno_targets:
    val = tech_df_base[tech_df_base['Technologies param'].str.strip() == t][colonne].values[0]
    initial_gwp[t] = float(val)

gwp_val = min(initial_gwp.values())  # On part de la valeur min
found_appearance = {t: False for t in techno_targets}
found_dominance = {t: False for t in techno_targets}
i = 0

while gwp_val >= gwp_min:
    run_name = f"{i}_Techno_{techno_targets[0]}_gwp{int(round(gwp_val))}"
    #run_name = f"{i}_Techno_BIOFUELS_gwp{int(round(gwp_val))}"
    print(f"\n🔁 Itération {i} – GWP = {gwp_val} → Run : {run_name}")

    # Modifier GWP pour toutes les technos
    tech_df = pd.read_csv(technos_csv_path, sep=';')
    tech_df = update_gwp(tech_df, techno_targets, gwp_val)
    tech_df.to_csv(technos_csv_path, sep=';', index=False)

    # Simulation
    run_simulation(run_name)

        

    # Lire résultats
    assets = read_assets(run_name)
    if assets is None:
        break

    # Extraire parts des techno de stockage
    stockage_fs = assets.loc[assets.index.isin(stockage_technos), ' f']
    techno_max = stockage_fs.idxmax()
    f_max = stockage_fs.max()

    for techno in techno_targets:
        if techno not in assets.index:
            print(f"⚠️ {techno} absente de assets.txt")
            continue

        f_tech = float(assets.loc[techno, ' f'])

        if not found_appearance[techno] and f_tech > 1:
            print(f"✅ Apparition de {techno} détectée à GWP = {gwp_val:.2f}")
            found_appearance[techno] = True

        if not found_dominance[techno] and techno == techno_max:
            print(f"🏆 {techno} devient dominante avec f = {f_tech:.4f}")
            found_dominance[techno] = True
        else:
            print(f"➡️ Dominante actuelle : {techno_max} (f = {f_max:.4f})")

        # Logging
        entry = {
            "run_name": run_name,
            "iteration": i,
            "techno": techno,
            "gwp_constr": gwp_val,
            "f_value": f_tech,
            "appeared": found_appearance[techno],
            "dominant": (techno == techno_max)
        }
        try:
            df_log = pd.read_csv(log_path)
            df_log = pd.concat([df_log, pd.DataFrame([entry])], ignore_index=True)
        except FileNotFoundError:
            df_log = pd.DataFrame([entry])
        df_log.to_csv(log_path, index=False)

    # Arrêt si gwp = 0
    if gwp_val == 0:
        print("⛔ GWP = 0 atteint → fin des itérations")
        break

    gwp_val += delta
    gwp_val = max(gwp_val, 0)
    i += 1