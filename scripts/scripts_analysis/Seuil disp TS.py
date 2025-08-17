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
techno_targets = ["H2_STORAGE"]

delta = 1                # 🔁 Pas de décrément
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
i = 5

while True:
    run_name = f"{i}_CAS_Techno_{'_'.join(techno_targets)}_gwp{int(round(gwp_val))}"
    print(f"\n🔁 Itération {i} – GWP = {gwp_val:.2f} → Run : {run_name}")

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

    # Vérifier les f
    disappeared_flags = []

    for techno in techno_targets:
        if techno not in assets.index:
            print(f"⚠️ {techno} absente de assets.txt")
            disappeared_flags.append(True)
            continue

        f_tech = float(assets.loc[techno, ' f'])
        print(f"ℹ️ {techno} → f = {f_tech:.4f}")

        disappeared_flags.append(f_tech <= 1)

        # Logging
        entry = {
            "run_name": run_name,
            "iteration": i,
            "techno": techno,
            "gwp_constr": gwp_val,
            "f_value": f_tech,
            "disappeared": (f_tech <= 1)
        }

        try:
            df_log = pd.read_csv(log_path)
            df_log = pd.concat([df_log, pd.DataFrame([entry])], ignore_index=True)
        except FileNotFoundError:
            df_log = pd.DataFrame([entry])
        df_log.to_csv(log_path, index=False)

    # Si toutes les techno ont disparu (f ≤ 1)
    if all(disappeared_flags):
        print("✅ Toutes les technologies ciblées ont disparu du mix → arrêt des itérations")
        break

    # Mise à jour de l’itération
    gwp_val += delta
    i += 1
