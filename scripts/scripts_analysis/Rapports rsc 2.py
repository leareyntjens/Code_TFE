import pandas as pd
import subprocess
import os
import yaml
from pathlib import Path
import time

# === CONFIGURATION ===
base_path = Path("C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py")
resources_csv_path = base_path / "Data/2050/Resources.csv"
yaml_path = base_path / "scripts/config_ref.yaml"
script_path = base_path / "scripts/run_energyscope.py"
case_study_dir = base_path / "case_studies"
resources_breakdown_path = lambda run: case_study_dir / run / "output" / "resources_breakdown.txt"

# === Ressources locales
local_resources = {"WOOD", "WET_BIOMASS", "URANIUM", "RES_WIND", "RES_SOLAR", "RES_HYDRO", "RES_GEO", "WASTE"}
delta = 0.01

# === Backup
backup_path = resources_csv_path.with_name("Resources_backup.csv")
if not backup_path.exists():
    with open(resources_csv_path, "r", encoding="utf-8") as f_in, open(backup_path, "w", encoding="utf-8") as f_out:
        f_out.write(f_in.read())
    print(f"✅ Backup sauvegardé sous : {backup_path.name}")

# === Initialisation des GWP initiaux
with open(resources_csv_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

header_line = 2
header = lines[header_line].strip().split(';')
data_lines = lines[header_line + 1:]
df_init = pd.DataFrame([l.strip().split(';') for l in data_lines], columns=header)
df_init["parameter name"] = df_init["parameter name"].str.strip()
initial_gwps = df_init.set_index("parameter name")["gwp_op"].astype(float).to_dict()

# === Fonction de détection de la ressource dominante (locale ou non)
def detect_dominante(run_name, allow_local=True):
    path = resources_breakdown_path(run_name)
    if not path.exists():
        return None
    df = pd.read_csv(path, sep='\t')
    df = df[df["Used"] > 1e-4]
    if not allow_local:
        df = df[~df["Name"].isin(local_resources)]
    if df.empty:
        return None
    return df.sort_values("Used", ascending=False).iloc[0]["Name"]

# === Boucle principale
i = 0
step_counter = {}

while True:
    run_temp = f"AutoRun_{i:02d}"

    # Mise à jour du fichier YAML
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config["case_study"] = run_temp
    config["cs_path"] = "case_studies"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Lancer EnergyScope
    result = subprocess.run(["python", str(script_path)], shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ EnergyScope a échoué")
        print(result.stderr)
        break

    # Attendre la génération du fichier output
    timeout = 10
    while not resources_breakdown_path(run_temp).exists() and timeout > 0:
        time.sleep(1)
        timeout -= 1
    if not resources_breakdown_path(run_temp).exists():
        print("❌ Fichier resources_breakdown.txt manquant")
        break

    # Identifier la ressource dominante parmi toutes (locales et non-locales)
    res_dominante_all = detect_dominante(run_temp, allow_local=True)
    if res_dominante_all is None:
        print("✅ Plus aucune ressource significative utilisée")
        break

    print(f"🔍 Ressource dominante : {res_dominante_all}")

    # Si la ressource dominante est locale → on arrête
    if res_dominante_all in local_resources:
        print(f"✅ {res_dominante_all} est locale et dominante. Fin.")
        break

    # Sinon, on modifie son GWP et relance
    res_dominante = res_dominante_all  # ici elle est forcément non-locale

    if res_dominante not in step_counter:
        step_counter[res_dominante] = 0
    step_counter[res_dominante] += 1
    new_gwp = initial_gwps[res_dominante] + step_counter[res_dominante] * delta

    # Lire et modifier le fichier Resources.csv
    with open(resources_csv_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    header_line = 2
    header = all_lines[header_line].strip().split(';')
    data_lines = all_lines[header_line + 1:]
    df = pd.DataFrame([l.strip().split(';') for l in data_lines], columns=header)
    df["parameter name"] = df["parameter name"].str.strip()

    mask = df["parameter name"] == res_dominante
    if not mask.any():
        print(f"❌ Ressource {res_dominante} non trouvée dans Resources.csv")
        break
    df.loc[mask, "gwp_op"] = f"{new_gwp:.5f}"

    # Réécriture manuelle du fichier CSV
    with open(resources_csv_path, 'w', encoding='utf-8') as f:
        f.writelines(all_lines[:header_line + 1])
        for _, row in df.iterrows():
            f.write(';'.join(str(val) for val in row.values) + '\n')

    # Renommer le dossier de simulation
    run_final = f"{res_dominante}_{new_gwp:.2f}"
    old_path = case_study_dir / run_temp
    final_path = case_study_dir / run_final
    if old_path.exists():
        old_path.rename(final_path)
        print(f"📁 Simulation enregistrée : {run_final}")
    else:
        print(f"⚠️ Dossier {run_temp} introuvable")

    i += 1
