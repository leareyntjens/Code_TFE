# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 12:03:46 2025

@author: reynt
"""

import pandas as pd
import subprocess
import os
import yaml
import csv
from pathlib import Path

# === CONFIGURATION ===
base_path = Path("C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py")
resources_csv_path = base_path / "Data/2050/Resources.csv"
yaml_path = base_path / "scripts/config_ref.yaml"
script_path = base_path / "scripts/run_energyscope.py"
case_study_dir = base_path / "case_studies"

# === Ressources à modifier ===
# Format : {nom_dans_csv: delta}
modifications = {
    "METHANOL_RE": 0.01,
}

# === Sauvegarde initiale
backup_path = resources_csv_path.with_name("Resources_backup.csv")
if not backup_path.exists():
    with open(resources_csv_path, "r", encoding="utf-8") as f_in:
        with open(backup_path, "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())
    print(f" Fichier sauvegardé sous : {backup_path.name}")

# === Initialisation
initial_values = {}
i = 0

# === Fonction de vérification d’usage dans le fichier output
def check_resources_unused(run_name, resources_to_check):
    file_path = os.path.join(case_study_dir, run_name, "output", "resources_breakdown.txt")
    if not os.path.exists(file_path):
        print(f"❌ Fichier non trouvé : {file_path}")
        return False
    try:
        df = pd.read_csv(file_path, sep='\t', skiprows=0)
        df['Name'] = df['Name'].str.strip()

        for res in resources_to_check:
            match = df[df['Name'] == res]
            if match.empty:
                print(f"⚠️ Ressource non trouvée dans output : {res}")
                return False
            used = float(match['Used'].values[0])
            print(f"🔎 {res} ➜ Used = {used:.5f}")
            if used > 1e-4:
                return False  # Encore utilisée
        return True  # Toutes à 0
    except Exception as e:
        print(f"⚠️ Erreur de lecture du fichier resources_breakdown.txt : {e}")
        return False

# === Boucle principale
while True:
    with open(resources_csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_line = 2
    header = lines[header_line].strip().split(';')
    data_lines = lines[header_line + 1:]
    df = pd.DataFrame([l.strip().split(';') for l in data_lines], columns=header)

    if i == 0:
        for res in modifications:
            mask = df['parameter name'].str.strip() == res
            if not mask.any():
                print(f" Ressource non trouvée : {res}")
            else:
                val = float(df.loc[mask, 'gwp_op'].values[0])
                initial_values[res] = val

    for res, delta in modifications.items():
        mask = df['parameter name'].str.strip() == res
        if not mask.any():
            continue
        if i == 0:
            continue
        new_val = initial_values[res] + i * delta
        df.loc[mask, 'gwp_op'] = f"{new_val:.5f}"

    with open(resources_csv_path, 'w', encoding='utf-8') as f:
        f.writelines(lines[:header_line + 1])
        for _, row in df.iterrows():
            f.write(';'.join(str(val) for val in row.values) + '\n')

    print(f"\n Simulation {i} | GWP_op ajusté de {i * list(modifications.values())[0]:.5f}")

    # Modifier le fichier YAML
    run_name = f"Resources_GWP_op_{i:02d}"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['case_study'] = run_name
    config['cs_path'] = 'case_studies'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Lancer EnergyScope
    result = subprocess.run(["python", str(script_path)], shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(" Échec de l'exécution d'EnergyScope.")
        print(" STDERR :\n", result.stderr)
        break

    # Vérifier si toutes les ressources ne sont plus utilisées
    if check_resources_unused(run_name, list(modifications.keys())):
        print("Aucune ressource cible utilisée. Fin.")
        break

    i += 1
