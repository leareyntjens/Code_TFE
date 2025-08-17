# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 18:43:37 2025

@author: reynt
"""

import pandas as pd
import subprocess
from pathlib import Path
import yaml
from io import StringIO

# === CONFIGURATION ===
base_path = Path("C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py")
tech_csv_path = base_path / "Data/2050/Technologies.csv"
yaml_path = base_path / "scripts/config_ref.yaml"
script_path = base_path / "scripts/run_energyscope.py"
case_study_dir = base_path / "case_studies"

# === Paramètres
target_tech = "CAR_GASOLINE"  # Techno qu'on veut voir redevenir dominante
modification_column = "gwp_constr"
starting_value = 5000  # Tu adaptes à ta valeur limite trouvée
delta = -50            # On réduit progressivement
dominance_column = "f"  # Colonne de comparaison dans assets.txt

watch_list = [
    "CAR_GASOLINE",
    "CAR_DIESEL",
    "CAR_NG",
    "CAR_HEV",
    "CAR_PHEV",
    "CAR_BEV",
    "CAR_FUEL_CELL"
]

# === Chargement fichier de base
tech_df = pd.read_csv(tech_csv_path, sep=';')
row_idx = tech_df[tech_df["Technologies name"].str.strip() == target_tech].index[0]

# === Initialisation
i = 0
found = False
dominant_history = []

while True:
    current_val = starting_value + i * delta
    if current_val < 0:
        print("⛔ Valeur GWP devenue négative, arrêt.")
        break

    print(f"\n🔁 Simulation {i} — {target_tech} = {current_val:.2f}")

    # Modifier GWP dans le CSV
    tech_df.loc[row_idx, modification_column] = current_val
    tech_df.to_csv(tech_csv_path, sep=';', index=False)

    # Modifier config
    run_name = f"ReverseSearch_{target_tech}_{current_val:.0f}".replace('.', '_')
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['case_study'] = run_name
    config['cs_path'] = 'case_studies'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

    # Lancer EnergyScope
    result = subprocess.run(["python", str(script_path)], shell=True)
    if result.returncode != 0:
        print("❌ Échec de l'exécution d'EnergyScope.")
        break

    # Lire assets.txt
    assets_path = case_study_dir / run_name / "output" / "assets.txt"
    if not assets_path.exists():
        print(f"⚠️ Fichier non trouvé : {assets_path}")
        break

    with open(assets_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split('\t')
    data_lines = lines[2:]
    csv_like = StringIO(''.join(data_lines))
    assets_df = pd.read_csv(csv_like, sep='\t', header=None)
    assets_df.columns = [col.strip() for col in header]
    assets_df.set_index("TECHNOLOGIES", inplace=True)

    # Identifier la techno dominante
    dominant_tech = None
    max_val = -1
    for tech in watch_list:
        try:
            val = assets_df.loc[tech, dominance_column]
            if val > max_val:
                max_val = val
                dominant_tech = tech
        except KeyError:
            continue

    print(f"🏁 Dominante actuelle : {dominant_tech}")
    dominant_history.append(dominant_tech)

    if dominant_tech == target_tech:
        print(f"✅ {target_tech} est redevenue dominante à GWP = {current_val:.2f}")
        found = True
        break

    i += 1

if not found:
    print("❌ Aucun retour de dominance détecté.")
