# -*- coding: utf-8 -*-
"""
Pénalisation dynamique du GWP de construction pour forcer un changement de technologie dominante
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

# === Liste des technologies surveillées ===
watch_list = [
    "CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_HEV",
    "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"
]

dominance_column = "f"  # Capacité installée en GW
delta = 10  # Pénalisation en GWP
tech_history = []  # Historique des dominantes
initial_values = {}  # Valeurs de départ pour chaque techno pénalisée

# === Boucle principale ===
i = 0
while True:
    print(f"\n🔁 Simulation {i}")
    tech_df = pd.read_csv(tech_csv_path, sep=';')

    # === Appliquer la pénalisation à la techno dominante courante
    if i > 0:
        current_tech = tech_history[-1]
        idx = tech_df[tech_df["Technologies param"].str.strip() == current_tech].index
        if not idx.empty:
            idx = idx[0]
            if current_tech not in initial_values:
                initial_values[current_tech] = float(tech_df.loc[idx, "gwp_constr"])
            new_val = float(tech_df.loc[idx, "gwp_constr"]) + delta
            tech_df.loc[idx, "gwp_constr"] = new_val
            print(f"⬆️ Pénalisation de {current_tech} → gwp_constr = {new_val:.2f}")
        else:
            print(f"❌ Techno dominante non trouvée dans Technologies.csv : {current_tech}")
            break
    else:
        new_val = None
        current_tech = None

    tech_df.to_csv(tech_csv_path, sep=';', index=False)

    # === Définir le nom de simulation
    if i == 0:
        run_name = "initial_run"
    else:
        run_name = f"{current_tech}_{int(new_val)}".replace('.', '_')

    # === Modifier config YAML
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['case_study'] = run_name
    config['cs_path'] = 'case_studies'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

    # === Lancer EnergyScope
    result = subprocess.run(["python", str(script_path)], shell=True)
    if result.returncode != 0:
        print("❌ Échec de l'exécution d'EnergyScope.")
        break

    # === Lire le fichier assets.txt
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

    # === Identifier la techno dominante
    dominant_tech = None
    max_val = -1
    for tech in watch_list:
        try:
            val = assets_df.loc[tech, dominance_column]
            print(f"➡️ {tech} : {val} GW")
            if val > max_val:
                max_val = val
                dominant_tech = tech
        except KeyError:
            print(f"❌ Techno absente : {tech}")

    print(f"🏁 Nouvelle techno dominante : {dominant_tech}")

    # === Condition d'arrêt : techno déjà vue
    if dominant_tech in tech_history:
        print("✅ Retour à une techno déjà dominante. Fin de boucle.")
        break

    tech_history.append(dominant_tech)
    i += 1
