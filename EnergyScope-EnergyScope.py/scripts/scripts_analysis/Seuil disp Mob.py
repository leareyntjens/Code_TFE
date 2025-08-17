# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 14:02:18 2025

@author: reynt
"""

# -*- coding: utf-8 -*-
"""
Script de seuil d’apparition et de domination pour une ou plusieurs familles de technologies de mobilité privée.
"""

import pandas as pd
import subprocess
from pathlib import Path
import yaml
from io import StringIO
from collections import defaultdict

# === CONFIGURATION ===
base_path = Path("C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py")
technos_csv_path = base_path / "Data/2050/Technologies.csv"
assets_path = base_path / "case_studies"
yaml_path = base_path / "scripts/config_ref.yaml"
script_path = base_path / "scripts/run_energyscope.py"
log_path = base_path / "scripts/log_app_disparition.csv"

# === DÉFINITION DES FAMILLES ===
familles = {
    "electrique": ["CAR_BEV"],
    "hybride": ["CAR_PHEV", "CAR_HEV"],
    "gaz": ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL"],
    "fuel_cell": ["CAR_FUEL_CELL"]
}

# === FAMILLES À MODIFIER (paramétrable ici) ===
familles_modifiees = ["hybride"]  # Exemple : modifier seulement le GWP des fuel_cell

# === PARAMÈTRES DE BOUCLE ===
delta = 20
gwp_min = 0
colonne = "gwp_constr"

# === Fonctions Utiles ===

def update_gwp(df, techno_list, value):
    for t in techno_list:
        mask = df['Technologies param'].str.strip() == t
        df.loc[mask, colonne] = value
    return df

def run_simulation(run_name):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['case_study'] = run_name
    config['cs_path'] = 'case_studies'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    subprocess.run(["python", str(script_path)], shell=True)

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

# === Initialisation des technos ciblées
techno_targets = [t for f, group in familles.items() if f in familles_modifiees for t in group]

# Lecture des valeurs initiales
tech_df_base = pd.read_csv(technos_csv_path, sep=';')
initial_gwp = {t: float(tech_df_base[tech_df_base['Technologies param'].str.strip() == t][colonne].values[0])
               for t in techno_targets}

gwp_val = min(initial_gwp.values())
found_appearance = {fam: False for fam in familles}
found_dominance = {fam: False for fam in familles}
i = 0


while gwp_val >= gwp_min:
    run_name = f"{i}_SD_IS_MOB_{familles_modifiees[0]}_gwp{int(round(gwp_val))}"
    print(f"\n🔁 Itération {i} – GWP = {gwp_val} → Run : {run_name}")

    # Modifier fichier Technologies.csv avec le nouveau GWP
    tech_df = pd.read_csv(technos_csv_path, sep=';')
    tech_df = update_gwp(tech_df, techno_targets, gwp_val)
    tech_df.to_csv(technos_csv_path, sep=';', index=False)

    # Lancer simulation
    run_simulation(run_name)

    # Lire résultats
    assets = read_assets(run_name)
    if assets is None:
        break

    # Regrouper les f par famille
    fs_by_fam = defaultdict(list)
    for fam, technos in familles.items():
        for t in technos:
            if t in assets.index:
                fs_by_fam[fam].append(float(assets.loc[t, ' f']))
            else:
                fs_by_fam[fam].append(0.0)

    # Sommes par famille
    fam_sums = {fam: sum(fs) for fam, fs in fs_by_fam.items()}
    fam_max = max(fam_sums, key=fam_sums.get)

    # Analyse et logging
    for fam, total_f in fam_sums.items():
        if not found_appearance[fam] and total_f < 1:
            print(f"✅ Disparition de {fam} détectée à GWP = {gwp_val:.2f}")
            found_appearance[fam] = True
        
        # Détection de perte de dominance
        if fam_max != familles_modifiees[0]:
            print(f"⚠️  La famille '{familles_modifiees[0]}' n'est plus dominante à GWP = {gwp_val:.2f} (dominante = {fam_max})")

        else:
            print(f"➡️ Dominante actuelle : {fam_max} (f = {fam_sums[fam_max]:.4f})")

        entry = {
            "run_name": run_name,
            "iteration": i,
            "famille": fam,
            "gwp_constr": gwp_val,
            "f_total": total_f,
            "appeared": found_appearance[fam],
            "dominant": (fam == fam_max),
            "familles_modifiees": ','.join(familles_modifiees)
        }
        try:
            df_log = pd.read_csv(log_path)
            df_log = pd.concat([df_log, pd.DataFrame([entry])], ignore_index=True)
        except FileNotFoundError:
            df_log = pd.DataFrame([entry])
        df_log.to_csv(log_path, index=False)

    # ❗ AJOUT : arrêt si f_total < 1 pour la famille modifiée
    f_target = fam_sums.get(familles_modifiees[0], None)
    if f_target is not None and f_target < 1:
        print(f"🛑 Seuil de disparition atteint pour {familles_modifiees[0]} (f = {f_target:.4f}) → arrêt.")
        break

    gwp_val += delta
    gwp_val = max(gwp_val, 0)
    i += 1
