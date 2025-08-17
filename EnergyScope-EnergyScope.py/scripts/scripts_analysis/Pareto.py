# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 13:26:35 2025

@author: reynt
"""

import yaml
import subprocess
import os

# === Paramètres à adapter ===
config_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/scripts/config_ref.yaml"
script_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/scripts/run_energyscope.py"
valeurs_cost = list(range(55000, 100001, 5000))  # De 55 000 à 100 000 par pas de 5 000
base_case_study_name = "GWP_min_"

# === Chargement de la config initiale
with open(config_path, 'r') as file:
    config_data = yaml.safe_load(file)

# === Boucle sur chaque valeur de cost_limit
for valeur in valeurs_cost:
    print(f"[🔁] Lancement pour cost_limit = {valeur}")

    # Modification de la valeur de coût
    config_data['cost_limit'] = valeur
    config_data['case_study'] = f"{base_case_study_name}{valeur}"

    # Écriture du fichier temporaire
    with open(config_path, 'w') as file:
        yaml.dump(config_data, file)

    # Exécution du script EnergyScope
    try:
        subprocess.run(["python", script_path], check=True)
        print(f"[✅] Simulation {valeur} terminée avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"[❌] Erreur lors de l’exécution pour cost_limit = {valeur} : {e}")
