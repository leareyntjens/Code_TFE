# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:16:22 2025

@author: reynt
"""

import csv
import os
import matplotlib.pyplot as plt

# === Paramètres ===
base_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/Limite de prix"
prix_values = [55000, 58000, 60500, 63000, 65500, 68500]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

# === Scénarios analysés ===
scenarios = {
    "0 fossil": ["Prix 0 fossil/lim_prix_55000_0_fossil", "Prix 0 fossil/lim_prix_58000_0_fossil", "Prix 0 fossil/lim_prix_60500_0_fossil",
                 "Prix 0 fossil/lim_prix_63000_0_fossil", "Prix 0 fossil/lim_prix_65500_0_fossil", "Prix 0 fossil/lim_prix_68500_0_fossil"],
    "150%_Res": ["Prix 150% ressources/lim_prix_55000_150%", "Prix 150% ressources/lim_prix_58000_150%", "Prix 150% ressources/lim_prix_60500_150%",
                 "Prix 150% ressources/lim_prix_63000_150%", "Prix 150% ressources/lim_prix_65500_150%", "Prix 150% ressources/lim_prix_68500_150%"],
    "0 fossil rapports": ["Prix 0 fossil rapports/lim_prix_55000_0_fossil_rap", "Prix 0 fossil rapports/lim_prix_58000_0_fossil_rap", "Prix 0 fossil rapports/lim_prix_60500_0_fossil_rap",
                          "Prix 0 fossil rapports/lim_prix_63000_0_fossil_rap", "Prix 0 fossil rapports/lim_prix_65500_0_fossil_rap", "Prix 0 fossil rapports/lim_prix_68500_0_fossil_rap"],
    "150% rapports": ["Prix 150% rapports/lim_prix_55000_150%_rap", "Prix 150% rapports/lim_prix_58000_150%_rap", "Prix 150% rapports/lim_prix_60500_150%_rap",
                      "Prix 150% rapports/lim_prix_63000_150%_rap", "Prix 150% rapports/lim_prix_65500_150%_rap", "Prix 150% rapports/lim_prix_68500_150%_rap"]
}

# === Fonctions ===
def get_percentage(run_name, keyword):
    path = os.path.join(base_path, run_name, "output", "sankey", "input2sankey.csv")
    total = relevant = 0.0
    try:
        with open(path, newline='', encoding='utf-8') as csvfile:
            rows = list(csv.reader(csvfile))[1:]
            targets = {row[1].strip() for row in rows if len(row) > 1}
            for row in rows:
                source = row[0].strip()
                try:
                    val = float(row[2])
                    if keyword in source.lower():
                        relevant += val
                    if source not in targets:
                        total += val
                except:
                    continue
        return (relevant / total * 100) if total else None
    except FileNotFoundError:
        print(f"[{run_name}] Fichier non trouvé")
        return None

# === Récupération des données ===
results_import = {label: [] for label in scenarios}
results_storage = {label: [] for label in scenarios}

for label, runs in scenarios.items():
    for run in runs:
        results_import[label].append(get_percentage(run, "imp."))
        results_storage[label].append(get_percentage(run, "sto"))

# === Graphique 1 : Importation ===
plt.figure(figsize=(10, 5))
for idx, (label, values) in enumerate(results_import.items()):
    plt.plot(prix_values, values, 'o-', label=label)

plt.axvline(x=60500, color='red', linestyle='--', label='Limite prix')  # Ligne verticale

# Lignes horizontales légères en pointillé
for y in range(40, 81, 10):  # Ajuste selon l'échelle de tes données
    plt.axhline(y=y, color='gray', linestyle=':', linewidth=0.5)

plt.xlabel("Prix limite [M€]")
plt.ylabel("Importations [%]")
plt.legend()

# Suppression des axes du haut et de droite
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()

# === Graphique 2 : Stockage ===
plt.figure(figsize=(10, 5))
for idx, (label, values) in enumerate(results_storage.items()):
    plt.plot(prix_values, values, 'o-', label=label)

plt.axvline(x=60500, color='red', linestyle='--', label='Limite de prix')  # Ligne verticale

# Lignes horizontales légères en pointillé
for y in range(2, 20, 2):  # Ajuste selon l'échelle de tes données
    plt.axhline(y=y, color='gray', linestyle=':', linewidth=0.5)

plt.xlabel("Prix limite [M€]")
plt.ylabel("Stockage [%]")
plt.legend()

# Suppression des axes du haut et de droite
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.grid(False)
plt.tight_layout()
plt.show()
