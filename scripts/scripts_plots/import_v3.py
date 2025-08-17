# -*- coding: utf-8 -*-
"""
Created on Wed May  7 11:45:27 2025

@author: reynt
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np

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
    #"0 fossil rapports": ["Prix 0 fossil rapports/lim_prix_55000_0_fossil_rap", "Prix 0 fossil rapports/lim_prix_58000_0_fossil_rap", "Prix 0 fossil rapports/lim_prix_60500_0_fossil_rap",
                   #       "Prix 0 fossil rapports/lim_prix_63000_0_fossil_rap", "Prix 0 fossil rapports/lim_prix_65500_0_fossil_rap", "Prix 0 fossil rapports/lim_prix_68500_0_fossil_rap"],
    #"150% rapports": ["Prix 150% rapports/lim_prix_55000_150%_rap", "Prix 150% rapports/lim_prix_58000_150%_rap", "Prix 150% rapports/lim_prix_60500_150%_rap",
                  #    "Prix 150% rapports/lim_prix_63000_150%_rap", "Prix 150% rapports/lim_prix_65500_150%_rap", "Prix 150% rapports/lim_prix_68500_150%_rap"]
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

def get_import_value(run_name):
    path = os.path.join(base_path, run_name, "output", "sankey", "input2sankey.csv")
    total_imp = 0.0
    try:
        with open(path, newline='', encoding='utf-8') as csvfile:
            rows = list(csv.reader(csvfile))[1:]
            for row in rows:
                source = row[0].strip().lower()
                try:
                    val = float(row[2])
                    if "imp." in source:
                        total_imp += val
                except:
                    continue
        return total_imp  # valeur brute en TWh
    except FileNotFoundError:
        print(f"[{run_name}] Fichier non trouvé")
        return None

def get_storage_value(run_name):
    path = os.path.join(base_path, run_name, "output", "sankey", "input2sankey.csv")
    total_sto = 0.0
    try:
        with open(path, newline='', encoding='utf-8') as csvfile:
            rows = list(csv.reader(csvfile))[1:]
            for row in rows:
                source = row[0].strip().lower()
                try:
                    val = float(row[2])
                    if "sto" in source:
                        total_sto += val
                except:
                    continue
        return total_sto  # valeur brute en TWh
    except FileNotFoundError:
        print(f"[{run_name}] Fichier non trouvé")
        return None

def get_real_value_sum(run_name):
    path = os.path.join(base_path, run_name, "output", "sankey", "input2sankey.csv")
    total = 0.0
    try:
        with open(path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            real_value_index = headers.index("realValue")
            for row in reader:
                try:
                    total += float(row[real_value_index])
                except:
                    continue
        return total
    except FileNotFoundError:
        print(f"[{run_name}] Fichier non trouvé")
        return None

# === Données ===
results_import_percent = {label: [] for label in scenarios}
results_import_twh = {label: [] for label in scenarios}
results_storage = {label: [] for label in scenarios}

for label, runs in scenarios.items():
    print(f"\n--- {label} ---")
    for run in runs:
        imp_pct = get_percentage(run, "imp.")
        imp_twh = get_import_value(run)
        sto = get_storage_value(run)
        total_real = get_real_value_sum(run)

        results_import_percent[label].append(imp_pct)
        results_import_twh[label].append(imp_twh)
        results_storage[label].append(sto)

        print(f"{run}: total realValue = {total_real:.2f} TWh")

# === Graphique Importation [%] ===
plt.figure(figsize=(10, 5))
for idx, (label, values) in enumerate(results_import_percent.items()):
    plt.plot(prix_values, values, 'o-', label=label)

plt.axvline(x=60500, color='red', linestyle='--', label='Limite prix')
for y in range(40, 81, 10):
    plt.axhline(y=y, color='gray', linestyle=':', linewidth=0.5)

plt.xlabel("Prix limite [M€]")
plt.ylabel("Importations [%]")
plt.legend()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(False)
plt.tight_layout()
plt.show()

# === Graphique Stockage en TWh ===
plt.figure(figsize=(10, 5))
for idx, (label, values) in enumerate(results_storage.items()):
    plt.plot(prix_values, values, 'o-', label=label)

plt.axvline(x=60500, color='red', linestyle='--', label='Limite prix')

plt.xlabel("Prix limite [M€]")
plt.ylabel("Stockage [TWh]")
plt.legend()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(False)
plt.tight_layout()
plt.show()

# === Graphique barres : Importation & Stockage en TWh ===
def plot_bar_import_vs_storage(prix_values, results_import_twh, results_storage):
    for label in scenarios:
        import_values = results_import_twh[label]
        storage_values = results_storage[label]

        x = np.arange(len(prix_values))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, import_values, width, label='Importations [TWh]', color='skyblue')
        ax.bar(x + width/2, storage_values, width, label='Stockage [TWh]', color='orange')

        ax.set_ylabel("Énergie [TWh]")
        ax.set_xlabel("Prix limite [M€]")
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in prix_values])
        ax.set_title(f"Importations et Stockage – Scénario: {label}")
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

# === Appel du graphique à barres ===
plot_bar_import_vs_storage(prix_values, results_import_twh, results_storage)
