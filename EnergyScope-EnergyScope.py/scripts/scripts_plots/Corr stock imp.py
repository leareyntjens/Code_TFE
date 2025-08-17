# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:15:06 2025

@author: reynt
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np

# === Paramètres ===
base_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/Scénarios de base"

# === Fonctions ===
def get_import_value(run_path):
    path = os.path.join(run_path, "output", "sankey", "input2sankey.csv")
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
        return total_imp
    except FileNotFoundError:
        print(f"[{run_path}] Fichier non trouvé")
        return None

def get_storage_value(run_path):
    path = os.path.join(run_path, "output", "sankey", "input2sankey.csv")
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
        return total_sto
    except FileNotFoundError:
        print(f"[{run_path}] Fichier non trouvé")
        return None

# === Collecte des données ===
scenarios = []
import_values = []
storage_values = []

for entry in sorted(os.listdir(base_path)):  # tri alphabétique pour cohérence visuelle
    full_path = os.path.join(base_path, entry)
    if os.path.isdir(full_path):
        imp = get_import_value(full_path)
        sto = get_storage_value(full_path)
        if imp is not None and sto is not None:
            scenarios.append(entry)
            import_values.append(imp)
            storage_values.append(sto)

# === Affichage graphique ===
plt.figure(figsize=(12, 6))
x = range(len(scenarios))

plt.plot(x, import_values, 'o-', label='Importations [TWh]', color='skyblue')
plt.plot(x, storage_values, 's-', label='Stockage [TWh]', color='orange')

#plt.xticks(x, scenarios, rotation=45, ha='right')
plt.xlabel("Scénarios")
plt.ylabel("Énergie [TWh]")
plt.xticks([])  # supprime tous les ticks de l'axe X
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle=':', linewidth=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

# === Collecte des données ===
scenario_names = []
import_values = []
storage_values = []

for entry in sorted(os.listdir(base_path)):
    full_path = os.path.join(base_path, entry)
    if os.path.isdir(full_path):
        imp = get_import_value(full_path)
        sto = get_storage_value(full_path)
        if imp is not None and sto is not None:
            scenario_names.append(entry)
            import_values.append(imp)
            storage_values.append(sto)



import numpy as np
import matplotlib.pyplot as plt

# Matrice de corrélation
data = np.array([storage_values, import_values])
corr_matrix = np.corrcoef(data)

# Heatmap stylée
fig, ax = plt.subplots(figsize=(5, 4.5))
cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Titres des axes
labels = ['Stockage', 'Importation']
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)

# Valeurs dans les cases
for i in range(2):
    for j in range(2):
        value = corr_matrix[i, j]
        ax.text(j, i, f"{value:.2f}", ha='center', va='center',
                color='white' if abs(value) > 0.5 else 'black',
                fontsize=13, fontweight='bold')

# Cadres fins autour des cellules
for edge in ['top', 'bottom', 'left', 'right']:
    ax.spines[edge].set_visible(False)

# Retrait des ticks
ax.tick_params(top=False, bottom=False, left=False, right=False)

# Barre de couleur
# cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', shrink=0.75, pad=0.1)
# cbar.set_label('Corrélation (Pearson)', fontsize=10)

# Titre et ajustement
plt.title("Matrice de corrélation", fontsize=13, weight='bold', pad=12)
plt.tight_layout()
plt.show()

# Barre de couleur
#fig.colorbar(cax, ax=ax, label='Corrélation')

#plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()

# === Graphe de corrélation ===
plt.figure(figsize=(8, 6))

# Scatter + lignes entre les points dans l'ordre des scénarios
#plt.plot(storage_values, import_values, 'o-', color='purple', linewidth=1.5, label='Scénarios')
plt.scatter(storage_values, import_values, color='purple', s=60, label='Scénarios')
#for i, name in enumerate(scenario_names):
 #   plt.annotate(name, (storage_values[i], import_values[i]), fontsize=8, xytext=(5,5), textcoords='offset points')

plt.xlabel("Stockage [TWh]")
plt.ylabel("Importations [TWh]")

plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()