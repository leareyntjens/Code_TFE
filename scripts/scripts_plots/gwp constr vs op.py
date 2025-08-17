# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 12:01:36 2025
@author: reynt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Paramètres utilisateur ===
base_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/Scénarios de base"

scenarios_to_include = [
    "Mod_techno_gwp_constr", "Mod_techno_gwp_tot", "Capa_30%_gwp_constr", "Capa_30%_gwp_tot",
    "Capa_50%_gwp_constr", "Capa_50%_gwp_tot", "Ressources_150%_gwp_constr", "Ressources_150%_gwp_tot"
]

scenario_names = [
    "Mod_techno_gwp_constr", "Mod_techno_gwp_tot", "Capa_30%_gwp_constr", "Capa_30%_gwp_tot",
    "Capa_50%_gwp_constr", "Capa_50%_gwp_tot", "Ressources_150%_gwp_constr", "Ressources_150%_gwp_tot"
]

gwp_constr_values = []
gwp_op_values = []

print("\n📊 GWP d'opération par scénario :\n")

for scenario in scenarios_to_include:
    file_path = os.path.join(base_path, scenario, "output", "gwp_breakdown.txt")

    try:
        df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
        gwp_constr = df.iloc[:, 1].sum()
        gwp_op = df.iloc[:, 2].sum()

        gwp_constr_values.append(gwp_constr)
        gwp_op_values.append(gwp_op)

        print(f"▶️ {scenario:<30} ➜ GWP d'opération : {gwp_op:.2f} ktCO₂eq/an")

    except Exception as e:
        print(f"[{scenario}] ❌ Erreur : {e}")
        gwp_constr_values.append(0)
        gwp_op_values.append(0)

# === Création du graphe ===
x = np.arange(len(scenario_names))
gwp_tot_values = [c + o for c, o in zip(gwp_constr_values, gwp_op_values)]

fig, ax = plt.subplots(figsize=(12, 6))

bars_constr = ax.bar(x, gwp_constr_values, label="GWP de construction", color="lightcoral", width=0.4)
bars_op = ax.bar(x, gwp_op_values, bottom=gwp_constr_values, label="GWP d'opération", color="#9ECAE1", width=0.4)

# Affichage des pourcentages dans les barres
for i in range(len(scenario_names)):
    total = gwp_tot_values[i]
    if total > 0:
        pct_constr = 100 * gwp_constr_values[i] / total
        pct_op = 100 * gwp_op_values[i] / total

        ax.text(x[i], gwp_constr_values[i] / 2, f"{pct_constr:.1f}%", ha='center', va='center', fontsize=10, color='black')
        ax.text(x[i], gwp_constr_values[i] + gwp_op_values[i] / 2, f"{pct_op:.1f}%", ha='center', va='center', fontsize=10, color='black')

# Mise en forme
ax.set_xticks(x)
ax.set_xticklabels(scenario_names, rotation=45, ha='right')
ax.set_ylabel("GWP [ktCO₂eq/an]")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.show()

def plot_gwp_operation_only(scenario_names, gwp_op_values):
    # Filtrer uniquement les scénarios contenant "tot"
    filtered_names = []
    filtered_values = []
    for name, value in zip(scenario_names, gwp_op_values):
        if "tot" in name.lower():
            filtered_names.append(name)
            filtered_values.append(value)

    x = np.arange(len(filtered_names))
    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter plot
    ax.scatter(x, filtered_values, color="#9ECAE1", s=80, label="GWP d'opération")

    # Ligne horizontale à 1000 ktCO₂eq/an
    ax.axhline(y=1000, color='grey', linestyle='--', linewidth=1, label='Seuil fixé')

    # Moyenne
    if filtered_values:
        moyenne = np.mean(filtered_values)
        ax.axhline(y=moyenne, color="#9ECAE1", linestyle='--', linewidth=1.2, label="Moyenne")

    # Mise en forme
    ax.set_xticks(x)
    ax.set_xticklabels(filtered_names, rotation=45, ha='right')
    ax.set_ylabel("GWP d'opération [ktCO₂eq/an]")

    # Style épuré
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Affichage
    #ax.legend()
    plt.tight_layout()
    plt.show()


plot_gwp_operation_only(scenario_names, gwp_op_values)
