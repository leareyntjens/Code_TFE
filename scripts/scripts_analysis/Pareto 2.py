# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 14:39:28 2025

@author: reynt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


# === Fonctions fournies ===

def cost(scenario_path, txt_file_name="cost_breakdown.txt"):
    txt_path = os.path.join(scenario_path, "output", txt_file_name)
    csv_path = txt_path.replace('.txt', '.csv')

    if not os.path.exists(txt_path):
        print(f"[❌] Fichier non trouvé : {txt_path}")
        return None

    try:
        df = pd.read_csv(txt_path, sep=None, engine='python', skiprows=[1])
        df.columns = df.columns.str.strip()
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"[❌] Erreur lors de la conversion : {e}")
        return None

    if df.shape[1] < 4:
        print(f"[⚠️] Format inattendu : {df.shape[1]} colonnes.")
        return None

    col2_sum = df.iloc[:, 1].sum()
    col3_sum = df.iloc[:, 2].sum()
    col4_sum = df.iloc[:, 3].sum()
    total_sum = col2_sum + col3_sum + col4_sum

    return total_sum

def gwp(scenario_path, txt_file_name="gwp_breakdown.txt"):
    txt_path = os.path.join(scenario_path, "output", txt_file_name)
    csv_path = txt_path.replace('.txt', '.csv')

    if not os.path.exists(txt_path):
        print(f"[❌] Fichier non trouvé : {txt_path}")
        return None

    try:
        df = pd.read_csv(txt_path, sep=None, engine='python', skiprows=[1])
        df.columns = df.columns.str.strip()
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"[❌] Erreur lors de la conversion : {e}")
        return None

    if df.shape[1] < 3:
        print(f"[⚠️] Format inattendu : {df.shape[1]} colonnes.")
        return None

    col2_sum = df.iloc[:, 1].sum()
    col3_sum = df.iloc[:, 2].sum()
    total_sum = col2_sum + col3_sum

    return total_sum

# === Paramètres généraux ===
base_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies"
scenarios = list(range(55000, 100001, 5000))
suffix = "GWP_min_"

# === Collecte des résultats
results = []

for val in scenarios:
    run_name = f"{suffix}{val}"
    run_path = os.path.join(base_path, run_name)

    ctot = cost(run_path)
    gwptot = gwp(run_path)

    if ctot is not None and gwptot is not None:
        results.append({"Cost_limit": val, "C_tot": ctot, "GWP_tot": gwptot})
        print(f"[✅] {run_name} ➜ C_tot={ctot:.1f} M€, GWP_tot={gwptot:.1f} ktCO2eq")
    else:
        print(f"[⚠️] Résultats incomplets pour {run_name}")

# === Analyse
df = pd.DataFrame(results).sort_values("C_tot")
df["delta_C"] = df["C_tot"].diff()
df["delta_GWP"] = -df["GWP_tot"].diff()
df["Coût_marginal"] = df["delta_C"] / df["delta_GWP"]
print("\n=== Coûts marginaux de réduction des émissions ===")
for i in range(1, len(df)):
    print(f"De {df.loc[i-1, 'Cost_limit']} M€ à {df.loc[i, 'Cost_limit']} M€ : "
          f"{df.loc[i, 'Coût_marginal']:.2f} M€/ktCO₂eq")

# === Export CSV
df.to_csv("résumé_GWP_vs_Cout.csv", index=False)

# === Graphe Pareto
plt.figure(figsize=(8,6))
plt.plot(df["C_tot"], df["GWP_tot"], marker='o', label="Optimized scenarios")
plt.xlabel("Total annual cost of the energy system (M€/y.)")
plt.ylabel("Total yearly GHG emissions of the energy system (ktCO₂eq/y.)")
plt.title("Pareto Front in 2050")

# Ajouter les points rouges et verts
point_red = df.iloc[(df["C_tot"] - 96000).abs().argmin()]
point_green = df.iloc[(df["C_tot"] - 80000).abs().argmin()]
plt.plot(point_red["C_tot"], point_red["GWP_tot"], 'ro', label="No cost constraint (96k M€)")
plt.plot(point_green["C_tot"], point_green["GWP_tot"], 'go', label="Selected limit (80k M€)")

# === Graphe coût marginal (échelle linéaire) - Désactivé
# fig_lin, ax_lin = plt.subplots(figsize=(7, 6))
# ax_lin.plot(df["C_tot"][1:], df["Coût_marginal"][1:], marker='s', linestyle='--', color='orange', label="Marginal cost")
# ax_lin.set_xlabel("Total annual cost of the energy system (M€/y.)")
# ax_lin.set_ylabel("Marginal cost (M€/ktCO₂eq)")
# ax_lin.set_title("Marginal cost - Linear scale")
# ax_lin.spines['top'].set_visible(False)
# ax_lin.spines['right'].set_visible(False)
# ax_lin.set_xlim(60000, 90000)
# ax_lin.set_ylim(0, 1000)
# ax_lin.plot(point_red["C_tot"], point_red["Coût_marginal"], 'ro', label="No cost constraint")
# ax_lin.plot(point_green["C_tot"], point_green["Coût_marginal"], 'go', label="Selected limit")
# ax_lin.legend()
# plt.tight_layout()
# plt.savefig("courbe_cout_marginal_lineaire.png")
# plt.show()

# === Nouvelle figure combinée : Pareto + coût marginal log
fig, (ax_pareto, ax_log) = plt.subplots(1, 2, figsize=(14, 7))

# --- Graphe 1 : Pareto
ax_pareto.plot(df["C_tot"], df["GWP_tot"], marker='o', color="#4B6C8B",label="Optimized scenarios")
ax_pareto.plot(point_red["C_tot"], point_red["GWP_tot"], 'ro', label="No cost constraint (96k M€)")
ax_pareto.plot(point_green["C_tot"], point_green["GWP_tot"], 'go', label="Selected limit (80k M€)")
ax_pareto.set_xlabel("Total annual cost (M€/y.)")
ax_pareto.set_ylabel("Total GWP (ktCO₂eq/y.)")
ax_pareto.set_title("Pareto Front")
ax_pareto.spines['top'].set_visible(False)
ax_pareto.spines['right'].set_visible(False)
#ax_pareto.legend()

# --- Graphe 2 : Coût marginal (log)
ax_log.plot(df["C_tot"][1:], df["Coût_marginal"][1:], marker='s', linestyle='--', color="#A0B3C3", label="Marginal cost")
ax_log.plot(point_red["C_tot"], point_red["Coût_marginal"], 'ro')
ax_log.plot(point_green["C_tot"], point_green["Coût_marginal"], 'go')
ax_log.set_xlabel("Total annual cost (M€/y.)")
ax_log.set_ylabel("Marginal cost (M€/ktCO₂eq)")
ax_log.set_title("Marginal cost - Log scale")
ax_log.set_yscale("log")
ax_log.set_xlim(60000, 90000)
ax_log.set_ylim(1, 1000)
ax_log.spines['top'].set_visible(False)
ax_log.spines['right'].set_visible(False)
#ax_log.legend()

# Titre principal
#fig.suptitle("Trade-off between cost and emissions reduction in 2050", fontsize=16)

# Laisse plus de place en bas (augmente le `bottom` à 0.02 ou moins)
plt.tight_layout(rect=[0, 0.04, 1, 0.93])

# Légende unique, sans doublon, sur une ligne
handles_labels = {}
for ax in [ax_pareto, ax_log]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in handles_labels:
            handles_labels[label] = handle

# ncol = nombre exact d'éléments pour forcer une seule ligne
fig.legend(handles_labels.values(), handles_labels.keys(), loc='lower center', ncol=len(handles_labels), frameon=False)

plt.savefig("fig_pareto_marginal_log.png")
plt.show()


# === Affichage des points clés
print("\n=== Points d’intérêt ===")
print(f"Point rouge ➜ Sans contrainte : {point_red['C_tot']:.0f} M€, "
      f"{point_red['GWP_tot']:.1f} ktCO₂eq, "
      f"{point_red['Coût_marginal']:.1f} M€/kt")
print(f"Point vert  ➜ Limite choisie  : {point_green['C_tot']:.0f} M€, "
      f"{point_green['GWP_tot']:.1f} ktCO₂eq, "
      f"{point_green['Coût_marginal']:.1f} M€/kt")


from scipy.signal import savgol_filter
import numpy as np

# Étape 1 : Nettoyer les valeurs nulles ou négatives
df_clean = df.dropna(subset=["Coût_marginal"]).copy()
df_clean = df_clean[df_clean["Coût_marginal"] > 0].reset_index(drop=True)

# Étape 2 : Calcul du log
df_clean["log_cout_marginal"] = np.log10(df_clean["Coût_marginal"])

# Étape 3 : Vérifier qu'on a assez de points pour appliquer un filtre (min 5)
if len(df_clean) >= 5:
    y = df_clean["log_cout_marginal"].values
    x = df_clean["C_tot"].values
    d2 = savgol_filter(y, window_length=5, polyorder=2, deriv=2)  # Dérivée seconde

    df_clean["courbure"] = d2
    idx_max_courbure = np.argmax(np.abs(d2))
    point = df_clean.iloc[idx_max_courbure]

    # Affichage
    print("\n=== Point d'inflexion détecté ===")
    print(f"Coût total        : {point['C_tot']:.0f} M€")
    print(f"GWP total         : {point['GWP_tot']:.1f} ktCO₂eq/an")
    print(f"Coût marginal     : {point['Coût_marginal']:.1f} M€/ktCO₂eq")
    print(f"Courbure approx.  : {point['courbure']:.4f}")
else:
    print("[⚠️] Pas assez de points valides pour appliquer le filtre de Savitzky-Golay.")
