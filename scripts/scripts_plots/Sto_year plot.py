# -*- coding: utf-8 -*-
"""
Created on Sun May 18 16:29:30 2025
@author: reynt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# === Paramètres ===
base_path = r"C:\Users\reynt\LMECA2675\EnergyScope-EnergyScope.py\case_studies\Scénarios de base"
run_names = ["10MGT", "GWP_tot_10MGT_corr"]
legende = ["Reference scenario", "Add GWP"]
couleurs = ['blue', 'orange']
threshold = 3.0  # TWh : seuil pour considérer deux valeurs comme "quasi identiques"

# === Collecte des données ===
data_scenarios = {}
non_utilisees = set()

for run_name in run_names:
    txt_path = os.path.join(base_path, run_name, 'output', 'sto_year.txt')
    csv_path = txt_path.replace('.txt', '.csv')

    print(f"\n🔍 Traitement de : {run_name}")
    if not os.path.exists(txt_path):
        print(f"[❌] Fichier introuvable : {txt_path}")
        continue

    if not os.path.exists(csv_path):
        try:
            df = pd.read_csv(txt_path, sep=None, engine='python')
            df.columns = df.columns.str.strip()
            df.to_csv(csv_path, index=False)
            print(f"✅ Fichier converti : {csv_path}")
        except Exception as e:
            print(f"[❌] Erreur conversion {txt_path}: {e}")
            continue

    try:
        df = pd.read_csv(txt_path, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')
        df = df[df.iloc[:, 0].notna()]
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
    except Exception as e:
        print(f"[❌] Erreur lecture {txt_path}: {e}")
        continue

    # Détection des technologies utilisées
    techno_dict = {}
    for _, row in df.iterrows():
        techno = str(row.iloc[0]).strip()
        valeurs = pd.to_numeric(row.iloc[1:], errors='coerce')
        valeurs_utiles = valeurs[valeurs != 8.76]
        if not valeurs_utiles.empty:
            techno_dict[techno] = valeurs_utiles.iloc[0]
        else:
            non_utilisees.add(techno)

    if techno_dict:
        data_scenarios[run_name] = techno_dict
    else:
        print(f"[⚠️] Aucune techno utilisée pour {run_name}")

# === Filtrage : seuil d'utilisation minimale
techno_globale = set().union(*[set(techs.keys()) for techs in data_scenarios.values()])
techno_utiles = set()

for techno in techno_globale:
    valeurs = [data_scenarios[run].get(techno, 0) for run in data_scenarios]
    if any(val >= 0.01 for val in valeurs):
        techno_utiles.add(techno)
    else:
        non_utilisees.add(techno)

# Nettoyage
for run in data_scenarios:
    data_scenarios[run] = {t: v for t, v in data_scenarios[run].items() if t in techno_utiles}

# === Tracé du graphe ===
plt.figure(figsize=(12, 6))
all_technos = sorted(set().union(*[d.keys() for d in data_scenarios.values()]))
cross_drawn = False  # Pour éviter les doublons dans la légende

for j, techno in enumerate(all_technos):
    valeurs = []
    missing = False
    for run in run_names:
        if run not in data_scenarios or techno not in data_scenarios[run]:
            missing = True
            break
        valeurs.append(data_scenarios[run][techno])
    if missing:
        continue

    # Si les deux scénarios ont une valeur proche, tracer une croix noire
    if abs(valeurs[0] - valeurs[1]) <= threshold:
        moyenne = sum(valeurs) / 2
        plt.scatter(j, moyenne, marker='x', color='black', s=100,
                    label="Equivalent values " if not cross_drawn else "")
        cross_drawn = True
    else:
        for i, val in enumerate(valeurs):
            plt.scatter(j, val, color=couleurs[i], label=legende[i] if j == 0 else "")

plt.xticks(range(len(all_technos)), all_technos, rotation=45, ha='right', fontsize=14)
plt.ylabel("Annual storage flows [GWh/year]", fontsize=16)
#plt.title("Comparison of storage technologies", fontsize=16)
plt.legend(fontsize = 14)
plt.grid(False)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("C:\\Users\\reynt\\OneDrive - UCL\\MASTER\\Mémoire\\Overleaf\\Section 4.2\\Sto_year_comp.svg", format="svg", bbox_inches="tight")
plt.show()

# === Affichage final ===
if non_utilisees:
    print("\n[🕳️] Technologies non utilisées dans aucun scénario :")
    for t in sorted(non_utilisees):
        print(" -", t)
else:
    print("\n✅ Toutes les technologies ont été utilisées dans au moins un scénario.")
