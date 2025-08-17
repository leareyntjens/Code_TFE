# -*- coding: utf-8 -*-
"""
Analyse des pourcentages d'importation et de stockage à partir des fichiers input2sankey.csv
@author: reynt
"""

import csv
import os
import matplotlib.pyplot as plt

# === Paramètres à adapter ===
base_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies"

# Regroupements personnalisés : (label affiché, nom du run constr, nom du run tot)
scenarios = [
    ("Minimisation de cout",'Scénarios de base/10MGT', 'gwp_tot_TS' )
        #("Sans limite", "Scenario_ref_60500","Lim_prix_68500"),
        #("150%", "Scénarios de base/Ressources_150%_sans_fossil","Lim_prix_68500_150%")
    ]


# === Fonction 1 : Pourcentage d'importation
def get_import_percentage(run_name):
    if run_name is None:
        return None

    csv_path = os.path.join(base_path, run_name, "output", "sankey", "input2sankey.csv")
    import_sum = 0.0
    filtered_sum = 0.0

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = list(csv.reader(csvfile))
            data_rows = reader[1:]  # Ignore header
            all_targets = set(row[1].strip() for row in data_rows if len(row) > 1)

            for row in data_rows:
                try:
                    source = row[0].strip()
                    value = float(row[2])

                    if source.startswith("Imp."):
                        import_sum += value

                    if source not in all_targets:
                        filtered_sum += value

                except (IndexError, ValueError):
                    print(f"[{run_name}] ⚠️ Erreur de lecture sur une ligne.")

        if filtered_sum > 0:
            return (import_sum / filtered_sum) * 100
        else:
            print(f"[{run_name}] ❌ Dénominateur nul ou invalide.")
            return None

    except FileNotFoundError:
        print(f"[{run_name}] ❌ Fichier non trouvé : {csv_path}")
        return None

# === Fonction 2 : Pourcentage de stockage
def get_storage_percentage(run_name):
    if run_name is None:
        return None

    csv_path = os.path.join(base_path, run_name, "output", "sankey", "input2sankey.csv")
    storage_sum = 0.0
    filtered_sum = 0.0

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = list(csv.reader(csvfile))
            data_rows = reader[1:]  # Ignore header
            all_targets = set(row[1].strip() for row in data_rows if len(row) > 1)

            for row in data_rows:
                try:
                    source = row[0].strip()
                    value = float(row[2])

                    if "sto" in source.lower():
                        storage_sum += value

                    if source not in all_targets:
                        filtered_sum += value

                except (IndexError, ValueError):
                    print(f"[{run_name}] ⚠️ Erreur de lecture sur une ligne.")

        if filtered_sum > 0:
            return storage_sum 
        else:
            print(f"[{run_name}] ❌ Dénominateur nul ou invalide.")
            return None

    except FileNotFoundError:
        print(f"[{run_name}] ❌ Fichier non trouvé : {csv_path}")
        return None

# === Traitement général ===
x_labels = []
y_imp_constr = []
y_imp_tot = []
y_sto_constr = []
y_sto_tot = []

print("\n📊 Pourcentages d'importation et de stockage :\n")

for label, constr_run, tot_run in scenarios:
    x_labels.append(label)

    constr_imp = get_import_percentage(constr_run)
    tot_imp = get_import_percentage(tot_run)

    constr_sto = get_storage_percentage(constr_run)
    tot_sto = get_storage_percentage(tot_run)

    y_imp_constr.append(constr_imp)
    y_imp_tot.append(tot_imp)
    y_sto_constr.append(constr_sto)
    y_sto_tot.append(tot_sto)

    print(f"📁 {label}")
    if constr_imp is not None:
        print(f"  ➤ Importation constr : {constr_imp:.2f} %")
    if tot_imp is not None:
        print(f"  ➤ Importation tot    : {tot_imp:.2f} %")
    if constr_sto is not None:
        print(f"  ➤ Stockage constr    : {constr_sto:.2f} %")
    if tot_sto is not None:
        print(f"  ➤ Stockage tot       : {tot_sto:.2f} %")

# === Trier pour affichage
combined_imp = list(zip(x_labels, y_imp_constr, y_imp_tot))
combined_imp.sort(key=lambda t: min(v for v in t[1:] if v is not None))
x_labels_imp, y_imp_constr, y_imp_tot = zip(*combined_imp)

combined_sto = list(zip(x_labels, y_sto_constr, y_sto_tot))
combined_sto.sort(key=lambda t: min(v for v in t[1:] if v is not None))
x_labels_sto, y_sto_constr, y_sto_tot = zip(*combined_sto)

# === Graphique 1 : Importations
plt.figure(figsize=(10, 5))
plt.plot([], [], 'o', color="tab:blue", label="Import constr")
plt.plot([], [], 'o', color="tab:red", label="Import tot")

for i, (x, y1, y2) in enumerate(zip(x_labels_imp, y_imp_constr, y_imp_tot)):
    if y1 is not None:
        plt.plot(i, y1, 'o', color="tab:blue")
    if y2 is not None:
        plt.plot(i, y2, 'o', color="tab:red")

plt.xticks(range(len(x_labels_imp)), x_labels_imp, rotation=30)
plt.ylabel("Importations [%]")
plt.title("Part des importations par scénario")
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# === Graphique 2 : Stockage
plt.figure(figsize=(10, 5))
plt.plot([], [], 'o', color="tab:blue", label="Stockage constr")
plt.plot([], [], 'o', color="tab:red", label="Stockage tot")

for i, (x, y1, y2) in enumerate(zip(x_labels_sto, y_sto_constr, y_sto_tot)):
    if y1 is not None:
        plt.plot(i, y1, 'o', color="tab:blue")
    if y2 is not None:
        plt.plot(i, y2, 'o', color="tab:red")

plt.xticks(range(len(x_labels_sto)), x_labels_sto, rotation=30)
plt.ylabel("Stockage [%]")
plt.title("Part des flux provenant du stockage")
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
