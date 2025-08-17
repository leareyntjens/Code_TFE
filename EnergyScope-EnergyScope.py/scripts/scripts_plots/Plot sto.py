# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 16:40:12 2025

@author: reynt
"""

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

# === Données d'ordre des scénarios ===
couleur_bleu_froid = "#4B6C8B"
couleur_gris_foncé = "#666666"
couleur_bleu_pâle = "#A0B3C3"
colors = [couleur_bleu_froid, "darkgrey", couleur_bleu_pâle]
import re
plt.rcParams.update({'font.size': 20})
def extraire_scenarios_depuis_log(log_path):
    """
    Extrait les noms de scénarios qui commencent par 'Stockage_' depuis un fichier log.
    """
    scenarios = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r"(Stockage_[\w\d_]+)", line.strip())
            if match:
                scenarios.append(match.group(1))
    return scenarios

# Exemple d’utilisation :
log_file_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/scripts/log.csv"
ordered_scenarios = extraire_scenarios_depuis_log(log_file_path)
print(f"{len(ordered_scenarios)} scénarios trouvés.")

base_directory = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/Stockage dom"

def extraire_scenarios_depuis_log(log_path):
    """
    Extrait les noms de scénarios qui commencent par 'Stockage_' depuis un fichier log.
    """
    scenarios = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r"(Stockage_[\w\d_]+)", line.strip())
            if match:
                scenarios.append(match.group(1))
    return scenarios

def convert_txt_en_csv(scenario_path, txt_file_name):
    txt_path = os.path.join(scenario_path, "output", txt_file_name)
    csv_path = txt_path.replace('.txt', '.csv')
    if not os.path.exists(txt_path):
        print(f" Fichier non trouvé : {txt_path}")
        return None
    try:
        df = pd.read_csv(txt_path, sep=None, engine='python', skiprows=[1])
        df.columns = df.columns.str.strip()
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        print(f" Erreur de conversion : {e}")
        return None


def analyze_scenario_storage(scenario_path):
    results = {"scenario": os.path.basename(scenario_path), "assets_storage_gwh": None, "sankey_storage_gwh": None}

    # 1. Convertir et lire assets.txt
    assets_csv_path = convert_txt_en_csv(scenario_path, "assets.txt")
    if assets_csv_path and os.path.exists(assets_csv_path):
        try:
            df_assets = pd.read_csv(assets_csv_path)
            df_assets["TECHNOLOGIES"] = df_assets["TECHNOLOGIES"].astype(str).str.strip()

            technos_storage_elec = ["BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS"]
            technos_storage_thermal = ["TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS",
                                       "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS",
                                       "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP"]
            technos_storage_other = ["GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE", "LFO_STORAGE",
                                     "AMMONIA_STORAGE", "METHANOL_STORAGE"]
            technos_stockage_cibles = technos_storage_elec + technos_storage_thermal + technos_storage_other

            df_storage = df_assets[df_assets["TECHNOLOGIES"].isin(technos_stockage_cibles)]
            results["assets_storage_gwh"] = df_storage["f"].sum()
        except Exception as e:
            print(f"[{results['scenario']}] Erreur lecture CSV assets : {e}")

    # 2. input2sankey.csv
    sankey_path = os.path.join(scenario_path, "output", "sankey", "input2sankey.csv")
    if os.path.exists(sankey_path):
        try:
            with open(sankey_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)
                storage_sum_twh = 0.0
                for row in reader:
                    if len(row) >= 3 and "sto" in row[1].strip().lower():
                        try:
                            value = float(row[2])
                            storage_sum_twh += value
                        except ValueError:
                            continue
                results["sankey_storage_gwh"] = storage_sum_twh # TWh → GWh
        except Exception as e:
            print(f"[{results['scenario']}] Erreur lecture sankey : {e}")

    return results


def analyze_all_scenarios(base_dir, scenario_list):
    """
    Parcourt les dossiers de scenario_list et analyse chaque scénario.
    """
    all_results = []
    for scenario_name in scenario_list:
        scenario_path = os.path.join(base_dir, scenario_name)
        if os.path.isdir(scenario_path):
            result = analyze_scenario_storage(scenario_path)
            all_results.append(result)
    return pd.DataFrame(all_results)

def detect_stabilisation(values, tolerance=100):
    """
    Retourne l’index où la variation relative devient inférieure à `tolerance`
    entre deux points consécutifs (et reste stable ensuite).
    """
    for i in range(1, len(values)):
        if abs(values[i] - values[i-1]) / max(values[i-1], 1e-6) < tolerance:
            # On vérifie qu’après ce point, ça reste dans la même tendance
            stable = all(abs(values[j] - values[j-1]) / max(values[j-1], 1e-6) < tolerance for j in range(i+1, len(values)))
            if stable:
                return i
    return None

def get_scenario_index(df, scenario_name):
    """
    Retourne l'index (ordre) d’un scénario donné dans le DataFrame.
    """
    try:
        return df[df["scenario"] == scenario_name].index[0]
    except IndexError:
        print(f"⚠️ Scénario {scenario_name} introuvable.")
        return None



# === Lancement analyse ===
ordered_scenarios = extraire_scenarios_depuis_log(log_file_path)
df = analyze_all_scenarios(base_directory, ordered_scenarios)

# === Tri selon ordre de log ===
df["order"] = df["scenario"].apply(lambda x: ordered_scenarios.index(x) if x in ordered_scenarios else -1)
df = df[df["order"] >= 0].sort_values("order")

# === Détection des points de stabilisation ===
stable_idx_assets = detect_stabilisation(df["assets_storage_gwh"].values, tolerance=0.1)
stable_idx_sankey = detect_stabilisation(df["sankey_storage_gwh"].fillna(0).values, tolerance=0.1)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

scenario_marque = "Stockage_HIGH_TEMP_constr_171"

# === Tracé 1 : Capacité installée ===
ax1 = axes[0]
ax1.scatter(df["order"], df["assets_storage_gwh"], marker='x', label="Scenarios", color=couleur_bleu_pâle)

if scenario_marque in df["scenario"].values:
    idx_marque = df[df["scenario"] == scenario_marque].index[0]
    valeur_y = df.loc[idx_marque, "assets_storage_gwh"]
    ax1.axvline(x=idx_marque, color=couleur_bleu_froid, linestyle='-.', linewidth=1.5, label='Threshold')
    ax1.axhline(y=valeur_y, color=couleur_bleu_froid, linestyle=':', linewidth=1.2)
    ax1.text(len(df)-1, valeur_y, f"{valeur_y:.1f} GWh", va='bottom', ha='right', fontsize=20, color=couleur_bleu_froid)

ax1.set_ylabel("Installed capacity (GWh)")
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
legend1 = ax1.legend()
legend1.get_frame().set_linewidth(0)
legend1.get_frame().set_facecolor('none')


# === Tracé 2 : Flux de stockage ===
ax2 = axes[1]
ax2.scatter(df["order"], df["sankey_storage_gwh"], marker='x', color=couleur_bleu_pâle)

if scenario_marque in df["scenario"].values:
    valeur_y = df.loc[idx_marque, "sankey_storage_gwh"]
    ax2.axvline(x=idx_marque, color=couleur_bleu_froid, linestyle='-.', linewidth=1.5)
    ax2.axhline(y=valeur_y, color=couleur_bleu_froid, linestyle=':', linewidth=1.2)
    ax2.text(len(df)-1, valeur_y, f"{valeur_y:.1f} TWh", va='bottom', ha='right', fontsize=20, color=couleur_bleu_froid)

ax2.set_ylabel("Annual storage flow (TWh)")  # déjà converti depuis TWh
ax2.set_xlabel("Scenarios ordered by sequential increase in storage GWP")
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
legend2 = ax2.legend()
legend2.get_frame().set_linewidth(0)
legend2.get_frame().set_facecolor('none')


# Final layout
plt.tight_layout()
plt.show()
