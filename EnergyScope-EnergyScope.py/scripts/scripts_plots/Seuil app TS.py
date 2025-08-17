# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 19:00:37 2025

@author: reynt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re
import numpy as np
from pathlib import Path

plt.rcParams.update({'font.size': 20})
couleur_bleu_froid = "#4B6C8B"
couleur_gris_foncé = "#666666"
couleur_bleu_pâle = "#A0B3C3"
colors = [couleur_bleu_froid, "darkgrey", couleur_bleu_pâle]
colors = [couleur_bleu_froid, couleur_bleu_froid, couleur_bleu_froid]

def extract_metadata_storage(folder_name):
    parts = folder_name.split('_')

    # Supprimer les éventuels préfixes numériques (ex: "0_")
    if parts[0].isdigit():
        parts = parts[1:]

    # Trouver l'index du champ "gwp"
    if not any(p.startswith("gwp") for p in parts):
        return None, None

    # Extraire la valeur GWP (dernière partie qui commence par "gwp")
    for i in reversed(range(len(parts))):
        if parts[i].startswith("gwp"):
            try:
                gwp_val = float(parts[i].replace("gwp", ""))
                gwp_index = i
                break
            except ValueError:
                return None, None
    else:
        return None, None  # Aucun champ gwp trouvé

    # Identifier le nom de la techno modifiée : tous les morceaux entre "Techno" et "gwp"
    if "Techno" in parts:
        techno_index = parts.index("Techno")
        techno = '_'.join(parts[techno_index + 1 : gwp_index])
    else:
        # fallback : prendre tout entre le début et "gwp"
        techno = '_'.join(parts[:gwp_index])

    return techno, gwp_val





def read_gwp_from_dat_file(dat_path, techno_name):
    """
    Extrait la valeur gwp_constr d'une techno depuis un fichier .dat (ex : ESTD_data.dat).
    """
    try:
        with open(dat_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Fichier introuvable : {dat_path}")
        return None

    start = False
    for line in lines:
        if line.strip().startswith("param :") and "gwp_constr" in line:
            start = True
            continue
        if start:
            if line.strip() == ';':
                break
            if line.strip().startswith(techno_name):
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 4:
                    try:
                        return float(parts[3])  # 3ᵉ valeur après techno = gwp_constr
                    except ValueError:
                        return None
    print(f"⚠️ Techno {techno_name} non trouvée dans {dat_path}")
    return None





def read_assets_txt(path):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split('\t')
        data = pd.read_csv(StringIO(''.join(lines[2:])), sep='\t', header=None)
        data.columns = header
        data.set_index('TECHNOLOGIES', inplace=True)
        return data
    except Exception:
        return None
def build_storage_dataframe_pivoted(base_dir, storage_technos, export_csv=True):
    """
    Construit un DataFrame pivoté avec : scénario, techno modifiée, GWP, f techno modifiée, f autres techno surveillées.
    Trie les lignes par techno_modifiée puis par GWP croissant.
    """
    records = []

    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue

        techno_mod, _ = extract_metadata_storage(folder)
        if techno_mod is None:
            continue

        # Redirection si nécessaire
        redirect_targets = {
            "BIOFUELS": "AMMONIA_STORAGE"
        }
        techno_real_target = redirect_targets.get(techno_mod, techno_mod)

        dat_path = os.path.join(full_path, "ESTD_data.dat")
        gwp_val = read_gwp_from_dat_file(dat_path, techno_real_target)
        if gwp_val is None:
            continue

        assets_path = os.path.join(full_path, "output", "assets.txt")
        df_assets = read_assets_txt(assets_path)
        if df_assets is None:
            continue

        row = {
            "folder": folder,
            "techno_modifiee": techno_mod,
            "gwp_constr": gwp_val,
        }

        # f de la techno modifiée
        row[f"f_{techno_mod}"] = df_assets.at[techno_real_target, ' f'] if techno_real_target in df_assets.index else 0.0

        # f des autres technologies surveillées
        for tech in storage_technos:
            row[f"f_{tech}"] = df_assets.at[tech, ' f'] if tech in df_assets.index else 0.0

        records.append(row)

    df = pd.DataFrame(records)

    # 🔁 TRIER PAR techno_modifiee PUIS gwp_constr
    df.sort_values(by=["techno_modifiee", "gwp_constr"], inplace=True)


    # 🧠 Trier les colonnes f_... par moyenne décroissante (hors techno modifiée)
    techno_cols = [col for col in df.columns if col.startswith("f_")]
    
    # Identifier la colonne correspondant à la techno modifiée (varie selon ligne)
    techno_mod_col_names = df["techno_modifiee"].apply(lambda x: f"f_{x}")
    unique_mod_cols = techno_mod_col_names.unique()
    
    # Liste des colonnes f_ sauf les techno_modifiées (on garde leur colonne en premier)
    other_cols = sorted(set(techno_cols) - set(unique_mod_cols))
    
    # Calcul de la moyenne pour trier
    mean_by_col = df[other_cols].mean().sort_values(ascending=False)
    sorted_other_cols = list(mean_by_col.index)
    
    # Colonnes fixes + colonnes techno_mod + autres colonnes triées
    fixed_cols = ["folder", "techno_modifiee", "gwp_constr"]
    final_cols = fixed_cols + sorted(set(unique_mod_cols)) + sorted_other_cols
    
    df = df[final_cols]


    if export_csv:
        base_name = os.path.basename(base_dir.rstrip("\\/"))
        output_csv = f"points_data_pivoted_{base_name}_conv.csv"
        df.to_csv(output_csv, index=False)
        print(f"✅ CSV pivoté sauvegardé : {output_csv}")

    return df



def build_storage_dataframe(base_dir, storage_technos, export_csv=True):
    """
    Construit un DataFrame avec : dossier, techno modifiée, techno surveillée,
    capacité installée, GWP de construction depuis le .dat.
    """
    data = []

    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue

        techno_target, _ = extract_metadata_storage(folder)
        if techno_target is None:
            continue

        # 🔁 Redirection pour les groupes génériques
        redirect_targets = {
            "BIOFUELS": "AMMONIA_STORAGE",  # techno représentative
        }
        techno_real_target = redirect_targets.get(techno_target, techno_target)

        dat_path = os.path.join(full_path, "ESTD_data.dat")
        gwp_val = read_gwp_from_dat_file(dat_path, techno_real_target)
        if gwp_val is None:
            continue

        assets_path = os.path.join(full_path, "output", "assets.txt")
        df_assets = read_assets_txt(assets_path)
        if df_assets is None:
            continue

        for tech in storage_technos:
            capacity = df_assets.at[tech, ' f'] if tech in df_assets.index else 0.0
            data.append({
                "folder": folder,
                "techno_modifiee": techno_target,
                "technology": tech,
                "f": capacity,
                "gwp_constr": gwp_val
            })

    df = pd.DataFrame(data)

    # Tri pour graphe : d’abord techno modifiée, ensuite techno surveillée
    df.sort_values(by=["techno_modifiee", "technology", "gwp_constr"], inplace=True)

    if export_csv:
        base_name = os.path.basename(base_dir.rstrip("\\/"))
        output_csv = f"points_data_{base_name}.csv"
        df.to_csv(output_csv, index=False)
        print(f"✅ Fichier CSV sauvegardé : {output_csv}")

    return df






import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_storage_impact(df, technos_modifiees, technos_a_surveiller, output_dir="plots_storage"):
    os.makedirs(output_dir, exist_ok=True)
    non_modifiees = {}  # 📦 Stocke les technos filtrées pour chaque techno modifiée

    for techno_mod in technos_modifiees:
        df_mod = df[df["techno_modifiee"] == techno_mod]
        df_mod = df_mod[df_mod["technology"].isin(technos_a_surveiller)]

        technos_valides = []
        technos_ignores = []

        for tech in df_mod["technology"].unique():
            tech_data = df_mod[df_mod["technology"] == tech]["f"]
            if tech_data.max() >= 1:
                relative_range = (tech_data.max() - tech_data.min()) / (tech_data.max() + 1e-6)
                if relative_range < 0.010:
                    technos_ignores.append(tech)
                else:
                    technos_valides.append(tech)

        df_mod = df_mod[df_mod["technology"].isin(technos_valides)]

        if df_mod.empty:
            print(f"⚠️ Aucun résultat significatif pour {techno_mod}")
            continue

        # 📝 Enregistrer les technologies non modifiées
        non_modifiees[techno_mod] = technos_ignores

        plt.figure(figsize=(15, 8))
        sns.lineplot(data=df_mod, x="gwp_constr", y="f", hue="technology", marker="o", ci=None)

        plt.xlabel(f"Construction GWP of {techno_mod} [kgCO2eq/kWh]")
        plt.ylabel("Installed capacities [Mt$\cdot$km]")
 # Optionnel : adapte selon tes besoins
        ax2 = plt.gca()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.legend(title="Technologies", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    # 📢 Résumé final
    print("\n📋 Technologies considérées comme non modifiées (variation < 10 %) :\n")
    for mod, techs in non_modifiees.items():
        if techs:
            print(f"→ {mod} : {', '.join(techs)}")
        else:
            print(f"→ {mod} : (aucune)")


def plot_presence_percentage(base_dirs, storage_technos, seuil_f=1.0, output_path="presence_plot.png"):
    """
    Analyse plusieurs répertoires base_dir et calcule le pourcentage de runs où f > seuil_f pour chaque techno.
    Affiche un barplot horizontal pour les techno > 0.3%, ajoute les valeurs (arrondies à l'unité) si < 100%, 
    et log les techno ≤ 0.3%.
    """
    presence_count = {tech: 0 for tech in storage_technos}
    total_runs = 0

    for base_dir in base_dirs:
        for folder in os.listdir(base_dir):
            full_path = os.path.join(base_dir, folder)
            if not os.path.isdir(full_path):
                continue

            assets_path = os.path.join(full_path, "output", "assets.txt")
            df_assets = read_assets_txt(assets_path)
            if df_assets is None:
                continue

            total_runs += 1
            for tech in storage_technos:
                try:
                    if tech in df_assets.index and float(df_assets.at[tech, ' f']) > seuil_f:
                        presence_count[tech] += 1
                except Exception:
                    continue

    if total_runs == 0:
        print("❌ Aucun run valide trouvé.")
        return pd.DataFrame()

    # Calcul des pourcentages
    percentages = {tech: (count / total_runs) * 100 for tech, count in presence_count.items()}

    # Séparation entre les techno à afficher et celles à ignorer (< 0.3%)
    below_threshold = {tech: pct for tech, pct in percentages.items() if pct <= 0.3}
    above_threshold = {tech: pct for tech, pct in percentages.items() if pct > 0.3}

    # Log console des techno ignorées
    if below_threshold:
        print("⚠️ Technologies avec une présence ≤ 0.3 % (non affichées) :")
        for tech, pct in sorted(below_threshold.items(), key=lambda x: x[1]):
            print(f"→ {tech}: {pct:.2f} %")

    if not above_threshold:
        print("❌ Aucune technologie avec une présence > 0.3 %")
        return pd.DataFrame()

    df_percent = pd.DataFrame.from_dict(above_threshold, orient='index', columns=['Presence (%)'])
    df_percent.sort_values(by='Presence (%)', ascending=True, inplace=True)

    # Plot
    plt.figure(figsize=(10, max(6, len(df_percent) * 0.35)))
    ax = sns.barplot(x='Presence (%)', y=df_percent.index, data=df_percent, color = couleur_bleu_pâle)
    plt.xlabel("Technology used in all simulations [%]")

    # Ajouter les annotations arrondies à l’unité, sauf si ça donne 100
    for i, (value, label) in enumerate(zip(df_percent['Presence (%)'], df_percent.index)):
        if value < 1.0 and value > 0.0:
            ax.text(value + 1, i, "< 1%", va='center', ha='left', fontsize=16)
        else:
            rounded = round(value)
            if rounded < 100:
                ax.text(value + 1, i, f" {rounded}%", va='center', ha='left', fontsize=16)

    plt.xlim(0, 105)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    print(f"✅ Graphique sauvegardé dans : {output_path}")
    return df_percent



def analyze_covariation_distribution_by_category_from_folder(folder_path, trigger_technos, threshold=0.01, output_dir="distribution_by_category"):

    

    os.makedirs(output_dir, exist_ok=True)
    folder = Path(folder_path)
    files = list(folder.glob("*.csv"))
    if not files:
        print("❌ Aucun fichier CSV trouvé.")
        return

    # === Catégories définies ===
    categories = {
        "Storage": [
            "BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS",
             "TS_DEC_HP_ELEC", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP",
            "GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE",
            "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE"
        ],
        "Mobility_Public": ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL", "BUS_COACH_HYDIESEL",
                            "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"],
        "Mobility_Private": ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL",
                             "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"],
        "Mobility_Freight": ["TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG",
                             "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL", "TRUCK_METHANOL", "TRUCK_FUEL_CELL",
                             "TRUCK_ELEC", "TRUCK_NG"],
        "Electricity": ["NUCLEAR", "CCGT", "CCGT_AMMONIA", "COAL_US", "COAL_IGCC",
                        "PV", "WIND_ONSHORE", "WIND_OFFSHORE", "HYDRO_RIVER", "GEOTHERMAL"],
        "Heat_High": ["IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_BOILER_GAS",
                      "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE", "IND_DIRECT_ELEC"],
        "Heat_Low_Central": ["DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE",
                             "DHN_COGEN_WET_BIOMASS", "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS",
                             "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO", "DHN_SOLAR"],
        "Heat_Low_Decentral": ["DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL",
                               "DEC_ADVCOGEN_GAS", "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS",
                               "DEC_BOILER_WOOD", "DEC_BOILER_OIL", "DEC_SOLAR", "DEC_DIRECT_ELEC"],
        "Conversion": ["HABER_BOSCH", "SYN_METHANOLATION", "METHANE_TO_METHANOL", "BIOMASS_TO_METHANOL",
                       "OIL_TO_HVC", "GAS_TO_HVC", "BIOMASS_TO_HVC", "METHANOL_TO_HVC",
                       "AMMONIA_TO_H2", "PYROLYSIS_TO_LFO", "PYROLYSIS_TO_FUELS"]
    }

    global_results = {
        trigger: {}  # techno_observée: {up, down, stable}
        for trigger in trigger_technos
    }
    total_events = {trigger: 0 for trigger in trigger_technos}

    for file in files:
        try:
            df = pd.read_csv(file, sep=None, engine="python")
        except Exception as e:
            print(f"⚠️ Erreur lecture {file.name} : {e}")
            continue

        for col in df.columns:
            if col.startswith("f_"):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values(by="gwp_constr").reset_index(drop=True)

        for trigger in trigger_technos:
            col_trigger = f"f_{trigger}"
            if col_trigger not in df.columns:
                continue

            for i in range(1, len(df)):
                prev, curr = df.loc[i - 1], df.loc[i]
                if pd.isna(prev[col_trigger]) or pd.isna(curr[col_trigger]):
                    continue

                delta_trigger = curr[col_trigger] - prev[col_trigger]
                if abs(delta_trigger) < threshold:
                    continue

                direction_trigger = np.sign(delta_trigger)
                total_events[trigger] += 1

                for col in df.columns:
                    if not col.startswith("f_") or col == col_trigger:
                        continue
                    other = col.replace("f_", "")
                    if pd.isna(prev[col]) or pd.isna(curr[col]):
                        continue

                    delta_other = curr[col] - prev[col]
                    if abs(delta_other) < threshold:
                        state = "stable"
                    elif np.sign(delta_other) == direction_trigger:
                        state = "up"
                    else:
                        state = "down"

                    if other not in global_results[trigger]:
                        global_results[trigger][other] = {"up": 0, "down": 0, "stable": 0}
                    global_results[trigger][other][state] += 1

    for trigger, techno_data in global_results.items():
        if total_events[trigger] == 0:
            print(f"ℹ️ Aucun événement pour {trigger}")
            continue

        for cat_name, techno_list in categories.items():
            techs, ups, downs, stables = [], [], [], []

            for tech in techno_list:
                if tech not in techno_data:
                    continue
                counts = techno_data[tech]
                total = counts["up"] + counts["down"] + counts["stable"]
                if total == 0:
                    continue
                techs.append(tech)
                ups.append(100 * counts["up"] / total)
                downs.append(100 * counts["down"] / total)
                stables.append(100 * counts["stable"] / total)

            if not techs:
                continue

            plot_df = pd.DataFrame({
                "techno": techs,
                "Increase": ups,
                "Decrease": downs,
                "Stable": stables
            })
            plot_df.set_index("techno", inplace=True)
            plot_df = plot_df.sort_index()

          # === Barres horizontales avec annotations personnalisées ===
            colors = {
    "Increase": "#A8E6A3",  # vert pastel vif
    "Decrease": "#F7A9A8",  # rouge pastel vif
    "Stable": "#D3D3D3"    # gris neutre clair
}
            
            plot_df = plot_df[["Increase", "Decrease", "Stable"]]  # assure l'ordre
            plot_df = plot_df.loc[~((plot_df.sum(axis=1) == 0) | (plot_df.max(axis=1) < 1))]  # remove techno < 1% ou vides
            
            if plot_df.empty:
                print(f"⚠️ Aucun changement significatif à tracer pour {trigger} dans {cat_name}")
                continue
            
            fig, ax = plt.subplots(figsize=(10, 0.4 * len(plot_df) + 1))
            plot_df.plot(kind="barh", stacked=True, color=[colors[c] for c in plot_df.columns], ax=ax)
            
            # Ajouter les annotations dans les barres
            for i, (techno, row) in enumerate(plot_df.iterrows()):
                left = 0
                for label, value in row.items():
                    if value >= 1:  # on n’écrit que si > 1 %
                        ax.text(left + value / 2, i, f"{int(round(value))}%", va='center', ha='center', fontsize=14)
                    left += value
            
            # Nettoyage du graphe
            ax.set_title(f"Impact of variation in {trigger} on {cat_name}", fontsize=16)
            ax.set_xlabel("")
            ax.set_xlim(0, 100)
            ax.set_ylabel("")
            ax.set_yticklabels(plot_df.index, fontsize=14)
            ax.xaxis.set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),  # 👈 place sous le graphe
            ncol=3,
            frameon=False,
            fontsize=14
        )

            plt.tight_layout()
            plt.show()
     

def plot_storage_impact_from_df(df, technos_modifiees, technos_a_surveiller, output_dir="plots_storage"):
    os.makedirs(output_dir, exist_ok=True)
    non_modifiees = {}

    for techno_mod in technos_modifiees:
        df_mod = df[df["techno_modifiee"] == techno_mod]
        df_mod = df_mod[df_mod["technology"].isin(technos_a_surveiller)]

        technos_valides = []
        technos_ignores = []

        for tech in df_mod["technology"].unique():
            tech_data = df_mod[df_mod["technology"] == tech]["f"]
            if tech_data.max() >= 1:
                relative_range = (tech_data.max() - tech_data.min()) / (tech_data.max() + 1e-6)
                if relative_range < 0.010:
                    technos_ignores.append(tech)
                else:
                    technos_valides.append(tech)

        df_mod = df_mod[df_mod["technology"].isin(technos_valides)]

        if df_mod.empty:
            print(f"⚠️ Aucun résultat significatif pour {techno_mod}")
            continue

        non_modifiees[techno_mod] = technos_ignores

        # 🌈 Plot
        plt.figure(figsize=(13, 12))
        sns.lineplot(data=df_mod, x="gwp_constr", y="f", hue="technology", marker="o", ci=None)

        plt.xlabel(f"Construction GWP of {techno_mod} [kgCO2eq/kWh]")
        plt.ylabel("Installed capacities [Mt$\cdot$km]")
        ax2 = plt.gca()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
       # Légende au-dessus du graphique
        
        # Réduit la zone des axes pour laisser de la place au-dessus (vers le haut = valeur < 1)
        plt.tight_layout(rect=[0, 0, 1, 0.5])
        ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize='small')
        plt.title(f"Impact of {techno_mod} GWP on storage technologies")
        plt.show()

    return non_modifiees



# === Application ===
base_dir = r"C:\Users\reynt\LMECA2675\EnergyScope-EnergyScope.py\case_studies\Seuil_app_storage"

technos_storage = [
    "BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS",
    "TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS",
    "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS",
    "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP",
    "GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE",
    "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE"
]
technos_modifiees_CAS= ["H2_STORAGE", "METHANOL_STORAGE"]
technos_modifiees_IS= ["PHS", "BATT_LI", "TS_DEC_HP_ELEC", "TS_HIGH_TEMP"]
technos_modifiees= [
    "PHEV_BATT","BEV_BATT",  "BATT_LI", "TS_DHN_DAILY",  "TS_HIGH_TEMP", "GAS_STORAGE",
    "H2_STORAGE","TS_DEC_HP_ELEC","TS", "BIOFUELS"]
technos_double = ["BATT_LI", "TS_DEC_HP_ELEC", "TS_HIGH_TEMP"]
techno_sea = ["TS_DHN_SEASONAL"]
technos_mobility_public = ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL",
                           "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"]

technos_mobility_private = ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG",
                            "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"]

technos_mobility_freight = ["TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG",
                            "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL", "TRUCK_METHANOL", "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"]

technos_electricity = ["NUCLEAR", "CCGT", "CCGT_AMMONIA", "COAL_US", "COAL_IGCC",
                       "PV", "WIND_ONSHORE", "WIND_OFFSHORE", "HYDRO_RIVER", "GEOTHERMAL"]


technos_heat_high = [
     "IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_BOILER_GAS",
     "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE", "IND_DIRECT_ELEC"]


technos_heat_low_central = [
    "DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE",
     "DHN_COGEN_WET_BIOMASS", "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS",
     "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO", "DHN_SOLAR"]

technos_heat_low_decentral = [
     "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL", "DEC_ADVCOGEN_GAS",
     "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL", "DEC_SOLAR", "DEC_DIRECT_ELEC"]
technos_conversion = [
     "HABER_BOSCH", "SYN_METHANOLATION", "METHANE_TO_METHANOL", "BIOMASS_TO_METHANOL",
     "OIL_TO_HVC", "GAS_TO_HVC", "BIOMASS_TO_HVC", "METHANOL_TO_HVC",
     "AMMONIA_TO_H2", "PYROLYSIS_TO_LFO", "PYROLYSIS_TO_FUELS"]
technos_a_surveiller = technos_mobility_freight # ou une sous-liste personnalisée

csv_paths = [
    "seuil_appearance_BATT_LI.csv",
    "inverse_substitution_BATT_LI.csv"
]

# 📁 Dossiers à fusionner (tu peux en ajouter plus)
base_dirs = [
    "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/IS_techno",
    "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/Seuil_app_storage"
]

# 🔍 Technologies concernées
technos_modifiees = ["BATT_LI"]
technos_a_surveiller = technos_mobility_freight

# 📦 Récupération et fusion
dfs = [build_storage_dataframe(d, technos_a_surveiller, export_csv=False) for d in base_dirs]
df_combined = pd.concat(dfs, ignore_index=True)

# 📊 Plot combiné
plot_storage_impact_from_df(df_combined, technos_modifiees, technos_a_surveiller)


# df_storage = build_storage_dataframe(base_dir, technos_a_surveiller)
#  #df_storage_pivot = build_storage_dataframe_pivoted(base_dir, technos_a_surveiller)
# plot_storage_impact(df_storage, technos_modifiees, technos_a_surveiller)
# base_dirs = [
#     r"C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/IS_techno",
#     r"C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/Seuil_app_storage"
#     # r"C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/techno_ts_dhn_seasonal",
#     # r"C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/CAS_technos"
#     # ajoute d'autres chemins ici
# ]
df_presence = plot_presence_percentage(base_dirs, technos_storage)



# csv_file = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/scripts/CSV PB/points_data_pivoted_IS_techno_mob_priv.csv"
# analyze_covariation_distribution_by_category_from_folder(
#     folder_path="C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/scripts/CSV PB",
#     trigger_technos=["H2_STORAGE", "METHANOL_STORAGE", "PHS", "BATT_LI", "TS_DEC_HP_ELEC", 
#                                 "TS_HIGH_TEMP","PHEV_BATT","BEV_BATT", "TS_DHN_DAILY", "GAS_STORAGE",
#                                 "TS_DEC_HP_ELEC","TS_DHN_SEASONAL"],
#     threshold=0.01
# )
