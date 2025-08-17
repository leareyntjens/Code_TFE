# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 18:30:36 2025

@author: reynt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO

couleur_bleu_froid = "#4B6C8B"
couleur_gris_foncé = "#666666"
couleur_bleu_pâle = "#A0B3C3"
colors = [couleur_bleu_froid, "darkgrey", couleur_bleu_pâle]
plt.rcParams.update({'font.size': 14})

def convert_txt_to_csv(txt_path):
    """
    Convertit un fichier .txt (assets.txt) en .csv avec séparation auto.
    Retourne le chemin du fichier .csv ou None si échec.
    """
    csv_path = txt_path.replace(".txt", ".csv")
    try:
        df = pd.read_csv(txt_path, sep=None, engine="python", skiprows=[1])
        df.columns = df.columns.str.strip()
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        print(f"[ERREUR] Conversion {txt_path} : {e}")
        return None


def extract_gwp_from_folder(folder_name):
    """
    Extrait la ressource et la valeur de GWP du nom du dossier.
    Exemple : 'METHANOL_RE_0.02' → ('METHANOL_RE', 0.02)
    """
    parts = folder_name.split('_')
    try:
        gwp_value = float(parts[-1])
        resource = '_'.join(parts[:-1])
        return resource, gwp_value
    except ValueError:
        return None, None

def build_techno_timeseries(base_dir, techno_list, assets_filename='assets.csv'):
    """
    Parcourt tous les dossiers du répertoire, extrait le GWP et lit les capacités installées
    pour chaque technologie d'intérêt.
    """
    data = []
    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue

        resource, gwp_value = extract_gwp_from_folder(folder)
        if resource is None or gwp_value is None:
            continue

        try : 
            txt_path = os.path.join(full_path, "output/assets.txt")
            csv_path = convert_txt_to_csv(txt_path)
            if not csv_path:
                continue
            df = pd.read_csv(csv_path, index_col=0)

        except Exception as e:
            print(f"[ERREUR] Lecture échouée pour {folder} : {e}")
            continue

        for tech in techno_list:
            if tech in df.index:
                capacity = df.at[tech, 'f']
            else:
                capacity = 0.0
            data.append({
                "Scenario": folder,
                "Resource": resource,
                "GWP_value": gwp_value,
                "Technology": tech,
                "Capacity": capacity
            })

    return pd.DataFrame(data)

def plot_techno_evolution(df, group_by_resource=True):
    """
    Trace des courbes d’évolution des capacités installées selon le GWP.
    Une courbe par technologie. Option de regroupement par ressource dominante.
    """
    import seaborn as sns

    if group_by_resource:
        grouped = df.groupby("Resource")
        for resource, group in grouped:
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=group, x="GWP_value", y="Capacity", hue="Technology", marker="o")
            plt.title(f"Évolution des capacités installées pour la ressource {resource}")
            plt.xlabel("GWP d'opération de la ressource dominante")
            plt.ylabel("Capacité installée (unité)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    else:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x="GWP_value", y="Capacity", hue="Technology", style="Resource", marker="o")
        plt.title("Évolution des capacités installées par technologie")
        plt.xlabel("GWP d'opération")
        plt.ylabel("Capacité installée (unité)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
def detect_bascule_disparition(df, disparition_threshold=1.0):
    """
    Pour chaque technologie et chaque ressource dominante :
    - identifie le GWP à partir duquel la technologie n’est plus dominante (bascule),
    - identifie le GWP à partir duquel elle devient négligeable (disparition).
    """
    results = []

    for resource in df["Resource"].unique():
        df_r = df[df["Resource"] == resource]

        for tech in df_r["Technology"].unique():
            df_tech = df_r[df_r["Technology"] == tech].sort_values(by="GWP_value")

            # GWP où la techno tombe sous le seuil de disparition
            disappeared_at = df_tech[df_tech["Capacity"] < disparition_threshold]
            gwp_disparition = disappeared_at["GWP_value"].min() if not disappeared_at.empty else None

            # GWP où la techno cesse d’être dominante (capacité max du groupe)
            gwp_bascule = None
            df_grouped = df_r.groupby("GWP_value")
            for gwp_val, group in df_grouped:
                max_capacity = group["Capacity"].max()
                tech_capacity = group[group["Technology"] == tech]["Capacity"].values[0]
                if tech_capacity < max_capacity:
                    gwp_bascule = gwp_val
                    break

            results.append({
                "Resource": resource,
                "Technology": tech,
                "GWP_bascule": gwp_bascule,
                "GWP_disparition": gwp_disparition
            })

    return pd.DataFrame(results)



def read_gwp_op_from_dat_file(dat_path, resource_names):
    try:
        with open(dat_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {}

    gwp_ops = {}
    start = False
    for line in lines:
        if line.strip().startswith("param :") and "gwp_op" in line:
            start = True
            continue
        if start:
            if line.strip() == ';':
                break
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 4:
                resource = parts[0]
                if resource in resource_names:
                    try:
                        gwp_ops[resource] = float(parts[3])
                    except ValueError:
                        gwp_ops[resource] = None
    for r in resource_names:
        if r not in gwp_ops:
            gwp_ops[r] = None
    return gwp_ops

def read_gwp_from_dat_file(dat_path, techno_name):
    try:
        with open(dat_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
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
                        return float(parts[3])
                    except ValueError:
                        return None
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

def build_extended_dataframe(base_dir, storage_technos, resource_names, export_csv=True):
    records = []
    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue

        dat_path = os.path.join(full_path, "ESTD_data.dat")
        if not os.path.exists(dat_path):
            continue

        # Extract gwp_op for all resources
        gwp_ops = read_gwp_op_from_dat_file(dat_path, resource_names)

        # Guess modified resource
        modified_resource = max(gwp_ops.items(), key=lambda x: x[1] if x[1] is not None else -1)[0]
        modified_value = gwp_ops[modified_resource]

        # Read assets
        assets_path = os.path.join(full_path, "output", "assets.txt")
        df_assets = read_assets_txt(assets_path)
        if df_assets is None:
            continue

        row = {
            "folder": folder,
            "resource_modified": modified_resource,
            "gwp_op_modified": modified_value
        }

        for res in resource_names:
            row[f"gwp_op_{res}"] = gwp_ops.get(res)

        for tech in storage_technos:
            row[f"f_{tech}"] = df_assets.at[tech, ' f'] if tech in df_assets.index else 0.0

        records.append(row)

    df = pd.DataFrame(records)
    if export_csv:
        output_csv = f"gwp_op_f_technos_summary.csv"
        df.to_csv(output_csv, index=False)
    return df






def read_gwp_op_from_dat_file(dat_path, resource_names):
    """
    Extrait les valeurs gwp_op (3ᵉ colonne) de plusieurs ressources depuis un fichier .dat (ex : ESTD_data.dat).
    """
    gwp_ops = {}
    try:
        with open(dat_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Fichier introuvable : {dat_path}")
        return {}

    start = False
    for line in lines:
        if line.strip().startswith("param :") and "gwp_op" in line:
            start = True
            continue
        if start:
            if line.strip() == ';':
                break
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 3:
                res_name = parts[0]
                if res_name in resource_names:
                    try:
                        gwp_ops[res_name] = float(parts[2])  # ✅ colonne 3
                    except ValueError:
                        gwp_ops[res_name] = None
    for r in resource_names:
        gwp_ops.setdefault(r, None)
    return gwp_ops

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

def build_extended_dataframe_fixed(base_dir, storage_technos, resource_names, export_csv=True):
    """
    Construit un DataFrame résumant pour chaque run :
    - le nom du dossier
    - la ressource modifiée (celle avec le GWP le plus élevé)
    - le GWP de chaque ressource (GAS_RE, H2_RE, etc.)
    - la part installée (f) de chaque techno de stockage
    """
    records = []

    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue

        dat_path = os.path.join(full_path, "ESTD_data.dat")
        if not os.path.exists(dat_path):
            continue

        gwp_ops = read_gwp_op_from_dat_file(dat_path, resource_names)
        modified_resource = max(gwp_ops.items(), key=lambda x: x[1] if x[1] is not None else -1)[0]
        modified_value = gwp_ops[modified_resource]

        assets_path = os.path.join(full_path, "output", "assets.txt")
        df_assets = read_assets_txt(assets_path)
        if df_assets is None:
            continue

        row = {
            "folder": folder,
            "resource_modified": modified_resource,
            "gwp_op_modified": modified_value
        }

        for res in resource_names:
            row[f"gwp_op_{res}"] = gwp_ops.get(res)

        for tech in storage_technos:
            row[f"f_{tech}"] = df_assets.at[tech, ' f'] if tech in df_assets.index else 0.0

        records.append(row)

    df = pd.DataFrame(records)

    if export_csv:
        output_csv = f"gwp_op_f_technos_summary_fixed_sto.csv"
        df.to_csv(output_csv, index=False)
        print(f"✅ CSV sauvegardé : {output_csv}")

    return df

def build_imports_nd_dataframe(base_dir, import_technos, export_csv=True):
    records = []

    for folder in os.listdir(base_dir):
        if not folder.startswith("IMPORTS_ND_"):
            continue
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue

        # 🔍 Extrait la valeur de GWP depuis le nom du dossier (ex: 0_01 → 0.01)
        try:
            gwp_str = folder.split("_")[-2] + "." + folder.split("_")[-1]
            gwp_val = float(gwp_str)
        except (IndexError, ValueError):
            print(f"⚠️ Impossible d'extraire la valeur GWP depuis : {folder}")
            continue

        assets_path = os.path.join(full_path, "output", "assets.txt")
        df_assets = read_assets_txt(assets_path)
        if df_assets is None:
            continue

        row = {
            "folder": folder,
            "gwp_imports": gwp_val
        }

        for tech in import_technos:
            row[f"f_{tech}"] = df_assets.at[tech, ' f'] if tech in df_assets.index else 0.0

        records.append(row)

    df = pd.DataFrame(records)
    df.sort_values(by="gwp_imports", inplace=True)

    if export_csv:
        df.to_csv("imports_nd_techno_shares_sto.csv", index=False)
        print("✅ CSV sauvegardé : imports_nd_techno_shares.csv")

    return df

import matplotlib.pyplot as plt

def plot_import_shares(df, import_technos):
    plt.figure(figsize=(8, 5))
    
    for tech in import_technos:
            plt.plot(
                df["gwp_imports"],
                df[f"f_{tech}"],
                marker="o",
                label=tech.replace("_", " ").title()
            )
    
    plt.xlabel("GWP biofuels [kgCO2eq/Mpkm]")
    plt.ylabel("Installed share (f)")
    plt.title("Impact of biofuel GWP on imported transport technologies")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("✅ Graphiques sauvegardés.")

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
def add_cross_markers_by_column_pairs(df, column_pairs, index_pairs_dict,
                                      color="red", marker="x", size=80, labels=None,
                                      custom_y_values=None):
    """
    Ajoute des marqueurs entre deux colonnes, à une position y définie ou moyenne.
    
    Args:
        ...
        custom_y_values (dict): {("col1", "col2"): [y1, y2, ...]} (optionnel)
    """
    for i, (col1, col2) in enumerate(column_pairs):
        y1 = df[col1].astype(str).str.replace(",", ".").astype(float)
        y2 = df[col2].astype(str).str.replace(",", ".").astype(float)
        pairs = index_pairs_dict.get((col1, col2), [])
        y_custom = custom_y_values.get((col1, col2), []) if custom_y_values else []
        label = labels[i] if labels and i < len(labels) else None

        for j, (idx1, idx2) in enumerate(pairs):
            if idx1 < len(df) and idx2 < len(df):
                x = (idx1+ idx2) / 2
                y = y_custom[j] if j < len(y_custom) else (y1[idx1] + y2[idx2]) / 2
                plt.scatter(x, y, color=color, marker=marker, s=size,
                            label=label if j == 0 else "")

def plot_selected_technos_from_csv(csv_path, selected_technos, x_column="GWP ND", output_name="selected_storage_plot.png"):
    """
    Trace un graphe avec des technologies sélectionnées à partir d’un CSV, en gardant l’ordre des lignes.
    
    Args:
        csv_path (str): chemin vers le fichier CSV
        selected_technos (list of str): noms des technologies à tracer (ex: ["METHANOL_STORAGE"])
        x_column (str): colonne à utiliser pour l’axe x (ex: "GWP ND")
        output_name (str): nom du fichier de sortie
    """
    # Lire avec le bon séparateur
    df = pd.read_csv(csv_path, sep=";")
    
    # Corriger les colonnes si nécessaire (par exemple convertir les virgules en points)
    df[x_column] = df[x_column].astype(str).str.replace(",", ".").astype(float)
    x_vals = list(range(len(df)))

    for tech in selected_technos:
        y_col = f"f_{tech}"
        if y_col in df.columns:
            df[y_col] = df[y_col].astype(str).str.replace(",", ".").astype(float)
        else:
            print(f"⚠️ Colonne manquante dans le CSV : {y_col}")

    # Plot
    plt.figure(figsize=(10, 6))
    #custom_labels = ["Methanol Storage", "Seasonal Storage", "Ammonia Storage"]
    custom_labels = ["CAR_METHANOL", "CAR_FUEL_CELL", "CAR_GASOLINE","CAR_HEV"]
    # Tracer chaque techno
    for i, tech in enumerate(selected_technos):
        y_col = f"f_{tech}"
        if y_col in df.columns:
            label_c = custom_labels[i]
            color = colors[i] if colors and i < len(colors) else None
            plt.plot(x_vals, df[y_col], marker="o", label=label_c, color=color)


    plt.xlabel("Scenario index")
    plt.ylabel("Installed capacities [Mp$\cdot$km]")
    plt.title("Private mobility technology trends under increasing GWP of renewable imports")
    ax2 = plt.gca()
    column_pairs = [
        ("f_CAR_FUEL_CELL", "f_CAR_GASOLINE"),
        ("f_CAR_FUEL_CELL", "f_CAR_HEV"),
        ("f_CAR_GASOLINE", "f_CAR_HEV")
    ]
    
    index_pairs_dict = {
        ("f_CAR_FUEL_CELL", "f_CAR_GASOLINE"): [(2, 3)],
        ("f_CAR_FUEL_CELL", "f_CAR_HEV"): [(42, 43)],
        ("f_CAR_GASOLINE", "f_CAR_HEV") : [(37, 38)]
    }
    
    labels = ["Tipping points"]
    
    custom_y = {
    ("f_CAR_FUEL_CELL", "f_CAR_GASOLINE"): [70],
    ("f_CAR_FUEL_CELL", "f_CAR_HEV"): [90],
    ("f_CAR_GASOLINE", "f_CAR_HEV") : [95]
    }
    
    add_cross_markers_by_column_pairs(
        df,
        column_pairs=column_pairs,
        index_pairs_dict=index_pairs_dict,
        labels=labels,
        color="palevioletred",
        marker="s",
        size=50,
        custom_y_values=custom_y
    )
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)  
    plt.axvline(x=36, color="palevioletred", linestyle="--", linewidth=2, label = 'Fixation point for dominant GWP values')
    legend1 = plt.legend()
    legend1.get_frame().set_linewidth(0)
    legend1.get_frame().set_facecolor('none')
    # Exemple : barre verticale sur le scénario 12


    plt.tight_layout()
    plt.savefig(output_name)
    plt.show()



# Exécution avec la correction

# Définition des paramètres
#base_dir = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/PB_imports"  # Dossier contenant les dossiers de simulation
technos_mobility_private = ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG",
                            "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"]
resource_names = ["GAS_RE", "H2_RE", "METHANOL_RE", "AMMONIA_RE"]
technos_storage_elec = ["BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS"]

technos_storage_thermal = ["TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS",
                           "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP"]

technos_storage_other = ["GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE",
                         "GASOLINE_STORAGE", "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"]

technos_storage = technos_storage_elec + technos_storage_thermal + technos_storage_other


#df = build_extended_dataframe_fixed(base_dir, technos_storage, resource_names)

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

technos_mobility_public = ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL",
                           "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"]

technos_mobility_private = ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG",
                            "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"]

technos_mobility_freight = ["TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG",
                            "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL", "TRUCK_METHANOL", "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"]

technos_mobility = technos_mobility_public  + technos_mobility_private + technos_mobility_freight


# technos_conversion = [
#      "HABER_BOSCH", "SYN_METHANOLATION", "METHANE_TO_METHANOL", "BIOMASS_TO_METHANOL",
#      "OIL_TO_HVC", "GAS_TO_HVC", "BIOMASS_TO_HVC", "METHANOL_TO_HVC",
#      "AMMONIA_TO_H2", "PYROLYSIS_TO_LFO", "PYROLYSIS_TO_FUELS"]
# base_dir = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/PB_imports"
# technos_clefs =   technos_storage
# df = build_techno_timeseries(base_dir, technos_clefs)
# plot_techno_evolution(df, group_by_resource=True)


# # ➕ Ajout : tableau des points de rupture
# df_bascule = detect_bascule_disparition(df)
# df_bascule.to_csv("points_de_bascule_et_disparition.csv", index=False)
# print(df_bascule)

# base_dir = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/GWP op ND"  # ← adapte ici

# df_imports = build_imports_nd_dataframe(base_dir, technos_storage)
# plot_import_shares(df_imports, technos_storage)

csv_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/GWP_op_mob_all.csv"
#selected_technos = ["METHANOL_STORAGE", "TS_DHN_SEASONAL", "AMMONIA_STORAGE"]
selected_technos = ["CAR_METHANOL", "CAR_FUEL_CELL", "CAR_GASOLINE","CAR_HEV"]
plot_selected_technos_from_csv(csv_path, selected_technos)
