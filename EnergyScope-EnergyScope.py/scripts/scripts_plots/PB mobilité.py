# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 19:32:51 2025

@author: reynt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO


def extract_metadata_from_folder(folder_name):
    parts = folder_name.split('_')
    
    if len(parts) < 4:
        return None, None, None
    
    cat = parts[0]
    mod_type = parts[-2]  # constr ou op
    try:
        gwp_val = float(parts[-1])
    except ValueError:
        return None, None, None

    sousfamille = '_'.join(parts[1:-2])  # tout ce qui est entre cat et mod_type
    return cat.lower(), sousfamille.lower(), gwp_val


def read_assets_txt(path):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split('\t')
        data = pd.read_csv(StringIO(''.join(lines[2:])), sep='\t', header=None)
        data.columns = header
        data.set_index('TECHNOLOGIES', inplace=True)
        return data
    except Exception as e:
        print(f"[Erreur] {path}: {e}")
        return None

def build_mobility_dataframe(base_dir, tech_list):
    data = []
    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue

        cat, sousfamille, gwp = extract_metadata_from_folder(folder)
        if cat is None:
            continue

        assets_path = os.path.join(full_path, "output", "assets.txt")
        df_assets = read_assets_txt(assets_path)
        if df_assets is None:
            continue

        for tech in tech_list:
            capacity = df_assets.at[tech, ' f'] if tech in df_assets.index else 0.0
            data.append({
                "Scenario": folder,
                "Categorie": cat,
                "SousFamille": sousfamille,
                "GWP": gwp,
                "Technology": tech,
                "Capacity": capacity
            })

    return pd.DataFrame(data)

def plot_evolution(df, cat="publique"):
    df_cat = df[df["Categorie"] == cat]
    grouped = df_cat.groupby("SousFamille")
    for sf, group in grouped:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=group, x="GWP", y="Capacity", hue="Technology", marker="o")
        plt.title(f"{cat.capitalize()} — {sf.capitalize()} : Capacité installée vs GWP")
        plt.xlabel("GWP (constr ou op)")
        plt.ylabel("Capacité installée")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def detect_bascule_disparition_stable_with_replacement(df, seuil_disparition=1.0, n_stable=3):
    results = []
    for (cat, sf), df_grp in df.groupby(["Categorie", "SousFamille"]):
        for tech in df_grp["Technology"].unique():
            df_tech = df_grp[df_grp["Technology"] == tech].sort_values(by="GWP")
            gwp_disparition = df_tech[df_tech["Capacity"] < seuil_disparition]["GWP"].min() if not df_tech[df_tech["Capacity"] < seuil_disparition].empty else None

            gwp_values = sorted(df_grp["GWP"].unique())
            non_dominant_streak = 0
            remplacement_streak = []
            gwp_bascule = None
            remplacant = None

            for gwp_val in gwp_values:
                group = df_grp[df_grp["GWP"] == gwp_val]
                max_cap = group["Capacity"].max()
                tech_cap = group[group["Technology"] == tech]["Capacity"].values[0]

                if tech_cap < max_cap:
                    non_dominant_streak += 1
                    dominant_tech = group.loc[group["Capacity"].idxmax(), "Technology"]
                    remplacement_streak.append(dominant_tech)
                else:
                    non_dominant_streak = 0
                    remplacement_streak = []

                if non_dominant_streak >= n_stable:
                    gwp_bascule = gwp_val - (n_stable - 1) * (gwp_values[1] - gwp_values[0])
                    remplacant = remplacement_streak[0]  # premier techno remplaçante dans la séquence stable
                    break

            results.append({
                "Categorie": cat,
                "SousFamille": sf,
                "Technology": tech,
                "GWP_bascule_stable": gwp_bascule,
                "GWP_disparition": gwp_disparition,
                "Remplacee_par": remplacant
            })
    return pd.DataFrame(results)

def plot_impact_on_other_technos(df, cat="publique", familles=None):
    """
    Affiche l'évolution de la capacité de toutes les technologies de la catégorie donnée,
    mais uniquement dans les scénarios où une sous-famille donnée a été modifiée.
    """
    if familles is None:
        print("⚠️ Aucune structure de sous-familles n'a été fournie.")
        return

    df_cat = df[df["Categorie"] == cat]

    for sf_ref in familles.keys():
        # ⬇️ Filtrer les scénarios où la sous-famille modifiée est bien sf_ref
        df_filtered = df_cat[df_cat["SousFamille"] == sf_ref]

        if df_filtered.empty:
            print(f"⚠️ Aucun scénario pour la sous-famille : {sf_ref}")
            continue

        plt.figure(figsize=(12, 5))
        sns.lineplot(data=df_filtered, x="GWP", y="Capacity", hue="Technology", marker="o")
        plt.title(f"Impact du GWP sur toutes les technologies — GWP modifié : {sf_ref}")
        plt.xlabel(f"GWP (appliqué à {sf_ref})")
        plt.ylabel("Capacité installée")
        plt.grid(True)
        plt.tight_layout()
        plt.show()





# === USAGE ===

base_dir = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/Publique fin"

# Tu peux choisir ici la catégorie
categorie = "privee"

# Les technologies à analyser
techs_publique = ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL", "BUS_COACH_HYDIESEL",
                  "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"]
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

technos_storage_elec = ["BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS"]

technos_storage_thermal = ["TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS",
                           "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP"]

technos_storage_other = ["GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE",
                         "GASOLINE_STORAGE", "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"]

technos_storage = technos_storage_elec + technos_storage_thermal + technos_storage_other

technos_conversion = [
     "HABER_BOSCH", "SYN_METHANOLATION", "METHANE_TO_METHANOL", "BIOMASS_TO_METHANOL",
     "OIL_TO_HVC", "GAS_TO_HVC", "BIOMASS_TO_HVC", "METHANOL_TO_HVC",
     "AMMONIA_TO_H2", "PYROLYSIS_TO_LFO", "PYROLYSIS_TO_FUELS"]

df = build_mobility_dataframe(base_dir, techs_publique)
plot_evolution(df, cat=categorie)

df_bascule = detect_bascule_disparition_stable_with_replacement(df, seuil_disparition=1.0, n_stable=1)
df_bascule.to_csv("points_bascule_stables.csv", index=False)

print(df_bascule)

familles_mob_privee = {
     "fuel_cell": ["CAR_FUEL_CELL"],
    "electrique": ["CAR_BEV"],
        "hybride": ["CAR_PHEV", "CAR_HEV"],
        "gaz": ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL"]
       
}

familles_mob_publique = {
    "electrique": ["TRAMWAY_TROLLEY", "TRAIN_PUB"],
    "hybride": ["BUS_COACH_HYDIESEL"],
    "gaz": ["BUS_COACH_CNG_STOICH", "BUS_COACH_DIESEL"],
    "fuel_cell": ["BUS_COACH_FC_HYBRIDH2"]
}

plot_impact_on_other_technos(df, cat="publique", familles=familles_mob_publique)

