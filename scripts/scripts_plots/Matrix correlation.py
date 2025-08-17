# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 09:24:09 2025

@author: reynt
"""

import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

def build_correlation_matrix(base_dir):
    import os
    import pandas as pd

    # === GROUPES DE TECHNOLOGIES (selon ta demande)
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
    technos_storage_elec = ["BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS"]
    technos_storage_thermal = ["TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS",
                               "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP"]
    technos_storage_other = ["GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE",
                             "GASOLINE_STORAGE", "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"]
    technos_conversion = [
        "HABER_BOSCH", "SYN_METHANOLATION", "METHANE_TO_METHANOL", "BIOMASS_TO_METHANOL",
        "OIL_TO_HVC", "GAS_TO_HVC", "BIOMASS_TO_HVC", "METHANOL_TO_HVC",
        "AMMONIA_TO_H2", "PYROLYSIS_TO_LFO", "PYROLYSIS_TO_FUELS"]

    categories_tech = {
        "Electricité": technos_electricity,
        "Chaleur HT": technos_heat_high,
        "Chaleur BT centralisée": technos_heat_low_central,
        "Chaleur BT décentralisée": technos_heat_low_decentral,
        "Mobilité publique": technos_mobility_public,
        "Mobilité privée": technos_mobility_private,
        "Mobilité fret": technos_mobility_freight,
        "Stockage électrique": technos_storage_elec,
        "Stockage thermique": technos_storage_thermal,
        "Stockage autre": technos_storage_other,
        "Conversion": technos_conversion
    }

    #technos_storage = technos_mobility_private
    technos_storage = technos_storage_elec + technos_storage_thermal + technos_storage_other
    sto_utiles = ["BATT_LI","TS_DHN_SEASONAL", "TS_HIGH_TEMP", "TS_DEC_HP_ELEC", "METHANOL_STORAGE", ]
    #sto_utiles = ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG",
                          #      "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"]
    #sto_utiles= technos_storage
    
    new_cat_tech = {'technos_electricity' : ['PV'],
   'technos_heat_high' : ['IND_BOILER_GAS', 'IND_BOILER_WOOD', 'IND_COGEN_GAS', 'IND_DIRECT_ELEC'],
    'technos_heat_low_central': ['DHN_BOILER_GAS', 'DHN_BOILER_OIL', 'DHN_BOILER_WOOD', 'DHN_HP_ELEC'],
    'technos_heat_low_decentral' : ['DEC_BOILER_GAS', 'DEC_DIRECT_ELEC', 'DEC_HP_ELEC', 'DEC_SOLAR'],
    'technos_mobility_public' : ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL",
                               "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"],
    'technos_mobility_private' : ['CAR_DIESEL', 'CAR_FUEL_CELL', 'CAR_GASOLINE', 'CAR_HEV', 'CAR_METHANOL', 'CAR_NG', 'CAR_BEV', 'CAR_PHEV'],
    'technos_mobility_freight' : ["TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG",
                                "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL", "TRUCK_METHANOL", "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"],
    'technos_storage' : [
        "BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS",
        "TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS",
        "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS",
        "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP",
        "GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE",
        "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE"
    ],
    'technos_storage_thermal' : ['TS_DEC_HP_ELEC', 'TS_DHN_SEASONAL', 'TS_HIGH_TEMP'],
    'technos_storage_other' : ['CO2_STORAGE', 'GASOLINE_STORAGE', 'LFO_STORAGE', 'METHANOL_STORAGE'],
    'technos_conversion' : ['METHANE_TO_METHANOL', 'METHANOL_TO_HVC', 'SYN_METHANOLATION']}
    data = []
    def Electrification_from_df(df_assets, col_label="f"):
        """
        Calcule les taux d’électrification pour chaque usage final
        à partir d’un DataFrame assets déjà chargé.
        """
        techs_elec = {
            "High Temp Heat": ["IND_DIRECT_ELEC"],
            "Low Temp Heat": ["DHN_HP_ELEC", "DEC_HP_ELEC", "DEC_DIRECT_ELEC"],
            "Public Mobility": ["TRAMWAY_TROLLEY", "TRAIN_PUB", 'BUS_COACH_FC_HYBRIDEH2', 'BUS_COACH_HYDIESEL'],
            
            "Freight Mobility": ["TRAIN_FREIGHT", "TRUCK_ELEC"],
        "Private Mobility": ["CAR_PHEV", "CAR_BEV"]}
    
        techs_total = {
            "High Temp Heat": ["IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_BOILER_GAS",
                               "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE", "IND_DIRECT_ELEC"],
            "Low Temp Heat": ["DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE", "DHN_COGEN_WET_BIOMASS",
                              "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS", "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO",
                              "DHN_SOLAR", "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL", "DEC_ADVCOGEN_GAS",
                              "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL", "DEC_SOLAR", "DEC_DIRECT_ELEC"],
            "Public Mobility": ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL", "BUS_COACH_HYDIESEL",
                                "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"],
           
            "Freight Mobility": ["TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG",
                                 "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL", "TRUCK_METHANOL",
                                 "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"],
         "Private Mobility": ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL",
                              "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"]}
    
        results = {}
        for usage in techs_total:
            total = df_assets.loc[df_assets.index.intersection(techs_total[usage]), col_label].sum()
            elec = df_assets.loc[df_assets.index.intersection(techs_elec[usage]), col_label].sum()
            if elec  < 0.001 and total > 0:
                results[f"Elec_{usage.replace(' ', '_')}"] = None  # ou np.nan
            else : 
                results[f"Elec_{usage.replace(' ', '_')}"] = 100 * elec / total if total > 0 else 0
        return results

    all_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    

    for run in all_runs:
        run_path = os.path.join(base_dir, run, "output")
        if not os.path.exists(run_path):
            continue

        run_data = {}

        try:
            df_gwp = pd.read_csv(os.path.join(run_path, "gwp_breakdown.txt"), sep="\t", index_col=0)
            run_data["GWP_total"] = df_gwp["GWP_op"].sum() + df_gwp["GWP_constr"].sum()
            for tech in sto_utiles:
                run_data[f"gwp_constr_{tech}"] = df_gwp.loc[tech, "GWP_constr"] if tech in df_gwp.index else None
        except Exception as e:
            print(f"[{run}] ERREUR gwp_breakdown.txt : {e}")
            continue

        try:
            df_assets = pd.read_csv(os.path.join(run_path, "assets.txt"), sep="\t", index_col=0, skiprows=[1])
            df_assets.columns = df_assets.columns.str.strip()
            # Taux d’électrification par usage final
            electrif_secteurs = Electrification_from_df(df_assets)
            run_data.update(electrif_secteurs)

            if "f" not in df_assets.columns:
                raise KeyError("Colonne 'f' absente")
            run_data["storage_capacity_total"] = df_assets.loc[df_assets.index.isin(technos_storage), "f"].sum()
            for tech in df_assets.index:
                if "f" in df_assets.columns:
                    run_data[f"f_{tech}"] = df_assets.loc[tech, "f"]
        except Exception as e:
            print(f"[{run}] ERREUR assets.txt : {e}")
            continue

        try:
            df_res = pd.read_csv(os.path.join(run_path, "resources_breakdown.txt"), sep="\t", index_col=0)
            for res in ["ELECTRICITY", "GASOLINE", "DIESEL", "BIOETHANOL", "BIODIESEL", "LFO",
            "GAS", "GAS_RE", "WOOD", "WET_BIOMASS", "COAL", "WASTE", "H2_RE", "AMMONIA", "METHANOL", "AMMONIA_RE", "METHANOL_RE",
            "RES_WIND", "RES_SOLAR", "RES_HYDRO",]:
                run_data[f"Used_{res}"] = df_res.loc[res, "Used"] if res in df_res.index else 0
            
        except Exception as e:
            print(f"[{run}] ERREUR resources_breakdown.txt : {e}")
            continue

        try:
            df_cost = pd.read_csv(os.path.join(run_path, "cost_breakdown.txt"), sep="\t", index_col=0)
            for tech in df_cost.index:
                val = df_cost.loc[tech, "C_inv"]
                if val > 0.1:
                    run_data[f"c_inv_{tech}"] = val
        except Exception as e:
            print(f"[{run}] ERREUR cost_breakdown.txt : {e}")
            continue

        data.append(run_data)

    df_raw = pd.DataFrame(data)
    df_raw = df_raw.dropna(thresh=int(0.6 * df_raw.shape[1]))

    if df_raw.empty:
        print("⚠️ Aucun run exploitable.")
        return {}, df_raw
    
    

    # === Colonnes
    param_cols = [col for col in df_raw.columns if col.startswith("gwp_constr_")]
    assets_cols = [col for col in df_raw.columns if col.startswith("f_") or col == "storage_capacity_total"]
    resources_cols = [col for col in df_raw.columns if col.startswith("Used_")]
    cost_cols = [col for col in df_raw.columns if col.startswith("c_inv_")]
    other_cols = [col for col in df_raw.columns if col not in param_cols + assets_cols + resources_cols + cost_cols]
    electrif_cols = [col for col in df_raw.columns if col.startswith("electrification_")]



    # === Corrélations principales
    df_corr_assets = df_raw[param_cols + assets_cols].corr().loc[param_cols, assets_cols]
    df_corr_resources = df_raw[param_cols + resources_cols].corr().loc[param_cols, resources_cols]
    df_corr_cost = df_raw[param_cols + cost_cols].corr().loc[param_cols, cost_cols]
    df_corr_other = df_raw[param_cols + other_cols].corr().loc[param_cols, other_cols]
    df_corr_electrif = df_raw[param_cols + electrif_cols].corr().loc[param_cols, electrif_cols]

    # === Corrélations par catégorie (C_inv)
    df_corr_cinv_by_cat = {}
    for cat_name, tech_list in categories_tech.items():
        cols = [f"c_inv_{tech}" for tech in tech_list if f"c_inv_{tech}" in df_raw.columns]
        if cols:
            df_corr_cinv_by_cat[cat_name] = df_raw[param_cols + cols].corr().loc[param_cols, cols]

    # === Corrélations par catégorie (f)
    df_corr_f_by_cat = {}
    for cat_name, tech_list in new_cat_tech.items():
        cols = [f"f_{tech}" for tech in tech_list if f"f_{tech}" in df_raw.columns]
        if cols:
            df_corr_f_by_cat[cat_name] = df_raw[param_cols + cols].corr().loc[param_cols, cols]
        # === Corrélation combinée sur toutes les technologies de new_cat_tech (f uniquement)
    all_techs_f = []
    for techs in new_cat_tech.values():
        all_techs_f.extend(techs)
    all_f_cols = [f"f_{tech}" for tech in all_techs_f if f"f_{tech}" in df_raw.columns]

    df_corr_f_all = None
    if all_f_cols:
        df_corr_f_all = df_raw[param_cols + all_f_cols].corr().loc[param_cols, all_f_cols]
        
        # === Corrélation pour les technologies de mobilité (f uniquement)
    mobility_keys = ["technos_mobility_public", "technos_mobility_freight"]
    techs_mobility = []
    for key in mobility_keys:
        techs_mobility.extend(new_cat_tech[key])
    f_cols_mobility = [f"f_{tech}" for tech in techs_mobility if f"f_{tech}" in df_raw.columns]

    df_corr_f_mobility = None
    if f_cols_mobility:
        df_corr_f_mobility = df_raw[param_cols + f_cols_mobility].corr().loc[param_cols, f_cols_mobility]

    # === Corrélation pour les autres technologies (hors mobilité)
    other_keys = [k for k in new_cat_tech if k not in mobility_keys]
    techs_other = []
    for key in other_keys:
        techs_other.extend(new_cat_tech[key])
    f_cols_other = [f"f_{tech}" for tech in techs_other if f"f_{tech}" in df_raw.columns]

    df_corr_f_other = None
    if f_cols_other:
        df_corr_f_other = df_raw[param_cols + f_cols_other].corr().loc[param_cols, f_cols_other]
    

        return {
        "assets": df_corr_assets,
        "resources": df_corr_resources,
        "cost": df_corr_cost,
        "other": df_corr_other,
        "cost_by_category": df_corr_cinv_by_cat,
        "f_by_category": df_corr_f_by_cat,
        "f_mobility": df_corr_f_mobility,
        "f_other": df_corr_f_other, 
        "electrification": df_corr_electrif

    }, df_raw






def plot_correlation_heatmap(df_corr, figsize=(12, 8), threshold=None):
    """
    Affiche une heatmap de la matrice de corrélation.
    
    Arguments :
    - df_corr : DataFrame de corrélation
    - figsize : taille de la figure
    - threshold : si défini, masque les corrélations dont la valeur absolue est < threshold
    """
    plt.figure(figsize=figsize)

    if threshold:
        mask = df_corr.abs() < threshold
    else:
        mask = None

    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, mask=mask)
    plt.tight_layout()
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.show()
    
  
import os
import pandas as pd

def convert_txt_to_csv(txt_path):
    csv_path = txt_path.replace(".txt", ".csv")
    try:
        df = pd.read_csv(txt_path, sep=None, engine="python", skiprows=[1])
        df.columns = df.columns.str.strip()
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        print(f"[ERREUR] Conversion {txt_path} : {e}")
        return None

def extract_top3_from_nested_assets(base_dir, tech_groups, col="f"):
    top3_by_group = {group: set() for group in tech_groups}

    for subdir in os.listdir(base_dir):
        subpath = os.path.join(base_dir, subdir, "output", "assets.txt")
        if os.path.exists(subpath):
            csv_path = convert_txt_to_csv(subpath)
            if not csv_path:
                continue
            try:
                df = pd.read_csv(csv_path, index_col=0)
                if col not in df.columns:
                    print(f"[WARN] Colonne '{col}' absente dans {csv_path}")
                    continue
                for group, techs in tech_groups.items():
                    valid_techs = [(t, df.at[t, col]) for t in techs if t in df.index and pd.notna(df.at[t, col]) and df.at[t, col] > 0]
                    top3 = sorted(valid_techs, key=lambda x: x[1], reverse=True)[:3]
                    top3_by_group[group].update([t[0] for t in top3])
            except Exception as e:
                print(f"[ERREUR] Lecture {csv_path} : {e}")
        else:
            print(f"[SKIP] Fichier non trouvé : {subpath}")
    
    return top3_by_group

# À adapter avec ton propre chemin :
base_dir = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/SA_TS_20%_imp_fin"
#base_dir = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/SA_TS_20%_imp_fin"
df_corr_all, df_raw = build_correlation_matrix(base_dir)
corrs, df = build_correlation_matrix(base_dir)
#sto_utiles = ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG",
 #                               "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"]
sto_utiles = ["BATT_LI","TS_DHN_SEASONAL", "TS_HIGH_TEMP", "TS_DEC_HP_ELEC", "METHANOL_STORAGE", ]

 
import matplotlib.pyplot as plt
import seaborn as sns

def plot_gwp_vs_f(df_raw, technos, seuil_f=1e-4):
    for tech in technos:
        col_gwp = f"gwp_constr_{tech}"
        col_f = f"f_{tech}"

        if col_gwp in df_raw.columns and col_f in df_raw.columns:
            df_plot = df_raw[[col_gwp, col_f]].copy()
            df_plot = df_plot.dropna()
            df_plot = df_plot[df_plot[col_f] > seuil_f]

            if df_plot.empty:
                print(f"[INFO] {tech} ignorée (aucune valeur f > {seuil_f})")
                continue

            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df_plot, x=col_gwp, y=col_f)
            sns.lineplot(data=df_plot, x=col_gwp, y=col_f, ci=None, color="orange", lw=1)
            plt.title(f"{tech} – f vs gwp_constr")
            plt.xlabel("GWP de construction")
            plt.ylabel("Capacité installée (f)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"[WARN] Colonnes manquantes pour {tech}")

# Exemple d'appel
plot_gwp_vs_f(df_raw, sto_utiles)

plot_correlation_heatmap(corrs["f_mobility"], figsize=(12, 8), threshold=0)
plot_correlation_heatmap(corrs["f_other"], figsize=(12, 8), threshold=0.2)
plot_correlation_heatmap(df_corr_all["resources"], figsize=(12, 8), threshold=0.2)
# plot_correlation_heatmap(df_corr_all["cost"], figsize=(12, 6), threshold=0.1)
plot_correlation_heatmap(df_corr_all["other"], figsize=(8, 5), threshold=0.1)
# plot_correlation_heatmap(corrs["electrification"], figsize=(10, 6), threshold=0.2)


for cat, df_cat in df_corr_all["f_by_category"].items():
    print(f"\n==== Corrélation GWP_constr vs f pour {cat} ====")
    plot_correlation_heatmap(df_cat, figsize=(12, 6), threshold=0)


tech_groups = {   

"technos_electricity": ["NUCLEAR", "CCGT", "CCGT_AMMONIA", "COAL_US", "COAL_IGCC",
                       "PV", "WIND_ONSHORE", "WIND_OFFSHORE", "HYDRO_RIVER", "GEOTHERMAL"],

"technos_heat_high":[
     "IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_BOILER_GAS",
     "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE", "IND_DIRECT_ELEC"],


"technos_heat_low_central":[
    "DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE",
     "DHN_COGEN_WET_BIOMASS", "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS",
     "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO", "DHN_SOLAR"],

"technos_heat_low_decentral":[
     "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL", "DEC_ADVCOGEN_GAS",
     "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL", "DEC_SOLAR", "DEC_DIRECT_ELEC"],

"technos_mobility_public": ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL",
                           "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"],

"technos_mobility_private":["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG",
                            "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"],

"technos_mobility_freight": ["TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG",
                            "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL", "TRUCK_METHANOL", "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"],

"technos_storage_elec": ["BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS"],

"technos_storage_thermal":["TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS",
                           "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP"],

"technos_storage_other" :["GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE",
                         "GASOLINE_STORAGE", "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"],

"technos_conversion" : [
     "HABER_BOSCH", "SYN_METHANOLATION", "METHANE_TO_METHANOL", "BIOMASS_TO_METHANOL",
     "OIL_TO_HVC", "GAS_TO_HVC", "BIOMASS_TO_HVC", "METHANOL_TO_HVC",
     "AMMONIA_TO_H2", "PYROLYSIS_TO_LFO", "PYROLYSIS_TO_FUELS"]}


# === Technologies et ressources spécifiques à corréler
techs_f_cibles = [
    "TRUCK_DIESEL", "TRAIN_FREIGHT", "CAR_FUEL_CELL", "CAR_GASOLINE",
    "IND_BOILER_GAS", "IND_DIRECT_ELEC",
    "TS_DEC_HP_ELEC", "TS_HIGH_TEMP", "TS_DHN_SEASONAL", "METHANOL_STORAGE"
]

resources_cibles = ["GAS_RE", "AMMONIA_RE"]

# === Colonnes associées dans df_raw
f_cols = [f"f_{tech}" for tech in techs_f_cibles if f"f_{tech}" in df_raw.columns]
used_cols = [f"Used_{res}" for res in resources_cibles if f"Used_{res}" in df_raw.columns]
param_cols = [col for col in df_raw.columns if col.startswith("gwp_constr_")]

# === Combinaison des colonnes d'intérêt
cols_corr = param_cols + f_cols + used_cols

# === Filtrage du DataFrame et calcul de la matrice de corrélation
df_subset = df_raw[cols_corr]
df_corr_custom = df_subset.corr().loc[param_cols, f_cols + used_cols]

# === Affichage de la matrice de corrélation
plot_correlation_heatmap(df_corr_custom, figsize=(10, 6), threshold=0)



top3_result = extract_top3_from_nested_assets(base_dir, tech_groups)

for group, techs in top3_result.items():
    print(f"{group} : {sorted(techs)}")


