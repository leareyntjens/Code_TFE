import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Étape 1 : Chargement des données ===
def collect_data(scenarios, file_name, col_label):
    techno_data = {sc_name: {} for sc_name in scenarios}

    for sc_name, base_path in scenarios.items():
        run_folders = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

        for run in run_folders:
            file_path = os.path.join(base_path, run, 'output', file_name)
            if not os.path.exists(file_path):
                continue
            try:
                df = pd.read_csv(file_path, sep=None, engine='python', skiprows=[1])
                df.columns = df.columns.str.strip()
                for _, row in df.iterrows():
                    techno = row[0]
                    value = row[col_label]
                    techno_data[sc_name].setdefault(techno, []).append(value)
            except Exception as e:
                print(f"❌ Erreur lecture {file_path} : {e}")
    return techno_data

# === Étape 2 : Tracer les boxplots par catégorie ===
def tracer_boxplots_par_categorie(techno_data, categories):
    scenarios = list(techno_data.keys())
    n_scenarios = len(scenarios)

    def should_keep_tech(tech):
        for sc in scenarios:
            values = techno_data.get(sc, {}).get(tech, [])
            if any(v > 0 for v in values):
                return True
        return False

    def plot_boxplot_techs(tech_list, title_suffix):
        data_to_plot = []
        positions = []
        xtick_positions = []
        xtick_labels = []
        width = 0.8 / n_scenarios

        fig, ax = plt.subplots(figsize=(len(tech_list) * 1.4, 6))

        for i, tech in enumerate(tech_list):
            if not should_keep_tech(tech):
                continue

            base = len(xtick_labels)
            xtick_positions.append(base + width * (n_scenarios - 1) / 2)
            xtick_labels.append(tech)

            for j, sc_name in enumerate(scenarios):
                pos = base + j * width
                positions.append(pos)
                values = techno_data[sc_name].get(tech, [0])
                data_to_plot.append(values)

        if not data_to_plot:
            print(f"[ℹ️] Aucune techno affichable pour : {title_suffix}")
            return

        ax.boxplot(data_to_plot, positions=positions, widths=width, patch_artist=True)

        colors = plt.cm.tab10.colors
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors[i % len(colors)]) for i in range(n_scenarios)
        ]
        for patch, color in zip(ax.artists, [colors[i % n_scenarios] for i in range(len(data_to_plot))]):
            patch.set_facecolor(color)

        # Moyennes
        y_min = -max([max(vals) if len(vals) > 0 else 0 for vals in data_to_plot]) * 0.05 

        for idx, (vals, pos) in enumerate(zip(data_to_plot, positions)):
            if len(vals) > 0:
                moyenne = np.mean(vals)
                min_val = np.min(vals)
                max_val = np.max(vals)
                
                # Affichage optionnel dans le graphe (désactivé ici)
                # ax.text(pos, y_min, f"{moyenne:.1f}", ha='center', va='top', fontsize=8, rotation=90)
                
                tech_label = xtick_labels[idx // n_scenarios]
                sc_label = scenarios[idx % n_scenarios]
                
                print(f"📊 {tech_label} ({sc_label})")
                print(f"   ↳ Moyenne : {moyenne:.2f} GWh")
                print(f"   ↳ Min     : {min_val:.2f} GWh")
                print(f"   ↳ Max     : {max_val:.2f} GWh")
            else:
                print(f"📊 {xtick_labels[idx // n_scenarios]} ({scenarios[idx % n_scenarios]}) : Aucun résultat")

            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
        #ax.set_title(f"{title_suffix} – Comparaison des scénarios")
        ax.set_ylabel("Installed capacities [GWh]")
        #ax.legend(legend_handles, scenarios, title="Scénarios")
        #ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.subplots_adjust(bottom=0.2)
        plt.show()

    # Tracer les deux technos spécifiques à part
    for techno_isolée in ["GAS_STORAGE", "TS_DHN_SEASONAL"]:
        plot_boxplot_techs([techno_isolée], techno_isolée)

    # Tracer les autres catégories
    for cat_name, tech_list in categories.items():
        # Exclure GAS_STORAGE et TS_DHN_SEASONAL des catégories
        filtered_techs = [t for t in tech_list if t not in ["GAS_STORAGE", "TS_DHN_SEASONAL"]]
        plot_boxplot_techs(filtered_techs, cat_name)

import os
def extract_storage_capacity(assets_csv_path, col_label="f"):
    """
    Extrait la capacité totale de stockage installée à partir du fichier assets.csv.

    Arguments:
    - assets_csv_path : chemin vers le fichier assets.csv
    - col_label : nom de la colonne contenant la capacité (par défaut "f")

    Retourne :
    - total_storage_capacity : somme des capacités de toutes les technologies de stockage
    """

    stockage_technos = [
        "BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS",
        "TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL",
        "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD",
        "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP",
        "GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE", "LFO_STORAGE",
        "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"
    ]

    try:
        df = pd.read_csv(assets_csv_path)
        df.set_index(df.columns[0], inplace=True)
        total_capacity = sum(df.loc[tech, col_label] for tech in stockage_technos if tech in df.index)
    except Exception as e:
        total_capacity = None  # ou np.nan

    return total_capacity

def Electrification (csv_path, col_label="f"):
    """
    Calcule le taux d’électrification pour chaque usage final :
    - High Temp Heat
    - Low Temp Heat
    - Public Mobility
    - Private Mobility
    - Freight Mobility
    ...et pour le stockage
    """

    df = pd.read_csv(csv_path)
    df.set_index(df.columns[0], inplace=True)

    # === Usages finaux ===
    techs_elec = {
        "High Temp Heat": ["IND_DIRECT_ELEC"],
        "Low Temp Heat": ["DHN_HP_ELEC", "DEC_HP_ELEC", "DEC_DIRECT_ELEC"],
        "Public Mobility": ["TRAMWAY_TROLLEY", "TRAIN_PUB"],
        "Private Mobility": ["CAR_PHEV", "CAR_BEV"],
        "Freight Mobility": ["TRAIN_FREIGHT", "TRUCK_ELEC"]
    }

    techs_total = {
        "High Temp Heat": ["IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_BOILER_GAS",
                           "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE", "IND_DIRECT_ELEC"],
        "Low Temp Heat": ["DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE", "DHN_COGEN_WET_BIOMASS",
                          "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS", "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO",
                          "DHN_SOLAR", "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL", "DEC_ADVCOGEN_GAS",
                          "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL", "DEC_SOLAR", "DEC_DIRECT_ELEC"],
        "Public Mobility": ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL", "BUS_COACH_HYDIESEL",
                            "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"],
        "Private Mobility": ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL",
                             "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"],
        "Freight Mobility": ["TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG",
                             "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL", "TRUCK_METHANOL",
                             "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"]
    }

    # === Stockage ===
    technos_storage_elec = ["BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS"]
    technos_storage_thermal = ["TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS",
                               "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS",
                               "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP"]
    technos_storage_other = ["GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE", "LFO_STORAGE",
                             "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"]
    technos_storage_total = technos_storage_elec + technos_storage_thermal + technos_storage_other


    def somme(tech_list):
        return sum(df.loc[tech, col_label] for tech in tech_list if tech in df.index)

    resultats = {}

    for usage, total_list in techs_total.items():
        total = somme(total_list)
        elec = somme(techs_elec[usage])
        pourcentage = 100 * elec / total if total > 0 else 0
        resultats[usage] = {
            "Total": total,
            "Electric": elec,
            "Electrification [%]": round(pourcentage, 2)
        }

    # Stockage : électrification
    sto_total = somme(technos_storage_total)
    sto_elec = somme(technos_storage_elec)
    sto_ratio = 100 * sto_elec / sto_total if sto_total > 0 else 0
    resultats["Storage Electrification"] = {
        "Total": sto_total,
        "Electric": sto_elec,
        "Electrification [%]": round(sto_ratio, 2)
    }

    return resultats 




# === Paramètres utilisateur ===
scenarios = {
    "sto": "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/SA_TS_20%_imp_fin",  # 🔁 remplace par le bon chemin
}
file_name = "assets.txt"
col_label = "f"

categories = {   

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

"technos_storage_thermal ":["TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS",
                           "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP"],

"technos_storage_other" :["GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE",
                         "GASOLINE_STORAGE", "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"],

"technos_conversion" : [
     "HABER_BOSCH", "SYN_METHANOLATION", "METHANE_TO_METHANOL", "BIOMASS_TO_METHANOL",
     "OIL_TO_HVC", "GAS_TO_HVC", "BIOMASS_TO_HVC", "METHANOL_TO_HVC",
     "AMMONIA_TO_H2", "PYROLYSIS_TO_LFO", "PYROLYSIS_TO_FUELS"]}



# === Exécution ===
techno_data = collect_data(scenarios, file_name, col_label)
tracer_boxplots_par_categorie(techno_data, categories)


