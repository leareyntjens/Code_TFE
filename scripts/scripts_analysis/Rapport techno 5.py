# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 08:42:40 2025

@author: reynt
"""

# -*- coding: utf-8 -*-
"""
Script de points de bascule mobilité avec gwp_op et gwp_constr.
"""

import pandas as pd
import subprocess
from pathlib import Path
import yaml
from io import StringIO

# === CONFIGURATION ===
base_path = Path('C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py')
technos_csv_path = base_path / 'Data/2050/Technologies.csv'
resources_csv_path = base_path / 'Data/2050/Resources.csv'
yaml_path = base_path / 'scripts/config_ref.yaml'
script_path = base_path / 'scripts/run_energyscope.py'

categorie = "stockage"
delta_constr = 5
colonne_constr = "gwp_constr"

familles = {
    "stockage": [
        "BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS",
        "TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL",
        "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL",
        "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP",
        "GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE", "LFO_STORAGE",
        "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"
    ]
}

sous_familles = {
    "stockage": {
        "electrique": ["BATT_LI", "BEV_BATT"],
        "hybride": ["PHEV_BATT"],
        "PHS": ["PHS"],         
        "thermique": [
            "TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL",
            "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD",
            "TS_DEC_BOILER_OIL"], 
        "Daily": ["TS_DHN_DAILY"],
        "Season": ["TS_DHN_SEASONAL"],
        "High_temp": ["TS_HIGH_TEMP"], 
        "Gas_sto": ["GAS_STORAGE"], 
        "H2_sto": ["H2_STORAGE"],
        "fuel_sto": [
            "DIESEL_STORAGE", "GASOLINE_STORAGE", "LFO_STORAGE",
            "AMMONIA_STORAGE", "METHANOL_STORAGE"], 
        "CO2_sto": ["CO2_STORAGE"
        ]
    }
}



mapping_csv = {
    "BATT_LI": "Batteries",
    "BEV_BATT": "Batteries of electric cars",
    "PHEV_BATT": "Batteries of plug-in hybrid cars",
    "PHS": "Pumped hydro storage",
    "TS_DEC_DIRECT_ELEC": "Decentralised direct elec",
    "TS_DEC_HP_ELEC": "Decentralised electrical heat pumps",
    "TS_DEC_THHP_GAS": "Decentralised thermal heat pumps gas",
    "TS_DEC_COGEN_GAS": "Decentralised cogeneration gas",
    "TS_DEC_COGEN_OIL": "Decentralised cogenration oil",
    "TS_DEC_ADVCOGEN_GAS": "Decentralised fuel cell cogeneration gas",
    "TS_DEC_ADVCOGEN_H2": "Decentralised fuel cell cogenration H2",
    "TS_DEC_BOILER_GAS": "Decentralised boiler gas",
    "TS_DEC_BOILER_WOOD": "Decentralised boiler wood",
    "TS_DEC_BOILER_OIL": "Decentralised boiler oil",
    "TS_DHN_DAILY": "Centralised daily",
    "TS_DHN_SEASONAL": "Centralised seasonal",
    "TS_HIGH_TEMP": "Thermal storage industries",
    "GAS_STORAGE": "Gas storage",
    "H2_STORAGE": "Hydrogen storage",
    "DIESEL_STORAGE": "Diesel storage",
    "GASOLINE_STORAGE": "Gasoline storage",
    "LFO_STORAGE": "LFO storage",
    "AMMONIA_STORAGE": "Ammonia storage",
    "METHANOL_STORAGE": "Methanol storage",
    "CO2_STORAGE": "CO2 storage"
}



# categorie = "fret"
# delta_constr = 10
# colonne_constr = "gwp_constr"

# ressources_non_modifiees = {"BIOETHANOL", "BIODIESEL"}

# familles = {
#     "privee": ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"],
#     "publique": ["BUS_COACH_DIESEL", "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAMWAY_TROLLEY", "TRAIN_PUB"],
#     "fret": ["TRUCK_DIESEL", "TRUCK_METHANOL", "TRUCK_NG", "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG", "BOAT_FREIGHT_METHANOL"]
# }

# sous_familles = {
#     "privee": {
#         "electrique": ["CAR_BEV"],
#         "hybride": ["CAR_PHEV", "CAR_HEV"],
#         "gaz": ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL"],
#         "fuel_cell": ["CAR_FUEL_CELL"]
#     },
#     "publique": {
#         "electrique": ["TRAMWAY_TROLLEY", "TRAIN_PUB"],
#         "hybride": ["BUS_COACH_HYDIESEL"],
#         "gaz": ["BUS_COACH_CNG_STOICH", "BUS_COACH_DIESEL"],
#         "fuel_cell": ["BUS_COACH_FC_HYBRIDH2"]
#     },
#     "fret": {
#         "electrique_camion": ["TRUCK_ELEC"],
#         "electrique_train": ["TRAIN_FREIGHT"],
#         "gaz_camion": ["TRUCK_NG", "TRUCK_DIESEL", "TRUCK_METHANOL"],
#         "gaz_boat": [ "BOAT_FREIGHT_NG",  "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_METHANOL"],
#         "fuel_cell": ["TRUCK_FUEL_CELL"]
#     }
# }

# mapping_csv = {
#     "CAR_GASOLINE": "Car gasoline",
#     "CAR_DIESEL": "Car diesel",
#     "CAR_NG": "Car gas",
#     "CAR_METHANOL": "Car methanol",
#     "CAR_HEV": "Car hybrid (gasoline)",
#     "CAR_PHEV": "Car plug-in hybrid",
#     "CAR_BEV": "Car electric",
#     "CAR_FUEL_CELL": "Car fuel cell (H2)",
#     "BUS_COACH_DIESEL": "Bus diesel",
#     "BUS_COACH_HYDIESEL": "Bus diesel hybrid",
#     "BUS_COACH_CNG_STOICH": "Bus gas",
#     "BUS_COACH_FC_HYBRIDH2": "Bus fuel cell (H2)",
#     "TRAMWAY_TROLLEY": "Tram or metro",
#     "TRAIN_PUB": "Train (passenger)",
#     "TRUCK_DIESEL": "Trucks diesel",
#     "TRUCK_METHANOL": "Truck methanol",
#     "TRUCK_NG": "Truck gas",
#     "TRUCK_FUEL_CELL": "Truck fuel cell (hydrogen)",
#     "TRUCK_ELEC": "Truck electric",
#     "TRAIN_FREIGHT": "Train (freight)",
#     "BOAT_FREIGHT_DIESEL": "Boat diesel",
#     "BOAT_FREIGHT_NG": "Boat gas",
#     "BOAT_FREIGHT_METHANOL": "Boat methanol"
# }

# techno_to_ressource = {
#     "CAR_GASOLINE": "BIOETHANOL",
#     "CAR_DIESEL": "BIODIESEL",
#     "CAR_NG": "GAS",
#     "CAR_METHANOL": "METHANOL",
#     "CAR_HEV": "BIOETHANOL",
#     "CAR_PHEV": "BIOETHANOL",
#     "CAR_BEV": "ELECTRICITY",
#     "CAR_FUEL_CELL": "H2",
#     "BUS_COACH_DIESEL": "BIODIESEL",
#     "BUS_COACH_HYDIESEL": "BIODIESEL"
# }

def read_assets(run_name):
    path = base_path / 'case_studies' / run_name / 'output' / 'assets.txt'
    if not path.exists():
        print(f"❌ Fichier non trouvé : {path}")
        return None
    with open(path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split('\t')
    data = pd.read_csv(StringIO(''.join(lines[2:])), sep='\t', header=None)
    data.columns = header
    data.set_index('TECHNOLOGIES', inplace=True)
    return data

def read_resources_csv(path, header_line=2):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header = lines[header_line].strip().split(';')
    data_lines = lines[header_line + 1:]
    df = pd.DataFrame([l.strip().split(';') for l in data_lines], columns=header)
    df["parameter name"] = df["parameter name"].str.strip()
    return df, lines[:header_line + 1]

def run_simulation_with_name(run_name):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['case_study'] = run_name
    config['cs_path'] = 'case_studies'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    subprocess.run(["python", str(script_path)], shell=True)

# === INIT
i = 0
visited_sousfamilles = set()
last_dominant_sousfamille = None
non_dominant_counter = 0
last_newly_visited = None  # Dernière sous-famille à avoir complété visited_sousfamilles
sousfamilles_cat = sous_familles[categorie]
non_dominant_counters = {sf: 0 for sf in sousfamilles_cat}  # ✅ ici

run_name = f"Init_{categorie.capitalize()}"
run_simulation_with_name(run_name)

while True:
    print(f"\n🔁 Itération {i} — Lecture de {run_name}")
    assets_df = read_assets(run_name)
    if assets_df is None:
        break

    tech_df = pd.read_csv(technos_csv_path, sep=';')
    res_df, res_header = read_resources_csv(resources_csv_path)

    technos = familles[categorie]
    sousfamilles_cat = sous_familles[categorie]
    techno_f = {t: assets_df.loc[t, ' f'] for t in technos if t in assets_df.index and assets_df.loc[t, ' f'] > 1e-4}
    if not techno_f:
        print("❌ Aucune techno dominante.")
        break

    techno_dom = max(techno_f, key=techno_f.get)
    sousfamille_dom = next((sf for sf, lst in sousfamilles_cat.items() if techno_dom in lst), None)
    if sousfamille_dom is None:
        print(f"❌ Sous-famille inconnue pour {techno_dom}")
        break
    print(f"➡️ Techno dominante : {techno_dom} ({sousfamille_dom})")
    
    # === Mise à jour des compteurs de non-dominance
    for sf in non_dominant_counters:
        if sf == sousfamille_dom:
            non_dominant_counters[sf] = 0
        else:
            non_dominant_counters[sf] += 1
            print(f"⏱️ La sous-famille {sf} n’est plus dominante depuis {non_dominant_counters[sf]} itérations.")
    
    # === Ajout d’une nouvelle sous-famille dominante
    if sousfamille_dom not in visited_sousfamilles:
        visited_sousfamilles.add(sousfamille_dom)
        if len(visited_sousfamilles) == len(sousfamilles_cat):
            last_newly_visited = sousfamille_dom
            print(f"✅ Toutes les sous-familles ont été dominantes. Dernière ajoutée : {last_newly_visited}")
    
    # === Test d’éviction de la dernière sous-famille découverte
    if last_newly_visited and sousfamille_dom != last_newly_visited:
        if non_dominant_counters[last_newly_visited] >= 4:
            print(f"🎯 La dernière sous-famille dominante ({last_newly_visited}) a été évincée pendant 4 itérations consécutives.")
            break
    

    nom_csv = mapping_csv.get(techno_dom, techno_dom)



    techs_sf = sous_familles[categorie][sousfamille_dom]
    noms_csv = [mapping_csv.get(t, t) for t in techs_sf]
    gwp_new = None  # pour être utilisé dans le nom du run
    for nom in noms_csv:
        mask = tech_df['Technologies name'].str.strip() == nom
        if not mask.any():
            continue
        gwp_old = float(pd.to_numeric(tech_df.loc[mask, colonne_constr], errors='coerce').iloc[0])
        gwp_new = gwp_old + delta_constr
        tech_df.loc[mask, colonne_constr] = gwp_new
        print(f"🔧 {nom} gwp_constr : {gwp_old:.1f} → {gwp_new:.1f}")
    
    # ✅ Sauvegarde
    tech_df.to_csv(technos_csv_path, sep=';', index=False)

    run_name = f"{categorie.capitalize()}_{sousfamille_dom.upper()}_constr_{int(round(gwp_new))}"

    run_simulation_with_name(run_name)
    # === Logging
    log_path = base_path / 'scripts' / 'log.csv'
    log_entry = {
        "run_name": run_name,
        "iteration": i,
        "techno_dominante": techno_dom,
        "sous_famille": sousfamille_dom,
        "type_modification": "constr",
        "valeur_modifiee": round(gwp_new, 4)
    }
    
    try:
        df_log = pd.read_csv(log_path)
        df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
    except FileNotFoundError:
        df_log = pd.DataFrame([log_entry])
    df_log.to_csv(log_path, index=False)

    last_dominant_sousfamille = sousfamille_dom
    i += 1