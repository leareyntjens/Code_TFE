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

# === Choix de la catégorie de mobilité ===
categorie = "publique"

familles = {
    "privee": ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"],
    "publique": ["BUS_COACH_DIESEL", "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAMWAY_TROLLEY", "TRAIN_PUB"],
    "fret": ["TRUCK_DIESEL", "TRUCK_METHANOL", "TRUCK_NG", "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG", "BOAT_FREIGHT_METHANOL"]
}

sous_familles = {
    "privee": {
        "electrique": ["CAR_BEV"], 
        "hybride": ["CAR_PHEV", "CAR_HEV"],
        "gaz": ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL"],
        "fuel_cell": ["CAR_FUEL_CELL"]
    },
    "publique": {
        "electrique": ["TRAMWAY_TROLLEY", "TRAIN_PUB"],
        "hybride": ["BUS_COACH_HYDIESEL"],
        "gaz": ["BUS_COACH_CNG_STOICH", "BUS_COACH_DIESEL"],
        "fuel_cell": ["BUS_COACH_FC_HYBRIDH2"]
    },
    "fret": {
        "electrique": ["TRUCK_ELEC", "TRAIN_FREIGHT"],
        "gaz": ["TRUCK_NG", "BOAT_FREIGHT_NG", "TRUCK_DIESEL", "BOAT_FREIGHT_DIESEL", "TRUCK_METHANOL", "BOAT_FREIGHT_METHANOL"],
        "fuel_cell": ["TRUCK_FUEL_CELL"]
    }
}

mapping_csv = {
    "CAR_GASOLINE": "Car gasoline",
    "CAR_DIESEL": "Car diesel",
    "CAR_NG": "Car gas",
    "CAR_METHANOL": "Car methanol",
    "CAR_HEV": "Car hybrid (gasoline)",
    "CAR_PHEV": "Car plug-in hybrid",
    "CAR_BEV": "Car electric",
    "CAR_FUEL_CELL": "Car fuel cell (H2)",
    "BUS_COACH_DIESEL": "Bus diesel",
    "BUS_COACH_HYDIESEL": "Bus diesel hybrid",
    "BUS_COACH_CNG_STOICH": "Bus gas",
    "BUS_COACH_FC_HYBRIDH2": "Bus fuel cell (H2)",
    "TRAMWAY_TROLLEY": "Tram or metro",
    "TRAIN_PUB": "Train (passenger)",
    "TRUCK_DIESEL": "Trucks diesel",
    "TRUCK_METHANOL": "Truck methanol",
    "TRUCK_NG": "Truck gas",
    "TRUCK_FUEL_CELL": "Truck fuel cell (hydrogen)",
    "TRUCK_ELEC": "Truck electric",
    "TRAIN_FREIGHT": "Train (freight)",
    "BOAT_FREIGHT_DIESEL": "Boat diesel",
    "BOAT_FREIGHT_NG": "Boat gas",
    "BOAT_FREIGHT_METHANOL": "Boat methanol"
}

techno_to_ressource = {
    "CAR_GASOLINE": "BIOETHANOL",
    "CAR_DIESEL": "BIODIESEL",
    "CAR_NG": "GAS",
    "CAR_METHANOL": "METHANOL",
    "CAR_HEV": "BIOETHANOL",
    "CAR_PHEV": "BIOETHANOL",
    "CAR_BEV": "ELECTRICITY",
    "CAR_FUEL_CELL": "H2",
    "BUS_COACH_DIESEL": "BIODIESEL",
    "BUS_COACH_HYDIESEL": "BIODIESEL",
}

ressources_non_modifiees = {"BIOETHANOL","BIODIESEL"}
seuil_gwp_op = 0.11
delta_constr = 10
delta_op = 0.01
colonne_constr = "gwp_constr"
colonne_op = "gwp_op"

def read_resources_csv(path, header_line=2):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header = lines[header_line].strip().split(';')
    data_lines = lines[header_line + 1:]
    df = pd.DataFrame([l.strip().split(';') for l in data_lines], columns=header)
    df["parameter name"] = df["parameter name"].str.strip()
    return df

i = 0
initial_values = {}
technos_modifies = {}
tech_dominante = None
sousfamilles_visitees = set()

while True:
    print(f"\n🔁 Simulation {i}")
    tech_df = pd.read_csv(technos_csv_path, sep=';')
    res_df = read_resources_csv(resources_csv_path)

    if i == 0 or tech_dominante is None:
        run_name = f"DetectDominant_{categorie}"
    else:
        nom_csv_prec = mapping_csv.get(tech_dominante, tech_dominante)
        ressource_prec = techno_to_ressource.get(tech_dominante, None)
    
        if ressource_prec in ressources_non_modifiees:
            res_index = res_df[res_df["parameter name"] == ressource_prec].index
            if not res_index.empty:
                val_op = float(res_df.loc[res_index[0], colonne_op])
                run_name = f"{categorie}_{tech_dominante}_op{val_op:.3f}_{i}"
            else:
                run_name = f"{categorie}_{tech_dominante}_opNA_{i}"
        else:
            val_constr = technos_modifies.get(tech_dominante, "NA")
            if isinstance(val_constr, float):
                run_name = f"{categorie}_{tech_dominante}_{val_constr:.0f}_{i}"
            else:
                run_name = f"{categorie}_{tech_dominante}_NA_{i}"

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['case_study'] = run_name
    config['cs_path'] = 'case_studies'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

    result = subprocess.run(["python", str(script_path)], shell=True)
    if result.returncode != 0:
        print("❌ Échec de l'exécution d'EnergyScope.")
        break

    assets_path = base_path / 'case_studies' / run_name / 'output' / 'assets.txt'
    if not assets_path.exists():
        print(f"⚠️ Fichier non trouvé : {assets_path}")
        break

    with open(assets_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split('\t')
    data_lines = lines[2:]
    csv_like = StringIO(''.join(data_lines))
    assets_df = pd.read_csv(csv_like, sep='\t', header=None)
    assets_df.columns = [col.strip() for col in header]
    assets_df.set_index('TECHNOLOGIES', inplace=True)

    technos_categorie = familles[categorie]
    sousfamilles_cat = sous_familles[categorie]

    technos_presents = {tech: assets_df.loc[tech, 'f'] for tech in technos_categorie if tech in assets_df.index and assets_df.loc[tech, 'f'] > 1e-4}
    if not technos_presents:
        print("Aucune technologie présente dans la catégorie.")
        break

    techno_max = max(technos_presents, key=technos_presents.get)
    capacite_max = technos_presents[techno_max]
    print(f"➡️ Techno dominante actuelle : {techno_max} ({capacite_max:.3f} GW)")

    sousfamille_actuelle = None
    for sf_name, sf_techs in sousfamilles_cat.items():
        if techno_max in sf_techs:
            sousfamille_actuelle = sf_name
            break

    if sousfamille_actuelle:
        sousfamilles_visitees.add(sousfamille_actuelle)
        print(f"✅ Sous-famille dominante : {sousfamille_actuelle}")
    else:
        print("❌ Sous-famille inconnue pour la techno dominante")

    nom_csv = mapping_csv.get(techno_max, techno_max)
    ressource = techno_to_ressource.get(techno_max, None)

    if ressource in ressources_non_modifiees:
        res_index = res_df[res_df["parameter name"] == ressource].index
        if not res_index.empty:
            val = float(res_df.loc[res_index[0], colonne_op])
            if val < seuil_gwp_op:
                res_df.loc[res_index[0], colonne_op] = f"{val + delta_op:.5f}"
                print(f"🔧 gwp_op de {ressource} modifié : {val:.3f} → {val + delta_op:.3f}")

                with open(resources_csv_path, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                with open(resources_csv_path, 'w', encoding='utf-8') as f:
                    f.writelines(all_lines[:3])
                    for _, row in res_df.iterrows():
                        f.write(';'.join(str(val) for val in row.values) + '\n')

                i += 1
                continue

    if techno_max not in initial_values:
        match = tech_df[tech_df['Technologies name'].str.strip() == nom_csv]
        if match.empty:
            print(f"❌ ERREUR : technologie '{techno_max}' (→ '{nom_csv}') introuvable dans Technologies.csv.")
            break
        initial_values[techno_max] = float(pd.to_numeric(match[colonne_constr].values[0], errors='coerce'))
        technos_modifies[techno_max] = initial_values[techno_max]

    nouvelle_valeur = technos_modifies[techno_max] + delta_constr
    tech_df.loc[tech_df['Technologies name'] == nom_csv, colonne_constr] = nouvelle_valeur
    technos_modifies[techno_max] = nouvelle_valeur
    tech_df.to_csv(technos_csv_path, sep=';', index=False)
    print(f"🔧 {colonne_constr} de {techno_max} mis à {nouvelle_valeur}")

    tech_dominante = techno_max

    if len(sousfamilles_visitees) == len(sousfamilles_cat):
        print(f"🎯 Toutes les sous-familles ont été visitées : {sousfamilles_visitees}")
        break

    i += 1