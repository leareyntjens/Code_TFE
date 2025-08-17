import pandas as pd 
import subprocess
from pathlib import Path
import yaml
from io import StringIO

# === CONFIGURATION ===
base_path = Path('C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py')
tech_csv_path = base_path / 'Data/2050/Technologies.csv'
yaml_path = base_path / 'scripts/config_ref.yaml'
script_path = base_path / 'scripts/run_energyscope.py'

# === Technologies à modifier ===
# Format : {nom_dans_csv: (colonne_à_modifier, delta)}
modifications = {
    'Bus diesel hybrid': ('gwp_constr', 10),
}
                                  
                                              
                                                                                                                                         
# === Technologies à surveiller dans assets.txt ===
# Format : {tech_name_in_assets: col_to_check}
watch_list = {
    'BUS_COACH_HYDIESEL': 'f',

}

# === Dictionnaire Nom usuel -> EnergyScope
nom_techno = {
    "Tram or metro": "TRAMWAY_TROLLEY",
    "Bus diesel": "BUS_COACH_DIESEL",
    "Bus diesel hybrid": "BUS_COACH_HYDIESEL",
    "Bus gas": "BUS_COACH_CNG_STOICH",
    "Bus fuel cell (H2)": "BUS_COACH_FC_HYBRIDH2",
    "Train (passenger)": "TRAIN_PUB",
    "Car gasoline": "CAR_GASOLINE",
    "Car diesel": "CAR_DIESEL",
    "Car gas": "CAR_NG",
    "Car methanol": "CAR_METHANOL",
    "Car hybrid (gasoline)": "CAR_HEV",
    "Car plug-in hybrid": "CAR_PHEV",
    "Car electric": "CAR_BEV",
    "Car fuel cell (H2)": "CAR_FUEL_CELL",
    "Train (freight)": "TRAIN_FREIGHT",
    "Boat diesel": "BOAT_FREIGHT_DIESEL",
    "Boat gas": "BOAT_FREIGHT_NG",
    "Boat methanol": "BOAT_FREIGHT_METHANOL",
    "Trucks diesel": "TRUCK_DIESEL",
    "Truck methanol": "TRUCK_METHANOL",
    "Truck fuel cell (hydrogen)": "TRUCK_FUEL_CELL",
    "Truck electric": "TRUCK_ELEC",
    "Truck gas": "TRUCK_NG",
}

# === Stock des valeurs initiales ===
initial_values = {}

# === Boucle principale ===
i = 0
while True:
    tech_df = pd.read_csv(tech_csv_path, sep=';')
    print(f"\n🔁 Simulation {i}")

    for tech, (col, delta) in modifications.items():
        row_index = tech_df[tech_df['Technologies name'].str.strip() == tech].index
        if row_index.empty:
            print(f"❌ Technologie non trouvée : {tech}")
            continue

        idx = row_index[0]

        # Stocker la valeur initiale une seule fois
        if i == 0:
            initial_values[tech] = float(tech_df.loc[idx, col])
        initial = initial_values[tech]
        new_value = initial + i * delta
        tech_df.loc[idx, col] = new_value
        print(f" - {col} de {tech} mis à {new_value:.2f}")

    tech_df.to_csv(tech_csv_path, sep=';', index=False)

    # === Modifier le fichier YAML ===
    run_suffix = '_'.join([
        f"{k.replace(' ', '_')}_{tech_df.loc[tech_df['Technologies name'] == k, modifications[k][0]].values[0]:.2f}".replace('.', '_')
        for k in modifications
    ])
    run_name = f"MultiTech_{run_suffix}"
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['case_study'] = run_name
    config['cs_path'] = 'case_studies'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

    # === Lancer EnergyScope ===
    result = subprocess.run(["python", str(script_path)], shell=True)
    if result.returncode != 0:
        print("❌ Échec de l'exécution d'EnergyScope.")
        break

    # === Lire le fichier assets.txt ===
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

    # === Vérification de l'utilisation des technologies ciblées ===
    fin = True
    for tech, col in watch_list.items():
        try:
            val = assets_df.loc[tech, col]
            print(f"➡️  Capacité installée de {tech} : {val} GW")
            if val >= 1e-4:
                fin = False
        except KeyError:
            print(f"❌ Clé absente : {tech} ou colonne {col}")
            fin = False

    if fin:
        print("✅ Toutes les technologies surveillées sont désactivées. Fin de boucle.\n")
        
        # === Affichage final des technologies présentes avec capacité
        print("📋 Résumé des technologies présentes :\n")
        print("{:<30} {:<30} {:>20}".format("Nom usuel", "Nom EnergyScope", "Capacité installée (GW)"))
        print("-" * 85)

        for nom_usuel, nom_scope in nom_techno.items():
            try:
                val = assets_df.loc[nom_scope, 'f']
                print("{:<30} {:<30} {:>20.5f}".format(nom_usuel, nom_scope, val))
            except KeyError:
                continue

        break

    i += 1
