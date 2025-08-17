import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIGURATION ===
categorie = "4privé"
base_path = Path("C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py")
case_study_subdir = base_path / "case_studies" / categorie

mapping_csv = {
    "CAR_GASOLINE": "Car gasoline",
    "CAR_DIESEL": "Car diesel",
    "CAR_NG": "Car gas",
    "CAR_METHANOL": "Car methanol",
    "CAR_HEV": "Car hybrid (gasoline)",
    "CAR_PHEV": "Car plug-in hybrid",
    "CAR_BEV": "Car electric",
    "CAR_FUEL_CELL": "Car fuel cell (H2)",
}

# === Initialisation
current_techno = None
gwp_vals = []
capacities = []

# === Parcours des dossiers triés
for folder in sorted(os.listdir(case_study_subdir)):
    folder_path = case_study_subdir / folder / "output" / "assets.txt"
    if not folder_path.exists():
        continue

    parts = folder.split("_")
    if len(parts) < 3:
        continue
    techno = "_".join(parts[1:-1])
    if techno in {"None", ""}:
        continue

    try:
        df_assets = pd.read_csv(folder_path, sep='\t', skiprows=0)
        df_assets.columns = df_assets.columns.str.strip()
        if "TECHNOLOGIES" not in df_assets.columns:
            raise ValueError("Colonne TECHNOLOGIES absente")

        df_assets.set_index("TECHNOLOGIES", inplace=True)

        if techno not in df_assets.index:
            raise ValueError(f"{techno} absent de assets.txt")

        capacity = df_assets.at[techno, "f"]
        gwp_constr = df_assets.at[techno, "gwp_constr"]

        # Si changement de techno : afficher le graphe précédent
        if current_techno is not None and techno != current_techno:
            plt.figure(figsize=(8, 6))
            plt.plot(gwp_vals, capacities, marker='o', linestyle='--', color='blue')
            plt.xlabel("GWP de construction [kgCO2eq/kW]")
            plt.ylabel("Capacité installée [GW]")
            plt.title(f"Évolution de {mapping_csv.get(current_techno, current_techno)}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            # Réinitialiser pour nouvelle techno
            gwp_vals = []
            capacities = []

        # Enregistrer les valeurs pour la techno actuelle
        gwp_vals.append(gwp_constr)
        capacities.append(capacity)
        current_techno = techno

    except Exception as e:
        print(f"Erreur dans {folder}: {e}")
        continue

# Dernier graphe après la boucle
if current_techno and gwp_vals:
    plt.figure(figsize=(8, 6))
    plt.plot(gwp_vals, capacities, marker='o', linestyle='--', color='blue')
    plt.xlabel("GWP de construction [kgCO2eq/kW]")
    plt.ylabel("Capacité installée [GW]")
    plt.title(f"Évolution de {mapping_csv.get(current_techno, current_techno)}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
