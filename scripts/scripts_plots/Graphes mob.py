import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# === Dossier pour sauvegarder les graphiques ===
dossier_images = "C:/Users/reynt/OneDrive - UCL/MASTER/Mémoire/Résultats/12 mai"
os.makedirs(dossier_images, exist_ok=True)


# === Colonne dans le fichier CSV où se trouve le nom des technologies ===
colonne_techno = "technologies"

# === Mappage : affichage sur le graphe → nom réel dans assets.csv ===
technos_a_analyser= {
    "Car electric": "CAR_BEV",
    "car_diesel": "CAR_DIESEL",
    "car_fuel_cell": "CAR_FUEL_CELL",
    "car_gas": "CAR_NG",
    "car_methanol": "CAR_METHANOL",
    
}
technos_a_analyser_pub = {
    "Bus_diesel": "BUS_COACH_DIESEL",
    "Bus_diesel_hybrid": "BUS_COACH_HYDIESEL",
    "Bus_fuel_cell(H2)": "BUS_COACH_CNG_STOICH",
    #Train_(passenger)": "TRAIN_PUB",
    #"Tram_or_metro": "TRAMWAY_TROLLEY",
    
}
technos_a_analyser_freight = {
    "boat_diesel": "BOAT_FREIGHT_DIESEL",
    "boat_methanol": "BOAT_FREIGHT_METHANOL",
    #"Train(freight)": "TRAIN_FREIGHT",
    "Truck_fuel_cell(hydrogen)": "TRUCK_FUEL_CELL",
    "Truck_methanol": "TRUCK_METHANOL",
    "Trucks_diesel": "TRUCK_DIESEL",
    
}

# === Fonction pour lire la capacité installée dans assets.csv ===
def lire_capacite_depuis_assets(chemin_assets, techno_reelle, colonne_techno):
    if chemin_assets.endswith(".txt"):
        chemin_csv = chemin_assets.replace('.txt', '.csv')
        if not os.path.exists(chemin_csv):
            try:
                df = pd.read_csv(chemin_assets, sep=None, engine='python', skiprows=[1])
                df.columns = df.columns.str.strip()
                df.to_csv(chemin_csv, index=False)
                print(f"✅ Converti : {chemin_csv}")
            except Exception as e:
                print(f"❌ Erreur de conversion {chemin_assets} → {e}")
                return None
        else:
            print(f"🔄 CSV déjà existant : {chemin_csv}")
    else:
        chemin_csv = chemin_assets

    try:
        df = pd.read_csv(chemin_csv)
        df.columns = df.columns.str.strip().str.lower()
    except Exception as e:
        print(f"[❌] Erreur lecture {chemin_csv} : {e}")
        return None

    col = colonne_techno.lower()
    if col not in df.columns or 'f' not in df.columns:
        print(f"[❌] Colonnes '{col}' ou 'f' manquantes dans {chemin_csv}")
        return None

    ligne = df[df[col] == techno_reelle]
    if ligne.empty:
        print(f"[⚠️] Ligne '{techno_reelle}' non trouvée dans {chemin_csv}")
        return None

    return ligne['f'].values[0]

# === Dossier racine contenant les technos ===
chemin_base = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/Rapports_150%"

# === Dictionnaire pour stocker les points de chaque techno
courbes = {}  # { "Car electric": [(f1, gwp1), (f2, gwp2), ...], ... }

for label_affiche, nom_techno_reel in technos_a_analyser.items():
    chemin_techno = os.path.join(chemin_base, label_affiche)
    if not os.path.isdir(chemin_techno):
        print(f"[⚠️] Dossier {label_affiche} introuvable à {chemin_techno}")
        continue

    points = []
    for sous_dossier in os.listdir(chemin_techno):
        chemin_sous_dossier = os.path.join(chemin_techno, sous_dossier)
        if not os.path.isdir(chemin_sous_dossier):
            continue

        match = re.search(r'_(\d+(?:\.\d+)?)', sous_dossier)
        if not match:
            print(f"[⚠️] GWP non trouvé dans {sous_dossier}")
            continue
        gwp = float(match.group(1))

        chemin_assets = os.path.join(chemin_sous_dossier, "output/assets.txt")
        if not os.path.exists(chemin_assets):
            print(f"[⚠️] Fichier assets.txt manquant dans : {chemin_sous_dossier}")
            continue

        capacite = lire_capacite_depuis_assets(chemin_assets, nom_techno_reel, colonne_techno)
        if capacite is None:
            continue

        points.append((capacite, gwp))

    if points:
        points.sort(key=lambda p: p[1])  # GWP = deuxième élément du tuple
        courbes[label_affiche] = points


# === Tracé du graphique ===
plt.figure(figsize=(10, 6))

for label, points in courbes.items():
    x_vals, y_vals = zip(*points)
    plt.plot(y_vals, x_vals, linestyle='-', label=label)
    #plt.step(y_vals, x_vals, where='post', label=label, linewidth=2)

plt.xlabel("GWP de construction [ktCO2eq/Mpkm]")
plt.ylabel("Capacité installée [Mpkm]")
#plt.grid(True)
plt.legend()
plt.tight_layout()

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
chemin_image = os.path.join(dossier_images, "Secteur_mob_privé.png")
plt.savefig(chemin_image, dpi=300)

plt.show()



def tracer_sensibilite_transversale(techno_pivot_label, techno_pivot_nom, autres_technos_dict, chemin_base, colonne_techno):
    dossier_pivot = os.path.join(chemin_base, techno_pivot_label)
    if not os.path.isdir(dossier_pivot):
        print(f"[❌] Dossier pivot introuvable : {dossier_pivot}")
        return

    courbes = {label: [] for label in autres_technos_dict.keys()}
    gwp_vals = []

    for sous_dossier in os.listdir(dossier_pivot):
        chemin_sous = os.path.join(dossier_pivot, sous_dossier)
        if not os.path.isdir(chemin_sous):
            continue

        match = re.search(r'_(\d+(?:\.\d+)?)', sous_dossier)
        if not match:
            print(f"[⚠️] GWP non trouvé dans {sous_dossier}")
            continue
        gwp = float(match.group(1))
        gwp_vals.append(gwp)

        chemin_assets = os.path.join(chemin_sous, "output/assets.txt")
        if not os.path.exists(chemin_assets):
            print(f"[⚠️] Fichier assets.txt manquant : {chemin_assets}")
            continue

        for label, techno_nom in autres_technos_dict.items():
            cap = lire_capacite_depuis_assets(chemin_assets, techno_nom, colonne_techno)
            courbes[label].append(cap if cap is not None else 0.0)

    if not gwp_vals:
        print("[❌] Aucune valeur GWP collectée.")
        return

    # Tri par GWP croissant
    zipped = sorted(zip(gwp_vals, *courbes.values()))
    gwp_vals_sorted = [z[0] for z in zipped]
    autres_capacites = [list(z[1:]) for z in zipped]  # une liste par techno (transposé)

    plt.figure(figsize=(10, 6))
    for idx, (label, _) in enumerate(autres_technos_dict.items()):
        y_vals = [val[idx] for val in autres_capacites]
        plt.plot(gwp_vals_sorted, y_vals, marker='o', linestyle='-', label=label)

    plt.xlabel(f"GWP de construction de {techno_pivot_label} [ktCO2eq/Mtkm]")
    plt.ylabel("Capacité installée [Mtkm]")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    chemin_image = os.path.join(dossier_images, "Comparaison_truck_fuel_cell.png")
    plt.savefig(chemin_image, dpi=300)
    plt.show()


# Exemple d'utilisation
techno_pivot_label = "Truck_fuel_cell(hydrogen)"
techno_pivot_nom = "TRUCK_FUEL_CELL"

# On veut voir l'impact de la variation du GWP de CAR_BEV sur d'autres voitures
autres_technos = {
    "boat_diesel": "BOAT_FREIGHT_DIESEL",
    "boat_methanol": "BOAT_FREIGHT_METHANOL",
    #"Train(freight)": "TRAIN_FREIGHT",
    "Truck_fuel_cell(hydrogen)": "TRUCK_FUEL_CELL",
    "Truck_methanol": "TRUCK_METHANOL",
    "Trucks_diesel": "TRUCK_DIESEL",
    
}


tracer_sensibilite_transversale(techno_pivot_label, techno_pivot_nom, autres_technos, chemin_base, colonne_techno)
