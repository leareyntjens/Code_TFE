import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# === Dossier racine contenant les scénarios ===
dossier_base = 'C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/Limite de prix'

# === Dictionnaire pour stocker les données par scénario ===
donnees_par_scenario = {}  # { "Scénario_1": [(prix, somme_gwp), ...], ... }

# === Parcours des dossiers de scénarios ===
for nom_scenario in os.listdir(dossier_base):
    chemin_scenario = os.path.join(dossier_base, nom_scenario)
    if not os.path.isdir(chemin_scenario):
        continue

    donnees_graph = []

    # Parcours des sous-dossiers "Limite_X"
    for nom_dossier_limite in os.listdir(chemin_scenario):
        chemin_limite = os.path.join(chemin_scenario, nom_dossier_limite)
        if not os.path.isdir(chemin_limite):
            continue

        # Extraction de la limite de prix
        match = re.search(r'(\d+(?:\.\d+)?)', nom_dossier_limite)
        if not match:
            print(f"[⚠️] Pas de limite de prix détectée dans : {nom_dossier_limite}")
            continue
        limite_prix = float(match.group(1))

        # Lecture du fichier gwp_breakdown.txt
        chemin_txt = os.path.join(chemin_limite, "output", "gwp_breakdown.txt")
        chemin_csv = chemin_txt.replace(".txt", ".csv")

        if not os.path.exists(chemin_txt):
            print(f"[❌] Fichier manquant : {chemin_txt}")
            continue

        # Conversion .txt → .csv
        try:
            df = pd.read_csv(chemin_txt, sep=None, engine='python', skiprows=[1])
            df.columns = df.columns.str.strip()
            df.to_csv(chemin_csv, index=False)
            print(f"✅ Converti : {chemin_csv}")
        except Exception as e:
            print(f"[❌] Erreur conversion {chemin_txt} : {e}")
            continue

        # Lecture et somme des deux dernières colonnes
        try:
            df = pd.read_csv(chemin_csv)
            if df.shape[1] < 2:
                print(f"[⚠️] Pas assez de colonnes dans : {chemin_csv}")
                continue
            somme_gwp = df.iloc[:, -2:].sum().sum()
            donnees_graph.append((limite_prix, somme_gwp))
        except Exception as e:
            print(f"[❌] Erreur traitement {chemin_csv} : {e}")
            continue

    if donnees_graph:
        donnees_graph.sort()  # tri par limite de prix croissante
        donnees_par_scenario[nom_scenario] = donnees_graph

# === Tracé du graphique combiné ===
if donnees_par_scenario:
    plt.figure(figsize=(10, 6))

    for nom_scenario, donnees in donnees_par_scenario.items():
        x_vals, y_vals = zip(*donnees)
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=nom_scenario)
    
    
    plt.axvline(x=60500, color='red', linestyle='--', label='Limite de prix')
    plt.xlabel("Limite de prix [M€]")
    plt.ylabel("GWP total [ktCO2eq/an]")
    #plt.title("Impact de la limite de prix sur le GWP total par scénario")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Sauvegarde
    dossier_images = "C:/Users/reynt/OneDrive - UCL/MASTER/Mémoire/Résultats/12 mai"
    os.makedirs(dossier_images, exist_ok=True)
    chemin_image = os.path.join(dossier_images, "gwp_vs_limite_prix_par_scenario.png")
    plt.savefig(chemin_image, dpi=300)
    print(f"📈 Graphique sauvegardé : {chemin_image}")
    plt.show()
else:
    print("[❌] Aucune donnée utilisable pour tracer un graphique.")
