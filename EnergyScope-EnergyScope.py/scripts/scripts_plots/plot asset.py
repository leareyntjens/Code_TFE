"""
Script d'analyse EnergyScope – Technologies centrales, stockage, mobilité, synthétiques
Générique et adaptable pour toute liste de technologies et scénarios
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Listes de technologies ===
techno_centrale = [
    "NUCLEAR", "CCGT", "CCGT_AMMONIA", "COAL_US", "COAL_IGCC",
    "PV", "WIND_ONSHORE", "WIND_OFFSHORE", "HYDRO_RIVER", "GEOTHERMAL", "H2_ELECTROLYSIS"]

IND =   ["IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_DIRECT_ELEC",
    "IND_BOILER_GAS", "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE"]
    
DEC=    ["DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL", "DEC_ADVCOGEN_GAS",
    "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL",
    "DEC_SOLAR", "DEC_DIRECT_ELEC"]
    
DHN=    ["DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE",
    "DHN_COGEN_WET_BIOMASS", "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS",
    "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO", "DHN_SOLAR"
]

techno_stockage = [
    "BATT_LI", "PHEV_BATT", "PHS","GAS_STORAGE",
    "TS_DEC_DIRECT_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL",
    "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD",
    "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE",
    "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE", "TS_DHN_SEASONAL",
    "TS_HIGH_TEMP", "TS_DEC_HP_ELEC", "BEV_BATT", "CO2_STORAGE"
] 

techno_mob_pub = [
    "TRAMWAY_TROLLEY", "BUS_COACH_DIESEL", "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH",
    "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"
]

techno_mob_priv = [
    "CAR_GASOLINE", "CAR_DIESEL", "CAR_NG",
    "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"
]

techno_mob_freight = [
    "TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG", "BOAT_FREIGHT_METHANOL",
    "TRUCK_DIESEL", "TRUCK_METHANOL", "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"
]

techno_Synthetic_fuels = [
    "HABER_BOSCH", "SYN_METHANOLATION", "METHANE_TO_METHANOL", "BIOMASS_TO_METHANOL",
    "OIL_TO_HVC", "GAS_TO_HVC", "BIOMASS_TO_HVC", "METHANOL_TO_HVC", 
      "SMR", "H2_BIOMASS", "GASIFICATION_SNG", "SYN_METHANATION",
    "BIOMETHANATION", "BIO_HYDROLYSIS", "PYROLYSIS_TO_LFO", "PYROLYSIS_TO_FUELS",
    "ATM_CCS", "INDUSTRY_CCS", "AMMONIA_TO_H2"
]
#GRID, "EFFICIENCY",
# === Fonctions ===
def convert_to_csv(base_path, run_names, file_name):
    for run_name in run_names:
        txt_path = os.path.join(base_path, run_name, 'output', file_name)
        csv_path = txt_path.replace('.txt', '.csv')
        if not os.path.exists(txt_path):
            print(f"Fichier non trouvé : {txt_path}")
            continue
        try:
            df = pd.read_csv(txt_path, sep=None, engine='python', skiprows=[1])
            df.columns = df.columns.str.strip()
            df.to_csv(csv_path, index=False)
            print(f"✅ Fichier converti : {csv_path}")
        except Exception as e:
            print(f"❌ Erreur conversion {txt_path}: {e}")

def tracer_histogramme_technos(base_path, run_names, file_name_csv, col_label, x_labels_custom, liste_technos, titre):
    if isinstance(liste_technos, tuple):
        liste_technos = liste_technos[0]

    dfs = []
    for run_name in run_names:
        path = os.path.join(base_path, run_name, 'output', file_name_csv)
        try:
            dfs.append(pd.read_csv(path))
        except Exception as e:
            print(f"[❌] Problème avec {path} : {e}")
            dfs.append(pd.DataFrame())

    valeurs_par_techno, techno_labels = [], []
    for tech in liste_technos:
        valeurs = []
        for df in dfs:
            if col_label not in df.columns:
                valeurs.append(0.0)
            else:
                match = df[df.iloc[:, 0] == tech]
                valeurs.append(match[col_label].values[0] if not match.empty else 0.0)
        if any(v > 0 for v in valeurs):
            valeurs_par_techno.append(valeurs)
            techno_labels.append(tech)

    if not techno_labels:
        print("[ℹ️] Aucune technologie significative à tracer.")
        return

    x = np.arange(len(techno_labels))
    bar_width = 0.12
    plt.figure(figsize=(14, 6))
    for i, label in enumerate(x_labels_custom):
        valeurs = [valeurs_par_techno[j][i] for j in range(len(techno_labels))]
        plt.bar(x + i * bar_width, valeurs, width=bar_width, label=label)

    plt.xticks(x + bar_width * len(run_names) / 2 - bar_width / 2, techno_labels, rotation=45, ha="right")
    plt.ylabel("Capacités installées [GWh]")
    plt.title(titre)
    plt.legend(title="Scénarios")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def tracer_variation_pourcentages(base_path, run_names, file_name_csv, col_label, liste_technos):
    if isinstance(liste_technos, tuple):
        liste_technos = liste_technos[0]

    if len(run_names) != 2:
        print("⚠️ Cette fonction nécessite exactement deux scénarios.")
        return

    dfs = []
    for run_name in run_names:
        path = os.path.join(base_path, run_name, 'output', file_name_csv)
        try:
            dfs.append(pd.read_csv(path))
        except Exception as e:
            print(f"[❌] Problème avec {path} : {e}")
            return

    df1, df2 = dfs
    variations, techno_labels, annotations = [], [], []

    for tech in liste_technos:
        val1 = df1[df1.iloc[:, 0] == tech][col_label].values if col_label in df1.columns else []
        val2 = df2[df2.iloc[:, 0] == tech][col_label].values if col_label in df2.columns else []
        v1 = val1[0] if len(val1) > 0 else 0.0
        v2 = val2[0] if len(val2) > 0 else 0.0

        if v1 <= 1e-4 and v2 <= 1e-4:
            continue  # 👉 Ignore les technos trop faibles
        if v1 == 0 and v2 == 0:
            continue
      # elif v1 == 0:
       #     variation = 200
        #    annotation = ">+200%"
        else:
            #variation_calc = ((v2 - v1) / v1) * 100
            variation_calc = v2 - v1
            if abs(variation_calc) < 0.1:
                continue
            #variation = max(min(variation_calc, 200), -200)
            variation = variation_calc
            #annotation =  f">+200%" if variation_calc > 200 else (f"<-200%" if variation_calc < -200 else f"{variation:.0f}%")
            annotation =  f"{variation:.1f}"

        variations.append(variation)
        techno_labels.append(tech)
        annotations.append(annotation)

    if not techno_labels:
        print("[ℹ️] Aucune variation significative à afficher.")
        return

    x = np.arange(len(techno_labels))
    plt.figure(figsize=(14, 6))
    colors = ['#2ca02cAA' if v >= 0 else '#d62728AA' for v in variations]
    bars = plt.bar(x, variations, color=colors)
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(x, techno_labels, rotation=45, ha='right', fontsize=14)
    #plt.xticks(x, techno_labels, fontsize=14)

    plt.yticks(fontsize = 14)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        y_offset = 3 if height > 0 else -5
        va = 'bottom' if height > 0 else 'top'
        plt.text(bar.get_x() + bar.get_width()/2, height + y_offset, annotations[i], ha='center', va=va, fontsize=14)
    plt.ylim(-210, 210)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.ylabel('Installed capacity variation [%]', fontsize = 16)
    plt.ylabel('Installed capacity variation [GWh]', fontsize = 16)
    
    plt.tight_layout()
    plt.savefig("C:\\Users\\reynt\\OneDrive - UCL\\MASTER\\Mémoire\\Overleaf\\Section 4.2\\Impact_stock_cout.svg", format='svg')
    plt.show()


    
def afficher_top3_par_scenario(base_path, run_names, file_name_csv, col_label, liste_technos):
    if isinstance(liste_technos, tuple):
        liste_technos = liste_technos[0]

    print("\n🔝 Top 3 des technologies ciblées par scénario :\n")
    for run_name in run_names:
        path = os.path.join(base_path, run_name, 'output', file_name_csv)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[❌] Erreur lecture {path} : {e}")
            continue

        if col_label not in df.columns:
            print(f"[❌] Colonne '{col_label}' absente dans {path}")
            continue

        techno_values = []
        for tech in liste_technos:
            match = df[df.iloc[:, 0] == tech]
            value = match[col_label].values[0] if not match.empty else 0.0
            techno_values.append((tech, value))

        # Trier par valeur décroissante
        techno_values_sorted = sorted(techno_values, key=lambda x: x[1], reverse=True)

        print(f"▶️ Scénario : {run_name}")
        for i, (tech, val) in enumerate(techno_values_sorted[:3], start=1):
            print(f"   {i}. {tech} — {val:.2f} GWh")
        print("-" * 40)

def top3_sto_year_global(base_path, run_names, seuil_negligeable=6.79):
    print("\n🔎 Top 3 global des technologies de stockage par scénario (toutes colonnes confondues) :\n")

    for run_name in run_names:
        txt_path = os.path.join(base_path, run_name, 'output', 'sto_year.txt')
        csv_path = txt_path.replace('.txt', '.csv')

        # Convertir si nécessaire
        if not os.path.exists(csv_path):
            try:
                df = pd.read_csv(txt_path, sep=None, engine='python', skiprows=[1])
                df.columns = df.columns.str.strip()
                df.to_csv(csv_path, index=False)
                print(f"✅ Fichier converti : {csv_path}")
            except Exception as e:
                print(f"[❌] Erreur conversion {txt_path}: {e}")
                continue

        # Lire le fichier CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[❌] Erreur lecture {csv_path}: {e}")
            continue

        if df.empty:
            print(f"[ℹ️] Fichier vide pour {run_name}")
            continue

        # Préparer les valeurs : (TECHNO, VECTEUR, VALEUR)
        valeurs = []
        techno_col = df.columns[0]
        for _, row in df.iterrows():
            techno = row[techno_col]
            for vecteur in df.columns[1:]:
                try:
                    val = float(row[vecteur])
                    if val > seuil_negligeable:
                        valeurs.append((techno, vecteur, val))
                except:
                    continue

        if not valeurs:
            print(f"[ℹ️] Aucune valeur significative (> {seuil_negligeable}) pour {run_name}")
            continue

        # Trier et afficher top 3
        top3 = sorted(valeurs, key=lambda x: x[2], reverse=True)[:3]
        print(f"▶️ Scénario : {run_name}")
        for i, (tech, vect, val) in enumerate(top3, start=1):
            print(f"   {i}. {tech} – {vect} : {val:.2f}")
        print("-" * 40)





# === Section personnalisable ===
if __name__ == "__main__":
    base_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies"
    #, 'Scénarios de base/Mod_techno_gwp_constr'
    scenarios =['Scénarios de base/10MGT', 'Scénarios de base/GWP_tot_10MGT_corr']#]#Scenario_ref_60500', 'Limite de prix/Prix 0 fossil/Lim_prix_60500_0_fossil']

    labels = ["GWP10MGT", "10MGT"]
    techno_cible = techno_Synthetic_fuels+ techno_mob_pub + techno_mob_freight #["BATT_LI", "H2_STORAGE", "TS_DHN_SEASONAL", "TS_HIGH_TEMP", "TS_DEC_HP_ELEC", "BEV_BATT", "CAR_BEV", "CAR_FUEL_CELL"]#"GAS_STORAGE"# + techno_mob_priv + techno_mob_pub + techno_mob_freight #techno_stockage ## 👈 Choisis ici la liste à visualiser
    fichier_txt = "assets.txt"
    fichier_csv = "assets.csv"
    colonne = "f"

    convert_to_csv(base_path, scenarios, fichier_txt)
    tracer_histogramme_technos(base_path, scenarios, fichier_csv, colonne, labels, techno_cible, "Histogramme des capacités installées")
    tracer_variation_pourcentages(base_path, scenarios, fichier_csv, colonne, techno_cible)
    afficher_top3_par_scenario(base_path, scenarios, fichier_csv, colonne, techno_centrale)
    afficher_top3_par_scenario(base_path, scenarios, fichier_csv, colonne, IND)
    afficher_top3_par_scenario(base_path, scenarios, fichier_csv, colonne, DEC)
    afficher_top3_par_scenario(base_path, scenarios, fichier_csv, colonne, DHN)
    afficher_top3_par_scenario(base_path, scenarios, fichier_csv, colonne, techno_stockage)
    afficher_top3_par_scenario(base_path, scenarios, fichier_csv, colonne, techno_mob_priv)
    afficher_top3_par_scenario(base_path, scenarios, fichier_csv, colonne, techno_mob_pub)
    afficher_top3_par_scenario(base_path, scenarios, fichier_csv, colonne, techno_mob_freight)
    afficher_top3_par_scenario(base_path, scenarios, fichier_csv, colonne, techno_Synthetic_fuels)
    top3_sto_year_global(base_path, scenarios)
    # Ajoute ici les groupes de technologies à inclure dans la corrélation
    groupes = [techno_centrale, IND, DEC, DHN, techno_stockage, techno_mob_priv, techno_mob_pub, techno_mob_freight, techno_Synthetic_fuels]

    # Graphe barplot Top 3 – Production élec
 


