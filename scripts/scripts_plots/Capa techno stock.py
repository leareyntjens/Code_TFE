import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Étape 1 : Fonction de conversion TXT ➜ CSV ===
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
            print(f"✅ Fichier converti en CSV : {csv_path}")
        except Exception as e:
            print(f"❌ Erreur lors de la conversion de {txt_path}: {e}")

# === Étape 2 : Tracé des histogrammes ===
def tracer_histogramme_sep_gas(base_path, run_names, file_name_csv, col_label, row_start, row_end, x_labels_custom):
    # Charger tous les fichiers CSV complets
    dfs = []
    for run_name in run_names:
        csv_path = os.path.join(base_path, run_name, 'output', file_name_csv)
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as e:
            print(f"[❌] Problème avec {csv_path} : {e}")
            dfs.append(pd.DataFrame())  # Placeholder vide

    # === Catégories de technologies ===
    techno_stockage = [
        "BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS",
        "TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL",
        "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD",
        "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP",
        "GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE", "LFO_STORAGE",
        "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"
    ]

    techno_centrale = [
        "NUCLEAR", "CCGT", "CCGT_AMMONIA", "COAL_US", "COAL_IGCC",
        "PV", "WIND_ONSHORE", "WIND_OFFSHORE", "HYDRO_RIVER", "GEOTHERMAL", "H2_ELECTROLYSIS"
    ]
    techno_industrielle = [
        "IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_DIRECT_ELEC",
        "IND_BOILER_GAS", "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE"
    ]
    techno_decentralisee = [
        "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL", "DEC_ADVCOGEN_GAS",
        "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL",
        "DEC_SOLAR", "DEC_DIRECT_ELEC"
    ]
    techno_dhn = [
        "DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE",
        "DHN_COGEN_WET_BIOMASS", "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS",
        "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO", "DHN_SOLAR"
    ]

    # Fonction générique pour tracer des barres (horizontales ou non)
    def tracer_barres_par_nom(tech_list, titre, horizontal=False, remove_zeros=True):
        labels_present = []
        valeurs_par_scenario = []

        for df in dfs:
            valeurs = []
            for label in tech_list:
                matching = df[df.iloc[:, 0] == label]
                if not matching.empty:
                    valeur = matching[col_label].values[0]
                else:
                    valeur = 0.0
                valeurs.append(valeur)
            valeurs_par_scenario.append(valeurs)

        # Transposition : [techno][scénario]
        valeurs_transposees = list(map(list, zip(*valeurs_par_scenario)))
        filtered_labels = []
        filtered_data = []

        for i, (label, valeurs) in enumerate(zip(tech_list, valeurs_transposees)):
            if remove_zeros and all(v < 1 for v in valeurs):
                continue
            filtered_labels.append(label)
            filtered_data.append(valeurs)

        if not filtered_labels:
            print(f"[ℹ️] Aucun élément à tracer pour : {titre}")
            return

        x = np.arange(len(filtered_labels))
        bar_width = 0.12

        # Totaux
        print(f"\n📊 Totaux pour : {titre}")
        for i, run_name in enumerate(run_names):
            total = sum(filtered_data[j][i] for j in range(len(filtered_labels)))
            print(f"  {run_name:<20} ➜  {total:.2f}")

        # Tracé
        plt.figure(figsize=(10, 6))
        for i, run_label in enumerate(x_labels_custom):
            scenario_values = [filtered_data[j][i] for j in range(len(filtered_labels))]
            if horizontal:
                plt.barh(x + i * bar_width, scenario_values, height=bar_width, label=run_label)
            else:
                plt.bar(x + i * bar_width, scenario_values, width=bar_width, label=run_label)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if horizontal:
            plt.yticks(x + bar_width * (len(run_names) / 2), filtered_labels)
            plt.xlabel('Capacité installée [GWh]')
            plt.ylabel("")
        else:
            plt.xticks(x + bar_width * (len(run_names) / 2), filtered_labels, rotation=45, ha="right")
            plt.ylabel('Capacités installées [GWh]')
            plt.xlabel("")

        plt.title(titre)
        #plt.legend(title="Scénarios")
        plt.tight_layout()
        plt.show()

    # === Stockage ===
    stockage_hors_specifique = [t for t in techno_stockage if t not in ["GAS_STORAGE", "TS_DHN_SEASONAL","BEV_BATT"]]
    tracer_barres_par_nom(["GAS_STORAGE"], "GAS_STORAGE", horizontal=True)
    tracer_barres_par_nom(["TS_DHN_SEASONAL"], "TS_DHN_SEASONAL", horizontal=True)
    #tracer_barres_par_nom(["BEV_BATT"], "BEV_BATT", horizontal=True)
    # Diviser selon un seuil de 20 GWh dans tous les scénarios
    valeurs_moyennes = {}
    for df in dfs:
        for tech in stockage_hors_specifique:
            val = df[df.iloc[:, 0] == tech]
            if not val.empty:
                valeurs_moyennes.setdefault(tech, []).append(val[col_label].values[0])
            else:
                valeurs_moyennes.setdefault(tech, []).append(0)
    
    # Moyenne sur les scénarios
    moyennes = {k: np.mean(v) for k, v in valeurs_moyennes.items()}
    
    # Séparer les technos
    seuil = 20
    techs_basses = [k for k, v in moyennes.items() if v < seuil]
    techs_hautes = [k for k, v in moyennes.items() if v >= seuil]
    
    # Tracer séparément
    tracer_barres_par_nom(techs_basses, "Autres technologies de stockage (< 20 GWh)")
    tracer_barres_par_nom(techs_hautes, "Autres technologies de stockage (≥ 20 GWh)")


    # === Autres catégories ===
    tracer_barres_par_nom(techno_centrale, "Production centralisée")
    tracer_barres_par_nom(techno_industrielle, "Production industrielle")
    #tracer_barres_par_nom(techno_decentralisee, "Production décentralisée")
    #tracer_barres_par_nom(techno_dhn, "Réseau de chaleur (DHN)")


# === Paramètres ===
base_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/scénarios_finaux"
run_names = [
    "10MGT", 
    "Gwp_constr_TS"
]
txt_file_name = "assets.txt"
csv_file_name = "assets.csv"
col_label = "f"  # ⚠️ adapte au nom réel de la colonne
row_start = 72
row_end = 96
x_labels_custom = [
    "Sans f_min",
    "Opti -30%"
]

# === Lancement ===
convert_to_csv(base_path, run_names, txt_file_name)
tracer_histogramme_sep_gas(base_path, run_names, csv_file_name, col_label, row_start, row_end, x_labels_custom)
