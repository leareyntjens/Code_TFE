# -*- coding: utf-8 -*-
"""
Script de comparaison des variations de technologies (assets.csv)
entre plusieurs scenarii EnergyScope, avec regroupement par grandes catégories technologiques.
@author: reynt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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

def get_techno_categories():
    categories = {
        "Production d'énergie": [
            "NUCLEAR", "CCGT", "CCGT_AMMONIA", "COAL_US", "COAL_IGCC",
            "PV", "WIND_ONSHORE", "WIND_OFFSHORE", "HYDRO_RIVER", "GEOTHERMAL",
            "IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_DIRECT_ELEC",
            "IND_BOILER_GAS", "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE",
            "DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE", "DHN_COGEN_WET_BIOMASS",
            "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS", "DHN_BOILER_WOOD", "DHN_BOILER_OIL",
            "DHN_DEEP_GEO", "DHN_SOLAR", "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL",
            "DEC_ADVCOGEN_GAS", "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL",
            "DEC_SOLAR", "DEC_DIRECT_ELEC"
        ],
        "Transport": [
            "TRAMWAY_TROLLEY", "BUS_COACH_DIESEL", "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH",
            "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB", "CAR_GASOLINE", "CAR_DIESEL", "CAR_NG",
            "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL", "TRAIN_FREIGHT",
            "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG", "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL",
            "TRUCK_METHANOL", "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"
        ],
        "Conversion & synthèse": [
            "HABER_BOSCH", "SYN_METHANOLATION", "METHANE_TO_METHANOL", "BIOMASS_TO_METHANOL",
            "OIL_TO_HVC", "GAS_TO_HVC", "BIOMASS_TO_HVC", "METHANOL_TO_HVC", "H2_ELECTROLYSIS",
            "SMR", "H2_BIOMASS", "GASIFICATION_SNG", "SYN_METHANATION", "BIOMETHANATION", "BIO_HYDROLYSIS"
        ],
        "Stockage d'énergie": [
            "PHS", "BATT_LI", "BEV_BATT", "PHEV_BATT", "TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC",
            "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS",
            "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL",
            "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP", "GAS_STORAGE", "H2_STORAGE",
            "DIESEL_STORAGE", "GASOLINE_STORAGE", "LFO_STORAGE", "AMMONIA_STORAGE",
            "METHANOL_STORAGE", "CO2_STORAGE"
        ],
        "Système & réseau": [
            "EFFICIENCY", "DHN", "GRID", "ATM_CCS", "INDUSTRY_CCS", "AMMONIA_TO_H2"
        ]
    }
    return categories

def tracer_variation_par_categorie(base_path, run_names, file_name_csv, col_label, x_labels_custom):
    dfs = []
    for run_name in run_names:
        csv_path = os.path.join(base_path, run_name, 'output', file_name_csv)
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            dfs.append(df)
        except Exception as e:
            print(f"[❌] Problème avec {csv_path} : {e}")
            return

    ref_df = dfs[0]
    categories = get_techno_categories()
    techno_to_cat = {tech: cat for cat, techs in categories.items() for tech in techs}
    data_by_cat = defaultdict(list)

    for tech in ref_df.iloc[:, 0]:
        cat = techno_to_cat.get(tech, None)
        if not cat:
            continue

        ref_val = ref_df[ref_df.iloc[:, 0] == tech][col_label].values
        ref_val = ref_val[0] if len(ref_val) > 0 else 0.0
        techno_vals_all = []
        for df in dfs:
            val = df[df.iloc[:, 0] == tech][col_label].values
            val = val[0] if len(val) > 0 else 0.0
            techno_vals_all.append(val)

        if all(v == 0 for v in techno_vals_all) or all(v < 0.001 for v in techno_vals_all):
            continue

        variations = []
        annotations = []
        for i, val in enumerate(techno_vals_all):
            if ref_val == 0 and val == 0:
                variation = 0
                annotation = "0%"
            elif ref_val == 0 and val != 0:
                variation = 200
                annotation = ">+200%"
            else:
                variation_calc = ((val - ref_val) / ref_val) * 100
                if variation_calc > 200:
                    variation = 200
                    annotation = ">+200%"
                elif variation_calc < -200:
                    variation = -200
                    annotation = "<-200%"
                else:
                    variation = variation_calc
                    annotation = f"{variation:.0f}%"
            variations.append(variation)
            annotations.append(annotation)

        data_by_cat[cat].append((tech, variations, annotations))

    for cat, entries in data_by_cat.items():
        labels = [e[0] for e in entries]
        all_var = [e[1] for e in entries]
        all_ann = [e[2] for e in entries]
        x = np.arange(len(labels))
        bar_width = 0.12
        n_scenarios = len(run_names)

        plt.figure(figsize=(max(16, len(labels) * 0.35), 6))
        for i in range(1, n_scenarios):
            variations = [v[i] for v in all_var]
            annotations = [a[i] for a in all_ann]
            colors = ['#2ca02cAA' if v >= 0 else '#d62728AA' for v in variations]
            bars = plt.bar(x + (i - 1) * bar_width, variations, width=bar_width,
                           label=x_labels_custom[i], color=colors, edgecolor='black', linewidth=0.5)
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if abs(height) > 5:
                    plt.text(bar.get_x() + bar.get_width()/2, height + (3 if height > 0 else -5),
                             annotations[j], ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

        plt.xticks(x + bar_width * (n_scenarios - 1) / 2, labels, rotation=45, ha="right")
        plt.axhline(0, color='black', linewidth=1)
        plt.title(f"Variation (%) - {cat}")
        plt.ylim(-210, 210)
        plt.legend(title="Scénarios vs Réf.")
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.show()

# === PARAMÈTRES ===
base_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/scénarios_finaux"
run_names = [
    "10MGT", "COST_TS"
]
x_labels_custom = [
    "Ref.", "Opti cout", "Opti GWP", "150% Ressoruces"
]
txt_file_name = "assets.txt"
csv_file_name = "assets.csv"
col_label = "f"

# === EXÉCUTION ===
convert_to_csv(base_path, run_names, txt_file_name)
tracer_variation_par_categorie(base_path, run_names, csv_file_name, col_label, x_labels_custom)

#
