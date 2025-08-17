import os
import pandas as pd
import matplotlib.pyplot as plt

# === Paramètres à adapter ===
base_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies"

# === Catégories technologiques ===
importation_gaz = {'GAS_RE', 'H2_RE', 'AMMONIA_RE', 'METHANOL_RE', 'GAS', 'BIODIESEL'}
production = {
    'NUCLEAR','PV', 'CCGT', 'CCGT_AMMONIA', 'COAL_US', 'COAL_IGCC', 'WIND_ONSHORE', 'WIND_OFFSHORE', 'HYDRO_RIVER', 'GEOTHERMAL',
    'IND_COGEN_GAS', 'IND_COGEN_WOOD', 'IND_COGEN_WASTE', 'IND_BOILER_GAS', 'IND_BOILER_WOOD', 'IND_BOILER_OIL', 'IND_BOILER_COAL',
    'IND_BOILER_WASTE', 'IND_DIRECT_ELEC', 'DHN_HP_ELEC', 'DHN_COGEN_GAS', 'DHN_COGEN_WOOD', 'DHN_COGEN_WASTE', 'DHN_COGEN_WET_BIOMASS',
    'DHN_COGEN_BIO_HYDROLYSIS', 'DHN_BOILER_GAS', 'DHN_BOILER_WOOD', 'DHN_BOILER_OIL', 'DHN_DEEP_GEO', 'DHN_SOLAR', 'DEC_HP_ELEC',
    'DEC_THHP_GAS', 'DEC_COGEN_GAS', 'DEC_COGEN_OIL', 'DEC_ADVCOGEN_GAS', 'DEC_ADVCOGEN_H2', 'DEC_BOILER_GAS', 'DEC_BOILER_WOOD',
    'DEC_BOILER_OIL', 'DEC_SOLAR', 'DEC_DIRECT_ELEC', 'H2_ELECTROLYSIS', 'SMR', 'H2_BIOMASS', 'GASIFICATION_SNG', 'SYN_METHANATION',
    'BIOMETHANATION', 'BIO_HYDROLYSIS', 'PYROLYSIS_TO_LFO', 'PYROLYSIS_TO_FUELS', 'AMMONIA_TO_H2'
}
stockage = {
    'PHS', 'BATT_LI', 'BEV_BATT', 'PHEV_BATT', 'TS_DEC_DIRECT_ELEC', 'TS_DEC_HP_ELEC', 'TS_DEC_THHP_GAS', 'TS_DEC_COGEN_GAS',
    'TS_DEC_COGEN_OIL', 'TS_DEC_ADVCOGEN_GAS', 'TS_DEC_ADVCOGEN_H2', 'TS_DEC_BOILER_GAS', 'TS_DEC_BOILER_WOOD', 'TS_DEC_BOILER_OIL',
    'TS_DHN_DAILY', 'TS_DHN_SEASONAL', 'TS_HIGH_TEMP', 'GAS_STORAGE', 'H2_STORAGE', 'DIESEL_STORAGE', 'GASOLINE_STORAGE',
    'LFO_STORAGE', 'AMMONIA_STORAGE', 'METHANOL_STORAGE', 'CO2_STORAGE'
}

# === Scénarios à traiter ===
scenarios = ['Scénarios de base/Aucune_mod_gwp_constr', 'Scénarios de base/Mod_techno_gwp_constr']

# === Traitement ===
results = []

for scenario in scenarios:
    folder_path = os.path.join(base_path, scenario)
    txt_path = os.path.join(folder_path, "output", "gwp_breakdown.txt")

    if not os.path.isfile(txt_path):
        print(f"Fichier manquant : {txt_path}")
        continue

    try:
        df = pd.read_csv(txt_path, sep="\t", header=None, names=["Name", "gwp_constr", "gwp_op"])
        df["Name"] = df["Name"].str.strip()
        df["gwp_constr"] = pd.to_numeric(df["gwp_constr"], errors="coerce")
        df["gwp_op"] = pd.to_numeric(df["gwp_op"], errors="coerce")

        # Sommes par catégorie
        def sum_cat(cat, col): return df[df["Name"].isin(cat)][col].sum()

        prod_tot = sum_cat(production, "gwp_constr") + sum_cat(production, "gwp_op")
        imp_tot = sum_cat(importation_gaz, "gwp_constr") + sum_cat(importation_gaz, "gwp_op")
        sto_tot = sum_cat(stockage, "gwp_constr") + sum_cat(stockage, "gwp_op")
        total = df["gwp_constr"].sum() + df["gwp_op"].sum()

        constr_total = df["gwp_constr"].sum()
        op_total = df["gwp_op"].sum()

        print(f"\nScénario : {scenario}")
        print(f"GWP total     : {total:.2f}")
        print(f"GWP constr    : {constr_total:.2f}")
        print(f"GWP opération : {op_total:.2f}")

        results.append({
            "scenario": scenario.split("/")[-1],
            "Production (%)": prod_tot / total * 100 if total > 0 else 0,
            "Importation (%)": imp_tot / total * 100 if total > 0 else 0,
            "Stockage (%)": sto_tot / total * 100 if total > 0 else 0,
            "Total GWP": total,
            "GWP Constr": constr_total,
            "GWP Op": op_total
        })

    except Exception as e:
        print(f"Erreur dans {scenario} : {e}")

# === Affichage du graphe ===
df_res = pd.DataFrame(results)
x = range(len(df_res))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar([i - width for i in x], df_res["Production (%)"], width=width, label="Production")
plt.bar(x, df_res["Importation (%)"], width=width, label="Importation")
plt.bar([i + width for i in x], df_res["Stockage (%)"], width=width, label="Stockage")

plt.xticks(ticks=x, labels=df_res["scenario"], rotation=45, ha="right")
plt.ylabel("Part du GWP total (%)")
plt.title("Répartition du GWP (construction + opération) par catégorie")
plt.legend()
plt.tight_layout()
plt.show()
