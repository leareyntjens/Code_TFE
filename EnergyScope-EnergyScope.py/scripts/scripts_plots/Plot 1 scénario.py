import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
plt.rcParams.update({'font.size': 14})


# === PARAMETERS ===
ressources = [
        "ELECTRICITY", "GASOLINE", "DIESEL", "BIOETHANOL", "BIODIESEL", "LFO",
        "GAS", "GAS_RE", "WOOD", "WET_BIOMASS", "COAL", "URANIUM", "WASTE",
        "H2", "H2_RE", "AMMONIA", "METHANOL", "AMMONIA_RE", "METHANOL_RE",
        "ELEC_EXPORT", "CO2_EMISSIONS", "RES_WIND", "RES_SOLAR", "RES_HYDRO",
        "RES_GEO", "CO2_ATM", "CO2_INDUSTRY", "CO2_CAPTURED"]

techno_par_ressource = {
    "electricite": {
        "CAR_BEV", "TRAMWAY_TROLLEY", "TRAIN_PUB", "TRAIN_FREIGHT",
        "BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS",
        "NUCLEAR", "CCGT", "CCGT_AMMONIA", "COAL_US", "COAL_IGCC", "PV",
        "WIND_ONSHORE", "WIND_OFFSHORE", "HYDRO_RIVER", "GEOTHERMAL", "BUS_COACH_HYDIESEL","CAR_HEV"},
    "gaz": {
        "BOAT_FREIGHT_NG", "CAR_GASOLINE","BUS_COACH_CNG_STOICH", "TRUCK_FUEL_CELL", #"BOAT_FREIGHT_DIESEL",
        "GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE",
        "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"},
    "chaleur": {
        "TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS",
        "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2",
        "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL",
        "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP",
        "IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE",
        "IND_BOILER_GAS", "IND_BOILER_WOOD", "IND_BOILER_OIL",
        "IND_BOILER_COAL", "IND_BOILER_WASTE", "IND_DIRECT_ELEC"},
    "low_T": {
        "DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE",
        "DHN_COGEN_WET_BIOMASS", "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS",
        "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO", "DHN_SOLAR",
        "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL",
        "DEC_ADVCOGEN_GAS", "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS",
        "DEC_BOILER_WOOD", "DEC_BOILER_OIL", "DEC_SOLAR", "DEC_DIRECT_ELEC"},
    "diesel":{"BOAT_FREIGHT_DIESEL","BUS_COACH_DIESEL", "BOAT_FREIGHT_DIESEL", "TRUCK_DIESEL", "CAR_DIESEL"}}


technos_electricity = ["NUCLEAR", "CCGT", "CCGT_AMMONIA", "COAL_US", "COAL_IGCC",
                       "PV", "WIND_ONSHORE", "WIND_OFFSHORE", "HYDRO_RIVER", "GEOTHERMAL"]


technos_heat_high = [
     "IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_BOILER_GAS",
     "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE", "IND_DIRECT_ELEC"]


technos_heat_low_central = [
    "DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE",
     "DHN_COGEN_WET_BIOMASS", "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS",
     "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO", "DHN_SOLAR"]

technos_heat_low_decentral = [
     "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL", "DEC_ADVCOGEN_GAS",
     "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL", "DEC_SOLAR", "DEC_DIRECT_ELEC"]

technos_mobility_public = ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL",
                           "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"]

technos_mobility_private = ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG",
                            "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"]

technos_mobility_freight = ["TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG",
                            "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL", "TRUCK_METHANOL", "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"]

technos_mobility = technos_mobility_public  + technos_mobility_private + technos_mobility_freight

technos_storage_elec = ["BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS"]

technos_storage_thermal = ["TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS",
                           "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP"]

technos_storage_other = ["GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE",
                         "GASOLINE_STORAGE", "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"]


technos_storage = technos_storage_elec + technos_storage_thermal + technos_storage_other

technos_conversion = [
     "HABER_BOSCH", "SYN_METHANOLATION", "METHANE_TO_METHANOL", "BIOMASS_TO_METHANOL",
     "OIL_TO_HVC", "GAS_TO_HVC", "BIOMASS_TO_HVC", "METHANOL_TO_HVC",
     "AMMONIA_TO_H2", "PYROLYSIS_TO_LFO", "PYROLYSIS_TO_FUELS"]

groupes_technos = {
     "Electricity": technos_electricity,
     "High temp heat": technos_heat_high,
     "Low temp heat (central)": technos_heat_low_central,
     "Low temp heat (decentral)": technos_heat_low_decentral,
     "Mobility": technos_mobility,
     "Storage": technos_storage,
     "Conversion": technos_conversion}

palettes = {
    "electricite": ['forestgreen', 'olivedrab', 'darkseagreen', 'yellowgreen', 'lightgreen'],
    "gaz": ['steelblue', 'lightblue'],
    "chaleur": ['firebrick', 'crimson', 'palevioletred', 'pink'],
    "low_T": ['orange', 'peachpuff'],
    "diesel": ['blueviolet','purple', 'orchid', 'thistle']}

# === FUNCTIONS===

def convert_txt_en_csv(scenario_path, txt_file_name):
    txt_path = os.path.join(scenario_path, "output", txt_file_name)
    csv_path = txt_path.replace('.txt', '.csv')
    if not os.path.exists(txt_path):
        print(f" Fichier non trouvé : {txt_path}")
        return None
    try:
        df = pd.read_csv(txt_path, sep=None, engine='python', skiprows=[1])
        df.columns = df.columns.str.strip()
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        print(f" Erreur de conversion : {e}")
        return None


def colors_def(csv_path, groupes, col_label, techno_par_ressource, palettes):
    df = pd.read_csv(csv_path)
    df.set_index(df.columns[0], inplace=True)

    toutes_technos = set()
    for techs in groupes.values():
        toutes_technos.update(techs)

    techno_valeurs = [(tech, df.loc[tech, col_label])
                      for tech in toutes_technos if tech in df.index and df.loc[tech, col_label] > 0]
    techno_valeurs.sort(key=lambda x: x[1], reverse=True)

    couleurs = {}
    compteurs_palette = {ressource: 0 for ressource in palettes}

    for tech, _ in techno_valeurs:
        ressource_tech = None
        for ressource, tech_set in techno_par_ressource.items():
            if tech in tech_set:
                ressource_tech = ressource
                break
        if ressource_tech:
            palette = palettes[ressource_tech]
            i = compteurs_palette[ressource_tech]
            if i < len(palette):
                couleur = palette[i]
                couleurs[tech] = couleur
                compteurs_palette[ressource_tech] += 1
            else:
                couleurs[tech] = "#CCCCCC"
        else:
            couleurs[tech] = "#999999"

    return couleurs


def capacities(csv_path, groupes, col_label="f", seuil=15, seuil_texte=1, bar_width=0.4, couleurs_fixes=None):
    try:
        df = pd.read_csv(csv_path)
        df.set_index(df.columns[0], inplace=True)
    except Exception as e:
        print(f" Erreur lecture CSV : {e}")
        return

    noms_barres = list(groupes.keys())
    x = np.arange(len(noms_barres))
    fig_width = 2.5 if len(noms_barres) == 1 else 10
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bottom = np.zeros(len(noms_barres))

    labels_deja_affiches = set()
    for i, (nom_barre, technos) in enumerate(groupes.items()):
        valeurs = []
        techno_valides = []
        for techno in technos:
            if techno in df.index:
                val = df.loc[techno, col_label]
                if val > seuil:
                    valeurs.append(val)
                    techno_valides.append(techno)
        if not valeurs:
            continue
        sorted_pairs = sorted(zip(techno_valides, valeurs),
                              key=lambda x: x[1], reverse=True)
        techno_sorted, valeurs_sorted = zip(*sorted_pairs)
        couleurs = [couleurs_fixes.get(tech, "#CCCCCC")
                    for tech in techno_sorted]

        for tech, val, col in zip(techno_sorted, valeurs_sorted, couleurs):
            label = tech if tech not in labels_deja_affiches else "_nolegend_"
            bar = ax.bar(x[i], val, width=bar_width,
                         bottom=bottom[i], label=label, color=col)
            labels_deja_affiches.add(tech)
            if val >= seuil_texte:
                ax.text(bar[0].get_x() + bar[0].get_width()/2,
                        bottom[i] + val/2,
                        f"{val:.1f}",
                        ha='center', va='center', fontsize=11)
            bottom[i] += val

    ax.set_xticks(x)
    ax.set_xticklabels(noms_barres)
    #ax.set_title(titre)
    ax.set_ylabel("Installed power generation capacity in [GW]")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if len(noms_barres) == 1:
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    else:
        ax.legend(ncol = 2)
    plt.tight_layout()
    plt.show()
    


def cost(scenario_path, txt_file_name="cost_breakdown.txt"):


    txt_path = os.path.join(scenario_path, "output", txt_file_name)
    csv_path = txt_path.replace('.txt', '.csv')

    if not os.path.exists(txt_path):
        print(f" Fichier non trouvé : {txt_path}")
        return None

    try:
        # Conversion TXT → CSV
        df = pd.read_csv(txt_path, sep=None, engine='python', skiprows=[1])
        df.columns = df.columns.str.strip()
        df.to_csv(csv_path, index=False)

    except Exception as e:
        print(f" Erreur lors de la conversion : {e}")
        return None

    # Vérification qu’il y a au moins 4 colonnes
    if df.shape[1] < 4:
        print(f" Format inattendu : seulement {df.shape[1]} colonnes trouvées.")
        return None

    # Somme des colonnes 2, 3 et 4
    col2_sum = df.iloc[:, 1].sum()
    col3_sum = df.iloc[:, 2].sum()
    col4_sum = df.iloc[:, 3].sum()
    total_sum = col2_sum + col3_sum + col4_sum

    # Récupération des noms
    colnames = df.columns[1:4].tolist()
    resultats = {
        colnames[0]: col2_sum,
        colnames[1]: col3_sum,
        colnames[2]: col4_sum,
        "Total": total_sum
    }

    print("---COSTS---")
    for k, v in resultats.items():
        print(f"  {k} : {v:.2f} M€")

    return resultats


def gwp(scenario_path, txt_file_name="gwp_breakdown.txt"):


    txt_path = os.path.join(scenario_path, "output", txt_file_name)
    csv_path = txt_path.replace('.txt', '.csv')

    if not os.path.exists(txt_path):
        print(f" Fichier non trouvé : {txt_path}")
        return None

    try:
        # Conversion TXT → CSV
        df = pd.read_csv(txt_path, sep=None, engine='python', skiprows=[1])
        df.columns = df.columns.str.strip()
        df.to_csv(csv_path, index=False)
        
    except Exception as e:
        print(f" Erreur lors de la conversion : {e}")
        return None

    # Vérification qu’il y a au moins 4 colonnes
    if df.shape[1] < 3:
        print(f" Format inattendu : seulement {df.shape[1]} colonnes trouvées.")
        return None

    # Somme des colonnes 2, 3 et 4
    col2_sum = df.iloc[:, 1].sum()
    col3_sum = df.iloc[:, 2].sum()
    total_sum = col2_sum + col3_sum

    # Récupération des noms
    colnames = df.columns[1:3].tolist()
    resultats = {
        colnames[0]: col2_sum,
        colnames[1]: col3_sum,
        "Total": total_sum
    }

    print("---GWP--- ")
    for k, v in resultats.items():
        print(f"  {k} : {v:.2f} ktCO2_eq")

    return resultats


import os
import csv

def imports_storage(scenario_path):
    """
    Analyse les importations et le stockage pour un seul scénario EnergyScope.
    Retourne un dictionnaire contenant :
    - % d'importation (incluant les lignes 'Imp.' et 'Electricity')
    - énergie importée en Wh
    - énergie issue du stockage en Wh
    """
    sankey_path = os.path.join(scenario_path, "output", "sankey", "input2sankey.csv")

    if not os.path.exists(sankey_path):
        print(f" Fichier non trouvé : {sankey_path}")
        return None

    import_sum = 0.0
    storage_sum = 0.0
    total_system_input = 0.0
    all_targets = set()

    try:
        with open(sankey_path, newline='', encoding='utf-8') as csvfile:
            reader = list(csv.reader(csvfile))
            data_rows = reader[1:]  # ignore header
            all_targets = set(row[1].strip() for row in data_rows if len(row) > 1)

            for row in data_rows:
                try:
                    source = row[0].strip()
                    target = row[1].strip()
                    value = float(row[2])

                    # Compte comme importation si ligne commence par 'Imp.' ou est 'Electricity'
                    if source.startswith("Imp.") or source == "Electricity":
                        import_sum += value

                    # Compte comme stockage si le mot "sto" est dans la source
                    if "sto" in source.lower():
                        storage_sum += value

                    # Source non utilisée comme cible ailleurs = entrée primaire
                    if source not in all_targets:
                        total_system_input += value

                except (IndexError, ValueError):
                    print(f"Erreur de lecture dans une ligne du fichier {sankey_path}")

        if total_system_input == 0:
            print(f"Dénominateur nul : total des flux d'entrée = 0.")
            return None

        pourcentage_import = (import_sum / total_system_input) * 100

        return {
            "import_percent": pourcentage_import,
            "import_wh": import_sum,
            "storage_wh": storage_sum
        }

    except Exception as e:
        print(f"Erreur lors de l’analyse : {e}")
        return None



def plot_gwp(scenario_path):
    """
    Affiche deux barplots côte à côte :
    - à gauche : GWP de construction par groupe de technologies (hors ressources)
    - à droite : GWP d'opération par vecteur énergétique
    """
   
    gwp_path = os.path.join(scenario_path, "output", "gwp_breakdown.txt")
    if not os.path.exists(gwp_path):
        print(f"Fichier GWP non trouvé : {gwp_path}")
        return

    df = pd.read_csv(gwp_path, sep=None, engine='python', skiprows=[1], names=["Name", "GWP_constr", "GWP_op"])
    df = df.dropna()
    df["GWP_constr"] = pd.to_numeric(df["GWP_constr"], errors="coerce")
    df["GWP_op"] = pd.to_numeric(df["GWP_op"], errors="coerce")
    df.set_index("Name", inplace=True)

    noms_groupes = []
    valeurs_constr = []

    for nom, techno_list in groupes_technos.items():
        subset = df.loc[df.index.intersection(techno_list)]
        valeurs_constr.append(subset["GWP_constr"].sum())
        noms_groupes.append(nom)

    # tri décroissant
    group_sorted = sorted(zip(noms_groupes, valeurs_constr), key=lambda x: x[1], reverse=True)
    noms_groupes, valeurs_constr = zip(*group_sorted)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})

    ax1.bar(noms_groupes, valeurs_constr, color="#A0B3C3")
    ax1.set_title("Technology construction GHG emissions")
    ax1.set_ylabel("Construction's global warming\n potential in [kt$CO_2$-eq]")
    ax1.set_xticklabels(noms_groupes, rotation=45, ha='right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    op_data = df.loc[df.index.intersection(ressources)]
    op_data = op_data[op_data["GWP_op"] > 0].sort_values("GWP_op", ascending=False)

    ax2.bar(op_data.index, op_data["GWP_op"], color="#A0B3C3")
    ax2.set_title("Total GHG emissions of resources")
    ax2.set_ylabel("Operation's global warming\n potential in [kgt$CO_2$-eq/y]")
    ax2.set_xticklabels(op_data.index, rotation=90)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

couleur_bleu_froid = "#4B6C8B"
couleur_gris_foncé = "#666666"
couleur_bleu_pâle = "#A0B3C3"

def plot_costs(scenario_path):
    """
    Affiche deux barplots :
    - à gauche : coût de construction (C_inv + C_maint) par groupe de technologies
    - à droite : coût d’opération (C_op) par vecteur énergétique
    """
    cost_path = os.path.join(scenario_path, "output", "cost_breakdown.txt")
    if not os.path.exists(cost_path):
        print(f" Fichier coût non trouvé : {cost_path}")
        return

    df = pd.read_csv(cost_path, sep=None, engine='python', skiprows=[1], names=["Name", "C_inv", "C_maint", "C_op"])
    df = df.dropna()
    df[["C_inv", "C_maint", "C_op"]] = df[["C_inv", "C_maint", "C_op"]].apply(pd.to_numeric, errors="coerce")
    df.set_index("Name", inplace=True)

    noms_groupes = []
    valeurs_inv = []
    valeurs_maint = []

    for nom, techno_list in groupes_technos.items():
        subset = df.loc[df.index.intersection(techno_list)]
        valeurs_inv.append(subset["C_inv"].sum())
        valeurs_maint.append(subset["C_maint"].sum())
        noms_groupes.append(nom)

    # tri des groupes par ordre décroissant d'investissement
    sorted_inv = sorted(zip(noms_groupes, valeurs_inv), key=lambda x: x[1], reverse=True)
    noms_inv, valeurs_inv = zip(*sorted_inv)

    sorted_maint = sorted(zip(noms_groupes, valeurs_maint), key=lambda x: x[1], reverse=True)
    noms_maint, valeurs_maint = zip(*sorted_maint)

    op_data = df.loc[df.index.intersection(ressources)]
    op_data = op_data[op_data["C_op"] > 0].sort_values("C_op", ascending=False)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1.2, 1.2, 1]})

    ax1.bar(noms_inv, valeurs_inv, color="#A0B3C3")
    ax1.set_title("Technology total investment cost")
    ax1.set_ylabel("Investment cost [M€]")
    ax1.set_xticklabels(noms_inv, rotation=45, ha='right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.bar(noms_maint, valeurs_maint, color="#A0B3C3")
    ax2.set_title("Technology yearly maintenance cost")
    ax2.set_ylabel("Maintenance cost [M€/y]")
    ax2.set_xticklabels(noms_maint, rotation=45, ha='right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax3.bar(op_data.index, op_data["C_op"], color="#A0B3C3")
    ax3.set_title("Total cost of resources")
    ax3.set_ylabel("Operating cost [M€/y]")
    ax3.set_xticklabels(op_data.index, rotation=90)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_resources(scenario_path):
    """
    Affiche un barplot représentant la quantité totale d'énergie utilisée par ressource
    à partir du fichier resources_breakdown.txt.
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt



    file_path = os.path.join(scenario_path, "output", "resources_breakdown.txt")
    if not os.path.exists(file_path):
        print(f" Fichier non trouvé : {file_path}")
        return

    # Lecture du fichier
    df = pd.read_csv(file_path, sep=None, engine="python", skiprows=[1])
    df.columns = df.columns.str.strip()

    # Sélection et nettoyage
    df = df[df["Name"].isin(ressources)]
    df["Used"] = pd.to_numeric(df["Used"], errors="coerce")
    df = df.dropna(subset=["Used"])
    df = df[df["Used"] > 0]
    df = df.sort_values("Used", ascending=False)

    # Tracé
    plt.figure(figsize=(12, 6))
    plt.bar(df["Name"], df["Used"]/1000, color="#A0B3C3")
    plt.ylabel("Resources used [TWh/y]")  # à adapter si besoin
    plt.xticks(rotation=45,  ha='right')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def Electrification (csv_path, col_label="f"):
    """
    Calcule le taux d’électrification pour chaque usage final :
    - High Temp Heat
    - Low Temp Heat
    - Public Mobility
    - Private Mobility
    - Freight Mobility
    ...et pour le stockage
    """

    df = pd.read_csv(csv_path)
    df.set_index(df.columns[0], inplace=True)

    # === Usages finaux ===
    techs_elec = {
        "High Temp Heat": ["IND_DIRECT_ELEC"],
        "Low Temp Heat": ["DHN_HP_ELEC", "DEC_HP_ELEC", "DEC_DIRECT_ELEC"],
        "Public Mobility": ["TRAMWAY_TROLLEY", "TRAIN_PUB"],
        "Private Mobility": ["CAR_PHEV", "CAR_BEV"],
        "Freight Mobility": ["TRAIN_FREIGHT", "TRUCK_ELEC"]
    }

    techs_total = {
        "High Temp Heat": ["IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_BOILER_GAS",
                           "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE", "IND_DIRECT_ELEC"],
        "Low Temp Heat": ["DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE", "DHN_COGEN_WET_BIOMASS",
                          "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS", "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO",
                          "DHN_SOLAR", "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL", "DEC_ADVCOGEN_GAS",
                          "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL", "DEC_SOLAR", "DEC_DIRECT_ELEC"],
        "Public Mobility": ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL", "BUS_COACH_HYDIESEL",
                            "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"],
        "Private Mobility": ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL",
                             "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"],
        "Freight Mobility": ["TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG",
                             "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL", "TRUCK_METHANOL",
                             "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"]
    }

    # === Stockage ===
    technos_storage_elec = ["BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS"]
    technos_storage_thermal = ["TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS",
                               "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS",
                               "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP"]
    technos_storage_other = ["GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE", "LFO_STORAGE",
                             "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE"]
    technos_storage_total = technos_storage_elec + technos_storage_thermal + technos_storage_other


    def somme(tech_list):
        return sum(df.loc[tech, col_label] for tech in tech_list if tech in df.index)

    resultats = {}

    for usage, total_list in techs_total.items():
        total = somme(total_list)
        elec = somme(techs_elec[usage])
        pourcentage = 100 * elec / total if total > 0 else 0
        resultats[usage] = {
            "Total": total,
            "Electric": elec,
            "Electrification [%]": round(pourcentage, 2)
        }

    # Stockage : électrification
    sto_total = somme(technos_storage_total)
    sto_elec = somme(technos_storage_elec)
    sto_ratio = 100 * sto_elec / sto_total if sto_total > 0 else 0
    resultats["Storage Electrification"] = {
        "Total": sto_total,
        "Electric": sto_elec,
        "Electrification [%]": round(sto_ratio, 2)
    }

    return resultats

def capacities_dual_uniform(csv_path, groupes1, groupes2,
                            col_label="f", titre1="", titre2="",
                            seuil=5, seuil_texte=1,
                            couleurs1=None, couleurs2=None,
                            bar_width=0.5,
                            ylabel1="Installed capacity", ylabel2="Installed capacity"):
    try:
        df = pd.read_csv(csv_path)
        df.set_index(df.columns[0], inplace=True)
    except Exception as e:
        print(f"[❌] Erreur lecture CSV : {e}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                   gridspec_kw={'width_ratios': [len(groupes1), len(groupes2)]})

    # ✅ Ensemble global pour éviter doublons entre les deux graphes
    labels_deja_affiches_global = set()

    def tracer(ax, groupes, titre, couleurs, labels_utilises, ylabel="Installed capacity"):
        noms_barres = list(groupes.keys())
        nb_groupes = len(groupes)
        
        if nb_groupes == 1:
            x_positions = [0.2]
            bar_width_effective = 0.3
            ax.set_xlim(0, 1)
        else:
            x_positions = np.arange(nb_groupes)
            bar_width_effective = 0.3

        bottom = np.zeros(nb_groupes)

        for i, (nom_barre, technos) in enumerate(groupes.items()):
            valeurs = []
            techno_valides = []
            for techno in technos:
                if techno in df.index:
                    val = df.loc[techno, col_label]
                    if val > seuil:
                        valeurs.append(val)
                        techno_valides.append(techno)

            if not valeurs:
                continue

            sorted_pairs = sorted(zip(techno_valides, valeurs), key=lambda x: x[1], reverse=True)
            techno_sorted, valeurs_sorted = zip(*sorted_pairs)
            couleurs_barres = [couleurs.get(tech, "#CCCCCC") for tech in techno_sorted]

            for tech, val, col in zip(techno_sorted, valeurs_sorted, couleurs_barres):
                if col == "#CCCCCC" or tech in labels_utilises:
                    label = "_nolegend_"
                else:
                    label = tech
                    labels_utilises.add(tech)  # ✅ Ajout global

                bar = ax.bar(x_positions[i], val, width=bar_width_effective,
                             bottom=bottom[i], label=label, color=col)
                if val >= seuil_texte:
                    ax.text(bar[0].get_x() + bar[0].get_width()/2,
                            bottom[i] + val/2,
                            f"{val:.1f}",
                            ha='center', va='center', fontsize=12)
                bottom[i] += val

        ax.set_xticks(x_positions)
        ax.set_xticklabels(noms_barres, rotation=0, ha='center')
        ax.set_title(titre)
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ⚠️ Pas de legend ici

    # ✅ Utilisation du même set pour éviter doublons
    tracer(ax1, groupes1, titre1, couleurs1, labels_deja_affiches_global, ylabel1)
    tracer(ax2, groupes2, titre2, couleurs2, labels_deja_affiches_global, ylabel2)

    # Création d'une légende commune
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

def capacities_dual_uniform(csv_path, groupes1, groupes2, col_label="f", titre1="", titre2="",
                            seuil=5, seuil_texte=1,
                            couleurs=None,  # ✅ une seule table de couleurs
                            bar_width=0.5,
                            ylabel1="Installed capacity", ylabel2="Installed capacity"):
    try:
        df = pd.read_csv(csv_path)
        df.set_index(df.columns[0], inplace=True)
    except Exception as e:
        print(f"[❌] Erreur lecture CSV : {e}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                   gridspec_kw={'width_ratios': [len(groupes1), len(groupes2)]})

    labels_deja_affiches_global = set()  # pour ne labeller chaque techno qu'une seule fois

    def tracer(ax, groupes, titre, couleurs_map, labels_utilises, ylabel="Installed capacity"):
        noms_barres = list(groupes.keys())
        nb_groupes = len(groupes)

        if nb_groupes == 1:
            x_positions = [0.2]
            bar_width_effective = 0.3
            ax.set_xlim(0, 1)
        else:
            x_positions = np.arange(nb_groupes)
            bar_width_effective = 0.3

        bottom = np.zeros(nb_groupes)

        for i, (nom_barre, technos) in enumerate(groupes.items()):
            valeurs, techno_valides = [], []
            for techno in technos:
                if techno in df.index:
                    val = df.loc[techno, col_label]
                    if val > seuil:
                        valeurs.append(val)
                        techno_valides.append(techno)
            if not valeurs:
                continue

            techno_sorted, valeurs_sorted = zip(*sorted(zip(techno_valides, valeurs), key=lambda x: x[1], reverse=True))
            couleurs_barres = [couleurs_map.get(tech, "#CCCCCC") for tech in techno_sorted]

            for tech, val, col in zip(techno_sorted, valeurs_sorted, couleurs_barres):
                label = "_nolegend_" if (tech in labels_utilises) else tech
                labels_utilises.add(tech)

                bar = ax.bar(x_positions[i], val, width=bar_width_effective,
                             bottom=bottom[i], label=label, color=col)
                if val >= seuil_texte:
                    ax.text(bar[0].get_x() + bar[0].get_width()/2,
                            bottom[i] + val/2, f"{val:.1f}",
                            ha='center', va='center', fontsize=12)
                bottom[i] += val

        ax.set_xticks(x_positions)
        ax.set_xticklabels(noms_barres, rotation=0, ha='center')
        if titre:
            ax.set_title(titre)
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # --- Tracés avec la même table de couleurs ---
    tracer(ax1, groupes1, titre1, couleurs, labels_deja_affiches_global, ylabel1)
    tracer(ax2, groupes2, titre2, couleurs, labels_deja_affiches_global, ylabel2)

    # Légende commune (sans doublons)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    seen = set()
    handles, labels = [], []
    for h, lab in zip(h1 + h2, l1 + l2):
        if lab != "_nolegend_" and lab not in seen:
            handles.append(h); labels.append(lab); seen.add(lab)

    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

def colors_def_unique_dual(csv_path, groupes1, groupes2, col_label,
                           techno_par_ressource, palettes):
    """
    Assigne des couleurs SANS duplication sur l’ensemble (groupes1 + groupes2).
    Priorité: palettes par ressource -> fallback sur le cycle matplotlib -> gris.
    """
    df = pd.read_csv(csv_path)
    df.set_index(df.columns[0], inplace=True)

    # 1) Toutes les technos concernées, triées par valeur décroissante (col_label)
    toutes_technos = set()
    for techs in list(groupes1.values()) + list(groupes2.values()):
        toutes_technos.update(techs)

    techno_valeurs = [(tech, df.loc[tech, col_label])
                      for tech in toutes_technos
                      if tech in df.index and pd.notna(df.loc[tech, col_label]) and df.loc[tech, col_label] > 0]
    techno_valeurs.sort(key=lambda x: x[1], reverse=True)

    # 2) Attribution de couleurs sans répétition
    couleurs = {}
    used_colors = set()
    compteurs_palette = {ressource: 0 for ressource in palettes}
    fallback_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    fallback_idx = 0

    def next_palette_color(resource):
        i = compteurs_palette[resource]
        pal = palettes[resource]
        # cherche une couleur non utilisée dans la palette
        while i < len(pal) and pal[i] in used_colors:
            i += 1
        if i < len(pal):
            compteurs_palette[resource] = i + 1
            return pal[i]
        return None

    def next_fallback_color():
        nonlocal fallback_idx
        while fallback_idx < len(fallback_cycle) and fallback_cycle[fallback_idx] in used_colors:
            fallback_idx += 1
        if fallback_idx < len(fallback_cycle):
            c = fallback_cycle[fallback_idx]
            fallback_idx += 1
            return c
        # Dernier recours : gris clair (différencié)
        step = 180 + (len(used_colors) * 7) % 60
        return f"#{step:02x}{step:02x}{step:02x}"

    # 3) Parcours des technos par importance et affectation
    for tech, _ in techno_valeurs:
        ressource_tech = None
        for ressource, tech_set in techno_par_ressource.items():
            if tech in tech_set:
                ressource_tech = ressource
                break

        color = next_palette_color(ressource_tech) if ressource_tech else None
        if color is None:
            color = next_fallback_color()

        # Ceinture & bretelles : évite toute collision
        while color in used_colors:
            color = next_fallback_color()

        used_colors.add(color)
        couleurs[tech] = color

    return couleurs


# === Lancement
scenario_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/scénarios_finaux/10MGT"
txt_file_name = "assets.txt"
col_label = "f"
csv_path = convert_txt_en_csv(scenario_path, txt_file_name)

if csv_path:
    
    groupes_prod = {
        "Electricity": technos_electricity,
        "High temperature heat": technos_heat_high,
        "Low temperature heat": technos_heat_low_central + technos_heat_low_decentral
    }
    couleurs_prod = colors_def(
        csv_path, groupes_prod, col_label, techno_par_ressource, palettes)
    capacities(csv_path, groupes_prod, col_label,
                                    seuil=0.01, seuil_texte=1, couleurs_fixes=couleurs_prod)

    groupes_mob = {
        "Public mobility": technos_mobility_public,
        "Private mobility": technos_mobility_private
    }
    couleurs_mob = colors_def(
        csv_path, groupes_mob, col_label, techno_par_ressource, palettes)
    #capacities(csv_path, groupes_mob, col_label,
              #                       "People mobility in [Mpass$\cdot$km]", seuil=0.01, seuil_texte=1, couleurs_fixes=couleurs_mob)

    groupes_fret = {
        "Freight mobility ": technos_mobility_freight
    }
    couleurs_fret =colors_def(
        csv_path, groupes_fret, col_label, techno_par_ressource, palettes)
   # capacities(csv_path, groupes_fret, col_label, "Freight mobility in [Mt$\cdot$km]",
          #                           seuil=0.01, seuil_texte=1, couleurs_fixes=couleurs_fret, bar_width=0.2)

    groupes_sto = {
        "Electric storage": technos_storage_elec,
        "Thermal storage": technos_storage_thermal
    }
    couleurs_sto = colors_def(
        csv_path, groupes_sto, col_label, techno_par_ressource, palettes)
    #capacities(csv_path, groupes_sto, col_label,
        #                             "Electric and thermal storage in [GWh]", seuil=1, seuil_texte=10, couleurs_fixes=couleurs_sto)

    groupes_autres = {
        "Other storages": technos_storage_other
    }
    couleurs_autres = colors_def(
        csv_path, groupes_autres, col_label, techno_par_ressource, palettes)
    #capacities(csv_path, groupes_autres, col_label,
                   #                  "Storage of chemical energy carriers in [GWh]", seuil=10, seuil_texte=1, couleurs_fixes=couleurs_autres)
    
    costs = cost(scenario_path)
    gwps = gwp(scenario_path)
    
    # imports = imports_storage(scenario_path)
    # print('---IMPORTS & STORAGE ---')
    # print(f"   Imports : {imports['import_wh']:.2f} TWh = {imports['import_percent']:.2f} %")
    # print(f"   Storage : {imports['storage_wh']:.2f} TWh")

    res = Electrification(csv_path)
    print('---ELECTRIFICATION---')
    for usage, valeurs in res.items():
        print(f"{usage} → {valeurs['Electrification [%]']} %")
        
    plot_gwp(scenario_path)
    plot_costs(scenario_path)
    plot_resources(scenario_path)
#     groupes_mob1 = {
#         "Public mobility": technos_mobility_public,
#         "Private mobility": technos_mobility_private
#     }
#     groupes_mob2 = {
#         "Freight mobility": technos_mobility_freight
#     }

#     couleurs_mob1 = colors_def(csv_path, groupes_mob1, col_label, techno_par_ressource, palettes)
#     couleurs_mob2 = colors_def(csv_path, groupes_mob2, col_label, techno_par_ressource, palettes)

#     capacities_dual_uniform(csv_path,
#     groupes1=groupes_mob1,
#     groupes2=groupes_mob2,
#     #titre1="Passenger mobility in [Mpass·km]",
#     #titre2="",
#     couleurs1=couleurs_mob1,
#     couleurs2=couleurs_mob2,
#     seuil=5,
#     seuil_texte=1,
#     ylabel1="Passenger mobility in [Mpass·km]",
#     ylabel2="Freight mobility in [Mt·km]"
# )

#     groupes_sto1 = {
#         "Electric storage": technos_storage_elec,
#         "Thermal storage": technos_storage_thermal
#     }
#     groupes_sto2 = {
#         "Chemical storage": technos_storage_other
#     }

#     couleurs_sto1 = colors_def(csv_path, groupes_sto1, col_label, techno_par_ressource, palettes)
#     couleurs_sto2 = colors_def(csv_path, groupes_sto2, col_label, techno_par_ressource, palettes)

#     capacities_dual_uniform(
#     csv_path,
#     groupes1=groupes_sto1,
#     groupes2=groupes_sto2,
#     #titre1="Installed electrical and thermal storage capacity in [GWh]",
#     #titre2="Installed capacity for chemical energy storage in [GWh]",
#     couleurs1=couleurs_sto1,
#     couleurs2=couleurs_sto2,
#     seuil=5,
#     seuil_texte=10,
#     ylabel1="Installed electrical and thermal \n storage capacity in [GWh]",
#     ylabel2="Installed capacity for chemical \n energy storage in [GWh]"
# )
# --- MOBILITÉ ---
groupes_mob1 = {"Public mobility": technos_mobility_public,
                "Private mobility": technos_mobility_private}
groupes_mob2 = {"Freight mobility": technos_mobility_freight}

couleurs_mob_unique = colors_def_unique_dual(
    csv_path, groupes_mob1, groupes_mob2, col_label, techno_par_ressource, palettes
)

capacities_dual_uniform(
    csv_path,
    groupes1=groupes_mob1,
    groupes2=groupes_mob2,
    couleurs=couleurs_mob_unique,     # ✅ une seule table
    seuil=5,
    seuil_texte=1,
    ylabel1="Passenger mobility in [Mpass·km]",
    ylabel2="Freight mobility in [Mt·km]"
)

# --- STOCKAGE ---
groupes_sto1 = {"Electric storage": technos_storage_elec,
                "Thermal storage": technos_storage_thermal}
groupes_sto2 = {"Chemical storage": technos_storage_other}

couleurs_sto_unique = colors_def_unique_dual(
    csv_path, groupes_sto1, groupes_sto2, col_label, techno_par_ressource, palettes
)

capacities_dual_uniform(
    csv_path,
    groupes1=groupes_sto1,
    groupes2=groupes_sto2,
    couleurs=couleurs_sto_unique,     # ✅ une seule table
    seuil=5,
    seuil_texte=10,
    ylabel1="Installed electrical and thermal \n storage capacity in [GWh]",
    ylabel2="Installed capacity for chemical \n energy storage in [GWh]"
)

    

