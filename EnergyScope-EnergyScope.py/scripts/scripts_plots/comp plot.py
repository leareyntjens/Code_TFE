import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

plt.rcParams.update({'font.size': 16})

def convert_txt_to_csv(txt_path):
    csv_path = txt_path.replace(".txt", ".csv")
    try:
        df = pd.read_csv(txt_path, sep=None, engine="python", skiprows=[1])
        df.columns = df.columns.str.strip()
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        print(f"[ERREUR] Conversion {txt_path} : {e}")
        return None
    

couleur_bleu_froid = "#4B6C8B"
couleur_gris_foncé = "#666666"
couleur_bleu_pâle = "#A0B3C3"
colors = [couleur_bleu_froid, "darkgrey", couleur_bleu_pâle]
colors = [couleur_bleu_froid, couleur_bleu_froid, couleur_bleu_froid]
alphas = [1.0, 0.7, 0.3]  # #1 = opaque, #2 = semi, #3 = clair


def get_top3(df, tech_list, col="f"):
    return sorted([(t, df.at[t, col]) for t in tech_list if t in df.index and df.at[t, col] > 0],
                  key=lambda x: x[1], reverse=True)[:3]



def plot_top3_all_scenarios_grid(scenarios, tech_groups, colors, alphas):
    n = len(tech_groups)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for idx, (group, techs) in enumerate(tech_groups.items()):
        ax = axes[idx]
        top3_by_scenario = {}
        for name, path in scenarios.items():
            df = pd.read_csv(path, index_col=0)
            top3 = sorted([(t, df.at[t, "f"]) for t in techs if t in df.index and df.at[t, "f"] > 0],
                          key=lambda x: x[1], reverse=True)[:3]
            top3_by_scenario[name] = top3

        x = np.arange(len(scenarios))
        width = 0.25

        for i in range(3):
            vals, labels = [], []
            for s in scenarios:
                tups = top3_by_scenario[s]
                labels.append(tups[i][0] if i < len(tups) else "")
                vals.append(tups[i][1] if i < len(tups) else 0)
            bars = ax.bar(x + i * width - width, vals, width, label=f"#{i+1}",
                          color=colors[i], alpha=alphas[i])

            for j, label in enumerate(labels):
                rotation = 0 if i == 0 else 45
                ha = 'center' if i == 0 else 'left'
                ax.text(x[j] + i * width - width, vals[j] + 0.01 * max(vals), label,
                        ha=ha, va='bottom', fontsize=11, rotation=rotation)
                ylabel_mapping = {
    "Electricity": "Installed capacities (GW)",
    "Heat - HT": "Installed capacities (GW)",
    "Heat - LT": "Installed capacities (GW)",
    "Mobility - Private": "Installed capacities (Mpass$\cdot$ km)",
    "Mobility - Public": "Installed capacities (Mpass$\cdot$ km)",
    "Mobility - Freight": "Installed capacities (Mt$\cdot$km)",
    "Storage": "Installed capacities (GWh)",
    "Conversion": "Installed capacities (GW)"
}

        ax.set_ylabel(ylabel_mapping.get(group, "Capacity (GW)"))


        ax.set_xticks(x)
        ax.set_xticklabels(list(scenarios.keys()))
        ax.set_title(f"{group}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Remove unused subplots
    for idx in range(len(tech_groups), len(axes)):
        fig.delaxes(axes[idx])

    legend = fig.legend([f"Top  {i+1}" for i in range(3)], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.99))
    legend.get_frame().set_linewidth(0)  # enlève le contour
    legend.get_frame().set_facecolor('none')  # rend l'arrière-plan transparent (optionnel)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    for group, techs in tech_groups.items():
        top3_by_scenario = {}
        for name, path in scenarios.items():
            df = pd.read_csv(path, index_col=0)
            top3 = get_top3(df, techs)
            top3_by_scenario[name] = top3

        x = np.arange(len(scenarios))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(3):
            vals, labels = [], []
            for s in scenarios:
                tups = top3_by_scenario[s]
                labels.append(tups[i][0] if i < len(tups) else "")
                vals.append(tups[i][1] if i < len(tups) else 0)
            bars = ax.bar(x + i * width - width, vals, width, label=f"#{i+1}", color=colors[i], alpha=alphas[i])

            for j, label in enumerate(labels):
                rotation = 0 if i == 0 else 45  # Pas d'inclinaison pour la 1ère barre
                ha = 'center' if i == 0 else 'left'
                
                ax.text(
                    x[j] + i * width - width,
                    vals[j] + 0.01 * max(vals),
                    label,
                    ha=ha,
                    va='bottom',
                    fontsize=11,
                    rotation=rotation
                )

        ax.set_xticks(x)
        ax.set_xticklabels(list(scenarios.keys()))
        ax.set_title(f"Top 3 technologies – {group}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        plt.tight_layout()
        plt.show()

def plot_resource_use_from_csv(file_paths, scenario_labels=None, seuil=1.0):
    color_mapping = {
        'GAS_RE': 'royalblue', 'H2_RE': 'royalblue', 'AMMONIA_RE': 'royalblue', 'METHANOL_RE': 'royalblue',
        'WOOD': 'olivedrab', 'WET_BIOMASS': 'olivedrab', 'WASTE': 'olivedrab',
        'RES_WIND': 'orange', 'RES_SOLAR': 'orange',
        'CO2_EMISSIONS': 'gainsboro'
    }

    alpha_mapping = {
        'GAS_RE': 1.0, 'H2_RE': 0.3, 'AMMONIA_RE': 0.5, 'METHANOL_RE': 0.75,
        'WOOD': 0.65, 'WET_BIOMASS': 1, 'WASTE': 0.45,
        'RES_WIND': 0.3, 'RES_SOLAR': 0.6
    }

    resource_categories = {
        'Synthetic RE vectors': ['GAS_RE', 'H2_RE', 'AMMONIA_RE', 'METHANOL_RE'],
        'Biomass & waste': ['WOOD', 'WET_BIOMASS', 'WASTE'],
        'Direct renewables': ['RES_WIND', 'RES_SOLAR'],
        'CO₂ indicators': ['CO2_EMISSIONS']
    }

    dataframes = []
    for path in file_paths:
        df = pd.read_csv(path, index_col=0)
        df.columns = df.columns.str.strip()
        df = df[df.columns[0]].rename("GWh")
        df = df / 1000
        df.name = "TWh"
        dataframes.append(df)

    if scenario_labels is None:
        scenario_labels = [f"Scenario {i+1}" for i in range(len(dataframes))]

    combined_df = pd.concat(dataframes, axis=1)
    combined_df.columns = scenario_labels
    combined_df = combined_df.fillna(0)

    mask = (combined_df > seuil).any(axis=1)
    filtered_df = combined_df[mask]
    if 'CO2_EMISSIONS' in filtered_df.index:
        filtered_df = filtered_df.drop(index='CO2_EMISSIONS')

    ordered_resources = []
    for category, resources in resource_categories.items():
        cat_resources = [res for res in resources if res in filtered_df.index]
        cat_resources_sorted = sorted(cat_resources, key=lambda r: -filtered_df.loc[r].sum())
        ordered_resources.extend(cat_resources_sorted)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0] * len(scenario_labels)
    bars_by_resource = {}
    category_labels = defaultdict(list)

    for resource in ordered_resources:
        values = filtered_df.loc[resource]
        color = color_mapping.get(resource, 'grey')
        alpha = alpha_mapping.get(resource, 1.0)
        bar = ax.bar(
            scenario_labels,
            values,
            bottom=bottom,
            color=color,
            alpha=alpha,
            label=resource,
        )
        bars_by_resource[resource] = bar[0]
        for category, res_list in resource_categories.items():
            if resource in res_list:
                category_labels[category].append(resource)
        bottom = [b + v for b, v in zip(bottom, values)]

    handles = []
    labels = []
    for category, resources in category_labels.items():
        handles.append(plt.Line2D([0], [0], color='none', linestyle=''))
        labels.append(f"$\cdot$ {category}")
        for res in resources:
            handles.append(bars_by_resource[res])
            labels.append(f"  {res}")

    legend = ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1.05, 1))
    legend.get_frame().set_linewidth(0)  # enlève le contour
    legend.get_frame().set_facecolor('none')  # rend l'arrière-plan transparent (optionnel)
    for text in legend.get_texts():
        if text.get_text().startswith("$\cdot$"):
            text.set_fontweight('bold')

    ax.set_ylabel("Annual resources used (TWh/y)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_storage_diff_percent(ref_path, cost_path, gwp_path, storage_techs, seuil=1.0):
    # Charger les données
    ref_df = pd.read_csv(ref_path, index_col=0)
    cost_df = pd.read_csv(cost_path, index_col=0)
    gwp_df = pd.read_csv(gwp_path, index_col=0)

    # S'assurer que 'f' est bien la colonne cible
    ref_df = ref_df[['f']] if 'f' in ref_df.columns else ref_df
    cost_df = cost_df[['f']] if 'f' in cost_df.columns else cost_df
    gwp_df = gwp_df[['f']] if 'f' in gwp_df.columns else gwp_df

    # Initialisation
    techs_to_plot = []
    diffs_cost = []
    diffs_gwp = []

    for tech in storage_techs:
        val_ref = ref_df.at[tech, 'f'] if tech in ref_df.index else 0
        val_cost = cost_df.at[tech, 'f'] if tech in cost_df.index else 0
        val_gwp = gwp_df.at[tech, 'f'] if tech in gwp_df.index else 0

        # if val_ref < seuil and val_cost < seuil and val_gwp < seuil:
        #     print(f"[IGNORÉ] {tech} < {seuil} TWh dans tous les scénarios.")
        #     continue

        # if val_cost < seuil and val_gwp < seuil:
        #     print(f"[DISPARU] {tech} présent dans ref mais absent dans les deux autres.")
        #     continue

        # Calcul des % de variation
        diff_cost = 100 * (val_cost - val_ref) / val_ref if val_ref > 0 else 0
        diff_gwp = 100 * (val_gwp - val_ref) / val_ref if val_ref > 0 else 0

        techs_to_plot.append(tech)
        diffs_cost.append(diff_cost)
        diffs_gwp.append(diff_gwp)

    # Création du graphique
    x = range(len(techs_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar([i - width/2 for i in x], diffs_cost, width, 
                   color=['green' if v > 0 else 'red' for v in diffs_cost], label='Min Cost vs Ref')

    bars2 = ax.bar([i + width/2 for i in x], diffs_gwp, width, 
                   color=['green' if v > 0 else 'red' for v in diffs_gwp], label='Min GWP vs Ref')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(techs_to_plot, rotation=45, ha='right')
    ax.set_ylabel("Difference vs REF (%)")
    ax.set_title("Percentage Change in Installed Storage Capacity Compared to Reference")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_storage_variations_heatmap_and_barplot(assets_paths, storage_techs, seuil=2):
    # Lire les 3 fichiers
    df_ref = pd.read_csv(assets_paths["Reference"], index_col=0)
    df_cost = pd.read_csv(assets_paths["Min GWP"], index_col=0)
    df_gwp = pd.read_csv(assets_paths["Min GWP corr"], index_col=0)

    variations_gwh = []
    variations_pct = []
    techs_valides = []
    exclude_from_barplot = {"GAS_STORAGE", "TS_DHN_SEASONAL"}


    for tech in storage_techs:
        if tech not in df_ref.index and tech not in df_cost.index and tech not in df_gwp.index:
            continue

        val_ref = df_ref.at[tech, 'f'] if tech in df_ref.index else 0
        val_cost = df_cost.at[tech, 'f'] if tech in df_cost.index else 0
        val_gwp = df_gwp.at[tech, 'f'] if tech in df_gwp.index else 0

        # Nouveau filtre : on garde si une des trois valeurs est > seuil
        if max(val_ref, val_cost, val_gwp) <= seuil:
            continue

        # Différences absolues
        delta_cost = val_gwp - val_cost
        delta_gwp = val_gwp - val_ref
        variations_gwh.append([delta_cost, delta_gwp])

        # Différences en %
        pct_cost = 100 * delta_cost / val_cost if val_cost != 0 else 0
        pct_gwp = 100 * delta_gwp / val_ref if val_ref != 0 else 0
        variations_pct.append([pct_cost, pct_gwp])

        techs_valides.append(tech)

    # ---- HEATMAP (variation en %) ----
    df_pct = pd.DataFrame(variations_pct, index=techs_valides, columns=["Min gwp corr vs gwp", "Min GWP corr vs Ref"])

    # Création d’un DataFrame de strings pour l’annotation
    df_annots = df_pct.copy()
    for i in df_annots.index:
        for j in df_annots.columns:
            val = df_annots.loc[i, j]
            if val > 100:
                df_annots.loc[i, j] = ">100%"
            elif val < -100:
                df_annots.loc[i, j] = "<-100%"
            else:
                df_annots.loc[i, j] = f"{val:.1f}%"

    plt.figure(figsize=(8, len(techs_valides) * 0.5 + 1))
    sns.heatmap(df_pct.clip(-100, 100), annot=df_annots, fmt="", cmap="RdYlGn", center=0)
    plt.tight_layout()
    plt.show()

        # ---- BARPLOT HORIZONTAL (variation en GWh) ----
    df_gwh = pd.DataFrame(variations_gwh, index=techs_valides, columns=["Min Cost vs Ref", "Min GWP vs Ref"])

    # Exclure certaines technos du barplot
    exclude_from_barplot = {"GAS_STORAGE", "TS_DHN_SEASONAL"}
    df_gwh_filtered = df_gwh[~df_gwh.index.isin(exclude_from_barplot)]

    # Trier les technologies par variation max absolue
    df_gwh_filtered["abs_max"] = df_gwh_filtered.abs().max(axis=1)
    df_gwh_filtered = df_gwh_filtered.sort_values("abs_max", ascending=False)
    df_gwh_filtered.drop(columns="abs_max", inplace=True)

    fig, ax = plt.subplots(figsize=(12, len(df_gwh_filtered) * 0.7 + 1))

    bar_width = 0.7
    y_pos = np.arange(0, len(df_gwh_filtered) * 2, 2)

    bars1 = ax.barh(y_pos - bar_width/2, df_gwh_filtered["Min Cost vs Ref"], height=bar_width, label="Min Cost vs Ref", color=couleur_bleu_froid)
    bars2 = ax.barh(y_pos + bar_width/2, df_gwh_filtered["Min GWP vs Ref"], height=bar_width, label="Min GWP vs Ref", color=couleur_bleu_pâle)

    # Ajouter les valeurs au bout des barres (arrondies, avec espace)
    for i, (v1, v2) in enumerate(zip(df_gwh_filtered["Min Cost vs Ref"], df_gwh_filtered["Min GWP vs Ref"])):
        if abs(v1) > 1:
            ax.text(v1 + (1.5 if v1 >= 0 else -1.5), y_pos[i] - bar_width/2, f"{round(v1):d}", va="center", ha="left" if v1 >= 0 else "right")
        if abs(v2) > 1:
            ax.text(v2 + (1.5 if v2 >= 0 else -1.5), y_pos[i] + bar_width/2, f"{round(v2):d}", va="center", ha="left" if v2 >= 0 else "right")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_gwh_filtered.index)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Change in installed capacity in GWh", fontsize = 18)
    legend = ax.legend()
    legend.get_frame().set_linewidth(0)  # enlève le contour
    legend.get_frame().set_facecolor('none')  # rend l'arrière-plan transparent (optionnel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(-200,60)
    plt.tight_layout()
    plt.show()

    
def plot_resource_variation_heatmap(assets_paths, resource_list, seuil=2):
    # Chargement des fichiers
    df_ref = pd.read_csv(assets_paths["Reference"], index_col=0)
    df_gwp = pd.read_csv(assets_paths["Min GWP"], index_col=0)
    df_corr = pd.read_csv(assets_paths["Min GWP corr"], index_col=0)

    variations_pct = []
    valid_resources = []
    
    for res in resource_list:
        if res not in df_ref.index and res not in df_gwp.index and res not in df_corr.index:
            continue

        val_ref = df_ref.at[res, 'Used'] if res in df_ref.index else 0
        val_gwp = df_gwp.at[res, 'Used'] if res in df_gwp.index else 0
        val_corr = df_corr.at[res, 'Used'] if res in df_corr.index else 0

        if max(val_ref, val_gwp, val_corr) <= seuil:
            continue

        pct_vs_gwp = 100 * (val_corr - val_gwp) / val_gwp if val_gwp != 0 else None
        pct_vs_ref = 100 * (val_corr - val_ref) / val_ref if val_ref != 0 else None
        variations_pct.append([pct_vs_gwp, pct_vs_ref])
        valid_resources.append(res)

    df_pct = pd.DataFrame(variations_pct, index=valid_resources, columns=["Corr vs GWP", "Corr vs Ref"])

    # Annotations textuelles
    df_annots = df_pct.copy()
    for i in df_annots.index:
        for j in df_annots.columns:
            val = df_annots.loc[i, j]
            if pd.isna(val):
                df_annots.loc[i, j] = "–"
            elif val > 100:
                df_annots.loc[i, j] = ">100%"
            elif val < -100:
                df_annots.loc[i, j] = "<-100%"
            else:
                df_annots.loc[i, j] = f"{val:.1f}%"

    df_pct_clipped = df_pct.clip(-100, 100)

    # Plot
    plt.figure(figsize=(8, len(df_pct) * 0.5 + 1))
    sns.heatmap(df_pct_clipped, annot=df_annots, fmt="", cmap="RdYlGn", center=0, linewidths=0.5, linecolor='gray')
    #plt.title("Variation relative de l’usage des ressources dans Min GWP corrigé")
    plt.tight_layout()
    plt.show()


def plot_installed_storage_capacities(scenario_paths, storage_technologies):
    """
    Graphe en barres horizontales des capacités installées (GWh) par technologie de stockage.
    Trié de la plus grande à la plus petite capacité maximale (tous scénarios confondus).
    Affiche une seule valeur si les capacités sont proches à 10% près. Ignore les valeurs < 1 GWh.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Charger les données
    data = {}
    for scenario, path in scenario_paths.items():
        df = pd.read_csv(path, index_col=0)
        data[scenario] = df

    # Créer le DataFrame des capacités
    capacity_data = pd.DataFrame({
        scenario: [data[scenario].at[tech, 'f'] if tech in data[scenario].index else 0
                   for tech in storage_technologies]
        for scenario in scenario_paths
    }, index=storage_technologies)

    # Trier les technologies selon leur capacité maximale
    capacity_data["max"] = capacity_data.max(axis=1)
    capacity_data = capacity_data.sort_values("max", ascending=False).drop(columns="max")

    # Tracé du graphique
    fig, ax = plt.subplots(figsize=(12, len(capacity_data) * 0.5 + 2))
    y_pos = np.arange(len(capacity_data))
    bar_width = 0.2
    scenarios = list(capacity_data.columns)

    for i, scenario in enumerate(scenarios):
        offset = (i - len(scenarios) / 2) * bar_width + bar_width / 2
        ax.barh(y_pos + offset, capacity_data[scenario], height=bar_width, label=scenario, color = colors[i] , alpha= alphas[i])


    ax.set_yticks(y_pos)
    ax.set_yticklabels(capacity_data.index)
    ax.set_xlabel("Installed capacity (GWh)")

    legend = ax.legend()
    legend.get_frame().set_linewidth(0)  # enlève le contour
    legend.get_frame().set_facecolor('none') 
    ax.grid(True, axis='x', linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def main(base_path):
    scenarios = {"Reference": "scénarios_finaux/10MGT", "Min GWP": "GWP_min_TS", "Min GWP corr": "GWP_min_TS_imp_4"}
    assets_paths, cost_paths, gwp_paths, res_paths = {}, {}, {}, {}

    for label, sub in scenarios.items():
        folder = os.path.join(base_path, sub, "output")
        assets_paths[label] = convert_txt_to_csv(os.path.join(folder, "assets.txt"))
        cost_paths[label] = convert_txt_to_csv(os.path.join(folder, "cost_breakdown.txt"))
        gwp_paths[label] = convert_txt_to_csv(os.path.join(folder, "gwp_breakdown.txt"))
        res_paths[label] = convert_txt_to_csv(os.path.join(folder, "resources_breakdown.txt"))

    techno_groups = {
         "Electricity": ["NUCLEAR", "CCGT", "CCGT_AMMONIA", "COAL_US", "COAL_IGCC",
                                "PV", "WIND_ONSHORE", "WIND_OFFSHORE", "HYDRO_RIVER", "GEOTHERMAL"],
         "Heat - HT": [
              "IND_COGEN_GAS", "IND_COGEN_WOOD", "IND_COGEN_WASTE", "IND_BOILER_GAS",
              "IND_BOILER_WOOD", "IND_BOILER_OIL", "IND_BOILER_COAL", "IND_BOILER_WASTE", "IND_DIRECT_ELEC"],
         "Heat - LT": [
             "DHN_HP_ELEC", "DHN_COGEN_GAS", "DHN_COGEN_WOOD", "DHN_COGEN_WASTE",
              "DHN_COGEN_WET_BIOMASS", "DHN_COGEN_BIO_HYDROLYSIS", "DHN_BOILER_GAS",
              "DHN_BOILER_WOOD", "DHN_BOILER_OIL", "DHN_DEEP_GEO", "DHN_SOLAR", "DEC_HP_ELEC", "DEC_THHP_GAS", "DEC_COGEN_GAS", "DEC_COGEN_OIL", "DEC_ADVCOGEN_GAS",
              "DEC_ADVCOGEN_H2", "DEC_BOILER_GAS", "DEC_BOILER_WOOD", "DEC_BOILER_OIL", "DEC_SOLAR", "DEC_DIRECT_ELEC"],
         "Mobility - Private": ["CAR_GASOLINE", "CAR_DIESEL", "CAR_NG",
                                     "CAR_METHANOL", "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"],
         "Mobility - Public": ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL",
                                    "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"],
         "Mobility - Freight": ["TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG",
                                     "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL", "TRUCK_METHANOL", "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"],
         "Storage": ["BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS", "TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS",
                                    "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY",  "TS_HIGH_TEMP",  "H2_STORAGE", "DIESEL_STORAGE",
                                  "GASOLINE_STORAGE", "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE", "CO2_STORAGE","GAS_STORAGE", "TS_DHN_SEASONAL"],
        "Conversion":  [
             "HABER_BOSCH", "SYN_METHANOLATION", "METHANE_TO_METHANOL", "BIOMASS_TO_METHANOL",
             "OIL_TO_HVC", "GAS_TO_HVC", "BIOMASS_TO_HVC", "METHANOL_TO_HVC",
             "AMMONIA_TO_H2", "PYROLYSIS_TO_LFO", "PYROLYSIS_TO_FUELS"]
    }

    print("\n=== TOP 3 TECHNOLOGIES PAR GROUPE ===")
    colors = ['#4B6C8B', '#4B6C8B', '#4B6C8B']
    alphas = [1.0, 0.7, 0.4]
    plot_top3_all_scenarios_grid(assets_paths, techno_groups, colors, alphas)

    file_paths = [
        
        "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/scénarios_finaux/10MGT/output/resources_breakdown.csv",
        "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/GWP_min_TS/output/resources_breakdown.csv",
        "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/GWP_min_TS_imp_4/output/resources_breakdown.csv"
    ]
    labels = ["REF", "MIN GWP", "MIN GWP corr"]
    #plot_resource_use_from_csv(file_paths, labels,seuil=10)
    print("\n=== VARIATIONS STOCKAGE ===")
    #plot_storage_variations_heatmap_and_barplot(assets_paths, techno_groups["Storage"], seuil=2)
    
    resources_to_plot = [
    "GASOLINE", "DIESEL", "BIOETHANOL", "BIODIESEL", "LFO", "GAS", "GAS_RE",
    "WOOD", "WET_BIOMASS", "COAL", "URANIUM", "WASTE", "H2", "H2_RE",
    "AMMONIA", "METHANOL", "AMMONIA_RE", "METHANOL_RE", "RES_WIND", "RES_SOLAR", "RES_HYDRO", "RES_GEO"]
    
    storage_technologies =  ["BATT_LI", "BEV_BATT",  "PHS","TS_DEC_HP_ELEC",  "TS_HIGH_TEMP", 
                              "METHANOL_STORAGE",  "TS_DHN_SEASONAL"]
    sto_pas_used = ["GAS_STORAGE","CO2_STORAGE", "GASOLINE_STORAGE", "LFO_STORAGE", "AMMONIA_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "TS_DEC_BOILER_GAS", "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS", "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS",
                               "TS_DEC_ADVCOGEN_H2",  "TS_DEC_DIRECT_ELEC", "PHEV_BATT" ]

    plot_resource_variation_heatmap(res_paths, resources_to_plot)
    plot_installed_storage_capacities(assets_paths, storage_technologies)


    
    
# === APPEL FINAL ===
main("C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies")


