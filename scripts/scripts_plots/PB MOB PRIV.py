import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re
plt.rcParams.update({'font.size': 18})
# === PARAMÈTRES À ADAPTER ===
base_dir = r"C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/SD_MOB"
technos_mobility_private = [
    "CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL",
    "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"
]

# Famille -> techno représentative pour lire le GWP dans ESTD_data.dat
famille_to_techno = {
    "electrique": "CAR_BEV",
    "hybride": "CAR_HEV",
    "gaz": "CAR_GASOLINE",
    "fuel_cell": "CAR_FUEL_CELL"
}
technos_storage = [
    "BATT_LI", "BEV_BATT", "PHEV_BATT", "PHS",
    "TS_DEC_DIRECT_ELEC", "TS_DEC_HP_ELEC", "TS_DEC_THHP_GAS", "TS_DEC_COGEN_GAS",
    "TS_DEC_COGEN_OIL", "TS_DEC_ADVCOGEN_GAS", "TS_DEC_ADVCOGEN_H2", "TS_DEC_BOILER_GAS",
    "TS_DEC_BOILER_WOOD", "TS_DEC_BOILER_OIL", "TS_DHN_DAILY", "TS_DHN_SEASONAL", "TS_HIGH_TEMP",
    "GAS_STORAGE", "H2_STORAGE", "DIESEL_STORAGE", "GASOLINE_STORAGE",
    "LFO_STORAGE", "AMMONIA_STORAGE", "METHANOL_STORAGE"
]

technos_mobility_public = ["TRAMWAY_TROLLEY", "BUS_COACH_DIESEL",
                           "BUS_COACH_HYDIESEL", "BUS_COACH_CNG_STOICH", "BUS_COACH_FC_HYBRIDH2", "TRAIN_PUB"]

technos_mobility_freight = ["TRAIN_FREIGHT", "BOAT_FREIGHT_DIESEL", "BOAT_FREIGHT_NG",
                            "BOAT_FREIGHT_METHANOL", "TRUCK_DIESEL", "TRUCK_METHANOL", "TRUCK_FUEL_CELL", "TRUCK_ELEC", "TRUCK_NG"]


def read_assets_txt(path):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split('\t')
        data = pd.read_csv(StringIO(''.join(lines[2:])), sep='\t', header=None)
        data.columns = header
        data.set_index('TECHNOLOGIES', inplace=True)
        return data
    except Exception:
        return None

def read_gwp_from_dat_file(dat_path, techno_name):
    try:
        with open(dat_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Fichier introuvable : {dat_path}")
        return None

    start = False
    for line in lines:
        if line.strip().startswith("param :") and "gwp_constr" in line:
            start = True
            continue
        if start:
            if line.strip() == ';':
                break
            if line.strip().startswith(techno_name):
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 4:
                    try:
                        return float(parts[3])
                    except ValueError:
                        return None
    print(f"⚠️ Techno {techno_name} non trouvée dans {dat_path}")
    return None

def extract_famille_from_folder(folder_name):
    match = re.search(r"MOB_(\w+)_gwp", folder_name)
    if match:
        return match.group(1)
    return None

def build_mobility_dataframe(base_dir, technos, export_csv=True):
    records = []

    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue

        famille = extract_famille_from_folder(folder)
        if famille is None or famille not in famille_to_techno:
            print(f"⚠️ Dossier ignoré : {folder}")
            continue

        techno_ref = famille_to_techno[famille]
        dat_path = os.path.join(full_path, "ESTD_data.dat")
        gwp_val = read_gwp_from_dat_file(dat_path, techno_ref)
        if gwp_val is None:
            continue

        assets_path = os.path.join(full_path, "output", "assets.txt")
        df_assets = read_assets_txt(assets_path)
        if df_assets is None:
            continue

        for tech in technos:
            capacity = df_assets.at[tech, ' f'] if tech in df_assets.index else 0.0
            records.append({
                "folder": folder,
                "famille": famille,
                "technology": tech,
                "f": capacity,
                "gwp_constr": gwp_val
            })

    df = pd.DataFrame(records)
    if df.empty:
        print("❌ Aucun résultat généré.")
        return df

    df.sort_values(by=["famille", "technology", "gwp_constr"], inplace=True)

    if export_csv:
        df.to_csv("points_data_mobility.csv", index=False)
        print("✅ CSV exporté : points_data_mobility.csv")

    return df

def plot_mobility_impact(df, familles, technos_a_surveiller):
    os.makedirs("plots_mobility", exist_ok=True)

    for fam in familles:
        df_mod = df[df["famille"] == fam]
        df_mod = df_mod[df_mod["technology"].isin(technos_a_surveiller)]

        if df_mod.empty:
            continue

        # Filtrer les technologies toujours < 1
        techs_significatives = []
        for tech in df_mod["technology"].unique():
            if df_mod[df_mod["technology"] == tech]["f"].max() >= 1:
                techs_significatives.append(tech)

        df_mod = df_mod[df_mod["technology"].isin(techs_significatives)]

        if df_mod.empty:
            continue

        # Tracer le graphique
        plt.figure(figsize=(13, 6))
        sns.lineplot(data=df_mod, x="gwp_constr", y="f", hue="technology", marker="o", ci=None)
        plt.xlabel(f"GWP construction for {fam} mobility technologies [kgCO2eq/Mpkm/h]")
        plt.ylabel("Installed capacities [Mt$\cdot$km]")
        ax2 = plt.gca()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.legend( # Rapproche légèrement la légende du graphe
    loc='best',
    bbox_to_anchor=(1.02, 1),
    ncol=1                    # Affiche la légende en deux colonnes
)
        plt.tight_layout()
        plt.show()

        

def build_mobility_dataframe_pivoted(base_dir, techs_a_surveiller, export_csv=True):
    """
    Construit un DataFrame pivoté avec :
    - dossier, famille modifiée, GWP,
    - f de la techno modifiée (famille → techno représentative),
    - f des autres techno surveillées (techs_a_surveiller).
    """
    famille_to_techno = {
        "electrique": "CAR_BEV",
        "hybride": "CAR_HEV",
        "gaz": "CAR_GASOLINE",
        "fuel_cell": "CAR_FUEL_CELL"
    }

    records = []

    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if not os.path.isdir(full_path):
            continue

        # Extraire la famille depuis le nom du dossier
        match = re.search(r"MOB_(\w+)_gwp", folder)
        if not match:
            print(f"⚠️ Dossier ignoré (pas de famille détectée) : {folder}")
            continue
        famille = match.group(1)

        if famille not in famille_to_techno:
            print(f"⚠️ Famille non reconnue : {famille}")
            continue

        techno_modifiee = famille_to_techno[famille]

        dat_path = os.path.join(full_path, "ESTD_data.dat")
        gwp_val = read_gwp_from_dat_file(dat_path, techno_modifiee)
        if gwp_val is None:
            continue

        assets_path = os.path.join(full_path, "output", "assets.txt")
        df_assets = read_assets_txt(assets_path)
        if df_assets is None:
            continue

        row = {
            "folder": folder,
            "techno_modifiee": famille,  # ici on garde la famille comme identifiant
            "gwp_constr": gwp_val,
        }

        # f de la techno modifiée (famille → techno)
        row[f"f_{techno_modifiee}"] = df_assets.at[techno_modifiee, ' f'] if techno_modifiee in df_assets.index else 0.0

        # f des autres techno surveillées
        for tech in techs_a_surveiller:
            row[f"f_{tech}"] = df_assets.at[tech, ' f'] if tech in df_assets.index else 0.0

        records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        print("❌ Aucun scénario valide trouvé.")
        return df

    # 🔁 Tri
    df.sort_values(by=["techno_modifiee", "gwp_constr"], inplace=True)

    # 🧠 Tri des colonnes f_... par moyenne décroissante
    techno_cols = [col for col in df.columns if col.startswith("f_")]
    techno_mod_col_names = df["techno_modifiee"].apply(lambda fam: f"f_{famille_to_techno[fam]}")
    unique_mod_cols = techno_mod_col_names.unique()
    other_cols = sorted(set(techno_cols) - set(unique_mod_cols))

    mean_by_col = df[other_cols].mean().sort_values(ascending=False)
    sorted_other_cols = list(mean_by_col.index)

    fixed_cols = ["folder", "techno_modifiee", "gwp_constr"]
    final_cols = fixed_cols + sorted(set(unique_mod_cols)) + sorted_other_cols
    df = df[final_cols]

    if export_csv:
        base_name = os.path.basename(base_dir.rstrip("\\/"))
        output_csv = f"points_data_pivoted_{base_name}_mob_fret.csv"
        df.to_csv(output_csv, index=False)
        print(f"✅ CSV pivoté sauvegardé : {output_csv}")

    return df



def plot_mobility_impact_subfigures(df, familles, technos_a_surveiller):
    os.makedirs("plots_mobility", exist_ok=True)

    familles = [fam for fam in familles if fam in df["famille"].unique()]
    familles = sorted(familles)
    n = len(familles)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 6* nrows), squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.1)

    # Trouver les bornes Y communes
    y_min, y_max = float('inf'), float('-inf')
    for fam in familles:
        df_mod = df[df["famille"] == fam]
        df_mod = df_mod[df_mod["technology"].isin(technos_a_surveiller)]

        techs_significatives = [
            tech for tech in df_mod["technology"].unique()
            if df_mod[df_mod["technology"] == tech]["f"].max() >= 1
        ]
        df_mod = df_mod[df_mod["technology"].isin(techs_significatives)]
        if df_mod.empty:
            continue

        y_min = min(y_min, df_mod["f"].min())
        y_max = max(y_max, df_mod["f"].max())

    handles, labels = None, None  # Pour la légende globale

    for idx, fam in enumerate(familles):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        df_mod = df[df["famille"] == fam]
        df_mod = df_mod[df_mod["technology"].isin(technos_a_surveiller)]

        techs_significatives = [
            tech for tech in df_mod["technology"].unique()
            if df_mod[df_mod["technology"] == tech]["f"].max() >= 1
        ]
        df_mod = df_mod[df_mod["technology"].isin(techs_significatives)]
        if df_mod.empty:
            ax.set_visible(False)
            continue

        plot = sns.lineplot(data=df_mod, x="gwp_constr", y="f", hue="technology", marker="o", ci=None, ax=ax)
        if handles is None and labels is None:
            handles, labels = plot.get_legend_handles_labels()
            ref_line = ax.axvline(x=388, color='gray', linestyle='--', linewidth=2, label="Tipping point")
            handles.append(ref_line)
            labels.append("Tipping point")

        ax.set_ylim(y_min, y_max)

        if col == 0:
            ax.set_ylabel("Installed capacities [Mp·km]")
            ax.set_xlabel("GWP construction of \n biofuel car [kgCO2eq/Mpkm/h]")
            ax.get_legend().remove()


        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis='y', left=False, labelleft=False)
        if col == 1:
            ax.set_xlabel("GWP construction of \n hybrid car [kgCO2eq/Mpkm/h]")
            ax.spines['left'].set_visible(False)
            ref_line2 = ax.axvline(x=533, color='gray', linestyle='--', linewidth=2)
            ax.get_legend().remove()



        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Légende globale au-dessus
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.87])
  # laisse de la place en haut pour la légende
    plt.show()

# === EXÉCUTION ===
df_mob = build_mobility_dataframe(base_dir, technos_mobility_public)
# if not df_mob.empty:
#     plot_mobility_impact(df_mob, df_mob["famille"].unique(), technos_mobility_freight)



if not df_mob.empty:
    plot_mobility_impact_subfigures(df_mob, df_mob["famille"].unique(), technos_mobility_public)

#df_pivot_mob = build_mobility_dataframe_pivoted(base_dir, technos_mobility_freight)
