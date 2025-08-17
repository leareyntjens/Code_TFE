import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# Chemin de base
base_path = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies/GWP op ND/IMPORTS_ND_0_0{}"
gwp_values = [round(i * 0.01, 2) for i in range(10)]

# Technologies de mobilité privée
mobility_technologies = [
    "CAR_GASOLINE", "CAR_HEV", "CAR_FUEL_CELL"
]
# mobility_technologies = [
#     "CAR_GASOLINE", "CAR_DIESEL", "CAR_NG", "CAR_METHANOL",
#     "CAR_HEV", "CAR_PHEV", "CAR_BEV", "CAR_FUEL_CELL"
# ]

couleur_bleu_froid = "#4B6C8B"
couleur_gris_foncé = "#666666"
couleur_bleu_pâle = "#A0B3C3"
colors = [couleur_bleu_froid, "darkgrey", couleur_bleu_pâle]
selected_techs = [ "CAR_FUEL_CELL", "CAR_HEV", "CAR_GASOLINE"]
color_map = dict(zip(selected_techs, colors))

# Initialiser les résultats
results = {tech: [] for tech in mobility_technologies}

# Parcourir les dossiers et lire les fichiers
for i, gwp in enumerate(gwp_values):
    folder = base_path.format(i)
    file_path = os.path.join(folder, "output", "assets.txt")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep="\t", comment="#")  # Lecture fichier TSV

        for tech in mobility_technologies:
            row = df[df['TECHNOLOGIES'].str.strip() == tech]
            if not row.empty:
                results[tech].append(float(row[' f'].values[0]))
            else:
                results[tech].append(0.0)
    else:
        for tech in mobility_technologies:
            results[tech].append(None)

# Supprimer les technologies avec f < 1 dans tous les cas
filtered_results = {tech: vals for tech, vals in results.items() if any(v is not None and v >= 1.0 for v in vals)}

# Nettoyage des données : remplacer None par 0.0 avant calculs d'intersections
gasoline = [0.0 if v is None else v for v in results["CAR_GASOLINE"]]
hev = [0.0 if v is None else v for v in results["CAR_HEV"]]
fc = [0.0 if v is None else v for v in results["CAR_FUEL_CELL"]]

# Tracer à nouveau
plt.figure(figsize=(10, 6))
for tech, vals in filtered_results.items():
    if tech in color_map:
        plt.plot(gwp_values, vals, marker='o', label=tech, color=color_map[tech])

# Intersection CAR_GASOLINE vs CAR_HEV
for i in range(1, len(gwp_values)):
    if gasoline[i - 1] > hev[i - 1] and gasoline[i] <= hev[i]:
        x1, x2 = gwp_values[i - 1], gwp_values[i]
        y1_diff = gasoline[i - 1] - hev[i - 1]
        y2_diff = gasoline[i] - hev[i]
        slope = (y2_diff - y1_diff) / (x2 - x1)
        x_cross = x1 - y1_diff / slope if slope != 0 else x1
        y_cross = gasoline[i - 1] + (x_cross - x1) * (gasoline[i] - gasoline[i - 1]) / (x2 - x1)
        plt.scatter(x_cross, y_cross, color="palevioletred", zorder=5, marker = 's',label = 'Tipping point')
        break

# Intersection CAR_HEV vs CAR_FUEL_CELL
for i in range(1, len(gwp_values)):
    if hev[i - 1] > fc[i - 1] and hev[i] <= fc[i]:
        x1, x2 = gwp_values[i - 1], gwp_values[i]
        y1_diff = hev[i - 1] - fc[i - 1]
        y2_diff = hev[i] - fc[i]
        slope = (y2_diff - y1_diff) / (x2 - x1)
        x_cross = x1 - y1_diff / slope if slope != 0 else x1
        y_cross = hev[i - 1] + (x_cross - x1) * (hev[i] - hev[i - 1]) / (x2 - x1)
        plt.scatter(x_cross, y_cross, color="palevioletred", zorder=5,marker='s')
        break

# Mise en forme
plt.xlabel("Operational GWP of non-dominant gases in kg$CO_2$eq/kWh")
plt.ylabel("Installed capacities   in Mpkm/h ")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend = plt.legend()
legend.get_frame().set_linewidth(0)
legend.get_frame().set_facecolor('none')
plt.tight_layout()
plt.show()
