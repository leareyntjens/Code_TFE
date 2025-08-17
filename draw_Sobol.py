# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:45:40 2024

@author: reynt
"""

# Importations
import rheia.POST_PROCESS.post_process as rheia_pp
import matplotlib.pyplot as plt
import numpy as np

# ========================================
# Initialisation des paramètres de base
# ========================================
case = 'ENERGYSCOPE'
objective = 'gwp_op'
couleur_bleu_froid = "#4B6C8B"
couleur_gris_foncé = "#666666"
couleur_bleu_pâle = "#A0B3C3"

# Création d'une instance PostProcessUQ
pol_order = 1
my_post_process_uq = rheia_pp.PostProcessUQ(case, pol_order)

# Dossiers de résultats
result_dir_1 = 'SA_MOB_PRIV_20%_final'
result_dir_2 = 'SA_TS_20%_imp_fin'



# Pour chaque dossier :
for i, result_dir in enumerate([result_dir_1, result_dir_2], start=1):

    names, sobol = my_post_process_uq.get_sobol(result_dir, objective)
    mean, std = my_post_process_uq.get_mean_std(result_dir, objective)
    x_pdf, y_pdf = my_post_process_uq.get_pdf(result_dir, objective)

    # === SECTION 1 : Graphique des indices de Sobol ===
    threshold = 0.01
    filtered_names = [name for name, value in zip(names, sobol) if abs(value) > threshold]
    filtered_sobol = [value for value in sobol if abs(value) > threshold]

    plt.figure(figsize=(10, 3))
    y_pos = np.arange(len(filtered_names))
    bar_height = 0.35
    bars = plt.barh(y_pos, filtered_sobol, height=bar_height, color = couleur_bleu_froid)

    plt.yticks(y_pos, filtered_names, fontsize=16)
    for j, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{filtered_sobol[j]:.2f}", va="center", fontsize=14)

    plt.title("Sobol index of construction GWPs for private mobility technologies", fontsize=16)
    plt.xticks([])  # Supprime les ticks de l'axe des x
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"Sobol_{result_dir}.svg", format="svg", bbox_inches="tight")
    plt.show()

    # === SECTION 2 : PDF des résultats ===
    plt.figure()
    plt.plot(x_pdf, y_pdf, label=f"{result_dir}")
    plt.axvline(mean, color="firebrick", linestyle="--", linewidth=2)

    plt.xlabel("Total GWP for construction [ktCO2_eq/year]", fontsize=12)
    plt.ylabel("Probability Density [-]", fontsize=12)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"PDF_{result_dir}.svg", format="svg", bbox_inches="tight")
    plt.show()
    
    
# Dossiers de résultats
result_dirs = [
    ('SA_TS_20%_imp_fin', '20% uncertainty '),
    ('SA_TS_75%_imp_fin', '75% uncertainty')
]

# Couleurs (modifiable si tu veux des codes précis)
colors = [couleur_bleu_froid, couleur_bleu_pâle]

plt.figure(figsize=(10, 6))

# Tracer tous les PDF avec leur moyenne
for i, (result_dir, label) in enumerate(result_dirs):
    x_pdf, y_pdf = my_post_process_uq.get_pdf(result_dir, objective)
    mean, std = my_post_process_uq.get_mean_std(result_dir, objective)

    plt.plot(x_pdf, y_pdf, label=label, color=colors[i], linewidth=2)
    plt.axvline(mean, color=colors[i], linestyle='--', linewidth=2)

# Mise en forme du graphe
plt.xlabel("Total global warming potential [ktCO$_2$ eq/year]", fontsize=16)
plt.ylabel("Probability Density [-]", fontsize=16)

ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

legend = plt.legend(fontsize=14)
legend.get_frame().set_linewidth(0)  # enlève le contour
legend.get_frame().set_facecolor('none')
plt.tight_layout()
plt.savefig("PDF_comparison_all_scenarios.svg", format="svg", bbox_inches="tight")
plt.show()
