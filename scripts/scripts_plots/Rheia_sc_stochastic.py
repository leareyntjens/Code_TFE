# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 16:47:16 2025

@author: reynt
"""

#Run_90 : run de référence à 20%
#Run_92: stochastique à 50%
#Run_75%: stochastique à 75% 
#Run_imp_150%_lim: stochastique à 20% pour le scénario des ressources à 150% avec limite naturelle
import rheia.POST_PROCESS.post_process as rheia_pp
import matplotlib.pyplot as plt
import numpy as np

# ========================================
# Paramètres généraux
# ========================================
case = 'ENERGYSCOPE'
objective = 'gwp_op'
pol_order = 1
result_dirs = ['C:/Users/reynt/LMECA2675/rheia/RESULTS/ENERGYSCOPE/UQ/Run_90',  'C:/Users/reynt/LMECA2675/rheia/RESULTS/ENERGYSCOPE/UQ/Run_imp_150%_lim'] #'C:/Users/reynt/LMECA2675/rheia/RESULTS/ENERGYSCOPE/UQ/Run_92', 'C:/Users/reynt/LMECA2675/rheia/RESULTS/ENERGYSCOPE/UQ/Run_75%',
labels = ["Sans f_min"] #"Incertitude à 50%", "Incertitude à 75%"
colors = ["steelblue"] # "orange", "seagreen",

# ========================================
# Chargement des résultats
# ========================================
post_process = rheia_pp.PostProcessUQ(case, pol_order)
sobol_all = []
names_all = []
pdf_all = []
mean_std_all = []

for dir in result_dirs:
    names, sobol = post_process.get_sobol(dir, objective)
    sobol_all.append(sobol)
    names_all.append(names)
    x_pdf, y_pdf = post_process.get_pdf(dir, objective)
    pdf_all.append((x_pdf, y_pdf))
    mean_std_all.append(post_process.get_mean_std(dir, objective))

# ========================================
# SECTION 1 : Indices de Sobol (barres)
# ========================================
# ========================================
# SECTION 1 : Indices de Sobol (barres)
# ========================================
threshold = 0.1

# Filtrage des indices globalement non-significatifs
names_ref = names_all[0]
indices_to_keep = []

for i in range(len(names_ref)):
    keep = any(abs(sobol[i]) > threshold for sobol in sobol_all)
    if keep:
        indices_to_keep.append(i)

# Construction des noms filtrés
filtered_names = [names_ref[i] for i in indices_to_keep]
x = np.arange(len(filtered_names))
bar_width = 0.18

# Création du graphique
plt.figure(figsize=(5, 4))
for run_index, (sobol, label, color) in enumerate(zip(sobol_all, labels, colors)):
    sobol_filtered = [sobol[i] if abs(sobol[i]) > threshold else 0 for i in indices_to_keep]
    bars = plt.bar(x + run_index * bar_width, sobol_filtered, width=bar_width, label=label, color=color)

    # Affichage des valeurs
    for j, bar in enumerate(bars):
        if sobol_filtered[j] > 0:
            plt.text(
                bar.get_x(),
                bar.get_height() + 0.01,
                f"{sobol_filtered[j]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )
    # for j, bar in enumerate(bars):
    #     if sobol_filtered[j] > 0:
    #         plt.text(
    #             bar.get_x() + bar.get_width() / 2,
    #             bar.get_height() + 0.01,
    #             f"{sobol_filtered[j]:.2f}",
    #             ha="center",
    #             va="bottom",
    #             fontsize=8
    #         )

plt.xticks(x + bar_width * 1.5, filtered_names, rotation=45, ha="right")
plt.ylabel("Sobol' index")
#plt.title("Sobol indices comparés (filtrés)")
plt.legend()
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

# ========================================
# SECTION 2 : Courbes PDF + Moyennes & Intervalles
# ========================================
# ========================================
# SECTION 2 : Courbes PDF (séparées)
# ========================================

# # 1. Graphe avec Run 2 (orange) et Run 3 (vert)
# plt.figure(figsize=(8, 5))
# for i in [1, 2]:  # Run 2 et Run 3
#     x_pdf, y_pdf = pdf_all[i]
#     mean, std = mean_std_all[i]
#     plt.plot(x_pdf, y_pdf, label=labels[i], color=colors[i])
#     plt.axvline(mean, color=colors[i], linestyle="--", linewidth=1.5)
#     #plt.axvspan(mean - std, mean + std, color=colors[i], alpha=0.2)
# plt.xlabel("Total GWP [ktCO2_eq/year]")
# plt.ylabel("Probability Density [-]")
# #plt.title("PDF – Runs 2 & 3 superposés")
# #plt.legend()
# ax = plt.gca()
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# plt.tight_layout()
# plt.show()

# # 2. Graphe pour Run 1 (bleu)
# plt.figure(figsize=(8, 5))
# x_pdf, y_pdf = pdf_all[0]
# mean, std = mean_std_all[0]
# plt.plot(x_pdf, y_pdf, label=labels[0], color=colors[0])
# plt.axvline(mean, color=colors[0], linestyle="--", linewidth=1.5)
# #plt.axvspan(mean - std, mean + std, color=colors[0], alpha=0.2)
# plt.xlabel("Total GWP [ktCO2_eq/year]")
# plt.ylabel("Probability Density [-]")
# #plt.title("PDF – Run 1 seul")
# plt.xlim(4060,4085)
# #plt.legend()
# ax = plt.gca()
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# plt.tight_layout()
# plt.show()

# # 3. Graphe pour Run 4 (rouge)
# plt.figure(figsize=(8, 5))
# x_pdf, y_pdf = pdf_all[3]
# mean, std = mean_std_all[3]
# plt.plot(x_pdf, y_pdf, label=labels[3], color=colors[3])
# plt.axvline(mean, color=colors[3], linestyle="--", linewidth=1.5)
# #plt.axvspan(mean - std, mean + std, color=colors[3], alpha=0.2)
# plt.xlabel("Total GWP [ktCO2_eq/year]")
# plt.ylabel("Probability Density [-]")
# #plt.title("PDF – Run 4 seul")
# plt.xlim(5370, 5395)
# #plt.legend()
# ax = plt.gca()
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# plt.tight_layout()
# plt.show()
