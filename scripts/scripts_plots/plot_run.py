# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:21:53 2025
@author: reynt
Ce script charge la configuration et les outputs déjà générés depuis le dossier spécifié,
puis affiche les graphiques et le Sankey.
"""

import os
from pathlib import Path
import energyscope as es
import webbrowser
import pathlib


if __name__ == '__main__':
    # === Chemins ===
    working_dir = os.getcwd()
    cs_path = Path("C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies")
    case_study_name = "Sans_lim_5"
   

    # === Chargement de la config YAML ===
    config = es.load_config(config_fn='config_ref.yaml')
    config['Working_directory'] = working_dir
    config['cs_path'] = cs_path
    config['case_study'] = case_study_name

    # === Chargement des données du scénario ===
    es.import_data(config)

    # === Affichage du Sankey ===
    
    if config.get('print_sankey', True):  # tu peux forcer à True ici
        sankey_path = cs_path / case_study_name / 'output' / 'sankey'
        es.drawSankey(path=sankey_path)
    
    # === Lecture des outputs déjà générés ===
    outputs = es.read_outputs(
        case_study=config['cs_path'] / config['case_study'],
        hourly_data=False,
        layers=['layer_ELECTRICITY', 'layer_HEAT_LOW_T_DECEN']
    )

    # === Graphique des ressources primaires utilisées ===
    fig2, ax2 = es.plot_barh(outputs['resources_breakdown'][['Used']],
                             title='Primary energy [GWh/y]')

    # === Graphique des actifs électriques ===
    elec_assets = es.get_assets_l(layer='ELECTRICITY',
                                  eff_tech=config['all_data']['Layers_in_out'],
                                  assets=outputs['assets'])
    fig3, ax3 = es.plot_barh(elec_assets[['f']],
                             title='Electricity assets [GW_e]',
                             x_label='Installed capacity [GW_e]')

    # === Graphique de la couche ELECTRICITY (facultatif)
    # fig4 = es.plot_layer_elec_td(outputs['layer_ELECTRICITY'])

    # === Graphique horaire de la chaleur décentralisée (facultatif)
    # fig5, ax5 = es.hourly_plot(plotdata=outputs['layer_HEAT_LOW_T_DECEN'], nbr_tds=12)
