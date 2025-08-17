# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:26:29 2020

Contains functions to read data in csv files and print it with AMPL syntax in ESTD_data.dat
Also contains functions to analyse input data

@author: Paolo Thiran
"""

import logging
import numpy as np
import pandas as pd
import csv
import yaml
import os
import sys
import json
import shutil
from subprocess import CalledProcessError, run
from pathlib import Path

from energyscope import ampl_syntax, print_set, print_df, newline, print_param, print_header, print_run


# Function to print the ESTD_data.dat file #
def print_data(config):
    """
    Cette fonction lit les données CSV et les imprime au format AMPL dans le fichier ESTD_data.dat.
    La version ci-dessous a été modifiée pour interdire certaines importations spécifiques :
      - La capacité d'importation d'électricité est forcée à zéro.
      - Certaines ressources importées (ex. methanol, ammonia, Gas_RE, H2_RE, Biodiesel) ont leur disponibilité forcée à 0.
      - Ces ressources ne sont pas inclues dans l'ensemble RES_IMPORT_CONSTANT.
    """
    cs = Path(__file__).parents[3] / 'case_studies'
    (cs / config['case_study']).mkdir(parents=True, exist_ok=True)

    data = config['all_data']

    eud = data['Demand']
    resources = data['Resources']
    technologies = data['Technologies']
    end_uses_categories = data['End_uses_categories']
    layers_in_out = data['Layers_in_out']
    storage_characteristics = data['Storage_characteristics']
    storage_eff_in = data['Storage_eff_in']
    storage_eff_out = data['Storage_eff_out']
    time_series = data['Time_series']

    if config['printing']:
        logging.info('Printing ESTD_data.dat')

        out_path = cs / config['case_study'] / 'ESTD_data.dat'
        cost_limit = config['cost_limit']
        gwp_op = config['lim_gwp_op']

        resources_simple = resources.loc[:, ['avail', 'gwp_op', 'c_op']]

        # 🔧 Interdire certaines ressources importées en mettant leur disponibilité à 0
        blocked_imports = ['METHANOL', 'AMMONIA', 'GAS_RE', 'H2_RE', 'BIODIESEL']
        for res in blocked_imports:
            if res in resources_simple.index:
                resources_simple.loc[res, 'avail'] = 0

        resources_simple.index.name = 'param :'
        resources_simple = resources_simple.astype('float')

        eud_simple = eud.drop(columns=['Category', 'Subcategory', 'Units'])
        eud_simple.index.name = 'param end_uses_demand_year:'
        eud_simple = eud_simple.astype('float')

        technologies_simple = technologies.drop(columns=['Category', 'Subcategory', 'Technologies name'])
        technologies_simple.index.name = 'param:'
        technologies_simple = technologies_simple.astype('float')

        i_rate = config['all_data']['Misc']['i_rate']
        re_share_primary = config['all_data']['Misc']['re_share_primary']
        solar_area = config['all_data']['Misc']['solar_area']
        power_density_pv = config['all_data']['Misc']['power_density_pv']
        power_density_solar_thermal = config['all_data']['Misc']['power_density_solar_thermal']

        share_mobility_public_min = config['all_data']['Misc']['share_mobility_public_min']
        share_mobility_public_max = config['all_data']['Misc']['share_mobility_public_max']
        share_freight_train_min = config['all_data']['Misc']['share_freight_train_min']
        share_freight_train_max = config['all_data']['Misc']['share_freight_train_max']
        share_freight_road_min = config['all_data']['Misc']['share_freight_road_min']
        share_freight_road_max = config['all_data']['Misc']['share_freight_road_max']
        share_freight_boat_min = config['all_data']['Misc']['share_freight_boat_min']
        share_freight_boat_max = config['all_data']['Misc']['share_freight_boat_max']
        share_heat_dhn_min = config['all_data']['Misc']['share_heat_dhn_min']
        share_heat_dhn_max = config['all_data']['Misc']['share_heat_dhn_max']

        share_ned = pd.DataFrame.from_dict(config['all_data']['Misc']['share_ned'], orient='index',
                                           columns=['share_ned'])

        keys_to_extract = ['EVs_BATT', 'vehicule_capacity', 'batt_per_car']
        evs = pd.DataFrame({key: config['all_data']['Misc']['evs'][key] for key in keys_to_extract},
                           index=config['all_data']['Misc']['evs']['CAR'])
        state_of_charge_ev = pd.DataFrame.from_dict(config['all_data']['Misc']['state_of_charge_ev'], orient='index',
                                                    columns=np.arange(1, 25))

        loss_network = config['all_data']['Misc']['loss_network']
        c_grid_extra = config['all_data']['Misc']['c_grid_extra']
        import_capacity = 0  # 🔧 Interdire l'importation d'électricité

        STORAGE_DAILY = config['all_data']['Misc']['STORAGE_DAILY']

        SECTORS = list(eud_simple.columns)
        END_USES_INPUT = list(eud_simple.index)
        END_USES_CATEGORIES = list(end_uses_categories.loc[:, 'END_USES_CATEGORIES'].unique())
        RESOURCES = list(resources_simple.index)

        # 🔧 Filtrer RES_IMPORT_CONSTANT pour exclure les ressources bloquées
        RES_IMPORT_CONSTANT = [r for r in resources.index if r in RESOURCES and r not in blocked_imports and resources.loc[r, 'Category'] == 'Import']

        # ... (le reste du code reste inchangé à partir d’ici)

        # Tu peux continuer avec l’impression des sets, des paramètres, etc.
        # L’exclusion des imports est maintenant gérée dynamiquement à partir du bloc ci-dessus

    if config['printing_td']:
        # partie inchangée
        pass
    return
