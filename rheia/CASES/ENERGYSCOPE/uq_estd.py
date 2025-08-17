import pandas as pd
import os
import energyscope as es
from pathlib import Path
import rheia
import rheia.CASES.ENERGYSCOPE.uq_estd as uq_estd


# TODO adapt to updates and test
# TODO write doc
def get_gwp_constr (config):
    """Read the cost breakdown and computes the total cost

        Parameters
        ----------
        config: dictionnary
        Dictionnary defining the case study

        case: str
        Set to 'deter' for determinist run and 'uq' for uncertainty quatification run

        Returns
        -------
        Total annualised cost of the system (float) [M€/y]
        """
    two_up = Path(__file__).parents[2]
    file_path = os.path.join("C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py", 
                      "case_studies", config["case_study"], "output", "gwp_breakdown.txt")
    print(f"[DEBUG] Searching for 'gwp_breakdown.txt' at: {file_path}")
    if not os.path.exists(file_path):
        print("[ERROR] File not found!")
        raise FileNotFoundError(f"File not found: {file_path}")

    gwp_constr = pd.read_csv(file_path, index_col=0, sep='\t')
   
    return gwp_constr.sum().sum()


   
def run_ESTD_UQ(sample):


    path_rheia = os.path.dirname(rheia.__file__)    

    # loading the config file into a python dictionnary
    
    # SPECIFY PATH WHERE THE CONFIG_REF_UQ FILE WILL BE
    config = es.load_config(config_fn=os.path.join(path_rheia,'CASES','ENERGYSCOPE','config_ref_UQ.yaml'))
    config['Working_directory'] = os.getcwd() # keeping current working directory into config

    # Reading the data of the csv
    es.import_data(config)

    s = sample[0]
    sample_index = s[0]
    sample_dict = s[1]

    name = sample[1]

    config['case_study'] = name + '/Run_{}'.format(sample_index)

    config['print_sankey'] = False #For the UQ analysis, no need to print the Sankey

    # # Test to update uncertain parameters
    uncer_params = sample_dict

    config['all_data'] =  uq_estd.transcript_uncertainties(uncer_params,config)

    # Printing the .dat files for the optimisation problem
    es.print_data(config)

    # Running EnergyScope
    es.run_es(config)

    # Example to get total cost
    #total_cost = es.get_total_cost(config)
    total_gwp_constr = get_gwp_constr(config)
    #return total_cost
    return total_gwp_constr
    


def transcript_uncertainties(uncer_params, config):
    """Applique les incertitudes de stochastic_space.csv aux paramètres de design_space.csv"""

    path_rheia = os.path.dirname(rheia.__file__)    
    
    # Charger les fichiers design_space et stochastic_space
    design_space_path = os.path.join(path_rheia, 'CASES', 'ENERGYSCOPE', 'design_space.csv')
    stochastic_space_path = os.path.join(path_rheia, 'CASES', 'ENERGYSCOPE', 'stochastic_space.csv')

    design_space_df = pd.read_csv(design_space_path)
    stochastic_space_df = pd.read_csv(stochastic_space_path)

    # Extraire les valeurs d'incertitude
    up = dict(zip(stochastic_space_df.iloc[:, 0], stochastic_space_df.iloc[:, 3]))  # Associe le paramètre à sa valeur

    # Mettre à jour les incertitudes à partir de Rheia
    for key in uncer_params:
        up[key] = uncer_params[key]
        print(up)

    # Sauvegarde des valeurs initiales pour éviter la multiplication cumulative
    if "gwp_constr_original" not in config:
        config["gwp_constr_original"] = config['all_data']['Technologies']['gwp_constr'].copy()
        print("Valeurs originales de gwp_constr :")
        print(config['gwp_constr_original'])

    # Appliquer les incertitudes une seule fois
    for tech in config['all_data']['Technologies'].index:
        gwp_key = 'gwp_' + tech.lower()
        if gwp_key in up:
            print(f"Avant modification : {tech} gwp_constr = {config['gwp_constr_original'][tech]}")
            print(f"up['{gwp_key}'] = {up[gwp_key]}")

            # Appliquer une seule fois sans cumul
            config['all_data']['Technologies'].loc[tech, 'gwp_constr'] = up[gwp_key]


            # # Limite des valeurs pour éviter des explosions
            # config['all_data']['Technologies'].loc[tech, 'gwp_constr'] = min(
            #     max(config['all_data']['Technologies'].loc[tech, 'gwp_constr'], 0), 1e6
            # )

    return config['all_data']

# def transcript_uncertainties(uncer_params, config):

#     # to fill the undefined uncertainty parameters
#     #uncert_param = config['all_data']['Uncertainty_ranges']
    

#     path_rheia = os.path.dirname(rheia.__file__)    
    
#     df = pd.read_csv( os.path.join(path_rheia,'CASES','ENERGYSCOPE','stochastic_space.csv'), header=None)
    
#     up = dict.fromkeys(df[0])

#     rel_uncert_param = df
                
#     for key in up:

#         element_to_find = key
#         result = rel_uncert_param.loc[rel_uncert_param[0] == element_to_find, 3].values[0]
#         up[key] = result 
        
#     # Set here the nominal value of the absolute uncertain parameters
#     #up['f_max_nuc'] = config['all_data']['Technologies'].loc['NUCLEAR', 'f_max']

#     # Extract the new value from the RHEIA sampling
#     for key in uncer_params:
#         up[key] = uncer_params[key]
#         print(f"Avant modification : PHS gwp_constr = {config['all_data']['Technologies'].loc['PHS', 'gwp_constr']}")
#         print(f"up['gwp_phs'] = {up['gwp_phs']}")
#         config['all_data']['Technologies'].loc['BATT_LI', 'gwp_constr'] *= (1. + up['gwp_batt_li'])
#         config['all_data']['Technologies'].loc['BEV_BATT', 'gwp_constr'] *= (1. + up['gwp_bev_batt'])
#         config['all_data']['Technologies'].loc['PHEV_BATT', 'gwp_constr'] *= (1. + up['gwp_phev_batt'])
#         config['all_data']['Technologies'].loc['PHS', 'gwp_constr'] *= (1. + up['gwp_phs'])
#         config['all_data']['Technologies'].loc['TS_DEC_DIRECT_ELEC', 'gwp_constr'] *= (1. + up['gwp_ts_dec_direct_elec'])
#         config['all_data']['Technologies'].loc['TS_DEC_HP_ELEC', 'gwp_constr'] *= (1. + up['gwp_ts_dec_hp_elec'])
#         config['all_data']['Technologies'].loc['TS_DEC_THHP_GAS', 'gwp_constr'] *= (1. + up['gwp_ts_dec_thhp_gas'])
#         config['all_data']['Technologies'].loc['TS_DEC_COGEN_GAS', 'gwp_constr'] *= (1. + up['gwp_ts_dec_cogen_gas'])
#         config['all_data']['Technologies'].loc['TS_DEC_COGEN_OIL', 'gwp_constr'] *= (1. + up['gwp_ts_dec_cogen_oil'])
#         config['all_data']['Technologies'].loc['TS_DEC_ADVCOGEN_GAS', 'gwp_constr'] *= (1. + up['gwp_ts_dec_advcogen_gas'])
#         config['all_data']['Technologies'].loc['TS_DEC_ADVCOGEN_H2', 'gwp_constr'] *= (1. + up['gwp_ts_dec_advcogen_h2'])
#         config['all_data']['Technologies'].loc['TS_DEC_BOILER_GAS', 'gwp_constr'] *= (1. + up['gwp_ts_dec_boiler_gas'])
#         config['all_data']['Technologies'].loc['TS_DEC_BOILER_WOOD', 'gwp_constr'] *= (1. + up['gwp_ts_dec_boiler_wood'])
#         config['all_data']['Technologies'].loc['TS_DEC_BOILER_OIL', 'gwp_constr'] *= (1. + up['gwp_ts_dec_boiler_oil'])
#         config['all_data']['Technologies'].loc['TS_DHN_DAILY', 'gwp_constr'] *= (1. + up['gwp_ts_dhn_daily'])
#         config['all_data']['Technologies'].loc['TS_DHN_SEASONAL', 'gwp_constr'] *= (1. + up['gwp_ts_dhn_seasonal'])
#         config['all_data']['Technologies'].loc['TS_HIGH_TEMP', 'gwp_constr'] *= (1. + up['gwp_ts_high_temp'])
#         config['all_data']['Technologies'].loc['GAS_STORAGE', 'gwp_constr'] *= (1. + up['gwp_gas_storage'])
#         config['all_data']['Technologies'].loc['H2_STORAGE', 'gwp_constr'] *= (1. + up['gwp_h2_storage'])
#         config['all_data']['Technologies'].loc['DIESEL_STORAGE', 'gwp_constr'] *= (1. + up['gwp_diesel_storage'])
#         config['all_data']['Technologies'].loc['GASOLINE_STORAGE', 'gwp_constr'] *= (1. + up['gwp_gasoline_storage'])
#         config['all_data']['Technologies'].loc['LFO_STORAGE', 'gwp_constr'] *= (1. + up['gwp_lfo_storage'])
#         config['all_data']['Technologies'].loc['AMMONIA_STORAGE', 'gwp_constr'] *= (1. + up['gwp_ammonia_storage'])
#         config['all_data']['Technologies'].loc['METHANOL_STORAGE', 'gwp_constr'] *= (1. + up['gwp_methanol_storage'])
#         config['all_data']['Technologies'].loc['CO2_STORAGE', 'gwp_constr'] *= (1. + up['gwp_co2_storage'])


#     # Set here the absolute value for the absolute uncertain parameters. It's "=" and not "*="
#     #config['all_data']['Technologies'].loc['NUCLEAR', 'f_max'] = up['f_max_nuc']

#     return config['all_data']