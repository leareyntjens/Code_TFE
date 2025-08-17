import pandas as pd
import os
import energyscope as es
from pathlib import Path
import rheia
import rheia.CASES.ENERGYSCOPE.uq_estd as uq_estd

# TODO adapt to updates and test
# TODO write doc

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
    total_cost = es.get_total_cost(config)

    return total_cost
    

def transcript_uncertainties(uncer_params, config):

    # to fill the undefined uncertainty parameters
    #uncert_param = config['all_data']['Uncertainty_ranges']
    

    path_rheia = os.path.dirname(rheia.__file__)    
    
    df = pd.read_csv( os.path.join(path_rheia,'CASES','ENERGYSCOPE','design_space_ref.csv'), header=None)
    
    up = dict.fromkeys(df[0])

    rel_uncert_param = df
                
    for key in up:

        element_to_find = key
        result = rel_uncert_param.loc[rel_uncert_param[0] == element_to_find, 2].values[0]
        up[key] = result 
        
    # Set here the nominal value of the absolute uncertain parameters
    #up['f_max_nuc'] = config['all_data']['Technologies'].loc['NUCLEAR', 'f_max']

    # Extract the new value from the RHEIA sampling
    for key in uncer_params:
        up[key] = uncer_params[key]
        config['all_data']['Technologies'].loc['BATT_LI', 'gwp_constr'] *= (1. + up['gwp_batt_li'])
        config['all_data']['Technologies'].loc['BEV_BATT', 'gwp_constr'] *= (1. + up['gwp_bev_batt'])
        config['all_data']['Technologies'].loc['PHEV_BATT', 'gwp_constr'] *= (1. + up['gwp_phev_batt'])
        config['all_data']['Technologies'].loc['PHS', 'gwp_constr'] *= (1. + up['gwp_phs'])
        config['all_data']['Technologies'].loc['TS_DEC_DIRECT_ELEC', 'gwp_constr'] *= (1. + up['gwp_ts_dec_direct_elec'])
        config['all_data']['Technologies'].loc['TS_DEC_HP_ELEC', 'gwp_constr'] *= (1. + up['gwp_ts_dec_hp_elec'])
        config['all_data']['Technologies'].loc['TS_DEC_THHP_GAS', 'gwp_constr'] *= (1. + up['gwp_ts_dec_thhp_gas'])
        config['all_data']['Technologies'].loc['TS_DEC_COGEN_GAS', 'gwp_constr'] *= (1. + up['gwp_ts_dec_cogen_gas'])
        config['all_data']['Technologies'].loc['TS_DEC_COGEN_OIL', 'gwp_constr'] *= (1. + up['gwp_ts_dec_cogen_oil'])
        config['all_data']['Technologies'].loc['TS_DEC_ADVCOGEN_GAS', 'gwp_constr'] *= (1. + up['gwp_ts_dec_advcogen_gas'])
        config['all_data']['Technologies'].loc['TS_DEC_ADVCOGEN_H2', 'gwp_constr'] *= (1. + up['gwp_ts_dec_advcogen_h2'])
        config['all_data']['Technologies'].loc['TS_DEC_BOILER_GAS', 'gwp_constr'] *= (1. + up['gwp_ts_dec_boiler_gas'])
        config['all_data']['Technologies'].loc['TS_DEC_BOILER_WOOD', 'gwp_constr'] *= (1. + up['gwp_ts_dec_boiler_wood'])
        config['all_data']['Technologies'].loc['TS_DEC_BOILER_OIL', 'gwp_constr'] *= (1. + up['gwp_ts_dec_boiler_oil'])
        config['all_data']['Technologies'].loc['TS_DHN_DAILY', 'gwp_constr'] *= (1. + up['gwp_ts_dhn_daily'])
        config['all_data']['Technologies'].loc['TS_DHN_SEASONAL', 'gwp_constr'] *= (1. + up['gwp_ts_dhn_seasonal'])
        config['all_data']['Technologies'].loc['TS_HIGH_TEMP', 'gwp_constr'] *= (1. + up['gwp_ts_high_temp'])
        config['all_data']['Technologies'].loc['GAS_STORAGE', 'gwp_constr'] *= (1. + up['gwp_gas_storage'])
        config['all_data']['Technologies'].loc['H2_STORAGE', 'gwp_constr'] *= (1. + up['gwp_h2_storage'])
        config['all_data']['Technologies'].loc['DIESEL_STORAGE', 'gwp_constr'] *= (1. + up['gwp_diesel_storage'])
        config['all_data']['Technologies'].loc['GASOLINE_STORAGE', 'gwp_constr'] *= (1. + up['gwp_gasoline_storage'])
        config['all_data']['Technologies'].loc['LFO_STORAGE', 'gwp_constr'] *= (1. + up['gwp_lfo_storage'])
        config['all_data']['Technologies'].loc['AMMONIA_STORAGE', 'gwp_constr'] *= (1. + up['gwp_ammonia_storage'])
        config['all_data']['Technologies'].loc['METHANOL_STORAGE', 'gwp_constr'] *= (1. + up['gwp_methanol_storage'])
        config['all_data']['Technologies'].loc['CO2_STORAGE', 'gwp_constr'] *= (1. + up['gwp_co2_storage'])


    # Set here the absolute value for the absolute uncertain parameters. It's "=" and not "*="
    #config['all_data']['Technologies'].loc['NUCLEAR', 'f_max'] = up['f_max_nuc']

    return config['all_data']