# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:39:50 2024

@author: reynt
"""

import rheia.UQ.uncertainty_quantification as rheia_uq
import pandas as pd

dict_uq = {'case':                  'ENERGYSCOPE',
           'pol order':             1,
           #'objective names':       ['cost', 'gwp_op'],
           'objective names':       ['gwp_op'],
           'objective of interest': 'gwp_op',
           'draw pdf cdf':          [True, 1e5],
           'results dir':           'SA_MOB_PRIV_20%_final'
           }

if __name__ == '__main__':
    
    rheia_uq.run_uq(dict_uq, design_space = 'design_space.csv')
    
