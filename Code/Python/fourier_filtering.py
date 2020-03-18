# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:06:42 2020

@author: Callum Shaw
"""

import numpy as np
import analysis_functions as af

data_path = 'D:/batch 2/3echo/run8/transit/results/transformation_result.npz'
data = np.load(data_path)

t_data = data['result']
t_data = np.dstack(t_data)
#t_data = t_data[:,1:-1,1:-1] #removing boundary nans
