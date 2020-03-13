# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:59:40 2020

@author: Callum Shaw

This code uses the density fields in the ridge frame of reference (produced by
'centering_topo.py') and finds the the velocity field using the buoyancy and
continuity equations. 
"""

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures


data_path='E:/batch 2/3echo/run8/transit/results/data.npz'
data = np.load(data_path, allow_pickle=True)



def topograghy()
