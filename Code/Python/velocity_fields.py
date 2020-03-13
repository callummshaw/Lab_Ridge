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


data_path='E:/batch 2/3echo/run8/transit/results/centre_data.npz'
data = np.load(data_path)

rho = data['centre_rho']

#remving nans at base
min_nan=np.min(np.sum(np.isnan(rho[0]),axis=0)) #summing the Nans in the vertical direction
offset = 15
rho_c = rho[:,:-(min_nan+offset),:]

def topograghy(rho, no_hills = 1):
    base = rho[0]
    y,x = base.shape
    domain = np.arange(x)
    
    if no_hills == 1:
        nan_array = np.sum(np.isnan(base),axis=0)
        max_amp = np.max(nan_array)
        max_loc = np.argmax(nan_array)
        
        h_m = max_amp//2
        h_m_array  = base[h_m,:]
        h_m_w = np.sum(h_m_array)//2
        
    if no_hills == 2: