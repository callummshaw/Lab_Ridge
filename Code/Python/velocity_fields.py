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

#data_path='E:/batch 2/3echo/run8/transit/results/centre_data.npz'
#data = np.load(data_path)

#rho = data['centre_rho']

#remving nans at base
#min_nan=np.min(np.sum(np.isnan(rho[0]),axis=0)) #summing the Nans in the vertical direction
#offset = 15
#rho_c = rho[:,:-(min_nan+offset),:]

def topograghy_mask(rho, no_hills=1,lensing=33 ):
    t,y,x = rho.shape
    base = rho[t//2]
    domain = np.arange(x)
    
    if no_hills == 1:
        
        height_increase=20 #pixels we want to increase the topo height by (otherwise cuts off sides of the top)
        
        nan_array = np.sum(np.isnan(base),axis=0)
        max_amp = np.max(nan_array)
        nan_array = np.float32(nan_array) #need to convert from int to float
        nan_array[nan_array<max_amp-5]=np.nan #removing everything but the top of the hill
        
        nan_count=nan_array*0+1
        nan_count=nan_count*np.arange(x) #creating an array where the value is the index
        max_loc = np.nanmean(nan_count) #finding the average index of top of the hill
     
        
        h_m = max_amp//2
        h_m_array  = base[-h_m,:]*0+1
        h_m_w = (x-np.nansum(h_m_array))//2+lensing
        
        max_amp = max_amp + height_increase
        
        topo_function=-max_amp*np.exp(-(domain-max_loc)**2/(2*h_m_w**2))+y
        
        
    if no_hills == 2:
        print('unfinished')
    
    
    return topo_function