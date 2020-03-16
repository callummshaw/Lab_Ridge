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
import analysis_functions as af

#data_path='E:/batch 2/3echo/run8/transit/results/centre_data.npz'
#data = np.load(data_path)

#rho = data['centre_rho']

#remving nans at base
#in_nan=np.min(np.sum(np.isnan(rho[0]),axis=0)) #summing the Nans in the vertical direction
#offset = 15
#rho_c = rho[:,:-(min_nan+offset),:]


    
def topograghy_mask(rho, no_hills, lensing):
    '''
    A function that reads in a dataset and returns a function that can be used to mask
    the topography (needed for fourier transform)

    Parameters
    ----------
    rho : Dataset that is being read in
    no_hills : The number of hills in the topo. The default is 1.
    lensing : A sort of fudge factor- to overcome the effect of lensing on the topo
        The default is 33.

    Returns
    -------
    topo_function : An funtion in the form of a 1D array that can be used to mask
        the topography.

    '''
    t,y,x = rho.shape
    base = rho[t//2]
    domain = np.arange(x)
    

    if no_hills == 1:
        
        height_increase=20 #pixels we want to increase the topo height by (otherwise cuts off sides of the top)
        
        max_amp, max_loc = af.max_and_loc(base)
        
        h_m = max_amp//2
        h_m_array  = base[-h_m,:]*0+1
        h_m_w = (x-np.nansum(h_m_array))//2+lensing
        
        max_amp = max_amp + height_increase
        
        topo_function=-max_amp*np.exp(-(domain-max_loc)**2/(2*h_m_w**2))+y
        
        return topo_function
        
    if no_hills == 2:
        
        print('Currently Untested')
        
        mid = x//2
        
        side1 = base[:, :mid]
        side2 = base[:, mid:]
    
        #splitting the domain in two parts to find the hills, will assume hills are the same shape, 
        #only difference is a horizontal translation
    
        height_increase=20
        
        max_amp_1, max_loc_1 = af.max_and_loc(side1)
        max_amp_2, max_loc_2 = af.max_and_loc(side2)
    
        
        
        h_m = max_amp_1//2
        h_m_array  = base[-h_m,:]*0+1
        h_m_w = (x-np.nansum(h_m_array))//2+lensing
        
        topo_function=-max_amp_1*np.exp(-(domain-max_loc_1)**2/(2*h_m_w**2))-max_amp_2*np.exp(-(domain-max_loc_2)**2/(2*h_m_w**2))+y
         
        return topo_function
  
def transformation(data, no_hills=1, lensing=33):
   
    t,z,x=data.shape
    topo_function = topograghy_mask(data, no_hills, lensing)
    
    #creating a meshgrid of pixel locations
    x_array = np.arange(x)
    z_array = np.arange(z)
    xx,zz=np.meshgrid(x_array,z_array)
    
    zt=np.zeros((z,x))
    for i in range(x):
        topo=topo_function[i]
        transformed_array=z*(zz[:,i]-topo)/(-topo) #function
        zt[:,i]=-np.round(transformed_array)+z
    
    #creating and filling a new transformed array that usese the values from zt
    transformed = np.zeros((t,z,x))
    for k in range(t):
        rho_data=data[k,:,:]
        if k%100 == 0:
            print('{} done Images!'.format(k))
        for i in range(x):
            for j in range(z):
                wanted_data=rho_data[j,i]
                z_loc = int(zt[j,i])
            
                if z_loc<580:
                    transformed[k,z_loc,i]=wanted_data
    
    transformed[transformed==0]=np.nan
    
    return transformed