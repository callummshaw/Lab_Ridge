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
import analysis_functions as af
from multiprocessing import Pool
from scipy import interpolate 

data_path='E:/batch 2/3echo/run8/transit/results/centre_data.npz'
data = np.load(data_path)

rho = data['centre_rho']

#remving nans at base
min_nan=np.min(np.sum(np.isnan(rho[0]),axis=0)) #summing the Nans in the vertical direction
offset = 15

rho_c = rho[:,:-(min_nan+offset),:]

    

  
def transformation(data, no_hills=1, lensing=33):
   
    t,z,x=data.shape
    topo_function = af.topograghy_mask(data, no_hills, lensing)
    
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

transformed_data = transformation(rho_c)


def interp_nan(time):
    data = transformed_data
    t,z,x=data.shape
    
    #creating a meshgrid of pixel locations
    x_array = np.arange(x)
    z_array = np.arange(z)
    xx,zz=np.meshgrid(x_array,z_array)
    
    
    array = np.ma.masked_invalid(data[time])
    x1=xx[~array.mask]
    z1=zz[~array.mask]
    newarr=array[~array.mask]

    interp = interpolate.griddata((x1, z1), newarr.ravel(),(xx, zz), method='cubic')
    return interp
 


if __name__ == '__main__':
   
    numbers=np.arange(transformed_data.shape[0])
    p=Pool()
    result = p.map(interp_nan, numbers)
    p.close()
    p.join()

 



