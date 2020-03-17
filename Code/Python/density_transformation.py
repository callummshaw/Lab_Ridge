# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:59:40 2020

@author: Callum Shaw

This code uses the density fields in the ridge frame of reference (produced by
'centering_topo.py') and removes the topography through a transformation 
and then fills in the nans with interpolation to get the data ready for fourier
filtering. The interpolation is very slow and has been speed up partially with 
parallel processing
"""

import numpy as np
import analysis_functions as af
from multiprocessing import Pool
from scipy import interpolate 
import os
import time

data_path='E:/batch 2/3echo/run8/transit/results/centre_data.npz'
data = np.load(data_path)

rho = data['centre_rho']

#remving nans at base
min_nan=np.min(np.sum(np.isnan(rho[0]),axis=0)) #summing the Nans in the vertical direction
offset = 15

rho_c = rho[:,:-(min_nan+offset),:]

    

  
def transformation(data, no_hills=1, lensing=33):
    '''
    This function transforms data to remove the topography, this is done by
    creating z'. The removal of topography allows for fourier filtering

    Parameters
    ----------
    data : Dataset we want to transform
    no_hills : the number of hills in the data, defualt is 1
    lensing : a fudge factor to overcome some lensing that occurs around the hiil
                the default is 33.

    Returns
    -------
     The transformed dataset 

    '''   
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
            print('{} Images Transformed!'.format(k))
        for i in range(x):
            for j in range(z):
                wanted_data=rho_data[j,i]
                z_loc = int(zt[j,i])
            
                if z_loc<580:
                    transformed[k,z_loc,i]=wanted_data
    
    transformed[transformed==0]=np.nan
    
    return transformed

transformed_data = transformation(rho_c)
np.savez('test_data',transformed_data=transformed_data)
print('Now Interpolating Images!')

# def interp_nan(i):
#     '''
#     This function interpolates the transformed data to remove the nans. 
#     Interpolation is very slow so it is set up to be parallelised.

#     Parameters
#     ----------
#     time : reads in time which we are iterating ovrt

#     Returns
#     -------
#     interp : interped data set

#     '''
#     data = transformed_data
#     t,z,x=data.shape
    
#     #creating a meshgrid of pixel locations
#     x_array = np.arange(x)
#     z_array = np.arange(z)
#     xx,zz=np.meshgrid(x_array,z_array)
    
    
#     array = np.ma.masked_invalid(data[i])
#     x1=xx[~array.mask]
#     z1=zz[~array.mask]
#     newarr=array[~array.mask]

#     interp = interpolate.interpn((x1, z1), newarr.ravel(),(xx, zz), method='linear')
#     return interp
 


# if __name__ == '__main__':
#     start = time.time()
#     numbers=np.arange(1)
#     p=Pool()
#     result = p.map(interp_nan, numbers)
#     p.close()
#     p.join()
#     print(time.time()-start)
# # print('Now Saving Data!')

# np.savez('{}/transformation_result'.format(os.path.dirname(data_path)),result=result)


