# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:07:34 2020

@author: Callum Shaw
"""

import numpy as np
import matplotlib.pyplot as plt
import analysis_functions as af
import pandas as pd
from multiprocessing import Pool
import os

# data_path='E:/batch 2/3echo/run8/transit/results/centre_data.npz'
# data = np.load(data_path)

# rho = data['centre_rho']

# #remving nans at base
# min_nan=np.min(np.sum(np.isnan(rho[0]),axis=0)) #summing the Nans in the vertical direction
# offset = 15

# rho_c = rho[:,:-(min_nan+offset),:]
# topo_function = af.topograghy_mask(rho_c) #finding the topography shape

# t,z,x=rho_c.shape

# #creating data used in transform
# x_array = np.arange(x)
# z_array = np.arange(z)
# xx,zz=np.meshgrid(x_array,z_array)

# zt=np.zeros((z,x)) #where we are storing the transformed z coordinate

# for i in range(x):
#     topo=topo_function[i]
#     transformed_array=z*(zz[:,i]-topo)/(-topo) #function
#     zt[:,i]=-np.round(transformed_array)+z


#checking topo looks alright
#plt.figure(figsize=(10,10))
#plt.title('Topography Function')
#plt.imshow(rho_c[0])
#plt.plot(topo_function)

def transform_and_interp(k):
    # data = rho_c[k]
    # dummy = np.zeros((z,x)) #variable to store date during transform
    
    # #applying the transform on the data
    # for i in range(x):
    #         for j in range(z):
    #             wanted_data=data[j,i]
    #             z_loc = int(zt[j,i])
    #             if z_loc<580:
    #                 dummy[z_loc,i]=wanted_data
    
    # dummy[dummy==0]=np.nan
    
    # #now interpolating data
    # dummy_frame = pd.DataFrame(dummy)
    # dummy_int = dummy_frame.interpolate()
    sq = k**2
    return sq 


if __name__ == '__main__':
      numbers=range(10)
      p=Pool()
      result = p.map(transform_and_interp, numbers)
      p.close()
      p.join()

# for l in range(10):
#     a = transform_and_interp(l)
#     print(a)

#np.savez('{}/transformation_result'.format(os.path.dirname(data_path)),result=result)
