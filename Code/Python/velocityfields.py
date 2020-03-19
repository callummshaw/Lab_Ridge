# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:51:04 2020

@author: u6201343
"""
import time
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
import analysis_functions as af
import matplotlib.pyplot as plt

transform_data_path='E:/batch 2/3echo/run8/transit/results/centre_data.npz'
filtered_data_path='E:/batch 2/3echo/run8/transit/results/filtered_data.npy'
unfiltered_data_path='E:/batch 2/3echo/run8/transit/results/centre_data.npz'
uf=np.load(unfiltered_data_path)
rho = uf['centre_rho']

zt = af.transformed_coords(transform_data_path, return_dataset='no')
filtered_data =  np.load(filtered_data_path)
zt=zt[1:-1,1:-1]
t,z,x=filtered_data.shape

def back_transform(k):
    data = filtered_data[k]
    dummy = np.zeros(data.shape)
    for i in range(x): 
        for j in range(z):
            z_val = zt[j,i]
            if z_val < z-1:
                dummy[j,i]=data[int(z_val),i]
    dummy=np.float32(dummy)
    dummy[dummy==0]=np.nan
    
    return(dummy)       

if __name__ == '__main__':
      numbers=range(t)
      p = Pool()
      result = p.map(back_transform, numbers)
      p.close()
      p.join()

result=np.dstack(result)
result = np.moveaxis(result,2,0)


plt.figure()
plt.imshow(result[0],vmin=1002,vmax=1008)
plt.colorbar()

plt.figure()
plt.imshow(rho[0],vmin=1002,vmax=1008)
plt.colorbar()

plt.figure()
plt.imshow(result[1]-result[0])
plt.colorbar()
