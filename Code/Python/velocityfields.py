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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path
import matplotlib.animation as animation
import cmocean as cmo
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

test1 = np.gradient(result,axis=0)

t,z,x=test1.shape


ims=[]
fig = plt.figure(figsize=(10,5))

for i in range(t):
    
    image=test1[i]
    
    cmap = cmo.cm.balance
    vmin=-0.02
    vmax=-vmin
        

    im=plt.imshow(image, cmap=cmap, animated=True, vmin=vmin,vmax=vmax)
    title = 'Sample Vertical Velocity'
    plt.title(title, fontsize=20)
    
    plt.xlabel('Length')
    plt.ylabel('Depth')
    
    ims.append([im])

        
ani = animation.ArtistAnimation(fig, ims, interval=125, blit=True,
                                repeat_delay=1000)

ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar=plt.colorbar(im, cax=cax)
cbar.set_label(r'Velocity (m s$^{-1}$)', rotation=90)


print('Saving!')

writer = animation.writers['ffmpeg']
save_name = 'vertical_velocity_05'
ani.save('{}/{}.mp4'.format(os.path.dirname(transform_data_path),save_name), dpi=250)