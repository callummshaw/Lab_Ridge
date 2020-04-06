# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:26:53 2020

@author: Callum Shaw

A program that will do everything all in one.
"""
import analysis_functions as af
import cv2
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
import pandas as pd
from numpy import fft

from analysis_functions import background, foreground, topography


excel_path='E:/records.xlsx'
run = 8 



def light_attenuation_analysis( vertical_crop=1000, no_hills=1, sigma=0.005, moving_anom = 'no', moving_abs = 'no', fixed_anom = 'no', fixed_abs = 'no', w_vel = 'no', u_vel = 'no'):
  
    #Analysing background image
    print('\n Select Background Image')
    background_path = askopenfilename()
    b_d = background(run,excel_path, background_path)
    b_d.load_data()
    
    while True:
        #choosing region that you want (cropping photo to just show water)
        b_d.plot_raw()
        b_d.density_profile()
        b_d.density_plot()
        
        ok = int(input(' Happy with background density? 1 for yes, 2 for no: '))
        
        if ok == 2:
            print('\n Please redo cropping region')
        if ok == 1:
            break
 
    
    print('\n Select Foreground Images')
    foreground_paths = askopenfilenames()
    f_d = foreground(foreground_paths)
    
    print(f'\n Analysing Short Term Density Variations For {f_d.no_images} Images')
    af.foreground_profile(b_d, f_d, vertical_crop, moving_anom, moving_abs)
    
    print('\n Finding background topography')
    t_d = topography(no_hills, f_d, b_d)
    t_d.topo_locator()
    t_d.topo_location_plot()
    
    print('\n Centering Fields!')
    af.centred_field(t_d, b_d, f_d, vertical_crop, fixed_anom, fixed_abs)
   
    print('\n Transforming data for Fourier Filtering')
    
    while True:
        #choosing region that you want (cropping photo to just show water)
        af.topograghy_mask(t_d,f_d)
        ok = int(input(' Happy with topography mask? 1 for yes, 2 for no: '))
        
        if ok == 2:
            lensing = int(input(f' New Lensing Value (Previous was {t_d.lensing}): '))
            bottom_offset = int(input(f' New Bottom Offset Value (Previous was {t_d.bottom_offset}): '))
            
            t_d.lensing = lensing
            t_d.bottom_offset = bottom_offset
        if ok == 1:
            break
    return('Test complete')
    z_prime = af.transformed_coords(cropped_data, topo_mask)
    t,z,x=cropped_data.shape

    filt = af.low_pass_filter(z-2, x-2, sigma=sigma)
    print('\n Now filtering Data')
    def fourier_filter_and_trans(i):
        
        data = cropped_data[i]
        z,x=data.shape
        trans_dummy = np.zeros((z,x)) #variable to store date during transform
        
        #applying the transform on the data
        for k in range(x):
                for j in range(z):
                    wanted_data=data[j,k]
                    z_loc = int(z_prime[j,k])
                    if z_loc<z:
                        trans_dummy[z_loc,k]=wanted_data
        
        trans_dummy[trans_dummy==0]=np.nan
        
        #now interpolating data
        dummy_frame = pd.DataFrame(trans_dummy)
        dummy_int = dummy_frame.interpolate()
        dummy_int = dummy_int.values
        cropped = dummy_int[1:-1,1:-1]
        z_prime_c=z_prime[1:-1,1:-1]
        
        
                
        fft_data = fft.fft2(cropped) #preforming fft
        filtered_data = fft_data*filt #filtering
        ifft_data = fft.ifft2(filtered_data) #going back to cartesian space
        
        filt_d = np.float32(ifft_data.real)
    
        filt_dummy = np.zeros(filt_d.shape)
        z,x=filt_d.shape
        
        for k in range(x): 
            for j in range(z):
                z_val = z_prime_c[j,k]
                if z_val < z-1:
                    filt_dummy[j,k]=filt_d[int(z_val),k]
        filt_dummy=np.float32(filt_dummy)
        filt_dummy[filt_dummy==0]=np.nan
        
        return(filt_dummy)       
    
    if __name__ == '__main__':
          numbers=range(t)
          p = Pool()
          result = p.map(fourier_filter_and_trans, numbers)
          p.close()
          p.join()
    
    result=np.dstack(result)
    result = np.moveaxis(result,2,0)    
    
    print('\n Filtering Finished')
    
    print('\n Now calculating W')
    time_scale = 0.125
    scale=-g/(rho_0*buoyancy_freq**2)
    w = scale*np.gradient(result-rho_0,time_scale,axis=0)

    if w_vel == 'yes':
        print('\n Plotting W')
        af.plot_w(w, run, foreground_paths[0])  
    
    if u_vel == 'yes':
        print('Warning!! Not Yet Finished')
    np.save('vertical_test',w)