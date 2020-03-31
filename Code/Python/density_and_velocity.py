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



excel_path='E:/records.xlsx'
run = 8 

def light_attenuation_analysis(run, excel_path, vertical_crop=1000, no_hills=1, sigma=0.005, moving_anom = 'no', moving_abs = 'no', fixed_anom = 'no', fixed_abs = 'no', w_vel = 'no', u_vel = 'no'):
    exp_rho, depth = af.load_data(excel_path, run)
    #Analysing background image
    print('\n Select Background Image')
    background_path = askopenfilename() 
    b_image = cv2.imread(background_path,0)
    
    while True:
        #choosing region that you want (cropping photo to just show water)
        plt.figure()
        plt.title('Choose Area to Average over- Top then bottom')
        plt.axis('off')
        plt.imshow(b_image, cmap='gist_gray')
        density_locations = plt.ginput(2)

        depth_array, background_data = af.background_analysis(b_image, density_locations, exp_rho, depth)
 
        
        ok = int(input(' Happy with background density? 1 for yes, 2 for no: '))
        if ok == 2:
            print('\n Please redo cropping region')
        if ok == 1:
            break
        
    
    print('\n Select Foreground Images')
    foreground_paths = askopenfilenames()
    length = len(foreground_paths)
    print(f'\n Analysing Short Term Density Variations For {length} Images')
    density_profile, plot_ratio = af.foreground_profile(foreground_paths, background_data, density_locations,excel_path, run, vertical_crop, moving_anom, moving_abs)

    exp_rho, depth = af.load_data(excel_path, run)
    
    rho_bottom=exp_rho[0]
    rho_top=exp_rho[1]
    rho_ref=background_data[2]
    
    g=9.81
    rho_0 = 1000
    buoyancy_freq = np.sqrt(g/rho_0*(rho_bottom-rho_top)/depth)
   
    print('\n Finding background topography')
    topo_location = af.topo_locator(density_profile,rho_bottom)
    
    print('\n Centering Fields!')
    data = af.centred_field(topo_location, density_profile, rho_ref, rho_top, rho_bottom, run, foreground_paths[0], vertical_crop, plot_ratio, fixed_anom, fixed_abs)
   
    print('\n Transforming data for Fourier Filtering')
    bottom_offset=15 #adjusting bottom cutoff of mask
    lensing = 33 #adjusting width of mask
    while True:
        #choosing region that you want (cropping photo to just show water)
        topo_mask, cropped_data = af.topograghy_mask(data, no_hills, bottom_offset=bottom_offset, lensing=lensing)
        ok = int(input(' Happy with topography mask? 1 for yes, 2 for no: '))
        if ok == 2:
            lensing = int(input(f' New Lensing Value (Previous was {lensing}): '))
            bottom_offset = int(input(f' New Bottom Offset Value (Previous was {bottom_offset}): '))
        if ok == 1:
            break
    
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