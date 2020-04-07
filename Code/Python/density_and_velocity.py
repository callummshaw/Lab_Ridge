# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:26:53 2020

@author: Callum Shaw

A program that will do everything all in one. Just update the run number and
excel spreadsheet path if changed and then run light_attenuation_analysis(), with 
the desired videos set to 'yes'
"""
import analysis_functions as af
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
from functools import partial
import numpy as np
from multiprocessing.pool import ThreadPool as Pool


from analysis_functions import background, foreground, topography, fourier_filter


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
    t_d = topography(no_hills)
    t_d.topo_locator(f_d.density_abs, b_d.rho_bottom)
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
   
    shape = f_d.centre_rho.shape
    l_d = fourier_filter(sigma, shape)
    l_d.transformed_coords(t_d.topo_function)
    l_d.low_pass_filter()
    
    print('\n Now filtering Data')

    if __name__ == '__main__':
          numbers=range(l_d.t)
          p = Pool()
          func = partial(af.fourier_filter_and_trans, f_d, l_d)
          result = p.map(func, numbers)
          p.close()
          p.join()
          
          result=np.dstack(result)
          result = np.moveaxis(result,2,0)
          
          f_d.filtered_rho = result
    
    print('\n Filtering Finished')
    
    print('\n Now calculating W')
    
    time_scale = 0.125
    rho_0=1000
    g=9.81
    
    scale=-g/(rho_0*b_d.b_freq**2)
    
    f_d.wvel = scale*np.gradient(f_d.filtered_rho-rho_0, time_scale,axis=0)

    if w_vel == 'yes':
        print('\n Plotting W')
        af.plot_w(b_d,f_d)  
    
    if u_vel == 'yes':
        print('Warning!! Not Yet Finished')
    
    return f_d,b_d,t_d,l_d