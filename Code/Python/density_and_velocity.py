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
import time 
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
#Doing everything in one--- choose yes or no for each animation that you want:



excel_path='E:/records.xlsx'

run = 8 #run number

def light_attenuation_analysis(run, excel_path, no_hills=1, moving_anom = 'no', moving_abs = 'no', fixed_anom = 'no', fixed_abs = 'no', w_vel = 'no', uvel = 'no'):
    
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
 
        
        ok = int(input('Happy with background density? 1 for yes, 2 for no: '))
        
        if ok == 1:
            break
        
    
    print('\n Select Foreground Images')
    foreground_paths = askopenfilenames()
    print('\n Analysing Short Term Density Variations')
    density_profile = af.foreground_profile(foreground_paths, background_data, density_locations,excel_path, run, moving_anom, moving_abs)
    
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
    data = af.centred_field(topo_location, density_profile, rho_ref, rho_top, run, foreground_paths[0], fixed_anom, fixed_abs)
    
    print('\n Transforming and filtering data for Fourier Filtering')
    bottom_offset=15 #adjusting bottom cutoff of mask
    lensing = 33 #adjusting width of mask
    while True:
        #choosing region that you want (cropping photo to just show water)
        topo_mask, cropped_data = af.topograghy_mask(data, no_hills, bottom_offset=bottom_offset, lensing=lensing)
        ok = int(input('\n Happy with topography mask? 1 for yes, 2 for no: '))
        if ok == 2:
            lensing = int(input(f'\n New Lensing Value (Previous was {lensing}): '))
            bottom_offset = int(input(f'\n New Bottom Offset Value (Previous was {bottom_offset}): '))
        if ok == 1:
            break
    
    