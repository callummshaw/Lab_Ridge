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
#Doing everything in one--- choose yes or no for each animation that you want:

moving_anom = 'yes'
moving_abs = 'yes'

fixed_anom = 'yes' #density fields with topography fixed in centre of domain
fixed_abs = 'yes'

w_vel = 'yes'
u_vel = 'yes' 

excel_path='E:/records.xlsx'

run = 8 #run number

def light_attenuation_analysis(run, excel_path, moving_anom = 'no', moving_abs = 'no', fixed_anom = 'no', fixed_abs = 'no', w_vel = 'no', uvel = 'no'):
    exp_rho, depth = af.load_data(excel_path, run)

    #Analysing background image
    print('Select Background Image')
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
        
    
    print('Select Foreground Images')
    foreground_paths = askopenfilenames()
    
    
    density_profile = af.foreground_profile(foreground_paths, background_data, density_locations,excel_path, run, moving_anom, moving_abs)
    
    return density_profile
