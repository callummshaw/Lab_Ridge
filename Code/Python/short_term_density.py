# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:51:28 2020

@author: Callum Shaw
"""

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames

import analysis_functions as af

## %matplotlib auto before start!!!!!


#loading in data that will be used to determine 

#excel_path = askopenfilename()
excel_path='E:/records.xlsx'
run = 8

exp_rho, depth = af.load_data(excel_path, run)

#Analysing background image
background_path = askopenfilename() 
b_image = cv2.imread(background_path,0)

#choosing region that you want (cropping photo to just show water)
plt.figure()
plt.title('Choose Area to Average over- Top then bottom')
plt.axis('off')
plt.imshow(b_image, cmap='gist_gray')
density_locations = plt.ginput(2)


# %matplotlib inline before going on!
#now analysing the foreground images


def analysis(i):
    #1 for saving data, 2 for anom vid and 3 for abs vid
    depth_array, background_data = af.background_analysis(b_image, density_locations, exp_rho, depth)
    foreground_paths = askopenfilenames()
    af.foreground_profile(i, foreground_paths, background_data, density_locations,excel_path, run)

