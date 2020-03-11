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


#now calculating background density
depth_array, background_data = af.background_profile(b_image, density_locations, exp_rho, depth)

rho_ref=background_data[2]

plt.figure()
im=plt.imshow(rho_ref,vmin=exp_rho[0],vmax=exp_rho[1], extent=[0,depth/rho_ref.shape[0]*rho_ref.shape[1],-depth,0])
plt.title('Background Density')
plt.xlabel('Length')
plt.ylabel('Depth')
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar=plt.colorbar(im, cax=cax)
cbar.set_label(r'Density (kg m$^{-3}$)', rotation=90)
cbar.ax.invert_yaxis()


#now analysing the foreground images
foreground_paths = askopenfilenames()
what = 2 #1 for saving data, 2 for anom vid and 3 for abs vid
af.foreground_profile(what, foreground_paths, background_data, density_locations,excel_path, run)
