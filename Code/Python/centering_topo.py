# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:34:00 2020

@author: Callum Shaw

For filtering and visualistion purposes it would be convienent if the hill was stationairy and the fluid moved. 
The code below does this. In short it sets the topography to Nan, then sums the Nans in the vertical direction 
in order to find the locations in x where the topography is 'most present'. I find the average location of the 
maximal Nan area and then crop the flow fields to keep the topography in the same spot.
"""
import numpy as np
from tkinter.filedialog import askopenfilename
import analysis_functions as af

#loading in density data

# #data_path = askopenfilename()
data_path='E:/batch 2/3echo/run8/transit/results/data.npz'
data = np.load(data_path, allow_pickle=True)

#loading in run data

excel_path='E:/records.xlsx'
run = 8
exp_rho, depth = af.load_data(excel_path, run)


density_abs=data['density_abs']
background_data=data['background_data']

rho_ref=background_data[2]
rho_bottom=exp_rho[0]
rho_top=exp_rho[1]

topo_location = af.topo_locator(density_abs, rho_bottom)
#check that the image produced matches what you expect it to be

def centre_analysis(i):
    #1 for saving data, 2 for anom vid and 3 for abs vid
    af.centred_field(i, topo_location, density_abs, rho_ref, rho_top, run, data_path)
