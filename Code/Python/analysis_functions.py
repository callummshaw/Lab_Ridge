# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 08:49:14 2020

@author: Callum Shaw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path
import cv2
from matplotlib import rc

import matplotlib.animation as animation
import cmocean as cmo


def index_find(depths,click):
    '''

    Parameters
    ----------
    depths : An array of depths
    click : Mouse click

    Returns
    -------
    i : the location in depth array that corresponds to where click was

    '''
    lst = depths<click
    for i,v in enumerate(lst):
        if v==True:
            return i
        
def load_data(path,run_num):
    '''
    Function that loads data

    Parameters
    ----------
    path : Path to excel sheet with experiment data
    run_num : the run number that is being analysed

    Returns
    -------
    rho_bottom : Bottom tank dentisty (k/m^3)
    rho_top : Top of tank density (k/m^3)
    depth : Distance between density samples (m)
    '''
    data = pd.read_excel(path)
    
    rho_bottom = data.iloc[run_num-1, 7]
    rho_top = data.iloc[run_num-1, 6]
    depth = data.iloc[run_num-1, 2]
    
    return [rho_bottom, rho_top], depth

def background_profile(b_image, density_locations, exp_rho, depth):
    '''
    Function that finds the background density profile given an image

    Parameters
    ----------
    b_image : Background image that you want to average over
    density_locations : Top and bottom of water column
    rho_bottom : Bottom tank dentisty (k/m^3)
    rho_top : Top of tank density (k/m^3)
    depth : Distance between density samples (m).

    Returns
    -------
    depth_array : Array of all tank depth
    background_data : Background density  data

    '''
    zbot=int(np.round(density_locations[0][1]))
    ztop=int(np.round(density_locations[1][1]))
    
    rho_bottom=exp_rho[0]
    rho_top=exp_rho[1]

    back_crop= b_image[zbot:ztop,:]

    #taking the log and removing unwanted inf values
    log_back = np.log(back_crop)
    log_back[np.isinf(log_back)]=np.nan
    
    intensity = np.float32(log_back)
    
    depth_array = -np.linspace(0,depth,intensity.shape[0])
    
    #finding the average intensity over the middle of the image
    middle= int(intensity.shape[1]/2)
    intensity_average=np.mean(intensity[:,middle-10:middle+10],axis=1)
    
    beta = (rho_bottom-rho_top)/(intensity_average[-1]-intensity_average[0])
    bottom_ref = intensity_average[-1]
    
    rho_ref=rho_bottom+np.float64(beta*(intensity-bottom_ref))
    
    background_data = [beta, bottom_ref, rho_ref]
    return depth_array, background_data

def foreground_profile(what, foreground_path, background_data, density_locations, path, run):
    '''
    

    Parameters
    ----------
    what : If = 1, then will only save animation, =2 then will make anom vid, if =3 then will make abs vid
    foreground_path : Location of all the foreground pictures
    background_data : Produced by background profile, contains beta, bottom_ref and rho_ref
    density_locations : Top and bottom of water column
    path : path to excel doco
    run : run number

    Returns
    -------
    None.

    '''
    exp_rho, depth = load_data(path, run)
    
    rho_bottom=exp_rho[0]
    rho_top=exp_rho[1]
    
    no_images = len(foreground_path)
    os.mkdir('{}/results'.format(os.path.dirname(foreground_path[0])))
    
    zbot=int(np.round(density_locations[0][1]))
    ztop=int(np.round(density_locations[1][1]))
    
    
    beta=background_data[0]
    bottom_ref=background_data[1]
    rho_ref=background_data[2]
    
    
    #the save only option (no plotting)
    if what == 1:
        
        y,x=rho_ref.shape
        density_abs = np.zeros((no_images,y,x))
        
        for i in range(no_images):

            f_image=cv2.imread(foreground_path[i],0)
            f_image_crop=f_image[zbot:ztop,:]
            absorbtion = np.log(f_image_crop)

            #getting rid of unwated inf_values and converting to density
            absorbtion[np.isinf(absorbtion)]=np.nan
            density = rho_bottom+np.float64(beta*(absorbtion-bottom_ref))
            
            #putting density data into array
            density_abs[i]=density
        
        np.savez('{}/results/data'.format(os.path.dirname(foreground_path[0])),density_abs=density_abs, background_data=background_data)
    #plotting anom
    if what == 2:
        ims=[]
        fig = plt.figure(figsize=(10,5))
        for i in range(no_images):

            f_image=cv2.imread(foreground_path[i],0)
            f_image_crop=f_image[zbot:ztop,:]
            absorbtion = np.log(f_image_crop)

            #getting rid of unwated inf_values and converting to density
            absorbtion[np.isinf(absorbtion)]=np.nan
            density = rho_bottom+np.float64(beta*(absorbtion-bottom_ref))
    
    
            cmap = cmo.cm.balance
            vmin=-2
            vmax=-vmin
        
            den=density[:,:]-rho_ref
            den[den>4]=np.nan
            density_filt=cv2.medianBlur(np.float32(den[50:,:]),3)
        
            plt_depth=depth
            tank_length=depth/rho_ref.shape[0]*rho_ref.shape[1]

            im=plt.imshow(density_filt, cmap=cmap, animated=True, vmin=vmin,vmax=vmax, extent=[0,tank_length,-plt_depth,0])
            title = 'Run {}- Density Anomaly'.format(run)
            plt.title(title, fontsize=20)
            
            plt.xlabel('Length (m)')
            plt.ylabel('Depth (m)')
            
            ims.append([im])
            if i % 25 == 0:
                print('{} of {} Images Done!'.format(i,no_images))
                
                
        ani = animation.ArtistAnimation(fig, ims, interval=125, blit=True,
                                        repeat_delay=1000)
        
        
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_label(r'Density (kg m$^{-3}$)', rotation=90)
        
        
    
        print('Saving!')
        
        writer = animation.writers['ffmpeg']
        save_name = 'run_{}_anomaly'.format(run)
        ani.save('{}/densityfields/{}.mp4'.format(os.path.dirname(foreground_path[0]),save_name), dpi=250)
    #plotting abs    
    if what == 3:
        ims=[]
        fig = plt.figure(figsize=(10,5))
        
        
        
        for i in range(no_images):
        
            f_image=cv2.imread(foreground_path[i],0)
            f_image_crop=f_image[zbot:ztop,:]
            absorbtion = np.log(f_image_crop)
        
            #getting rid of unwated inf_values and converting to density
            absorbtion[np.isinf(absorbtion)]=np.nan
            density = rho_bottom+np.float64(beta*(absorbtion-bottom_ref))
            
            cut=600
            plt_depth=depth/718*(718-cut)
                    
            den=density[:,50:-50]
            den[den>rho_bottom+2]=np.nan
            density_filt=cv2.medianBlur(np.float32(den[:cut,:]),3)
        
            
            tank_length=depth/rho_ref.shape[0]*rho_ref.shape[1]
            
            cmap = cmo.cm.dense
            vmin=rho_top
            vmax=rho_top+4
                
          
        
            im=plt.imshow(density_filt, cmap=cmap, animated=True, vmin=vmin,vmax=vmax, extent=[0,tank_length,-plt_depth,0])
            title = 'Run {}- Density'.format(run)
            plt.title(title, fontsize=20)
            
            plt.xlabel('Length (m)')
            plt.ylabel('Depth (m)')
            
            ims.append([im])
        
            
            if i % 25 == 0:
                print('{} of {} Images Done!'.format(i,no_images))
                
                
        ani = animation.ArtistAnimation(fig, ims, interval=125, blit=True,
                                        repeat_delay=1000)
        
        
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_label(r'Density (kg m$^{-3}$)', rotation=90)
        
        
        print('Saving!')
        
        writer = animation.writers['ffmpeg']
        save_name = 'run_{}_abs'.format(run)
        ani.save('{}/densityfields/{}.mp4'.format(os.path.dirname(foreground_path[0]),save_name), dpi=250)
