# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 08:49:14 2020

@author: Callum Shaw

This is module that contains all the functions needed for the lab ridge analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path
import cv2

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

def background_analysis(b_image, density_locations, exp_rho, depth):
    depth_array, background_data = background_profile(b_image, density_locations, exp_rho, depth)
    
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
    
    return depth_array, background_data

def foreground_profile(what, foreground_path, background_data, density_locations, path, run):
    '''
    

    Parameters
    ----------
    what : If = 1, then will only save animation, =2 then will make abs vid, if =3 then will make anom vid
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
    
    if not os.path.exists('{}/results'.format(os.path.dirname(foreground_path[0]))):
        os.makedirs('{}/results'.format(os.path.dirname(foreground_path[0])))
    
    zbot=int(np.round(density_locations[0][1]))
    ztop=int(np.round(density_locations[1][1]))
    
    
    beta=background_data[0]
    bottom_ref=background_data[1]
    rho_ref=background_data[2]
    
    
    #the save only option (no plotting)
    if what == 1:
        crop_points=600 #how much you want to crop in vertical
        y,x=rho_ref.shape
        density_abs = np.zeros((no_images,crop_points,x))
        #only taking the crop_points closest to the top to save
        for i in range(no_images):

            f_image=cv2.imread(foreground_path[i],0)
            f_image_crop=f_image[zbot:ztop,:]
            absorbtion = np.log(f_image_crop)

            #getting rid of unwated inf_values and converting to density
            absorbtion[np.isinf(absorbtion)]=np.nan
            density = rho_bottom+np.float64(beta*(absorbtion-bottom_ref))
            
            #putting density data into array
            density_abs[i]=density[:crop_points,:][::-1]  #cropping and flipping data
        
        np.savez('{}/results/data'.format(os.path.dirname(foreground_path[0])),density_abs=density_abs, background_data=background_data)
    #plotting anom
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
        ani.save('{}/results/{}.mp4'.format(os.path.dirname(foreground_path[0]),save_name), dpi=250)
    #plotting abs    
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
        ani.save('{}/results/{}.mp4'.format(os.path.dirname(foreground_path[0]),save_name), dpi=250)

def max_and_loc(data):
    '''
    Simple function that finds the maximum height of topo and the location of
    the maximum. It takes an average of values from around the max, to avoid
    the problem of the maximum shifting frame to frame

    Parameters
    ----------
    data : data set that we want to find the topography

    Returns
    -------
    max_amp : The maximum height of the topography
    max_loc : The location (in x) of the topography

    '''
    y,x = data.shape
    
    nan_array = np.sum(np.isnan(data),axis=0)
    max_amp = np.max(nan_array)
    nan_array = np.float32(nan_array) #need to convert from int to float
    nan_array[nan_array<max_amp-5]=np.nan #removing everything but the top of the hill
        
    nan_count=nan_array*0+1
    nan_count=nan_count*np.arange(x) #creating an array where the value is the index
    max_loc = np.nanmean(nan_count) #finding the average index of top of the hill
    
    return max_amp, max_loc

def topo_locator(density_abs,rho_bottom):
    '''
    

    Parameters
    ----------
    density_abs : Dataset that has all the images that we will run over
    rho_bottom : Max rho from experiment (used to determine the topography)

    Returns
    -------
    topo_location : An array that has the average location (X Pixel) of tip of topography for each image in dataset

    '''
    
    t,y,x=density_abs.shape
    topo_location=np.zeros(t)
    
    density_abs[density_abs>rho_bottom-3]=np.nan #setting topo to nan
    
    for i in range(t):
        image=density_abs[i]
        
        nan_count=np.float32(np.sum(np.isnan(image),axis=0)) #summing the Nans in the vertical direction
        max_nan =  np.max((np.sum(np.isnan(image),axis=0)))
        nan_count[nan_count<max_nan-5]=np.nan #removing everything but the top of the hill
        nan_count=nan_count*0+1
        nan_count=nan_count*np.arange(x) #creating an array where the value is the index
    
        max_loc = np.nanmean(nan_count) #finding the average index of top of the hill
        topo_location[i]=max_loc
    
    plt.figure(figsize=(10,5))
    plt.plot(topo_location,color='b')
    plt.title('Topography Location')
    plt.xlabel('Image Number')
    plt.ylabel('Topography Location (Pixel)')
    
    return topo_location

def crop_centre(i, topo_location, field, rho_ref):
    '''
    

    Parameters
    ----------
    i: if I = 1 or 2 then returns abs density if rho =3 then returns anom 
    topo_location : An array that has the average location (X Pixel) of tip of topography for each image in dataset
    field : Dataset that we are cropping (either density abs or density anom)
    rho_ref : Dataset that that has the background density

    Returns
    -------
    cropped_abs : The cropped abs with the topography stationairy 
    cropped_anom : The cropped anom with the topography stationairy

    '''

    t,y,x=field.shape
    
    right = int(x-max(topo_location))
    left = int(min(topo_location))    
    
    cropped_field = np.zeros((t,y,right+left))
    
    
    if i ==1 or i == 2:
        
        for j in range(t):
            topo=int(topo_location[j])
            
            image=field[j]
            
            cropped_image=image[:,int(topo-left):int(topo+right)]
            
            
            cropped_field[j]=cropped_image
        
        return cropped_field
    
    elif i == 3:
        
        for j in range(t):
            topo=int(topo_location[j])
            
            image=field[j]
            
            cropped_image=image[:,int(topo-left):int(topo+right)]
            cropped_ref=rho_ref[:600,int(topo-left):int(topo+right)]
            
            delta=cropped_image-cropped_ref
            cropped_field[j] = delta
            
        return cropped_field

    

def centred_field(i, topo_location, field, rho_ref, rho_top, run, data_path):
    
    #saving data
    if i == 1:
        
        centre_rho = crop_centre(i,topo_location,field, rho_ref)
        np.savez('{}/centre_data'.format(os.path.dirname(data_path)),centre_rho=centre_rho)
        
    if i == 2:
        centre_rho = crop_centre(i,topo_location,field, rho_ref)
        
        ims=[]
        fig = plt.figure(figsize=(10,5))
        
        t,y,x=centre_rho.shape
        
        for i in range(t):
            
            image=centre_rho[i]
            
            cmap = cmo.cm.dense
            vmin=rho_top
            vmax=rho_top+4
                
        
            im=plt.imshow(image, cmap=cmap, animated=True, vmin=vmin,vmax=vmax)
            title = 'Run {}- Density'.format(run)
            plt.title(title, fontsize=20)
            
            plt.xlabel('Length (m)')
            plt.ylabel('Depth (m)')
            
            ims.append([im])
        
                
        ani = animation.ArtistAnimation(fig, ims, interval=125, blit=True,
                                        repeat_delay=1000)
        
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_label(r'Density (kg m$^{-3}$)', rotation=90)
        
        
        print('Saving!')
        
        writer = animation.writers['ffmpeg']
        save_name = 'run_{}_abs_centre'.format(run)
        ani.save('{}/{}.mp4'.format(os.path.dirname(data_path),save_name), dpi=250)
    
    if i == 3:
        centre_anom = crop_centre(i,topo_location,field, rho_ref)
        
        ims=[]
        fig = plt.figure(figsize=(10,5))
        
        t,y,x=centre_anom.shape
        
        vmin=np.nanmin(centre_anom[0])
        vmax=-vmin
        
        
        for i in range(t):
            
            image=centre_anom[i]
            
            cmap = cmo.cm.balance
            
                
            density_filt=cv2.medianBlur(np.float32(image),3)

        
            im=plt.imshow(density_filt, cmap=cmap, animated=True, vmin=vmin,vmax=vmax)
            title = 'Run {}- Density Anomaly'.format(run)
            plt.title(title, fontsize=20)
            
            plt.xlabel('Length (m)')
            plt.ylabel('Depth (m)')
            
            ims.append([im])
                
                
        ani = animation.ArtistAnimation(fig, ims, interval=125, blit=True,
                                        repeat_delay=1000)
        
        
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_label(r'Density (kg m$^{-3}$)', rotation=90)
        
        
        print('Saving!')
        
        writer = animation.writers['ffmpeg']
        save_name = 'run_{}_anomaly_centre'.format(run)
        ani.save('{}/{}.mp4'.format(os.path.dirname(data_path),save_name), dpi=250)
                    
                    