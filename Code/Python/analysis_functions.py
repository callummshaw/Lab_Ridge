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
import sys

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
    plt.pause(5)
    
    return depth_array, background_data

def foreground_profile(foreground_path, background_data, density_locations, path, run, moving_anom = 'no', moving_abs = 'no'):
    '''
    

    Parameters
    ----------
    foreground_path : Location of all the foreground pictures
    background_data : Produced by background profile, contains beta, bottom_ref and rho_ref
    density_locations : Top and bottom of water column
    path : path to excel doco
    run : run number
    moving_anom :if yes will make animation. Default is no
    moving_abs : if yes will make animation. Default is no

    Returns
    -------
    Density data

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
        
    #plotting anom
    if moving_anom == 'yes':
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
    if moving_abs == 'yes':
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
        
    return density_abs

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
                    
def topograghy_mask(rho, no_hills=1, lensing=33):
    '''
    A function that reads in a dataset and returns a function that can be used to mask
    the topography (needed for fourier transform)

    Parameters
    ----------
    rho : Dataset that is being read in
    no_hills : The number of hills in the topo. The default is 1.
    lensing : A sort of fudge factor- to overcome the effect of lensing on the topo
        The default is 33.

    Returns
    -------
    topo_function : An funtion in the form of a 1D array that can be used to mask
        the topography.

    '''
    t,y,x = rho.shape
    base = rho[t//2]
    domain = np.arange(x)
    

    if no_hills == 1:
        
        height_increase=20 #pixels we want to increase the topo height by (otherwise cuts off sides of the top)
        
        max_amp, max_loc = max_and_loc(base)
        
        h_m = max_amp//2
        h_m_array  = base[-h_m,:]*0+1
        h_m_w = (x-np.nansum(h_m_array))//2+lensing
        
        max_amp = max_amp + height_increase
        
        topo_function=-max_amp*np.exp(-(domain-max_loc)**2/(2*h_m_w**2))+y
        
        return topo_function
        
    if no_hills == 2:
        
        print('Currently Untested')
        
        mid = x//2
        
        side1 = base[:, :mid]
        side2 = base[:, mid:]
    
        #splitting the domain in two parts to find the hills, will assume hills are the same shape, 
        #only difference is a horizontal translation
    
        height_increase=20
        
        max_amp_1, max_loc_1 = max_and_loc(side1)
        max_amp_2, max_loc_2 = max_and_loc(side2)
    
        
        
        h_m = max_amp_1//2
        h_m_array  = base[-h_m,:]*0+1
        h_m_w = (x-np.nansum(h_m_array))//2+lensing
        
        topo_function=-max_amp_1*np.exp(-(domain-max_loc_1)**2/(2*h_m_w**2))-max_amp_2*np.exp(-(domain-max_loc_2)**2/(2*h_m_w**2))+y
         
        return topo_function

def transformed_coords(data_path, bottom_offset=15, return_dataset='yes'):
    '''
    A function that calculates the transformed coordinates needed to mask the 
    the topography

    Parameters
    ----------
    data_path : Path to the data that is being transformed
    bottom_offset : The amount of datapoints that you want to remove from
                    bottom of the dataset. The default is 15.
    dataset: if the function will return the cropped dataset (with the bottom 
             part removed). The default is yes, if not desired then 'no'

    Returns
    -------
    zt : The transformed array, that contains the z' coordinates
    rho_c : The cropped dataset

    '''
    print(f'Return Bottom Cropped Dataset = {return_dataset}')
    if return_dataset not in {'yes', 'no'}:
        print('Return Dataset must be either yes or no, please rerun')
        sys.exit(1)
         
    data = np.load(data_path)

    rho = data['centre_rho']
    #remving nans at base
    min_nan=np.min(np.sum(np.isnan(rho[0]),axis=0)) #summing the Nans in the vertical direction
    
    rho_c = rho[:,:-(min_nan+bottom_offset),:]
    
    topo_function = topograghy_mask(rho_c) #finding the topography shape
    
    t,z,x=rho_c.shape
    
    #creating data used in transform
    x_array = np.arange(x)
    z_array = np.arange(z)
    xx,zz=np.meshgrid(x_array,z_array)
    
    zt=np.zeros((z,x)) #where we are storing the transformed z coordinate
    
    for i in range(x):
        topo=topo_function[i]
        transformed_array=z*(zz[:,i]-topo)/(-topo) #function
        zt[:,i]=-np.round(transformed_array)+z
    
    if return_dataset  == 'yes':
        return zt, rho_c
    
    if return_dataset  == 'no':
        return zt