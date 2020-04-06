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

      
class background:
    
    def __init__(self, run,excel_path,picture_path):
        self.run = run
        self.excel_path = excel_path
        self.picture_path = picture_path
        self.picture =  cv2.imread(picture_path,0)
    
    def load_data(self):
        data = pd.read_excel(self.excel_path)
        
        rho_bottom = data.iloc[self.run-1, 7]
        rho_top = data.iloc[self.run-1, 6]
        depth = data.iloc[self.run-1, 2]
        b_freq = np.sqrt(9.81/1000*(rho_bottom-rho_top)/depth)
        
        self.rho_bottom = rho_bottom
        self.rho_top = rho_top
        self.depth = depth
        self.b_freq = b_freq
        
        print('\n Experiment Data Loaded')
        
    def plot_raw(self):
       
        plt.figure()
        plt.title('Choose Area to Average over- Top then bottom')
        plt.axis('off')
        plt.imshow(self.picture, cmap='gist_gray')
        density_locations = plt.ginput(2)
        
        self.density_locations = density_locations
        
    def density_profile(self):
        np.seterr(divide='ignore')
        
        zbot=int(np.round(self.density_locations[0][1]))
        ztop=int(np.round(self.density_locations[1][1]))
        back_crop= self.picture[zbot:ztop,:]

        #taking the log and removing unwanted inf values
        log_back = np.log(back_crop)
        log_back[np.isinf(log_back)]=np.nan
        
        intensity = np.float32(log_back)
        
        #finding the average intensity over the middle of the image
        middle= int(intensity.shape[1]/2)
        intensity_average=np.mean(intensity[:,middle-10:middle+10],axis=1)
        
        beta = (self.rho_bottom-(self.rho_top))/(intensity_average[-1]-intensity_average[0])
        bottom_ref = intensity_average[-1]
        
        rho_ref=self.rho_bottom+np.float64(beta*(intensity-bottom_ref))
        
        self.zbot = zbot
        self.ztop = ztop
        self.rho_ref = rho_ref
        self.beta = beta
        self.bottom_ref = bottom_ref
        
        z,x = rho_ref.shape
        
        self.z = z
        self.x = x
        
    def density_plot(self):
        
        plt.figure()
        im=plt.imshow(self.rho_ref,vmin=self.rho_top,vmax=self.rho_bottom, extent=[0,self.depth/self.rho_ref.shape[0]*self.rho_ref.shape[1],-self.depth,0])
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

class foreground:
    
    def __init__(self,foreground_path):
        
        self.foreground_path=foreground_path
        self.no_images = len(foreground_path)
        self.save_path = os.path.dirname(foreground_path[0])
        
        
class topography:
    
    def __init__(self,no_hills,f_d,b_d):
        bottom_offset=15
        lensing=33
        
        self.no_hills = no_hills
        self.bottom_offset = bottom_offset
        self.lensing = lensing
        self.rho_bottom = b_d.rho_bottom
        self.density_abs = f_d.density_abs
        
    def topo_locator(self):
        np.seterr(invalid='ignore')
        t,y,x=self.density_abs.shape
        
        topo_crop = y-450 #cropping data to isolate topography
    
        topo_location=np.zeros(t)
    
        topo_dens = self.density_abs[:,topo_crop:,:]
        topo_dens[topo_dens>self.rho_bottom-5]=np.nan #setting topo to nan
        
        self.density_abs[:,topo_crop:,:] = topo_dens
        
        for i in range(t):
            image=self.density_abs[i]
            
            nan_count=np.float32(np.sum(np.isnan(image),axis=0)) #summing the Nans in the vertical direction
            max_nan =  np.max((np.sum(np.isnan(image),axis=0)))
            nan_count[nan_count<max_nan-5]=np.nan #removing everything but the top of the hill
            nan_count=nan_count*0+1
            nan_count=nan_count*np.arange(x) #creating an array where the value is the index
        
            max_loc = np.nanmean(nan_count) #finding the average index of top of the hill
            topo_location[i]=max_loc
            
            
        self.topo_location = topo_location
        
    def topo_location_plot(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.topo_location,color='b')
        plt.title('Topography Location')
        plt.xlabel('Image Number')
        plt.ylabel('Topography Location (X-Pixel)')
        plt.pause(5)
        
    def topo_mask(self):
        fdsf
        
def foreground_profile(b_d, f_d, vertical_crop, moving_anom = 'no', moving_abs = 'no'):
    '''
    
    Parameters
    ----------
    b_d : background data
    f_d : foreground_data
  
    moving_anom :if yes will make animation. Default is no
    moving_abs : if yes will make animation. Default is no

    Returns
    -------
    Density data

    '''
    np.seterr(divide='ignore')
   
    
    if not os.path.exists('{}/results'.format(f_d.save_path)):
        os.makedirs('{}/results'.format(f_d.save_path))
    
    
    crop_points=b_d.z-vertical_crop
    ratio = crop_points/b_d.z
    density_abs = np.zeros((f_d.no_images,crop_points,b_d.x))
    #only taking the crop_points closest to the top to save
    for i in range(f_d.no_images):

        f_image=cv2.imread(f_d.foreground_path[i],0)
        f_image_crop=f_image[b_d.zbot:b_d.ztop,:]
        absorbtion = np.log(f_image_crop)

        #getting rid of unwated inf_values and converting to density
        absorbtion[np.isinf(absorbtion)]=np.nan
        density = b_d.rho_bottom+np.float64(b_d.beta*(absorbtion-b_d.bottom_ref))
        
        #putting density data into array
        density_abs[i]=density[:crop_points,:][::-1]  #cropping and flipping data
    
    f_d.density_abs = density_abs
    f_d.ratio=ratio
    print ('\n Done Analysing Images!')  
    
    #plotting anom
    if moving_anom == 'yes':
        
        print('\n Making Anomaly Animation')
        ims=[]
        fig = plt.figure(figsize=(10,5))
        for i in range(f_d.no_images):
            density = density_abs[i]
    
            cmap = cmo.cm.balance
            vmin=-2
            vmax=-vmin
        
            den=density-b_d.rho_ref
            den[den>4]=np.nan
            density_filt=cv2.medianBlur(np.float32(den[50:,:]),3)
        
            plt_depth=b_d.depth
            tank_length=b_d.depth/b_d.z*b_d.x

            im=plt.imshow(density_filt, cmap=cmap, animated=True, vmin=vmin,vmax=vmax, extent=[0,tank_length,-plt_depth,0])
            title = 'Run {}- Density Anomaly'.format(b_d.run)
            plt.title(title, fontsize=20)
            
            plt.xlabel('Length (m)')
            plt.ylabel('Depth (m)')
            
            ims.append([im])
            if i % 25 == 0:
                print('{} of {} Images Done!'.format(i,f_d.no_images))
                
                
        ani = animation.ArtistAnimation(fig, ims, interval=125, blit=True,
                                        repeat_delay=1000)
        
        
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_label(r'Density (kg m$^{-3}$)', rotation=90)
    
        print('Saving!')
        
        writer = animation.writers['ffmpeg']
        save_name = 'run_{}_anomaly'.format(b_d.run)
        ani.save('{}/results/{}.mp4'.format(f_d.save_path,save_name), dpi=250)
        
    #plotting abs    
    if moving_abs == 'yes':
        print('\n Making Abseloute Animation')
        ims=[]
        fig = plt.figure(figsize=(10,5))
        
        for i in range(f_d.no_images):
        
            density = density_abs[i]
            
            cut=600
            plt_depth=b_d.depth/718*(718-cut)
                    
            den=density[:,50:-50]
            den[den>b_d.rho_bottom+2]=np.nan
            density_filt=cv2.medianBlur(np.float32(den[:cut,:]),3)
        
            tank_length=b_d.depth/b_d.z*b_d.x
            
            cmap = cmo.cm.dense
            vmin=b_d.rho_top
            vmax=b_d.rho_top+4
                
          
        
            im=plt.imshow(density_filt, cmap=cmap, animated=True, vmin=vmin,vmax=vmax, extent=[0,tank_length,-plt_depth,0])
            title = 'Run {}- Density'.format(b_d.run)
            plt.title(title, fontsize=20)
            
            plt.xlabel('Length (m)')
            plt.ylabel('Depth (m)')
            
            ims.append([im])
        
            
            if i % 25 == 0:
                print('{} of {} Images Done!'.format(i,f_d.no_images))
                
                
        ani = animation.ArtistAnimation(fig, ims, interval=125, blit=True,
                                        repeat_delay=1000)
        
        
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(im, cax=cax)
        cbar.set_label(r'Density (kg m$^{-3}$)', rotation=90)
        
        
        print('Saving!')
        
        writer = animation.writers['ffmpeg']
        save_name = 'run_{}_abs'.format(b_d.run)
        ani.save('{}/results/{}.mp4'.format(f_d.save_path,save_name), dpi=250)



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
                    


def crop_centre(t_d, b_d, vertical_crop, anom ='no'):
    '''

    Parameters
    ----------
    topo_location : An array that has the average location (X Pixel) of tip of topography for each image in dataset
    field : Dataset that we are cropping (either density abs or density anom)
    rho_ref : Dataset that that has the background density
    fixed_anom :  Generate data for anom

    Returns
    -------
    cropped_abs : The cropped abs with the topography stationairy 
    cropped_anom : The cropped anom with the topography stationairy

    '''

    t,y,x=t_d.density_abs.shape
    crop_points=y-vertical_crop
    right = int(x-max(t_d.topo_location))
    left = int(min(t_d.topo_location))    
    
    cropped_field = np.zeros((t,y,right+left))
    
    if anom == 'yes':
        for j in range(t):
            topo=int(t_d.topo_location[j])
            
            image=t_d.density_abs[j]
            
            cropped_image=image[:,int(topo-left):int(topo+right)]
            cropped_ref=b_d.rho_ref[:crop_points,int(topo-left):int(topo+right)]
            
            delta=cropped_image-cropped_ref
            cropped_field[j] = delta
            
        return cropped_field
        
    else:
        
        for j in range(t):
            topo=int(t_d.topo_location[j])
            
            image=t_d.density_abs[j]
            
            cropped_image=image[:,int(topo-left):int(topo+right)]
            
            
            cropped_field[j]=cropped_image
        
        return cropped_field
    


def centred_field(t_d, b_d, f_d, vertical_crop, fixed_anom, fixed_abs):
    
    centre_rho = crop_centre(t_d, b_d, vertical_crop)
    
    f_d.centre_rho = centre_rho
    
    if not os.path.exists('{}/results'.format(f_d.save_path)):
        os.makedirs('{}/results'.format(f_d.save_path))
    

    if fixed_abs == 'yes':
        print('\n Making Abseloute Animation')
        ims=[]
        fig = plt.figure(figsize=(10,5))
        
        t,y,x=centre_rho.shape
        
        for i in range(t):
            
            image=centre_rho[i]
            
            cmap = cmo.cm.dense
            diff = b_d.rho_bottom-b_d.rho_top
            vmin=b_d.rho_top
            vmax=b_d.rho_top+diff*f_d.ratio
                
        
            im=plt.imshow(image, cmap=cmap, animated=True, vmin=vmin,vmax=vmax)
            title = 'Run {}- Density'.format(b_d.run)
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
        save_name = 'run_{}_abs_centre'.format(b_d.run)
        ani.save('{}/results/{}.mp4'.format(f_d.save_path,save_name), dpi=250)
    
    if fixed_anom == 'yes':
        print('\n Making Anomaly Animation')
        centre_anom = crop_centre(t_d, b_d, vertical_crop, anom='yes')
        
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
            title = 'Run {}- Density Anomaly'.format(b_d.run)
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
        save_name = 'run_{}_anomaly_centre'.format(b_d.run)
        ani.save('{}/results/{}.mp4'.format(f_d.save_path,save_name), dpi=250)
    

                    
def topograghy_mask(t_d, f_d):
    '''
    A function that reads in a dataset and returns a function that can be used to mask
    the topography (needed for fourier transform)

    Parameters
    ----------
    t_d : Topography Data
    f_d :  Foreground data

    Returns
    -------
    topo_function : An funtion in the form of a 1D array that can be used to mask
        the topography.
    rho_c : Cropped dataset (bottom nan's removed')

    '''
    rho = f_d.centre_rho
    
    min_nan=np.min(np.sum(np.isnan(rho[0]),axis=0)) #summing the Nans in the vertical direction
    rho_c = rho[:,:-(min_nan+t_d.bottom_offset),:]
    
    t,y,x = rho_c.shape
    base = rho_c[t//2]
    domain = np.arange(x)
    

    if t_d.no_hills == 1:
        
        height_increase=20 #pixels we want to increase the topo height by (otherwise cuts off sides of the top)
        
        max_amp, max_loc = max_and_loc(base)
        
        h_m = max_amp//2
        h_m_array  = base[-h_m,:]*0+1
        h_m_w = (x-np.nansum(h_m_array))//2+t_d.lensing
        
        max_amp = max_amp + height_increase
        
        topo_function=-max_amp*np.exp(-(domain-max_loc)**2/(2*h_m_w**2))+y
        
        plt.figure()
        plt.imshow(rho_c[0])
        plt.plot(topo_function)
        plt.title('Does Topography Mask Cover Topography?')
        plt.xlabel('Length')
        plt.ylabel('Depth')
        plt.pause(5)
        
        t_d.topo_function = topo_function
        f_d.cropped_centre = rho_c
        
    if t_d.no_hills == 2:
        
        print('Warning Currently Untested')
        
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
        h_m_w = (x-np.nansum(h_m_array))//2+t_d.lensing
        
        topo_function=-max_amp_1*np.exp(-(domain-max_loc_1)**2/(2*h_m_w**2))-max_amp_2*np.exp(-(domain-max_loc_2)**2/(2*h_m_w**2))+y
       
        
        plt.figure()
        plt.imshow(rho_c[0])
        plt.plot(topo_function)
        plt.title('Does Topography Mask Cover Topography?')
        plt.xlabel('Length')
        plt.ylabel('Depth')
        plt.pause(5)
        
        t_d.topo_function = topo_function
        f_d.cropped_centre = rho_c

def transformed_coords(data, topography_mask):
    '''
    A function that calculates the transformed coordinates needed to mask the 
    the topography

    Parameters
    ----------
    data : Data that is being transformed
    Topogrpahy Function

    Returns
    -------
    zt : The transformed array, that contains the z' coordinates

    '''

    
    
    t,z,x=data.shape
    
    #creating data used in transform
    x_array = np.arange(x)
    z_array = np.arange(z)
    xx,zz=np.meshgrid(x_array,z_array)
    
    zt=np.zeros((z,x)) #where we are storing the transformed z coordinate
    
    for i in range(x):
        topo=topography_mask[i]
        transformed_array=z*(zz[:,i]-topo)/(-topo) #function
        zt[:,i]=-np.round(transformed_array)+z
        
  
    return zt

def low_pass_filter(z,x,sigma=0.05,mu=0):
    '''
    Simple function that generates a low pass filter (using a gaussian)

    Parameters
    ----------
    z : height of array
    y : length of array
    sigma : As this uses a guassian filter sigma is width of gaussian which
            coresponds to the strength of filter. The default is 0.1.
    mu : The mean of the gaussian. The default is 0.

    Returns
    -------
    filt : Returns a 2d array containing the low pass filter

    '''
    ratio=.8
    
    x_fft = np.fft.fftfreq(x)
    z_fft = np.fft.fftfreq(z)

    x_filt =np.exp(-(x_fft-mu)**2/(2*(ratio*sigma)**2))
    z_filt =np.exp(-(z_fft-mu)**2/(2*sigma**2))

    filt = x_filt*z_filt[:,None]
    
    return filt

def plot_w(data, run, path):
    
    if not os.path.exists('{}/results'.format(os.path.dirname(path))):
        os.makedirs('{}/results'.format(os.path.dirname(path)))
    
    ims=[]
    fig = plt.figure(figsize=(10,5))
    
    t,y,x=data.shape
    
    for i in range(t):
        
        image=data[i]
        
        cmap = cmo.cm.balance
        vmin=-5e-4
        vmax=-vmin
            
    
        im=plt.imshow(image, cmap=cmap, animated=True, vmin=vmin,vmax=vmax)
        title = 'Run {}- Vertical Velocity'.format(run)
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
    cbar.set_label(r'Vertical Velocity (m s$^{-1}$)', rotation=90)
    
    
    print('Saving!')
    
    writer = animation.writers['ffmpeg']
    save_name = 'run_{}_w_vel'.format(run)
    ani.save('{}/results/{}.mp4'.format(os.path.dirname(path),save_name), dpi=250)

