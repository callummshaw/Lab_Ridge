# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:06:42 2020

@author: Callum Shaw
"""
from numpy import fft
import numpy as np
#import analysis_functions as af

data_path = 'D:/batch 2/3echo/run8/transit/results/transformation_result.npz'
data = np.load(data_path)

t_data = data['result']
t_data = np.moveaxis(t_data,2,0) #move time to first axis like rest of data
data = t_data[:,1:-1,1:-1] #removing boundary nans

def fourier_filter(data,sigma=.1):
    '''
    A function the preforms low pass filter, removing all the high frequency
    noise present in the data (using a gaussian filter).

    Parameters
    ----------
    data : Data set that needs to be filtered
    sigma : As this uses a guassian filter sigma is width of gaussian which
            coresponds to the strength of filter

    Returns
    -------
    Data set that has undergone low pass filter

    '''
    mu=0 #mean of guassian filter should be kept to zero
    
    #building filter
    t,z,x = data.shape
    x_fft = np.fft.fftfreq(x)
    z_fft = np.fft.fftfreq(z)
    x_filt =np.exp(-(x_fft-mu)**2/(2*sigma**2))
    z_filt =np.exp(-(z_fft-mu)**2/(2*sigma**2))
    filt = x_filt*z_filt[:,None]
    
    fft_data = fft.fft2(data) #preforming fft
    filtered_data = fft_data*filt #filtering
    ifft_data = fft.ifft2(filtered_data) #going back to cartesian space
    
    return ifft_data.real

