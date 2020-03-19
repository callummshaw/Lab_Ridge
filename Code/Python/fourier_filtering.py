# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:06:42 2020

@author: Callum Shaw
"""
import os
from numpy import fft
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
import time
#import analysis_functions as af
st = time.time()
data_path = 'E:/batch 2/3echo/run8/transit/results/transformation_result.npz'
data = np.load(data_path)

t_data = data['result']
t_data = np.moveaxis(t_data,2,0) #move time to first axis like rest of data
data = t_data[:,1:-1,1:-1] #removing boundary nans
t,z,x = data.shape
#filter:
def low_pass_filter(z,x,sigma=0.1,mu=0):
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

    x_fft = np.fft.fftfreq(x)
    z_fft = np.fft.fftfreq(z)

    x_filt =np.exp(-(x_fft-mu)**2/(2*sigma**2))
    z_filt =np.exp(-(z_fft-mu)**2/(2*sigma**2))

    filt = x_filt*z_filt[:,None]
    
    return filt

filt = low_pass_filter(z, x)

def fourier_filter(i):
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
    cropped = data[i]
   #mean of guassian filter should be kept to zero
    
    #building filter
    
    
    fft_data = fft.fft2(cropped) #preforming fft
    filtered_data = fft_data*filt #filtering
    ifft_data = fft.ifft2(filtered_data) #going back to cartesian space
    
    return np.float32(ifft_data.real)

if __name__ == '__main__':
      numbers=range(t)
      p = Pool()
      result = p.map(fourier_filter, numbers)
      p.close()
      p.join()

result=np.dstack(result)
result = np.moveaxis(result,2,0)
et = time.time()
tt=et-st
print(f'This process took {tt} seconds')

np.savez('{}/filtered_data'.format(os.path.dirname(data_path)),result)