from multiprocessing.pool import ThreadPool as Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

data_path='D:/batch 2/3echo/run8/transit/results/centre_data.npz'
data = np.load(data_path)

rho = data['centre_rho']

#remving nans at base
min_nan=np.min(np.sum(np.isnan(rho[0]),axis=0)) #summing the Nans in the vertical direction
offset = 15

rho_c = rho[:,:-(min_nan+offset),:]
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


#checking topo looks alright
plt.figure(figsize=(10,10))
plt.title('Topography Function')
plt.imshow(rho_c[0])
plt.plot(topo_function)

def transform_and_interp(k):
    data = rho_c[k]
    dummy = np.zeros((z,x)) #variable to store date during transform
    
    #applying the transform on the data
    for i in range(x):
            for j in range(z):
                wanted_data=data[j,i]
                z_loc = int(zt[j,i])
                if z_loc<580:
                    dummy[z_loc,i]=wanted_data
    
    dummy[dummy==0]=np.nan
    
    #now interpolating data
    dummy_frame = pd.DataFrame(dummy)
    dummy_int = dummy_frame.interpolate()
    return dummy_int.values


if __name__ == '__main__':
      numbers=range(t)
      p = Pool()
      result = p.map(transform_and_interp, numbers)
      p.close()
      p.join()

result=np.dstack(result)

np.savez('{}/transformation_result'.format(os.path.dirname(data_path)),result=result)
