# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:34:13 2020

@author: u6201343
"""

import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

w = np.load('vertical_test.npy')
t,y,x=w.shape
dwdz =np.gradient(w,axis=1)

tester = dwdz[100]

dummy=np.ones(tester.shape)

nan_array = np.isnan(tester)*dummy
nan_sum=np.sum(nan_array,axis=0)
topo_top = int(max(nan_sum))

topo_inc = dwdz[:,y-topo_top:,:]
topo_free = dwdz[:,:y-topo_top,:]

u=cumtrapz(topo_free, axis=2)