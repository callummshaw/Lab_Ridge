# -*- coding: utf-8 -*-
"""
Python version of Callum Shakespeare's mitgcm 2d initial condition
script

@author: Callum Shaw
"""
import numpy as np

#background constants
save_location= '/g/data/nm03/mitgcm/input/'

data_type = '>f8'

alpha_t = 2e-4
g = 9.81
n2 = 3e-6

#grid size
nx = 1024
ny = 1
nz = 2000

#grid lengths
lx = 100e3
hz = 3000

#grid arrays
z = -np.linspace(0,hz,nz+1)
z = (z[:-1]+z[1:])/2

x = np.linspace(0,lx,nx+1)
x = (x[:-1]+x[1:])/2

X, Z = np.meshgrid(x,z)

#Initial Temp (constant strat)
dtdz = n2/(alpha_t*g)
T = dtdz*(Z+hz)

save_name ='{}Tinit_N2_3e6.field'.format(save_location)
T.astype(data_type).tofile(save_name)

#Initial Velocity
omega0=0.000138889
f0=1e-4
F0=1e-6
u0=omega0/(f0**2-omega0**2)*F0
u_init = u0*(X*0+1)

save_name = '{}uinit_super.field'.format(save_location)
u_init.astype(data_type).tofile(save_name)

# Making hills
h1=np.cos(np.pi*(x-lx/2)/lx)**2
h1=h1-np.mean(h1)
h1=h1/np.std(h1) 
h1=h1-np.min(h1) 

H=np.min(z)
hill_heights=[200, 350, 500, 650,]#desired hill heights

for i in hill_heights:
    h = H+h1*i
    
    save_name = '{}very_big_topo_h{}.field'.format(save_location, i)
    h.astype(data_type).tofile(save_name)

