# -*- coding: utf-8 -*-
"""
Python version of Callum Shakespeare's mitgcm 2d initial condition
script

@author: Callum Shaw
"""
import numpy as np

#background constants
width=8000

hill_heights=[200, 250, 300, 350, 400]#desired hill heights

save_location= ''

data_type = '>f8'

alpha_t = 2e-4
g = 9.81
n2 = 3e-6

#grid size
nx = 2048
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
c=width/(2*np.sqrt(2*np.log(2))) #fwhm
h1 = np.exp(-(x-lx/2)**2/(c**2))
H=-hz

for i in hill_heights:
    h = H+h1*i
    
    save_name = '{}very_big_topo_h{}.field'.format(save_location, i)
    h.astype(data_type).tofile(save_name)

