# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:51:04 2020

@author: u6201343
"""
import time
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
import analysis_functions as af

transform_data_path='E:/batch 2/3echo/run8/transit/results/centre_data.npz'
filtered_data_path='E:/batch 2/3echo/run8/transit/results/filtered_data.np'

zt = af.transformed_coords(transform_data_path, return_dataset='no')