# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 23:47:26 2021

@author: lenovo
"""
import pyfeng as pf
import numpy as np
import imp
m = pf.BsmNdMc(np.array([0.1,0.1,0.1,0.1]), cor=0, intr=0.0, divr=0.0, rn_seed=1234)
texp = 5
tobs = np.arange(101)*texp/100
haha = m.simulate(tobs = tobs,n_path = 1000)
