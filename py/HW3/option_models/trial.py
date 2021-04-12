import imp
import numpy as np
import matplotlib.pyplot as plt
import pyfeng as pf
import bsm
import sabr
# Parameters
strike = np.linspace(75,125,num=25)
forward = 100
sigma = 0.2
texp = 1
vov = 0.5
rho = 0.25
beta = 1
sabr_bsm = pf.SabrHagan2002(sigma, vov=vov, rho=rho, beta=beta)
sabr_bsm_mc = sabr.ModelBsmMC(sabr_bsm.sigma, vov=sabr_bsm.vov, rho=sabr_bsm.rho, beta=1)