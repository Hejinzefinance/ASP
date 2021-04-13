#%% import modules
import imp
import numpy as np
import matplotlib.pyplot as plt
import pyfeng as pf
import option_models as opt
#%% 1.1.1
strike = np.linspace(75,125,num=25)
forward = 100
sigma = 0.2
texp = 1
vov = 0.5
rho = 0.25
beta = 1
sabr_bsm = pf.SabrHagan2002(sigma, vov=vov, rho=rho, beta=beta)
price = sabr_bsm.price(strike, forward, texp)
bsm_vol = sabr_bsm.vol_smile(strike, forward, texp)
#%% 1.1.2
strike = np.linspace(75,125,num=25)
forward = 100
sigma = 20
texp = 1
alpha = 0.5
rho = -0.25
sabr_norm = pf.SabrHagan2002(sigma, vov=vov, rho=rho, beta=0)
price = sabr_norm.price(strike, forward, texp)
nvol = sabr_norm.vol_smile(strike, forward, texp, model='norm')
#%% 1.1.3
strike3 = np.array([90, 100, 110])
price3 = sabr_bsm.price(strike3, forward, texp)
vol3 = sabr_bsm.vol_smile(strike3, forward, texp)
sabr_bsm.calibrate3(vol3, strike3, forward, texp, is_vol=True)
sabr_bsm.calibrate3(price3, strike3, forward, texp, is_vol=False)
#%% 1.2
strike = np.linspace(75,125,num=25)
forward = 100
sabr_bsm_mc = opt.sabr.ModelBsmMC(sabr_bsm.sigma, vov=sabr_bsm.vov, rho=sabr_bsm.rho, beta=1)
price_hagan = sabr_bsm.price(strike, forward, texp)
price_mc = sabr_bsm_mc.price(strike, forward, texp)
#%%
sabr_norm.sigma
sabr_norm_mc = opt.sabr.ModelNormalMC(sabr_norm.sigma, vov=sabr_norm.vov, rho=sabr_norm.rho)
price_hagan = sabr_norm.price(strike, forward, texp)
price_mc = sabr_norm_mc.price(strike, forward, texp)
#%%
strike = np.linspace(75,125,num=25)
forward = 100
print(sabr_bsm.__dict__)
sabr_bsm_cmc = opt.sabr.ModelBsmCondMC(sabr_bsm.sigma, vov=sabr_bsm.vov, rho=sabr_bsm.rho, beta=1)
price_hagan = sabr_bsm.price(strike, forward, texp)
price_mc = sabr_bsm_cmc.price(strike, forward, texp)
# make sure the two prices are similar
#%% 1.4
sabr_norm_cmc = opt.sabr.ModelNormalCondMC(sabr_norm.sigma, vov=sabr_norm.vov, rho=sabr_norm.rho, beta=0)
price_hagan = sabr_norm.price(strike, forward, texp)
price_mc = sabr_norm_cmc.price(strike, forward, texp)
tt = 1