    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm
import pyfeng as pf
import scipy.integrate as spint

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        if self.option_prices == None:
            self.option_prices = self.price(strike, spot, texp, sigma)
        self.iv = np.zeros(strike.shape[0])
        for i, t_strike in enumerate(strike):
            self.iv[i] = bsm.bsm_impvol(self.option_prices, strike, spot)
        return self.iv
    
    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        option_num = strike.shape[0]
        self.option_prices = np.zeros(option_num)
        dt = 0.01
        time_num = int(texp//dt)
        path_num = 10000
        stock_price_paths = np.zeros((time_num+1, path_num))
        stock_price_paths[0,:] = spot
        sigma_paths = np.zeros((time_num+1, path_num))
        sigma_paths[0,:] = self.sigma
        for i in range(time_num):
            t_row = i+1
            z_1 = np.random.randn(path_num)
            x_1 = np.random.randn(path_num)
            w_1 = self.rho*z_1+np.sqrt(1-self.rho**2)*x_1
            sigma_paths[t_row,:] = sigma_paths[i,:]*np.exp(self.vov*np.sqrt(dt)*z_1-0.5*self.vov**2*dt)
            stock_price_paths[t_row,:] = stock_price_paths[i,:]*np.exp(sigma_paths[i,:]*np.sqrt(dt)*w_1-0.5*sigma_paths[i,:]**2*dt)
        self.stock_price_paths = stock_price_paths
        self.sigma_paths = sigma_paths
        self.option_prices = np.fmax(stock_price_paths[-1,:].reshape(1,-1)-strike.reshape(-1,1),0).mean(axis=1)
        return self.option_prices
    
'''stock_price_paths = np.array([[1,1,1,1,1,1],[1,2,3,4,5,6]])
strike = np.array([2,3,4,5])
tt = stock_price_paths[-1,:].reshape(1,-1)-strike.reshape(-1,1)
'''

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        return 0

        
    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        option_num = strike.shape[0]
        self.option_prices = np.zeros(option_num)
        dt = 0.01
        time_num = int(texp // dt)
        path_num = 10000
        stock_price_paths = np.zeros((time_num + 1, path_num))
        stock_price_paths[0, :] = spot
        sigma_paths = np.zeros((time_num + 1, path_num))
        sigma_paths[0, :] = self.sigma
        for i in range(time_num):
            t_row = i + 1
            z_1 = np.random.randn(path_num)
            x_1 = np.random.randn(path_num)
            w_1 = self.rho * z_1 + np.sqrt(1 - self.rho ** 2) * x_1
            sigma_paths[t_row, :] = sigma_paths[i, :] * np.exp(self.vov * np.sqrt(dt) * z_1 - 0.5 * self.vov ** 2 * dt)
            stock_price_paths[t_row, :] = stock_price_paths[i, :] + sigma_paths[i, :] * np.sqrt(dt) * w_1
        self.stock_price_paths = stock_price_paths
        self.sigma_paths = sigma_paths
        self.option_prices = np.fmax(stock_price_paths[-1, :].reshape(1, -1) - strike.reshape(-1, 1), 0).mean(axis=1)
        return self.option_prices

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        return 0
    
    def price(self, strike, spot, texp, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        option_num = strike.shape[0]
        self.option_prices = np.zeros(option_num)
        dt = 0.01
        time_num = int(texp // dt)
        path_num = 10000
        stock_price_paths = np.zeros((time_num + 1, path_num))
        stock_price_paths[0, :] = spot
        sigma_paths = np.zeros((time_num + 1, path_num))
        sigma_paths[0, :] = self.sigma
        for i in range(time_num):
            t_row = i + 1
            z_1 = np.random.randn(path_num)
            sigma_paths[t_row, :] = sigma_paths[i, :] * np.exp(self.vov * np.sqrt(dt) * z_1 - 0.5 * self.vov ** 2 * dt)
        self.I_T = spint.simps(sigma_paths**2, dx = dt, axis=0)
        self.stock_prices = spot * np.exp(self.rho/self.vov*(sigma_paths[-1, :]-self.sigma)-
                                          self.rho**2/2*self.I_T)
        self.sigma_bs = np.sqrt((1-self.rho**2)*self.I_T/texp)
        self.sigma_paths = sigma_paths
        self.option_prices = self.bsm_model()
        return self.option_prices
'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        return 0
        
    def price(self, strike, spot, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        np.random.seed(12345)
        return 0