#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:34:45 2023

@author: david
"""
import numpy as np
from scipy.signal import convolve2d

from particles import distributions as dists
from particles import state_space_models as ssm


'''
    Bootstrap Model
'''

class Fast_Ice_Model_Bootstrap(ssm.StateSpaceModel):
    
    def __init__(self, grid_size, distance, vel_appr, uncertainty, first_guess, land_mask, season_func, dp, dp_f):
        self.grid_size   = grid_size
        self.distance    = distance
        self.vel_appr    = vel_appr
        self.uncertainty = uncertainty
        self.first_guess = first_guess
        self.land_mask   = land_mask
        self.season_func = season_func
        
        super().__init__(**(dp|dp_f))
    
    
    ###############################
    ## Simulation for next steps ##
    ###############################
    def PX0(self):
        return self.PX(0, self.first_guess.reshape(1,-1))
    
    
    def PX(self, t, xp):
        vals     = self.neighbouring_fast_ice(xp)
        p_f, p_d = self.season_func(vals, t, self.p0, self.p1, self.EPS_u, self.EPS_l,
                                    self.b_0_0, self.b_0_1, self.b_1_0, self.b_1_1,
                                    self.a_0_0f, self.a_0_1f, self.a_1_0f, self.a_1_1f,
                                    self.a_0_0d, self.a_0_1d, self.a_1_0d, self.a_1_1d)
        p = np.where(xp, p_d, p_f)
        p = np.where(self.land_mask, 0, p)
        return dists.IndepProd(*[dists.Binomial(1, pn) for pn in p.T])

        
    def PY(self, t, xp, x):
        mus    = self.vel(t) * x
        mus    = np.where(self.land_mask, -1, mus)
        sigmas = self.variances(t)
        dis    = np.array([dists.Normal(loc=mu, scale=sig) for mu, sig in zip(mus.T, sigmas)])
        return dists.IndepProd(*[dists.MixMissing(pmiss=self.p_miss, 
                                                  base_dist=d) for d in dis])
        

    ############################
    ## Extra Functions needed ##
    ############################
    def vel(self, t):
        return self.vel_appr[t].flatten()
        
    def variances(self, t):
        return self.sigma + self.alpha * self.uncertainty[t] + self.beta * np.square(self.uncertainty[t])
       
    def neighbouring_fast_ice(self, xp):
        return np.stack(np.array([convolve2d(x.reshape(self.grid_size), self.distance, mode='same', boundary='symm').flatten() for x in xp]), axis=0)
        
'''
    Guided Model
'''       
        
        
class Fast_Ice_Model_Guided(Fast_Ice_Model_Bootstrap):
    
    def upper_bound_log_pt(self, t):
        return 0
    
    def proposal0(self, data):
        return self.proposal(0, self.first_guess.reshape(1,-1), data)
    
    def proposal(self, t, xp, data): 
        vals = self.neighbouring_fast_ice(xp)
        p_f, p_d = self.season_func(vals, t, self.p0, self.p1, self.EPS_u, self.EPS_l,
                                    self.b_0_0, self.b_0_1, self.b_1_0, self.b_1_1,
                                    self.a_0_0f, self.a_0_1f, self.a_1_0f, self.a_1_1f,
                                    self.a_0_0d, self.a_0_1d, self.a_1_0d, self.a_1_1d)
        p = np.where(xp, p_d, p_f)
        p = np.where(self.land_mask, 0, p)
        
        transition_zero = self.transition_density(t, data[t], 0)
        transition_one  = self.transition_density(t, data[t], 1)
        # transition = np.where(xp, transition_one, transition_zero)
        
        probs = transition_one * p / ( transition_zero * (1 - p) + transition_one * p)
        probs = np.where(np.isnan(probs), p, probs)
        return dists.IndepProd(*[dists.Binomial(1, pn) for pn in probs.T])
    
    def transition_density(self, t, yt, xt):
        mus    = self.vel(t) * xt
        sigmas = self.variances(t)
        return np.exp( -np.square((yt - mus) / sigmas) / 2) / (sigmas * np.sqrt(2* np.pi))
    