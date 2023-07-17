#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:08:22 2023

@author: david
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:50:12 2023

@author: david
"""

import particles as pt
from particles import distributions as dists
from particles import state_space_models as ssm
from particles.collectors import Moments
from particles import mcmc

# Drift Simulations
import simulation as sim
import models as mo

# Some imports
import numpy as np

import scipy.interpolate as sci
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d

# For neighbourhood checking:
h2 = 1 / 2
s2 = 1 / np.sqrt(2)
s5 = 1 / np.sqrt(5)
s8 = 1 / np.sqrt(8)
# euclid distance 5x5
n_complex = np.array([[s8, s5, h2, s5, s8],
                      [s5, s2,  1, s2, s5],
                      [h2,  1,  0,  1, h2],
                      [s5, s2,  1, s2, s5],
                      [s8, s5, h2, s5, s8]])

n_complex /= np.sum(n_complex)



class Handler():
    
    def __init__(self, grid_size=(50,65), T=50):
        '''
            initializes all variables / list of all variables
        '''
        # --------------
        # Main variables
        self.grid_size = grid_size
        self.y, self.x = grid_size
        self.total_xs  = self.y * self.x
        self.T         = T
        
        self.t_0, self.t_1 = 0, 1
        
        # ---------------------
        # simulation parameters
        self.seed        = 0
        self.amountFI    = 0
        self.amountLand  = 0
        self.fluctuation = 0
        self.noise       = 0 
        self.errorRate   = 0
        self.arealError  = 0
        self.clustered   = 0
        self.maxSpeed    = 0
        
        self.simulation = []
        self.fast_ice   = []
        
        self.sim = None
        
        # Velocity approximation
        self.vel_appr = []
        
        # constans
        self.LAND_PIX = -1
        
        # -----------------------
        # For Monte Carlo - basis
        self.default_params      = {} # inside the model
        self.default_params_func = {} # only for the outer likelihood function
        # Everything else
        self.distance  = n_complex
        self.land_mask = None
        self.vel_appr  = None
        
        # the basis for the 2 Feyman_Kac Models (Bootstrap and Guided)
        self.Model_Bootstrap = None
        self.Model_Guided    = None
        
        # the (forward) program
        self.my_alg = None             # forward
        self.ssm_guide_backward = None # the backward filtering, if needed
        
        # the actual outcomes
        self.filtering_outcome = {} # filtering
        self.smoothing_outcome = [] # smoothing, only mean, no variance!
        
        # --------------------
        # parameter estimation
        self.prior_dists = {}
        self.estimated   = None
        
        # ----------
        # the errors
        self.mean_square = 0
        self.true_true_abs  = 0
        self.true_true_perc = 0
        
        self.true_false_abs  = 0
        self.true_false_perc = 0
        
        self.false_true_abs  = 0
        self.false_true_perc = 0
        
        self.false_false_abs  = 0
        self.false_false_perc = 0
        
        self.true_fi_abs  = 0
        self.true_fi_perc = 0

        self.detected_fi_abs  = 0
        self.detected_fi_perc = 0
    
        
    #################################################
    #################################################
    #################################################    
    #################
    ## MONTE CARLO ##
    #################
    #################################################
    #################################################
    #################################################
    
    def smoothing(self, N, default_par_model, default_par_function, verbose=False, boots=False, already_filtered=False, seed=None, smoothing='two_filter', M=50):
        '''
            Does a smoothing depending on the algorithm chosen (first always forward-filtering).
            Every param is explained in different functions, M is the amount of appr. used for loggamma function in two_filter
        '''
        if not already_filtered:
            if verbose: print('------------- \n start filtering')
            self.filtering(N, default_par_model, default_par_function, verbose=verbose, store_history=True, boots=boots, seed=seed)
        
        if verbose: print('\n------------- \n start smoothing')
        if smoothing=='mcmc':
            paths = self.my_alg.hist.backward_sampling_mcmc(M=N)
            self.smoothing_outcome     = [np.mean(p, axis=0) for p in paths]
            
        if smoothing=='reject':
            paths = self.my_alg.hist.backward_sampling_reject(M=N)
            self.smoothing_outcome     = [np.mean(p, axis=0) for p in paths]

        if smoothing=='two_filter':
            if verbose: print('\n------------- \n start backward filtering')
            my_alg_backward = self._backward_filtering(N=N, verbose=verbose) # backward filtering
            # actual smoothing
            phi = lambda x,y: x
            if verbose: print('\n------------- \n start actual smoothing')
            for t in np.arange(self.T-1):
                if verbose: print(f'{t+1}/{self.T-1}')
                ti       = self.T - 2 - t
                loggamma = self._loggamma(ti, M)
                path = self.my_alg.hist.two_filter_smoothing(t, my_alg_backward, phi, loggamma, linear_cost=True)
                self.smoothing_outcome.append(path)
            # Add final decision (T)
            self.smoothing_outcome.append(self.filtering_outcome['mean'][-1])
                
        self.smoothing_outcome = np.array(self.smoothing_outcome)
        if verbose: print('FINISHED!')
            
        
    
    def filtering(self, N, default_par_model, default_par_function, verbose=False, store_history=False, boots=False, seed=None):
        '''
            Does a filtering of seed=seed, see self.create_model doc for infos on params.
        '''
        self.create_model(default_par_model, default_par_function, seed=seed)
        _ = self.particles(N=N, verbose=verbose, store_history=store_history, boots=boots)
        
        self.my_alg.run()
        
        self.filtering_outcome['mean'] = [s['mean'] for s in self.my_alg.summaries.moments]
        self.filtering_outcome['var']  = [s['var']  for s in self.my_alg.summaries.moments]
        
    
    def _backward_filtering(self, N, verbose=False):
        '''
            Does a backwards filtering, always Guided in this case!  # CHANGE
        '''
        first_guess = self._define_first_guess(self.fast_ice[-1])
        
        
        model_guided_backward = mo.Fast_Ice_Model_Guided(grid_size=self.grid_size, distance=self.distance, vel_appr=self.vel_appr[::-1,:,:], uncertainty=self.uncertainty, first_guess=first_guess.flatten(),
                                                       land_mask=self.land_mask.flatten(), season_func=self.season_func_back,
                                                       dp=self.default_params, dp_f=self.default_params_func)
        self.ssm_guide_backward = ssm.GuidedPF(ssm=model_guided_backward, data=self.data[::-1,:,:])
        my_alg_backward    = pt.SMC(fk=self.ssm_guide_backward, N=N, verbose=verbose, store_history=True, collect=[Moments()])
        my_alg_backward.run()
        
        return my_alg_backward

    def _loggamma(self, ti, M):
        '''
            returns the loggamma function for the two-filter smoothing for step t and appr. M
        '''
        def loggamma(xti):
            s = np.zeros(len(xti))
            for i in range(M):
                xt  = self.ssm_guide_backward.ssm.proposal(ti, xti, self.ssm_guide_backward.data).rvs()
                ft  = self.ssm_guide_backward.ssm.PY(ti, None, xt)
                val = ft.logpdf(self.ssm_guide_backward.data[ti,:,:])
                s   += val
            return s / M
        
        return loggamma
    
    
    def create_model(self, default_par_model, default_par_function, seed=None):
        '''
            Creates the base models to work with Monte Carlo simulations

        Parameters
        ----------
        Following parameters should be set in default_par_model (DISCTIONARY)
            p_miss : float -> the coverage of missing values. The default is .3.
            sigma  : float -> the standard deviation if the observation is NOT missing. The default is .2.
            alpha  : float -> linear dependence of regional uncertainty. Default is .3.
            beta   : float -> quadratic dependenc --||--. Default is .1.
            

        Following parameters should be set in default_function_par (DISCTIONARY)
            pk           : float [0,oo] -> exponent of the function
            EPS_u, EPS_l : float [0,1]  -> minimal and maximal (1-eps) value of the function
            b_i_J        : float [0,1]  -> values of the function at the thresholds a_i_j d/f
            a_i_j d/f    : float [0,1]  -> threshold for the shift in function 
            
            for i=0,1; j=0,1
        
        seed : int, optional
            the seed for generation of the fast ice environment. The default is None -> random.
        '''
        # set the seed for simulation
        self.set_simulation_parameters(amountFI=self.amountFI, amountLand=self.amountLand, fluctuation=self.fluctuation, noise=self.noise, 
                                      errorRate=self.errorRate, arealError=self.arealError, clustered=self.clustered, maxSpeed=self.maxSpeed, seed=seed)
        # simulate the fast ice
        self.simulate()
        self.vel_appr    = np.array([self.vel_approx(vel, sigma=1) for vel in self.simulation[1:-1]])  
        self.uncertainty = np.array([self.regional_uncertainty(vel).flatten() for vel in self.simulation[1:-1]])  
        self.data        = self.simulation[1:-1].reshape((self.T, 1, self.total_xs))

        # compute the time parameters
        self.determine_t0_t1()
        
        
        # set the fast ice base-parameters for the sampling models
        defaults_model = {'p_miss' : .3,
                          'sigma'  : .2,
                          'alpha'  : .3,
                          'beta'   : .1}
        self.default_params = default_par_model
        # check
        for key, val in defaults_model.items():
            if key not in self.default_params: self.default_params[key] = val
        
        defaults_function = {'p0' : 1, 'p1' : 1,
                             'EPS_u' : .0005, 'EPS_l' : .0005,
                             'b_0_0' : .050, 'b_0_1' : .900, 'b_1_0': .100, 'b_1_1' : .95, 
                             'a_0_0f' : .50, 'a_0_1f' : .20, 'a_1_0f' : .9, 'a_1_1f' : .7,
                             'a_0_0d' : .35, 'a_0_1d' : .15, 'a_1_0d' : .9, 'a_1_1d' : .6}
        self.default_params_func = default_par_function
        # check
        for key, val in defaults_function.items():
            if key not in self.default_params_func: self.default_params_func[key] = val
                              
        
        # creates the first guess for the modelling
        first_guess = self._define_first_guess(self.fast_ice[0]).flatten()
        
        self.Model_Bootstrap = mo.Fast_Ice_Model_Bootstrap(grid_size=self.grid_size, distance=self.distance, vel_appr=self.vel_appr, uncertainty=self.uncertainty, first_guess=first_guess,
                                                        land_mask=self.land_mask.flatten(), season_func=self.season_func, 
                                                        dp=self.default_params, dp_f=self.default_params_func)
        
        self.Model_Guided   = mo.Fast_Ice_Model_Guided(grid_size=self.grid_size, vel_appr=self.vel_appr, uncertainty=self.uncertainty, first_guess=first_guess,
                                                    land_mask=self.land_mask.flatten(), season_func=self.season_func, distance=self.distance, 
                                                    dp=self.default_params, dp_f=self.default_params_func)
                                                    
        
        
    def particles(self, N, verbose=False, store_history=False, boots=False):
        if boots:
            ssm_boots = ssm.Bootstrap(ssm=self.Model_Bootstrap, data=self.data)
            self.my_alg = pt.SMC(fk=ssm_boots, N=N, verbose=verbose, store_history=store_history, collect=[Moments()])
        else:
            ssm_guide = ssm.GuidedPF(ssm=self.Model_Guided, data=self.data)
            self.my_alg = pt.SMC(fk=ssm_guide, N=N, verbose=verbose, store_history=store_history, collect=[Moments()])
        return self.my_alg
    
    
    
    def _define_first_guess(self, observation):
        '''
            creates the first guess for the programm, for now it's the actual fast ice (so param 'observation' is just the actual fast ice)
        '''
        return 1 - (observation + self.land_mask)
    
    
    
    #################################################
    #################################################
    #################################################    
    ##########################
    ## PARAMETER ESTIMATION ##
    ##########################
    #################################################
    #################################################
    #################################################
    def parameter_estimation(self, params, n_part=200, n_iter=1000, verbose=0, rw_cov=None, starting_guess=None, adaptive=True):
        '''
            estimates the default parameters for this one model (needs to be done again if the obersavtions change).
            Needs to run create_params first!
            Parameters:
                params is a dictionary of these parameters with their corresponding prior distribution. BOTH the 'default_params' and 'default_params_func'
                n_part number of parts each iteration
                n_iter number of mcmc iterations
                verbose print information 0=false.
        '''
        p_miss = .3 # irrelevant parameter
        grid_size = self.grid_size
        
        self.prior_dists = params
        
        # local references for the Guided Model
        distance = self.distance
        vel_appr = self.vel_appr
        uncertainty = self.uncertainty
        first_guess = self._define_first_guess(self.fast_ice[0]).flatten()
        land_mask   = self.land_mask.flatten()
        season_func = self.season_func
        
        ### Add a local Model here, and declare local variables for reference.
        class Fast_Ice_Model_Bootstrap_local(ssm.StateSpaceModel):
            
            ###############################
            ## Simulation for next steps ##
            ###############################
            def PX0(self):
                return self.PX(0, first_guess.reshape(1,-1))
            
            
            def PX(self, t, xp):
                vals     = self.neighbouring_fast_ice(xp)
                p_f, p_d = season_func(vals, t, self.p0, self.p1, self.EPS_u, self.EPS_l,
                                            self.b_0_0, self.b_0_1, self.b_1_0, self.b_1_1,
                                            self.a_0_0f, self.a_0_1f, self.a_1_0f, self.a_1_1f,
                                            self.a_0_0d, self.a_0_1d, self.a_1_0d, self.a_1_1d)
                p = np.where(xp, p_d, p_f)
                p = np.where(land_mask, 0, p)
                return dists.IndepProd(*[dists.Binomial(1, pn) for pn in p.T])

                
            def PY(self, t, xp, x):
                mus    = self.vel(t) * x
                mus    = np.where(land_mask, -1, mus)
                sigmas = self.variances(t)
                sigmas = np.where(land_mask, 10e-10, sigmas)
                dis    = np.array([dists.Normal(loc=mu, scale=sig) for mu, sig in zip(mus.T, sigmas)])
                return dists.IndepProd(*[dists.MixMissing(pmiss=p_miss, 
                                                          base_dist=d) for d in dis])
                

            ############################
            ## Extra Functions needed ##
            ############################
            def vel(self, t):
                return vel_appr[t].flatten()
                
            def variances(self, t):
                return self.sigma + self.alpha * uncertainty[t].flatten() + self.beta * np.square(uncertainty[t])
               
            def neighbouring_fast_ice(self, xp):
                return np.stack(np.array([convolve2d(x.reshape(grid_size), distance, mode='same', boundary='symm').flatten() for x in xp]), axis=0)
                
        '''
            Guided Model
        '''       
                
                
        class Fast_Ice_Model_Guided_local(Fast_Ice_Model_Bootstrap_local):
            
            def upper_bound_log_pt(self, t):
                return 0
            
            def proposal0(self, data):
                return self.PX0()   # Is already good enough, since it is approximated via the -1st observation
            
            def proposal(self, t, xp, data): 
                vals = self.neighbouring_fast_ice(xp)
                p_f, p_d = season_func(vals, t, self.p0, self.p1, self.EPS_u, self.EPS_l,
                                            self.b_0_0, self.b_0_1, self.b_1_0, self.b_1_1,
                                            self.a_0_0f, self.a_0_1f, self.a_1_0f, self.a_1_1f,
                                            self.a_0_0d, self.a_0_1d, self.a_1_0d, self.a_1_1d)
                p = np.where(xp, p_d, p_f)
                p = np.where(land_mask, 0, p)
                
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
            
        self.estimation = mcmc.PMMH(niter=n_iter, verbose=verbose, ssm_cls=Fast_Ice_Model_Guided_local, prior=self.prior_dists, 
                                    data=self.data, Nx=n_part, fk_cls=ssm.GuidedPF, rw_cov=rw_cov, theta0=starting_guess, adaptive=adaptive)
        
        self.estimation.run()
        
        
        
    
    
    
    #################################################
    #################################################
    #################################################
    ## COMPARISON ##
    #################################################
    #################################################
    #################################################
    
    def transform_results(self, results, thresh_sure_fi=.2, thresh_sure_di=.8, thresh_diff=.5, thresh_fi=.5):
        '''
        Transforms the 'mean' results into {0,1} decisions depending on the sorrounding of every pixel. Also changes
        the 0 is fast ice into 1 is fast ice

        Parameters
        ----------
        results : np.array, shape=(T, y,x)
            The results of either a filtering or smoothing step
        thresh_sure_fi : float, in [0,1]
            threshhold for which fast ice is assumed to be true
        thresh_diff : thresh_diff, in [0,1], optional
            flags extreme values if the surrounding values are this distance away from the origin value. Default is .5.
        thresh_diff : thresh_diff, in [0,1], optional
            flags extreme values if the surrounding values are this distance away from the origin value. Default is .5.
        thresh_fi ; float, in [0,1], optional
            changes the mean values according to this threshhold. Default is .5
        
        Returns
        -------
        None.

        '''
        #e = 1/8
        #matrix = np.array([[e,e,e], [e,0,e], [e,e,e]])
        
        final = []
        
        #results = np.where(results <= thresh_sure_fi, 0, results)
        #results = np.where(results >= thresh_sure_di, 1, results)
        for result in results:
            result = result.reshape(self.grid_size)
            
            #surroundings = convolve2d(result, matrix, mode='same', boundary='symm')
            #diff         = np.abs(result - surroundings)
            #potential    = np.where(diff >= thresh_diff)
            
            #result[potential] = np.where(surroundings[potential] < .5, 0, 1)
            result = np.where(result < thresh_fi, 1, 0)
            # changes extreme cases into the surrounding value.
            fin = np.where(self.land_mask, 0, result)
            
            final.append(fin)
            
        return np.array(final)
    
    ###########
    ## ERROR ##
    ###########
    
    def compute_errors(self, detected):
        land_pixels   = np.sum(self.land_mask)
        total_xs      = self.total_xs - land_pixels
        fast_ice_here = self.fast_ice[1:-1]

        # true ice
        self.true_fi_abs  = np.sum(fast_ice_here, axis=(1,2))
        self.true_fi_perc = self.true_fi_abs / total_xs

        # detected ice
        self.detected_fi_abs  = np.sum(detected, axis=(1,2))
        self.detected_fi_perc = self.detected_fi_abs / total_xs

        ## error values
        # mean square
        self.mean_square = np.mean(np.square(fast_ice_here - detected), axis=(1,2))

        # true / false
        self.true_true_abs  = np.sum(np.where(fast_ice_here + detected == 2, 1, 0), axis=(1,2))
        self.true_true_perc = self.true_true_abs / self.true_fi_abs 

        self.false_true_abs  = np.sum(np.where(fast_ice_here - detected < 0, 1, 0), axis=(1,2))
        self.false_true_perc = self.false_true_abs/ (total_xs - self.true_fi_abs)

        self.true_false_abs  = np.sum(np.where(fast_ice_here + detected == 0, 1, 0), axis=(1,2))
        self.true_false_perc = self.true_false_abs / (total_xs - self.true_fi_abs)

        self.false_false_abs  = np.sum(np.where(detected - fast_ice_here < 0, 1, 0), axis=(1,2))
        self.false_false_perc = self.false_false_abs / self.true_fi_abs 

    
    
    
    
    
    #################################################
    #################################################
    #################################################
    ################
    ## SIMULATION ##
    ################
    #################################################
    #################################################
    #################################################
    
    def set_simulation_parameters(self, amountFI=.18, amountLand=.1, fluctuation=.005, noise=.07, 
                                  errorRate=.3, arealError=.9, clustered=2.5, maxSpeed=5, seed=None):
        '''
            See Simulation documentation for details!
        '''
        self.seed        = seed
        self.amountFI    = amountFI
        self.amountLand  = amountLand
        self.fluctuation = fluctuation
        self.noise       = noise
        self.errorRate   = errorRate
        self.arealError  = arealError
        self.clustered   = clustered
        self.maxSpeed    = maxSpeed
        
        self.sim =  sim.Simulate_Observations(shape=self.grid_size, cycleLength=self.T+2, amountFI=self.amountFI, amountLand=self.amountLand, 
                                  fluctuation=self.fluctuation, noise=self.noise, errorRate=self.errorRate, 
                                  arealError=self.arealError, clustered=self.clustered, maxSpeed=self.maxSpeed, seed=self.seed)
    
    
    def simulate(self):
        '''
            Simulates a timeline corresponding to the parameters.
        '''
        simu, fast_ice = [], []
        for i in np.arange(self.T+2):
            s, fi = self.sim.simulate()
            simu.append(s)
            fast_ice.append(fi)

        self.simulation = np.array(simu)
        self.fast_ice   = np.array(fast_ice)
        self.land_mask  = np.where(self.simulation[0] == self.LAND_PIX, 1, 0)


    #################################################
    #################################################
    ###########################
    ## PREPARATION FUNCTIONS ##
    ###########################
    #################################################
    #################################################

    def vel_approx(self, vels, sigma=1):
        '''
            creates the approximation for the velocity observations, sigma determines the smoothing degree.
        '''
        no_land = np.where(vels == self.LAND_PIX, 0, vels)
        valid = np.where(np.logical_not(np.isnan(no_land.flatten())))
        pY = np.repeat(np.arange(self.y), self.x)
        pX = np.tile(np.arange(self.x), self.y)
        vals  = no_land.flatten()[valid]
        inter = sci.LinearNDInterpolator(list(zip(pX[valid], pY[valid])), vals)
        Z = inter(np.stack([pX, pY]).T).reshape(self.grid_size)
        Z_gaussian = gaussian_filter(Z, sigma=sigma, mode='nearest')
        return Z_gaussian
    
    
    def regional_uncertainty(self, vels):
        '''
            computes the relative uncertainty in every region, returns values in [0,1]. Used for the variance in f_t(y_t | x_t)
        '''
        
        ones_zeros = np.where(np.isnan(vels), 1, 0)
        return convolve2d(ones_zeros, np.ones((3,3)), mode='same', boundary='symm') / 9
        
    
    
    def determine_t0_t1(self, tau=0.2, eps=0.025):
        '''
            Determines the two time parameters depending on the 'world' see chapter 2.5.4 for mathematical details and description of parameters.

        '''
        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        def extent_zero(x, N=1):
            return np.concatenate([np.zeros(N), np.array(x), np.zeros(N)])


        def r(tau):
            T = np.sum( np.where(np.isnan(self.data), 0, 1), axis=(1,2) )
            
            ice  = np.where(self.data >= 0, 1, 0)
            slow = np.where(self.data <= tau, 1, 0)
            
            t = np.sum( np.logical_and(ice, slow), axis=(1,2) )
            
            return t / T
        
        rho = extent_zero(running_mean(r(tau), 3), 1)
        # two start values a, b
        a = np.argmax(rho >= eps)
        b = self.T - np.argmax(rho[::-1] >= eps) - 1
        
        # numerical integration and determining of c
        num_int = np.trapz(rho[a:b], np.arange(a, b))
        c       = num_int / (b - a)
        
        # t0 and t1
        t_0 = np.argmax(rho >= c)
        t_1 = self.T - np.argmax(rho[::-1] >= c) - 1
        
        self.t_0 = t_0 / self.T
        self.t_1 = t_1 / self.T
    
    
    
    #################################################
    #################################################
    ##########################
    ## FOR THE MODEL ITSELF ##
    ##########################
    #################################################
    #################################################
    def season_func_back(self, x_vals, t, p0, p1, EPS_u, EPS_l, b_0_0, b_0_1, b_1_0, b_1_1, a_0_0f, a_0_1f, a_1_0f, a_1_1f, a_0_0d, a_0_1d, a_1_0d, a_1_1d):
        return self.season_func(x_vals, t, p0, p1, EPS_u, EPS_l, b_0_0, b_0_1, b_1_0, b_1_1, a_0_0f, a_0_1f, a_1_0f, a_1_1f, a_0_0d, a_0_1d, a_1_0d, a_1_1d, reverse=True)
    
    def season_func(self, x_vals, t, p0, p1, EPS_u, EPS_l, b_0_0, b_0_1, b_1_0, b_1_1, a_0_0f, a_0_1f, a_1_0f, a_1_1f, a_0_0d, a_0_1d, a_1_0d, a_1_1d, reverse=False):
        '''
            determines the values p(x_t | x_{t-1}) corresponding to their neighbourhood
        '''
        t_0 = self.t_0
        t_1 = self.t_1
        if reverse:
            t_0 = 1 - self.t_1
            t_1 = 1 - self.t_0
        
        def create_at(a_0_0f, a_0_1f, a_1_0f, a_1_1f, a_0_0d, a_0_1d, a_1_0d, a_1_1d):
            def a_0(t):
                perc   = np.clip((t/self.T-t_0) / (t_1 - t_0), 0, 1)
                t_lower = self.lerp(a_0_0f, a_0_1f, self.inter(perc))
                t_upper = self.lerp(a_1_0f, a_1_1f, self.inter(perc))
                
                return t_lower, t_upper
            
            def a_1(t):
                perc   = np.clip((t/self.T-t_0) / (t_1 - t_0), 0, 1)
                t_lower = self.lerp(a_0_0d, a_0_1d, self.inter(perc))
                t_upper = self.lerp(a_1_0d, a_1_1d, self.inter(perc))
                
                return t_lower, t_upper
            return a_0, a_1
        
        def _xp(vals, t, a, b_0, b_1):
            t_lower, t_upper = a(t)
            
            # linear interpolation of saved values
            x_vals = np.array([0, t_lower, t_upper, 1])
            y_vals = np.array([0, b_0, b_1, 1])
            
            final = np.interp(vals, x_vals, y_vals, left=0, right=1)
            return final
        
        def trafo(xvals, pi):
            return np.power(xvals, pi) * (1 - (EPS_u + EPS_l)) + EPS_u
        
        
        a_0, a_1 = create_at(a_0_0f, a_0_1f, a_1_0f, a_1_1f, a_0_0d, a_0_1d, a_1_0d, a_1_1d)

        pf = trafo(_xp(x_vals, t, a_0, b_0_0, b_0_1), p0)
        pd = trafo(_xp(x_vals, t, a_1, b_1_0, b_1_1), p1)
        return pf, pd
    
    
    
    #############
    ## HELPERS ##
    #############
    def inter(self, t):
        '''
            Interpolation function
        '''
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def lerp(self, a, b, x):
        '''
            Linear interpolation
        '''
        return a + x * (b - a)
    

    



































