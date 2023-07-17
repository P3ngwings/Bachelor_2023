#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:28:08 2023

@author: david
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:12:47 2023

@author: david
"""

# Imports
import numpy as np

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

import Perlin_Noise as pn

  
class Simulate_Observations():
    
    def __init__(self, shape, cycleLength, amountFI=.2, amountLand=.1, fluctuation=.005, noise=.03, errorRate=0, arealError=.9, clustered=2, maxSpeed=5, steepness=23, seed=None, no_starting_error=False):
        '''
        Generates velocities which are distributed random but clustered and change they're appearance over time.
        For fast ice simulation, so there is some greater areas with 'no' movement.

        Parameters
        ----------
        shape : tuple
            Size of grid to simulate (y,x).
        cycleLength : float
            Time period over which the ice freezes and start breaking again.
        amountFI : float in [0,1], optional
            ~percentage of fast ice when fully 'grown'. The default is 0.2
        amountLand : float in [0,1], optional
            ~percentage of land. The default is 0.1
        fluctuation : float, optional
            Determines the fluctuation of fast ice and errorrates in each observation, VERY sensitive -> std of normal. The default is 0.01
        noise : float, optional
            Determines the noise cast onto the observation, VERY sensitive -> std of normal. The default is 0.03
        errorRate : float in [0,1], optional
            expected percentage of missing data points and also the relative size of areal errors compared to pointwise. The default is 0.
        arealError: float in [0,1], optional
            expected chance for an areal error, only occurs if errorRate > 0. The default is .9
        clustered : float, optional
            How clustered the fast ice is, the lower the fewer clusters there are -> has to do with operating of perlin noise. The default is 2.
        maxSpeed : float > 0, optional
            the max speed a pixel can have. The default is 5.
        steepness : float > 0, optional
            the steepness of the threshhold function for the fast ice development. The default is 17. (recommended between 10-25)
        seed : int, optional
            seed for the simulation. The default is None => a random seed between 0-99999999 is taken.
        no_starting_error : bool, optional
            determines if the first observation has errors or not. Default is False (Errors)

        '''
        self.y, self.x   = shape
        self.cycle       = cycleLength
        self.amountFI    = amountFI
        self.amountLand  = amountLand
        self.fluctuation = fluctuation
        self.noise       = noise
        self.errorRate   = errorRate
        self.arealError  = arealError
        self.clustered   = clustered
        self.maxSpeed    = maxSpeed
        self.steepness   = steepness
        self.seed        = seed
        self.start_err   = no_starting_error
        
        self.NaN     = np.NaN
        self.landVal = -1
        
        # Create perlin object
        self.Perlin = pn.PerlinNoise2D()
        
        # check seed, else randomize it
        if self.seed == None: self.seed = np.random.randint(0, 100000000)
        self.rs = RandomState(MT19937(SeedSequence(self.seed)))
        
        # Initialize everything else needed
        self.initialize()
        
        
    def initialize(self):
        # Create base noise for 'sea depth'
        yLin, xLin   = np.linspace(0, self.clustered, self.y), np.linspace(0, self.clustered, self.x)     # The smaller 'clustered' the more the values are in one area
        xGrid, yGrid = np.meshgrid(xLin, yLin)
        self.perlinFix = self.Perlin.perlin(xGrid, yGrid, self.seed)
        
        # Compute sorted list
        sortNoise = np.sort(self.perlinFix.flatten())
        
        ## Thresholding parameters
        # Land
        self.landTresh = sortNoise[int(self.y * self.x * (1- self.amountLand))] # Everything above is land
        # Parameters for Threshholding fast ice
        minThresh = sortNoise[int(self.y * self.x  * (1- self.amountLand - self.amountFI/10))]   # Around 10% of all fast ice is shown at minimum
        maxThresh = sortNoise[int(self.y * self.x  * (1- self.amountLand - self.amountFI))]      # Around the amountFI of all pixels are fast ice
        
        # Land values
        self.land = np.where(self.perlinFix >= self.landTresh, 1, 0)
        # Generate coast pixel
        self.coast = self.coastPixel(self.land)
        
        # create thresholding function
        mean_begin = .2
        mean_end   = .75
        # pseudo truncated normal, since it does not exist in numpy but I need the seed dependence
        begin = np.clip(self.rs.normal(loc=mean_begin, scale=.1), 0, 1)
        end   = np.clip(self.rs.normal(loc=mean_end, scale=.1), 0, 1)
        if end > begin: 
            a = begin
            begin = end
            end   = a
        self.create_threshold(begin, end, self.steepness, minThresh, maxThresh)
        
                
        # Parameters for error creation
        self.keepOld = .8
        self.aErr = .9 * self.arealError
        self.pErr = 1  - self.aErr
        # Initialize step
        self.step = 0
        # safe last perlin noise
        self.lastPerlin = self.perlinFix
        # create first error noise
        xGrid, yGrid = np.meshgrid(np.linspace(0, 2, self.x), np.linspace(0, 2, self.y))
        self.lastError  = self.Perlin.perlin(xGrid, yGrid, self.rs.randint(0, 100000000))

        
    ################
    ## Simulation ##
    ################ 
    def simulate(self):
        # Increase step by 1
        self.step += 1
        # Threshhold Fast Ice, everywhere else new Perlin 'Movement'
        yLin, xLin   = np.linspace(0, 2, self.y), np.linspace(0, 2, self.x)     # Fixed Clustered=2, so it's more stable movement
        xGrid, yGrid = np.meshgrid(xLin, yLin)
        # seed for perlin noise
        seed      = self.rs.randint(0, 100000000)
        # Compute new fast and drift ice
        factor = 1 - np.clip(np.abs(self.rs.normal(scale=.4)), 0, 1)
        newPerlin = (factor * self.prepPerlin(self.Perlin.perlin(xGrid, yGrid, seed)) + (1-factor) * self.lastPerlin)
        
        maskFI = np.logical_and(self.perlinFix >= self.threshold(self.step), self.perlinFix < self.landTresh)
        speeds = np.where(maskFI, 0, newPerlin)

        # create noise and add to speeds:
        noise = self.rs.normal(scale=self.noise, size=(self.y, self.x))
        total = np.abs(speeds + noise)
        # set Coast Errors to special noise:
        coastVals = self.rs.random(size=(self.y, self.x)) * self.maxSpeed / 5
        total  = np.where(self.coast, coastVals, total)
            
        # Add Errors
        errors = self.createErrors()
        # Remove corner point errors for convenience (interpolating problems and such)
        errors[ 0, 0] = 0
        errors[-1,-1] = 0
        errors[0, -1] = 0
        errors[-1, 0] = 0
        # --------------- #
        preFinal = np.where(errors, self.NaN, total)
        final    = np.where(self.land, self.landVal, preFinal)
        
        # Safe last perlin noise
        self.lastPerlin = newPerlin
        
        return final, maskFI
    
    ###################
    ## Create Errors ##
    ###################  
    def createErrors(self):
        if self.start_err and (self.step == 1 or self.step == self.cycle): return np.zeros((self.y, self.x))
        
        # Pointswise Errors
        errorPoints = self.rs.binomial(1, self.errorRate * self.pErr, size=(self.y, self.x))   # Pointwise errors per step / later half (or 2/3) the errorRate for the area errors
        # Area Errors
        yLin, xLin   = np.linspace(0, 2, self.y), np.linspace(0, 2, self.x)     # Fixed Clustered=2, so it's more coherent
        xGrid, yGrid = np.meshgrid(xLin, yLin)
        seed      = self.rs.randint(0, 100000000)
        factor = np.clip(self.rs.normal(loc=self.keepOld, scale=.5), 0, 1)
        perlin  = factor * self.lastError + (1-factor) * self.Perlin.perlin(xGrid, yGrid, seed)
        thresh = self.generateThresh(perlin) # threshhold of last 0
        area  = np.where(perlin < thresh, 1, 0)
        # setting old error
        self.lastError = perlin
        return np.logical_or(errorPoints, area)
    
    def generateThresh(self, perlin):
        # generates the threshhold for every step
        cover = self.rs.random() > 1 - self.arealError
        fluct = np.abs(self.rs.normal(loc=0, scale=self.fluctuation)) if self.errorRate > 0 else 0
        return np.sort(perlin.flatten())[int(self.errorRate * self.y * self.x * self.aErr)* cover] + fluct
    
    ######################
    ## Preparate Perlin ##
    ######################    
    def prepPerlin(self, perlin):
        return (perlin - np.min(perlin)) * self.maxSpeed
    
    ################
    ## Threshhold ##
    ################  
    def create_threshold(self, begin, end, alpha, min_thresh, max_thresh):
        def function(t):
            x = t / self.cycle
            return min_thresh + (( np.tanh(alpha * (x-begin)) - np.tanh(alpha * (x-end)) ) / 2) * (min_thresh - max_thresh) + self.rs.normal(scale=self.fluctuation)
        self.threshold = function

    ###############
    ## CoastLine ##
    ###############  
    def coastPixel(self, pixel):
        '''
        Computes the coastline of a 2D pixelSet, the pixel in question should be masked!

        Parameters
        ----------
        pixel : np.array(y,x)
            2D array of pixel, every 'land'pixel is set to 1.

        Returns
        -------
        coast : np.array(y,x)
            returns a mask of the coast line.

        '''
        def neighbours(row, col):

            rows, cols = pixel.shape
            out = []

            for i in range(row - 1, row + 2):
                row = []
                for j in range(col - 1, col + 2):

                    if 0 <= i < rows and 0 <= j < cols:
                        row.append(pixel[i,j])
                    else:
                        row.append(0)

                out.append(row)
            return np.array(out).any()
        
        y, x = pixel.shape
        # create empty canvas
        empty = np.zeros((y,x))
        
        for yy in np.arange(y):
            for xx in np.arange(x):
                if neighbours(yy, xx): empty[yy, xx] = 1
        return empty





