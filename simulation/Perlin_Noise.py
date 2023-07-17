#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:30:07 2023

@author: david
"""
import numpy as np

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence



class PerlinNoise2D():
    
    def __init__(self):
        '''
        Initilaizes an object for perlin noise creation. Use perlin(x, y, seed) for creating the wished perlin noise.

        Returns
        -------
        Object.

        '''
        return 
  
    def perlin(self, x, y, seed=-1):    
        if seed != -1: rs = RandomState(MT19937(SeedSequence(seed)))
        else:          rs = RandomState(MT19937(SeedSequence(np.random.randint(0,100000))))
        # permutation table
        p = np.arange(256, dtype=int)
        rs.shuffle(p)
        p = np.stack([p, p]).flatten()
        # coordinates of the top-left
        xi, yi = x.astype(int), y.astype(int)
        # internal coordinates / vectors
        xf, yf = x - xi, y - yi
        # fade/interpolation factors
        wx, wy = self.inter(xf), self.inter(yf)
        # noise components
        v00 = self.gradients(p[p[xi] + yi], xf, yf)
        v01 = self.gradients(p[p[xi] + yi + 1], xf, yf - 1)
        v11 = self.gradients(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        v10 = self.gradients(p[p[xi + 1] + yi], xf - 1, yf)
        # combine noises
        x1 = self.lerp(v00, v10, wx)
        x2 = self.lerp(v01, v11, wx)  
        return self.lerp(x1, x2, wy)  
    
    def lerp(self, a, b, x):
        # Linear interpolation
        return a + x * (b - a)
    
    def inter(self, t):
        # Interpolation function
        return 6 * t**5 - 15 * t**4 + 10 * t**3
        
    def gradients(self, h, x, y):
        v = 1/np.sqrt(2)
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [v, v], [v, -v], [-v, v], [-v, -v]]) 
        # Choosing pseudo-random vectors dependend on the order of p
        g = vectors[h % 8]
        return g[:, :, 0] * x + g[:, :, 1] * y
     
        
     

