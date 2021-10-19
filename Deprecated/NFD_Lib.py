#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:00:21 2021

@author: SAB
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def NFD_Lib(x,y,imsize):
    
# Get dimensions of input contours
    r, c, nsamp = np.shape(x)
# Set up library variables

#    surface = np.zeros(num_k,nsamp,rxinc,ryinc)  ; Library variables
#    mag = fltarr(rxinc,ryinc)
#    dc = complexarr(rxinc,ryinc)
#    lib_angle = fltarr(num_k,rxinc,ryinc)

    kmax = 5    # arbitrary number of normalizations - probably ends up being 2 or 3
    k_norm = np.array([2,-1,-2,-3,-4])

    dc      = np.zeros((r,c),dtype='c16')
    mag     = np.zeros((r,c))
    surface = np.zeros((r,c,kmax,nsamp),dtype='c16')
    lib_angle = np.zeros((r,c,kmax))
    index_vect = np.linspace(1,nsamp,nsamp)-nsamp/2

#    px = np.linspace(0, 2 * np.pi, nsamp)
    max_norms = 0
    for i in range(r):
        for j in range(c):
            x_new = x[i,j,:]
            y_new = y[i,j,:]
            fcoord = np.fft.fft((x_new+(imsize-y_new)*1j),nsamp)
            fcoord = np.fft.fftshift(fcoord)    # shift so DC is in center
#            dc[i,j] = abs(fcoord[int(nsamp/2)])
            dc[i,j] = (fcoord[int(nsamp/2)])
            fcoord[int(nsamp/2)] = 0            # normalize x,y position
            idx = np.argsort(abs(fcoord))       # sort fft coeffs by magnitude
            idx = idx[::-1]                     # sort descending
            num_norms = abs(idx[1]-nsamp/2-1)   # number of normalizations
            print('Number of normalizations:',num_norms, 'Index',idx[1])

            if num_norms == 0:
                print('No valid normalizations')
                dc[i,j]         = 0.0
                mag[i,j]        = 0.0
                surface[i,j,:,:] = 0.0
                lib_angle[i,j,:] = 0.0
                break
            
            if num_norms > max_norms:
                max_norms = num_norms
                
            mag[i,j] = abs(fcoord[int(nsamp/2+1)])  # A(1) term to normalize size
            fcoord = fcoord/mag[i,j]                # Normalize for magnitude A(1)
            for norm in range(int(num_norms)):
                k = k_norm[norm]
                # Compute phase angles of A(1) and A(k)
                u = np.arctan2(fcoord.imag[int(nsamp/2+1)],fcoord.real[int(nsamp/2+1)])
                v = np.arctan2(fcoord.imag[int(idx[1])],fcoord.real[int(idx[1])])
#                print('u',u,'v',v)
                # Save reference angle for library matching - in degrees
                lib_angle[i,j,norm] = ((v-k*u)/(k-1))*180./(math.pi)
                # Compute complex angle to standardize in-plane rotation and contour starting point
                angle = ((index_vect - k)*u + (1-index_vect)*v)/(k-1)
                coeff = np.cos(angle)+np.sin(angle)*1j
#                print('Angle:',angle,'coeff',coeff)
                # Finish normalization
                surface[i,j,norm,:] = fcoord*coeff

    max_norms = int(max_norms)
    surface = surface[:,:,0:max_norms,:]
    lib_angle = lib_angle[:,:,0:max_norms]
    
    return dc, mag, lib_angle, surface
