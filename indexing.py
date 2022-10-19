#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 13:46:43 2022

@author: paytonbeeler
"""

import numpy as np
from numba import njit
import os

#@jit
def getName(index, bins):
    
    filename = os.getcwd() + '/gas_input.txt'
    data = np.loadtxt(filename, dtype='str', delimiter = ": ")
    gasses = ['z', 'T', 'P', 'SS', 'wv', 'wc']
    
    for i in range(0, len(data)):
        name = data[i][0]
        gasses.append(name + ' (gas)')
    
    
    filename = os.getcwd() + '/particle_input.txt'
    data = np.loadtxt(filename, dtype='str', delimiter = ": ")
    X = ['Ddry', 'Dp', 'kappa', 'N', 'density', 'pH']
    particle = []
    
    for x in X:
        for i in range(0, bins):
            particle.append(x + ', ' + str(i+1))

    for i in range(0, len(data)):
         for j in range(0, bins):
             name = data[i][0]
             particle.append(name + ' (aq,' + str(j+1) + ')')
    
    
    combined = np.hstack([gasses, particle])
    name = combined[index]
    #print(len(combined))
    
    #f = open('/Users/paytonbeeler/Desktop/output.txt', 'w')
    #f.close()
    #for i in range(0, len(combined)):
        #print(i,",", combined[i])
    #    f = open('/Users/paytonbeeler/Desktop/output.txt', 'a')
    #    f.write(str(i)+"	"+str(combined[i])+'\n')
    #    f.close()
    
    return name

def getIndex(search, bins):
    
    filename = os.getcwd() + '/gas_input.txt'
    data = np.loadtxt(filename, dtype='str', delimiter = ": ")
    gasses = ['z', 'T', 'P', 'SS', 'wv', 'wc']
    
    for i in range(0, len(data)):
        name = data[i][0]
        gasses.append(name + ' (gas)')
    
    filename = os.getcwd() + '/particle_input.txt'
    data = np.loadtxt(filename, dtype='str', delimiter = ": ")
    X = ['Ddry', 'Dp', 'kappa', 'N', 'density', 'pH']
    particle = []
    
    for x in X:
        for i in range(0, bins):
            particle.append(x + ', ' + str(i+1))

    for i in range(0, len(data)):
         for j in range(0, bins):
             name = data[i][0]
             particle.append(name + ' (aq,' + str(j+1) + ')')
            
    #combined = np.hstack([gasses, particle])
    combined = np.append(gasses, particle)
            
    for i in range(0, len(combined)):
        if search == combined[i]:
            index = i
            break
    
    return index

    
    