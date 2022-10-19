#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:57:26 2022

@author: paytonbeeler
"""

import numpy as np
import rates, read_partMC, run, indexing, tqdm, os, plot
from scipy.integrate import ode
import matplotlib.pyplot as plt


## initial run to check the equilibrium partitioning

# =================== SETTINGS =====================
partMC_input = False
constant_gas = False
constant_pressure = False
co_condens = False 
constant_pH = True
Kelvin_effect = True
aq_oxidation = True
# ==================================================

# =============== AEROSOL INPUTS ===================
Dpg = 0.150e-6 #2.0*1.3365046175719767e-05 #2.0*1.485e-7 # mean diameter of particles, m
sigma  = 1.8 # geometric standard deviation of lognormal size distribution
Ntot = 1E8 # total number concentration, #/m^3
bins = 6 # bins to split the distribution into
pH0 = 4.0
# ==================================================

# ================ PARCEL INPUTS ===================
stop_height = 500 # where to stop the calculation, m
dt = 1.0    # time interval for evaluation of particle growth
SS0 = -0.1 #supersaturation offset, will be added to supersaturation trajectory
T0 = 298 # surface temperature, K
P0 = 101325 # surface pressure, Pa
condens_coeff = 1.0 # condensation coefficient for water vapor
thermal_accom = 0.96 # thermal accomodation coefficient
V = 1.0  # velocity of parcel, m/s
# ==================================================

t_end = stop_height/V


if partMC_input == True:
    filename = '/Users/paytonbeeler/Desktop/1_urban_plume/out/urban_plume_0001_00000025.nc'
    y0 = read_partMC.read_input(bins, filename)
    t0 = 0
else:
    Ddrys, Ns = run.make_lognormal_dist(Dpg, sigma, Ntot, bins)
    gas_phase0 = run.setup_gas()
    particle_phase0 = run.setup_particles(bins)
    y0 = run.initialize(gas_phase0, particle_phase0, T0, P0, SS0, pH0, Ddrys, Ns)
    t0 = 0  


#for i in range(0, bins):
#    name = 'kappa, ' + str(i+1)
#    if y0[indexing.getIndex(name, bins)] < 0.17:
#        print('MIGHT FAIL TO RUN! kappa, ' + str(i+1) + ' = ', y0[indexing.getIndex(name, bins)])
#        print(' ')
    #y0[indexing.getIndex(name, bins)] = 0.5
    
#for i in range(0, len(y0)):
#    print(indexing.getName(i, bins), y0[i])

#dydt = run.ODEs(t0, y0, Ntot, bins, V, condens_coeff, thermal_accom, constant_gas, co_condens, constant_pH, Kelvin_effect, constant_pressure, aq_oxidation)


ode15s = ode(run.ODEs).set_integrator('lsoda', method='bdf', nsteps=1E4, rtol=1E-4, atol=1E-8, max_step=0.1)
ode15s.set_initial_value(y0, t0).set_f_params(Ntot, bins, V, condens_coeff, thermal_accom, constant_gas, co_condens, constant_pH, Kelvin_effect, constant_pressure, aq_oxidation)
soln = y0
t = np.array([t0])

print('Integrating...')
pbar = tqdm.tqdm(total = len(np.linspace(0., t_end, int(t_end/dt+1))))
while ode15s.successful() and ode15s.t < t_end:
    soln = np.vstack([soln, ode15s.integrate(ode15s.t+dt)])
    t = np.append(t, ode15s.t)
    pbar.update(1)
pbar.close()


for i in range(6, 79):
    if np.min(soln[:, i]) < 0:
        print(indexing.getName(i, bins), np.min(soln[:, i]))
for i in range(79+6*bins, len(soln[0])):
    if np.min(soln[:, i]) < 0:
        print(indexing.getName(i, bins), np.min(soln[:, i]))
print(' ')

for i in range(0, len(soln)):
    for j in range(6, 79):
        if soln[i, j] < 0:
            soln[i, j] = 0
    for j in range(79+6*bins, len(soln[0])):
        if soln[i, j] < 0:
            soln[i, j] = 0


zs = [10, 275, 400]
particles = [0, 1, 2, 3, 4, 5] #np.floor(bins*np.random.rand(6))
plot.make_figure(zs, particles, soln, bins, t_end, stop_height)

   

