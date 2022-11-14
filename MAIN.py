#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:57:26 2022

@author: paytonbeeler
"""

import numpy as np
import read_partMC, run, indexing, tqdm, os, hiscale, processing, plot
import pickle, time, sys
from scipy.integrate import ode
import matplotlib.pyplot as plt
import aqueous_chemistry as aq_chemistry


## initial run to check the equilibrium partitioning

# =================== SETTINGS =====================
partMC_input = False
hiscale_run = True
constant_gas = False
constant_pressure = False
co_condens = True 
constant_pH = True
Kelvin_effect = True
aq_oxidation = True
# ==================================================

# =============== AEROSOL INPUTS ===================
Dpg = 0.100e-6 # mean diameter of particles, m
sigma  = 1.8 # geometric standard deviation of lognormal size distribution
Ntot = 1E8 # total number concentration, #/m^3
bins = 25 # bins to split the distribution into
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
start_time = time.time()

#fractional mass of each species for particles to be put in that bin
mass_thresholds = {'BC': 0.6, 'IEPOX': 0.3, 'dust': 0.6,
                   'organics': 0.6, 
                   'nitrate': {'nitrate': 0.6, 'organics': 0.1},
                   'sulfate/nitrate': {'sulfate': 0.3, 'nitrate': 0.3, 'organics': 0.1}}

if partMC_input == True:
    filename = os.getcwd() + '/1_urban_plume/out/urban_plume_0001_00000025.nc'
    y0 = read_partMC.read_input(bins, filename)
    t0 = 0
elif hiscale_run == True:    
    cloud_flag = 0.0
    CVI_flag = 0.0
    sampling_time = [981, 986] # minutes since midnight
    y0 = hiscale.setup(bins, sampling_time, pH0, cloud_flag, CVI_flag, mass_thresholds, plots=False)
    t0 = 0
    #[981, 986]
    #[996, 1001]
    #[1087, 1092]
else:
    Ddrys, Ns = run.make_lognormal_dist(Dpg, sigma, Ntot, bins)
    gas_phase0 = run.setup_gas()
    particle_phase0 = run.setup_particles(bins)
    y0 = run.initialize(gas_phase0, particle_phase0, T0, P0, SS0, pH0, Ddrys, Ns)
    t0 = 0  

data = {}
data['y0'] = y0
filename = os.getcwd() + '/y0_' + str(bins) + '_part.pkl'
pickle.dump(data, open(filename,'wb'))

data = pickle.load(open(filename, 'rb'))
y0 = data['y0']


ode15s = ode(run.ODEs).set_integrator('lsoda', method='bdf', nsteps=1E4, rtol=1E-4, atol=1E-8, max_step=0.1)
ode15s.set_initial_value(y0, t0).set_f_params(Ntot, bins, V, condens_coeff, thermal_accom, constant_gas, co_condens, constant_pH, Kelvin_effect, constant_pressure, aq_oxidation)
soln = y0
t = np.array([t0])

print('Integrating...')
pbar = tqdm.tqdm(total = len(np.linspace(0., t_end, int(t_end/dt+1))))
while ode15s.successful() and ode15s.t < t_end:
    soln = np.vstack([soln, ode15s.integrate(ode15s.t+dt)])
    t = np.append(t, ode15s.t)
    if abs(np.sum(soln[-1, :])) >= 0:
        success = True
    else:
        success = False
    if success == False:
        sys.exit()
    pbar.update(1)
pbar.close()

for i in range(0, len(soln)):
    for j in range(6, 79):
        if soln[i, j] < 0:
            soln[i, j] = 0
    for j in range(79+6*bins, len(soln[0])):
        if soln[i, j] < 0:
            soln[i, j] = 0

elapsed_time = time.time() - start_time
print(' ')
print(' ')
print('Total solving time:', np.round(elapsed_time,2), 'seconds')
print(' ')
print(' ')


filename = os.getcwd() + '/particle_input.txt'
data = np.loadtxt(filename, dtype='str', delimiter = ": ")
output = {}

output['t'] = t
output['mass_thresholds'] = mass_thresholds

for i in range(0, 79):
    name = indexing.getName(i, bins)
    output[name] = soln[:, i]

names = ['Ddry', 'Dp', 'kappa', 'N', 'density', 'pH']

output[names[0]] = np.exp(soln[:, 79+0*bins:79+0*bins+bins])
output[names[1]] = np.exp(soln[:, 79+1*bins:79+1*bins+bins])
for i in range(2, 6):
    output[names[i]] = soln[:, 79+i*bins:79+i*bins+bins]

for i in range(0, len(data)):
    name = data[i][0]
    index = 79+((i+6)*bins)
    output[name] = soln[:, 79+((i+6)*bins):79+((i+6)*bins)+bins]



if hiscale_run == True:
    file_name = os.getcwd() + '/01ppb_' + str(bins) + '_particles_1.pkl'
elif partMC_input == True:
    file_name = os.getcwd() + '/PartMC_run_' + str(bins) + '_particles.pkl'
else:
    file_name = os.getcwd() + '/SD_run_' + str(bins) + '_particles.pkl'

pickle.dump(output, open(file_name,'wb'))


plot.SS_Dps(file_name, bins)
plt.plot(output['t'], soln[:, (indexing.getIndex('ISOP (gas)', bins))])
processing.classify_particles(file_name)


