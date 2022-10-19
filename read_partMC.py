#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 09:01:12 2022

@author: paytonbeeler
"""

import netCDF4, run, indexing, tqdm
import numpy as np
import modified_GAMMA_aq as aq_chemistry
from scipy.optimize import root

R_dry = 287.0 # Pa*m^3/kg*K

def read_input(num_particles, filename):
    
    partMC_data = netCDF4.Dataset(filename)
    
    temp = partMC_data.variables['time']
    x = temp[:]*1
    print('Initializing model with PartMC simulation at', x, 'seconds')
    
    
    y0 = np.zeros(79+55*num_particles+num_particles)
    
    temp = partMC_data.variables['temperature']
    T0 = temp[:]*1
    y0[indexing.getIndex('T', num_particles)] = T0
    temp = partMC_data.variables['relative_humidity']
    SS0 = (temp[:]*1) - 1
    y0[indexing.getIndex('SS', num_particles)] = SS0
    temp = partMC_data.variables['pressure']
    P0 = temp[:]*1
    y0[indexing.getIndex('P', num_particles)] = P0
    temp = partMC_data.variables['altitude']
    z = temp[:]*1
    y0[indexing.getIndex('z', num_particles)] = z
    es = 611.2*np.exp((17.67*(T0-273.15))/(T0-273.15+243.5))
    mixing_ratio = (SS0 + 1.0) * (0.622 * es / (P0 - es)) # kg water vapor per kg air
    Tv = T0*(1+0.61*mixing_ratio)
    rho_air = P0/(R_dry*Tv)
    
    #aero_particle(aero_particle)
    #aero_num_conc(aero_particle)
    #aero_species(aero_species)
    #aero_density(aero_species)
    #time()
    #aero_particle_mass(aero_species, aero_particle)
    #aero_kappa(aero_species)
    
    temp = partMC_data.variables['gas_mixing_ratio']
    partMC_gas = temp[:]*1
    partMC_gas_species = ['H2SO4', 'HNO3', 'HCl', 'NH3', 'NO', 'NO2', 'NO3', 
                   'N2O5', 'HONO', 'HNO4', 'O3', 'O1D', 'O3P', 'OH', 
                   'HO2', 'HOOH', 'CO', 'SO2', 'CH4', 'C2H6', 'CH3OO', 
                   'ETHP', 'FORM', 'CH3OH', 'ANOL', 'CH3O2H', 'ETHOOH', 
                   'ALD2', 'HCOOH', 'RCOOH', 'C2O3', 'PAN', 'ARO1', 'ARO2', 
                   'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 'PAR',
                   'AONE', 'MGLY', 'ETH', 'OLET', 'OLEI', 'TOL', 'XYL',
                   'CRES', 'TO2', 'CRO', 'OPEN', 'ONIT', 'ROOH', 'RO2', 
                   'ANO2', 'NAP', 'XO2', 'XPAR', 'ISOP', 'ISOPRD', 'ISOPP',
                   'ISOPN', 'ISOPO2', 'API', 'LIM', 'DMS', 'MSA', 'DMSO', 
                   'DMSO2', 'CH3SO2H', 'CH3SCH2OO', 'CH3SO2', 'CH3SO3', 
                   'CH3SO2OO', 'CH3SO2CH2OO', 'SULFHOX']
    
    for i in range(0, len(partMC_gas_species)):
        try:
            new_index = indexing.getIndex(partMC_gas_species[i] + ' (gas)', num_particles)
            y0[new_index] = partMC_gas[i]*1E-9
        except:
            new_index = -1
    
    temp = partMC_data.variables['aero_particle']
    x = temp[:]*1
    total_particles = len(x)
    selected_particles = np.floor(total_particles*np.random.rand(num_particles))
        
    partMC_aq_species = ['SO4--', 'NO3-', 'Cl', 'NH4+', 'MSA', 'ARO1', 'ARO2', 
             'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
             'CO3--', 'Na', 'Ca', 'OIN', 'OC', 'BC', 'H2O']
    
    temp = partMC_data.variables['aero_density']
    partMC_densities = temp[:]*1
    temp = partMC_data.variables['aero_kappa']
    partMC_kappas = temp[:]*1
    temp = partMC_data.variables['aero_molec_weight']
    partMC_mw = temp[:]*1
    
    dry_Ds = []
    kappas = []
    Ns = []
    lwc = 0
    
    #selected_particles = [361]
    
    pbar = tqdm.tqdm(total = num_particles)
    for i in range(0, num_particles):
        #print(int(selected_particles[i]))
        temp = partMC_data.variables['aero_num_conc']
        x = temp[:]*1
        N = x[int(selected_particles[i])] # particles/m^3
        name = 'N, ' + str(i+1)
        y0[indexing.getIndex(name, num_particles)] = N
        Ns.append(N)

        temp = partMC_data.variables['aero_particle_mass']
        x = temp[:]*1
        masses = x[:, int(selected_particles[i])] # kg
        Vtot = 0
        kj_Vj = 0
        dry_mass = 0
        for j in range(0, len(masses)-1):
            Vj = masses[j]/partMC_densities[j] # volume of species j, m^3
            Vtot += Vj
            dry_mass += masses[j]
            kj_Vj += Vj*partMC_kappas[j]


        Ddry = 2.0*np.power((3.0*Vtot)/(4.0*np.pi), 1/3)
        name = 'Ddry, ' + str(i+1)
        y0[indexing.getIndex(name, num_particles)] = Ddry
        dry_Ds.append(Ddry)

        name = 'density, ' + str(i+1)
        y0[indexing.getIndex(name, num_particles)] = dry_mass/Vtot
    
        kappa = kj_Vj/Vtot
        name = 'kappa, ' + str(i+1)
        y0[indexing.getIndex(name, num_particles)] = kappa
        kappas.append(kappa)
        
        density = dry_mass/Vtot
        name = 'density, ' + str(i+1)
        y0[indexing.getIndex(name, num_particles)] = density
        
        pH = 4.0
        name = 'pH, ' + str(i+1)
        y0[indexing.getIndex(name, num_particles)] = pH
        
           
        partMC_moles = np.zeros(len(masses))
        for j in range(0, len(masses)):
            partMC_moles[j] = masses[j]/partMC_mw[j]
            
        
        Dp, dummy1, dummy2 = aq_chemistry.equilibrate_h2o(np.array([Ddry]), np.array([kappa]), np.array([N]), SS0, P0, T0)
        
        name = 'Dp, ' + str(i+1)
        y0[indexing.getIndex(name, num_particles)] = Dp[0]
        Rp = 0.5*Dp[0]
        Rdry = 0.5*Ddry
        V_water = (4.0/3.0)*np.pi*(Rp**3-Rdry**3)
        mass_water = V_water*1000.0
        name = 'H2O (aq,' + str(i+1) + ')'
        y0[indexing.getIndex(name, num_particles)] = mass_water
        lwc += N*mass_water
            
        for j in range(0, len(masses)-1):
            name = partMC_aq_species[j] + ' (aq,' + str(i+1) + ')'
            y0[indexing.getIndex(name, num_particles)] = masses[j]
        
        pbar.update(1)
    pbar.close()
      
    lwc /= rho_air
    
    y0[indexing.getIndex('wv', num_particles)] = mixing_ratio
    y0[indexing.getIndex('wc', num_particles)] = lwc
    print(' ')
    
    return y0
