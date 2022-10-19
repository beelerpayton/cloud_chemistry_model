# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:09:00 2022

@author: beel083
"""

#from pyrcel import constants as c
import numpy as np
import gas_chemistry, parcel, os, indexing
import modified_GAMMA_aq as aq_chemistry
import warnings, sys, time
from scipy import special
from scipy.integrate import solve_ivp

R = 8.314 # m^3 Pa/mol K
Na = 6.022e23 # molecules per mole


def setup_gas():
    
    filename = os.getcwd() + '/gas_input.txt'
    
    data = np.loadtxt(filename, dtype='str', delimiter = ": ")
    
    gas_phase0 = {}

    for i in range(0, len(data)):
        
        name = data[i][0]
        value = float(data[i][1])
        gas_phase0[name] = value
    
    return gas_phase0


def setup_particles(N):
    
    filename = os.getcwd() + '/particle_input.txt'
    data = np.loadtxt(filename, dtype='str', delimiter = ": ")
    
    particle_phase0 = {}
    
    for i in range(0, len(data)):
        name = data[i][0]
        value = float(data[i][1])
        particle_phase0[name] = value
        
    return particle_phase0


def initialize(gas, particle_volumes, T, P, SS, pH, dry_diameters, Ns):
    
    bins = len(Ns)
    y0 = np.zeros(79+56*bins+bins)
    
    # ------------------------------------------
    #diameters = dry_diameters
    #dry_diameters = []
    
    #for i in range(0, len(kappas)):
    #    dry_diameters.append(0)
    # ------------------------------------------
    
    es = 611.2*np.exp((17.67*(T-273.15))/(T-273.15+243.5))
    wv = (SS + 1.0) * (0.622 * es / (P - es)) # kg water vapor per kg air
    Tv = T*(1+0.61*wv)
    rho_air = P/(287.0*Tv)
    
    y0[indexing.getIndex('z', bins)] = 0
    y0[indexing.getIndex('T', bins)] = T
    y0[indexing.getIndex('P', bins)] = P
    y0[indexing.getIndex('SS', bins)] = SS
    y0[indexing.getIndex('wv', bins)] = wv

    for gas_species in gas:
        name  = gas_species + ' (gas)'
        y0[indexing.getIndex(name, bins)] = gas[gas_species]
        

    density_i = {'SO4--': 1800, 'NO3-': 1800, 'Cl': 2200, 
                 'NH4+': 1800, 'MSA': 1800, 'ARO1': 1400, 
                 'ARO2': 1400, 'ALK1': 1400, 'OLE1': 1400,
                 'API1': 1400, 'API2': 1400, 'LIM1': 1400, 
                 'LIM2': 1400, 'CO3--': 2600, 'Na': 2200, 
                 'Ca': 2600, 'OIN': 2600, 'OC': 1000, 
                 'BC': 1800, 'H2O': 1000}
    
    kappa_i = {'SO4--': 0.65, 'NO3-': 0.65, 'Cl': 0.53, 
              'NH4+': 0.65, 'MSA': 0.53, 'ARO1': 0.1, 
              'ARO2': 0.1, 'ALK1': 0.1, 'OLE1': 0.1,
                 'API1': 0.1, 'API2': 0.1, 'LIM1': 0.1, 
                 'LIM2': 0.1, 'CO3--': 0.53, 'Na': 0.53, 
                 'Ca': 0.53, 'OIN': 0.1, 'OC': 0.001, 
                 'BC': 0., 'H2O': 0.}
    
    '''
    #-------------------------------------
    
    V_BCs = np.random.rand(bins)
    V_SO4s = np.linspace(1, 1, bins) - V_BCs
    kappas = np.zeros(bins)
    for i in range(0, len(dry_diameters)):
        name = 'kappa, ' + str(i+1)
        ki_Vi = 0
        dry_volume = (4.0/3.0)*np.pi*(0.5*dry_diameters[i])**3
        for aq_species in ['BC', 'SO4--']:
            if aq_species == 'BC':
                Vi = V_BCs[i]*dry_volume
                ki_Vi += kappa_i[aq_species]*Vi  
            else:
                Vi = V_SO4s[i]*dry_volume
                ki_Vi += kappa_i[aq_species]*Vi

        y0[indexing.getIndex(name, bins)] = ki_Vi/dry_volume
        kappas[i] = ki_Vi/dry_volume
        name = 'N, ' + str(i+1)
        y0[indexing.getIndex(name, bins)] = Ns[i]


    for aq_species in ['BC', 'SO4--']:
        for i in range(0, len(dry_diameters)):
            name = aq_species + ' (aq,' + str(i+1) + ')'
            dry_volume = (4.0/3.0)*np.pi*(0.5*dry_diameters[i])**3
            if aq_species == 'BC':
                Vi = V_BCs[i]*dry_volume
                density = density_i[aq_species]
                mass = density*Vi
                y0[indexing.getIndex(name, bins)] = mass
            else:
                Vi = V_SO4s[i]*dry_volume
                density = density_i[aq_species]
                mass = density*Vi
                y0[indexing.getIndex(name, bins)] = mass
                
    for i in range(0, len(dry_diameters)):
        dry_mass = 0
        dry_volume = (4.0/3.0)*np.pi*(0.5*dry_diameters[i])**3
        for aq_species in particle_volumes:
            name = aq_species + ' (aq,' + str(i+1) + ')'
            dry_mass += y0[indexing.getIndex(name, bins)]

        name = 'density, ' + str(i+1)
        y0[indexing.getIndex(name, bins)] = dry_mass/dry_volume
        
    diameters, wv, wc = aq_chemistry.equilibrate_h2o(dry_diameters, kappas, Ns, SS, P, T)
    
    wc = 0
    for i in range(0, len(dry_diameters)):
        name = 'Dp, ' + str(i+1)
        y0[indexing.getIndex(name, bins)] = diameters[i]
        name = 'Ddry, ' + str(i+1)
        y0[indexing.getIndex(name, bins)] = dry_diameters[i]
        name = 'pH, ' + str(i+1)
        y0[indexing.getIndex(name, bins)] = pH
        radius = 0.5*diameters[i]
        dry_radius = 0.5*dry_diameters[i]
        V_water = (4.0/3.0)*np.pi*(radius**3-dry_radius**3)
        mass_water = density_i['H2O']*V_water
        name = 'H2O (aq,' + str(i+1) + ')'
        y0[indexing.getIndex(name, bins)] = mass_water
        wc += Ns[i]*mass_water
    wc /= rho_air
    
    y0[indexing.getIndex('wc', bins)] = wc
    '''    
    
    
    
    kappas = np.zeros(bins)
    for i in range(0, len(dry_diameters)):
        name = 'kappa, ' + str(i+1)
        ki_Vi = 0
        dry_volume = (4.0/3.0)*np.pi*(0.5*dry_diameters[i])**3
        for aq_species in particle_volumes:
            if aq_species != 'H2O':
                try:
                    Vi = particle_volumes[aq_species]*dry_volume
                    ki_Vi += kappa_i[aq_species]*Vi                    
                except:
                    ki_Vi += 0
        y0[indexing.getIndex(name, bins)] = ki_Vi/dry_volume
        kappas[i] = ki_Vi/dry_volume
        name = 'N, ' + str(i+1)
        y0[indexing.getIndex(name, bins)] = Ns[i]
    

    for aq_species in particle_volumes:
        for i in range(0, len(dry_diameters)):
            if aq_species != 'H2O':
                try:
                    name = aq_species + ' (aq,' + str(i+1) + ')'
                    dry_volume = (4.0/3.0)*np.pi*(0.5*dry_diameters[i])**3
                    Vi = particle_volumes[aq_species]*dry_volume
                    density = density_i[aq_species]
                    mass = density*Vi
                    y0[indexing.getIndex(name, bins)] = mass
                except:
                    name = aq_species + ' (aq,' + str(i+1) + ')'
                    y0[indexing.getIndex(name, bins)] = 0
                    
    for i in range(0, len(dry_diameters)):
        dry_mass = 0
        dry_volume = (4.0/3.0)*np.pi*(0.5*dry_diameters[i])**3
        for aq_species in particle_volumes:
            name = aq_species + ' (aq,' + str(i+1) + ')'
            dry_mass += y0[indexing.getIndex(name, bins)]

        name = 'density, ' + str(i+1)
        y0[indexing.getIndex(name, bins)] = dry_mass/dry_volume
        
    diameters, wv, wc = aq_chemistry.equilibrate_h2o(dry_diameters, kappas, Ns, SS, P, T)

    wc = 0
    for i in range(0, len(dry_diameters)):
        name = 'Dp, ' + str(i+1)
        y0[indexing.getIndex(name, bins)] = diameters[i]
        name = 'Ddry, ' + str(i+1)
        y0[indexing.getIndex(name, bins)] = dry_diameters[i]
        name = 'pH, ' + str(i+1)
        y0[indexing.getIndex(name, bins)] = pH
        radius = 0.5*diameters[i]
        dry_radius = 0.5*dry_diameters[i]
        V_water = (4.0/3.0)*np.pi*(radius**3-dry_radius**3)
        mass_water = density_i['H2O']*V_water
        name = 'H2O (aq,' + str(i+1) + ')'
        y0[indexing.getIndex(name, bins)] = mass_water
        wc += Ns[i]*mass_water
    wc /= rho_air
    
    y0[indexing.getIndex('wc', bins)] = wc
    
    return y0
    

def make_lognormal_dist(Dp_avg, sigma, Ntot, bins):
   
    print('Initializing model given size distribution')
    if bins == 1:
        Dp = [Dp_avg]
        Ns = [Ntot]
    else:
        d_min = Dp_avg/(2.0*sigma)
        d_max = Dp_avg*2.0*sigma
    
        Dp = np.logspace(np.log10(d_min), np.log10(d_max), bins+1, endpoint=True)
        
        Ns = []
        
        for i in range(0,len(Dp)-1):
            
            Dp1 = 0.5*Ntot*special.erf((np.log(Dp[i])-np.log(Dp_avg))/(np.sqrt(2)*np.log(sigma)))
            Dp2 = 0.5*Ntot*special.erf((np.log(Dp[i+1])-np.log(Dp_avg))/(np.sqrt(2)*np.log(sigma)))
            Ns.append(Dp2-Dp1)
        
        Dp = Dp[:-1]
    
    return Dp, Ns


#def ODEs(y, Ntot, bins, V, condens_coeff, thermal_accom, state, constant_gas = False, co_condens = True, constant_pH = False, Kelvin_effect = True, constant_pressure = False):
def ODEs(t, y, Ntot, bins, V, condens_coeff, thermal_accom, constant_gas, co_condens, constant_pH, Kelvin_effect, constant_pressure, aq_oxidation):
         
    #for i in range(0, len(y)):
    #    if y[i] < 0:
    #        y[i] = 0
        
    dydt = np.zeros(len(y))
    dydt = parcel.parcel_properties(y, dydt, V, bins, thermal_accom, condens_coeff, constant_pressure=constant_pressure)
    
    dydt, dy_phasechange = aq_chemistry.particle_phase(y, dydt, bins, constant_pH = constant_pH, Kelvin_effect = Kelvin_effect, aq_oxidation = aq_oxidation)    
    
    if constant_gas == False:
        dydt = gas_chemistry.gas_phase(y, dydt, bins, Ntot, dy_phasechange, constant_pH = constant_pH, Kelvin_effect = Kelvin_effect)
    
    dydt = parcel.particle_changes(y, dydt, bins, co_condens)
    
    '''
    for i in range(0, 79):
        if y[i] <= 0 and dydt[i] < 0:
            dydt[i] = 0
            
    for i in range(79+6*bins, len(dydt)):
        if y[i] <= 0 and dydt[i] < 0:
            dydt[i] = 0
    '''

    return dydt



