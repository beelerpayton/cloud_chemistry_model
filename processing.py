#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:15:10 2022

@author: paytonbeeler
"""

import pickle, os, indexing
import matplotlib.pyplot as plt
from pyrcel import thermo
import numpy as np

    

def classify_particles(filename):
    
    file = open(filename, 'rb')
    data = pickle.load(file)
    
    mass_thresholds = data['mass_thresholds']
    
    activated = []
    unactivated = []
    
    for i in range(len(data['Dp'][0])):
        radius = 0.5*data['Dp'][-1, i]
        T = data['T'][-1]
        dry_radius = 0.5*data['Ddry'][-1, i]
        kappa = data['kappa'][-1, i]
        r_crit, SS_crit = thermo.kohler_crit(T, dry_radius, kappa)
        if radius >= r_crit:
            activated.append(i)
        else:
            unactivated.append(i)

    dry_species = ['SO4--', 'NO3-', 'Cl', 'NH4+', 'MSA', 'ARO1', 'ARO2', 
                   'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
                   'CO3--', 'Na', 'Ca', 'OIN', 'OC', 'BC']
    
    model_setup = {'organics': ['ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 'OC'],
                   'IEPOX': ['IEPOX', 'IEPOXOS', 'tetrol'],
                   'dust': ['dust'],
                   'BC': ['BC'],
                   'nitrate': ['NO3-'],
                   'sulfate': ['SO4--'],
                   'others': ['Cl', 'NH4+', 'MSA', 'CO3--', 'Na', 'Ca', 'OIN']}
    
    Ni_activated = {'organics': 0, 'IEPOX': 0, 'dust': 0, 'BC': 0, 'nitrate': 0, 
               'sulfate/nitrate': 0, 'other': 0}
    
    Nf_activated = {'organics': 0, 'IEPOX': 0, 'dust': 0, 'BC': 0, 'nitrate': 0, 
               'sulfate/nitrate': 0, 'other': 0}
    
    Ni_unactivated = {'organics': 0, 'IEPOX': 0, 'dust': 0, 'BC': 0, 'nitrate': 0, 
               'sulfate/nitrate': 0, 'other': 0}
    
    Nf_unactivated = {'organics': 0, 'IEPOX': 0, 'dust': 0, 'BC': 0, 'nitrate': 0, 
               'sulfate/nitrate': 0, 'other': 0}
        
    #print(activated)
    #activated = [0]
    
    # initial composition of activated particles
    for particle in activated:
        
        #find total mass of particle at t=0
        total_mass = 0
        for species in dry_species:
            total_mass += data[species][0, particle]
        
        #find mass of each species at t=0
        org_mass = 0
        other_mass = 0
        masses = {}
        masses['IEPOX'] = data['IEPOX'][0, particle]/total_mass
        masses['dust'] = data['dust'][0, particle]/total_mass
        masses['BC'] = data['BC'][0, particle]/total_mass
        masses['nitrate'] = data['NO3-'][0, particle]/total_mass
        masses['sulfate'] = data['SO4--'][0, particle]/total_mass

        for species in model_setup['organics']:
            org_mass += data[species][0, particle]
        org_mass = org_mass/total_mass
        masses['organics'] = org_mass
        
        for species in model_setup['others']:
            other_mass += data[species][0, particle]
        other_mass = other_mass/total_mass
        masses['others'] = other_mass
        
        # sort masses
        masses = dict(sorted(masses.items(), key=lambda item: item[1]))
        
        #find out which group the particle belongs to
        if list(masses.keys())[6] == 'sulfate' and list(masses.keys())[5] == 'nitrate':
            #print(particle, 'sulfate/nitrate')
            Ni_activated['sulfate/nitrate'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'nitrate' and list(masses.keys())[5] == 'sulfate':
            #print(particle, 'sulfate/nitrate')
            Ni_activated['sulfate/nitrate'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'nitrate':
            #print(particle, 'nitrate')
            Ni_activated['nitrate'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'BC':
            #print(particle, 'BC')
            Ni_activated['BC'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'IEPOX':
            #print(particle, 'IEPOX')
            Ni_activated['IEPOX'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'dust':
            #print(particle, 'dust')
            Ni_activated['dust'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'organics':
            #print(particle, 'organics')
            Ni_activated['organics'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'others':
            #print(particle, 'others')
            Ni_activated['other'] += data['N'][0, particle]
          
      
    # final composition of activated particles
    for particle in activated:
        
        #find total mass of particle at t=t_end
        total_mass = 0
        for species in dry_species:
            total_mass += data[species][-1, particle]
        
        #find mass of each species at t=0
        org_mass = 0
        other_mass = 0
        masses = {}
        masses['IEPOX'] = data['IEPOX'][-1, particle]/total_mass
        masses['dust'] = data['dust'][-1, particle]/total_mass
        masses['BC'] = data['BC'][-1, particle]/total_mass
        masses['nitrate'] = data['NO3-'][-1, particle]/total_mass
        masses['sulfate'] = data['SO4--'][-1, particle]/total_mass

        for species in model_setup['organics']:
            org_mass += data[species][-1, particle]
        org_mass = org_mass/total_mass
        masses['organics'] = org_mass
        
        for species in model_setup['others']:
            other_mass += data[species][-1, particle]
        other_mass = other_mass/total_mass
        masses['others'] = other_mass
        
        # sort masses
        masses = dict(sorted(masses.items(), key=lambda item: item[1]))
        
        #find out which group the particle belongs to
        if list(masses.keys())[6] == 'sulfate' and list(masses.keys())[5] == 'nitrate':
            #print(particle, 'sulfate/nitrate')
            Nf_activated['sulfate/nitrate'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'nitrate' and list(masses.keys())[5] == 'sulfate':
            #print(particle, 'sulfate/nitrate')
            Nf_activated['sulfate/nitrate'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'nitrate':
            #print(particle, 'nitrate')
            Nf_activated['nitrate'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'BC':
            #print(particle, 'BC')
            Nf_activated['BC'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'IEPOX':
            #print(particle, 'IEPOX')
            Nf_activated['IEPOX'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'dust':
            #print(particle, 'dust')
            Nf_activated['dust'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'organics':
            #print(particle, 'organics')
            Nf_activated['organics'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'others':
            #print(particle, 'others')
            Nf_activated['other'] += data['N'][-1, particle]
    
    

    # initial composition of unactivated particles
    for particle in unactivated:
        
        #find total mass of particle at t=0
        total_mass = 0
        for species in dry_species:
            total_mass += data[species][0, particle]
        
        #find mass of each species at t=0
        org_mass = 0
        other_mass = 0
        masses = {}
        masses['IEPOX'] = data['IEPOX'][0, particle]/total_mass
        masses['dust'] = data['dust'][0, particle]/total_mass
        masses['BC'] = data['BC'][0, particle]/total_mass
        masses['nitrate'] = data['NO3-'][0, particle]/total_mass
        masses['sulfate'] = data['SO4--'][0, particle]/total_mass

        for species in model_setup['organics']:
            org_mass += data[species][0, particle]
        org_mass = org_mass/total_mass
        masses['organics'] = org_mass
        
        for species in model_setup['others']:
            other_mass += data[species][0, particle]
        other_mass = other_mass/total_mass
        masses['others'] = other_mass
        
        # sort masses
        masses = dict(sorted(masses.items(), key=lambda item: item[1]))
        
        #find out which group the particle belongs to
        if list(masses.keys())[6] == 'sulfate' and list(masses.keys())[5] == 'nitrate':
            #print(particle, 'sulfate/nitrate')
            Ni_unactivated['sulfate/nitrate'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'nitrate' and list(masses.keys())[5] == 'sulfate':
            #print(particle, 'sulfate/nitrate')
            Ni_unactivated['sulfate/nitrate'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'nitrate':
            #print(particle, 'nitrate')
            Ni_unactivated['nitrate'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'BC':
            #print(particle, 'BC')
            Ni_unactivated['BC'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'IEPOX':
            #print(particle, 'IEPOX')
            Ni_unactivated['IEPOX'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'dust':
            #print(particle, 'dust')
            Ni_unactivated['dust'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'organics':
            #print(particle, 'organics')
            Ni_unactivated['organics'] += data['N'][0, particle]
        elif list(masses.keys())[6] == 'others':
            #print(particle, 'others')
            Ni_unactivated['other'] += data['N'][0, particle]
    
    
    
    
    
    # final composition of unactivated particles
    for particle in unactivated:
        
        #find total mass of particle at t=t_end
        total_mass = 0
        for species in dry_species:
            total_mass += data[species][-1, particle]
        
        #find mass of each species at t=0
        org_mass = 0
        other_mass = 0
        masses = {}
        masses['IEPOX'] = data['IEPOX'][-1, particle]/total_mass
        masses['dust'] = data['dust'][-1, particle]/total_mass
        masses['BC'] = data['BC'][-1, particle]/total_mass
        masses['nitrate'] = data['NO3-'][-1, particle]/total_mass
        masses['sulfate'] = data['SO4--'][-1, particle]/total_mass

        for species in model_setup['organics']:
            org_mass += data[species][-1, particle]
        org_mass = org_mass/total_mass
        masses['organics'] = org_mass
        
        for species in model_setup['others']:
            other_mass += data[species][-1, particle]
        other_mass = other_mass/total_mass
        masses['others'] = other_mass
        
        # sort masses
        masses = dict(sorted(masses.items(), key=lambda item: item[1]))
        
        #find out which group the particle belongs to
        if list(masses.keys())[6] == 'sulfate' and list(masses.keys())[5] == 'nitrate':
            #print(particle, 'sulfate/nitrate')
            Nf_unactivated['sulfate/nitrate'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'nitrate' and list(masses.keys())[5] == 'sulfate':
            #print(particle, 'sulfate/nitrate')
            Nf_unactivated['sulfate/nitrate'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'nitrate':
            #print(particle, 'nitrate')
            Nf_unactivated['nitrate'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'BC':
            #print(particle, 'BC')
            Nf_unactivated['BC'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'IEPOX':
            #print(particle, 'IEPOX')
            Nf_unactivated['IEPOX'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'dust':
            #print(particle, 'dust')
            Nf_unactivated['dust'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'organics':
            #print(particle, 'organics')
            Nf_unactivated['organics'] += data['N'][-1, particle]
        elif list(masses.keys())[6] == 'others':
            #print(particle, 'others')
            Nf_unactivated['other'] += data['N'][-1, particle]
    
    print(' ')
    print('BELOW CLOUD:')
    Ntot = 0
    for group in Ni_activated:
        Ntot += Ni_activated[group]
        Ntot += Ni_unactivated[group]
    for group in Nf_activated:
        if Ntot > 0:
            print(group, (Ni_activated[group] + Ni_unactivated[group])/Ntot)
        else:
            print(group, 'N/A')
    print(' ')
    
    print('IN CLOUD:')
    print('activated:')
    Ntot = 0
    for group in Nf_activated:
        Ntot += Nf_activated[group]
    for group in Nf_activated:
        if Ntot > 0:
            print(group, Nf_activated[group]/Ntot)
        else:
            print(group, 'N/A')
    print(' ')
    print('unactivated:')
    Ntot = 0
    for group in Nf_unactivated:
        Ntot += Nf_unactivated[group]
    for group in Nf_unactivated:
        if Ntot > 0:
            print(group, Nf_unactivated[group]/Ntot)
        else:
            print(group, 'N/A')
    
    
    #print(Ni_activated)
    #print(Nf_activated)    
    #print(' ')
    #print(Ni_unactivated)
    #print(Nf_unactivated) 


    return    
    
    
    
    
    
    
    