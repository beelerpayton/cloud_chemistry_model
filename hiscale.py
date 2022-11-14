#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:19:43 2022

@author: paytonbeeler
"""

#import os
import numpy as np
import matplotlib.pyplot as plt
import indexing, tqdm
import aqueous_chemistry as aq_chemistry

def setup(bins, sampling_times, pH0, cloud_flag, CVI_flag, mass_thresholds, plots=False):
    
    print('Initializing model with HISCALE data.')
    y0 = np.zeros(79+60*bins+bins)
    
    model_species = {'BC': ['soot'],
                     'sulfate/nitrate': ['sulfate_nitrate_org'],
                     'nitrate': ['nitrate_amine_org'], 
                     'organics': ['org28', 'org30_43', 'BB_SOA', 'org_amines', 'BB', 'pyridine'], 
                     'dust': ['Dust'],
                     'IEPOX': ['IEPOX_SOA']}
   
    Ddrys, Ns = sample_diameters(sampling_times[0], sampling_times[1], bins, cloud_flag, CVI_flag, plot = plots)
    particles = sample_composition(sampling_times[0], sampling_times[1], bins, model_species, cloud_flag, CVI_flag, plot = plots)
    
    kappa_i = {'SO4--': 0.65, 'NO3-': 0.65, 'Cl': 0.53, 'NH4+': 0.65, 
               'MSA': 0.53, 'ARO1': 0.1, 'ARO2': 0.1, 'ALK1': 0.1, 
               'OLE1': 0.1, 'API1': 0.1, 'API2': 0.1, 'LIM1': 0.1, 
               'LIM2': 0.1, 'CO3--': 0.53, 'Na': 0.53, 'Ca': 0.53, 'OIN': 0.1,
               'OC': 0.001, 'BC': 1.0E-5, 'H2O': 0., 'IEPOX': 0.1, 'dust': 1.0E-5,
               'IEPOXOS': 0.1, 'tetrol': 0.1}
    
    #kg/m^3
    density_i = {'SO4--': 1800., 'NO3-': 1800., 'Cl': 2200., 'NH4+': 1800., 
                 'MSA': 1800., 'ARO1': 1400., 'ARO2': 1400., 'ALK1': 1400., 
                 'OLE1': 1400., 'API1': 1400., 'API2': 1400., 'LIM1': 1400., 
                 'LIM2': 1400., 'CO3--': 2600., 'Na': 2200., 'Ca': 2600., 
                 'OIN': 2600., 'OC': 1000., 'BC': 1800., 'H2O': 1000., 
                 'IEPOX': 1400., 'dust': 2500., 'IEPOXOS': 1400., 'tetrol': 1400.}  
    
    pbar = tqdm.tqdm(total = bins)
    for i in range(0, bins):
        if particles[i] == 'organics':
            y0 = organics(y0, kappa_i, density_i, mass_thresholds['organics'], i, Ddrys, Ns, bins, pH0)
        
        elif particles[i] == 'IEPOX':
            y0 = IEPOX(y0, kappa_i, density_i, mass_thresholds['IEPOX'], i, Ddrys, Ns, bins, pH0)
        
        elif particles[i] == 'dust':
            y0 = dust(y0, kappa_i, density_i, mass_thresholds['dust'], i, Ddrys, Ns, bins, pH0)
        
        elif particles[i] == 'BC':
            y0 = BC(y0, kappa_i, density_i, mass_thresholds['BC'], i, Ddrys, Ns, bins, pH0)
        
        elif particles[i] == 'nitrate':
            nitrate_threshold = mass_thresholds['nitrate']['nitrate']
            organic_threshold = mass_thresholds['nitrate']['organics']
            mass_threshold = nitrate_threshold + organic_threshold
            y0 = nitrate(y0, kappa_i, density_i, mass_threshold, i, Ddrys, Ns, bins, pH0, nitrate_threshold, organic_threshold)
        
        elif particles[i] == 'sulfate/nitrate':
            sulfate_threshold = mass_thresholds['sulfate/nitrate']['sulfate']
            nitrate_threshold = mass_thresholds['sulfate/nitrate']['nitrate']
            organic_threshold = mass_thresholds['sulfate/nitrate']['organics']
            mass_threshold = sulfate_threshold + nitrate_threshold + organic_threshold
            y0 = sulfate_nitrate(y0, kappa_i, density_i, mass_threshold, i, Ddrys, Ns, bins, pH0, sulfate_threshold, nitrate_threshold, organic_threshold)
            
        pbar.update(1)
    pbar.close()
        
    y0, SS, P, T = gas_phase(y0, sampling_times[0], sampling_times[1], bins)
    
    y0[indexing.getIndex('ISOP (gas)', bins)] = 1e-9
    
    kappas = y0[79+2*bins:79+2*bins+bins]
    Dps, wv, wc = aq_chemistry.equilibrate_h2o(Ddrys, kappas, Ns, SS, P, T)
    
    for i in range(0, bins):
        name = 'Dp, ' + str(i+1)
        y0[indexing.getIndex(name, bins)] = np.log(Dps[i])
    y0[indexing.getIndex('wv', bins)] = wv
    y0[indexing.getIndex('wc', bins)] = wc

    return y0


def sample_diameters(start_time, end_time, bins, cloud_flag, CVI_flag, plot = True):
    
    filename = 'HISCALE_data.txt'
    raw_data = np.loadtxt(filename, dtype='str')
    hiscale = {}
    units = {}
    hiscale[str(raw_data[0, 0])] = np.array(raw_data[2:, 0], dtype='float64')
    units[str(raw_data[0, 0])] = raw_data[1, 0]
    hiscale['N_Dp'] = np.array(raw_data[2:, 1:56], dtype = 'float64')
    units['N_Dp'] = 'nm'
    hiscale['Dp_lower'] = np.array([9.38, 10.66, 12.1, 13.74, 15.61, 17.73, 20.13, 22.86, 25.96, 29.48, 33.48, 38.02, 43.18, 49.04, 55.69, 63.25, 71.82, 81.57, 92.63, 105.2, 119.47, 135.67, 154.08, 174.98, 198.71, 225.67, 256.28, 291.04, 330.52, 375.35, 426.27, 484.09, 549.75, 624.33, 709.02, 805.19, 914.41, 1038.45, 1179.31, 1339.28, 1520.95, 1727.27, 1961.57, 2227.65, 2529.82, 2872.99, 3262.7, 3705.27, 4207.88, 4778.67, 5426.89, 6163.03, 6999.03, 7948.42, 9026.61])
    units['Dp_lower'] = 'nm'
    hiscale['Dp_upper'] = np.array([10.66, 12.1, 13.74, 15.61, 17.73, 20.13, 22.86, 25.96, 29.48, 33.48, 38.02, 43.18, 49.04, 55.69, 63.25, 71.82, 81.57, 92.63, 105.2, 119.47, 135.67, 154.08, 174.98, 198.71, 225.67, 256.28, 291.04, 330.52, 375.35, 426.27, 484.09, 549.75, 624.33, 709.02, 805.19, 914.41, 1038.45, 1179.31, 1339.28, 1520.95, 1727.27, 1961.57, 2227.65, 2529.82, 2872.99, 3262.7, 3705.27, 4207.88, 4778.67, 5426.89, 6163.03, 6999.03, 7948.42, 9026.61, 10251.04])
    units['Dp_upper'] = 'nm'
    hiscale['Dp_mid'] = np.array([10.00, 11.36, 12.89, 14.65, 16.64, 18.89, 21.45, 24.36, 27.66, 31.42, 35.68, 40.52, 46.02, 52.26, 59.35, 67.40, 76.54, 86.92, 98.72, 112.11, 127.31, 144.58, 164.20, 186.47, 211.76, 240.49, 273.11, 310.15, 352.22, 400.00, 454.26, 515.88, 585.85, 665.33, 755.58, 858.06, 974.46, 1106.64, 1256.75, 1427.23, 1620.83, 1840.70, 2090.38, 2373.93, 2695.95, 3061.65, 3476.95, 3948.59, 4484.20, 5092.48, 5783.26, 6567.74, 7458.63, 8470.38, 9619.36])
    units['Dp_mid'] = 'nm'
    for i in range(56, len(raw_data[0])): 
        hiscale[str(raw_data[0, i])] = np.array(raw_data[2:, i], dtype = 'float64')
        units[str(raw_data[0, i])] = str(raw_data[1, i])
    
    start_index = -1
    stop_index = -1
    
    for i in range(0, len(hiscale['N_Dp'])):
        if hiscale['Start_UTC'][i]/60 >= start_time and start_index == -1:
            start_index = i
        elif hiscale['Start_UTC'][i]/60 >= end_time and stop_index == -1:
            stop_index = i+1
    
    reduced_data = np.zeros(0)
    sampled_points = np.zeros(0)
    for i in range(start_index, stop_index):
        if hiscale['Cloud_flag'][i] == cloud_flag and hiscale['CVI_flag'][i] == CVI_flag and hiscale['N_Dp'][i][0] >= 0:
            if len(reduced_data) == 0:
                reduced_data = hiscale['N_Dp'][i]
                sampled_points = np.append(sampled_points, i)
            else:
                reduced_data = np.vstack((reduced_data, hiscale['N_Dp'][i]))
                sampled_points = np.append(sampled_points, i)
    
    if len(reduced_data) > 0:
        avg_N = np.zeros(len(reduced_data[0]))
        error_N = np.zeros(len(reduced_data[0]))
        
        for i in range(0, len(hiscale['N_Dp'][0])): #columns
            temp_data = np.zeros(0)
            for j in range(0, len(reduced_data)): #rows
                if reduced_data[j, i] >= 0:
                    temp_data = np.append(temp_data, reduced_data[j, i])
            
            avg_N[i] = np.mean(temp_data)
            error_N[i] = np.std(temp_data)
        
        Ntot = np.sum(avg_N)
        normalized = avg_N/Ntot
        CDF = np.zeros(len(avg_N))
        for i in range(1, len(CDF)):
            CDF[i] = np.sum(normalized[0:i])
    
    if plot == True:
              
        fig, (ax1) = plt.subplots(1, 1, figsize=(1*6.4, 1*4.8), constrained_layout=True, sharex=True)    
        ax1.plot(hiscale['Start_UTC']/3600, hiscale['Radar_Alt'], '.', color='C0', markersize=1.5)#, label='in cloud')
        for point in sampled_points:
            ax1.plot(hiscale['Start_UTC'][int(point)]/3600, hiscale['Radar_Alt'][int(point)], '.', color='C1', markersize=1.5)#, label='in cloud')
        ax1.plot(-10, -10, 's', color = 'C1', label = 'INCLUDED IN AVERAGING')
        ax1.axvline(hiscale['Start_UTC'][start_index]/3600, 0, 1e6, color = 'k', linestyle = 'dashed')
        ax1.axvline(hiscale['Start_UTC'][stop_index]/3600, 0, 1e6, color = 'k', linestyle = 'dashed')
        ticks = [16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5]
        labels = ['16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30']
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0, 10000) 
        ax1.set_xlim(16, 19.5)   
        ax1.set_xlabel('Time on April 25, 2016')
        ax1.set_ylabel('Altitude (m)')
        ax1.legend(ncol=2, loc='upper center', frameon=False)
        #ax1.set_xlim(16.5, 16.75)
        
        if len(reduced_data) > 0:
            fig, ax1 = plt.subplots(1, 1, figsize=(1*6.4, 1*4.8), constrained_layout=True, sharex=True)    
            ax1.errorbar(hiscale['Dp_mid'], avg_N, yerr = error_N, fmt = 'bo')
            ax1.set_xscale('log')
            ax1.set_ylim(0, 150)
            ax1.set_ylabel(r'N (cm$^{-3}$)', fontsize=12)
            ax1.text(0.95, 0.95, str(np.round((hiscale['Start_UTC'][stop_index]/60)-(hiscale['Start_UTC'][start_index]/60),2))+' minute average', transform = ax1.transAxes, ha='right', va='top', fontsize=12)    
            #ax2.plot(hiscale['Dp_mid'], CDF, '-b')
            #ax2.set_ylabel('CDF')
            ax1.set_xlabel('Diameter (nm)', fontsize=12)
            #ax2.set_ylim(0, 1.1)
        
    Dps = np.zeros(bins)
    Ns = np.zeros(bins)
    
    if len(reduced_data) > 0:
        for i in range(0, bins):
            rand = np.random.rand(1)
            breaker = False
            for j in range(0, len(CDF)-1):
                if rand[0] >= CDF[j] and rand[0] < CDF[j+1] and breaker == False:
                    Dps[i] = hiscale['Dp_mid'][j]*1e-9 # m
                    Ns[i] = avg_N[j]*(100**3) # particles per m^3
                    breaker = True
    
    return Dps, Ns
  



  
def sample_composition(start_time, end_time, bins, model_species, cloud_flag, CVI_flag, plot = True):
    
    filename = 'HISCALE_data.txt'
    raw_data = np.loadtxt(filename, dtype='str')
    CVI_data = {}
    units = {}
    CVI_data[str(raw_data[0, 0])] = np.array(raw_data[2:, 0], dtype='float64')
    units[str(raw_data[0, 0])] = raw_data[1, 0]
    CVI_data['N_Dp'] = np.array(raw_data[2:, 1:56], dtype = 'float64')
    units['N_Dp'] = 'nm'
    CVI_data['Dp_lower'] = np.array([9.38, 10.66, 12.1, 13.74, 15.61, 17.73, 20.13, 22.86, 25.96, 29.48, 33.48, 38.02, 43.18, 49.04, 55.69, 63.25, 71.82, 81.57, 92.63, 105.2, 119.47, 135.67, 154.08, 174.98, 198.71, 225.67, 256.28, 291.04, 330.52, 375.35, 426.27, 484.09, 549.75, 624.33, 709.02, 805.19, 914.41, 1038.45, 1179.31, 1339.28, 1520.95, 1727.27, 1961.57, 2227.65, 2529.82, 2872.99, 3262.7, 3705.27, 4207.88, 4778.67, 5426.89, 6163.03, 6999.03, 7948.42, 9026.61])
    units['Dp_lower'] = 'nm'
    CVI_data['Dp_upper'] = np.array([10.66, 12.1, 13.74, 15.61, 17.73, 20.13, 22.86, 25.96, 29.48, 33.48, 38.02, 43.18, 49.04, 55.69, 63.25, 71.82, 81.57, 92.63, 105.2, 119.47, 135.67, 154.08, 174.98, 198.71, 225.67, 256.28, 291.04, 330.52, 375.35, 426.27, 484.09, 549.75, 624.33, 709.02, 805.19, 914.41, 1038.45, 1179.31, 1339.28, 1520.95, 1727.27, 1961.57, 2227.65, 2529.82, 2872.99, 3262.7, 3705.27, 4207.88, 4778.67, 5426.89, 6163.03, 6999.03, 7948.42, 9026.61, 10251.04])
    units['Dp_upper'] = 'nm'
    CVI_data['Dp_mid'] = np.array([10.00, 11.36, 12.89, 14.65, 16.64, 18.89, 21.45, 24.36, 27.66, 31.42, 35.68, 40.52, 46.02, 52.26, 59.35, 67.40, 76.54, 86.92, 98.72, 112.11, 127.31, 144.58, 164.20, 186.47, 211.76, 240.49, 273.11, 310.15, 352.22, 400.00, 454.26, 515.88, 585.85, 665.33, 755.58, 858.06, 974.46, 1106.64, 1256.75, 1427.23, 1620.83, 1840.70, 2090.38, 2373.93, 2695.95, 3061.65, 3476.95, 3948.59, 4484.20, 5092.48, 5783.26, 6567.74, 7458.63, 8470.38, 9619.36])
    units['Dp_mid'] = 'nm'
    for i in range(56, len(raw_data[0])): 
        CVI_data[str(raw_data[0, i])] = np.array(raw_data[2:, i], dtype = 'float64')
        units[str(raw_data[0, i])] = str(raw_data[1, i])
    
    #start_index = -1
    #stop_index = -1
    
    #for i in range(0, len(hiscale['N_Dp'])):
    #    if CVI_data['Start_UTC'][i]/60 >= start_time and start_index == -1:
    #        start_index = i
    #    elif CVI_data['Start_UTC'][i]/60 >= end_time and stop_index == -1:
    #        stop_index = i+1    
    
    
    filename = 'Splat_Composition_25-Apr-2016.txt'
    raw_data = np.loadtxt(filename, dtype='str')
    hiscale = {}
    for i in range(0, len(raw_data[0])): 
        hiscale[str(raw_data[0, i])] = np.array(raw_data[1:, i], dtype = 'float64')
     
    CVI_times = np.zeros(len(hiscale['Time']))
    cloud_times = np.zeros(len(hiscale['Time']))
    for i in range(0, len(hiscale['Time'])):
        for j in range(0, len(CVI_data['N_Dp'])):
            if CVI_data['Start_UTC'][j] == hiscale['Time'][i]:
                CVI_times[i] = CVI_data['CVI_flag'][j]
                cloud_times[i] = CVI_data['Cloud_flag'][j]
    hiscale['CVI_flag'] = CVI_times
    hiscale['Cloud_flag'] = cloud_times
    
    start_index = -1
    stop_index = -1
    for i in range(0, len(hiscale['Time'])):
        if hiscale['Time'][i]/60 >= start_time and start_index == -1:
            start_index = i
        elif hiscale['Time'][i]/60 >= end_time and stop_index == -1:
            stop_index = i+1
    
   
    minisplat_species = ['soot', 'sulfate_nitrate_org', 
                         'nitrate_amine_org', 'org28', 'org30_43', 
                         'BB_SOA', 'org_amines', 'BB', 'Dust', 'pyridine', 
                         'IEPOX_SOA']
    
    reduced_fraction = {}
    for reduced_species in model_species:
        summation = np.zeros(len(hiscale['Time']))
        for species in model_species[reduced_species]:
            for i in range(0, len(hiscale['Time'])):
                summation[i] = summation[i] + hiscale[species][i]
        reduced_fraction[reduced_species] = summation    
    
    if plot == True:
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']        
        fig, (ax1) = plt.subplots(1, 1, figsize=(1*6.4, 1*4.8), constrained_layout=True, sharex=True)    
        bottom = np.zeros(len(hiscale['Time']))     
        for species, color in zip(minisplat_species, colors):
            ax1.fill_between(hiscale['Time']/3600, bottom, bottom+hiscale[species], color=color, label=species)
            bottom = bottom+hiscale[species]
        ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)
        ticks = [16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5]
        labels = ['16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30']
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(np.min(hiscale['Time'])/3600, np.max(hiscale['Time'])/3600)
        ax1.set_xlabel('Time on April 25, 2016')
        ax1.set_ylabel('Number Fraction')
        ax1.axvline(hiscale['Time'][start_index]/3600, 0, 1, color='k', linestyle='dashed')
        ax1.axvline(hiscale['Time'][stop_index]/3600, 0, 1, color='k', linestyle='dashed')
        
        colors = ['C0', 'C1', 'C2', 'C4', 'C8', 'C6']        
        fig, (ax1) = plt.subplots(1, 1, figsize=(1*6.4, 1*4.8), constrained_layout=True, sharex=True)    
        bottom = np.zeros(len(hiscale['Time']))     
        for species, color in zip(model_species, colors):
            ax1.fill_between(hiscale['Time']/3600, bottom, bottom+reduced_fraction[species], color=color, label=species)
            bottom = bottom+reduced_fraction[species]
        ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)
        ticks = [16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5]
        labels = ['16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30']
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(np.min(hiscale['Time'])/3600, np.max(hiscale['Time'])/3600)
        #ax1.set_xlim(990/60, 1006/60)
        ax1.set_xlabel('Time on April 25, 2016')
        ax1.set_ylabel('Number Fraction')
        #ax1.axvline(hiscale['Time'][start_index]/3600, 0, 1, color='k', linestyle='dashed')
        #ax1.axvline(hiscale['Time'][stop_index]/3600, 0, 1, color='k', linestyle='dashed')
        ax1.axvline(996/60, 0, 1, color='k', linestyle='dashed')
        ax1.axvline(1001/60, 0, 1, color='k', linestyle='dashed')
        ax1.axvline(981/60, 0, 1, color='k', linestyle='dashed')
        ax1.axvline(986/60, 0, 1, color='k', linestyle='dashed')
        
    
    avg_comp = {}
    for species in model_species:
        data = 0
        points = 0
        for i in range(start_index, stop_index):
            if hiscale['CVI_flag'][i] == CVI_flag and hiscale['Cloud_flag'][i] == cloud_flag:
                data += reduced_fraction[species][i]
                points += 1.0
        if points > 0:
            avg_comp[species] = data/points
        else:
            avg_comp[species] = 0
    #avg_comp = dict(sorted(avg_comp.items(), key=lambda item: item[1]))
    #print(avg_comp)
    
    if plot == True:
        species = []
        value = []
        for x in avg_comp:
            species.append(x)
            value.append(avg_comp[x])
        if np.sum(value) > 0:    
            fig, (ax1) = plt.subplots(1, 1, figsize=(1*6.4, 1*4.8), constrained_layout=True, sharex=True)    
            ax1.pie(value, colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.0}, labels=species)
            if cloud_flag == 0.0 and CVI_flag == 0.0:
                ax1.set_title('Below Cloud')
            elif cloud_flag == 1.0 and CVI_flag == 0.0:
                ax1.set_title('Interstitials')
            else:
                ax1.set_title('Cloud Droplet Residuals')
    
    
    species = np.zeros(0)
    probability = np.zeros(0)
    for x in avg_comp:
        species = np.append(species, x)
        probability = np.append(probability, avg_comp[x])
    probability[-1] = 1.0 - np.sum(probability[:-1])
    
    #for j in range(1, len(species)+1):
    #    print(species[j-1], np.sum(probability[:j]))
    
    
    particles = np.array(np.zeros(bins), dtype=str)
    for i in range(0, bins):
        rand = np.random.rand(1)
        breaker = -1
        for j in range(1, len(species)+1):
            if rand <= np.sum(probability[:j]) and breaker == -1:
                particles[i] = species[j-1]
                breaker = 1.0

    return particles




def organics(y0, kappa_i, density_i, mass_threshold, i, Dps, Ns, bins, pH0):
    
    included = ['ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 
                'LIM2', 'OC']
    others = ['SO4--', 'NO3-', 'Cl', 'NH4+', 'MSA', 'CO3--', 'Na', 'Ca', 
              'OIN', 'BC', 'IEPOXOS', 'tetrol']
    
    incl_composition = np.zeros(len(included))
    other_composition = np.zeros(len(others))
    incl_mass = 0
    other_mass = 1.0
    volume_threshold = mass_threshold
    
    
    while incl_mass/(incl_mass+other_mass) < mass_threshold:
        incl_composition = np.zeros(len(included))
        other_composition = np.zeros(len(others))
        Vtot = 0
        while Vtot < volume_threshold:
            Vi = volume_threshold*np.random.rand(1)
            Vi = Vi[0]
            species = np.floor(len(included)*np.random.rand(1))
            incl_composition[int(species)] = incl_composition[int(species)] + Vi
            Vtot += Vi
        while Vtot <= 1.0:
            Vi = (1-volume_threshold)*np.random.rand(1)
            Vi = Vi[0]
            species = np.floor(len(others)*np.random.rand(1))
            other_composition[int(species)] = other_composition[int(species)] + Vi
            Vtot += Vi
    
        if Vtot > 1.0:
            multiplier = 1.0/Vtot
            for j in range(0, len(incl_composition)):
                incl_composition[j] = multiplier*incl_composition[j]
            for j in range(0, len(other_composition)):
                other_composition[j] = multiplier*other_composition[j]
        
        incl_mass = 0
        other_mass = 0
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        for j in range(0, len(incl_composition)):
            Vi = incl_composition[j]*dry_volume
            incl_mass += Vi*density_i[included[j]]
        for j in range(0, len(other_composition)):
            Vi = other_composition[j]*dry_volume
            other_mass += Vi*density_i[others[j]]
        volume_threshold += 0.001

    name = 'kappa, ' + str(i+1)
    ki_Vi = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        Vi = incl_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi    
    for j in range(0, len(others)):
        aq_species = others[j]
        Vi = other_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi   

    y0[indexing.getIndex(name, bins)] = ki_Vi/dry_volume      
    
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = incl_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = other_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    
    dry_mass = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]

    name = 'density, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = dry_mass/dry_volume

    name = 'Ddry, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = np.log(Dps[i])
    name = 'pH, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = pH0
    name = 'N, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = Ns[i]
    
    return y0




def IEPOX(y0, kappa_i, density_i, mass_threshold, i, Dps, Ns, bins, pH0):
    
    included = ['IEPOX', 'IEPOXOS', 'tetrol']
    others = ['SO4--', 'NO3-', 'Cl', 'NH4+', 'MSA', 'ARO1', 'ARO2', 
              'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 'CO3--', 
              'Na', 'Ca', 'OIN', 'OC', 'BC', 'dust']
    
    incl_composition = np.zeros(len(included))
    other_composition = np.zeros(len(others))
    incl_mass = 0
    other_mass = 1.0
    volume_threshold = mass_threshold
    
    
    while incl_mass/(incl_mass+other_mass) < mass_threshold:
        incl_composition = np.zeros(len(included))
        other_composition = np.zeros(len(others))
        Vtot = 0
        while Vtot < volume_threshold:
            Vi = volume_threshold*np.random.rand(1)
            Vi = Vi[0]
            species = np.floor(len(included)*np.random.rand(1))
            incl_composition[int(species)] = incl_composition[int(species)] + Vi
            Vtot += Vi
        while Vtot <= 1.0:
            Vi = (1-volume_threshold)*np.random.rand(1)
            Vi = Vi[0]
            species = np.floor(len(others)*np.random.rand(1))
            other_composition[int(species)] = other_composition[int(species)] + Vi
            Vtot += Vi
    
        if Vtot > 1.0:
            multiplier = 1.0/Vtot
            for j in range(0, len(incl_composition)):
                incl_composition[j] = multiplier*incl_composition[j]
            for j in range(0, len(other_composition)):
                other_composition[j] = multiplier*other_composition[j]
        
        incl_mass = 0
        other_mass = 0
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        for j in range(0, len(incl_composition)):
            Vi = incl_composition[j]*dry_volume
            incl_mass += Vi*density_i[included[j]]
        for j in range(0, len(other_composition)):
            Vi = other_composition[j]*dry_volume
            other_mass += Vi*density_i[others[j]]
        volume_threshold += 0.001

    name = 'kappa, ' + str(i+1)
    ki_Vi = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        Vi = incl_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi    
    for j in range(0, len(others)):
        aq_species = others[j]
        Vi = other_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi   

    y0[indexing.getIndex(name, bins)] = ki_Vi/dry_volume      
    
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = incl_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = other_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    
    dry_mass = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]

    name = 'density, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = dry_mass/dry_volume

    name = 'Ddry, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = np.log(Dps[i])
    name = 'pH, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = pH0
    name = 'N, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = Ns[i]
    
    return y0



def dust(y0, kappa_i, density_i, mass_threshold, i, Dps, Ns, bins, pH0):
    
    included = ['dust']
    others = ['SO4--', 'NO3-', 'Cl', 'NH4+', 'MSA', 'ARO1', 'ARO2', 
              'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 'CO3--', 
              'Na', 'Ca', 'OIN', 'OC', 'BC', 'IEPOX', 'IEPOXOS', 'tetrol']
    
    incl_composition = np.zeros(len(included))
    other_composition = np.zeros(len(others))
    incl_mass = 0
    other_mass = 1.0
    volume_threshold = mass_threshold
    
    
    while incl_mass/(incl_mass+other_mass) < mass_threshold:
        incl_composition = np.zeros(len(included))
        other_composition = np.zeros(len(others))
        Vtot = 0
        while Vtot < volume_threshold:
            Vi = volume_threshold*np.random.rand(1)
            Vi = Vi[0]
            species = np.floor(len(included)*np.random.rand(1))
            incl_composition[int(species)] = incl_composition[int(species)] + Vi
            Vtot += Vi
        while Vtot <= 1.0:
            Vi = (1-volume_threshold)*np.random.rand(1)
            Vi = Vi[0]
            species = np.floor(len(others)*np.random.rand(1))
            other_composition[int(species)] = other_composition[int(species)] + Vi
            Vtot += Vi
    
        if Vtot > 1.0:
            multiplier = 1.0/Vtot
            for j in range(0, len(incl_composition)):
                incl_composition[j] = multiplier*incl_composition[j]
            for j in range(0, len(other_composition)):
                other_composition[j] = multiplier*other_composition[j]
        
        incl_mass = 0
        other_mass = 0
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        for j in range(0, len(incl_composition)):
            Vi = incl_composition[j]*dry_volume
            incl_mass += Vi*density_i[included[j]]
        for j in range(0, len(other_composition)):
            Vi = other_composition[j]*dry_volume
            other_mass += Vi*density_i[others[j]]
        volume_threshold += 0.001

    name = 'kappa, ' + str(i+1)
    ki_Vi = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        Vi = incl_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi    
    for j in range(0, len(others)):
        aq_species = others[j]
        Vi = other_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi   

    y0[indexing.getIndex(name, bins)] = ki_Vi/dry_volume      
    
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = incl_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = other_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    
    dry_mass = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]

    name = 'density, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = dry_mass/dry_volume

    name = 'Ddry, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = np.log(Dps[i])
    name = 'pH, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = pH0
    name = 'N, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = Ns[i]
    
    return y0


def BC(y0, kappa_i, density_i, mass_threshold, i, Dps, Ns, bins, pH0):
    
    included = ['BC']
    others = ['SO4--', 'NO3-', 'Cl', 'NH4+', 'MSA', 'ARO1', 'ARO2',
              'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 'CO3--',
              'Na', 'Ca', 'OIN', 'OC', 'IEPOX', 'dust', 'IEPOXOS', 'tetrol']
    
    incl_composition = np.zeros(len(included))
    other_composition = np.zeros(len(others))
    incl_mass = 0
    other_mass = 1.0
    volume_threshold = mass_threshold
    
    
    while incl_mass/(incl_mass+other_mass) < mass_threshold:
        incl_composition = np.zeros(len(included))
        other_composition = np.zeros(len(others))
        Vtot = 0
        while Vtot < volume_threshold:
            Vi = volume_threshold*np.random.rand(1)
            Vi = Vi[0]
            species = np.floor(len(included)*np.random.rand(1))
            incl_composition[int(species)] = incl_composition[int(species)] + Vi
            Vtot += Vi
        while Vtot <= 1.0:
            Vi = (1-volume_threshold)*np.random.rand(1)
            Vi = Vi[0]
            species = np.floor(len(others)*np.random.rand(1))
            other_composition[int(species)] = other_composition[int(species)] + Vi
            Vtot += Vi
    
        if Vtot > 1.0:
            multiplier = 1.0/Vtot
            for j in range(0, len(incl_composition)):
                incl_composition[j] = multiplier*incl_composition[j]
            for j in range(0, len(other_composition)):
                other_composition[j] = multiplier*other_composition[j]
        
        incl_mass = 0
        other_mass = 0
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        for j in range(0, len(incl_composition)):
            Vi = incl_composition[j]*dry_volume
            incl_mass += Vi*density_i[included[j]]
        for j in range(0, len(other_composition)):
            Vi = other_composition[j]*dry_volume
            other_mass += Vi*density_i[others[j]]
        volume_threshold += 0.001

    name = 'kappa, ' + str(i+1)
    ki_Vi = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        Vi = incl_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi    
    for j in range(0, len(others)):
        aq_species = others[j]
        Vi = other_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi   

    y0[indexing.getIndex(name, bins)] = ki_Vi/dry_volume      
    
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = incl_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = other_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    
    dry_mass = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]

    name = 'density, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = dry_mass/dry_volume

    name = 'Ddry, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = np.log(Dps[i])
    name = 'pH, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = pH0
    name = 'N, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = Ns[i]
    
    return y0




def nitrate(y0, kappa_i, density_i, mass_threshold, i, Dps, Ns, bins, pH0, nitrate_threshold, org_threshold):
    
    included = ['NO3-', 'ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 
                'LIM2', 'OC']
    others = ['SO4--', 'Cl', 'NH4+', 'MSA', 'CO3--', 'Na', 
              'Ca', 'OIN', 'BC', 'IEPOX', 'dust', 'IEPOXOS', 'tetrol']
    
    incl_composition = np.zeros(len(included))
    other_composition = np.zeros(len(others))
    incl_mass = 0
    other_mass = 1.0

    volume_threshold = nitrate_threshold+org_threshold

    
    while incl_mass/(incl_mass+other_mass) < mass_threshold:
        incl_composition = np.zeros(len(included))
        other_composition = np.zeros(len(others))
        V_nit = 0
        V_org = 0
        Vtot = 0
        while V_nit < nitrate_threshold:
            Vi = nitrate_threshold*np.random.rand(1)
            Vi = Vi[0]
            species = 0
            incl_composition[int(species)] = incl_composition[int(species)] + Vi
            V_nit += Vi
            Vtot += Vi
        while V_org < org_threshold:
            Vi = org_threshold*np.random.rand(1)
            Vi = Vi[0]
            species = 1+np.floor((len(included)-1)*np.random.rand(1))
            incl_composition[int(species)] = incl_composition[int(species)] + Vi
            V_org += Vi
            Vtot += Vi
        while Vtot <= 1.0:
            Vi = (1-volume_threshold)*np.random.rand(1)
            Vi = Vi[0]
            species = np.floor(len(others)*np.random.rand(1))
            other_composition[int(species)] = other_composition[int(species)] + Vi
            Vtot += Vi
    
        if Vtot > 1.0:
            multiplier = 1.0/Vtot
            for j in range(0, len(incl_composition)):
                incl_composition[j] = multiplier*incl_composition[j]
            for j in range(0, len(other_composition)):
                other_composition[j] = multiplier*other_composition[j]
        
        incl_mass = 0
        other_mass = 0
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        for j in range(0, len(incl_composition)):
            Vi = incl_composition[j]*dry_volume
            incl_mass += Vi*density_i[included[j]]
        for j in range(0, len(other_composition)):
            Vi = other_composition[j]*dry_volume
            other_mass += Vi*density_i[others[j]]
        volume_threshold += 0.001
    
    name = 'kappa, ' + str(i+1)
    ki_Vi = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        Vi = incl_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi    
    for j in range(0, len(others)):
        aq_species = others[j]
        Vi = other_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi   

    y0[indexing.getIndex(name, bins)] = ki_Vi/dry_volume      
    
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = incl_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = other_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    
    dry_mass = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]

    name = 'density, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = dry_mass/dry_volume

    name = 'Ddry, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = np.log(Dps[i])
    name = 'pH, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = pH0
    name = 'N, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = Ns[i]
    
    return y0



def sulfate_nitrate(y0, kappa_i, density_i, mass_threshold, i, Dps, Ns, bins, pH0, sulfate_threshold, nitrate_threshold, org_threshold):
    
    included = ['NO3-', 'SO4--', 'ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 
                'LIM2', 'OC']
    others = ['Cl', 'NH4+', 'MSA', 'CO3--', 'Na', 
              'Ca', 'OIN', 'BC', 'IEPOX', 'dust', 'IEPOXOS', 'tetrol']
    
    incl_composition = np.zeros(len(included))
    other_composition = np.zeros(len(others))
    incl_mass = 0
    other_mass = 1.0

    volume_threshold = nitrate_threshold+org_threshold

    
    while incl_mass/(incl_mass+other_mass) < mass_threshold:
        incl_composition = np.zeros(len(included))
        other_composition = np.zeros(len(others))
        V_nit = 0
        V_sulf = 0
        V_org = 0
        Vtot = 0
        while V_nit < nitrate_threshold:
            Vi = nitrate_threshold*np.random.rand(1)
            Vi = Vi[0]
            species = 0
            incl_composition[int(species)] = incl_composition[int(species)] + Vi
            V_nit += Vi
            Vtot += Vi
        while V_sulf < sulfate_threshold:
            Vi = sulfate_threshold*np.random.rand(1)
            Vi = Vi[0]
            species = 1
            incl_composition[int(species)] = incl_composition[int(species)] + Vi
            V_sulf += Vi
            Vtot += Vi
        while V_org < org_threshold:
            Vi = org_threshold*np.random.rand(1)
            Vi = Vi[0]
            species = 2+np.floor((len(included)-2)*np.random.rand(1))
            incl_composition[int(species)] = incl_composition[int(species)] + Vi
            V_org += Vi
            Vtot += Vi
        while Vtot <= 1.0:
            Vi = (1-volume_threshold)*np.random.rand(1)
            Vi = Vi[0]
            species = np.floor(len(others)*np.random.rand(1))
            other_composition[int(species)] = other_composition[int(species)] + Vi
            Vtot += Vi
    
        if Vtot > 1.0:
            multiplier = 1.0/Vtot
            for j in range(0, len(incl_composition)):
                incl_composition[j] = multiplier*incl_composition[j]
            for j in range(0, len(other_composition)):
                other_composition[j] = multiplier*other_composition[j]
        
        incl_mass = 0
        other_mass = 0
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        for j in range(0, len(incl_composition)):
            Vi = incl_composition[j]*dry_volume
            incl_mass += Vi*density_i[included[j]]
        for j in range(0, len(other_composition)):
            Vi = other_composition[j]*dry_volume
            other_mass += Vi*density_i[others[j]]
        volume_threshold += 0.001
    
    name = 'kappa, ' + str(i+1)
    ki_Vi = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        Vi = incl_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi    
    for j in range(0, len(others)):
        aq_species = others[j]
        Vi = other_composition[j]*dry_volume
        ki_Vi += kappa_i[aq_species]*Vi   

    y0[indexing.getIndex(name, bins)] = ki_Vi/dry_volume      
    
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = incl_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
        Vi = other_composition[j]*dry_volume
        density = density_i[aq_species]
        mass = density*Vi
        y0[indexing.getIndex(name, bins)] = mass
    
    dry_mass = 0
    dry_volume = (4.0/3.0)*np.pi*(0.5*Dps[i])**3
    for j in range(0, len(included)):
        aq_species = included[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]
    for j in range(0, len(others)):
        aq_species = others[j]
        name = aq_species + ' (aq,' + str(i+1) + ')'
        dry_mass += y0[indexing.getIndex(name, bins)]

    name = 'density, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = dry_mass/dry_volume

    name = 'Ddry, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = np.log(Dps[i])
    name = 'pH, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = pH0
    name = 'N, ' + str(i+1)
    y0[indexing.getIndex(name, bins)] = Ns[i]
    
    return y0


def gas_phase(y0, start_time, end_time, bins):
    
    filename = 'HISCALE_data.txt'
    raw_data = np.loadtxt(filename, dtype='str')
    hiscale = {}
    units = {}
    hiscale[str(raw_data[0, 0])] = np.array(raw_data[2:, 0], dtype='float64')
    units[str(raw_data[0, 0])] = raw_data[1, 0]
    hiscale['N_Dp'] = np.array(raw_data[2:, 1:56], dtype = 'float64')
    units['N_Dp'] = 'nm'
    hiscale['Dp_lower'] = np.array([9.38, 10.66, 12.1, 13.74, 15.61, 17.73, 20.13, 22.86, 25.96, 29.48, 33.48, 38.02, 43.18, 49.04, 55.69, 63.25, 71.82, 81.57, 92.63, 105.2, 119.47, 135.67, 154.08, 174.98, 198.71, 225.67, 256.28, 291.04, 330.52, 375.35, 426.27, 484.09, 549.75, 624.33, 709.02, 805.19, 914.41, 1038.45, 1179.31, 1339.28, 1520.95, 1727.27, 1961.57, 2227.65, 2529.82, 2872.99, 3262.7, 3705.27, 4207.88, 4778.67, 5426.89, 6163.03, 6999.03, 7948.42, 9026.61])
    units['Dp_lower'] = 'nm'
    hiscale['Dp_upper'] = np.array([10.66, 12.1, 13.74, 15.61, 17.73, 20.13, 22.86, 25.96, 29.48, 33.48, 38.02, 43.18, 49.04, 55.69, 63.25, 71.82, 81.57, 92.63, 105.2, 119.47, 135.67, 154.08, 174.98, 198.71, 225.67, 256.28, 291.04, 330.52, 375.35, 426.27, 484.09, 549.75, 624.33, 709.02, 805.19, 914.41, 1038.45, 1179.31, 1339.28, 1520.95, 1727.27, 1961.57, 2227.65, 2529.82, 2872.99, 3262.7, 3705.27, 4207.88, 4778.67, 5426.89, 6163.03, 6999.03, 7948.42, 9026.61, 10251.04])
    units['Dp_upper'] = 'nm'
    hiscale['Dp_mid'] = np.array([10.00, 11.36, 12.89, 14.65, 16.64, 18.89, 21.45, 24.36, 27.66, 31.42, 35.68, 40.52, 46.02, 52.26, 59.35, 67.40, 76.54, 86.92, 98.72, 112.11, 127.31, 144.58, 164.20, 186.47, 211.76, 240.49, 273.11, 310.15, 352.22, 400.00, 454.26, 515.88, 585.85, 665.33, 755.58, 858.06, 974.46, 1106.64, 1256.75, 1427.23, 1620.83, 1840.70, 2090.38, 2373.93, 2695.95, 3061.65, 3476.95, 3948.59, 4484.20, 5092.48, 5783.26, 6567.74, 7458.63, 8470.38, 9619.36])
    units['Dp_mid'] = 'nm'
    for i in range(56, len(raw_data[0])): 
        hiscale[str(raw_data[0, i])] = np.array(raw_data[2:, i], dtype = 'float64')
        units[str(raw_data[0, i])] = str(raw_data[1, i])
    
    start_index = -1
    stop_index = -1
    
    for i in range(0, len(hiscale['N_Dp'])):
        if hiscale['Start_UTC'][i]/60 >= start_time and start_index == -1:
            start_index = i
        elif hiscale['Start_UTC'][i]/60 >= end_time and stop_index == -1:
            stop_index = i+1
    
    z = np.mean(hiscale['Radar_Alt'][start_index:stop_index])
    y0[indexing.getIndex('z', bins)] = z
    
    T = np.mean(hiscale['T_ambient'][start_index:stop_index])+273.15
    y0[indexing.getIndex('T', bins)] = T
    
    #V = np.mean(hiscale['Vert_Wind_Spd'][start_index:stop_index])
    
    SS = (np.mean(hiscale['RH_water'][start_index:stop_index])/100)-1.0
    y0[indexing.getIndex('SS', bins)] = SS
    
    P = np.mean(hiscale['P_ambient'][start_index:stop_index])*100
    y0[indexing.getIndex('P', bins)] = P
    
    filename = 'AGFL_atmosphere.txt'
    raw_data = np.loadtxt(filename, dtype='str')
    AGFL = {}
    AGFL[str(raw_data[0, 0])] = np.array(raw_data[2:, 0], dtype='float64')

    for i in range(0, len(raw_data[0])): 
        AGFL[str(raw_data[0, i])] = np.array(raw_data[2:, i], dtype = 'float64')
    
    for x in AGFL:
        name = x + ' (gas)'
        try:
            index = indexing.getIndex(name, bins)
            for i in range(0, len(AGFL['z'])-1):
                if z < 1000*AGFL['z'][i+1] and z >= 1000*AGFL['z'][i]:
                    slope = (AGFL[x][i+1]-AGFL[x][i])/((1000*AGFL['z'][i+1])-(1000*AGFL['z'][i]))
                    dx = (1000*AGFL['z'][i+1]) - (1000*AGFL['z'][i])
                    y = 1e-6*(AGFL[x][i] + (slope*dx)) # convert from ppm to mol A/mol air
            y0[index] = y
        except:
            index = 0
    
    return y0, SS, P, T