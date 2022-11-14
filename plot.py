#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:36:06 2022

@author: paytonbeeler
"""
import indexing, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as lines


def SS_Dps(filename, bins):
    
    file = open(filename, 'rb')
    data = pickle.load(file)
    
    plt.plot(data['SS']*100, data['z'], '-b')
    plt.xlabel('Supersaturation (%)', fontsize=16)
    plt.ylabel('Altitude (m)', fontsize=16)
    plt.ylim(np.min(data['z']), np.max(data['z']))
    plt.show()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.5*6.4, 1.2*4.8),constrained_layout=True, sharey=True)
    cmap = plt.get_cmap('viridis')
    for i in range(0, bins):
        color = cmap(data['N'][0, i]/np.max(data['N'][0, :]))
        ax1.plot(data['Dp'][:, i]*1e6, data['z'], '-', color = color)
        ax2.plot(data['Ddry'][:, i]*1e6, data['z'], '-', color = color)
    
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax1.set_xlabel(r'Wet Diameter ($\mu$m)', fontsize=16)
    ax2.set_xlabel(r'Dry Diameter ($\mu$m)', fontsize=16)
    ax1.set_ylabel('Altitude (m)', fontsize=16)
    x=np.empty(3)
    x[:]=-10
    y=np.empty(3)
    y[:]=-10
    c=np.array([[5,5,5],[5,5,5], [5,5,5]])
    im = ax1.pcolormesh(x, y, c, cmap=cmap, norm=None, vmin=0, vmax=np.max(data['N'][0, :]))
    cbar = fig.colorbar(im, ax=ax2, aspect=15, location='right')
    cbar.ax.set_title(r'N (cm$^{-3}$)'+'\n', fontsize=16)
    ax1.set_ylim(np.min(data['z']), np.max(data['z']))
    plt.show()
    
    return

def height_piecharts(zs, particles, soln, bins, t_end, start_height, stop_height):
    
    # set up figure
    fig = plt.figure(figsize=(12*6.4, 6*4.8), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig)
    axSS = fig.add_subplot(gs[:, 0])
    axDp = fig.add_subplot(gs[:, 1])
    
    for axis in ['top','bottom','left','right']:
        axSS.spines[axis].set_linewidth(6)
        axDp.spines[axis].set_linewidth(6)
    axSS.tick_params(which="major", axis='both', labelsize=50)
    axSS.tick_params(which="major", axis="both", direction="out", length=30, width=6, color="black")
    axDp.tick_params(which="major", axis='y', labelsize=0)
    axDp.tick_params(which="major", axis='x', labelsize=50)
    axDp.tick_params(which="major", axis="both", direction="out", length=30, width=6, color="black")
    

    # plot supersaturation and temperature
    axSS.plot(soln[:, indexing.getIndex('SS', bins)]*100, soln[:, indexing.getIndex('z', bins)], '-k', linewidth=10)
    axSS.hlines(zs[0], -100, 100, linewidth=6, color='k', linestyle='dashed')
    axSS.hlines(zs[1], -100, 100, linewidth=6, color='k', linestyle='dashed')
    axSS.hlines(zs[2], -100, 100, linewidth=6, color='k', linestyle='dashed')
    axSS.set_ylim(start_height, stop_height)
    axSS.set_xlim(np.floor(np.min(soln[:, indexing.getIndex('SS', bins)])*100), np.ceil(np.max(soln[:, indexing.getIndex('SS', bins)])*100)+1)
    axSS.set_xlabel('Supersaturation (%)', fontsize=70)
    axSS.set_ylabel('Altitude (m)', fontsize=70)
    axT = axSS.twiny()
    axT.tick_params(which="major", axis='x', direction="out", length=30, width=6, color="red")
    axT.tick_params(which="major", axis='x', colors = 'red', labelsize=50)
    axT.set_xlabel('Temperature (K)', fontsize=70, color='red')
    axT.spines['top'].set_linewidth(6)
    axT.spines['top'].set_color('red')
    axT.plot(soln[:, indexing.getIndex('T', bins)], soln[:, indexing.getIndex('z', bins)], '-r', linewidth=10)
    
    
    # plot wet diameters
    cmap = plt.get_cmap('viridis')
    for i in range(0, bins):
        name = 'Dp, ' + str(i+1)
        color = cmap(soln[0, 79+3*bins+i]/np.max(soln[0, 79+3*bins:79+3*bins+bins]))
        axDp.plot(soln[:, indexing.getIndex(name, bins)]*1e6, soln[:, indexing.getIndex('z', bins)], '-', linewidth=10, color = 'b')
        axDp.set_xscale('log')
    axDp.set_ylim(start_height, stop_height)
    axDp.set_xlabel(r'Wet Diameter ($\mu$m)', fontsize=70)
    axDp.hlines(zs[0], -100, 10000, linewidth=6, color='k', linestyle='dashed')
    axDp.hlines(zs[1], -100, 10000, linewidth=6, color='k', linestyle='dashed')
    axDp.hlines(zs[2], -100, 10000, linewidth=6, color='k', linestyle='dashed')
    axDp.set_xlim(1E-2, 1E2)    
    
    
    axDp.arrow(1.0, (zs[2]-start_height)/(stop_height-start_height), 0.3, 0.85-((zs[2]-start_height)/(stop_height-start_height)), transform=axDp.transAxes, linewidth=6, head_width=10*0.001, facecolor='k', clip_on = False)
    axDp.arrow(1.0, (zs[1]-start_height)/(stop_height-start_height), 0.3, 0.5-((zs[1]-start_height)/(stop_height-start_height)), transform=axDp.transAxes, linewidth=6, head_width=10*0.001, facecolor='k', clip_on = False)
    axDp.arrow(1.0, (zs[0]-start_height)/(stop_height-start_height), 0.3, 0.15-((zs[0]-start_height)/(stop_height-start_height)), transform=axDp.transAxes, linewidth=6, head_width=10*0.001, facecolor='k', clip_on = False)
    

    # make pie charts
    pie_size = 0.105
    
    partMC_aq_species = ['SO4--', 'NO3-', 'Cl', 'NH4+', 'MSA', 'ARO1', 'ARO2', 
             'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
             'CO3--', 'Na', 'Ca', 'OIN', 'OC', 'BC']
    
    cmap = plt.get_cmap('viridis')
    
    # top row
    place = zs[2]/stop_height
    
    axpie = fig.add_subplot(gs[0, 2], aspect=0.5)
    for axis in ['top', 'left', 'right', 'bottom']:
        axpie.spines[axis].set_linewidth(6)
    axpie.tick_params(which="major", axis='both', labelsize=0)
    
    colors = []
    diams = []
    max_diam = 0
    part = np.zeros((6, len(partMC_aq_species)))
    for i in range(0, len(partMC_aq_species)):
        colors.append(cmap(i/len(partMC_aq_species)))
    
    for i in range(0, 6):
        particle = int(particles[i])
        for j in range(0, len(partMC_aq_species)):
            aq = partMC_aq_species[j]
            name = aq + ' (aq,' + str(particle+1) + ')'
            part[i, j] = soln[int(place*t_end), indexing.getIndex(name, bins)]
        name = 'Dp, ' + str(particle+1)
        diam = soln[int(place*t_end), indexing.getIndex(name, bins)]
        #print(particle, name, diam)
        diams.append(diam)
        if diam > max_diam:
            max_diam = diam    
    max_diam = 1.1*max_diam
    
    #print(diams/max_diam)
    
    left, bottom, width, height = [0.68, 0.77, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[0], center=(0,0), colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[0]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68+0.5*pie_size, 0.77, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[1], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[1]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68+1.0*pie_size, 0.77, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[2], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[2]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68, 0.77-pie_size, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[3], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[3]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68+0.5*pie_size, 0.77-pie_size, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[4], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[4]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68+1.0*pie_size, 0.77-pie_size, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[5], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[5]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    
    # middle row
    place = zs[1]/stop_height
    
    axpie = fig.add_subplot(gs[1, 2], aspect=0.5)
    for axis in ['top', 'left', 'right', 'bottom']:
        axpie.spines[axis].set_linewidth(6)
    axpie.tick_params(which="major", axis='both', labelsize=0)
    
    colors = []
    diams = []
    max_diam = 0
    for i in range(0, len(partMC_aq_species)):
        colors.append(cmap(i/len(partMC_aq_species)))
    
    for i in range(0, 6):
        particle = int(particles[i])
        for j in range(0, len(partMC_aq_species)):
            aq = partMC_aq_species[j]
            name = aq + ' (aq,' + str(particle+1) + ')'
            part[i, j] = soln[int(place*t_end), indexing.getIndex(name, bins)]
        name = 'Dp, ' + str(particle+1)
        diam = soln[int(place*t_end), indexing.getIndex(name, bins)]
        diams.append(diam)
        if diam > max_diam:
            max_diam = diam    
    max_diam = 1.1*max_diam
    
    left, bottom, width, height = [0.68, 0.5, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[0], center=(0,0), colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[0]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    
    left, bottom, width, height = [0.68+0.5*pie_size, 0.5, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[1], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[1]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68+1.0*pie_size, 0.5, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[2], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[2]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68, 0.5-pie_size, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[3], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[3]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68+0.5*pie_size, 0.5-pie_size, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[4], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[4]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68+1.0*pie_size, 0.5-pie_size, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[5], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[5]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    
    
    # bottom row
    place = 0.0
    
    axpie = fig.add_subplot(gs[2, 2], aspect=0.5)
    for axis in ['top', 'left', 'right', 'bottom']:
        axpie.spines[axis].set_linewidth(6)
    axpie.tick_params(which="major", axis='both', labelsize=0)
    colors = []
    diams = []
    max_diam = 0
    for i in range(0, len(partMC_aq_species)):
        colors.append(cmap(i/len(partMC_aq_species)))
    
    for i in range(0, 6):
        particle = int(particles[i])
        for j in range(0, len(partMC_aq_species)):
            aq = partMC_aq_species[j]
            name = aq + ' (aq,' + str(particle+1) + ')'
            part[i, j] = soln[int(place*t_end), indexing.getIndex(name, bins)]
        name = 'Dp, ' + str(particle+1)
        diam = soln[int(place*t_end), indexing.getIndex(name, bins)]
        diams.append(diam)
        if diam > max_diam:
            max_diam = diam    
    max_diam = 1.1*max_diam
    
    left, bottom, width, height = [0.68, 0.235, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[0], center=(0,0), colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[0]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68+0.5*pie_size, 0.235, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[1], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[1]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68+1.0*pie_size, 0.235, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[2], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[2]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68, 0.235-pie_size, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[3], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[3]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68+0.5*pie_size, 0.235-pie_size, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[4], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[4]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    left, bottom, width, height = [0.68+1.0*pie_size, 0.235-pie_size, pie_size, pie_size]
    ax = fig.add_axes([left, bottom, width, height])
    ax.pie(part[5], colors=colors, wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}, radius=diams[5]/max_diam)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    
    #labels = partMC_aq_species
    #ax2.legend(labels, loc='right')
    fig.savefig(os.getcwd()+'/piechart.png')
    
    return