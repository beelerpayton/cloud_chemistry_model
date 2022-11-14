# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 09:53:30 2022

@author: beel083
"""
import numpy as np
import indexing
#from pyrcel import thermo
import matplotlib.pyplot as plt
from numba import njit
import sys

Kw = 1E-14
g = 9.8 # m/s
R = 8.314 # m^3*Pa/mol*K
Lv_H2O = 2.25e6 # latent heat of vaporization of water, Pa*m^3/kg*K
R_dry = 287.0 # Pa*m^3/kg*K
Cp = 1004.0 # specific heat of dry air at constant pressure
Na = 6.022E23

Mw = 0.018 # molar mass of water, kg/mol
Ma = 28.9/1000.0 # molar mass of dry air, kg/mol
rho_w = 1000.0 # density of water, kg/m^3


@njit
def parcel_properties(y, dydt, V, bins, thermal_accom, condens_coeff, constant_pressure=False):
                
    wv = y[4]
    T = y[1]
    P = y[2]
    SS = y[3]
    
    Ns = []
    dry_radii = []
    radii = []
    kappas = []
    Ntot = 0
    
    for i in range(0, bins):
        dry_radii.append(np.exp(y[79+0*bins+i])/2.0)
        radii.append(np.exp(y[79+1*bins+i])/2.0)
        Ns.append(y[79+3*bins+i])
        kappas.append(y[79+2*bins+i])
        Ntot += y[79+3*bins+i]

           
    dz_dt = V
    Tv = T*(1 + 0.61*wv)
    dP_dt = -((g*P)/(R_dry*Tv))*dz_dt
    
    # saturation vapor pressure, Pa
    Pv_sat = 611.2*np.exp((17.67*(T-273.15))/(T-273.15+243.5))
    
    ## Non-continuum diffusivity/thermal conductivity of air near
    ## near particle
    P_atm = P/101325
    Dv = np.power(10.0, -4)*(0.211/P_atm)*np.power(T/273.15, 1.94)
    
    ka = np.power(10.0, -3)*(4.39 + 0.071*T)
    
    if constant_pressure==True or Ntot==0:
        dz_dt = 0
        dT_dt = 0
        dP_dt = 0
        dSS_dt = 0
        dwv_dt = 0
        dwc_dt = 0
        dr_dts = np.zeros(len(dry_radii))
        Tv = T*(1+0.61*wv)
        rho_air = P/(R_dry*Tv)
        
    else:
        Tv = T*(1+0.61*wv)
        rho_air = P/(R_dry*Tv)
        
        dr_dts=[np.float64(x) for x in range(0)]
        dwc_dts=[np.float64(x) for x in range(0)]
        dwc_dt = 0
        i = 0
        
        for N, radius, dry_radius, kappa in zip(Ns, radii, dry_radii, kappas):
            
            ka_r = ka/(1+(ka/(thermal_accom*radius*rho_air*Cp))*np.sqrt((2*np.pi*Mw)/(R*T)))
            Dv_r = Dv/(1+(Dv/(condens_coeff*radius))*np.sqrt((2*np.pi*Mw)/(R*T)))
        
            G_a = (rho_w*R*T)/(Pv_sat*Dv_r*Mw)
            G_b = (Lv_H2O*rho_w*((Lv_H2O*Mw/(R*T))-1.0))/(ka_r*T)
            G = 1.0 / (G_a + G_b)
            
            aw_min = np.power(1.0+kappa*(np.power(dry_radius,3)/(np.power((1+1e-15)*dry_radius, 3)-np.power(dry_radius, 3))), -1)
            sigma_water = 0.0761-1.55E-4*(T-273.15) # J/m^2
            Seq_min = aw_min*np.exp((2.0*sigma_water*Mw)/(R*T*rho_w*(1+1e-15)*dry_radius)) - 1.0
            
            if SS >= Seq_min:
                if radius == dry_radius:
                    radius = (1.0+1e-15)*dry_radius
                    a_w = np.power(1.0+kappa*(np.power(dry_radius,3)/(np.power(radius, 3)-np.power(dry_radius, 3))), -1)
                    sigma_water = 0.0761-1.55E-4*(T-273.15) # J/m^2
                    Seq = a_w*np.exp((2.0*sigma_water*Mw)/(R*T*rho_w*radius)) - 1.0
                    dr_dt = (G / radius) * (SS-Seq) # m/s
                else:
                    a_w = np.power(1.0+kappa*(np.power(dry_radius,3)/(np.power(radius, 3)-np.power(dry_radius, 3))), -1)
                    sigma_water = 0.0761-1.55E-4*(T-273.15) # J/m^2
                    Seq = a_w*np.exp((2.0*sigma_water*Mw)/(R*T*rho_w*radius)) - 1.0
                    dr_dt = (G / radius) * (SS-Seq) # m/s
            else:
                dr_dt = 0

            dV_dt = 4.0*np.pi*(radius**2)*dr_dt
            dmH2O = dV_dt*rho_w
            #print(dmH2O)
            dydt[79+55*bins+i] = dmH2O
            i += 1

            if radius+dr_dt <= dry_radius:
                dr_dt = 0
            
            dwc_dts.append(((4.0*np.pi*rho_w)/rho_air)*N*np.power(radius, 2)*dr_dt)
            dwc_dt += ((4.0*np.pi*rho_w)/rho_air)*N*np.power(radius, 2)*dr_dt
            dr_dts.append(dr_dt)
        
        #print(' ')
        dwv_dt = -dwc_dt
        dT_dt = ((-g/Cp)*dz_dt)-((Lv_H2O/Cp)*dwv_dt)
        alpha = ((g*Mw*Lv_H2O)/(Cp*R*np.power(T, 2)))-((g*Mw)/(R*T))
        gamma = ((P*Ma)/(Pv_sat*Mw))+((Mw*np.power(Lv_H2O, 2))/(Cp*R*np.power(T, 2)))
        dSS_dt = alpha*(dz_dt)-gamma*(dwc_dt)
        
        dydt[0] = dz_dt
        dydt[1] = dT_dt
        dydt[2] = dP_dt
        dydt[3] = dSS_dt
        dydt[4] = dwv_dt
        dydt[5] = dwc_dt
        
    return dydt

@njit
def particle_changes(y, dydt, bins, co_condens):
        
    partMC_kappas = {'SO4--': 0.65, 
                     'NO3-': 0.65, 
                     'Cl': 0.53, 
                     'NH4+': 0.65, 
                     'MSA': 0.53, 
                     'ARO1': 0.1, 
                     'ARO2': 0.1, 
                     'ALK1': 0.1, 
                     'OLE1': 0.1, 
                     'API1': 0.1,
                     'API2': 0.1, 
                     'LIM1': 0.1, 
                     'LIM2': 0.1, 
                     'CO3--': 0.53, 
                     'Na': 0.53, 
                     'Ca': 0.53, 
                     'OIN': 0.1, 
                     'OC': 0.001, 
                     'BC': 1.0E-5, 
                     'H2O': 1.0E-5}
    
    #kg/m^3
    partMC_densities = {'SO4--': 1800., 
                     'NO3-': 1800., 
                     'Cl': 2200., 
                     'NH4+': 1800., 
                     'MSA': 1800., 
                     'ARO1': 1400., 
                     'ARO2': 1400., 
                     'ALK1': 1400., 
                     'OLE1': 1400., 
                     'API1': 1400.,
                     'API2': 1400., 
                     'LIM1': 1400., 
                     'LIM2': 1400., 
                     'CO3--': 2600., 
                     'Na': 2200., 
                     'Ca': 2600., 
                     'OIN': 2600., 
                     'OC': 1000., 
                     'BC': 1800., 
                     'H2O': 1000.}
    
    for i in range(0, bins):
        
        dVdry_dt = 0
        
        if co_condens == True:
            
            name = 'SO4--'
            index = 79+31*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'NO3-'
            index = 79+33*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'Cl'
            index = 79+40*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'NH4+'
            index = 79+35*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'MSA'
            index = 79+43*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'ARO1'
            index = 79+44*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'ARO2'
            index = 79+45*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'ALK1'
            index = 79+46*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'OLE1'
            index = 79+47*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'API1'
            index = 79+48*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'API2'
            index = 79+49*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'LIM1'
            index = 79+50*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'LIM2'
            index = 79+51*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'CO3--'
            index = 79+37*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'Na'
            index = 79+41*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'Ca' 
            index = 79+42*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'OIN'
            index = 79+52*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'OC'
            index = 79+53*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
            name = 'BC' 
            index = 79+54*bins+i
            density = partMC_densities[name]
            dVi_dt = dydt[index]/density
            dVdry_dt += dVi_dt
            
        name = 'H2O' 
        index = 79+55*bins+i
        density = partMC_densities[name]
        dVi_dt = dydt[index]/density
        dVtot_dt = dVdry_dt + dVi_dt
        
        radius = np.exp(y[79+1*bins+i])/2.0
        dry_radius = np.exp(y[79+0*bins+i])/2.0
        
        dydt[79+0*bins+i] = (2.0*(dVdry_dt/(4.0*np.pi*dry_radius**2)))/dry_radius
        dydt[79+1*bins+i] = (2.0*(dVtot_dt/(4.0*np.pi*radius**2)))/radius
 
        if co_condens == True:
            
            Vdry = (4.0/3.0)*np.pi*dry_radius**3
            Vtot = (4.0/3.0)*np.pi*radius**3
            summation = 0
            dm_dt = 0
            mass = 0
            
            name = 'SO4--'
            index = 79+31*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'NO3-'
            index = 79+33*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'Cl'
            index = 79+40*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'NH4+'
            index = 79+35*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'MSA'
            index = 79+43*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'ARO1'
            index = 79+44*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'ARO2'
            index = 79+45*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'ALK1'
            index = 79+46*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'OLE1'
            index = 79+47*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'API1'
            index = 79+48*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'API2'
            index = 79+49*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'LIM1'
            index = 79+50*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'LIM2'
            index = 79+51*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'CO3--'
            index = 79+37*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'Na'
            index = 79+41*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'Ca' 
            index = 79+42*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'OIN'
            index = 79+52*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'OC'
            index = 79+53*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            name = 'BC' 
            index = 79+54*bins+i
            density = partMC_densities[name]
            kappa = partMC_kappas[name]
            dVi_dt = dydt[index]/density
            Vi = y[index]/density
            summation += kappa*((Vdry*dVi_dt)-(Vi*dVdry_dt))
            dm_dt += dydt[index]
            mass += y[index]
            
            #name = 'H2O' 
            #index = 79+55*bins+i
            #density = partMC_densities[name]
            #kappa = partMC_kappas[name]
            #dVi_dt = dydt[index]/density
            #Vi = y[index]/density
            #summation += kappa*((Vtot*dVi_dt)-(Vi*dVtot_dt))
            
        
            dydt[79+2*bins+i] = (1/(Vdry**2))*summation
            dydt[79+4*bins+i] = (1/(Vdry**2))*((Vdry*dm_dt)-(mass*dVdry_dt))
            #print(dydt[79+1*bins+i], dydt[79+0*bins+i])
      
    
    return dydt

