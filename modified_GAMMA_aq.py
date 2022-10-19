# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:32:03 2022

@author: beel083
"""
#from pyrcel import constants as c
from pyrcel import thermo
import numpy as np
from scipy.optimize import fsolve
import run, indexing, rates
from numba import njit

R = 8.314 # m^3*Pa/mol*K
R_dry = 287.0 # Pa*m^3/kg*K
Mw = 18.015/1000.0 # molar mass of water, kg/mol
Ma = 28.9/1000.0 # molar mass of dry air, kg/mol
rho_w = 1000.0 # density of water, kg/m^3
kb = 1.380649e-23 #m^2 kg/s^2 K

def equilibrate_h2o(dry_diameters, kappas, Ns, SS, P, T):

    es = 611.2*np.exp((17.67*(T-273.15))/(T-273.15+243.5))
    mixing_ratio = (SS + 1.0) * (0.622 * es / (P - es)) # kg water vapor per kg air
    Tv = T*(1+0.61*mixing_ratio)
    rho_air = P/(R_dry*Tv)
    sigma_water = 0.0761-1.55E-4*(T-273.15) # J/m^2
    mass_water = lambda radius, dry_radius, Ni: (4.0*np.pi/3.0)*rho_w*Ni*(radius**3-dry_radius**3)
    
    diameters = []
    mH2O = []
    
    for D_dry, kappa, N in zip(dry_diameters, kappas, Ns):
       
        r_dry = D_dry/2.0
        #a_w = lambda r: np.power(1.0+kappa*(np.power(r_dry/r, 3)), -1)
        #Seq = lambda r: a_w(r)*np.exp((2.0*sigma_water*Mw)/(R*T*rho_w*r))
        #f = lambda r: Seq(r) - SS
        f = lambda r: thermo.Seq(r, r_dry, T, kappa) - SS
       
        r = fsolve(f, r_dry)
        if r[0] < r_dry:
            r[0] = r_dry
        mass_w = mass_water(r, r_dry, N)
        diameters.append(2.0*r[0])
        mH2O.append(mass_w)
    
    liquid_water_content = np.sum(mH2O)     # kg liqiud water per m^3
    liquid_water_content /= rho_air # kg liqiud water per kg air

    return np.array(diameters), mixing_ratio, liquid_water_content


@njit
def particle_phase(y, dydt, bins, Kelvin_effect = True, constant_pH = False, aq_oxidation = True):

    
    T = y[1]
    P = y[2]
     
    # taken from LERICHE ET AL.: MODELING STUDY OF STRONG ACIDS FORMATION
    alphas = {'O3': 0.05,
              'O2': 0.05,
              'HOOH': 0.11,
              'HO2': 0.01,
              'OH': 0.05,
              'NO': 0.0001,
              'NO2': 0.0015,
              'NO3': 0.0002,
              'N2O5': 0.0037,
              'HONO': 0.05,
              'HNO3': 0.054,
              'HNO4': 0.05,
              'NH3': 0.04,
              'SO2': 0.11,
              'H2SO4': 0.07,
              'CO2': 0.0002,
              'CH3OO': 0.05,
              'FORM': 0.02,
              'FORMIC': 0.012,
              'CH3O2H': 0.0038,
              'CH3OH': 0.015}
    
    # these units need to be in kg/mol
    molec_masses = {'O3': 0.048,
                    'O2': 0.032,
                    'HOOH': 0.034,
                    'HO2': 0.033,
                    'OH': 0.017,
                    'NO': 0.030,
                    'NO2': 0.046,
                    'NO3': 0.062,
                    'N2O5': 0.108,
                    'HONO': 0.047,
                    'HNO3': 0.063,
                    'HNO4': 0.079,
                    'NH3': 0.017,
                    'SO2': 0.064,
                    'H2SO4': 0.098,
                    'CO2': 0.044,
                    'CH3OO': 0.047,
                    'FORM': 0.030,
                    'FORMIC': 0.046,
                    'CH3O2H': 0.048,
                    'CH3OH': 0.032,
                    'HO2-': 0.033,
                    'HSO3-': 0.081,
                    'SO3--': 0.080,
                    'HSO4-': 0.097,
                    'SO4--': 0.096,
                    'O2-': 0.032,
                    'NO3-': 0.062,
                    'NO4-': 0.078,
                    'NH4+': 0.018,
                    'HCO3-': 0.061,
                    'CO3--': 0.060,
                    'NO2-': 0.046,
                    'HCOO-': 0.045,
                    'Cl': 0.03545,
                    'Na': 0.023,
                    'Ca': 0.040,
                    'MSA': 0.095,
                    'ARO1': 0.15,
                    'ARO2': 0.15,
                    'ALK1': 0.14,
                    'OLE1': 0.14,
                    'API1': 0.184,
                    'API2': 0.184,
                    'LIM1': 0.2,
                    'LIM2': 0.2,
                    'OIN': 0.001,
                    'OC': 0.001,
                    'BC': 0.001,
                    'H2O': 0,
                    'H2C(OH)2': 0.060}
    
    air_changes = np.zeros(len(y))
    
    for i in range(0, bins):

        pH = y[79+5*bins+i]
        Hplus = np.power(10, -pH) 
        radius = y[79+1*bins+i]/2.0
        dry_radius = y[79+0*bins+i]/2.0
        N = y[79+3*bins+i]

        # taken from S&P (primary) or LERICHE ET AL.: MODELING STUDY OF STRONG ACIDS FORMATION
        Henry_constants = {'O3': 1.1e-2*np.exp((-21087/8.314)*((1/298)-(1/T))),
                  'O2': 1.3e-3,
                  'HOOH': 1.0e5*np.exp((-60668/8.314)*((1/298)-(1/T))),
                  'HO2': 5.7e3,
                  'OH': 25.0,
                  'NO': 1.9e-3*np.exp((-13133/8.314)*((1/298)-(1/T))),
                  'NO2': 1.0e-2*np.exp((-20920/8.314)*((1/298)-(1/T))),
                  'NO3': 1.8,
                  'N2O5': 2.1*np.exp(3400*((1/298)-(1/T))), # from other source
                  'HONO': 49.0*np.exp((-39748/8.314)*((1/298)-(1/T))),
                  'HNO3': 2.1e5,
                  'HNO4': 1.2E4*np.exp(6900*((1/298)-(1/T))), # from other source
                  'NH3': 62.0*np.exp((-34183/8.314)*((1/298)-(1/T))),
                  'SO2': 1.23*np.exp((-26150/8.314)*((1/298)-(1/T))),
                  'H2SO4': 2.1E5*np.exp(-8700*((1/298)-(1/T))), # from other source
                  'CO2': 3.4e-2*np.exp((-20292/8.314)*((1/298)-(1/T))),
                  'CH3OO': 6.0*np.exp((-46442/8.314)*((1/298)-(1/T))),
                  'FORM': 2.5*np.exp((-53555/8.314)*((1/298)-(1/T))),
                  'FORMIC': 3.6e3*np.exp((-47697/8.314)*((1/298)-(1/T))),
                  'CH3O2H': 310.0*np.exp((-46442/8.314)*((1/298)-(1/T))),
                  'CH3OH': 220.0*np.exp((-40585/8.314)*((1/298)-(1/T)))}
                              
        
        ka = rates.simple_aqueous()
        V_water = (4.0/3.0)*np.pi*(radius**3-dry_radius**3)
    
        if Kelvin_effect==True:
            
            Tv = T*(1+0.61*y[4])
            rho_air = P/(R_dry*Tv)
            sigma_H2O = 0.0761-1.55E-4*(T-273.15) # surface tension of water, J/m^2
            P_curved = P - ((sigma_H2O*rho_air)/(rho_w-rho_air)) * (1/radius + 1/radius) # Pa
        else:
            P_curved = P
        
        #O3
        name = 'O3'
        index = 79+6*bins+i
        Pi = P_curved*y[42]
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name] # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        reactions = 0
        dydt[index] = molec_masses[name]*V_water*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)) + (1000*reactions))
        air_changes[index] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        #O2
        name = 'O2'
        index = 79+7*bins+i
        Pi = P_curved*0.21
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name] # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dydt[index] = molec_masses[name]*V_water*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        # total H2O2
        name = 'HOOH'
        Pi = P_curved*y[18]
        if V_water == 0:
            Ci = 0
        else:
            Ci = (y[79+8*bins+i]/(V_water*molec_masses['HOOH'])) + (y[79+27*bins+i]/(V_water*molec_masses['HO2-'])) + (y[79+32*bins+i]/(V_water*molec_masses['O2-']))
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        K1 = 2.2E-12*np.exp(-3730*((1/T)-(1/298))) #M, needs to match the unit of Hplus
        H = (1000/101325)*Henry_constants[name]*(1+(K1/Hplus)) # mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dHOOH_T = ((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T))
        air_changes[79+8*bins+i] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        # H2O2
        index = 79+8*bins+i
        dHOOH = dHOOH_T/(1+(K1/Hplus))
        dydt[index] = molec_masses['HOOH']*V_water*(dHOOH)
        
        # HO2-
        index = 79+27*bins+i
        dHO2rad = (dHOOH*K1)/Hplus
        dydt[index] = molec_masses['HO2-']*V_water*(dHO2rad)
        

        
        #total HO2
        name = 'HO2'
        Pi = P_curved*y[8]
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        K1 = 3.5e-5 #M, needs to match the unit of Hplus
        H = (1000/101325)*Henry_constants[name]*(1+(K1/Hplus)) # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dHO2_T = ((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T))
        air_changes[79+9*bins+i] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))

        # HO2
        index = 79+9*bins+i
        dHO2 = dHO2_T/(1+(K1/Hplus))
        dydt[index] = molec_masses['HO2']*V_water*(dHOOH)        

        # O2-
        index = 79+32*bins+i
        dO2rad = (dHO2*K1)/Hplus
        dydt[index] = molec_masses['O2-']*V_water*(dO2rad)
        
        # OH
        name = 'OH'
        index = 79+10*bins+i
        Pi = P_curved*y[19]
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name] # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dydt[index] = molec_masses[name]*V_water*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        air_changes[index] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        #NO
        name = 'NO'
        index = 79+11*bins+i
        Pi = P_curved*y[35]
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name] # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dydt[index] = molec_masses[name]*V_water*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        air_changes[index] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        #NO2
        name = 'NO2'
        index = 79+12*bins+i
        Pi = P_curved*y[36]
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name] # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dydt[index] = molec_masses[name]*V_water*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        air_changes[index] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        #NO3
        name = 'NO3'
        index = 79+13*bins+i
        Pi = P_curved*y[37]
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name] # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dydt[index] = molec_masses[name]*V_water*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        air_changes[index] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        #N2O5
        name = 'N2O5'
        index = 79+14*bins+i
        Pi = P_curved*y[38]
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name] # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dydt[index] = molec_masses[name]*V_water*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        air_changes[index] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        # total HONO
        name = 'HONO'
        Pi = P_curved*y[40]
        if V_water == 0:
            Ci = 0
        else:
            Ci = (y[79+15*bins+i]/(V_water*molec_masses['HONO'])) + (y[79+38*bins+i]/(V_water*molec_masses['NO2-']))
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        K1 = 5.1E-4*np.exp(-1260*((1/T)-(1/298))) #M, needs to match the unit of Hplus
        H = (1000/101325)*Henry_constants[name]*(1+(K1/Hplus)) # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dHONO_T = ((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T))
        air_changes[79+15*bins+i] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        #HONO
        index = 79+15*bins+i
        dHONO = dHONO_T/(1+(K1*Hplus))
        dydt[index] = molec_masses['HONO']*V_water*(dHONO)
        
        #NO2-
        index = 79+38*bins+i
        dNO2 = (dHONO*K1)/Hplus
        dydt[index] = molec_masses['NO2-']*V_water*(dNO2)
        
        #total HNO3
        name = 'HNO3'
        Pi = P_curved*y[39]
        if V_water == 0:
            Ci = 0
        else:
            Ci = (y[79+16*bins+i]/(V_water*molec_masses['HNO3'])) + (y[79+33*bins+i]/(V_water*molec_masses['NO3-']))
        K1 = 15.4*np.exp(8700*((1/T)-(1/298))) #M, needs to match the unit of Hplus
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name]*(1+(K1/Hplus)) # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dHNO3_T = ((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T))
        air_changes[79+16*bins+i] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        #HNO3
        index = 79+16*bins+i
        dHNO3 = dHNO3_T/(1+(K1/Hplus))
        dydt[index] = molec_masses['HNO3']*V_water*(dHNO3)
        
        #NO3-
        index = 79+33*bins+i
        dNO3 = (dHNO3*K1)/Hplus
        dydt[index] = molec_masses['NO3-']*V_water*(dNO3)
        
        #total HNO4
        name = 'HNO4'
        Pi = P_curved*y[41]
        if V_water == 0:
            Ci = 0
        else:
            Ci = (y[79+17*bins+i]/(V_water*molec_masses['HNO4'])) + (y[79+34*bins+i]/(V_water*molec_masses['NO4-']))
        K1 = 1.26E-6 #M, needs to match the unit of Hplus (from French paper)
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name]*(1+(K1/Hplus)) # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dHNO4_T = ((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T))
        air_changes[79+17*bins+i] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        #HNO4
        index = 79+17*bins+i
        dHNO4 = dHNO4_T/(1+(K1*Hplus))
        dydt[index] = molec_masses['HNO4']*V_water*(dHNO4)
        
        #NO4-
        index = 79+34*bins+i
        dNO4 = (dHNO4*K1)/Hplus
        dydt[index] = molec_masses['NO4-']*V_water*(dNO4)
        
        # total NH3
        Pi = P_curved*y[78]
        if V_water == 0:
            Ci = 0
        else:
            Ci = (y[79+18*bins+i]/(V_water*molec_masses['NH3'])) + (y[79+35*bins+i]/(V_water*molec_masses['NH4+']))
        w = np.sqrt((8*R*T)/(np.pi*molec_masses['NH3'])) # thermal velocity, m/s
        K1 = 1.7E-5*np.exp(-450*((1/T)-(1/298))) #M, needs to match the unit of Hplus (from French paper)
        Kw = 1.0E-14*np.exp(-6710*((1/T)-(1/298))) # M
        OH_conc = Kw/Hplus # M
        H = (1000/101325)*Henry_constants['NH3']*(1+(K1/OH_conc))
        Dg = (1/100**2)*1.9*np.power(molec_masses['NH3'], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas['NH3'])), -1.0) # mass uptake, 1/s
        dNH3_T = ((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T))
        air_changes[79+18*bins+i] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        # NH3
        index = 79+18*bins+i
        dNH3 = dNH3_T/((K1/OH_conc))
        dydt[index] = molec_masses['NH3']*V_water*(dNH3)
        
        # NH4+
        index = 79+35*bins+i
        dNH4 = (dNH3*K1)/OH_conc
        dydt[index] = molec_masses['NH4+']*V_water*(dNH4)
        
        # S4 from gas
        Pi = P_curved*y[29]
        if V_water == 0:
            Ci = 0
        else:
            Ci = (y[79+19*bins+i]/(V_water*molec_masses['SO2'])) + (y[79+28*bins+i]/(V_water*molec_masses['HSO3-'])) + (y[79+29*bins+i]/(V_water*molec_masses['SO3--']))
        w = np.sqrt((8*R*T)/(np.pi*molec_masses['SO2'])) # thermal velocity, m/s
        K1 = 1.3E-2*np.exp(1960*((1/T)-(1/298))) #M, needs to match the unit of Hplus
        K2 = 6.6E-8*np.exp(1500*((1/T)-(1/298))) #M
        H = (1000/101325)*Henry_constants['SO2']*(1+(K1/Hplus)+((K1*K2)/(Hplus*Hplus)))
        Dg = (1/100**2)*1.9*np.power(molec_masses['SO2'], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas['SO2'])), -1.0) # mass uptake, 1/s
        SO2_conc = 0.001*(y[79+19*bins+i]/(V_water*molec_masses['SO2'])) # M
        HSO3_conc = 0.001*(y[79+28*bins+i]/(V_water*molec_masses['HSO3-'])) # M
        SO3_conc = 0.001*(y[79+29*bins+i]/(V_water*molec_masses['SO3--'])) # M
        H2O2_conc = 0.001*(y[79+8*bins+i]/(V_water*molec_masses['HOOH'])) # M
        O3_conc = 0.001*(y[79+6*bins+i]/(V_water*molec_masses['O3'])) # M
        NO2_conc = 0.001*(y[79+12*bins+i]/(V_water*molec_masses['NO2'])) # M
        if aq_oxidation == True:
            S6_production_rate = (ka[0]*SO2_conc + ka[1]*HSO3_conc + ka[2]*SO3_conc)*O3_conc\
                + (ka[4]*Hplus*HSO3_conc*H2O2_conc)/(1 + ka[5]*Hplus)\
                + ka[7]*NO2_conc*0.001*Ci #M/s
        else:
            S6_production_rate = 0
            
        dS4_T = ((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)) - (1000*S6_production_rate)
        air_changes[79+19*bins+i] = N*V_water*((R*T)/P)*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        # SO2
        index = 79+19*bins+i
        dSO2 = dS4_T/(1+(K1/Hplus)+((K1*K2)/(Hplus*Hplus)))
        dydt[index] = molec_masses['SO2']*V_water*(dSO2)
        
        # HSO3-
        index = 79+28*bins+i
        dHSO3 = (dSO2*K1)/Hplus
        dydt[index] = molec_masses['HSO3-']*V_water*(dHSO3)
        
        # SO3--
        index = 79+29*bins+i
        dSO3 = (dHSO3*K2)/Hplus
        dydt[index] = molec_masses['SO3--']*V_water*(dSO3)
        
        # S6 from gas
        Pi = P_curved*y[77]
        if V_water == 0:
            Ci = 0
        else:
            Ci = (y[79+20*bins+i]/(V_water*molec_masses['H2SO4'])) + (y[79+30*bins+i]/(V_water*molec_masses['HSO4-'])) + (y[79+31*bins+i]/(V_water*molec_masses['SO4--']))
        w = np.sqrt((8*R*T)/(np.pi*molec_masses['H2SO4'])) # thermal velocity, m/s
        K1 = 1000
        K2 = 1.02E-2*np.exp(2720*((1/T)-(1/298))) #M
        H = (1000/101325)*Henry_constants['H2SO4']*(1+(K1/Hplus)+((K1*K2)/(Hplus*Hplus)))
        Dg = (1/100**2)*1.9*np.power(molec_masses['H2SO4'], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas['H2SO4'])), -1.0) # mass uptake, 1/s
        dS6_T = ((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)) + (1000*S6_production_rate)
        air_changes[79+20*bins+i] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        # H2SO4
        index = 79+20*bins+i
        dH2SO4 = dS6_T/(1+(K1/Hplus)+((K1*K2)/(Hplus*Hplus)))
        dydt[index] = molec_masses['H2SO4']*V_water*(dH2SO4)
        
        # HSO4-
        index = 79+30*bins+i
        dHSO4 = (dH2SO4*K1)/Hplus
        dydt[index] = molec_masses['HSO4-']*V_water*(dHSO4)
        
        # SO4--
        index = 79+31*bins+i
        dSO4 = (dHSO4*K2)/Hplus
        dydt[index] = molec_masses['SO4--']*V_water*(dSO4)
        
        # total CO2
        Pi = P_curved*0 #3.5659e-04
        if V_water == 0:
            Ci = 0
        else:
            Ci = (y[79+21*bins+i]/(V_water*molec_masses['CO2'])) + (y[79+36*bins+i]/(V_water*molec_masses['HCO3-'])) + (y[79+37*bins+i]/(V_water*molec_masses['CO3--']))
        w = np.sqrt((8*R*T)/(np.pi*molec_masses['CO2'])) # thermal velocity, m/s
        K1 = 4.3E-7*np.exp(-1000*((1/T)-(1/298))) #M, needs to match the unit of Hplus
        K2 = 4.68E-11*np.exp(-1760*((1/T)-(1/298))) #M
        H = (1000/101325)*Henry_constants['CO2']*(1+(K1/Hplus)+((K1*K2)/(Hplus*Hplus)))
        Dg = (1/100**2)*1.9*np.power(molec_masses['CO2'], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dCO2_T = ((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T))
        
        # CO2
        index = 79+21*bins+i
        dCO2 = dCO2_T/(1+(K1/Hplus)+((K1*K2)/(Hplus*Hplus)))
        dydt[index] = molec_masses['CO2']*V_water*(dCO2)
        
        # HCO3-
        index = 79+36*bins+i
        dHCO3 = (dCO2*K1)/Hplus
        dydt[index] = molec_masses['HCO3-']*V_water*(dHCO3)
        
        # CO3--
        index = 79+37*bins+i
        dCO3 = (dHCO3*K2)/Hplus
        dydt[index] = molec_masses['CO3--']*V_water*(dCO3)
        
        #CH3OO
        name = 'CH3OO'
        index = 79+22*bins+i
        Pi = P_curved*y[33]
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name] # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dydt[index] = molec_masses[name]*V_water*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        air_changes[index] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        #total FORM
        name = 'FORM'
        Pi = P_curved*y[20]
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        K1 = 2.53e3*np.exp(4020*((1/T)-(1/298))) #M, needs to match the unit of Hplus
        H = (1000/101325)*Henry_constants[name]*(1+(K1/Hplus)) # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dFORM_T = ((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T))
        air_changes[79+23*bins+i] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        # FORM
        index = 79+23*bins+i
        dFORM = dFORM_T/(1+(K1/Hplus))
        dydt[index] = molec_masses['FORM']*V_water*(dFORM)
        
        #H2C(OH)2
        index = 79+56*bins+i
        dH2COH2 = (dFORM*K1)/Hplus
        dydt[index] = molec_masses['H2C(OH)2']*V_water*(dH2COH2)
        
        #FORMIC
        Pi = P_curved*y[21]
        if V_water == 0:
            Ci = 0
        else:
            Ci = (y[79+24*bins+i]/(V_water*molec_masses['FORMIC'])) + (y[79+39*bins+i]/(V_water*molec_masses['HCOO-']))
        w = np.sqrt((8*R*T)/(np.pi*molec_masses['FORMIC'])) # thermal velocity, m/s
        K1 = 1.8E-4*np.exp(-20*((1/T)-(1/298))) #M, needs to match the unit of Hplus
        H = (1000/101325)*Henry_constants['FORMIC']*(1+(K1/Hplus))
        Dg = (1/100**2)*1.9*np.power(molec_masses['FORMIC'], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas['FORMIC'])), -1.0) # mass uptake, 1/s
        dFORMIC_T = ((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T))
        air_changes[79+24*bins+i] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        # FORMIC
        index = 79+24*bins+i
        dFORMIC = dFORMIC_T/(1+(K1/Hplus))
        dydt[index] = molec_masses['FORMIC']*V_water*(dFORMIC)
        
        #HCOO-
        index = 79+39*bins+i
        dHCOO = (dFORMIC*K1)/Hplus
        dydt[index] = molec_masses['HCOO-']*V_water*(dHCOO)
        
        #CH3O2H
        name = 'CH3O2H'
        index = 79+25*bins+i
        Pi = P_curved*y[53]
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name] # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dydt[index] = molec_masses[name]*V_water*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        air_changes[index] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        #CH3OH
        name = 'CH3OH'
        index = 79+26*bins+i
        Pi = P_curved*y[52]
        if V_water == 0:
            Ci = 0
        else:
            Ci = y[index]/(V_water*molec_masses[name])
        w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
        H = (1000/101325)*Henry_constants[name] # convert from M/atm to mol/m^3*Pa
        Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
        kmt = np.power((radius**2/(3*Dg))+((4.0*radius)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
        dydt[index] = molec_masses[name]*V_water*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        air_changes[index] = N*V_water*((R*T)/P)*(((kmt*Pi)/(R*T)) - ((kmt*Ci)/(H*R*T)))
        
        
        if constant_pH == False and V_water > 0:
            name = 'pH, ' + str(i+1)
            index = 79+5*bins+i
            '''
            reactions = (dydt[79+28*bins+i]/(V_water*molec_masses['HSO3-']))\
                + ((2.0*dydt[79+29*bins+i])/(V_water*molec_masses['SO3--']))\
                + (dydt[79+30*bins+i]/(V_water*molec_masses['HSO4-']))\
                + ((2.0*dydt[79+31*bins+i])/(V_water*molec_masses['SO4--']))\
                + (dydt[79+27*bins+i]/(V_water*molec_masses['HO2-']))\
                + ((2.0*dydt[79+32*bins+i])/(V_water*molec_masses['O2-']))\
                + (dydt[79+33*bins+i]/(V_water*molec_masses['NO3-']))\
                + (dydt[79+34*bins+i]/(V_water*molec_masses['NO4-']))\
                - (dydt[79+35*bins+i]/(V_water*molec_masses['NH4+']))\
                + (dydt[79+36*bins+i]/(V_water*molec_masses['HCO3-']))\
                + ((2.0*dydt[79+37*bins+i])/(V_water*molec_masses['CO3--']))\
                + (dydt[79+38*bins+i]/(V_water*molec_masses['NO2-']))\
                + (dydt[79+39*bins+i]/(V_water*molec_masses['HCOO-']))
            '''
            reactions = (dydt[79+28*bins+i]/(V_water*molec_masses['HSO3-']))\
                + ((2.0*dydt[79+29*bins+i])/(V_water*molec_masses['SO3--']))\
                + (dydt[79+30*bins+i]/(V_water*molec_masses['HSO4-']))\
                + ((2.0*dydt[79+31*bins+i])/(V_water*molec_masses['SO4--']))\
                + (dydt[79+33*bins+i]/(V_water*molec_masses['NO3-']))\
                + (dydt[79+34*bins+i]/(V_water*molec_masses['NO4-']))\
                - (dydt[79+35*bins+i]/(V_water*molec_masses['NH4+']))\
                + (dydt[79+36*bins+i]/(V_water*molec_masses['HCO3-']))\
                + ((2.0*dydt[79+37*bins+i])/(V_water*molec_masses['CO3--']))\
                + (dydt[79+38*bins+i]/(V_water*molec_masses['NO2-']))\
                + (dydt[79+39*bins+i]/(V_water*molec_masses['HCOO-']))
            dHplus = 0.001*reactions
            dpH = (-0.434294*dHplus)/Hplus
            #print(y[index], dpH)
            dydt[index] = dpH  
        
    return dydt, air_changes
    
    
    
    

