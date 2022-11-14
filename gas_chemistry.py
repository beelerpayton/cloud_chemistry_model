#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:55:17 2022

@author: paytonbeeler
"""
import numpy as np
import rates
from numba import njit
import warnings


warnings.filterwarnings("ignore")

R = 8.314 # m^3*Pa/mol*K
R_dry = 287.0 # Pa*m^3/kg*K
Mw = 18.015/1000.0 # molar mass of water, kg/mol
Ma = 28.9/1000.0 # molar mass of dry air, kg/mol
rho_w = 1000.0 # density of water, kg/m^3
kb = 1.380649e-23 #m^2 kg/s^2 K
Na = 6.022e23 #molecules/mol

@njit
def gas_phase(y, dydt, bins, N, dy_phasechange, Kelvin_effect = True, constant_pH = False):
    
    T = y[1]
    P = y[2]
    SS = y[3]
    
    Ns = y[78+3*bins:78+3*bins+bins]
    radii = y[78+1*bins:78+1*bins+bins]/2.0
    dry_radii = y[78+0*bins:78+0*bins+bins]/2.0
    
    V_H2Os = Ns*(4.0/3.0)*np.pi*(radii**3-dry_radii**3)
    k = rates.gaseous(T, SS+1)
    mult = ((P*Na)/(100**3*R*T))
    
    NO3 = np.zeros(bins)
    N2O5 = np.zeros(bins)
    HNO3 = np.zeros(bins)
    HONO = np.zeros(bins)
    HNO4 = np.zeros(bins)
    
    #ISOP
    index = 6
    reactions = -k[0]*mult*y[6]*mult*y[19]\
        - k[54]*mult*y[6]*mult*y[42] # molec/cm^3*s
    particles = 0 # (Na/100**3)*np.sum(dy_phasechange[78+40*bins:78+40*bins+bins]) # molec/cm^3*s
    dydt[index] = (reactions/mult) - particles # mol A/mol air
    
    #ISOPOO
    index = 7
    reactions = k[0]*mult*y[6]*mult*y[19]\
        - k[1]*mult*y[7]*mult*y[8]\
    	+ 0.7*k[3]*mult*y[9]*mult*y[19]\
    	- k[69]*(mult*y[30]+mult*y[32])*mult*y[7]\
    	- 2*k[70]*mult*y[7]*mult*y[7]
    particles = 0
    dydt[index] = (reactions/mult) - particles
        
    #HO2
    index = 8
    peroxy = y[30]+y[32]+y[31]+y[34]+y[48]+y[50]+y[57]+y[58]+y[59]+y[60]+y[61]+y[68]
    reactions = -k[1]*mult*y[7]*mult*y[8]\
        + 0.12*k[1]*mult*y[7]*mult*y[8]\
    	- k[5]*mult*y[13]*mult*y[8]\
    	+ 0.825*k[5]*mult*y[13]*mult*y[8]\
    	+ k[7]*mult*y[17]*mult*y[19]\
    	- 2*k[8]*mult*y[8]*mult*y[8]\
    	+ k[9]*mult*y[18]*mult*y[19]\
    	+ k[10]*mult*y[20]*mult*y[19]\
    	+ k[11]*mult*y[21]*mult*y[19]\
    	+ k[12]*mult*y[15]*mult*y[19]\
    	+ 0.75*k[16]*mult*y[14]*mult*y[19]\
    	+ 0.825*k[17]*mult*y[22]*mult*y[19]\
    	- k[19]*mult*y[8]*mult*y[35]\
    	+ k[20]*mult*y[14]*mult*y[19]\
    	+ 2*k[21]*mult*y[20]\
    	+ 0.8*k[24]*mult*y[15]\
    	+ k[25]*mult*y[16]\
    	+ 2*k[26]*mult*y[14]\
    	- k[27]*mult*y[19]*mult*y[8]\
    	+ k[41]*mult*y[19]*mult*y[42]\
    	- k[42]*mult*y[8]*mult*y[42]\
    	- k[50]*mult*y[36]*mult*y[8]\
    	+ k[51]*mult*y[41]\
    	+ k[53]*mult*y[45]*mult*y[19]\
    	+ k[57]*mult*y[33]*mult*y[35]\
    	+ 0.11*k[54]*mult*y[6]*mult*y[42]\
    	+ 0.23*k[55]*mult*y[46]*mult*y[42]\
    	+ k[59]*mult*y[47]\
    	- k[60]*mult*y[8]*mult*peroxy\
    	+ k[61]*mult*y[20]*mult*y[37]\
        + k[62]*mult*y[29]*mult*y[19]\
    	+ 0.265*k[65]*mult*y[32]*mult*y[35]\
    	+ 0.85*k[66]*mult*y[30]*mult*y[35]\
    	+ 1.2*k[69]*(mult*y[30]+mult*y[32])*mult*y[7]\
    	+ 1.3*k[70]*mult*y[7]*mult*y[7]\
    	- k[73]*mult*y[33]*mult*y[8]\
    	+ k[72]*mult*y[52]*mult*y[19]\
    	+ 0.937*k[85]*mult*y[64]*mult*y[19]\
    	+ 0.93*k[86]*mult*y[63]*mult*y[19]\
    	+ 0.38*k[87]*mult*y[19]*mult*y[65]\
    	+ 0.92*k[88]*mult*y[35]*mult*y[60]\
    	+ 0.40*k[89]*mult*y[66]*mult*y[19]\
    	+ k[93]*mult*y[23]\
    	+ k[94]*mult*y[22]\
    	+ 0.5*k[95]*mult*y[11]\
    	+ (k[96]+k[97])*mult*y[10]\
    	+ k[98]*mult*y[48]\
    	+ 0.98*k[99]*mult*y[54]\
    	+ k[101]*mult*y[59]*mult*y[35]\
    	+ k[102]*mult*y[68]*mult*y[35]\
    	+ k[103]*mult*y[57]*mult*y[35]\
    	+ 0.964*k[104]*mult*y[61]*mult*y[35]\
    	+ k[106]*mult*y[69]\
    	+ k[107]*mult*y[65]\
    	+ k[108]*mult*y[75]\
    	+ k[109]*mult*y[66]\
    	+ k[110]*mult*y[76]\
    	+ 0.93*k[112]*mult*y[73]*mult*y[19]\
    	+ 0.13*k[115]*mult*y[65]*mult*y[42]\
    	+ 0.13*k[116]*mult*y[66]*mult*y[42]\
    	+ k[117]*mult*y[15]*mult*y[37]\
    	+ 0.1*k[121]*mult*y[73]*mult*y[37]\
    	+ 0.1*k[125]*mult*y[63]*mult*y[37]\
        + 0.825*k[126]*mult*y[13]*mult*y[35]
    particles = np.sum(dy_phasechange[79+9*bins:79+9*bins+bins])
    dydt[index] = (reactions/mult) - particles
       
    #ISOPOOH
    index = 9
    reactions = 0.88*k[1]*mult*y[7]*mult*y[8]\
        - (k[2]+k[3])*mult*y[9]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles

    #MACR
    index = 10
    reactions = 0.047*k[1]*mult*y[7]*mult*y[8]\
        - k[14]*mult*y[10]*mult*y[19]\
     	+ 0.55*k[54]*mult*y[6]*mult*y[42]\
     	- k[63]*mult*y[10]*mult*y[42]\
     	- k[68]*mult*y[10]*mult*y[37]\
     	+ 0.3*k[69]*(mult*y[30]+mult*y[32])*mult*y[7]\
     	+ 0.34*k[70]*mult*y[7]*mult*y[7]\
     	- (k[96]+k[97])*mult*y[10]
    particles = 0 # (Na/100**3)*np.sum(dy_phasechange[78+7*bins:78+7*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #MVK
    index = 11
    reactions = 0.073*k[1]*mult*y[7]*mult*y[8]\
        - k[15]*mult*y[11]*mult*y[19]\
     	+ 0.21*k[54]*mult*y[6]*mult*y[42]\
     	- k[64]*mult*y[11]*mult*y[42]\
     	- k[67]*mult*y[11]*mult*y[37]\
     	+ 0.3*k[69]*(mult*y[30]+mult*y[32])*mult*y[7]\
     	+ 0.42*k[70]*mult*y[7]*mult*y[7]\
     	- k[95]*mult*y[11]
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+8*bins:78+8*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #IEPOX
    index = 12
    kmtgamma = 0
    reactions = -kmtgamma*mult*y[12]\
        + k[2]*mult*y[9]*mult*y[19]\
        - k[4]*mult*y[12]*mult*y[19]
    particles = np.sum(dy_phasechange[79+57*bins:79+57*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #IEPOXOO
    index = 13
    reactions = k[4]*mult*y[12]*mult*y[19]\
        - k[5]*mult*y[13]*mult*y[8]\
        - k[126]*mult*y[13]*mult*y[35]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #GLYC
    index = 14
    reactions = 0.275*k[5]*mult*y[13]*mult*y[8]\
        - k[16]*mult*y[14]*mult*y[19]\
     	- k[20]*mult*y[14]*mult*y[19]\
     	- k[26]*mult*y[14]\
     	+ 0.625*k[65]*mult*y[32]*mult*y[35]\
     	+ 0.5*k[93]*mult*y[23]\
        + 0.275*k[126]*mult*y[13]*mult*y[35]
    particles = 0 # (Na/100**3)*np.sum(dy_phasechange[78+10*bins:78+10*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #GLYX
    index = 15
    kmtgamma = 0
    reactions = -kmtgamma*mult*y[15]\
        + 0.275*k[5]*mult*y[13]*mult*y[8]\
     	- k[12]*mult*y[15]*mult*y[19]\
     	+ 0.13*k[16]*mult*y[14]*mult*y[19]\
     	+ k[20]*mult*y[14]*mult*y[19]\
     	- (k[23]+ k[24])*mult*y[15]\
     	+ k[52]*mult*y[45]*mult*y[19]\
     	+ 0.01*k[54]*mult*y[6]*mult*y[42]\
     	+ 0.238*k[84]*mult*y[62]*mult*y[19]\
     	+ 0.247*k[85]*mult*y[64]*mult*y[19]\
     	+ 0.13*k[86]*mult*y[63]*mult*y[19]\
     	+ 0.89*k[88]*mult*y[35]*mult*y[60]\
     	- k[117]*mult*y[15]*mult*y[37]\
     	+ 0.57*k[125]*mult*y[63]*mult*y[37]\
        + 0.275*k[126]*mult*y[13]*mult*y[35]
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+11*bins:78+11*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #MGLY
    index = 16
    kmtgamma = 0
    reactions = -kmtgamma*mult*y[16]\
        + 0.275*k[5]*mult*y[13]*mult*y[8]\
     	- k[13]*mult*y[16]*mult*y[19]\
     	+ 0.75*k[17]*mult*y[22]*mult*y[19]\
     	- k[25]*mult*y[16]\
     	+ 0.76*k[63]*mult*y[10]*mult*y[42]\
     	+ 0.76*k[64]*mult*y[11]*mult*y[42]\
     	+ 0.265*k[65]*mult*y[32]*mult*y[35]\
     	+ 0.425*k[66]*mult*y[30]*mult*y[35]\
     	+ k[67]*mult*y[11]*mult*y[37]\
     	+ 0.5*k[68]*mult*y[10]*mult*y[37]\
     	+ 0.6*k[69]*(mult*y[30]+mult*y[32])*mult*y[7]\
     	+ 0.1*k[70]*mult*y[7]*mult*y[7]\
     	+ 0.238*k[84]*mult*y[62]*mult*y[19]\
     	+ 0.14*k[85]*mult*y[64]*mult*y[19]\
     	+ 0.31*k[87]*mult*y[19]*mult*y[65]\
     	+ 0.11*k[88]*mult*y[35]*mult*y[60]\
     	+ 0.33*k[89]*mult*y[66]*mult*y[19]\
     	+  k[101]*mult*y[59]*mult*y[35]\
     	+ 0.14*k[112]*mult*y[73]*mult*y[19]\
     	+ 0.31*k[115]*mult*y[65]*mult*y[42]\
     	+ 0.30*k[116]*mult*y[66]*mult*y[42]\
     	- k[118]*mult*y[16]*mult*y[37]\
     	+ 0.57*k[121]*mult*y[73]*mult*y[37]\
        + 0.275*k[126]*mult*y[13]*mult*y[35]
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+12*bins:78+12*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #CO
    index = 17
    reactions = 0.251*k[5]*mult*y[13]*mult*y[8]\
        - k[7]*mult*y[17]*mult*y[19]\
     	+ k[10]*mult*y[20]*mult*y[19]\
     	+ 2*k[12]*mult*y[15]*mult*y[19]\
     	+ k[13]*mult*y[16]*mult*y[19]\
     	+ 0.52*k[16]*mult*y[14]*mult*y[19]\
     	+ 0.05*k[17]*mult*y[22]*mult*y[19]\
     	+ (k[21]+k[22])*mult*y[20]\
     	+ (1.87*k[23]+1.55*k[24])*mult*y[15]\
     	+ k[25]*mult*y[16]\
     	+ k[26]*mult*y[14]\
     	+ k[53]*mult*y[45]*mult*y[19]\
     	+ 0.18*k[54]*mult*y[6]*mult*y[42]\
     	+ 0.33*k[55]*mult*y[46]*mult*y[42]\
     	+ k[59]*mult*y[47]\
     	+ k[61]*mult*y[20]*mult*y[37]\
     	+ 0.24*k[64]*mult*y[11]*mult*y[42]\
     	+ 0.425*k[66]*mult*y[30]*mult*y[35]\
     	+ k[78]*mult*y[56]*mult*y[19]\
     	+ 0.10*k[87]*mult*y[19]*mult*y[65]\
     	+ 0.95*k[88]*mult*y[35]*mult*y[60]\
     	+ 0.10*k[89]*mult*y[66]*mult*y[19]\
     	+ k[92]*mult*y[67]*mult*y[19]\
     	+ 0.5*k[93]*mult*y[23]\
     	+ k[95]*mult*y[11]\
     	+ 1.65*k[96]*mult*y[10]\
     	+ k[105]*mult*y[31]*mult*y[35]\
     	+ k[107]*mult*y[65]\
     	+ k[108]*mult*y[75]\
     	+ k[109]*mult*y[66]\
     	+ k[110]*mult*y[76]\
     	+ 0.57*k[115]*mult*y[65]*mult*y[42]\
     	+ 0.57*k[116]*mult*y[66]*mult*y[42]\
     	+ 2*k[117]*mult*y[15]*mult*y[37]\
     	+ k[118]*mult*y[16]*mult*y[37]\
     	+ 0.10*k[120]*mult*y[65]*mult*y[37]\
     	+ 0.1*k[123]*mult*y[66]*mult*y[37]\
        + 0.251*k[126]*mult*y[13]*mult*y[35]
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+66*bins:78+66*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #HOOH
    index = 18
    reactions = -k[6]*mult*y[18]\
        + k[8]*mult*y[8]*mult*y[8]\
        - k[9]*mult*y[18]*mult*y[19]
    reactions = 0
    particles = np.sum(dy_phasechange[79+8*bins:79+8*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #OH
    index = 19
    CH4 = 41820000000000.0
    reactions = -k[0]*mult*y[6]*mult*y[19]\
        + 0.120*k[1]*mult*y[7]*mult*y[8]\
     	- k[3]*mult*y[9]*mult*y[19]\
     	+ 0.3*k[3]*mult*y[9]*mult*y[19]\
     	- k[4]*mult*y[12]*mult*y[19]\
     	+ 2*k[6]*mult*y[18]\
     	+ 1.125*k[5]*mult*y[13]*mult*y[8]\
     	- k[7]*mult*y[17]*mult*y[19]\
     	- k[9]*mult*y[18]*mult*y[19]\
     	- k[10]*mult*y[20]*mult*y[19]\
     	- k[11]*mult*y[21]*mult*y[19]\
     	- k[12]*mult*y[15]*mult*y[19]\
     	- k[13]*mult*y[16]*mult*y[19]\
     	- k[14]*mult*y[10]*mult*y[19]\
     	- k[15]*mult*y[11]*mult*y[19]\
     	- 0.75*k[16]*mult*y[14]*mult*y[19]\
     	- 0.9*k[17]*mult*y[22]*mult*y[19]\
     	- k[18]*mult*y[23]*mult*y[19]\
     	+ k[19]*mult*y[8]*mult*y[35]\
     	- k[20]*mult*y[14]*mult*y[19]\
     	- k[27]*mult*y[19]*mult*y[8]\
     	+ k[34]*mult*y[39]\
     	+ k[35]*mult*y[40]\
     	+ 2*k[40]*mult*y[43]\
     	- k[41]*mult*y[19]*mult*y[42]\
     	+ k[42]*mult*y[8]*mult*y[42]\
     	- k[43]*mult*y[35]*mult*y[19]\
     	- k[49]*mult*y[36]*mult*y[19]\
     	- k[53]*mult*y[45]*mult*y[19]\
     	- k[56]*CH4*mult*y[19]\
     	- k[58]*mult*y[47]*mult*y[19]\
     	+ 0.5*k[54]*mult*y[6]*mult*y[42]\
     	+ 0.1*k[55]*mult*y[46]*mult*y[42]\
     	- k[62]*mult*y[29]*mult*y[19]\
     	+ 0.46*k[63]*mult*y[10]*mult*y[42]\
     	+ 0.46*k[64]*mult*y[11]*mult*y[42]\
     	- k[72]*mult*y[52]*mult*y[19]\
     	- k[75]*(mult*y[24]+mult*y[25]+mult*y[26]+mult*y[28])*mult*y[19]\
     	- 0.7*k[74]*mult*y[53]*mult*y[19]\
     	- k[78]*mult*y[56]*mult*y[19]\
     	- k[79]*mult*y[46]*mult*y[19]\
     	- 0.5*k[80]*mult*y[48]*mult*y[19]\
     	- k[81]*mult*y[55]*mult*y[19]\
     	- k[82]*mult*y[51]*mult*y[19]\
     	- k[83]*mult*y[54]*mult*y[19]\
     	- k[84]*mult*y[62]*mult*y[19]\
     	- k[85]*mult*y[64]*mult*y[19]\
     	- k[86]*mult*y[63]*mult*y[19]\
     	- k[87]*mult*y[19]*mult*y[65]\
     	- k[89]*mult*y[66]*mult*y[19]\
     	- k[92]*mult*y[67]*mult*y[19]\
     	+ k[98]*mult*y[48]\
     	- k[111]*mult*y[69]*mult*y[19]\
     	- k[112]*mult*y[73]*mult*y[19]\
     	- k[113]*mult*y[19]*mult*y[75]\
     	- k[114]*mult*y[19]*mult*y[76]\
        + 0.125*k[126]*mult*y[13]*mult*y[35]
    particles = np.sum(dy_phasechange[79+10*bins:79+10*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #FORM
    index = 20
    reactions = 0.12*k[1]*mult*y[7]*mult*y[8]\
        + 0.375*k[5]*mult*y[13]*mult*y[8]\
     	- k[10]*mult*y[20]*mult*y[19]\
     	+ 0.71*k[16]*mult*y[14]*mult*y[19]\
     	- (k[21]+k[22])*mult*y[20]\
     	+ (0.13*k[23]+0.45*k[24])*mult*y[15]\
     	+ k[26]*mult*y[14]\
     	+ k[57]*mult*y[33]*mult*y[35]\
     	+ 0.89*k[54]*mult*y[6]*mult*y[42]\
     	+ 0.53*k[55]*mult*y[46]*mult*y[42]\
     	- k[61]*mult*y[20]*mult*y[37]\
     	+ 0.7*k[63]*mult*y[10]*mult*y[42]\
     	+ 0.7*k[64]*mult*y[11]*mult*y[42]\
     	+ 0.425*k[65]*mult*y[32]*mult*y[35]\
     	+ 0.425*k[66]*mult*y[30]*mult*y[35]\
     	+ k[71]*mult*y[33]*mult*y[33]\
     	+ k[72]*mult*y[52]*mult*y[19]\
     	+ 0.3*k[74]*mult*y[53]*mult*y[19]\
     	+ k[69]*(mult*y[30]+mult*y[32])*mult*y[7]\
     	+ 0.55*k[70]*mult*y[7]*mult*y[7]\
     	+ k[78]*mult*y[56]*mult*y[19]\
     	+ k[94]*mult*y[22]\
     	+ 0.5*k[95]*mult*y[11]\
     	+ k[96]*mult*y[10]\
     	+ k[103]*mult*y[57]*mult*y[35]\
     	+ 0.09*k[104]*mult*y[61]*mult*y[35]\
     	+ k[105]*mult*y[31]*mult*y[35]\
        + 0.375*k[126]*mult*y[13]*mult*y[35]
    particles = np.sum(dy_phasechange[79+23*bins:79+23*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #FORMIC
    index = 21
    reactions = 0.074*k[5]*mult*y[13]*mult*y[8]\
        - k[11]*mult*y[21]*mult*y[19]\
     	+ 0.16*k[16]*mult*y[14]*mult*y[19]\
     	+ 0.125*k[17]*mult*y[22]*mult*y[19]\
     	+ k[53]*mult*y[45]*mult*y[19]\
     	+ 0.07*k[54]*mult*y[6]*mult*y[42]\
     	+ 0.2*k[55]*mult*y[46]*mult*y[42]\
     	+ 0.11*k[63]*mult*y[10]*mult*y[42]\
     	+ 0.11*k[64]*mult*y[11]*mult*y[42]\
        + 0.074*k[126]*mult*y[13]*mult*y[35]
    particles = np.sum(dy_phasechange[79+24*bins:79+24*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #HAC
    index = 22
    reactions = 0.725*k[5]*mult*y[13]*mult*y[8]\
        - k[17]*mult*y[22]*mult*y[19]\
     	+ 0.425*k[66]*mult*y[30]*mult*y[35]\
     	- k[94]*mult*y[22]\
        + 0.725*k[126]*mult*y[13]*mult*y[35]
    particles = 0
    dydt[index] = (reactions/mult) - particles

    #HC5
    index = 23
    reactions = 0.3*k[3]*mult*y[9]*mult*y[19]\
        - k[18]*mult*y[23]*mult*y[19]\
        - k[93]*mult*y[23]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #GLYCAC
    index = 24
    reactions = -k[75]*mult*y[24]*mult*y[19]
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+32*bins:78+32*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #GLYAC
    index = 25
    reactions = -k[75]*mult*y[25]*mult*y[19]
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+16*bins:78+16*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #PYRAC
    index = 26
    reactions = -k[75]*mult*y[26]*mult*y[19]
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+34*bins:78+34*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #OXLAC
    index = 27
    reactions = 0
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+17*bins:78+17*bins+bins])
    dydt[index] = (reactions/mult) - particles

    #ACETIC
    index = 28
    reactions = 0.125*k[17]*mult*y[22]*mult*y[19]\
        + 0.15*k[54]*mult*y[6]*mult*y[42]\
     	+ 0.2*k[55]*mult*y[46]*mult*y[42]\
     	+ 0.23*k[63]*mult*y[10]*mult*y[42]\
     	- k[75]*mult*y[28]*mult*y[19]
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+36*bins:78+36*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #SO2
    index = 29
    reactions = -k[62]*mult*y[29]*mult*y[19]
    reactions = 0
    particles = np.sum(dy_phasechange[79+19*bins:79+19*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #MACP
    index = 30
    reactions = + 0.47*k[14]*mult*y[10]*mult*y[19]\
        - k[60]*mult*y[8]*mult*y[30]\
     	- k[66]*mult*y[30]*mult*y[35]\
     	- k[69]*mult*y[30]*mult*y[7]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #CH3CO3
    index = 31
    reactions = 0.53*k[14]*mult*y[10]*mult*y[19]\
        + k[13]*mult*y[16]*mult*y[19]\
     	+ k[25]*mult*y[16]\
     	+ k[58]*mult*y[47]*mult*y[19]\
     	- k[60]*mult*y[8]*mult*y[31]\
     	+ 0.625*k[65]*mult*y[32]*mult*y[35]\
     	+ 0.8*k[68]*mult*y[10]*mult*y[37]\
     	- k[76]*mult*y[31]*mult*y[36]\
     	+ k[77]*mult*y[56]\
     	+ 0.07*k[87]*mult*y[19]*mult*y[65]\
     	+ 0.05*k[88]*mult*y[35]*mult*y[60]\
     	+ 0.08*k[89]*mult*y[66]*mult*y[19]\
     	+ 0.5*k[93]*mult*y[23]\
     	+ k[94]*mult*y[22]\
     	+ 0.5*k[95]*mult*y[11]\
     	+ 0.5*k[96]*mult*y[10]\
     	+ 0.02*k[99]*mult*y[54]\
     	+ k[100]*mult*y[51]\
     	- k[105]*mult*y[31]*mult*y[35]\
     	+ 0.7*k[115]*mult*y[65]*mult*y[42]\
     	+ 0.7*k[116]*mult*y[66]*mult*y[42]\
     	+ k[118]*mult*y[16]*mult*y[37]\
     	+ k[119]*mult*y[47]*mult*y[37]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #MVKP
    index = 32
    reactions = k[15]*mult*y[11]*mult*y[19]\
        - k[60]*mult*y[8]*mult*y[32]\
     	- k[65]*mult*y[32]*mult*y[35]\
     	- k[69]*mult*y[32]*mult*y[7]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #CH3OO
    index = 33
    reactions = 0.125*k[17]*mult*y[22]*mult*y[19]\
     	- 2*k[28]*mult*y[33]*mult*y[33]\
     	+ 0.09*k[54]*mult*y[6]*mult*y[42]\
     	+ 0.22*k[55]*mult*y[46]*mult*y[42]\
     	+ k[56]*CH4*mult*y[19]\
     	- k[57]*mult*y[33]*mult*y[35]\
     	+ k[59]*mult*y[47]\
     	- k[60]*mult*y[8]*mult*y[33]\
     	+ 0.24*k[64]*mult*y[11]*mult*y[42]\
     	+ k[67]*mult*y[11]*mult*y[37]\
     	- 2*k[71]*mult*y[33]*mult*y[33]\
     	- k[73]*mult*y[33]*mult*y[8]\
     	+ 0.7*k[74]*mult*y[53]*mult*y[19]\
     	+ k[75]*(mult*y[24]+mult*y[25]+mult*y[26]+mult*y[28])*mult*y[19]\
     	+ 0.65*k[96]*mult*y[10]\
     	+ k[105]*mult*y[31]*mult*y[35]\
     	+ 0.5*k[107]*mult*y[65]\
     	+ 0.5*k[109]*mult*y[66]
    particles = np.sum(dy_phasechange[79+22*bins:79+22*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    '''
    #HC5OO
    index = 34
    reactions = k[18]*mult*y[23]*mult*y[19]\
        - k[60]*mult*y[8]*mult*y[34]\
        - k[90]*mult*y[34]*mult*y[36]+mult*y[34]/(mult*y[34]+mult*y[60])*k[91]*mult*y[67]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    '''
    
    #NO
    index = 35
    reactions = -k[19]*mult*y[8]*mult*y[35]\
        + k[31]*mult*y[36]\
     	+ k[33]*mult*y[37]\
     	+ k[35]*mult*y[40]\
     	- k[43]*mult*y[35]*mult*y[19]\
     	- k[44]*mult*y[35]*mult*y[42]\
     	- k[46]*mult*y[35]*mult*y[37]\
     	- k[57]*mult*y[33]*mult*y[35]\
     	- k[65]*mult*y[32]*mult*y[35]\
     	- k[66]*mult*y[30]*mult*y[35]\
     	- k[88]*mult*y[35]*mult*y[60]\
     	- k[101]*mult*y[59]*mult*y[35]\
     	- k[102]*mult*y[68]*mult*y[35]\
     	- k[103]*mult*y[57]*mult*y[35]\
     	- k[104]*mult*y[61]*mult*y[35]\
     	- k[105]*mult*y[31]*mult*y[35]\
        - k[126]*mult*y[13]*mult*y[35]
    particles = np.sum(dy_phasechange[79+11*bins:79+11*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #NO2
    index = 36
    reactions = k[19]*mult*y[8]*mult*y[35]\
        - k[31]*mult*y[36]\
     	+ k[32]*mult*y[37]\
     	+ k[34]*mult*y[39]\
     	+ k[36]*mult*y[38]\
     	+ k[44]*mult*y[35]*mult*y[42]\
     	- k[45]*mult*y[36]*mult*y[42]\
     	+ 2*k[46]*mult*y[35]*mult*y[37]\
     	- k[47]*mult*y[36]*mult*y[37]\
     	+ k[48]*mult*y[38]\
     	- k[49]*mult*y[36]*mult*y[19]\
     	- k[50]*mult*y[36]*mult*y[8]\
     	+ k[51]*mult*y[41]\
     	+ k[57]*mult*y[33]*mult*y[35]\
     	+ 0.89*k[65]*mult*y[32]*mult*y[35]\
     	+ 0.85*k[66]*mult*y[30]*mult*y[35]\
     	+ k[67]*mult*y[11]*mult*y[37]\
     	+ 0.5*k[68]*mult*y[10]*mult*y[37]\
     	- k[76]*mult*y[31]*mult*y[36]\
     	+ k[77]*mult*y[56]\
     	+ k[78]*mult*y[56]*mult*y[19]\
     	+ 1.32*k[86]*mult*y[63]*mult*y[19]\
     	+ 0.38*k[87]*mult*y[19]*mult*y[65]\
     	+ k[88]*mult*y[35]*mult*y[60]\
     	+ 1.27*k[89]*mult*y[66]*mult*y[19]\
     	- k[90]*(mult*y[34]+mult*y[60])*mult*y[36]\
     	+ k[91]*mult*y[67]\
     	+ k[92]*mult*y[67]*mult*y[19]\
     	+ k[101]*mult*y[59]*mult*y[35]\
     	+ k[102]*mult*y[68]*mult*y[35]\
     	+ k[103]*mult*y[57]*mult*y[35]\
     	+ 0.964*k[104]*mult*y[61]*mult*y[35]\
     	+ k[105]*mult*y[31]*mult*y[35]\
     	+ k[106]*mult*y[69]\
     	+ k[111]*mult*y[69]*mult*y[19]\
     	+ 0.68*k[112]*mult*y[73]*mult*y[19]\
     	+ k[113]*mult*y[19]*mult*y[75]\
     	+ k[114]*mult*y[19]*mult*y[76]\
     	+ 0.18*k[115]*mult*y[65]*mult*y[42]\
     	+ 0.18*k[116]*mult*y[66]*mult*y[42]\
     	+ 2.05*k[120]*mult*y[65]*mult*y[37]\
     	+ 3.76*k[121]*mult*y[73]*mult*y[37]\
     	+ k[122]*mult*y[75]*mult*y[37]\
     	+ 2.05*k[123]*mult*y[66]*mult*y[37]\
     	+ k[124]*mult*y[76]*mult*y[37]\
     	+ 3.69*k[125]*mult*y[63]*mult*y[37]\
        + k[126]*mult*y[13]*mult*y[35]
    particles = np.sum(dy_phasechange[79+12*bins:79+12*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #NO3
    index = 37
    kmtgamma = 0
    reactions = -kmtgamma*mult*y[37]\
        - (k[32]+k[33])*mult*y[37]\
     	+ k[36]*mult*y[38]\
     	+ k[45]*mult*y[36]*mult*y[42]\
     	- k[46]*mult*y[35]*mult*y[37]\
     	- k[47]*mult*y[36]*mult*y[37]\
     	+ k[48]*mult*y[38]\
     	- k[61]*mult*y[20]*mult*y[37]\
     	- k[67]*mult*y[11]*mult*y[37]\
     	- k[68]*mult*y[10]*mult*y[37]\
     	- k[117]*mult*y[15]*mult*y[37]\
     	- k[118]*mult*y[16]*mult*y[37]\
     	- k[119]*mult*y[47]*mult*y[37]\
     	- k[120]*mult*y[65]*mult*y[37]\
     	- k[121]*mult*y[73]*mult*y[37]\
     	- k[122]*mult*y[75]*mult*y[37]\
     	- k[123]*mult*y[66]*mult*y[37]\
     	- k[124]*mult*y[76]*mult*y[37]\
     	- k[125]*mult*y[63]*mult*y[37]
    particles = np.sum(dy_phasechange[79+13*bins:79+13*bins+bins])
    dydt[index] = (reactions/mult) - particles
  
    #N2O5
    index = 38
    kmtgamma = 0
    reactions = -kmtgamma*mult*y[38]\
        - k[36]*mult*y[38]\
        + k[47]*mult*y[36]*mult*y[37]\
        - k[48]*mult*y[38]
    particles = np.sum(dy_phasechange[79+14*bins:79+14*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #HNO3
    index = 39
    kmtgamma = 0
    reactions = -kmtgamma*mult*y[39]\
        - k[34]*mult*y[39]\
     	+ k[49]*mult*y[36]*mult*y[19]\
     	+ k[61]*mult*y[20]*mult*y[37]\
     	+ 0.5*k[68]*mult*y[10]*mult*y[37]\
     	+ k[117]*mult*y[15]*mult*y[37]\
     	+ k[118]*mult*y[16]*mult*y[37]\
     	+ k[119]*mult*y[47]*mult*y[37]\
     	+ k[120]*mult*y[65]*mult*y[37]\
     	+ k[122]*mult*y[75]*mult*y[37]\
     	+ k[123]*mult*y[66]*mult*y[37]\
     	+ k[124]*mult*y[76]*mult*y[37]\
     	+ 0.49*k[125]*mult*y[63]*mult*y[37]
    reactions = 0
    particles = np.sum(dy_phasechange[79+16*bins:79+16*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #HONO
    index = 40
    reactions = -k[35]*mult*y[40]\
        + k[43]*mult*y[35]*mult*y[19]
    particles = np.sum(dy_phasechange[79+15*bins:79+15*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #HNO4
    index = 41
    reactions = k[50]*mult*y[36]*mult*y[8]\
        - k[51]*mult*y[41]
    particles = np.sum(dy_phasechange[79+17*bins:79+17*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #O3
    index = 42
    reactions = -(k[29]+k[30])*mult*y[42]\
        + k[39]*mult*y[44]\
     	- k[41]*mult*y[19]*mult*y[42]\
     	- k[42]*mult*y[8]*mult*y[42]\
     	- k[44]*mult*y[35]*mult*y[42]\
     	- k[45]*mult*y[36]*mult*y[42]\
     	- k[54]*mult*y[6]*mult*y[42]\
     	- k[55]*mult*y[46]*mult*y[42]\
     	- k[63]*mult*y[10]*mult*y[42]\
     	- k[64]*mult*y[11]*mult*y[42]\
     	- k[115]*mult*y[65]*mult*y[42]\
     	- k[116]*mult*y[66]*mult*y[42]
    reactions = 0
    particles = np.sum(dy_phasechange[79+6*bins:79+6*bins+bins])
    dydt[index] = (reactions/mult) - particles    
    
    #O1D
    index = 43
    reactions = k[29]*mult*y[42]\
        - (k[37]+k[38])*mult*y[43]\
        - k[40]*mult*y[43]
    particles = 0
    dydt[index] = (reactions/mult) - particles
       
    #O3P
    index = 44
    reactions = k[30]*mult*y[42]\
        + k[31]*mult*y[36]\
     	+ k[32]*mult*y[37]\
     	+ (k[37]+k[38])*mult*y[43]\
     	- k[39]*mult*y[44]\
     	+ 0.3*k[54]*mult*y[6]*mult*y[42]\
     	+ 0.23*k[63]*mult*y[10]*mult*y[42]\
     	+ 0.23*k[64]*mult*y[11]*mult*y[42]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #C2H2
    index = 45
    reactions = -(k[52]+k[53])*mult*y[45]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #OLT
    index = 46
    reactions = 0.07*k[54]*mult*y[6]*mult*y[42]\
        - k[55]*mult*y[46]*mult*y[42]\
     	+ 0.4*k[69]*(mult*y[30]+mult*y[32])*mult*y[7]\
     	+ 0.3*k[70]*mult*y[7]*mult*y[7]\
     	- k[79]*mult*y[46]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #ALD
    index = 47
    reactions = 0.5*k[55]*mult*y[46]*mult*y[42]\
        - k[58]*mult*y[47]*mult*y[19]\
     	- k[59]*mult*y[47]\
     	+ 0.5*k[80]*mult*y[48]*mult*y[19]\
     	+ 0.66*k[87]*mult*y[19]*mult*y[65]\
     	+ 0.66*k[89]*mult*y[66]*mult*y[19]\
     	+ k[92]*mult*y[67]*mult*y[19]\
     	+ k[98]*mult*y[48]\
     	+ k[102]*mult*y[68]*mult*y[35]\
     	+ k[103]*mult*y[57]*mult*y[35]\
     	+ 0.75*k[104]*mult*y[61]*mult*y[35]\
     	+ 0.2*k[106]*mult*y[69]\
     	+ k[108]*mult*y[75]\
     	+ k[110]*mult*y[76]\
     	+ k[113]*mult*y[19]*mult*y[75]\
     	+ k[114]*mult*y[19]*mult*y[76]\
     	- k[119]*mult*y[47]*mult*y[37]\
     	+ k[122]*mult*y[75]*mult*y[37]\
     	+ k[124]*mult*y[76]*mult*y[37]
    particles = 0
    dydt[index] = (reactions/mult) - particles
   
    #OP2
    index = 48
    reactions = k[60]*mult*y[8]*mult*peroxy\
        - k[80]*mult*y[48]*mult*y[19]\
     	+ 0.5*k[93]*mult*y[23]\
     	+ k[97]*mult*y[10]\
     	- k[98]*mult*y[48]\
     	+ 0.02*k[121]*mult*y[73]*mult*y[37]\
     	+ 0.02*k[125]*mult*y[63]*mult*y[37]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #CH3O
    index = 49
    reactions = 2*k[28]*mult*y[33]*mult*y[33]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #XO2
    index = 50
    reactions = -k[60]*mult*y[8]*mult*y[50]\
        + 2*k[88]*mult*y[35]*mult*y[60]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #KET
    index = 51
    reactions = -k[82]*mult*y[51]*mult*y[19]\
        - k[100]*mult*y[51]\
     	+ 0.25*k[104]*mult*y[61]*mult*y[35]\
     	+ 0.8*k[106]*mult*y[69]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #CH3OH
    index = 52
    reactions = k[71]*mult*y[33]*mult*y[33]\
        -k[72]*mult*y[52]*mult*y[19]
    particles = np.sum(dy_phasechange[79+26*bins:79+26*bins+bins])
    dydt[index] = 0 # (reactions-particles)/mult
     
    #CH3O2H
    index = 53
    reactions = k[73]*mult*y[33]*mult*y[8]\
        - k[74]*mult*y[53]*mult*y[19]
    particles = np.sum(dy_phasechange[79+25*bins:79+25*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #DCB
    index = 54
    reactions = 0.1*k[69]*(mult*y[30]+mult*y[32])*mult*y[7]\
        + 0.06*k[70]*mult*y[7]*mult*y[7]\
     	- k[83]*mult*y[54]*mult*y[19]\
     	- k[99]*mult*y[54]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #OLI
    index = 55
    reactions = 0.3*k[69]*(mult*y[30]+mult*y[32])*mult*y[7]\
        + 0.1*k[70]*mult*y[7]*mult*y[7]\
        - k[81]*mult*y[55]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #PAN
    index = 56
    reactions = k[76]*mult*y[31]*mult*y[36]\
        - k[77]*mult*y[56]\
        - k[78]*mult*y[56]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #OLTP
    index = 57
    reactions = -k[60]*mult*y[8]*mult*y[57]\
        + k[79]*mult*y[46]*mult*y[19]\
        - k[103]*mult*y[57]*mult*y[35]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #OLIP
    index = 58
    reactions = -k[60]*mult*y[8]*mult*y[58]\
        + k[81]*mult*y[55]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #KETP
    index = 59
    reactions = -k[60]*mult*y[8]*mult*y[59]\
        + k[82]*mult*y[51]*mult*y[19]\
        - k[101]*mult*y[59]*mult*y[35]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    '''
    #TCO3
    index = 60
    if y[34] <= 1E-30 and y[60] <= 1E-30:
        R1 = 0
    else:
        R1 = k[90]*mult*y[34]*mult*y[36]+mult*y[34]/(mult*y[34]+mult*y[60])*k[91]*mult*y[67]
    reactions = - k[60]*mult*y[8]*mult*y[60]\
        + k[83]*mult*y[54]*mult*y[19]\
     	- k[88]*mult*y[35]*mult*y[60]\
     	- R1\
     	+ k[99]*mult*y[54]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    '''
    
    #HC3P
    index = 61
    reactions = -k[60]*mult*y[8]*mult*y[61]\
        + 0.5*k[80]*mult*y[48]*mult*y[19]\
     	- k[104]*mult*y[61]*mult*y[35]\
     	+ k[111]*mult*y[69]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #TOL
    index = 62
    reactions = -k[84]*mult*y[19]*mult*y[62]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #CSL
    index = 63
    reactions = 0.20*k[84]*mult*y[19]*mult*y[62]\
        - k[86]*mult*y[63]*mult*y[19]\
        - k[125]*mult*y[63]*mult*y[37]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #XYL
    index = 64
    reactions = -k[85]*mult*y[64]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #TOL_EPOX
    index = 65
    reactions = 0.20*k[84]*mult*y[62]*mult*y[19]\
     	- k[87]*mult*y[19]*mult*y[65]\
     	- k[107]*mult*y[65]\
     	- k[115]*mult*y[65]*mult*y[42]\
     	- k[120]*mult*y[65]*mult*y[37]
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+79*bins:78+79*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #XYL_EPOX
    index = 66
    reactions = 0.21*k[85]*mult*y[64]*mult*y[19]\
     	- k[89]*mult*y[66]*mult*y[19]\
     	- k[109]*mult*y[66]\
     	- k[116]*mult*y[66]*mult*y[42]\
     	- k[123]*mult*y[66]*mult*y[37]
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+85*bins:78+85*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #APN
    index = 67
    reactions = k[90]*(mult*y[34]+mult*y[60])*mult*y[36]\
     	- k[91]*mult*y[67]\
     	- k[92]*mult*y[67]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #ETHP
    index = 68
    reactions = -k[60]*mult*y[8]*mult*y[68]\
     	+ k[100]*mult*y[51]\
     	- k[102]*mult*y[68]*mult*y[35]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #ONIT
    index = 69
    reactions = 0.11*k[65]*mult*y[32]*mult*y[35]\
     	+ 0.15*k[66]*mult*y[30]*mult*y[35]\
     	+ 0.036*k[104]*mult*y[61]*mult*y[35]\
     	- k[106]*mult*y[69]\
     	- k[111]*mult*y[69]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #FURANONES
    index = 70
    reactions = 0.198*k[84]*mult*y[62]*mult*y[19]\
        + 0.07*k[85]*mult*y[64]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #AROMATIC_KETONES
    index = 71
    reactions = 0.106*k[84]*mult*y[62]*mult*y[19]\
        + 0.09*k[85]*mult*y[64]*mult*y[19]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #UNSAT_ALDEHYDES
    index = 72
    reactions = 0.277*k[84]*mult*y[62]*mult*y[19]\
     	+ 0.317*k[85]*mult*y[64]*mult*y[19]\
     	+ 0.13*k[86]*mult*y[63]*mult*y[19]\
     	+ 0.31*k[87]*mult*y[19]*mult*y[65]\
     	+ 0.26*k[89]*mult*y[66]*mult*y[19]\
     	+ 0.5*k[107]*mult*y[65]\
     	+ 0.5*k[109]*mult*y[66]\
     	+ 0.14*k[112]*mult*y[73]*mult*y[19]\
     	+ k[120]*mult*y[65]*mult*y[37]\
     	+ 0.57*k[121]*mult*y[73]*mult*y[37]\
     	+ k[123]*mult*y[66]*mult*y[37]\
     	+ 0.57*k[125]*mult*y[63]*mult*y[37]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #MCSL
    index = 73
    reactions = 0.16*k[85]*mult*y[64]*mult*y[19]\
     	- k[112]*mult*y[73]*mult*y[19]\
     	- k[121]*mult*y[73]*mult*y[37]
    particles = 0
    dydt[index] = (reactions/mult) - particles
   
    #POLY_FXNAL_AROMATIC
    index = 74
    reactions = 0.09*k[85]*mult*y[64]*mult*y[19]\
     	+ 0.86*k[86]*mult*y[63]*mult*y[19]\
     	+ 0.86*k[112]*mult*y[73]*mult*y[19]\
     	+ 0.40*k[121]*mult*y[73]*mult*y[37]\
     	+ 0.40*k[125]*mult*y[63]*mult*y[37]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #EPXC4DIAL
    index = 75
    reactions = 0.38*k[87]*mult*y[19]*mult*y[65]\
     	+ 0.5*k[107]*mult*y[65]\
     	- k[108]*mult*y[75]\
     	- k[113]*mult*y[19]*mult*y[75]\
     	+ k[115]*mult*y[65]*mult*y[42]\
     	- k[122]*mult*y[75]*mult*y[37]
    particles = 0
    dydt[index] = (reactions/mult) - particles
    
    #EPXC4MDIAL
    index = 76
    reactions = 0.41*k[89]*mult*y[66]*mult*y[19]\
        + 0.5*k[109]*mult*y[66]\
     	- k[110]*mult*y[76]\
     	- k[114]*mult*y[19]*mult*y[76]\
     	+ k[116]*mult*y[66]*mult*y[42]\
     	- k[124]*mult*y[76]*mult*y[37]
    particles = 0 #(Na/100**3)*np.sum(dy_phasechange[78+88*bins:78+88*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #H2SO4
    index = 77
    reactions = k[62]*mult*y[29]*mult*y[19]
    particles = np.sum(dy_phasechange[79+20*bins:79+20*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    #NH3
    index = 78
    reactions = -k[127]*mult*y[78]*mult*y[19]
    reactions = 0
    particles = np.sum(dy_phasechange[79+18*bins:79+18*bins+bins])
    dydt[index] = (reactions/mult) - particles
    
    return dydt
