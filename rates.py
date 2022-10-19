#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 10:21:25 2022

@author: paytonbeeler
"""

import numpy as np
from numba import njit

@njit
def gaseous(T, RH):

    k = np.zeros(128)
    H2O = (RH*100)/100*np.power(10, 8.10765-1750.286/(T-273.15+235))/760*2.46e19*(298.15/T)
    
    k[0] = 2.7e-11*np.exp(390/T)                                                  # cm3/molec/s Paulot    ISOP + OH --> ISOPOO
    k[1] = 0.074e-11*np.exp(700/T)                                                # Paulot                ISOPOO + HO2 --> 0.880 ISOPOOH + 0.120 OH + 0.047 MACR + 0.073 MVK + 0.120 HO2 + 0.120 FORM
    k[2] = 1.9e-11*np.exp(390/T)                                                  # Paulot                ISOPOOH + OH --> IEPOX + OH
    k[3] = 0.38e-11*np.exp(200/T)                                                 # Paulot                ISOPOOH + OH --> 0.7 ISOPOO + 0.3 HC5 + 0.3 OH
    k[4] = 5.78e-11*np.exp(-400/T)                                                # Paulot                IEPOX + OH --> IEPOXOO 
    k[5] = 0.074e-11*np.exp(700/T)                                                # Paulot                IEPOXOO + HO2 --> 0.725 HAC + 0.275 GLYC + 0.275 GLYX + 0.275 MGLY + 1.125 OH + 0.825 HO2 + 0.200 CO2 + 0.375 FORM + 0.074 FORMIC + 0.251 CO 
    k[6] = 7.657e-6                                                             # Saunders 2003, Atlanta GA, noon, June 21 (SZA = 13.5)                                                      
    k[7] = 1.5e-13                                                             # Seinfeld & Pandis     CO + OH --> CO2 + HO2
    k[8] = 2.3e-13*np.exp(600/T)+1.7e-33*2.46e19*298.15/T*np.exp(1000/T)             # Seinfeld & Pandis     HO2 + HO2 --> H2O2 + O2
    k[9] = 2.9e-12*np.exp(-160/T)                                                # Lim 2005              H2O2 + OH --> HO2 + H2O 
    k[10] = 1.20e-14*T*np.exp(287/T)                                              # Seinfeld & Pandis     FORM + OH --> HO2 + CO + H2O
    k[11] = 4.5e-13                                                            # Seinfeld & Pandis     FORMIC + OH --> HO2 + CO2 + H2O
    k[12] = 11.4e-12                                                           # Seinfeld & Pandis     GLYX + OH --> HO2 + 2CO + H2O
    k[13] = 17.2e-12                                                           # Seinfeld & Pandis     MGLY + OH --> CH3CO3 + CO + H2O
    k[14] = 2.95e-11                                                           # Paulot 2009a          MACR + OH --> 0.47 MACP + 0.53 CH3CO3  
    k[15] = 1.75e-11                                                           # Paulot 2009a          MVK + OH --> MVKP
    k[16] = 0.8e-11                                                            # Paulot 2009a          GLYC + OH --> 0.71 FORM + 0.16 FORMIC + 0.35 CO2  + 0.52 CO + 0.13 GLYX + 0.75 HO2 + 0.25 OH  
    k[17] = 0.6e-11                                                            # Paulot 2009a          HAC + OH --> 0.75 MGLY + 0.825 HO2 + 0.125 FORMIC + 0.1 OH + 0.125 CH3OO + 0.20 CO2 + 0.05 CO + 0.125 ACETIC
    k[18] = 11e-11                                                             # Paulot 2009a          HC5 + OH --> HC5OO
    k[19] = 3.5e-12*np.exp(250/T)                                                 # Seinfeld & Pandis     HO2 + NO --> NO2 + OH
    k[20] = 2.5e-12                                                            # Lim 2005              GLYC + OH --> GLYX + HO2
    k[21] = 3.161e-5
    k[22] = 4.851e-5
    k[23] = 5.546e-5
    k[24] = 2.709e-5
    k[25] = 1.235e-4
    k[26] = 1.928e-5
    k[27] = 4.8e-11*np.exp(250/T)                                                 # Lim 2005              HO2 + OH --> H2O + O2
    k[28] = 0                                                                  # CAPRAM (5.98e-14)     2 CH3OO --> 2 CH3O + O2 (REMOVED)
    k[29] = 3.553e-5
    k[30] = 2.080e-4
    k[31] = 8.792e-3
    k[32] = 1.5301e-1
    k[33] = 2.213e-2
    k[34] = 6.561e-7
    k[35] = 1.952e-3
    k[36] = 0
    k[37] = 1.8e-11*np.exp(100/T)*2.46e19*(298.15/T)*0.79                         # Lim 2005              O1D + N2 --> O3P + N2
    k[38] = 3.2e-11*np.exp(70/T)*2.46e19*(298.15/T)*0.21                          # Lim 2005              O1D + O2 --> O3P + O2
    k[39] = 6e-34*np.power(T/300, -2.3)*np.power(2.46e19, 2)*np.power(298.15/T, 2)*0.21                   # Lim 2005              O3P + O2 + M --> O3 + M
    k[40] = 2.2e-10*H2O                                                        # Lim 2005              O1D + H2O --> 2OH
    k[41] = 1.6e-12*np.exp(-940/T)                                                # Lim 2005              O3 + OH --> HO2 + O2
    k[42] = 1.1e-14*np.exp(-500/T)                                                # Lim 2005              O3 + HO2 --> OH + 2O2
    k[43] = 7.6e-12                                                            # Lim 2005              NO + OH (+M) --> HONO(+M)
    k[44] = 2e-12*np.exp(-1400/T)                                                 # Lim 2005              O3 + NO --> NO2 + O2
    k[45] = 1.2e-13*np.exp(-2450/T)                                               # Lim 2005              O3 + NO2 --> NO3 + O2
    k[46] = 1.5e-11*np.exp(170/T)                                                 # Lim 2005              NO + NO3 --> 2 NO2
    k[47] = 1.3e-12                                                            # Lim 2005              NO2 + NO3 (+M) --> N2O5(+M)
    k[48] = 6.8e-2*np.exp(-11080/T)                                               # Lim 2005              N2O5(+M) --> NO2 + NO3 (+M)
    k[49] = 9.7e-12                                                            # Lim 2005              NO2 + OH (+M)--> HNO3(+M)
    k[50] = 1.5e-12                                                            # Lim 2005              NO2 + HO2 (+M) --> HNO4(+M)
    k[51] = 7.7e-12*np.exp(-10420/T)                                              # Lim 2005              HNO4 (+M)--> NO2 + HO2(+M)
    k[52] = 8.8e-13*0.636                                                      # Atkinson 1984, MCM    C2H2 + OH --> GLYX + OH
    k[53] = 8.8e-13*0.364                                                      # Atkinson 1984, MCM    C2H2 + OH --> FORMIC + CO + HO2
    k[54] = 1.23e-14*np.exp(-2013/T)                                              # Lim                   ISOP + O3 --> 0.55 MACR + 0.21 MVK + 0.3 O3P + 0.5 OH + 0.89 FORM + 0.11 HO2 + 0.18 CO + 0.07 OLT + 0.15 ACETIC+ 0.07 FORMIC + 0.09 CH3OO + 0.01 GLY  2005 - assuming ORA2 = acetic, ORA1 = formic, MO2 = CH3OO. OLT is terminal alkene
    k[55] = 1.32e-14*np.exp(-2105/T)                                              # Lim                   OLT + O3 -->  0.53 FORM + 0.5 ALD + 0.33 CO + 0.2 FORMIC + 0.2 ACETIC + 0.23 HO2 + 0.22 CH3OO + 0.1 OH + 0.06 CH4
    k[56] = 2.45e-12*np.exp(-1775/T)                                              # Lim                   CH4 + OH --> CH3OO + H2O
    k[57] = 3e-12*np.exp(280/T)                                                   # Lim                   CH3OO + NO --> FORM + HO2 + NO2
    k[58] = 6.87e-12*np.exp(256/T)                                                # Lim                   ALD + OH --> CH3CO3 + H2O
    k[59] = 4.62e-06
    k[60] = 7.7e-14*np.exp(1300/T)                                                # Lim                   peroxy (MACP+CH3CO3+MVKP+CH3OO+HC5OO+OLTP+OLIP+KETP+HC3P+TCO3+XO2+TOLP+XYLP+ETHP) + HO2 --> OP2 'higher peroxides'
    k[61] = 5.8e-16                                                            # Lim                   FORM + NO3 --> HNO3 + HO2 + CO
    k[62] = 4.8e-13                                                            # Lim                   SO2 + OH --> SO3 + HO2 (note: not currently tracking SO3)
    k[63] = 4.4e-15*np.exp(-2500/T)                                               # Lim                   MACR + O3 --> 0.76 MGLY + 0.23 O3P + 0.46 OH + 0.7 FORM + 0.11 FORMIC + 0.23 ACETIC  subst ORA1 = FORMIC, ORA2 = ACETIC
    k[64] = 4e-15*np.exp(-2000/T)                                                 # Lim                   MVK + O3 --> 0.76 MGLY + 0.23 O3P + 0.46 OH + 0.7 FORM + 0.11 FORMIC + 0.24 CH3OO + 0.24 CO subst ORA1 = FORMIC, MO2 = CH3OO
    k[65] = 0.81e-11                                                           # Paulot 2009b          MVKP + NO --> 0.625 CH3CO3 + 0.625 GLYC + 0.265 FORM + 0.265 HO2 + 0.265 MGLY + 0.89 NO2 + 0.11 ONIT
    k[66] = 0.81e-11                                                           # Paulot 2009b          MACP + NO --> 0.425 HAC + 0.425 FORM + 0.425 MGLY + 0.85 HO2 + 0.85 NO2 + 0.15 ONIT + 0.425 CO
    k[67] = 6e-14                                                              # Lim                   MVK + NO3 --> NO2 + CH3OO + MGLY
    k[68] = 1e-14                                                              # Lim                   MACR + NO3 --> 0.5 HNO3 + 0.5 NO2 + 0.8 CH3CO3 + 0.5 MGLY
    k[69] = 0.07*5e-14                                                         # Lim THIS IMPROVES AGREEMENT w/Paulot.         MVKP or MACP + ISOPOO --> 0.3 MACR + 0.3 MVK + 0.6 MGLY + FORM + 1.2 HO2 + 0.1 DCB + 0.4 OLT + 0.3 OLI
    k[70] = 0.07*1e-14                                                         # Lim THIS IMPROVES AGREEMENT w/Paulot.         ISOPOO + ISOPOO --> 0.34 MACR + 0.42 MVK + 0.3 OLT + 0.06 DCB + 0.1 OLI + 1.3 HO2 + 0.55 FORM + 0.1 MGLY
    k[71] = 0                                                                  # CAPRAM (2.8e-13)      2CH3OO --> FORM + CH3OH + O2 (REMOVED)
    k[72] = 6.7e-12*np.exp(-600/T)                                                # Lim                   CH3OH + OH --> FORM + HO2 + H2O
    k[73] = 3.8e-13*np.exp(800/T)                                                 # Lim                   CH3OO + HO2 --> CH3O2H + O2
    k[74] = 3.8e-12*np.exp(200/T)                                                 # Lim                   CH3O2H + OH --> 0.3 FORM + 0.3 OH + 0.3 H2O + 0.7 CH3OO + 0.7 H2O
    k[75] = 4e-13*np.exp(200/T)                                                   # Lim                   ACETIC, GLYCAC, GLYAC, PYRAC + OH --> CH3OO
    k[76] = 2.8e-12*np.exp(181/T)                                                 # Lim                   CH3CO3 + NO2 --> PAN
    k[77] = 1.95e16*np.exp(-13543/T)                                              # Lim                   PAN --> CH3CO3 + NO2
    k[78] = 3e-14                                                              # MCM                   PAN + OH --> FORM + NO2 + CO
    k[79] = 5.32e-12*np.exp(504/T)                                                # Lim                   OLT + OH --> OLTP 
    k[80] = 1e-11                                                              # Lim                   OP2 + OH --> 0.5 HC3P + 0.5 ALD + 0.5 OH
    k[81] = 1.07e-11*np.exp(549/T)                                                # Lim                   OLI + OH --> OLIP
    k[82] = 1.2e-11*np.exp(-745/T)                                                # Lim                   KET + OH --> KETP + H2O
    k[83] = 2.8e-11                                                            # Lim                   DCB + OH --> TCO3 + H2O
    k[84] = 1.81e-12*np.exp(338/T)                                                # MCM                   TOL + OH --> 0.20 CSL + 0.238 GLYX + 0.238 MGLY + 0.198 FURANONES + 0.106 AROMATIC_KETONES + 0.20 TOL_EPOX + 0.277 UNSAT_ALDEHYDES
    k[85] = 1.43e-11                                                           # MCM                   XYL + OH --> 0.937 HO2 + 0.09 AROMATIC_KETONES + 0.16 MCSL + 0.09 POLY_FXNAL_AROMATIC + 0.21 XYL_EPOX + 0.317 UNSAT_ALDEHYDES + 0.247 GLYX + 0.14 MGLY + 0.07 FURANONES
    k[86] = 4.65e-11                                                           # MCM                   CSL + OH --> 0.13GLYX + 0.13UNSAT_ALDEHYDES+0.93HO2 + 0.86POLY_FXNAL_AROMATIC +0.68 NO2
    k[87] = 7.99e-11                                                           # MCM                   TOL_EPOX + OH --> 0.38EPXC4DIAL + 0.31UNSAT_ALDEHYDES + 0.070CH3CO3 + 0.31MGLY + 0.10CO + 0.38HO2 + 1.32 NO2 +0.61 ALD
    k[88] = 4.2e-12*np.exp(180/T)                                                 # Lim                   TCO3 + NO --> 0.89 GLY + 0.11 MGLY + 0.05 CH3CO3 + 0.95 CO + 2 XO2 + NO2 + 0.92 HO2
    k[89] = 7.88e-11                                                           # MCM                   XYL_EPOX + OH --> 0.41EPXC4MDIAL + 0.26UNSAT_ALDEHYDES + 0.08CH3CO3 + 0.33MGLY + 0.10CO + 0.40HO2 + 1.27NO2 + 0.66ALD
    k[90] = 1e-11                                                              # LaFranchi 2009        HC5OO, TCO3 + NO2 --> APN
    k[91] = 4.6e-4                                                              # LaFranchi 2009        APN --> HC5OO, MOBAOO, TCO3 + NO2
    k[92] = 3.2e-11                                                             # La Franchi            APN + OH --> ALD + NO2 + CO
    k[93] = 2*8.298e-06
    k[94] = 3.83e-6
    k[95] = 2*1.34e-5
    k[96] = 8.298e-06
    k[97] = 8.298e-06
    k[98] = 9.260e-06
    k[99] = 9.26e-6
    k[99] = 6.04e-04
    k[100] = 1.40e-06 
    k[101] = 4.2e-12*np.exp(180/T)                                                # Lim                   KETP + NO --> MGLY + NO2 + HO2
    k[102] = 4.2e-12*np.exp(180/T)                                                # Lim                   ETHP + NO --> ALD + HO2 + NO2
    k[103] = 4.2e-12*np.exp(180/T)                                                # Lim                   OLTP + NO --> ALD + FORM + NO2 + HO2
    k[104] = 4.2e-12*np.exp(180/T)                                                # Lim                   HC3P + NO --> 0.75 ALD + 0.25 KET + 0.09 FORM + 0.036 ONIT + 0.964 NO2 + 0.964 HO2
    k[105] = 2.1e-11                                                           # Paulot 2009b          CH3CO3 + NO --> NO2 + CO + CO2 + FORM + CH3OO
    k[106] = 1.47e-6
    k[107] = 8.8e-4
    k[108] = 1.07e-4
    k[109] = 8.8e-4
    k[110] = 1.07e-4
    k[111] = 1.55e-11*np.exp(-540/T)                                              # Lim                   ONIT + OH --> HC3P + NO2
    k[112] = 8e-11                                                             # MCM                   MCSL + OH --> 0.14MGLY + 0.14UNSAT_ALDEHYDES + 0.93HO2 + 0.86POLY_FXNAL_AROMATIC +0.68NO2
    k[113] = 4.32e-11                                                          # MCM                   EPXC4DIAL + OH --> ALD + NO2
    k[114] = 4.32e-11                                                          # MCM                   EPXC4MDIAL + OH --> ALD + NO2
    k[115] = 5e-18                                                             # MCM                   TOL_EPOX + O3 --> EPXC4DIAL + 0.70 CH3CO3 + 0.31 MGLY + 0.57 CO + 0.13 HO2 + 0.18 NO2
    k[116] = 5e-18                                                             # MCM                   XYL_EPOX + O3 --> EPXC4MDIAL + 0.70 CH3CO3 + 0.30 MGLY + 0.57 CO + 0.13 HO2 + 0.18 NO2
    k[117] = 6e-13*np.exp(-2058/T)                                                # Lim 2005              GLYX + NO3 --> HO2 + HNO3 + 2 CO
    k[118] = 1.4e-12*np.exp(-1900/T)                                              # Lim 2005              MGLY + NO3 --> CH3CO3 + HNO3 + CO
    k[119] = 1.4e-12*np.exp(-1900/T)                                              # Lim 2005              ALD + NO3 --> CH3CO3 + HNO3
    k[120] = 2.7254e-15                                                        # MCM                   TOL_EPOX + NO3 --> UNSAT_ALDEHYDES + HNO3 + 0.10 CO + 2.05 NO2
    k[121] = 3.48e-11                                                          # MCM                   MCSL + NO3 --> 0.57 MGLY + 0.57 UNSAT_ALDEHYDES + 0.1 HO2 + 0.40 POLY_FXNAL_AROMATIC + 3.76 NO2 + 0.02 OP2
    k[122] = 2.18e-14                                                          # MCM                   EPXC4DIAL + NO3 --> ALD + HNO3 + NO2
    k[123] = 1.16e-14                                                          # MCM                   XYL_EPOX + NO3 --> UNSAT_ALDEHYDES + HNO3 + 0.1 CO + 2.05 NO2
    k[124] = 4.63e-14                                                          # MCM                   EPXC4MDIAL + NO3 --> ALD + HNO3 + NO2
    k[125] = 1.4e-11                                                           # MCM                   CSL + NO3 --> 0.57 GLYX + 0.57 UNSAT_ALDEHYDES+0.1 HO2 + 0.40 POLY_FXNAL_AROMATIC + 0.49 HNO3 + 3.69 NO2 +0.02 OP2
    k[126] = 2.60e-12*np.exp(380/T)                                               # Xie ACPD 2013         IEPOXOO + NO --> 0.725HAC + 0.275GLYC + 0.275GLYX + 0.275MGLY + 0.125OH + 0.825HO2 + 0.200CO2 + 0.375 HCHO + 0.074 FORMIC + 0.251CO + NO2 
    k[127] = ((6.022E23/np.power(100.0,3))*1.7E-12*np.exp(-710*(1/298)))*(6.022E23/100**3)  # S&P         NH3 + OH --> H2O + NH2 

    return k

@njit
def full_aqueous(y, T, bins, O3, Hplus):
    
    ka = np.zeros(72)
    
    ka[1] = 0
    ka[2] = 0
    ka[3] = 7.0e9*np.exp(-1500*(1/T - 1/298))
    ka[4] = 1.0e10*np.exp(-1500*(1/T - 1/298))
    ka[5] = 2.7e7*np.exp(-1700*(1/T - 1/298))
    ka[6] = 8.6e5*np.exp(-2365*(1/T - 1/298))
    ka[7] = 1.0e8*np.exp(-1500*(1/T - 1/298))
    ka[8] = 0.2
    ka[9] = 0.5
    ka[10] = 0.13
    ka[11] = 2.0e9
    ka[12] = 1e4
    ka[13] = 1.5e9*np.exp(-1500*(1/T - 1/298))
    ka[14] = 70
    ka[15] = 2.8e6*np.exp(-2500*(1/T - 1/298))
    ka[16] = 7.8e-3*np.sqrt(O3)
    ka[17] = 1.5e7*np.exp(-1910*(1/T - 1/298))
    ka[18] = 1.5e6*np.exp(0*(1/T - 1/298))
    ka[19] = 4.0e8*np.exp(-1500*(1/T - 1/298))
    ka[20] = 8.0e5*np.exp(-2820*(1/T - 1/298))
    ka[21] = 4.3e9*np.exp(-1500*(1/T - 1/298))
    ka[22] = 6.1e9*np.exp(0*(1/T - 1/298))
    ka[23] = 2.1e10*Hplus*np.exp(0*(1/T - 1/298))
    ka[24] = 1.3e3*np.exp(0*(1/T - 1/298))
    ka[25] = 4.5e9*np.exp(-1500*(1/T - 1/298))
    ka[26] = 1.0e9*np.exp(-1500*(1/T - 1/298))
    ka[27] = 3.1e9*np.exp(-1500*(1/T - 1/298))
    ka[28] = 1.4e5*np.exp(-3370*(1/T - 1/298))
    ka[29] = 4.5e7*np.exp(0*(1/T - 1/298))
    ka[30] = 7.3e6*np.exp(-2160*(1/T - 1/298))
    ka[31] = 2.0e8*np.exp(-1500*(1/T - 1/298))
    ka[32] = 1.0e8*np.exp(-1500*(1/T - 1/298))
    ka[33] = 2.0e10*np.exp(-1500*(1/T - 1/298))
    ka[34] = 1.3e9*np.exp(-1500*(1/T - 1/298))
    ka[35] = 0 # photolysis
    ka[36] = 0 # photolysis
    ka[37] = 1.0e9*np.exp(-1500*(1/T - 1/298))
    ka[38] = 1.0e10*np.exp(-1500*(1/T - 1/298))
    ka[39] = 6.3e3*Hplus*np.exp(-6693*(1/T - 1/298))
    ka[40] = 5.0e5*np.exp(-6950*(1/T - 1/298))
    ka[41] = 4.0e5*np.exp(0*(1/T - 1/298))
    ka[42] = 2.5e8*np.exp(-1500*(1/T - 1/298))
    ka[43] = 1.2e9*np.exp(-1500*(1/T - 1/298))
    ka[44] = 0 # photolysis
    ka[45] = 0 # photolysis
    ka[46] = 4.5e9*np.exp(-1500*(1/T - 1/298))
    ka[47] = 1.0e9*np.exp(-1500*(1/T - 1/298))
    ka[48] = 1.0e6*np.exp(-2800*(1/T - 1/298))
    ka[49] = 1.0e8*np.exp(-1500*(1/T - 1/298))
    ka[50] = 2.0e9*np.exp(-1500*(1/T - 1/298))
    ka[51] = 0.1*np.exp(0*(1/T - 1/298))
    ka[52] = 2.0e8*np.exp(-1500*(1/T - 1/298))
    ka[53] = 4.6e-6*np.exp(-5180*(1/T - 1/298))
    ka[54] = 2.1e5*np.exp(-3200*(1/T - 1/298))
    ka[55] = 5.0*np.exp(0*(1/T - 1/298))
    ka[56] = 6.7e3*np.exp(-4300*(1/T - 1/298))
    ka[57] = 2.5e9*np.exp(-1500*(1/T - 1/298))
    ka[58] = 100.0*np.exp(0*(1/T - 1/298))
    ka[59] = 6.0e7*np.exp(-1500*(1/T - 1/298))
    ka[60] = 1.1e5*np.exp(-3400*(1/T - 1/298))
    ka[61] = 1.9e6*np.exp(-2600*(1/T - 1/298))
    ka[62] = 4.0e-4*np.exp(0*(1/T - 1/298))
    ka[63] = 4.3e5*np.exp(-3000*(1/T - 1/298))
    ka[64] = 5.0e7*np.exp(-1600*(1/T - 1/298))
    ka[65] = 0 # photolysis
    ka[66] = 2.7e7*np.exp(-1700*(1/T - 1/298))
    ka[67] = 4.5e8*np.exp(-1500*(1/T - 1/298))
    ka[68] = 2.6e3*np.exp(-4500*(1/T - 1/298))
    ka[69] = 3.5e3*np.exp(-4400*(1/T - 1/298))
    ka[70] = 1.9e7*np.exp(-1800*(1/T - 1/298))
    ka[71] = 1.0e6*np.exp(-2800*(1/T - 1/298))
    ka[72] = 0
    ka[73] = 0
    ka[74] = 0
    ka[75] = 5.2e9*np.exp(-1500*(1/T - 1/298))
    ka[76] = 4.5e9*np.exp(-1500*(1/T - 1/298))
    ka[77] = 2.5e4*np.exp(-3100*(1/T - 1/298))
    ka[78] = 1.0e8*np.exp(-2000*(1/T - 1/298))
    ka[79] = 200.0*np.exp(-1500*(1/T - 1/298))
    ka[80] = 1.4e4*np.exp(-5300*(1/T - 1/298))
    ka[81] = 6.0e8*np.exp(-4000*(1/T - 1/298))
    ka[82] = 7.1e6*np.exp(-1500*(1/T - 1/298))
    ka[83] = 1.7e7*np.exp(-3100*(1/T - 1/298))
    ka[84] = 1.0e5*np.exp(-1900*(1/T - 1/298))
    ka[85] = 0.31*np.exp(0*(1/T - 1/298))
    ka[86] = 1.8e-3*np.exp(-6650*(1/T - 1/298))
    ka[87] = 1.3e9*np.exp(-7050*(1/T - 1/298))
    ka[88] = 5.3e8*np.exp(-1500*(1/T - 1/298))
    ka[89] = 5.0e9*np.exp(-1500*(1/T - 1/298))
    ka[90] = 5.0e9*np.exp(-1500*(1/T - 1/298))
    ka[91] = 8.0e7*np.exp(-1500*(1/T - 1/298))
    ka[92] = 1.2e7*np.exp(-2000*(1/T - 1/298))
    ka[93] = 8.8e8*np.exp(-1500*(1/T - 1/298))
    ka[94] = 9.1e6*np.exp(-2100*(1/T - 1/298))
    ka[95] = 1.7e8*np.exp(-1500*(1/T - 1/298))
    ka[96] = 2.0e8*np.exp(-1500*(1/T - 1/298))
    ka[97] = 1.4e6*np.exp(-2700*(1/T - 1/298))
    ka[98] = 6.7e-3*np.exp(0*(1/T - 1/298))
    ka[99] = 2.3e7*np.exp(-3800*(1/T - 1/298))
    ka[100] = 0
    ka[101] = 0
    ka[102] = 2.5e7*np.exp(-1800*(1/T - 1/298))
    ka[103] = 1.0e8*np.exp(0*(1/T - 1/298))
    ka[104] = 2.0e6*np.exp(0*(1/T - 1/298))
    ka[105] = 0
    ka[106] = 0
    ka[107] = 3.6e3*np.exp(-4500*(1/T - 1/298))
    ka[108] = 2.6e8*np.exp(-1500*(1/T - 1/298))
    ka[109] = 3.4e8*np.exp(-1500*(1/T - 1/298))
    
    return ka # units are M^(1-n)/s

@njit
def simple_aqueous():
    
    ka = np.zeros(8)
    
    ka[0] = 2.4e4
    ka[1] = 3.7e5
    ka[2] = 1.5e9
    ka[4] = 7.45e7
    ka[5] = 13.0
    ka[7] = 2.0e6
    
    return ka # units are M^(1-n)/s