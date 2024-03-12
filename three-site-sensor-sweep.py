#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:23:13 2024

@author: jacob
"""

import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # normalization of data for plotting
import matplotlib as mpl
from qdarray.sparse import DotArray
from optimisation.lossfunctions import EsplitLoss


D = 1.0
TS = 0.5*D
TSO = 0.2*TS
EZ = 1.5*D
U = 5.0*D

TD = 0.1*TS
TSOD = 0.2*TD

pmm_array = DotArray(ndots=6, energies=[0, 0, 0, 0, 0, 0],
                    hoppings = [[TD, TSOD], [TS, TSO],[TS, TSO], [TS, TSO],[TS, TSO]],
                    zeeman = [[0,EZ], [0,EZ], [0,0], [0,EZ], [0,0], [0,EZ]],
                    coulomb = [U, U, 0, U, 0, U],
                    proxgap = [0, 0, D, 0, D, 0])



sweep = np.hstack([np.linspace(-10, -EZ-U-.25*D, 20),
                 np.linspace(-EZ-U-.25*D, -EZ-U+.25*D, 100),
                 np.linspace(-EZ-U+.25*D, -.25*D, 20),
                 np.linspace(-.25*D, .25*D, 100),
                 np.linspace(.25*D, 4, 20)]).flatten()


def SensorSweep(sys, X, sweep, sensor=1):
    
    esplit_gnd = []
    esplit_ex1 = []
    esplit_ex2 = []
    
    for end in tqdm(sweep):
        if sensor == 1:
            det = np.append(np.array([end]), X)
            eige, eigo = sys.get_eigvals(detunings=det, k=2)
            esplit_gnd.append(np.abs(eige[0]-eigo[0]))
            esplit_ex1.append(np.abs(eige[1]-eigo[0]))
            esplit_ex2.append(np.abs(eige[0]-eigo[1]))
        
        elif sensor == 2:
            det = np.append(np.array([end]), X[::-1])
            eige, eigo = sys.get_eigvals(detunings=det, k=2)
            esplit_gnd.append(np.abs(eige[0]-eigo[0]))
            esplit_ex1.append(np.abs(eige[1]-eigo[0]))
            esplit_ex2.append(np.abs(eige[0]-eigo[1]))
        
            
    return np.array(esplit_gnd), np.array(esplit_ex1), np.array(esplit_ex2)


X = np.array([-0.05139512810991555, -0.01896395885327153, -0.7999999999502164, -0.7999999953484033, -0.2225844198528545])
print(np.sum(X**2))
X = np.array([-0.1705349612223454, -0.05728202700517395, -0.14576811852512814, -0.05728202698242133, -0.1705349612335501])
print(np.sum(X**2))
#X = np.array([-0.15396558077287115, -0.7999998883461931, 0.015610073308011738, -0.7999996222488144, -0.15396553975810687])
X = np.array([-0.20725064215744776, -0.5508105896453253, -0.6817725392605115, 0.06258819891267656, -0.02066555002238109])


gnd, ex1, ex2 = SensorSweep(pmm_array, X, sweep, sensor=1)

plt.figure()
plt.plot(sweep, gnd, color='C0')
plt.plot(sweep, -gnd, color='C0')

plt.plot(sweep, ex1, color='C1')
plt.plot(sweep, -ex1, color='C1')

plt.plot(sweep, ex2, color='C2')
plt.plot(sweep, -ex2, color='C2')

plt.ylim(-.2, .2)

plt.show()