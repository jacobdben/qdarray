#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:01:26 2024

@author: jacob
"""

import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # normalization of data for plotting
import matplotlib as mpl
from qdarray.dense import DotArray
from optimisation.lossfunctions import EsplitLoss


D = 1.0
TS = 0.5*D
TSO = 0.2*TS
EZ = 1.5*D
U = 5.0*D

TD = 0.1*TS
TSOD = 0.2*TD

pmm_array = DotArray(ndots=4, energies=[0, 0, 0, 0],
                    hoppings = [[TD, TSOD], [TS, TSO],[TS, TSO]],
                    zeeman = [[0,EZ], [0,EZ], [0,0], [0,EZ]],
                    #zeeman = [EZ, EZ, 0, EZ],
                    coulomb = [U, U, 0, U],
                    proxgap = [0, 0, D, 0])



sweep = np.hstack([np.linspace(-10, -EZ-U-.25*D, 20),
                 np.linspace(-EZ-U-.25*D, -EZ-U+.25*D, 200),
                 np.linspace(-EZ-U+.25*D, -.25*D, 20),
                 np.linspace(-.25*D, .25*D, 200),
                 np.linspace(.25*D, 4, 20)]).flatten()


def SensorSweep(sys, X, sweep):
    
    esplit_gnd = []
    esplit_ex1 = []
    esplit_ex2 = []
    
    for end in sweep:
        det = np.append(np.array([end]), X)
        eige, eigo = sys.get_eigvals(detunings=det, k=2)
        esplit_gnd.append(np.abs(eige[0]-eigo[0]))
        esplit_ex1.append(np.abs(eige[1]-eigo[0]))
        esplit_ex2.append(np.abs(eige[0]-eigo[1]))
        
            
    return np.array(esplit_gnd), np.array(esplit_ex1), np.array(esplit_ex2)


X = np.array([-0.15355, -0.32894, -0.15355])


gnd, ex1, ex2 = SensorSweep(pmm_array, X, sweep)

plt.figure()
plt.plot(sweep, gnd, color='C0')
plt.plot(sweep, -gnd, color='C0')

plt.plot(sweep, ex1, color='C1')
plt.plot(sweep, -ex1, color='C1')

plt.plot(sweep, ex2, color='C2')
plt.plot(sweep, -ex2, color='C2')

plt.ylim(-.2, .2)

plt.show()