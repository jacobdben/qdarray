#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:57:48 2024

@author: jacob
"""

import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # normalization of data for plotting
import matplotlib as mpl
from qdarray.dense import DotArray
from optimisation.lossfunctions import MajoranaQuality


D = 1.0
TS = 0.5*D
TSO = 0.2*TS
EZ = 1.5*D
U = 5.0*D

TD = 0.1*TS
TSOD = 0.2*TD

pmm_array = DotArray(ndots=5, energies=[0, 0, 0, 0, 0],
                    hoppings = [[TS, TSO],[TS, TSO], [TS, TSO],[TS, TSO]],
                    zeeman = [[0,EZ], [0,0], [0,EZ], [0,0], [0,EZ]],
                    coulomb = [U, 0, U, 0, U],
                    proxgap = [0, D, 0, D, 0])





X = np.array([-0.05139512810991555, -0.01896395885327153, -0.7999999999502164, -0.7999999953484033, -0.2225844198528545])
#X = np.array([-0.1485, -0.0077, -0.06735, -0.0077, -0.1485])
#X = np.array([-0.07960022930547474, 0.15199895813585232, 0.21584427009351037, 0.15199894464475855, -0.07960023295260607])
#X = np.array([-0.15396558077287115, -0.7999998883461931, 0.015610073308011738, -0.7999996222488144, -0.15396553975810687])
X = np.array([-0.20725064215744776, -0.5508105896453253, -0.6817725392605115, 0.06258819891267656, -0.02066555002238109])

quality = MajoranaQuality(pmm_array)
print("dE:", quality.dE(X))
print("MP:", quality.MP(X))

even_energies, odd_energies = pmm_array.get_eigvals(k=4, detunings=X)

print(even_energies)
print(odd_energies)
print(max(abs(even_energies[0]-odd_energies[1]), abs(even_energies[1]-odd_energies[0])))

quality.plot(X)