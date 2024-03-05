#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:12:55 2024

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
                    coulomb = [U, U, 0, U],
                    proxgap = [0, 0, D, 0])


#sweep = np.linspace(-10, 4, 100)
sweep = np.hstack([np.linspace(-10, -EZ-U-.25*D, 20),
                 np.linspace(-EZ-U-.25*D, -EZ-U+.25*D, 50),
                 np.linspace(-EZ-U+.25*D, -.25*D, 20),
                 np.linspace(-.25*D, .25*D, 50),
                 np.linspace(.25*D, 4, 20)]).flatten()

def LossMap(LR_detunings, SC_detunings):
    
    
    loss_map = np.zeros((100,100))
    
    
    
    loss = EsplitLoss(pmm_array, sweep)
    
    for i in tqdm(range(LR_detunings.shape[0])):
        for j in range(SC_detunings.shape[0]):
            
            loss_map[i,j] = loss.prada_clarke([LR_detunings[i], SC_detunings[j], LR_detunings[i]])
            
    return loss_map

lr_bound = (-.4,.4)
s_bound = (-1,1)

sweep_s = np.linspace(s_bound[0],s_bound[1],100)
sweep_lr = np.linspace(lr_bound[0],lr_bound[1],100)

#loss_map = LossMap(sweep_lr, sweep_s)
#np.save("two-site-prada-lossmap.npy", loss_map)

#loss_map = np.load("two-site-max-lossmap.npy")
loss_map = np.load("two-site-prada-lossmap.npy")



min_index = np.unravel_index(loss_map.argmin(), loss_map.shape)


plt.figure(figsize=(10,6))
im = plt.imshow(loss_map, origin='lower', interpolation=None,
          extent=[s_bound[0], s_bound[1], lr_bound[0], lr_bound[1]])

plt.plot(sweep_s[min_index[1]], sweep_lr[min_index[0]], ls='', marker='x', color='r')
cb = plt.colorbar(im, fraction=0.0185)
cb.ax.set_yscale('linear')
plt.show()