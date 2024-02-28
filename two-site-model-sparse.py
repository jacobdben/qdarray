#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:33:48 2024

@author: jacob
"""

import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # normalization of data for plotting
import matplotlib as mpl
from qdarray.sparse import DotArray


D = 1.0
TS = 0.5*D
TSO = 0.2*TS
EZ = 1.5*D
U = 5.0*D

pmm_array = DotArray(ndots=3, energies=[0, 0, 0],
                    hoppings = [[TS, TSO],[TS, TSO]],
                    zeeman = [[0,EZ], [0,0], [0,EZ]],
                    coulomb = [U, 0, U],
                    proxgap = [0, D, 0])


def get_mps(sys, detunings=None):
        
        mps = []
        
        vece, veco = sys.get_eigvecs(detunings=detunings)
        
        
        even = np.zeros(2*vece.shape[0], dtype=np.complex128)
        odd = np.zeros(2*veco.shape[0], dtype=np.complex128)
        
        np.put(even,sys.index_even,vece[:,0])
        np.put(odd,sys.index_odd,veco[:,0])
        
        for n in range(sys.ndots):
            
            wu = odd @ ( sys.cpu(n) + sys.cmu(n) ) @ even
            zu = odd @ ( sys.cpu(n) - sys.cmu(n) ) @ even
            wd = odd @ ( sys.cpd(n) + sys.cmd(n) ) @ even
            zd = odd @ ( sys.cpd(n) - sys.cmd(n) ) @ even
            
            mps.append( (wu**2-zu**2+wd**2-zd**2)/(wu**2+zu**2+wd**2+zd**2) )
        
        return mps


def OddEvenMap(LR_detunings, SC_detunings):
    
    
    oe_map = np.zeros((100,100))
    
    
    hameven = pmm_array.get_even_ham()
    hamodd = pmm_array.get_odd_ham()
    
    for i in tqdm(range(LR_detunings.shape[0])):
        for j in range(SC_detunings.shape[0]):
            
            eige, eigo = pmm_array.get_eigvals(detunings=[LR_detunings[i], SC_detunings[j], LR_detunings[i]])
            
            oe_map[i,j] = eigo[0]-eige[0]
            
    return oe_map


def MPMap(LR_detunings, SC_detunings):
    
    
    mp_map = np.zeros((100,100,3), dtype=np.complex128)
    
    
    hameven = pmm_array.get_even_ham()
    hamodd = pmm_array.get_odd_ham()
    
    for i in tqdm(range(LR_detunings.shape[0])):
        for j in range(SC_detunings.shape[0]):

            mps = get_mps(pmm_array, detunings=[LR_detunings[i], SC_detunings[j], LR_detunings[i]])
            mp_map[i,j,0] = mps[0]
            mp_map[i,j,1] = mps[1]
            mp_map[i,j,2] = mps[2]
            
    return mp_map



lr_bound = (-.4,.4)
s_bound = (-1,1)

sweep_s = np.linspace(s_bound[0],s_bound[1],100)
sweep_lr = np.linspace(lr_bound[0],lr_bound[1],100)

oe_map = OddEvenMap(sweep_lr, sweep_s)

plt.figure(figsize=(10,6))
divnorm = mcolors.TwoSlopeNorm(vmin=-.1, vcenter=0, vmax=0.4)
im = plt.imshow(oe_map, origin='lower', cmap='seismic', interpolation=None, norm=divnorm,
          extent=[s_bound[0], s_bound[1], lr_bound[0], lr_bound[1]])


cb = plt.colorbar(im, fraction=0.0185)
cb.ax.set_yscale('linear')
plt.show()

mp_map = MPMap(sweep_lr, sweep_s)

plt.figure(figsize=(10,6))
offset = mcolors.TwoSlopeNorm(vcenter=0.95, vmax=1., vmin=0.)
im = plt.imshow(np.abs(mp_map[:,:,0]), origin='lower', cmap='hot', interpolation=None,
          extent=[s_bound[0], s_bound[1], lr_bound[0], lr_bound[1]], norm=offset)


cb = plt.colorbar(im)
cb.ax.set_yscale('linear')
plt.xlabel(r'$\varepsilon_S\, (\Delta)$', fontsize=16)
plt.ylabel(r'$\varepsilon_L$, $\varepsilon_R\, (\Delta)$', fontsize=16)
plt.show()

plt.figure(figsize=(10,6))
offset = mcolors.TwoSlopeNorm(vcenter=0.95, vmax=1., vmin=0.)
im = plt.imshow(np.abs(mp_map[:,:,2]), origin='lower', cmap='hot', interpolation=None,
          extent=[s_bound[0], s_bound[1], lr_bound[0], lr_bound[1]], norm=offset)


cb = plt.colorbar(im)
cb.ax.set_yscale('linear')
plt.xlabel(r'$\varepsilon_S\, (\Delta)$', fontsize=16)
plt.ylabel(r'$\varepsilon_L$, $\varepsilon_R\, (\Delta)$', fontsize=16)
plt.show()