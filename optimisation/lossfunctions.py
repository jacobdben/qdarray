#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:45:43 2024

@author: Jacob Benestad
"""

import numpy as np



class EsplitLoss():
    
    def __init__(self, dot_array, sensor_sweep, l2=0, sensor_weight='avg'):
        
        self.dot_array = dot_array
        self.l2_ = l2
        self.sw = sensor_weight
        self.sweep = sensor_sweep
        
    def sensormix(self, d1_loss, d2_loss):
        if self.sw == 'avg':
            return (d1_loss + d2_loss)/2
        elif self.sw == 'max':
            return max(d1_loss, d2_loss)
        else:
            assert False, "Error: sensor_weight must be 'avg' or 'max'"
        
    def prada_clarke(self, X):
        X = np.array(X)
        
        esplit_gnd_d1 = [[],[]]
        esplit_exc_d1 = [[],[]]
        esplit_gnd_d2 = [[],[]]
        esplit_exc_d2 = [[],[]]
        
        for end in self.sweep[:self.sweep.shape[0]//2]:
            det = np.append(np.array([end]), X)
            eige, eigo = self.dot_array.get_eigvals(detunings=det, k=2)
            esplit_gnd_d1[0].append(np.abs(eige[0]-eigo[0]))
            esplit_exc_d1[0].append(min(np.abs(eige[1]-eigo[0]), np.abs(eige[0]-eigo[1])))
            
            det = np.append(np.array([end]), X[::-1])
            eige, eigo = self.dot_array.get_eigvals(detunings=det, k=2)
            esplit_gnd_d2[0].append(np.abs(eige[0]-eigo[0]))
            esplit_exc_d2[0].append(min(np.abs(eige[1]-eigo[0]), np.abs(eige[0]-eigo[1])))
            
        for end in self.sweep[self.sweep.shape[0]//2:]:
            det = np.append(np.array([end]), X)
            eige, eigo = self.dot_array.get_eigvals(detunings=det, k=2)
            esplit_gnd_d1[1].append(np.abs(eige[0]-eigo[0]))
            esplit_exc_d1[1].append(min(np.abs(eige[1]-eigo[0]), np.abs(eige[0]-eigo[1])))
            
            det = np.append(np.array([end]), X[::-1])
            eige, eigo = self.dot_array.get_eigvals(detunings=det, k=2)
            esplit_gnd_d2[1].append(np.abs(eige[0]-eigo[0]))
            esplit_exc_d2[1].append(min(np.abs(eige[1]-eigo[0]), np.abs(eige[0]-eigo[1])))
            
        EM1_d1 = np.array(esplit_gnd_d1[0]).max()
        EM2_d1 = np.array(esplit_gnd_d1[1]).max()
        
        EM1_d2 = np.array(esplit_gnd_d2[0]).max()
        EM2_d2 = np.array(esplit_gnd_d2[1]).max()
        
        ED1_d1 = np.array(esplit_exc_d1[0]).min()
        ED2_d1 = np.array(esplit_exc_d1[1]).min()
        
        ED1_d2 = np.array(esplit_exc_d2[0]).min()
        ED2_d2 = np.array(esplit_exc_d2[1]).min()
        
        d1_loss = np.sqrt((EM1_d1/ED1_d1)**2+(EM2_d1/ED2_d1)**2)
        d2_loss = np.sqrt((EM1_d2/ED1_d2)**2+(EM2_d2/ED2_d2)**2)
        
        return self.sensormix(d1_loss, d2_loss) + self.l2_*np.sum(X**2)
    
    def max_esplit(self, X):
        X = np.array(X)
        
        esplit_gnd_d1 = []
        esplit_gnd_d2 = []

        
        for end in self.sweep[:self.sweep.shape[0]//2]:
            det = np.append(np.array([end]), X)
            eige, eigo = self.dot_array.get_eigvals(detunings=det, k=1)
            esplit_gnd_d1.append(np.abs(eige[0]-eigo[0]))
            
            det = np.append(np.array([end]), X[::-1])
            eige, eigo = self.dot_array.get_eigvals(detunings=det, k=1)
            esplit_gnd_d2.append(np.abs(eige[0]-eigo[0]))

            

            
        d1_loss = np.array(esplit_gnd_d1).max()
        d2_loss = np.array(esplit_gnd_d2).max()


        
        return self.sensormix(d1_loss, d2_loss) + self.l2_*np.sum(X**2)
    

class MajoranaQuality():
        
    def __init__(self, dot_array):
        
        self.dot_array = dot_array
        
    def dE(self, X):
        eige, eigo = self.dot_array.get_eigvals(detunings=X, k=2)
        return abs(eige[0]-eigo[0])
        
    def MP(self, X):
        
        mps = []

        vece, veco = self.dot_array.get_eigvecs(detunings=X)
        
        
        even = np.zeros(2*vece.shape[0], dtype=np.complex128)
        odd = np.zeros(2*veco.shape[0], dtype=np.complex128)
        
        np.put(even,self.dot_array.index_even,vece[:,0])
        np.put(odd,self.dot_array.index_odd,veco[:,0])
        
        for n in [0, self.dot_array.ndots-1]:
            
            wu = odd @ ( self.dot_array.cpu(n) + self.dot_array.cmu(n) ) @ even
            zu = odd @ ( self.dot_array.cpu(n) - self.dot_array.cmu(n) ) @ even
            wd = odd @ ( self.dot_array.cpd(n) + self.dot_array.cmd(n) ) @ even
            zd = odd @ ( self.dot_array.cpd(n) - self.dot_array.cmd(n) ) @ even
            
            mps.append( (wu**2-zu**2+wd**2-zd**2)/(wu**2+zu**2+wd**2+zd**2) )
        
        return (np.abs(mps[0]) + np.abs(mps[1])) / 2