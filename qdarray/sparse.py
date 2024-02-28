#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:18:36 2024

@author: Jacob Benestad, Jeroen Danon
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


# GLOBAL VARIABLES

cpu_mat = sparse.diags( [1,0,1], -1) # Up spin single dot creation matrix
cpd_mat = sparse.diags( [1,1], -2)   # Down spin single dot creation matrix
cmu_mat = sparse.diags( [1,0,1], 1)  # Up spin single dot destruction matrix
cmd_mat = sparse.diags( [1,1], 2)    # Down spin single dot destruction matrix






# HELPER FUNCTIONS

def binary_array(n,m):
    
    digits = []
    remainder = n
    
    for i in range(m):
        
        digits.append( np.divmod( remainder, 2**(m-1-i) )[0] )
        remainder = np.divmod( remainder, 2**(m-1-i) )[1]
    
    return digits








# DOT ARRAY CLASS

class DotArray():
    
    def __init__(self, ndots, energies, hoppings, zeeman, coulomb, proxgap):
        
        assert ndots > 1, "Error: Must have more than one dot"
        assert len(energies) == ndots, "Error: Wrong number of energies"
        assert len(hoppings) == ndots-1, "Error: Wrong number of hoppings"
        assert len(zeeman) == ndots, "Error: Wrong number of Zeeman fields"
        assert len(coulomb) == ndots, "Error: Wrong number of charging energies"
        assert len(proxgap) == ndots, "Error: Wrong number of superconductors"
        
        self.ndots = ndots
        self.ens = energies
        
        zeeman = np.array(zeeman)
        
        if len(zeeman.shape) == 1:
            zeeman = np.tile(zeeman, (2,1)).T / 2
        assert len(zeeman.shape) == 2 and zeeman.shape[1] == 2, "Error: Bad zeeman shape"
            
        
        self.EZs = zeeman
        self.ts = np.array(hoppings)[:,0]
        self.tsos = np.array(hoppings)[:,1]
        self.Us = coulomb
        self.Ds = proxgap
        
        self.occupations = []

        for n in range(4**ndots):
            self.occupations.append( binary_array( n, 2*ndots ) )
        
        self.pos = [ np.identity(4**n) for n in range(ndots) ]
        
        # Get parity indices
        parities = np.identity(1)

        for n in range(ndots):
            parities = np.kron( np.array([1,-1,-1,1]), parities)
        parities = parities[0]

        self.index_even = np.where(parities==1)[0]
        self.index_odd = np.where(parities==-1)[0]
        
        self.ham = self.make_hamiltonian()
        
    
    # Up spin creation operator
    def cpu(self, n):
        parity_factors = sparse.diags( (-1)**np.sum( np.array(self.occupations)[:,0:2*(n+1)-1], axis=1 ) )
        return sparse.kron( self.pos[n], sparse.kron( cpu_mat, self.pos[self.ndots-1-n] ) ) @ parity_factors

    # Down spin creation operator
    def cpd(self, n):
        parity_factors = sparse.diags( (-1)**np.sum( np.array(self.occupations)[:,0:2*(n+1)-2], axis=1 ) )
        return sparse.kron( self.pos[n], sparse.kron( cpd_mat, self.pos[self.ndots-1-n] ) ) @ parity_factors

    # Up spin destruction operator
    def cmu(self, n):
        return self.cpu(n).T

    # Down spin destruction operator
    def cmd(self, n):
        return self.cpd(n).T

    # Single dot energy
    def dia(self, h,n):
        return sparse.kron( self.pos[n], sparse.kron( h, self.pos[self.ndots-1-n] ) )
    
    
    def make_hamiltonian(self):
        
        # Dot energies
        ham = 0
        for n in range(self.ndots):
            ham += self.dia(sparse.diags([0, self.ens[n]-self.EZs[n,0], self.ens[n]+self.EZs[n,1], 2*self.ens[n]-self.EZs[n,0]+self.EZs[n,1]+self.Us[n]]),n)

        # Spin conserving hopping
        for n in range(0, self.ndots-1):
            ham += self.ts[n] * ( self.cpu(n)@self.cmu(n+1) + self.cpu(n+1)@self.cmu(n) + self.cpd(n)@self.cmd(n+1) + self.cpd(n+1)@self.cmd(n) )

        # Spin non-conserving hopping
        for n in range(0, self.ndots-1):
            ham += self.tsos[n] * ( self.cpu(n)@self.cmd(n+1) + self.cpd(n+1)@self.cmu(n) - self.cpd(n)@self.cmu(n+1) - self.cpu(n+1)@self.cmd(n) )

        for n in range(0, self.ndots):
            ham += ( self.Ds[n]*( self.cpu(n)@self.cpd(n) + self.cmd(n)@self.cmu(n) ) )

        return ham

    
    def get_even_ham(self):
        return self.ham[self.index_even,:][:,self.index_even]
    
    def get_odd_ham(self):
        return self.ham[self.index_odd,:][:,self.index_odd]
    
    def get_eigvals(self, k=1, detunings=None):
        
        if detunings is None:
            detunings = [0 for i in range(self.ndots)]
        assert len(detunings) == self.ndots, "Error: Wrong number of detunings"
        
        ham_det = detunings[0]*self.dia(sparse.diags([0, 1, 1, 2]), 0)
        for n in range(1, self.ndots):
            ham_det += detunings[n]*self.dia(sparse.diags([0, 1, 1, 2]), n)

        detune_even = sparse.csc_matrix(ham_det)[self.index_even,:][:,self.index_even]
        detune_odd = sparse.csc_matrix(ham_det)[self.index_odd,:][:,self.index_odd]
        
        eige = sparse.linalg.eigsh(self.get_even_ham() + detune_even, k=k, return_eigenvectors=False, which='SA')
        eigo = sparse.linalg.eigsh(self.get_odd_ham() + detune_odd, k=k, return_eigenvectors=False, which='SA')
        
        return np.sort(eige), np.sort(eigo)

    def get_eigvecs(self, k=1, detunings=None):
        
        if detunings is None:
            detunings = [0 for i in range(self.ndots)]
            
        assert len(detunings) == self.ndots, "Error: Wrong number of detunings"
        
        ham_det = detunings[0]*self.dia(sparse.diags([0, 1, 1, 2]), 0)
        for n in range(1, self.ndots):
            ham_det += detunings[n]*self.dia(sparse.diags([0, 1, 1, 2]), n)

        detune_even = sparse.csc_matrix(ham_det)[self.index_even,:][:,self.index_even]
        detune_odd = sparse.csc_matrix(ham_det)[self.index_odd,:][:,self.index_odd]
        
        
        eige, vece = sparse.linalg.eigsh(self.get_even_ham() + detune_even, k=k, return_eigenvectors=True, which='SA')
        eigo, veco = sparse.linalg.eigsh(self.get_odd_ham() + detune_odd, k=k, return_eigenvectors=True, which='SA')
        
        return vece[:,np.argsort(eige)], veco[:,np.argsort(eigo)]
    
    def get_eigvals_and_eigvecs(self, k=1, detunings=None):
        
        if detunings is None:
            detunings = [0 for i in range(self.ndots)]
            
        assert len(detunings) == self.ndots, "Error: Wrong number of detunings"
        
        ham_det = detunings[0]*self.dia(sparse.diags([0, 1, 1, 2]), 0)
        for n in range(1, self.ndots):
            ham_det += detunings[n]*self.dia(sparse.diags([0, 1, 1, 2]), n)

        detune_even = sparse.csc_matrix(ham_det)[self.index_even,:][:,self.index_even]
        detune_odd = sparse.csc_matrix(ham_det)[self.index_odd,:][:,self.index_odd]
        
        
        eige, vece = sparse.linalg.eigsh(self.get_even_ham() + detune_even, k=k, return_eigenvectors=True, which='SA')
        eigo, veco = sparse.linalg.eigsh(self.get_odd_ham() + detune_odd, k=k, return_eigenvectors=True, which='SA')
        
        return np.sort(eige), vece[:,np.argsort(eige)], np.sort(eigo), veco[:,np.argsort(eigo)]

