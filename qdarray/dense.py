"""
Created on Tue Feb 27 13:54:38 2024

@author: Jacob Benestad, Jeroen Danon
"""

import numpy as np
from scipy.linalg import eigh


# GLOBAL VARIABLES

cpu_mat = np.diag( [1,0,1], -1) # Up spin single dot creation matrix
cpd_mat = np.diag( [1,1], -2)   # Down spin single dot creation matrix
cmu_mat = np.diag( [1,0,1], 1)  # Up spin single dot destruction matrix
cmd_mat = np.diag( [1,1], 2)    # Down spin single dot destruction matrix






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
        parities = np.array([1,-1,-1,1])
        for n in range(ndots-1):
            parities = np.kron(np.array([1,-1,-1,1]), parities)
            
            
        self.index_even = np.where(parities==1)[0]
        self.index_odd = np.where(parities==-1)[0]
        
        self.ham = self.make_hamiltonian()
        
    
    # Up spin creation operator
    def cpu(self, n):
        parity_factors = np.diag( (-1)**np.sum( np.array(self.occupations)[:,0:2*(n-1)], axis=1 ) )
        return np.kron( self.pos[n], np.kron( cpu_mat, self.pos[self.ndots-1-n] ) ) @ parity_factors

    # Down spin creation operator
    def cpd(self, n):
        parity_factors = np.diag( (-1)**np.sum( np.array(self.occupations)[:,0:2*(n-1)-1], axis=1 ) )
        return np.kron( self.pos[n], np.kron( cpd_mat, self.pos[self.ndots-1-n] ) ) @ parity_factors

    # Up spin destruction operator
    def cmu(self, n):
        return self.cpu(n).T

    # Down spin destruction operator
    def cmd(self, n):
        return self.cpd(n).T

    # Single dot energy
    def dia(self, h,n):
        return np.kron( self.pos[n], np.kron( h, self.pos[self.ndots-1-n] ) )
    
    
    def make_hamiltonian(self):
        
        # Dot energies
        ham = self.dia(np.diag([0, self.ens[0]+self.EZs[0]/2, self.ens[0]-self.EZs[0]/2, 2*self.ens[0]+self.Us[0]]),0)
        for n in range(1, self.ndots):
            ham += self.dia(np.diag([0, self.ens[n]+self.EZs[n]/2, self.ens[n]-self.EZs[n]/2, 2*self.ens[n]+self.Us[n]]),n)

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
    
    def get_eigvals(self, detunings=None):
        
        if detunings == None:
            detunings = [0 for i in range(self.ndots)]
        assert len(detunings) == self.ndots, "Error: Wrong number of detunings"
        
        ham_det = detunings[0]*self.dia(np.diag([0, 1, 1, 2]), 0)
        for n in range(1, self.ndots):
            ham_det += detunings[n]*self.dia(np.diag([0, 1, 1, 2]), n)

        detune_even = ham_det[self.index_even,:][:,self.index_even]
        detune_odd = ham_det[self.index_odd,:][:,self.index_odd]
        
        eige = eigh(self.get_even_ham() + detune_even, eigvals_only=True)
        eigo = eigh(self.get_odd_ham() + detune_odd, eigvals_only=True)
        
        return eige, eigo

    def get_eigvecs(self, detunings=None):
        
        if detunings == None:
            detunings = [0 for i in range(self.ndots)]
            
        assert len(detunings) == self.ndots, "Error: Wrong number of detunings"
        
        ham_det = detunings[0]*self.dia(np.diag([0, 1, 1, 2]), 0)
        for n in range(1, self.ndots):
            ham_det += detunings[n]*self.dia(np.diag([0, 1, 1, 2]), n)

        detune_even = ham_det[self.index_even,:][:,self.index_even]
        detune_odd = ham_det[self.index_odd,:][:,self.index_odd]
        
        
        eige, vece = eigh(self.get_even_ham() + detune_even, eigvals_only=False)
        eigo, veco = eigh(self.get_odd_ham() + detune_odd, eigvals_only=False)
        
        return vece, veco