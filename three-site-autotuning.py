#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:07:27 2024

@author: jacob
"""

import sys
from os import cpu_count
import numpy as np
from qdarray.sparse import DotArray
from optimisation.lossfunctions import EsplitLoss, MajoranaQuality
from optimisation.parallel_cma import parallel_cma


runid = int(sys.argv[1])
losstype = sys.argv[2]
sigma0 = float(sys.argv[3])
L2 = float(sys.argv[4])

sensor_coupling = 0.5
if len(sys.argv) == 6:
    sensor_coupling = float(sys.argv[5])


print("Loss:", losstype, flush=True)
print("Sigma0:", sigma0, flush=True)
print("L2:", L2, flush=True)

D = 1.0
TS = 0.5*D
TSO = 0.2*TS
EZ = 1.5*D
U = 5.0*D

TD = sensor_coupling*TS
TSOD = 0.2*TD

print("td:", TD, flush=True)

pmm_array = DotArray(ndots=6, energies=[0, 0, 0, 0, 0, 0],
                    hoppings = [[TD, TSOD], [TS, TSO],[TS, TSO],[TS, TSO],[TS, TSO]],
                    zeeman = [[0,EZ], [0,EZ], [0,0], [0,EZ], [0,0], [0,EZ]],
                    coulomb = [U, U, 0, U, 0, U],
                    proxgap = [0, 0, D, 0, D, 0])


pmm_array_ref = DotArray(ndots=5, energies=[0, 0, 0, 0, 0],
                    hoppings = [[TS, TSO],[TS, TSO],[TS, TSO],[TS, TSO]],
                    zeeman = [[0,EZ], [0,0], [0,EZ], [0,0], [0,EZ]],
                    coulomb = [U, 0, U, 0, U],
                    proxgap = [0, D, 0, D, 0])

sweep = D*np.hstack([np.linspace(-10, -EZ-U-.25*D, 20),
                 np.linspace(-EZ-U-.25*D, -EZ-U+.25*D, 50),
                 np.linspace(-EZ-U+.25*D, -.25*D, 20),
                 np.linspace(-.25*D, .25*D, 50),
                 np.linspace(.25*D, 4, 20)]).flatten()



#options={'timeout':5*60*60,'popsize':cpu_count(), 'maxiter':200, 'bounds':[-0.8*D, 0.8*D], 'tolflatfitness':5}
options={'timeout':5*60*60,'popsize':cpu_count(), 'maxiter':200, 'tolflatfitness':5}
print("No bounds")
#starting_point = 0.1*np.random.uniform(-1,1,size=5)
starting_point = np.zeros(5)


l2type = 'std' # 'std', 'off'

print("Type of L2:", l2type)

loss = EsplitLoss(pmm_array, sweep, init_point=starting_point, l2=L2, l2_type=l2type)
fmin = None
if losstype=='prada_clarke':
    fmin = loss.prada_clarke
elif losstype=='max_esplit':
    fmin = loss.max_esplit


quality = MajoranaQuality(pmm_array_ref)
oms = {'dE': quality.dE, 'MP': quality.MP}

parallel_cma(fmin, starting_point, sigma0, runid, options, other_metrics=oms)

