#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:07:15 2024

@author: Jacob Benestad
"""

import numpy as np
from scipy.linalg import eigh
from scipy import sparse
from scipy.sparse.linalg import eigsh
from concurrent.futures import ProcessPoolExecutor
import cma
import os
import json
import pickle
import sys

import qdarray.dense as QDd
import qdarray.sparse as QDs
from optimisation.lossfunctions import EsplitLoss


def save_es(es,folder):
    string=es.pickle_dumps()
    with open(folder+'saved_es.pkl','wb') as file:
        file.write(string)

def load_es(folder):
    with open(folder+'saved_es.pkl','rb') as file:
        string=file.read()
        es=pickle.loads(string)
    return es


            
class CmaesData():
    def __init__(self):
        self.data = []
    
    def add(self, iteration, coordinate, loss, other_metric):
        
        if type(coordinate) is np.ndarray:
            coordinate = list(coordinate.flatten())
        
        if len(other_metric.items())==0:
            self.data.append({'iteration': iteration,  'coordinate': coordinate, 'loss': loss})
        else:
            dd = {'iteration': iteration,  'coordinate': coordinate, 'loss': loss}
            for key, res in other_metric.items():
                dd[key] = res
            self.data.append(dd)
    
    def save(self, folder):
        with open(folder+"datadict.txt",mode='w') as file_object:
            file_object.write(json.dumps(self.data))
    
    def load(self, folder):
        datadict = None
        with open(folder+'datadict.txt','rb') as file:
            datadict=json.load(file)
        return datadict
    
    


def parallel_cma(fmin, starting_point, sigma0, runid, options, other_metrics=None):
    
    savefolder = 'outcmaes_pmm/' + str(runid) + '/'
    
    options['verb_filenameprefix'] = savefolder
    
    es=cma.CMAEvolutionStrategy(starting_point,sigma0,options)

    es.logger.disp_header()
    
    cmaesdata = CmaesData()

    starting_results = fmin(starting_point)
    
    other_metrics_0 = {}
    for key, metric in other_metrics.items():
        other_metrics_0[key] = metric(starting_point)
    
    cmaesdata.add(0, starting_point, starting_results, other_metrics_0)
    
    
    
    iteration=1
    while not es.stop(ignore_list=['tolfun']):
        solutions=es.ask()
        solutions = [solutions[i]+starting_point for i in range(len(solutions))]
    
        with ProcessPoolExecutor(options['popsize']) as executor:
            results = list(executor.map(fmin, solutions))
            results = np.array(results).astype(float)
            
            other_metrics_res = {}
            
            
            for key, metric in other_metrics.items():
                other_metrics_res[key] = list(executor.map(metric, solutions))
            
            
            if not es.countiter%10:
                ss = str(solutions[np.argmin(results)]-starting_point)
                for key, res in other_metrics_res.items():
                    ss += ' ' + key + ': ' + str(res[np.argmin(results)])     
                print(ss, flush=True)
    
            for i in range(len(results)):
                other_metrics_i = {}
                for key, res in other_metrics_res.items():
                    other_metrics_i[key] = res[i]
                cmaesdata.add(iteration, solutions[i], results[i], other_metrics_i)
    
    
        es.tell(solutions,results)
        es.logger.add()
        es.disp()
        iteration+=1
    
    es.result_pretty()[0][0]
    
    #save the es instance
    save_es(es, savefolder)
    
    #save the datadict
    cmaesdata.save(savefolder)