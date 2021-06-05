from random import random
import numpy as np

def sim_e(no_iterations):
    
    e_total = 0
    
    for i in range(1, no_iterations):
        
        r = 0
        n_trials = 0
        while r < 1:
            r += random()
            n_trials += 1
            
        e_total += n_trials
        
    e_val = e_total/no_iterations
    print(e_val)