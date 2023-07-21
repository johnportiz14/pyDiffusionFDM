'''
Collection of tools and helper functions.
'''
import numpy as np

def calc_rmse(obs,pred):
    rmse = np.sqrt( np.sum( (obs-pred)**2 ) / len(obs) )
    return rmse

