'''
Extra plot for Case test.
'''
import os,sys
from os.path import join
#  sys.path.append('../..')
sys.path.append(os.environ['REPO'])
sys.path.append(join(os.environ['REPO'],'tools'))
from diffusion1D import read_inputs
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd


#-------------------------------------------------- 
#     READ PARAMS FROM INPUT FILE 
#-------------------------------------------------- 
# Default input file name
inputFile = 'input/inputFile.yml'
inputs    = read_inputs(inputFile)



C_0 = inputs['initial_conditions']['left']
D = inputs['D']
L = inputs['L']
dt = inputs['dt']
dx = inputs['dx']
bc_vals = inputs['bc_values']
xs = np.arange(0,L+dx,dx)
ts = np.arange(inputs['t_initial'], inputs['t_final']+dt,dt) #[s]

#-------------------------------------------------- 
#     READ IN MODEL RESULTS 
#-------------------------------------------------- 
#-- Model
df_mod = pd.read_csv(join('output','concs_.csv'))
#-- Prescribed BCs from file
df_bc = pd.read_csv(bc_vals[0])




#-------------------------------------------------- 
#     PLOT  
#-------------------------------------------------- 
# Plot concentration history @ x=0 and x=L
fig, (ax1,ax2) = plt.subplots(2)
# [1] (x=0)
#-- B.C. FROM FILE
ax1.plot(df_bc['time'],df_bc['C'],ls='',marker='*',color='r',mfc='r', ms=14,label='B.C. from inputFile')
#-- MODEL
x_plot=0.
ax1.plot(df_mod['time'],df_mod[str(x_plot)],ls='',marker='o',color='k',mfc='none', label='model')
ax1.set_title('@ x=0')
# [2] (x=L)
x_plot=L
ax2.plot(df_mod['time'],df_mod[str(x_plot)],ls='',marker='o',color='k',mfc='none')
ax2.set_title('@ x=L')
#  #----Properties
#  ax.set_title(r'@$x$ = {} m, $L$ = {} m'.format(x_plot,L))
#  ax.set_xlabel('time [s]')
#  ax.set_ylabel(r'$C$')
#  #  ax.legend(loc='center right')
ax1.legend()
plt.tight_layout()
plt.savefig(join('output', 'compare_plot_.pdf'))
plt.close('all')
