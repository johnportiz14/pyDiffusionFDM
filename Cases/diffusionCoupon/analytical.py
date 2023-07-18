'''
Analytical solution for 1-D Diffusion Coupon experiment.

Included as an additional check (besides the regression test) that the code is
behaving correctly.
'''
import os,sys
from os.path import join
sys.path.append('../..')
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from itertools import cycle
from diffusion1D import read_inputs
from glob import glob
import pandas as pd

def concentration(C_0,x,t,D):
    C_xt = C_0*special.erfc(x/(2.*np.sqrt(D*t)))
    return C_xt

#-------------------------------------------------- 
#     READ PARAMS FROM INPUT FILE 
#-------------------------------------------------- 
# Default input file name
inputFile = 'input/inputFile.yml'
inputs    = read_inputs(inputFile)



#  D = 2.e-6 #cm2/s
#  t_array = [5,10,20] #days
#  L = 10 #cm (length of column)
#  dx = 0.1
#  xs = np.arange(0,L+dx,dx)
D = inputs['D']
L = inputs['L']
dt = inputs['dt']
dx = inputs['dx']
xs = np.arange(0,L+dx,dx)

#  #-------------------------------------------------- 
#  #     (b)
#  #-------------------------------------------------- 
#  # Plot the instantaneous concentration profile C/C_0 within the coupon
#  # from x=0 to x=10cm at t=5,10, and 20 days
#  lines = ["-","--","-.",":"]
#  linecycler = cycle(lines)
#  
#  fig,(ax1,ax2) = plt.subplots(2,figsize=(8,12))
#  for time in t_array:
    #  C = concentration(C_0=1.,x=xs,t=time*86400.,D=D)
    #  ax1.plot(xs,C,next(linecycler),color='k',label='t = {} d'.format(time))
    #  ax1.set_xlabel(r'$x$ [cm]')
    #  ax1.set_ylabel(r'$C/C_0$')
    #  ax1.legend()

#-------------------------------------------------- 
#     CALCULATE BREAKTHROUGH CONCENTRATIONS  
#-------------------------------------------------- 
# Plot concentration history at x=L
ts = np.arange(inputs['t_initial'],inputs['t_final']+dt, dt) #[s] 
#  C = concentration(C_0=1.,x=x,t=ts*86400.,D=D)
C = concentration(C_0=inputs['initial_conditions']['left'],x=L,t=ts,D=D)

# Save the analytical concentrations
#---- Get the name of the standard file (*_std)
std_file = glob(join('output','*_std'))[0]
#---- Use the same name, but swap _std for _ana
outfile = std_file.replace('_std','_ana')
#---- Make DataFrame
twocols = np.column_stack( (ts, C) )
df_ana = pd.DataFrame(twocols, columns=['time',str(L)])
df_ana.to_csv(outfile, index=False)

#-------------------------------------------------- 
#     READ IN MODEL RESULTS 
#-------------------------------------------------- 
df_mod = pd.read_csv(join('output','concs_.csv'))


#-------------------------------------------------- 
#     PLOT TO COMPARE 
#-------------------------------------------------- 
fig, ax = plt.subplots()
# Model 
ax.plot(df_mod['time'],df_mod[str(L)],color='k', label='model')
# Analytical
ax.plot(ts,C,color='r', label='analytical')
ax.set_xlabel('time [s]')
ax.set_ylabel(r'$C$')
#  ax.legend(loc='center right')
ax.legend()
plt.savefig(join('output', 'compare_ana_plot_.pdf'))
plt.close('all')
