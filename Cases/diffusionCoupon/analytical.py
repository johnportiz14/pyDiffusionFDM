'''
Analytical solution for 1-D Diffusion Coupon experiment.

Included as an additional check (besides the regression test) that the code is
behaving correctly.
'''
import os,sys
from os.path import join
#  sys.path.append('../..')
sys.path.append(os.environ['REPO'])
sys.path.append(join(os.environ['REPO'],'tools'))
from diffusion1D import read_inputs
from tools import calc_rmse
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from glob import glob
import pandas as pd

def ana_infinite(C_0,x,t,D):
    '''
    Analytical solution for diffusion coupon assuming semi-infinite domain
    length.
    '''
    C_xt = C_0*special.erfc(x/(2.*np.sqrt(D*t)))
    C_xt = np.asarray(C_xt)
    return C_xt

def ana_finite(C_0,x,times,L,D,n=100):
    '''
    <<< TRY MAKING MORE GENERAL EVENTUALLLY, INCORPORATE C_0, IMG_SOURCE LOCATION...>>>

    Analytical solution for diffusion with no-flux B.C. on a finite domain
    using a summation of image sources..

    n : int
        Number of summation terms
    '''
    nn = np.arange(0,n+1,1)
    C_profile = []

    # Get conc at single node for all times
    for tt in times:
        uanal = 1 - 4 * np.sum(np.sin((2 * nn + 1) * np.pi * x / (2 * L)) *
                                  np.exp(-(2 * nn + 1) ** 2 * np.pi ** 2 * D * tt / (4 * L ** 2)) /
                                  ((2 * nn + 1) * np.pi))
        C_profile.append(uanal)
    C_profile=np.asarray(C_profile)
    return C_profile


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
C_0 = inputs['initial_conditions']['left']
D = inputs['D']
L = inputs['L']
dt = inputs['dt']
dx = inputs['dx']
xs = np.arange(0,L+dx,dx)
ts = np.arange(inputs['t_initial'], inputs['t_final']+dt,dt) #[s]
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
#     READ IN MODEL RESULTS 
#-------------------------------------------------- 
df_mod = pd.read_csv(join('output','concs_.csv'))

#-------------------------------------------------- 
#     CALCULATE CONCENTRATION PROFILE AT GIVEN TIME 
#-------------------------------------------------- 
t_plot = 3600.
t_ind = int(t_plot/dt)

#---- Infinite domain solution
C_ana_inf = [ana_infinite(C_0=inputs['initial_conditions']['left'],x=xx,t=ts,D=D) for xx in xs]
C_ana_inf=np.asarray(C_ana_inf) #convert to array
#---- Infinite domain solution
C_ana_fin = [ana_finite(C_0=inputs['initial_conditions']['left'],x=xx,times=ts,L=L,D=D,n=100) for xx in xs]
C_ana_fin=np.asarray(C_ana_fin) #convert to array

fig, ax = plt.subplots()
# Model 
x = xs
y = df_mod.iloc[t_ind,1:]; ymod=y
ax.plot(x,y,ls='',marker='o',color='k',mfc='none', label='model')
# Analytical (Infinite domain)
y = C_ana_inf[:,t_ind]; yinf=y
ax.plot(x,y, ls='',marker='*',color='r',mfc='r', label='analytical (infinite domain)')
err_inf = calc_rmse(yinf,ymod)
# Analytical (Finite domain)
y = C_ana_fin[:,t_ind]; yfin=y
ax.plot(x,y,ls='',marker='.',color='b',mfc='b', label='analytical (finite domain)')
err_fin = calc_rmse(yfin,ymod)
#---- Properties
ax.set_title(r'$t$ = {} s'.format(t_plot))
ax.set_xlabel(r'$x$ distance')
ax.set_ylabel(r'$C$')
ax.set_xlim(left=0.0)
ax.set_ylim(0,C_0)
# Display Errors
xt = 0.5; yt=0.7; yv=0.05
ax.text(xt,yt,    'RMSE={:.3f}'.format(err_inf), transform=ax.transAxes,color='r')
ax.text(xt,yt-yv, 'RMSE={:.3f}'.format(err_fin), transform=ax.transAxes,color='b')
ax.legend(loc='upper right')
plt.savefig(join('output', 'compare_ana_plot_1.pdf'))
plt.close('all')

#-------------------------------------------------- 
#     CALCULATE BREAKTHROUGH CONCENTRATIONS  
#-------------------------------------------------- 
# Plot concentration history at x=L
x_plot=L #  C = concentration(C_0=1.,x=x,t=ts*86400.,D=D)
#---- Infinite domain solution
C_ana_inf = ana_infinite(C_0=inputs['initial_conditions']['left'],x=x_plot,t=ts,D=D)
#---- Infinite domain solution
C_ana_fin = ana_finite(C_0=inputs['initial_conditions']['left'],x=x_plot,times=ts,L=L,D=D,n=100)

# Save the analytical concentrations
#---- Get the name of the standard file (*_std)
std_file = glob(join('output','*_std'))[0]
#---- Use the same name, but swap _std for _ana
outfile = std_file.replace('_std','_ana')
#---- Make DataFrame
twocols = np.column_stack( (ts, C_ana_inf) )
df_ana = pd.DataFrame(twocols, columns=['time',str(L)])
df_ana.to_csv(outfile, index=False)




#-------------------------------------------------- 
#     PLOT TO COMPARE 
#-------------------------------------------------- 
fig, ax = plt.subplots()
# Model 
ax.plot(df_mod['time'],df_mod[str(L)],ls='',marker='o',color='k',mfc='none', label='model')
# Analytical (Infinite domain)
ax.plot(ts,C_ana_inf,color='r', label='analytical (infinite domain)')
# Analytical (Finite domain)
ax.plot(ts,C_ana_fin,color='b', label='analytical (finite domain)')
#----Properties
ax.set_title(r'@$x$ = {} m, $L$ = {} m'.format(x_plot,L))
ax.set_xlabel('time [s]')
ax.set_ylabel(r'$C$')
#  ax.legend(loc='center right')
ax.legend()
plt.savefig(join('output', 'compare_ana_plot_2.pdf'))
plt.close('all')
