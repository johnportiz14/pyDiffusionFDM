'''
FDM Backward Euler for one-dimensional diffusion equation.
Calculate D_e and K_d via optimization.
'''
import os,sys
sys.path.append('/project/gas_seepage/jportiz/scripts')
try: import logger
except ModuleNotFoundError: pass
from subprocess import call
import numpy as np
from os.path import join
from decimal import Decimal
import pandas as pd
import matplotlib.pyplot as plt
#  from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
#  from scipy.optimize import minimize
#  from scipy.optimize import differential_evolution
#  from tools import sci_notation
import math
import matplotlib
#  from matplotlib import tri
import yaml
#  from yaml import SafeDumper
from yaml import SafeLoader
import argparse
from TDMAsolver import TDMAsolver
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
#-----------------------------------------------------
#MATPLOTLIBRC PLOTTING PARAMETERS
# Load up sansmath so that math --> helvetica font
# Also need to tell tex to turn on sansmath package
plt.rcParams['text.latex.preamble'] = [
    r'\usepackage{helvet}',
    r'\usepackage{sansmath}',
    r'\sansmath']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
#  plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.labelweight']=u'normal'
plt.rcParams['agg.path.chunksize'] = 10000  #agg has a hardcoded limit on points
#-----------------------------------------------------

#  #The following allows blank values in *.yml inputFile (doesn't work yet)
#  SafeLoader.add_constructor(
    #  type(None),
    #  lambda loader, value: loader.constructor_scalar(u'tag:yaml.org,2002:null', '')
#  )
#  SafeDumper.add_representer(
    #  type(None),
    #  lambda dumper, value: dumper.represent_scalar(u'tag:yaml.org,2002:null', '')
#  )


def read_inputs(inputFileName):
    '''
    Generic read of (YAML) input file.
    Parameters must have same names as fdm_model.'''
    with open(inputFileName,'r') as f:
        inp = yaml.safe_load(f)
        #  inp = yaml.safe_load(f, default_flow_style=False)
    return inp

def lookup_inputs(inputs_dict, key):
    '''
    Lookup individual values for keys in inputs dictionary.
    Does not work if entry is itself a dictionary.
    '''
    for k in inputs_dict:
        if k==key:
            value=inputs_dict[k]
    return value

#  def from_file(fileName):
    #  df = pd.read_csv(filename)
    #  return df


def diffusion_model(inputFileName):
    '''
    Objective function to be minimized.
    Euler Backward implicit diffusion model (FDM).
    '''
    #-------------------------------------------------- 
    #  PARAMETERS 
    #-------------------------------------------------- 
    # Read all params from inputFile 
    inp   = read_inputs(inputFileName)
    # Assign params to vars
    D     = inp['D']
    phi   = inp['phi']
    L     = inp['L']
    #  rho_b = inp['rho_b']
    #  if type(rho_b) is not float:
        #  rho_b = float(rho_b)
    #  A     = inp['A']
    #  A = 0.003167  #[m2] core area
    #  rho_b = 2.57e3  #[kg/m3] tuff bulk density (2.36-2.57 g/cc)
    #  L = 0.051  #[m] core length (5.1cm)
    #  #  # Fixed parameters
    #  #  q_sample = 6.75e-10  #[m3/s] #outlfow rate in sampling chamber
    #  V_in = 5e-4  #[m3] spike chamber volume (500 mL)
    #  V_out = 7.5e-5  #[m3] sampling chamber volume (75 mL)

    #-------------------------------------------------- 
    #  I.C.s and B.C.s  
    #-------------------------------------------------- 
    ic_values = inp['initial_conditions']
    ic_all    = ic_values['all']
    bc_types  = inp['bc_types']
    bc_values = inp['bc_values']
    # Does bc_values use a file rather than a value?
    if any(isinstance(b,str) for b in bc_values):
        # If yes, create a time interpolation object
        for i,b in enumerate(bc_values):
            if type(b)==str:
                # Read in file to DataFrame
                df = pd.read_csv(b)
                # Create an interpolation object
                #  f0 = interp1d(df['time'], df.iloc[:,-1], fill_value='extrapolate')
                f0 = interp1d(df['time'], df.iloc[:,-1])
                # Replace interp object in dictionary
                bc_values[i] = f0


    num_bcs   = len(bc_types)



    #  #  K_d = 1e-2#0.#8e-8#8e-4    #[?][m3/kg] soil adsorption coefficient
    #  R = (1+rho_b*K_d / phi)  #[-] retardation factor
    #  #  D_e = 2e-5#2.84e-7 #1e-2  #1 #[m2/s] diffusivity

    #-------------------------------------------------- 
    #  DOMAIN 
    #-------------------------------------------------- 
    #  J=201  #number of nodes
    #  delx=L/(J-1)    #[m] grid spacing
    # If specified dx spacing
    if 'dx' in inp:
        delx=inp['dx']     #[m] grid spacing
        J = int(L/delx+1)  #number of nodes
    # If specified number of nodes
    elif 'nx' in inp:
        J = inp['nx']   #number of nodes
        delx=L/(J-1)    #[m] grid spacing
    # If both are specified, use number of nodes
    elif 'dx' and 'nx' in inp:
        J = inp['nx']   #number of nodes
        delx=L/(J-1)    #[m] grid spacing
    x=np.linspace(0.,L,J)   #create x node vector 

    #-------------------------------------------------- 
    #  TIME 
    #-------------------------------------------------- 
    #  dt = 10. #1.0  #[s]
    dt = inp['dt']  #[s]
    t_initial = inp['t_initial']   #[s]
    t_final   = inp['t_final']     #[s]
    #  # r1 = q_sample/V_in*120/3480  #[1/s] sampling rate in spike chamber
    #  r2 = q_sample/V_out*360/3240  #[1/s] sampling rate in sampling chamber
    #  nt=fix(72000/dt) #number of time-steps to get to 20 hours
    #  duration = max(x_obs_outlet)*3600. #[s] get from experimental data 
    duration = t_final - t_initial  #[s]
    nt=math.floor(duration/dt) #number of time-steps to get to t_final 
    t = np.arange(t_initial,t_final+dt,dt)  #time vector

    #-------------------------------------------------- 
    #  INITIALIZE SOLUTION ARRAY 
    #-------------------------------------------------- 
    u = ic_all * np.ones((1,J)).flatten() #[kg/m3] initial condition (concentrations)
    u[0]  = ic_values['left']   #left  initial conc
    u[-1] = ic_values['right']  #right initial conc
    #  u[0] = 1.0  #LEFT initial relative concentration
    #  uplot=[u] #plot initial condition

    #-------------------------------------------------- 
    #  a,b,c VECTORS FOR THOMAS ALGORITHM 
    #-------------------------------------------------- 
    #this could be outside the time
    #loop - for indexing reasons, we are using a(1)=0, c(J)=0; that way
    #the Thomas function is easier to write - these do not occur explicitly
    #in the Thomas algorithm but allow a(2) to line up with b(2), etc.
    #filling in and then overwriting the first and last row for B.C.
    #---Recall----
    # In the tri-diagonal matrix, we have the following in the a,b,c vectors:
    #(@x=0)  a_0 = 0,        b_0 = c1-c2/delx, c_0 = -c2/delx
    #(@x=J)  a_J = -c4/delx, b_J = c3+c4/delx, c_J = 0

    #---------------------------------------- 
    #       INTERIOR NODES                  #
    #---------------------------------------- 
    # (should not be affected by BCs)
    # (boundary nodes will be overwritten separately)
    mu = phi * D * dt / delx**2  #lumped term that shows up in coefficients
    #---- a vector ----
    a = -mu*np.ones((J,1)).flatten()
    #---- b vector ----
    b = (1+2*mu)*np.ones((J,1)).flatten()
    #---- c vector ----
    c = -mu*np.ones((J,1)).flatten()

    #---------------------------------------- 
    #       LEFT BOUNDARY (x=0)             #
    #---------------------------------------- 
    #---- a vector ----
    a[0] = 0.   #(not in matrix)
    # (1st-Type BC: Dirichlet) --> c1=1, c2=0, f_1=specified value of u_1 (in rhs)
    if bc_types[0] == 1:
        #---- b vector ----
        b[0] = 1.
        #---- c vector ----
        c[0] = 0.
    # (2nd-Type BC: Neumann)   --> c1=0, c2=1, f_1=specified value of gradient (in rhs)
    elif bc_types[0] == 2:
        #---- b vector ----
        b[0] = -1/delx
        #---- c vector ----
        c[0] = -1/delx

    #---------------------------------------- 
    #       RIGHT BOUNDARY (x=J)            #
    #---------------------------------------- 
    #---- c vector ----
    c[-1] = 0.   #(not in matrix)
    # (1st-Type BC: Dirichlet) --> c3=1, c4=0, f_1=specified value of u_1 (in rhs)
    if bc_types[-1] == 1:
        #---- a vector ----
        a[-1] = 0.
        #---- b vector ----
        b[-1] = 1.
    # (2nd-Type BC: Neumann)   --> c3=0, c4=1, f_1=specified value of gradient (in rhs)
    elif bc_types[-1] == 2:
        #---- a vector ----
        a[-1] = -1/delx
        #---- b vector ----
        b[-1] = 1/delx

    #  #---- b vector ----
    #  #  b=(1+2*phi*D_e/R*dt/delx**2)*np.ones(J) #interior nodes
    #  b=(1+2*phi*D_e/R*dt/delx**2)*np.ones((J,1)).flatten() #interior nodes
    #  # b(J,1) = 1/delx; #no-flux BC on right
    #  # b(1,1) = 1 + r1*dt + A/V_in*phi*D_e*dt/delx;    #4th-type BC on left
    #  #  b(1,1) = 1; #Dirichlet BC on left (will be time-varying)
    #  #  b(J,1) = 1 + r2*dt + A/V_out*phi*D_e*dt/delx; #4th-type BC on right
    #  b[0] = 1 #Dirichlet BC on LEFT (will be time-varying)
    #  b[-1] = 1 + r2*dt + A/V_out*phi*D_e*dt/delx #4th-type BC on RIGHT
#  
    #  #---- a vector ----
    #  #  a = -(phi*D_e/R*dt/delx**2)*np.ones(J) #define a for interior nodes
    #  a = -(phi*D_e/R*dt/delx**2)*np.ones((J,1)).flatten() #define a for interior nodes
    #  # a(J,1) = -1/delx; #no-flux BC on right
    #  #  a(1,1) = 0.0;      # (not in matrix) left BC
    #  #  a(J,1) = -A/V_out*phi*D_e*dt/delx; #4th-type BC on right
    #  a[0] = 0.0      # (not in matrix) LEFT BC
    #  a[-1] = -A/V_out*phi*D_e*dt/delx; #4th-type BC on RIGHT
#  
    #  #---- c vector ----
    #  #  c=-(phi*D_e/R*dt/delx**2)*np.ones(J) #interior nodes
    #  c=-(phi*D_e/R*dt/delx**2)*np.ones((J,1)).flatten() #interior nodes
    #  #  # c(1,1) = -A/V_in*phi*D_e*dt*delx; #4th-type B.C. at left
    #  #  c(1,1) = 0.; #first type B.C. at left
    #  #  c(J,1) = 0.0; #not sure about this...
    #  c[0] = 0. #first type B.C. at LEFT
    #  c[-1] = 0.0 #(not in matrix) at RIGHT



    #  #-------------------------------------------------- 
    #  #  Inlet B.C.  (Transient Dirichlet)
    #  #-------------------------------------------------- 
    #  # Interpolate experimental concentrations to time series 
    #  f0 = interp1d(x_obs_inlet*3600., y_obs_inlet,fill_value='extrapolate')  #convert time to seconds
    #  y_obs_inlet_i = f0(t)  #use this as left B.C.
    #  # Make any values > 1.0 equal 1.0
    #  y_obs_inlet_i[y_obs_inlet_i>1.0] = 1.0
    #  u[0] = y_obs_inlet_i[0]  # Set the initial conc on left B.C.
    #  #  # Save
    #  #  uhist_L[0] = u[0]


    #-------------------------------------------------- 
    #  LOOP OVER TIME 
    #-------------------------------------------------- 
    uhist_L = np.zeros(len(t)) #save the concentration history near left boundary
    uhist_L[0] = u[0]
    uhist_R = np.zeros(len(t)) #save the concentration history near right boundary
    uhist_R[0] = u[-1]
    #  uhist_L = np.zeros(nt) #save the concentration history near left boundary
    #  uhist_R = np.zeros(nt) #save the concentration history near right boundary
    # Save concentration at every node
    #  uhist_all = np.zeros( (nt+1, len(x)) )
    uhist_all = np.zeros( (len(t), len(x)) )
    uhist_all[0] = u  #@t=0


    # Initialize the right hand side (rhs) vector 
    rhs = np.zeros(J)
    # Handle initial conditions
    # Left Boundary
    rhs[0]=ic_values['left']
    # Interior Nodes
    rhs[1:-1]=u[1:-1]
    # Right Boundary
    rhs[-1]=ic_values['right']

    for it in np.arange(1,nt+1):
        time = it*dt

        #---- Update rhs vector (within time loop)----
        #---------------------------------------- 
        #       INTERIOR NODES                  #
        #---------------------------------------- 
        rhs[1:-1]=u[1:-1]
        #---------------------------------------- 
        #       LEFT BOUNDARY (x=0)             #
        #---------------------------------------- 
        # (1st-Type BC: Dirichlet) --> c3=1, c4=0, f_1=specified value of u_1 
        if bc_types[0]==1:
            if type(bc_values[0]) is float:
                f_1 = bc_values[0]  #specified value
            else:
                f_1 = float(bc_values[0](time)) #interpolate from file
            rhs[0] = f_1
        # (2nd-Type BC: Neumann)   --> c3=0, c4=1, f_1=specified value of gradient
        elif bc_types[0]==2:
            if type(bc_[0]) is float:
                f_1 = bc_values[0]  #specified value
            else:
                f_1 = float(bc_values[0](time)) #interpolate from file
            rhs[0] = f_1
        #---------------------------------------- 
        #       RIGHT BOUNDARY (x=J)            #
        #---------------------------------------- 
        # (1st-Type BC: Dirichlet) --> c3=1, c4=0, f_1=specified value of u_1 
        if bc_types[-1]==1:
            if type(bc_values[-1]) is float:
                f_1 = bc_values[-1]  #specified value
            else:
                f_1 = float(bc_values[-1](time)) #interpolate from file
            rhs[-1] = f_1
        # (2nd-Type BC: Neumann)   --> c3=0, c4=1, f_1=specified value of gradient
        elif bc_types[-1]==2:
            if type(bc_values[-1]) is float:
                f_1 = bc_values[-1]  #specified value
            else:
                f_1 = float(bc_values[-1](time)) #interpolate from file
            rhs[-1] = f_1
        #  #---- Time-varying spike chamber concentration
        #  #  u0 = m*(it*dt) + C_init # time-varying Dirichlet on Left BC
        #  u0 = y_obs_inlet_i[it]
        #  #4th-type BC on Right
        #  uJ = (-A/V_out*phi*D_e*dt/delx)*u[-2] + (1+r2*dt+A/V_out*phi*D_e*dt/delx)*u[-1]

        #---- Call Thomas algorithm
        # (need to manually omit the unused values in each vector)
        #  unext=TDMAsolver(a,b,c,rhs)
        unext = TDMAsolver(a[1:],b,c[:-1],rhs)
        u=unext

        #---------------------
        # Store vals in array
        #---------------------
        # Update solution arrays
        uhist_L[it] = u[0]
        uhist_R[it] = u[-1]
        #  uhist_all[it,:J] = u
        uhist_all[it] = u

    #-------------------------------------------------- 
    #  SAVE CONCENTRATION HISTORIES 
    #-------------------------------------------------- 
    #  #  cols = np.column_stack( (t/3600., uhist_L, uhist_R) )
    #  cols = np.column_stack( (t, uhist_L, uhist_R) )
    #  df = pd.DataFrame(cols, columns=['time', 'C_L', 'C_R'])
    #  if not os.path.exists('output'):
        #  os.makedirs('output')
    #  outfile = join('output', 'concs_{}.csv'.format(inp['outfilePrefix']))
    #  df.to_csv(outfile, index=False)
    # Concs at all Nodes
    t_all = np.arange(t_initial,t_final+dt, dt)
    cols = np.column_stack( (t_all, uhist_all) )
    xval = [str(i) for i in x]
    #  df = pd.DataFrame(cols, columns=['time', 'C_L', 'C_R'])
    df = pd.DataFrame(cols, columns=['time']+xval)
    if not os.path.exists('output'):
        os.makedirs('output')
    outfile = join('output', 'concs_{}.csv'.format(inp['outfilePrefix']))
    df.to_csv(outfile, index=False)

    #  #-------------------------------------------------- 
    #  #  CALCULATE ERROR 
    #  #-------------------------------------------------- 
    #  #  err_spike  = calc_nrmse(t, uhist_L, t, y_obs_inlet_i)
    #  err_spike  = calc_nrmse(t, uhist_L, x_obs_inlet*3600, y_obs_inlet)
    #  err_sample = calc_nrmse(t, uhist_R, x_obs_outlet*3600, y_obs_outlet)
    #  total_err = calc_combined_nrmse(err_spike, w_spike, err_sample, w_sample)
#  
    #  print('  spike_error = {:.4}'.format(err_spike))
    #  print('  sampl_error = {:.4}'.format(err_sample))
    #  print('error = {:.4}'.format(total_err))
#  
    #  # -----------------------------------------------
    #  # Write pars and errors to file (manually for now) 
    #  # -----------------------------------------------
    #  outputdir = join(os.getcwd(), 'output')
    #  # Append a pre-created file
    #  with open(join(outputdir, 'errors', 'errors_{}_sat{}.csv'.format(gas,int(100*sat))), 'a') as f:
        #  outcols = [D_e,K_d,total_err]
        #  l='{},'*(len(outcols)-1)+'{}\n'
        #  f.write(l.format(*outcols))
    #  return total_err
    return df




if __name__=='__main__':

    cwd = os.getcwd()
    outputdir = join(cwd,'output')

    #-------------------------------------------------- 
    # Define args for user to choose which part(s) of this script to run 
    #-------------------------------------------------- 
    descr ='Run the ``diffusion1D.py`` script.'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('-i', '--inputFile', help='Path and name of inputFile.')
    #  parser.add_argument('-i', '--inputFile', action='store_true', help='Name of YAML input file.')

    args = parser.parse_args()
    #  print(args)                      #DEBUG



    #-------------------------------------------------- 

    #================================================== 
    #       RUN MODEL 
    #================================================== 
    # Specify Input file
    # ... using -i commandline arg
    if args.inputFile:
        #Read in all inputs
        print(args.inputFile)
        #  inputFile = join('input',args.inputFile)
        inputFile = args.inputFile
        inputs = read_inputs(args.inputFile)
        #  inputs = read_inputs(join('input',args.inputFile))
    # ... or manually here (not recommended)
    else:
        print('Did not specify an inputFile using -i arg.')
        #  inputFile = 'input/simple_inputFile.yml'
        # Default input file name
        inputFile = 'input/inputFile.yml'
        inputs    = read_inputs(inputFile)

    # Run the Model
    df = diffusion_model(inputFile)


    #================================================== 
    #       PLOTTING
    #================================================== 

    #----------------------------------------------
    #  BREAKTHROUGH TIME SERIES
    #----------------------------------------------
    fig, ax1 = plt.subplots(1,figsize=(6,4),sharex=False)
    #-- [1] --
    ax1.plot(df['time']/3600,df[str(inputs['L'])],marker='o',mfc='none',label='model')
    ax1.set_ylabel(r'$C$')
    ax1.set_xlabel('Time [h]')
    #----
    plt.tight_layout()
    plt.savefig( join(outputdir, f'plot_{inputs["outfilePrefix"]}.pdf') )
    #  plt.savefig('output/test.pdf')
    plt.close('all')



