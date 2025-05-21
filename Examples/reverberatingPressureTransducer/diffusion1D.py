'''
FDM Backward Euler for one-dimensional diffusion equation.
Calculate D_e and K_d via optimization.

Run as:

    python diffusion1D.py -i <inputFile_*.yml>

To skip modeling step and just plot only:

    python diffusion1D.py -p -i <inputFile_*.yml>

or

    python diffusion1D.py -pi <inputFile_*.yml>


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
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
sys.path.append('/project/gas_seepage/jportiz/scripts/pyDiffusionFDM')
from TDMAsolver import TDMAsolver
from tools import sci_notation

#  #The following allows blank values in *.yml inputFile (doesn't work yet)
#  SafeLoader.add_constructor(
    #  type(None),
    #  lambda loader, value: loader.constructor_scalar(u'tag:yaml.org,2002:null', '')
#  )
#  SafeDumper.add_representer(
    #  type(None),
    #  lambda dumper, value: dumper.represent_scalar(u'tag:yaml.org,2002:null', '')
#  )
#----Specific Functions for Pressure Diffusion Problem------- 
def pa_to_psi(pressure_in_pa):
    return pressure_in_pa * (145.038/1.e6)

def psi_to_pa(pressure_in_psi):
    return pressure_in_psi * 6894.76

def mpa_to_psi(pressure_in_mpa):
    return pressure_in_mpa *145.038

def psi_to_mpa(pressure_in_psi):
    return pressure_in_psi*0.00689476

def psi_to_pa(pressure_in_psi):
    return pressure_in_psi*0.00689476*1e6


def m_to_ft(meters):
    return meters/0.3048
#------------------------------------------------------------ 


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


def diffusion_model(inputDict):
#  def diffusion_model(inputFileName):
    '''
    Objective function to be minimized.
    Euler Backward implicit diffusion model (FDM).
    '''
    #-------------------------------------------------- 
    #  PARAMETERS 
    #-------------------------------------------------- 
    #  # Read all params from inputFile 
    #  inp   = read_inputs(inputFileName)
    # Read all params from inputDict (inputFile read outside of function)
    inp = inputDict
    # Assign params to vars
    phi   = inp['phi']
    L     = inp['L']
    D     = inp['D']
    #  # If flag triggers calculation of D from list of parameters
    #  if type(D)==list:
        #  print('Calculating diffusity from list of provided params...')
        #  # Check if all params in list are specified in inputFile
        #  for item in D:
            #  if item not in inp: print('Need to specify {} to calculate D!'.format(item))
            #  # Read in as variables
            #  #  exec(f'{item} = inp["{item}"]')
            #  #  exec(f'{item} = inp[{item}]')
            #  #  globals()[f'{item}'] = inp[f"{item}"]
        #  # Calculate D
        #  #  D = perm / (mu * phi* beta)
        #  D = inp['perm'] / (inp['mu'] * phi * inp['beta'])
        #  print('Done.')


    #-------------------------------------------------- 
    #  I.C.s and B.C.s  
    #-------------------------------------------------- 
    ic_values = inp['initial_conditions']
    ic_all    = ic_values['all']
    bc_types  = inp['bc_types']
    bc_values = inp['bc_values']
    # Does bc_values use a file rather than a value? (i.e., a time series)
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
                # Keep track of final time value

    num_bcs   = len(bc_types)

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
    #  zeta = phi * D * dt / delx**2  #lumped term that shows up in coefficients
    zeta = D * dt / delx**2  #lumped term that shows up in coefficients
    #---- a vector ----
    a = -zeta*np.ones((J,1)).flatten()
    #---- b vector ----
    b = (1+2*zeta)*np.ones((J,1)).flatten()
    #---- c vector ----
    c = -zeta*np.ones((J,1)).flatten()

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


    #-------------------------------------------------- 
    #  LOOP OVER TIME 
    #-------------------------------------------------- 
    uhist_L = np.zeros(len(t)) #save the concentration history near left boundary
    uhist_L[0] = u[0]
    uhist_R = np.zeros(len(t)) #save the concentration history near right boundary
    uhist_R[0] = u[-1]
    # Save concentration at every node
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

    complete_percent = 0. #
    for it in np.arange(1,nt+1):
        time = it*dt
        current_percent_int = (time *100) // t_final
        if current_percent_int > complete_percent:
            #  print('{:.0f}% complete'.format(current_percent_int))
            complete_percent = current_percent_int

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
                rhs[0] = f_1
            else:
                #  f_1 = float(bc_values[0](time)) #interpolate from file
                # !!! NEW !!! Attempt to use last time value if time is beyond file range
                try:
                    f_1 = float(bc_values[0](time)) #interpolate from file
                    rhs[0] = f_1
                except ValueError:  #if can't intepolate above file's time range
                    rhs[0] = rhs[0] #use previous value
            #  rhs[0] = f_1
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
                rhs[-1] = f_1
            else:
                #  f_1 = float(bc_values[-1](time)) #interpolate from file
                # !!! NEW !!! Attempt to use last time value if time is beyond file range
                try:
                    f_1 = float(bc_values[-1](time)) #interpolate from file
                    rhs[-1] = f_1
                except ValueError:  #if can't intepolate above file's time range
                    rhs[-1] = rhs[-1] #use previous value
            #  rhs[-1] = f_1
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
    outfile = join('output', 'press_{}.csv'.format(inp['outfilePrefix']))
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



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                               MAIN                                               #
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__=='__main__':

    cwd = os.getcwd()
    outputdir = join(cwd,'output')

    #-------------------------------------------------- 
    # Define args for user to choose which part(s) of this script to run 
    #-------------------------------------------------- 
    descr ='Run the ``diffusion1D.py`` script.'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('-p', '--plotOnly', action='store_true', help='Skip modeling and just plot using existing output data files.')
    parser.add_argument('-i', '--inputFile', help='Path and name of inputFile.')
    #  parser.add_argument('-i', '--inputFile', action='store_true', help='Name of YAML input file.')
    args = parser.parse_args()
    #-------------------------------------------------- 

    #-------------------------------------------------- 
    # Parse Input File and Insert Values (if Needed)
    #-------------------------------------------------- 
    # Parse inputFile
    # Specify Input file
    # ... using -i commandline arg
    if args.inputFile:
        #Read in all inputs
        print(args.inputFile)
        #  inputFile = join('input',args.inputFile)
        inputFile = args.inputFile
        #  inputs = read_inputs(inputFile)
        #  inputs = read_inputs(join('input',args.inputFile))
    # ... or manually here (not recommended)
    else:
        print('Did not specify an inputFile using -i arg.')
        #  inputFile = 'input/simple_inputFile.yml'
        # Default input file name
        inputFile = 'input/inputFile.yml'
        #  inputs    = read_inputs(inputFile)
    inputs    = read_inputs(inputFile)


    #-------------------------------------------------- 


    #-------------------------------------------------- 
    # Define Other Values Needed for Problem 
    #-------------------------------------------------- 
    transducer_limit_psi = 20000. #[psi]

    #  porosity_values = {0.10: 'blue',
                       #  #  0.20: 'green',
                       #  #  0.30: 'red',
                      #  }
    porosity_values = {0.10: {'color':'blue',
                              'linestyle': '-'},
                       0.20: {'color':'green',
                              'linestyle': '--'},
                       0.30: {'color':'red',
                              'linestyle': ':'},
                      }
    phi_list = porosity_values.keys()
    #  permeability_values = np.logspace(-17, -14, 10)
    #  permeability_values = np.logspace(-17, -12, 2)  #FOR TESTING
    #  permeability_values = np.logspace(-14, -12, 2)  #FOR TESTING
    permeability_values = np.logspace(-17, -12, 10)  #FOR TESTING


    #-------------------------------------------------- 
    # Loop through Parameters and Run the Diffusion Model
    #-------------------------------------------------- 
    if args.plotOnly==False:
        trans_df_list = []
        for phi in porosity_values.keys():
            print()
            print(f'phi = {phi}')
            print('-'*40)
            times_of_failure = []
            # Iterate over each permeability value
            for k in permeability_values:
                print(f'k = {k}')

                # Read InputFile anew (some dictionary values get overwritten later)
                inputs    = read_inputs(inputFile)
                #  print(inputs)  #DEBUG

                # Calculate hydraulic/pressure diffusivity ( D = D(k,mu,phi,beta) )
                D     = inputs['D']
                # If flag triggers calculation of D from list of parameters
                if type(D)==list:
                    #  print('Calculating diffusity from list of provided params...')
                    # Check if all params in list are specified in inputFile
                    #  for item in D:
                        #  if item not in inputs: print('Need to specify {} to calculate D!'.format(item))
                    # Calculate D
                    #  k = inputs['perm']
                    #  phi = inputs['phi']
                    inputs['k']   = k
                    inputs['phi'] = phi
                    mu = inputs['mu']
                    beta = inputs['beta']
                    #  D = inputs['perm'] / (inp['mu'] * phi * inp['beta'])
                    D = k/(mu*phi*beta)  #hydraulic diffusivity
                    print('Calculated hydraulic diffusivity (alpha) = {:.2g} m2/s'.format(D))
                    #  print('Done.')
                # Replace the value of D in inputs with the calculated value
                inputs['D'] = D


                #================================================== 
                #       RUN MODEL 
                #================================================== 
                # Run the Model
                df = diffusion_model(inputs)

                #-------------------------------------------------- 
                #  Save Pressure Histories (at all nodes)
                #-------------------------------------------------- 
                if not os.path.exists('output'):
                    os.makedirs('output')
                outfile = join('output', 'press_{}phi_{:.0f}_k_{:.1e}.csv'.format(inputs['outfilePrefix'], phi*100, k ))
                df.to_csv(outfile, index=False)

                #-------------------------------------------------- 
                # Calculate t_fail and P_max x=L (if it occurs)
                #-------------------------------------------------- 
                xs = [float(x) for x in df[df.columns[1:]]]
                Lcol = str(inputs['L'])              #x=L
                midLcol = str(xs[int(len(xs)/2)])    #x=L/2

                p_transducer = df[['time',Lcol]]
                p_mid        = df[['time',midLcol]]
                p_max_L = np.nanmax(p_transducer[ Lcol])
                p_max_midL = np.nanmax(df[ midLcol])
                try:
                    t_fail_L = float(p_transducer[ p_transducer[Lcol]>=psi_to_pa(transducer_limit_psi)].iloc[0]['time'])
                    #  t_fail_L = p_transducer[ p_transducer[Lcol]>=psi_to_pa(transducer_limit_psi)].iloc[0]
                except IndexError:
                    t_fail_L = np.nan
                try:
                    t_fail_midL = float(p_mid[ p_mid[midLcol]>=psi_to_pa(transducer_limit_psi)].iloc[0]['time'])
                except IndexError:
                    t_fail_midL = np.nan


                trans_data = [phi, k, t_fail_midL, p_max_midL, t_fail_L, p_max_L]
                trans_df_list.append(trans_data)

    # Create DataFrame of transducer results and Save
    table_name = 'table_transducer_data.csv'
    if args.plotOnly==False:
        trans_df = pd.DataFrame( trans_df_list, columns= ['phi', 'k', 't_fail_s(L/2)', 'P_max_Pa(L/2)','t_fail_s(L)', 'P_max_Pa(L)'] )
        #  trans_df = pd.DataFrame( trans_df_list, columns= ['phi', 'k', 't_fail_s', 'P_max_Pa', 'P_max_psi'] )
        trans_df.to_csv( join('output',table_name),index=False, na_rep='NaN')



    #================================================== 
    #       PLOTTING
    #================================================== 

    #----------------------------------------------
    #  Plot Failure Times 
    #----------------------------------------------
    print('Plotting t_fail...')
    trans_df = pd.read_csv( join('output',table_name) )
    # Plot a Line for each porosity value
    fig, ax = plt.subplots(1)
    # Load this DataFrame if not performing search again
    for phi in porosity_values.keys():
        dfss = trans_df.copy()[ trans_df['phi'] == phi ]
        perm_values  = dfss['k']
        times_of_failure = dfss['t_fail_s(L)'] #[s] @x=L
        # Plot the results for this porosity value
        ax.plot(perm_values, np.asarray(times_of_failure), color=porosity_values[phi]['color'], marker='o', label=r'$\phi$ = {:.0f}\%'.format(100*phi))
        #  if 'incompressible' not in cwd:
            #  ax.plot(perm_values, np.asarray(times_of_failure)/60., color=porosity_values[phi]['color'], marker='o', label=r'$\phi$ = {:.0f}\%'.format(100*phi))
        #  else:
            #  ax.plot(perm_values, np.asarray(times_of_failure), color=porosity_values[phi]['color'], marker='o', label=r'$\phi$ = {:.0f}\%'.format(100*phi))

    # Formatting the plot
    ax.set_xscale('log')
    #  ax.set_yscale('log')
    ax.set_xlabel(r"Permeability, $k$ [m$^2$]")
    #  plt.ylabel("Time of failure, $t_{{\mathrm{{fail}}}}$ [s]")
    #  if 'incompressible' not in cwd: ax.set_ylabel("$t_{{\mathrm{{fail}}}}$ [min]")
    #  else:                           ax.set_ylabel("$t_{{\mathrm{{fail}}}}$ [s]")
    ax.set_ylabel("$t_{{\mathrm{{fail}}}}$ [s]")
    # Second y-axis in hrs
    ax2 = ax.twinx()
    mn,mx =  ax.get_ylim()
    ax2.set_ylim(mn/60., mx/60.)
    #  if 'incompressible' not in cwd: ax2.set_ylabel(r'$t_{{\mathrm{{fail}}}}$ [h]',rotation=270,va='bottom')
    #  else:                           ax2.set_ylabel(r'$t_{{\mathrm{{fail}}}}$ [min]',rotation=270,va='bottom')
    ax2.set_ylabel(r'$t_{{\mathrm{{fail}}}}$ [min]',rotation=270,va='bottom')
    #  ax2.set_yscale('log')
    # Properties
    plt.title(r"Time of Sensor Failure ($t_{{\mathrm{{fail}}}}$) vs Permeability ($k$)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig( join(outputdir, 'transducer_failure.pdf') )
    plt.close('all')
    print('Done')


    #----------------------------------------------
    #  Pressure Time Series
    #----------------------------------------------
    print('Plotting pressure timeseries...')
    #  x_locs = [0.0, inputs['L']/2., inputs['L']]
    #  x_locs = [0.0, 0.15415172413793105, inputs['L']]
    fig, (ax1,ax2) = plt.subplots(2,figsize=(12,10),sharex=False)
    cmap_set  = plt.cm.plasma(np.linspace(0.2, 0.95, len(permeability_values)))  # color map for permeability lines 

    #----------------------------------------------
    # Porosity and Permeability loop
    #----------------------------------------------
    counter = -1
    for phi in porosity_values.keys():
        ls = porosity_values[phi]['linestyle']  #specific linestyle for this porosity
        counter+=1
        # Iterate over each permeability value
        for j,k in enumerate(permeability_values):
            datafile = join('output', 'press_{}phi_{:.0f}_k_{:.1e}.csv'.format(inputs['outfilePrefix'], phi*100, k ))
            # Read in full nodal pressure data 
            df = pd.read_csv(datafile)

            #-- [1] -- Inlet Pressure (x=0)
            if counter==0:
                x_loc = 0.0
                label=r'$k =$ {} m$^2$'.format(sci_notation(k,1))
                x,y = df['time'], df[str(x_loc)]
                #  ax1.plot(x,y, color='k', ls=ls, label=r'$k =$ {} m$^2$'.format(sci_notation(k,1)))
                ax1.plot(x,y, color='k', ls=ls, label=label)
                ax1.set_title(r'$\Delta P$ @ inlet $x=0$ ft')
            #-- [2] -- Transducer Pressure (x=L)
            # (x=0) Plot the late-time inlet pressure for comparison of signal
            if counter==0:
                late_inds = df[df['time']>0.03].index
                x_loc = 0.0
                label=r'$P(x=0)$ (late-time)'
                x,y = df['time'].loc[late_inds], df[str(x_loc)].loc[late_inds]
                ax2.plot(x,y, color='k', ls=ls, label=label)
            # (x=L)
            x_loc = inputs['L']
            label=r'$k =$ {} m$^2$'.format(sci_notation(k,1))
            x,y = df['time'], df[str(x_loc)]
            #  lines = ax2.plot(x,y, ls=ls, label=r'$k =$ {:.1e} m$^2$'.format(m_to_ft(x_loc)))
            lines = ax2.plot(x,y, ls=ls, c=cmap_set[j], label=label)
            ax2.set_title(r'$\Delta P$ @ transducer ($x=1$ ft)')
    #----Legends-----------
    #---- Put Legends outside of Plot
    #  # Shrink current axis by 20%
    #  box1 = ax1.get_position()
    #  ax1.set_position([box1.x0, box1.y0, box1.width * 0.50, box1.height])
    #  box2 = ax2.get_position()
    #  ax2.set_position([box2.x0, box2.y0, box2.width * 0.80, box2.height])

    #---- Shrink Subplots Width
    plt.subplots_adjust(right=0.72)  # Adjust right margin to make room for the legends
    #-- Line Styles (POROSITY)
    line_styles = [ plt.Line2D([0], [0], color='k', linestyle=porosity_values[list(porosity_values.keys())[i]]['linestyle'], label=r'$\phi =$ {:.0f}\%'.format(list(porosity_values.keys())[i]*100)) for i in range(len(porosity_values)) ]
    # Add the first legend (line styles)
    legend1 = ax1.legend(handles=line_styles, loc='upper left', bbox_to_anchor=(1.10, 0.10), fancybox=True, title=r'\textbf{Line Styles (porosity, $\phi$)}')
    ax1.add_artist(legend1)  # Add the first legend manually to the plot
    #-- Line Colors (PERMEABILITY)
    # Create proxy lines for colors (legend 2) using the colormap (loop through all colors)
    color_lines = [ plt.Line2D([0], [0], color=cmap_set[i], linestyle='-', label=r'$k =$ {} m$^2$'.format(sci_notation(permeability_values[i],1))) for i in range(len(permeability_values)) ]
    # Add the second legend (colors)
    legend2 = ax2.legend(handles=color_lines, loc='upper left', bbox_to_anchor=(1.10,0.95), fancybox=True, title=r'\textbf{Colors (permeability, $k$)}')
    # Legend Properties
    legend1.get_frame().set_edgecolor('k')
    legend2.get_frame().set_edgecolor('k')
    legend1.get_frame().set_linewidth(1.0)
    legend2.get_frame().set_linewidth(1.0)
    #----------------------

    #----------------------
    # [1] 2nd y-axis in kPSI
    ax3 = ax1.twinx()
    #  ax3.plot(x, pa_to_psi(y)/1000.,ls='none')
    ax3.set_ylabel(r'$\Delta P$ [kpsi]',rotation=270,va='bottom')
    mn,mx = ax1.set_ylim()
    ax3.set_ylim(pa_to_psi(mn)/1000, pa_to_psi(mx)/1000)
    # [2] 2nd y-axis in kPSI
    ax4 = ax2.twinx()
    #  ax4.plot(x, pa_to_psi(y)/1000.,ls='none')
    ax4.set_ylabel(r'$\Delta P$ [kpsi]',rotation=270,va='bottom')
    mn,mx = ax2.set_ylim()
    ax4.set_ylim(pa_to_psi(mn)/1000, pa_to_psi(mx)/1000)
    #---Properties
    # [1]
    ax1.set_xscale('log')
    #  ax1.set_yscale('log')
    ax1.set_ylabel(r'$\Delta P$ [Pa]')
    ax1.set_xlabel(r'$t$ [s]')
    #  ax1.legend()
    # [2]
    ax2.set_xscale('log')
    #  ax2.set_yscale('log')
    ax2.set_ylabel(r'$\Delta P$ [Pa]')
    ax2.set_xlabel(r'$t$ [s]')
    #  ax2.legend()
    # [1]---Plot line for transducer limit
    ax3.axhline(y=transducer_limit_psi/1000, color='grey', ls='--',lw=1.0)
    # [2]---Plot line for transducer limit
    ax4.axhline(y=transducer_limit_psi/1000, color='grey', ls='--',lw=1.0)
    ax4.text(1.05*ax2.get_xlim()[0], transducer_limit_psi/1000+1., f'transducer limit', ha='left', fontsize=12)
    #----
    plt.tight_layout()
    plt.savefig( join(outputdir, f'plot_{inputs["outfilePrefix"]}.pdf') )
    #  plt.savefig('output/test.pdf')
    plt.close('all')
    print('    Done.')




    #  #----------------------------------------------
    #  #  Transducer Pressure Time Series
    #  #----------------------------------------------
    #  print('Plotting pressure timeseries...')
    #  #  x_locs = [0.0, inputs['L']/2., inputs['L']]
    #  x_locs = [0.0, 0.15415172413793105, inputs['L']]
    #  fig, ax1 = plt.subplots(1,figsize=(6,4),sharex=False)
#  
    #  #----------------------------------------------
    #  # Porosity and Permeability loop
    #  #----------------------------------------------
    #  for phi in porosity_values.keys():
        #  # Iterate over each permeability value
        #  for k in permeability_values:
            #  datafile = join('output', 'press_{}phi_{:.0f}_k_{:.1e}.csv'.format(inputs['outfilePrefix'], phi*100, k ))
            #  # Read in full nodal pressure data 
            #  df = pd.read_csv(datafile)
#  
            #  for x_loc in x_locs:
                #  #-- [1] --
                #  #  x,y = df['time'], df[str(inputs['L'])]
                #  #  x,y = df['time'], df['0.050800000000000005']
                #  #  x,y = df['time'], df['0.0']
                #  x,y = df['time'], df[str(x_loc)]
                #  ax1.plot(x,y, label=r'$x =$ {:.1f} ft'.format(m_to_ft(x_loc)))
    #  # 2nd y-axis in kPSI
    #  ax9 = ax1.twinx()
    #  ax9.plot(x, pa_to_psi(y)/1000.,ls='none')
    #  ax9.set_ylabel(r'$\Delta P$ [kpsi]',rotation=270,va='bottom')
    #  mn,mx = ax1.set_ylim()
    #  ax9.set_ylim(pa_to_psi(mn)/1000, pa_to_psi(mx)/1000)
    #  #---Properties
    #  ax1.set_xscale('log')
    #  ax1.set_yscale('log')
    #  ax1.set_ylabel(r'$\Delta P$ [Pa]')
    #  ax1.set_xlabel(r'$t$ [s]')
    #  #  ax1.set_title('Pressure at transducer ($x = $1 ft)')
    #  ax1.set_title('Pressure Time Series')
    #  ax1.legend()
    #  #---Plot line for transducer limit
    #  ax9.axhline(y=transducer_limit_psi/1000, color='grey', ls='--',lw=1.0)
    #  ax9.text(1.05*ax1.get_xlim()[0], transducer_limit_psi/1000+5., f'transducer limit', ha='left', fontsize=12)
    #  #----
    #  plt.tight_layout()
    #  plt.savefig( join(outputdir, f'plot_{inputs["outfilePrefix"]}.pdf') )
    #  #  plt.savefig('output/test.pdf')
    #  plt.close('all')
    #  print('    Done.')

    #  #----------------------------------------------
    #  #  Pressure Profile at Several Time Slices 
    #  #----------------------------------------------
    #  print('Plotting pressure profiles at several time snapshots...')
    #  #  times = [3e-3, 3e-2, 3e-1, 3e0, 3e1, 3e2]
    #  all_times = np.asarray(df['time'])
    #  #  plot_times = all_times[::10000]
    #  # Manually choose
    #  dt = inputs['dt'] #[s]
    #  #  plot_times = [dt*10, dt*100, dt*1000, 10., 100., 200. ,300., 400., 500.]
    #  plot_times = all_times[::50]
    #  #  # Try just early times
    #  #  all_times = np.asarray(df['time'])
    #  #  early_times = all_times[ all_times<=2.0]
    #  #  plot_times = early_times[::1000]
    #  fig, ax1 = plt.subplots(1,figsize=(6,4),sharex=False)
    #  for time in plot_times:
        #  #-- [1] --
        #  xs = [float(x) for x in df[df.columns[1:]]]
        #  x,y = xs, np.asarray(df[ df['time']==time ])[0][1:]
        #  ax1.plot(x,y, label=r'$t =$ {} s'.format(time))
        #  ax1.set_ylabel(r'$\Delta P$ [Pa]')
        #  ax1.set_xlabel(r'$x$ [m]')
        #  #----
    #  #  ax1.set_yscale('log')
    #  # 2nd y-axis in kPSI
    #  ax9 = ax1.twinx()
    #  mn,mx =  ax1.get_ylim()
    #  ax9.plot(x, pa_to_psi(y)/1000.,ls='none')
    #  ax9.set_ylim(pa_to_psi(mn)/1000., pa_to_psi(mx)/1000.)
    #  #  ax9.set_yscale('log')
    #  #---Plot line for transducer limit
    #  ax9.axhline(y=transducer_limit_psi/1000, color='grey', ls='--',lw=1.0)
    #  #  ax9.text(1.05*ax1.get_xlim()[0], transducer_limit_psi/1000+5., f'transducer limit', ha='left', fontsize=12)
    #  ax9.set_ylabel(r'$\Delta P$ [kpsi]',rotation=270,va='bottom')
    #  ax9.set_ylim(pa_to_psi(mn)/1000.,pa_to_psi(mx)/1000.)
    #  plt.tight_layout()
    #  ax1.legend()
    #  plt.savefig( join(outputdir, f'plot_{inputs["outfilePrefix"]}profiles.pdf') )
    #  #  plt.savefig('output/test.pdf')
    #  plt.close('all')
    #  print('    Done')

    #----------------------------------------------
    # Plot p_max at a couple locations
    #----------------------------------------------
    #  trans_df = pd.read_csv( join('output',table_name) )

    fig, ax1 = plt.subplots(1)
    #  for phi,color in porosity_values.items():
    for phi in porosity_values.keys():
        color = porosity_values[phi]['color']
        # Plot Pmax at L/2
        x, y = trans_df[ trans_df['phi']==phi]['k'], trans_df[ trans_df['phi'] ==  phi]['P_max_Pa(L/2)']
        ax1.plot(x,y, c=color, ls='--')
        # Plot Pmax at L 
        x, y = trans_df[ trans_df['phi']==phi]['k'], trans_df[ trans_df['phi'] ==  phi]['P_max_Pa(L)']
        ax1.plot(x,y, c=color, ls='-', label=r'$\phi = ${:.0f}\%'.format(phi*100.))
    # 2nd y-axis in kPSI
    ax9 = ax1.twinx()
    mn,mx=ax1.get_ylim()
    #  ax9.plot(x, pa_to_psi(y)/1000.,ls='none')
    ax9.set_ylim(pa_to_psi(mn), pa_to_psi(mx))
    ax9.set_ylabel(r'$\Delta P$ [kpsi]',rotation=270,va='bottom')
    #---- Properties
    ax1.set_xlabel(r'$k$ [m$^2$]')
    ax1.set_ylabel(r'$\Delta P_{{\mathrm{{max}}}}$ [Pa]')
    ax1.set_xscale('log')
    ax1.legend()
    plt.savefig(join('output', 'max_pressures_plot.pdf'))
    plt.close('all')




