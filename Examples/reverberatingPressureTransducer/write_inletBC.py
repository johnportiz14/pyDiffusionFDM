import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
from scipy.interpolate import interp1d

def mpa_to_psi(pressure_in_mpa):
    return pressure_in_mpa *145.038

def pa_to_psi(pressure_in_pa):
    return pressure_in_pa *145.038/1e6

def psi_to_mpa(pressure_in_psi):
    return pressure_in_psi*0.00689476

def psi_to_pa(pressure_in_psi):
    return pressure_in_psi*0.00689476*1e6

# Gaussian function
def gaussian(t, amplitude, mean, std_dev, baseline=0):
    return baseline + amplitude * np.exp(-0.5 * ((t - mean) / std_dev) ** 2)

transducer_limit = 20000. #[psi] transducer pressure limit (for plotting)

# Time settings
time_start = 0  # seconds
time_end = 30e-3  # 30 ms = 30 * 10^-3 seconds
#  n_points = 1000  # Number of time points
#  time = np.linspace(time_start, time_end, n_points)
delta_t = 3.e-4#[s]
time = np.arange(time_start, time_end+delta_t, delta_t)

#------------------------------------------------------------ 
# Load late-time cavity pressures from CSV
#------------------------------------------------------------ 
path = '/lclscratch/jportiz/projects/gas_seepage/lynm/instrumentation/preDesignReview-2024/design_1-indirectPressure/sampling_B/lhs-03/analysis/median_cavity_pressures.csv'
p_cav = pd.read_csv(path)
# Set the first time to 0 seconds
p_cav['t_s'] = p_cav['t_s'] - p_cav['t_s'].iloc[0]
# Convert pressure to delta P by subtracting initial P
p_initial = 0.08 #[MPa]
# Interpolate to uniform timesteps from earlier
t_i = np.arange(p_cav['t_s'].iloc[0], p_cav['t_s'].iloc[-1], delta_t)
f0 = interp1d(p_cav['t_s'], p_cav['P_MPa_median'])
p_i = f0(t_i)
# Create new DataFrame with interpolate late-time pressures
p_late = pd.DataFrame(np.column_stack([t_i,p_i]), columns=['t_s','P_MPa'])
p_late['P_MPa'] -= p_initial
# Now add time so that it late pressure begins after the reverberations
t_end_reverb = time[-1] #[s]
p_late['t_s'] = p_late['t_s'] + t_end_reverb + delta_t
# Add a column converting MPa to Pa
p_late['P_Pa'] =  p_late['P_MPa']*1e6
# Add a column converting MPa to psi
p_late['P_psi'] = mpa_to_psi( p_late['P_MPa'] )
#------------------------------------------------------------ 

# Main pulse parameters
main_pulse_amp     = psi_to_pa(150e3)  #[Pa] 150 kPSI
main_pulse_mean    = 5e-3  #[s] Peak at 5 ms
main_pulse_std_dev = 1e-3  #[s] Pulse width
main_pulse_baseline = 0.#0.08*1e6 #[Pa]

# Reverberation parameters (multiple smaller pulses)
reverb_amp     = psi_to_pa(30e3)  #[Pa] 30 kPSI
reverb_means = [10e-3, 15e-3, 20e-3, 25e-3]  # Peaks at 10 ms, 15 ms, etc.
reverb_std_dev = 1e-3  #[s] Pulse width for all reverberations
reverb_baseline = p_late['P_Pa'].iloc[0]  #[Pa] 
#  reverb_baseline = psi_to_pa(50)  #[Pa] 

# Generate the pressure signal
pressure = gaussian(time, main_pulse_amp, main_pulse_mean, main_pulse_std_dev,baseline=main_pulse_baseline)
#  # Ensure pressure starts at 0 (can shift later)
#  pressure-=pressure[0]


# Add smaller reverberations
for mean in reverb_means:
    pressure += gaussian(time, reverb_amp, mean, reverb_std_dev, baseline=reverb_baseline)
# Ensure (AGAIN) that pressure starts at 0 (can shift later)
pressure-=pressure[0]

# Now we want to ensure the reverberations finish at around 50 psi, 
# so we'll shift the pressure to ensure the final value is around 50 psi
final_pressure_target = reverb_baseline  # Target final pressure to match late-time data
#  pressure_shift = final_pressure_target - pressure[-1]
#  pressure += pressure_shift
# (Different approach)
# Remove pressures at the end of the reverbs that are less than desired target (interpolate later)
# Work backwards and remove pressures below the target
index = len(pressure) - 1
while pressure[index] < final_pressure_target and index >= 0:
    index -= 1

# Keep pressures above the target and interpolate from the remaining pressure to the target pressure
remaining_pressure = pressure[:index + 1]
remaining_time = time[:index + 1]
removed_pressure_times = time[index+1:]

# Linearly interpolate from the last remaining pressure to the target pressure
#  blend_time = np.linspace(remaining_time[-1], time_end, 10)
#  blend_pressure = np.linspace(remaining_pressure[-1], final_pressure_target, 10)
blend_time = removed_pressure_times
blend_pressure = np.linspace(remaining_pressure[-1], final_pressure_target, len(blend_time))

# Extend the time and pressure with the interpolated values
time_extended = np.concatenate([remaining_time, blend_time])
pressure_extended = np.concatenate([remaining_pressure, blend_pressure])

# Save the early pressure for referecne
p_early = pressure_extended



#------------------------------------------------------------ 
# Plotting the early-time and full pressure time series
#------------------------------------------------------------ 
fig,(ax1,ax2) = plt.subplots(2, figsize=(10, 8))
ax1.plot(time_extended, p_early)
#  ax1.plot(time, pressure/1000, label='Pressure (kPSI)')
# Add 30ms vertical line and label
ax1.axvline(x=time_end, color='r', linestyle='--', label='t=30 ms')  # Vertical line at 30 ms
ax1.text(time_end + 0.001, pressure.max()/1000 * 0.9, 't=30 ms', color='r')  # Text label
# 2nd y-axis in kPSI
ax3 = ax1.twinx()
mn,mx =  ax1.get_ylim()
ax3.set_ylim(pa_to_psi(mn)/1000., pa_to_psi(mx)/1000.)
ax3.set_ylabel(r'$\Delta P$ [kpsi]',rotation=270,va='bottom')
# Plot Transducer Limit 
ax3.axhline(y=transducer_limit/1000, color='grey', ls='--',lw=1.0)
ax3.text(0.01*max(time_extended), transducer_limit/1000+5., f'transducer limit', ha='left', fontsize=12)
# Properties
ax1.set_title('Early-time Reverberations')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('$\Delta P$ [Pa]')
ax1.grid(True)
#  plt.savefig( join('output','inletPressureBC.pdf') )
#  plt.close('all')



#------------------------------------------------------------ 
# Add on cavity pressures as late-time
#------------------------------------------------------------ 


#------------------------------------------------------------ 
# Now, extend the time series with late-time  DataFrame
#------------------------------------------------------------ 
time_extended = np.concatenate([time_extended, p_late['t_s']])  # Extend time
pressure_extended = np.concatenate([pressure_extended, p_late['P_Pa']])  # Extend pressure

#  # Blend the transition smoothly (optional):
#  # Here we use a linear interpolation over a short period to smooth the transition
#  blend_time = np.linspace(time_end, p_late['t_s'].iloc[0], 10)
#  blend_pressure = np.linspace(pressure[-1], p_late['P_psi'].iloc[0], 10)
#  
#  time_extended = np.concatenate([time, blend_time, p_late['t_s']])
#  pressure_extended = np.concatenate([pressure, blend_pressure, p_late['P_psi']])

#------------------------------------------------------------ 
# Output the data to CSV for use as BCs
#------------------------------------------------------------ 
#  df = pd.DataFrame( [time,pressure], columns=['time','P'] )
df = pd.DataFrame( np.column_stack([time_extended,pressure_extended]), columns=['time','P'] )
df.to_csv( 'inletDirichletPressureBC.csv' , index=False)

#------------------------------------------------------------ 
# Plotting the full extended pressure-time series
#------------------------------------------------------------ 
#  plt.figure(figsize=(8, 4))
ax2.plot(time_extended, pressure_extended)
ax2.axvline(x=time_end, color='r', linestyle='--', label='t=30 ms')  # Vertical line at 30 ms
ax2.text(time_end + 0.001, pressure.max() * 0.9, 't=30 ms', color='r')  # Text label
# 2nd y-axis in kPSI
ax4 = ax2.twinx()
ax4.plot(time_extended, pa_to_psi(pressure_extended)/1000.,ls='none')
#  mn,mx =  ax2.get_ylim()
#  ax4.set_ylim(pa_to_psi(mn)/1000., pa_to_psi(mx)/1000.)
ax4.set_ylabel(r'$\Delta P$ [kpsi]',rotation=270,va='bottom')
# Plot Transducer Limit 
ax4.axhline(y=transducer_limit/1000, color='grey', ls='--',lw=1.0)
ax4.text(0.99*max(time_extended), transducer_limit/1000+10., f'transducer limit', ha='right', fontsize=12)
# Properties

ax2.set_title('Extended Pressure Time Series with Reverberations and Late-time Decay')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax4.set_yscale('log')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('$\Delta P$ [Pa]')
ax2.grid(True)
#  .legend()
plt.tight_layout()
plt.savefig( join('output','inletPressureBC.pdf') )
plt.close('all')
