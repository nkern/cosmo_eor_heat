""" Import global parameters from Boxes and write them out """

import os, sys, fnmatch
import numpy as np
from curve_interp import curve_interp

# Extract global signal and mass-weighted neutral fraction from Box Names
# Load filenames
files = os.listdir('../Boxes')
xH_files = fnmatch.filter(files,'xH_nohalos*')
delta_T_files = fnmatch.filter(files,'delta_T*')

# Sort by redshift
xH_files = sorted(xH_files)
delta_T_files = sorted(delta_T_files)

# Extract parameter values
xH_params = map(lambda x: x.split('_'), xH_files)
delta_T_params = map(lambda x: x.split('_'), delta_T_files)

# Extract spin temperature, kinetic temperature and the volume-filling factor
filename = fnmatch.filter(os.listdir('../Output_files/Ts_outs'), 'global_evolution*')[0]
zp, Q, Tk, Ts = np.loadtxt('../Output_files/Ts_outs/'+filename,dtype=float,usecols=(0,1,2,4),unpack=True)
zp, Q, Tk, Ts = zp[::-1], Q[::-1], Tk[::-1], Ts[::-1]

# Extract Tau
try:
	tz, tau, avg_x_HII_1_deltab = np.loadtxt('../tau.tab', unpack=True)
	try:
		assert(len(tz)==len(zp))
	except:
		tau = curve_interp(zp, tz, tau[:,np.newaxis]).ravel()
except:
	tz, tau = np.copy(zp), np.zeros(len(zp))

# Write to file
with open('../global_params.tab','w') as f:
	f.write('#z\t nf\t aveTb\t Ts\t Tk\t Q\t tau\n')
	for i in range(len(zp)):
		f.write(str(zp[i])+'\t'+xH_params[i][3][2:]+'\t'+delta_T_params[i][11][5:]+'\t'+str(Ts[i])+'\t'+str(Tk[i])+'\t'+str(Q[i])+'\t'+str(tau[i])+'\n')

# Convert PS files to shorter name
ps_files = os.listdir('../Output_files/Deldel_T_power_spec/')
ps_files = fnmatch.filter(ps_files,'ps_no_halos_z*v3.txt')
ps_files = sorted(ps_files)
for i in range(len(ps_files)):
	filename = ps_files[i]
	filebits = ps_files[i].split('_')
	index = np.where(np.array(map(lambda x: x[0],filebits))=='z')[0][0]
	new_name = '_'.join(np.array(filebits)[:index+1])+'.txt'
	os.system('mv ../Output_files/Deldel_T_power_spec/'+filename+' ../Output_files/Deldel_T_power_spec/'+new_name)
