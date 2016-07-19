""" Import global parameters from Boxes and write them out """

import os, sys, fnmatch
import numpy as np

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

# Write to file
f = open('../global_params.tab','w')
f.write('#z\t nf\t aveTb\n')

for i in range(len(xH_params)):
	f.write(xH_params[i][2][1:]+'\t'+xH_params[i][3][2:]+'\t'+delta_T_params[i][11][5:]+'\n')

f.close()

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
