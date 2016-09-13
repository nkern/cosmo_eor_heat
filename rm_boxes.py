"""
rm_boxes.py : remove box data but keep specific ones
"""
import numpy as np
import os
import fnmatch

keep_box = ['Ts_z','xH_nohalos_z','delta_T_v3_no_halos_z','Ts_evolution/Tk_zprime']
z_arr = np.array(map(lambda x: "%06.2f" % x, np.around(sorted(np.loadtxt('../Output_files/Ts_outs/'+fnmatch.filter(os.listdir('../Output_files/Ts_outs'),'global_evolution*')[0],usecols=(0,))),2)))
keep_z = np.array([7.0,8.0,9.0,10.0,10.5,11.0])

# Iterate through subdirecs
for dirpath,dirs,files in os.walk('.'):

	# Sort files in subdirec
	files = np.array(sorted(files))

	# Iterate through files
	for i in range(len(files)):
		keep_file = sum(map(lambda x: x in files[i], keep_box))
		if keep_file == 0:
			os.sytem('rm -r '+files[i])








