"""
rm_boxes.py : remove box data but keep specific ones
"""
import numpy as np
import os
import fnmatch

keep_box = ['Ts_z','xH_nohalos_z','delta_T_v3_no_halos_z','Tk_zprime','updated_smoothed_deltax_z']
z_arr = np.array(map(lambda x: "%06.2f" % x, np.around(sorted(np.loadtxt('../Output_files/Ts_outs/'+fnmatch.filter(os.listdir('../Output_files/Ts_outs'),'global_evolution*')[0],usecols=(0,))),2)))
z_arrf = np.array(z_arr,float)
keep_z = np.array([7.0,8.0,9.0,10.0,11.0,12.0])
closest_z = np.array(map(lambda x: z_arr[np.where(np.abs(x-z_arrf)==np.abs(x-z_arrf).min())[0][0]],keep_z))
keep_arr = np.array(map(lambda x: [x+closest_z[i] for i in range(len(closest_z))], keep_box)).ravel()

# Iterate through subdirecs
for dirpath,dirs,files in os.walk('../Boxes'):

	# Sort files in subdirec
	files = np.array(sorted(files))

	# Iterate through files
	for i in range(len(files)):
		keep_file = np.array(map(lambda x: x in files[i],keep_arr))
		keep_bool = True in keep_file
		if keep_bool is False:
			os.system('rm -r '+dirpath+'/'+files[i])
