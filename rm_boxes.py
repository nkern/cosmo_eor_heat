"""
rm_boxes.py : remove box data but keep specific ones
"""
import numpy as np
import os
import fnmatch

keep_box = ['Ts_z','xH_nohalos_z','delta_T_v3_no_halos_z','Ts_evolution/Tk_zprime']
z_arr = np.array(map(lambda x: "%06.2f" % x, np.around(sorted(np.loadtxt('../Output_files/Ts_outs/'+fnmatch.filter(os.listdir('../Output_files/Ts_outs'),'global_evolution*')[0],usecols=(0,))),2)))
keep_z = np.array([7.0,8.0,9.0,10.0,10.5,11.0])







