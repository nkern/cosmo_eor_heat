"""
run camb a bunch of times
"""

import numpy as np
import scipy.stats as stats
from pycape.drive_camb import drive_camb
from collections import OrderedDict
from IPython import get_ipython
ipython = get_ipython()
from pycape import common_priors

def extend_dict(orig_dict,new_dict):
	for n in new_dict.keys():
		try:
			orig_dict[n].append(new_dict[n])
		except:
			pass

# Import planck cov mat
planck_cov = np.loadtxt('base_TTTEEE_lowl_plik.covmat')[[0,1,2,4,5]].T[[0,1,2,4,5]].T

# Define parameter names for pycape and CAMB
p_names = ['ombh2','omch2','theta_mc100','lntentenAs','ns']
camb_params = ['ombh2','omch2','theta_mc','As','ns']

# Create running data lists
params = ['ombh2','omch2','theta_mc','As','ns','sigma8','hlittle']
vals = [[],[],[],[],[],[],[]]
data = OrderedDict(zip(params,vals))

# Initialize camb
C = drive_camb({})

# Create multidimensional gaussian
planck_mean = map(lambda x: common_priors.cmb_priors1[x],p_names)
mgauss = stats.multivariate_normal.rvs(mean = planck_mean, cov = planck_cov, size = 2000)

# Time a single run
timeit = False
if timeit == True:
	print '-'*30
	print '...timing set_params()'
	ipython.magic("%timeit -n10 C.set_params()")
	print '-'*30

# Iterate through parameters
for i in range(len(mgauss)):
	this_run = mgauss[i]*1.0
	this_run[2] = this_run[2]/100.0
	this_run[3] = np.exp(this_run[3])/1e10
	C.set_params(H0=None,**dict(zip(camb_params,this_run)))
	extend_dict(data,C.get_pars)

data = np.array(data.values())


# Take covariance and write to file
cov = np.cov(data)

# Write to file
f = open('new_planck_cov.tab','w')
f.write('#'+'\t'.join(params)+'\n')
for i in range(len(params)):
	f.write('\t'.join(list(map(str,cov[i])))+'\n')

f.close()













