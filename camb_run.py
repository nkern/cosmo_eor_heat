"""
run camb a bunch of times
"""

import numpy as np
from pycape.drive_camb import drive_camb
from collections import OrderedDict

# Import planck cov mat
planck_cov = np.loadtxt('base_TTTEEE_lowl_plik.covmat')[[0,1,2,4,5]].T[[0,1,2,4,5]].T
params = ['omegabh2','omegach2','100theta_mc','ln(1e10*As)','ns']
vals = [[],[],[],[],[]]

# Initialize camb
C = drive_camb({})

# Iterate through thousands of runs
data = OrderedDict(zip(params,vals))









