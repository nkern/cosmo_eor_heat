"""
=====================
config_multiprog.py
=====================
 Either:
	1.  Create the configuration file needed by srun
	2.	Send jobout files to appropriate directories
"""
import sys
import os

if sys.argv[1] == 'config':
	### Configure Config Files ###
	# Get parameters
	configID = sys.argv[2]
	direcs = sys.argv[3:]
	Ndirecs = len(direcs)

	# Write config file
	f = open('MPMD'+configID+'.conf','w')
	for i in range(Ndirecs):
		f.write(str(i)+'\t'+direcs[i]+'/run_21cmFAST.sh\n')
	f.close()

elif sys.argv[1] == 'jobout':
	### Send jobout files back to direcs ###
	# Get parameters
	prefix = sys.argv[2]
	direcs = sys.argv[3:]
	Ndirecs = len(direcs)
	direcID = map(lambda x: "%02i" % x, range(Ndirecs))
	# Iterate over direcs
	for i in range(Ndirecs):
		os.system('mv '+prefix+direcID[i]+'.out '+direcs[i]+'/')
