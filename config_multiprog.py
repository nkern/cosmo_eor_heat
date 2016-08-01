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
import numpy as np

if sys.argv[1] == 'config':
	### Configure Config Files ###
	# Get parameters
	jobID	= sys.argv[2]
	seqID	= sys.argv[3]
	nodeID	= sys.argv[4]
	direcs = sys.argv[5:]
	Ndirecs = len(direcs)

	# Write config file
	f = open('MPMD_job'+jobID+'_seq'+seqID+'_node'+nodeID+'.conf','w')
	for i in range(Ndirecs):
		f.write(str(i)+'\t'+'bash '+direcs[i]+'/run_21cmFAST.sh\n')
	f.close()

elif sys.argv[1] == 'jobout':
	### Send jobout files back to direcs ###
	# Get parameters
	jobID	= sys.argv[2]
	seqID	= sys.argv[3]
	Nnodes	= int(sys.argv[4])

	# Iterate over MPMD configuration files
	for i in range(Nnodes):
		tasks, direcs = np.loadtxt('MPMD_job'+jobID+'_seq'+seqID+'_node'+str(i)+'.conf',unpack=True,usecols=(0,2),dtype='str')
		Ntasks = len(tasks)
		for j in range(Ntasks):
			outfile1	= 'job'+jobID+'_seq'+seqID+'_node'+str(i)+'_task'+"%02i"%j+'.out'
			outfile2	= '/'.join(direcs[j].split('/')[:-1])+'/jobout.txt'
			os.system('mv '+outfile1+' '+outfile2)


