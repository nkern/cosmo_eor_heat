'''
--------------
mcmc_grid.py
--------------

Make grid of points to use in mcmc of heating parameters

Relies on pre-determined variables from mcmc_params.py

Nicholas Kern
January, 2016
'''

## Import Modules
import numpy as np
import numpy.linalg as la
import os, sys
from mcmc_params import *
from DictEZ import create as ezcreate
from grid_helper import *
import time
from collections import OrderedDict
import re
import astropy.io.fits as fits
import scipy.stats as stats
from fits_table import fits_table,fits_data,fits_append
from round_to_n import round_to_n

if __name__ == '__main__':

	if sample_grid == True:
		sample_gauss = True

		# Load Sample Bound Information
		par, parbound = np.loadtxt('sample_HERA127_limits.tab',dtype='str',unpack=True)
		parbound = np.array(parbound,float)/2.0

		# Draw random samples
		if sample_gauss == True:
			# Multivariate Gaussian
			cov = np.eye(len(par))*parbound**2
			samples = np.round(stats.multivariate_normal.rvs(mean=params_fid,cov=cov,size=N_samples),7)

			# Check for negative values
			while True:
				neg = np.where(samples < 0)
				if len(neg[0]) > 0:
					new_samples = np.round(stats.multivariate_normal.rvs(mean=params_fid,cov=cov,size=len(neg[0])),7)
					samples[neg[0]] = new_samples
				else:
					break

		else:
			# Sample uniformly from space
			p1_lim = [0.60,1.10]
			p2_lim = [1000,80000]
			a3_lim = [0.1,100]
			param_lims = [p1_lim,p2_lim,p3_lim]

			samples = []
			for i in range(len(params)):
				samples.append( np.round(stats.uniform.rvs(loc=param_lims[i][0],scale=param_lims[i][1]-param_lims[i][0],size=N_samples),6) )
			samples = np.array(samples)[::-1]

		# Write out samples to new fits file
		filename = 'TS_samples2.fits'
		ts_samples = {}
		N = len(params)
		keys = params
		for i in range(N):
			ts_samples[keys[i]] = samples.T[i]

		fits_table(ts_samples,keys,filename,clobber=True)

	if compile_direcs == True:
		grid = fits.open('CV_samples.fits')[1].data
		names = grid.names
		grid = fits_data(grid)
		gridf = np.array( map(lambda x: grid[x], names) ).T
		grid = np.array( map(lambda y: map(lambda x: "%07.3f" % x, y), gridf) )

		direcs = []
		zipped = np.array(map(lambda x: zip(np.array(params)[[6,10]],x),grid.T[[6,10]].T))
		zipped = np.array(map(lambda x: zip(np.array(params),x),grid))
		j = 0
		for i in range(len(grid)):
			direcs.append('_'.join(map(lambda x: '_'.join(x),zipped[i][j][np.newaxis,:])))
			if i % 50 == 0 and i != 0: j += 1

		direcs = np.array(direcs)	

		if write_direcs == True:	
			f = open('cv_direcs.tab','w')
			f.write('\n'.join(map(lambda x: base_direc+x,direcs)))
			f.close()


	if build_direcs == True:

		# Single or multiple directories?
		single_direc = False
		if single_direc == True:
			gridf = np.array(params_fid)[:,np.newaxis].T
			grid = np.array( map(lambda y: map(lambda x: "%07.3f" % x, y), gridf) )
			direcs = ['_'.join(map(lambda x: '_'.join(x),np.array(zip(np.array(params)[[6,10]],np.array(map(lambda x: "%07.3f" % x, grid[0]))[[6,10]]))))]

		# Iterate over Parameters
		N = len(grid)
		M = len(params)
		for i in np.arange(0,N):
			print ''
			print 'working on sample #:',i
			print 'directory name: '+direcs[i]
			print '-'*30
			for j in range(M):
				print params[j],'=',grid[i][j]
			print ''

			# Make Dictionary of values and directory string
			param_vals = list(gridf[i])
			param_dict = OrderedDict(zip(params,param_vals))
			working_direc = base_direc + direcs[i]

			if os.path.isdir(working_direc) == True:
				print working_direc,'already exists, overwriting...'

			os.system('mkdir '+working_direc)

			# Create dictionary of variables to insert into 21cmFAST files
			vars_21cmFAST = OrderedDict( param_dict.items() + variables.items() )
	
			# Place 21cmFAST files
			os.system('cp -r '+sim_root+'/Cosmo_c_files '+working_direc+'/')
			os.system('cp -r '+sim_root+'/Parameter_files '+working_direc+'/')	
			os.system('cp -r '+sim_root+'/Programs '+working_direc+'/')
			os.system('cp -r '+sim_root+'/External_tables '+working_direc+'/')
			os.system('cp mcmc_params.py '+working_direc+'/')
			os.system('cp global_params.py '+working_direc+'/')

			# Insert parameters
			insert_params(vars_21cmFAST,working_direc)

			# Write .tab file with parameter values
			f = open(working_direc+'/param_vals.tab','w')
			for k in range(M):
				f.write(params[k]+'\t'+str(param_vals[k])+'\n')
			f.close()

			# Insert PBS file
			full_direc = direc_root+'/'+working_direc
			os.system("sed -e 's#@@working_direc@@#"+full_direc+"#g;s#@@command@@#"+command+"#g;' < drive_21cmFAST.sh > "+working_direc+"/run_21cmFAST.sh")
			os.system("chmod 755 "+working_direc+"/run_21cmFAST.sh")


	if send_slurm_jobs == True:
		# Assign run variables
		Nruns       	= 4000						# Total number of simulations we need to run
		Njobs       	= 5							# Number of different SLURM jobs to submit
		Nnodes      	= 10						# Number of nodes to request per job
		tasks_per_node	= 16						# Number of tasks to run per node
		Ntasks      	= tasks_per_node * Nnodes	# Number of individual tasks (processes) to run per node
		cpus_per_task	= 2							# Number of CPUs to allocate per task (threads)
		Nseq        	= 5							# Number of sequential simulations to run per task
		direc_file		= 'direcs.tab'				# File containing directories to be run
		walltime		= '07:20:00'				# Amount of walltime for slurm job
		base_direc		= 'param_space/gauss_hera127/'

		Nstart = 0

		# Load in slurm file
		job_file = open('slurm_21cmFAST.sh','r')
		job_string = np.array(job_file.read().split('\n'))
		job_file.close()

		job_string[2]	= "#SBATCH --nodes="+str(Nnodes)
		job_string[3]	= "#SBATCH --ntasks-per-node="+str(tasks_per_node)
		job_string[4]	= "#SBATCH --cpus-per-task="+str(cpus_per_task)
		job_string[5]	= "#SBATCH --time="+str(walltime)
		job_string[16]	= "IFS=$'\\r\\n' command eval 'direcs=($(<"+str(direc_file)+"))'"
		job_string[17]	= "basedir='"+base_direc+"'"
		job_string[20]	= "begin="+str(Nstart)
		job_string[21]	= "tot_length="+str(Nruns)
		job_string[25]	= "Nseq="+str(Nseq)
		job_string[26]	= "begin="+str(Nstart)
		job_string[27]	= "length="+str(Ntasks)

		for i in range(Njobs):
			print ''
			print 'running job #'+str(i)
			print '-'*30

			this_job_string = np.copy(job_string)
			this_job_string[26] = 'begin='+str(Nstart + i*Ntasks*Nseq)
			file = open('slurm_21cmFAST_auto.sh','w')
			file.write('\n'.join(this_job_string))
			file.close()

		#	os.system('sbatch slurm_21cmFAST_auto.sh')

		Nleftover = Nruns % (Ntasks*Nseq*Njobs)
		Nseq = Nruns / (Ntasks*Nseq*Njobs)
		if Nleftover != 0 and Nseq > 0:
			print ''
			print 'running leftover job #'+str(i+1)
			print '-'*30

			this_job_string = np.copy(job_string)
			this_job_string[25] = 'Nseq='+str(1)
			this_job_string[26] = 'begin='+str(Nstart + (i+1)*Ntasks*Nseq)
			this_job_string[27] = 'length='+str(Nleftover)
			file = open('slurm_21cmFAST_auto.sh','w')
			file.write('\n'.join(this_job_string))
			file.close()

			#os.system('sbatch slurm_21cmFAST_auto.sh')


