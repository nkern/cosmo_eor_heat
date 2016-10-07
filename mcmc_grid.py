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
import pyDOE

if __name__ == '__main__':

	if sample_grid == True:
		sampler = 'lhs'

		# Load Sample Bound Information
		#par, parbound = np.loadtxt('sample_HERA331_limits.tab',dtype='str',unpack=True)
		par, parbound = np.loadtxt('sample_paramsearch_limits.tab',dtype='str',unpack=True)
		parbound = np.array(parbound,float)
		#parbound /= 2.0

		# Draw random samples
		if sampler == 'gauss':
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

		elif sampler == 'lhsfs':
			while True:
				samples = pyDOE.lhs(11,samples=N_samples*10,criterion='maximin') - np.array([0.5 for i in range(11)])
				R = np.sqrt(np.array(map(np.sum,samples**2)))
				within = R < 0.5
				if len(np.where(within==True)[0]) >= N_samples: break
			samples = samples[within]
			samples = samples[np.random.choice(np.arange(len(samples)),N_samples,replace=False)]
			samples *= 2*parbound
			samples += params_fid

		elif sampler == 'uniform':
			# Sample uniformly from space
			p1_lim = [0.60,1.10]
			p2_lim = [1000,80000]
			a3_lim = [0.1,100]
			param_lims = [p1_lim,p2_lim,p3_lim]

			samples = []
			for i in range(len(params)):
				samples.append( np.round(stats.uniform.rvs(loc=param_lims[i][0],scale=param_lims[i][1]-param_lims[i][0],size=N_samples),6) )
			samples = np.array(samples)[::-1]

		elif sampler == 'lhs':
			samples = pyDOE.lhs(len(params),samples=N_samples,criterion='maximin') - np.array([0.5 for i in range(len(params))])
			samples *= 2*parbound
			samples += params_fid

		# Write out samples to new fits file
		filename = 'TS_samples5.fits'
		ts_samples = {}
		N = len(params)
		keys = params
		for i in range(N):
			ts_samples[keys[i]] = samples.T[i]

		fits_table(ts_samples,keys,filename,clobber=True)

	if compile_direcs == True:
		grid = fits.open('TS_samples5.fits')[1].data
		names = grid.names
		grid = fits_data(grid)
		gridf = np.array( map(lambda x: grid[x], names) ).T
		#grid = np.array( map(lambda y: map(lambda x: "%09.5f" % x, y), gridf) )
		grid = np.array( map(lambda y: map(lambda x: "%07.3f" % x, y), gridf) )

		direcs = []
		zipped = np.array(map(lambda x: zip(np.array(params)[[6,10]],x),grid.T[[6,10]].T))
#		zipped = np.array(map(lambda x: zip(np.array(params),x),grid))
		j = 0
		for i in range(len(grid)):
#			if i % 50 == 0 and i != 0: j += 1
			direcs.append('_'.join(map(lambda x: '_'.join(x),zipped[i])))
			#direcs.append('_'.join(map(lambda x: '_'.join(x),zipped[i][j][np.newaxis,:])))

		direcs = np.array(direcs)	

		if write_direcs == True:	
			f = open('direcs.tab','w')
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
			os.system('cp rm_boxes.py '+working_direc+'/')

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
		Nruns       	= 2000								# Total number of simulations we need to run
		Njobs       	= 66								# Number of different SLURM jobs to submit
		Nnodes      	= 1								# Number of nodes to request per job
		tasks_per_node	= 6								# Number of tasks to run per node
		Ntasks      	= tasks_per_node * Nnodes		# Number of individual tasks (processes) to run across all nodes
		cpus_per_task	= 4								# Number of CPUs to allocate per task (threads)
		Nseq        	= 5								# Number of sequential simulations to run per task
		direc_file		= 'direcs.tab'					# File containing directories to be run
		walltime		= '30:00:00'						# Amount of walltime for slurm job
		base_dir		= 'param_space/lhs/'	# Base drectory
		mem_per_cpu		= 500							# Memory in MB per cpu
		Nstart			= 4000								# Start index in directory file
		partition		= 'regular'						# NERSC Partition to run on 
		job_name		= 'Small'						# Job name
		infile			= 'slurm_21cmFAST_old.sh'		# SLURM infile

		# Search and Replace
		def SaR(SaR_dic,infile,outfile):
			zipped = zip(SaR_dic.keys(),SaR_dic.values())
			os.system("sed -e '"+' '.join(map(lambda x: 's#'+str(x[0])+'#'+str(x[1])+'#g;',zipped))+"' < "+infile+" > "+outfile)

		# Create Search and Replace Dictionary
		search = ['@@Nruns@@','@@Nnodes@@','@@tasks_per_node@@','@@Ntasks@@','@@cpus_per_task@@',\
					'@@Nseq@@','@@direc_file@@','@@walltime@@','@@mem_per_cpu@@','@@Nstart@@',\
					'@@partition@@','@@job_name@@','@@base_dir@@']
		replace = [Nruns,Nnodes,tasks_per_node,Ntasks,cpus_per_task,Nseq,direc_file,walltime,\
					mem_per_cpu,Nstart,partition,job_name,base_dir]

		SaR_dic = OrderedDict(zip(search,replace))

		# Iterate over job submissions
		for i in range(Njobs):
			print ''
			print 'running job #'+str(i)
			print '-'*30

			# perform search and replace
			SaR_dic['@@Nstart@@'] = int(Nstart + i*Ntasks*Nseq)
			SaR(SaR_dic,infile,'slurm_21cmFAST_auto.sh')

			# Send jobs
			os.system('sbatch slurm_21cmFAST_auto.sh')

		# Leftover job
		Nleftover = Nruns % (Ntasks*Nseq*Njobs)
		Nseq_leftover = np.ceil(float(Nleftover)/tasks_per_node)
		#Nnodes = 0
		if Nnodes == 1 and Nleftover > 0:
			print ''
			print 'running leftover job #'+str(i+1)
			print '-'*30

			# Search and Replace
			SaR_dic['@@Nstart@@']			= int(Nstart + (i+1)*Ntasks*Nseq)
			SaR_dic['@@Nruns@@']			= Nleftover
			SaR_dic['@@Ntasks@@']			= Nleftover
			SaR_dic['@@Nnodes@@']			= 1
			SaR_dic['@@Nseq@@']				= int(Nseq_leftover)
			SaR(SaR_dic,infile,'slurm_21cmFAST_auto.sh')

			# Send job
			os.system('sbatch slurm_21cmFAST_auto.sh')


