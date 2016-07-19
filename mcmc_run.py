"""
==========
mcmc_fit.py
==========
- Use Emulator to describe PS within a region of 21cmFAST parameter space

Nicholas Kern
nkern@berkeley.edu
July, 2016
"""

## Import Modules
import os, sys
import numpy as np
import numpy.linalg as la
import numpy.random as rd
from mcmc_params import *
import fnmatch
from DictEZ import create as ezcreate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from fits_table import fits_table,fits_to_array,fits_data
from curve_poly_interp import curve_interp
from plot_ellipse import plot_ellipse
import astropy.io.fits as fits
import cPickle as pkl
from memory_profiler import memory_usage
import operator
from klip import klfuncs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import corner
import warnings
from pycape.toolbox import workspace
warnings.filterwarnings('ignore',category=DeprecationWarning)

## Flags
interp_ps				= False
calc_sense				= False
plot_scree				= False
plot_eigvecs_project	= False
plot_constructions		= False
plot_error_map			= False
plot_rms_err			= False

## Program
if __name__ == "__main__":

	###############################
	## Load Training Set samples ##
	grid1	= fits_to_array(fits.open('TS_samples1.fits')[1].data)
	grid2	= fits_to_array(fits.open('TS_samples2.fits')[1].data)
	grid	= np.vstack([grid1,grid2])
	gridf	= np.array(grid,float)
	grid	= np.array( map(lambda y: map(lambda x: "%07.3f" % x, y), grid) )

	direcs = []
	zipped = np.array(map(lambda x: zip(np.array(params)[[6,10]],x),grid.T[[6,10]].T))
	for i in range(len(grid)):
		direcs.append('_'.join(map(lambda x: '_'.join(x),zipped[i])))

	direcs = np.array(direcs)

	sort = np.argsort(direcs)
	indices = np.arange(len(grid))[sort]
	direcs = np.array(direcs)[sort]
	grid = np.array(grid)[sort]
	gridf	= np.array(gridf)[sort]

	# Get directories that have global_params.tab
	glob_sel = []
	for i in range(len(direcs)):
		if os.path.isfile('param_space/'+direcs[i]+'/global_params.tab'):
			glob_sel.append(i)
	N_samples = len(glob_sel)-1
	direcs	= direcs[glob_sel]
	grid	= grid[glob_sel]
	indices	= indices[glob_sel]
	gridf	= gridf[glob_sel]
	eval_samples = np.arange(0,N_samples)

	## Define redshift bins
	z_array     = np.arange(5.5,27.1,0.5)
	zbin        = 0.5

	freq_low    = 1.420405 / (z_array+zbin/2.0+1)
	freq_high   = 1.420405 / (z_array-zbin/2.0+1)
	freq_cent   = np.round(1.420405 / (z_array+1),4)
	bandwidth   = np.round(1.420405 / (z_array-zbin/2.0+1) - 1.420405 / (z_array+zbin/2.0+1), 4)

	ps_interp_files = sorted(map(lambda x: 'ps_interp_z%06.2f.txt'%x,z_array))

	# Interpolate redshift outputs to new redshift array
	if interp_ps == True:

		# Work on new direcs only?
		new_direcs = False
		if new_direcs == True:
			new_direcs = []
			for i in range(len(direcs)):
				if len(fnmatch.filter(os.listdir('param_space/'+direcs[i]+'/Output_file/Deldel_T_power_spec'),'ps_interp*')) == 0:
					new_direcs.append(direcs[i])
			direcs = np.array(new_direcs)

		single_direc = True
		if single_direc == True:
			direcs = ['mock_observation']

		for i in range(len(direcs)):
			print 'working on direc '+direcs[i]
			k_data = []
			ps_data = []
			z_data = []
			ps_files = fnmatch.filter(os.listdir('param_space/'+direcs[i]+'/Output_files/Deldel_T_power_spec'),'ps_no_halos*')
			ps_files = sorted(ps_files)
			for j in range(len(ps_files)):
				kdat,psdat = np.loadtxt('param_space/'+direcs[i]+'/Output_files/Deldel_T_power_spec/'+ps_files[j],usecols=(0,1),unpack=True)
				zdat = float(ps_files[j][13:19])
				z_data.append(zdat)
				k_data.append(kdat)
				ps_data.append(psdat)

			k_data = np.array(k_data)
			ps_data = np.array(ps_data)
			z_data = np.array(z_data)
			ps_pred = 10**curve_interp(z_array,z_data,np.log10(ps_data),n=3,degree=2)
			for j in range(len(z_array)):
				f = open('param_space/'+direcs[i]+'/Output_files/Deldel_T_power_spec/ps_interp_z%06.2f.txt'%z_array[j],'w')
				for k in range(k_data.shape[1]):
					f.write(str(k_data[0][k])+'\t'+str(ps_pred[j][k])+'\n')
				f.close()

	## Calculate Telescope Sensitivity ##
	if calc_sense == True:
		# Get fiducial parameters
		params_fid = np.loadtxt('param_space/mock_observation/param_vals.tab',usecols=(1,),unpack=True)

		# initialize workspace
		workspace_init = {'dir_pycape':'/global/homes/n/nkern/Software/pycape','dir_21cmSense':'/global/homes/n/nkern/Software/21cmSense'}
		W = workspace(workspace_init)
		W.obs_init(workspace_init)

		# get ps filenames
		ps_files = np.array(map(lambda x: 'param_space/mock_observation/Output_files/Deldel_T_power_spec/ps_interp_z%06.2f.txt'%(x),z_array))
		ps_filenum = len(ps_files)
		calc_sense_kwargs = {'foreground_model':'mod','buff':[0.1 for i in range(ps_filenum)],'ndays':180,'n_per_day':6,
							'bwidth':bandwidth,'nchan':[82 for i in range(ps_filenum)],'lowk_cut':0.10}
		data_filename = 'mockObs_hera37_allz.pkl'
		W.Obs.calc_sense('hera37',ps_files,freq=freq_cent,data_filename=data_filename,write_data=True,**calc_sense_kwargs)

	### Load in power spectrum data ###
	ps_direc        = 'Output_files/Deldel_T_power_spec/'
	ps_keep         = 'ps_interp_z??????.txt'
	z_select        = np.arange(len(z_array))
	k_range         = np.loadtxt('k_range.tab')
	k_select        = np.arange(len(k_range))
	g_array			= np.array([r'nf',r'Tb'])
	g_array_tex		= np.array([r'$\chi_{HI}$',r'$T_{b}$'])
	g_select		= np.arange(len(g_array))

	# Limit to zlimits
	if zmin != None:
		select = np.where(z_array[z_select] >= zmin)
		z_select = z_select[select]
	if zmax != None:
		select = np.where(z_array[z_select] <= zmax)
		z_select = z_select[select]

	z_len   = len(z_select)
	z_array = z_array[z_select]

	# Limit to klimits
	if kmin != None:
		select = np.where(k_range[k_select] >= kmin)
		k_select = k_select[select]
	if kmax != None:
		select = np.where(k_range[k_select] <= kmax)
		k_select = k_select[select]

	k_len   = len(k_select)
	k_range = k_range[k_select]

	# Limit to glimits
	if gmin != None:
		g_select = g_select[gmin:]
	if gmax != None:
		g_select = g_select[:gmax]

	g_len	= len(g_select)            # for nf and aveTb

	y_len   = k_len + g_len
	y_array = np.concatenate([k_range,g_array])
	x_len = y_len * z_len

	yz_data         = []
	map(lambda x: yz_data.extend(map(list,zip(y_array,[x]*y_len))),z_array)
	yz_data = np.array(yz_data)
	yz_data = yz_data.reshape(z_len,y_len,2)

	### Load Data ###
	print '...Loading Data\n'+'-'*30
	data		= []
	alldata_overwrite = False
	for direc in direcs:
		if os.path.isfile(base_direc+direc+'/ps_interp_alldata.tab') == False or alldata_overwrite == True:
			# Load PS and global data if file w/ all the info doesn't exist
			print '...working on direc = '+direc
			ps_files        = np.array(sorted(fnmatch.filter(os.listdir(base_direc+direc+'/'+ps_direc),ps_keep)))
			global_file	= base_direc+direc+'/'+'global_params.tab'
			global_data	= np.loadtxt(global_file,usecols=(1,2),unpack=True).T[::-1][z_select]
			sample_data = []
			for i in range(len(ps_files)):
				ps_data	= np.loadtxt(base_direc+direc+'/'+ps_direc+ps_files[i],delimiter='\t',usecols=(1,),unpack=True)[k_select]
				total_data = np.concatenate([ps_data,global_data[i]])
				sample_data.append(total_data)
			sample_data = np.array(sample_data)
			# Write out an all_data file
			with open(base_direc+direc+'/ps_interp_alldata.tab','w') as f:
				f.write('#z, k, ps\n')
				for j in range(z_len):
					for k in range(y_len):
						f.write(str(z_array[j])+'\t'+str(yz_data[j,k][0])+'\t'+str(sample_data[j,k])+'\n')

		z_data,y_data,samp_data = np.loadtxt(base_direc+direc+'/ps_interp_alldata.tab',dtype='str',unpack=True)
	
		data.append(samp_data)

	data = np.array(data,float)
	data[np.isnan(data)] = 0
	############################

	## Get fiducial data, remove from samples, slice out eval_samples ##
	fid_direc = 'zeta_040.000_numin_300.000'
	#fid_params = np.loadtxt(base_direc+fid_direc+'/param_vals.tab',usecols=(1,))
	#fid_ind = np.where(direcs==fid_direc)[0][0]
	#fid_data = data[fid_ind]
	fid_data = np.array(map(lambda x: np.median(x[np.where((np.isnan(x)!=True)|(x==0))]),data.T))
	fid_params = np.array(map(np.median,data.T))
	fid_rm = direcs!=fid_direc

	data	= data[fid_rm]
	grid	= grid[fid_rm]
	gridf	= gridf[fid_rm]
	direcs	= direcs[fid_rm]
	indices	= indices[fid_rm]
	indices	= np.argsort(indices)
	direcs	= direcs[fid_rm]

	## Write out data to file if desired
	write_data_to_file = False
	if write_data_to_file == True:
		diction = {'direcs':direcs,'data':data,'grid':grid,'indices':indices,'fid_data':fid_data,'fid_params':fid_params,'gridf':gridf}
		file = open('TS_1_data.pkl','wb')
		output = pkl.Pickler(file)
		output.dump(diction)
		file.close()

	############################################################
	### Load and Separate Training and Cross Validation Sets ###
	############################################################
	print '...loading and separating data'
	print '-'*30

	file = open('TS_1_data.pkl','rb')
	TS_data = pkl.Unpickler(file).load()
	TS_data['grid'] = np.array(TS_data['grid'],float)
	file.close()

	# Separate Data
	tr_len = 500 #500

	use_all_tr = False
	if use_all_tr == True:
		rando = np.arange(tr_len)
		rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))
	else:
		rando = np.random.choice(np.arange(tr_len),size=450,replace=False)
		rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))

	data_tr = TS_data['data'][TS_data['indices']][:tr_len][rando]
	grid_tr = TS_data[ 'gridf'][TS_data['indices']][:tr_len][rando]
	direcs_tr = TS_data['direcs'][TS_data['indices']][:tr_len][rando]

	cv_start = tr_len
	data_cv = TS_data['data'][TS_data['indices']][cv_start:]
	grid_cv = TS_data['gridf'][TS_data['indices']][cv_start:]
	direcs_cv = TS_data['direcs'][TS_data['indices']][cv_start:]

	fid_params = TS_data['fid_params']
	fid_data = TS_data['fid_data']

	add_more_data = False
	if add_more_data == True:
		file = open('TS_data_2_4.pkl','rb')
		#file = open('TS_3_4_data.pkl','rb')
		#file = open('TS_data_5_6.pkl','rb')
		#file = open('TS_data7.pkl','rb')
		TS_data = pkl.Unpickler(file).load()
		TS_data['grid'] = np.array(TS_data['grid'],float)
		file.close()

		# Separate Data
		tr_len = 800 #800, 1100
		#tr_len = 1000 #1000, 1300
		#tr_len = 343 # 343, 559
		tr_len = 150

		start = 800
		end = 1100
		data_cv = TS_data['data'][TS_data['indices']][start:end]
		grid_cv = TS_data['grid'][TS_data['indices']][start:end]
		direcs_cv = TS_data['direcs'][TS_data['indices']][start:end]


	# Separate Data By Dimensions
	#data_tr = data_tr.reshape(7,7,7,176)[::2,::2,::2,:].reshape(4**3,176)
	#grid_tr = grid_tr.reshape(7,7,7,3)[::2,::2,::2,:].reshape(4**3,3)
	#direcs_tr = direcs_tr.reshape(7,7,7)[::2,::2,::2].reshape(4**3)

	#data_cv = np.vstack([data_cv[25:75],data_cv[125:175],data_cv[225:275]])
	#grid_cv = np.vstack([grid_cv[25:75],grid_cv[125:175],grid_cv[225:275]])
	#direcs_cv = np.hstack([direcs_cv[25:75],direcs_cv[125:175],direcs_cv[225:275]])

	data_cv = TS_data['data'][TS_data['indices']][:tr_len][~rando]
	grid_cv = TS_data[ 'gridf'][TS_data['indices']][:tr_len][~rando]
	direcs_cv = TS_data['direcs'][TS_data['indices']][:tr_len][~rando]


	#########################
	### Plot Training Set ###
	#########################
	plot_ts = True
	if plot_ts == True:
		print '...plotting training set'
		print '-'*30
		fig = mp.figure(figsize=(14,9))
		fig.subplots_adjust(wspace=0.3)

		p1_lim = (0.75,0.95); p2_lim = (20000,65000); p3_lim = (15,45)
		p1_lim = None,None; p2_lim = None,None; p3_lim = None,None

		for i in range(6):
			ax=fig.add_subplot(2,3,i+1)
			if i == 5:
				ax.plot(grid_tr.T[0],grid_tr.T[2*i],'k.')
				ax.plot(grid_cv.T[0],grid_cv.T[2*i],'r.',alpha=0.5)
				ax.set_xlabel(p_latex[0],fontsize=18)
				ax.set_ylabel(p_latex[2*i],fontsize=18)
			else:
				ax.plot(grid_tr.T[2*i],grid_tr.T[2*i+1],'k.')
				ax.plot(grid_cv.T[2*i],grid_cv.T[2*i+1],'r.',alpha=0.5)
				ax.set_xlabel(p_latex[2*i],fontsize=16)
				ax.set_ylabel(p_latex[2*i+1],fontsize=16)
		fig.savefig('trainingset.png',dpi=100,bbox_inches='tight')
		mp.close()


	#######################
	### Set Up Emulator ###
	#######################

	def insert(dict_arr,name,value,scalar=False):
		arr_len = len(dict_arr)
		if type(value) == float or type(value) == int or type(value) == bool or value is None:
			for i in range(arr_len):
				dict_arr[i][name] = value
		elif type(value) is np.ndarray:
			if value.ndim == 1:
				for i in range(arr_len):
					dict_arr[i][name] = value
			elif value.ndim == 2:
				if value.shape[1] > 1:
					for i in range(arr_len):
						dict_arr[i][name] = value[i]
				else:
					for i in range(arr_len):
						dict_arr[i][name] = value[i][0]


	### Variables for Emulator ###
	N_modes = 200
	N_params = len(params)
	N_data = 660
	N_samples = len(data_tr)
	poly_deg = 2
	reg_meth='gaussian'
	theta0 = np.array([1.,1.,1.])
	thetaU = None#theta0*10.
	thetaL = None#theta0/10.

	gp_kwargs = {'regr':'linear','theta0':theta0,'thetaL':thetaL,'thetaU':thetaU,
					'random_start':1,'verbose':False,'corr':'squared_exponential'}

	variables.update({'params':params,'N_params':N_params,'N_modes':N_modes,'N_samples':N_samples,'N_data':N_data,
					'reg_meth':reg_meth,'poly_deg':poly_deg,'gp_kwargs':gp_kwargs})

	workspace_init = {'dir_pycape':'/Users/nkern/Desktop/Research/Software/pycape',
					'dir_21cmSense':'/Users/nkern/Desktop/Research/Software/21cmSense'}


	###########################
	### INITIALIZE EMULATOR ###
	###########################

	# Initialize workspace, emulator
	W = workspace(workspace_init)
	W.emu_init(variables)

	# Use a specific sample for fiducial point?
	feed_fiducial = False
	if feed_fiducial == True:
		fid_ind = 150
		fid_params = grid_cv[fid_ind]
		fid_data = data_cv[fid_ind]
	else:
		fid_params = None
		fid_data = None
		
	# Initialize Cholesky
	W.E.sphere(grid_tr,fid_params=fid_params,save_chol=True)
	fid_params = W.E.fid_params

	# Initialize Tree
	W.E.create_tree(W.E.Xsph,metric='euclidean',leaf_size=5*N_params)

	# Initialize KLT
	W.E.klt(data_tr,fid_data=fid_data)

	# Initialize Observation Class
	obs_dic = dict(zip(workspace_init.keys(),workspace_init.values()))
	W.obs_init(obs_dic)


	# Load Mock Observation and Configure
	file = open('mockObs_hera331_allz.pkl','rb')
	mock_data = pkl.Unpickler(file).load()
	file.close()
	true_direc = 'zeta_040.000_numin_300.000'
	p_true = params_fid

	### Variables for Mock Observation ###
	z_array         = np.arange(5.5,27.1,0.5)
	z_select        = np.arange(len(z_array))
	k_range         = np.loadtxt('k_range.tab')
	k_select        = np.arange(len(k_range))
	g_array         = np.array([r'nf',r'Tb'])
	g_array_tex     = np.array([r'$\chi_{HI}$',r'$T_{b}$'])
	g_select        = np.arange(len(g_array))

	# Limit to zlimits
	zmin = None
	zmax = None
	if zmin != None:
			select = np.where(z_array[z_select] >= zmin)
			z_select = z_select[select]
	if zmax != None:
			select = np.where(z_array[z_select] <= zmax)
			z_select = z_select[select]

	# Limit to klimits
	kmin = None
	kmax = None
	if kmin != None:
			select = np.where(k_range[k_select] >= kmin)
			k_select = k_select[select]
	if kmax != None:
			select = np.where(k_range[k_select] <= kmax)
			k_select = k_select[select]

	z_len   = len(z_select)
	z_array = z_array[z_select]
	k_len   = len(k_select)
	k_range = k_range[k_select]
	g_len   = len(g_select)            # for nf and aveTb

	y_len   = k_len + g_len
	y_array = np.concatenate([k_range,g_array])
	x_len = y_len * z_len

	yz_data         = []
	map(lambda x: yz_data.extend(map(list,zip(y_array,[x]*y_len))),z_array)
	yz_data = np.array(yz_data)
	yz_data = yz_data.reshape(z_len,y_len,2)

	freq_cent       = mock_data['freq']           # In GHz
	z_pick          = np.round(1420.405e6/(freq_cent*1e9) - 1,1)
	dz              = 0.5
	bandwidth       = 1420.405e6*1e-9 * (1/(1+z_pick-dz/2)-1/(1+z_pick+dz/2))     # In GHz
	z_num           = len(z_pick)
	z_lim = np.array(map(lambda x: x in z_pick, z_array))
	y_pick = k_range
	y_lim = np.array(map(lambda x: x in np.array(y_pick,str), y_array))
	model_lim = np.array([False]*N_data)
	for i in range(z_len):
		if z_lim[i] == True:
			model_lim[i*y_len:(i+1)*y_len] = np.copy(y_lim)

	W.E.model_lim = model_lim

	## Configure Mock Data and Append to Obs class
	names = ['sense_kbins','sense_PSdata','sense_PSerrs']
	for n in names:
		mock_data[n] = np.array(mock_data[n],object)
		for i in range(z_num):
				# Cut out inf and nans
				try: mock_data[n][i] = mock_data[n][i].T[mock_data['valid'][i]].T.ravel()
				except: mock_data[n]=list(mock_data[n]);mock_data[n][i]=mock_data[n][i].T[mock_data['valid'][i]].T.ravel()
				if n == 'sense_PSerrs':
					# Cut out sense_PSerrs / sense_PSdata > x%
					err_thresh = 4.0        # 400%
					small_errs = np.where(mock_data['sense_PSerrs'][i] / mock_data['sense_PSdata'][i] < err_thresh)[0]
					mock_data['sense_kbins'][i] = mock_data['sense_kbins'][i][small_errs]
					mock_data['sense_PSdata'][i] = mock_data['sense_PSdata'][i][small_errs]
					mock_data['sense_PSerrs'][i] = mock_data['sense_PSerrs'][i][small_errs]

	mock_data['sense_kbins'] = np.array( map(lambda x: np.array(x,float),mock_data['sense_kbins']))
	mock_data['sense_PSdata'] = np.array(np.concatenate(mock_data['sense_PSdata']),float)
	mock_data['sense_PSerrs'] = np.array(np.concatenate(mock_data['sense_PSerrs']),float)

	# If model data that comes with Mock Data does not conform to Predicted Model Data, interpolate!
	try:
		residual = k_range - mock_data['kbins'][0]
		if sum(residual) != 0:
			raise Exception
	except:
		# Interpolate onto k-bins of emulated power spectra
		new_kbins  = k_range*1
		new_PSdata = curve_interp(k_range, mock_data['kbins'][0], mock_data['PSdata'].T, n=2, degree=1 ).T
		mock_data['kbins'] = np.array([new_kbins for i in range(z_num)])
		mock_data['PSdata'] = new_PSdata
		
	globals().update(mock_data)
	sense_kbin_nums = np.array([len(x) for x in sense_kbins])

	# Feed Mock Observation to Workspace
	W.Obs.N_data = sense_PSdata.size
	W.obs_feed(kbins,sense_kbins,sense_PSdata,sense_PSerrs)


	##### Set Parameter values
	k = 30
	use_pca = True
	solve_simul = True
	split_modes = True
	split_var = 33.33
	calc_noise = False
	norm_noise = False
	fast = False
	compute_klt = False
	save_chol = False
	LAYG = False
	LAYG_pretrain = False
	NN_hyper = False
	NN_hyper_k = 30

	sample_noise = 1e-2 * np.linspace(1,5,N_modes).reshape(N_modes,1)

	if split_modes == True:
		W.E.group_eigenmodes(var_div=split_var)

	# Solve for GP Hyperparameters for every grid point
	#theta0 = np.array([1.0,1.0,1.0])
	#theta0 = np.array([0.1,0.1,0.1])
	#theta0 = np.array([0.17058725,  0.0139115,   0.10627892])
	theta0 = np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]*N_modes) *\
							np.linspace(1,10,N_modes).reshape(N_modes,1)
	thetaL = theta0/5
	thetaU = theta0*5
	thetaL = None
	thetaU = None

	if use_pca == False:
		data_tr = W.E.data_tr.T[W.E.model_lim].T
		W.E.data_tr = data_tr
		W.E.fid_data = W.E.fid_data[W.E.model_lim]
		N_modes = np.where(W.E.model_lim==True)[0].size
		W.E.N_modes = N_modes
		N_data = N_modes
		W.E.N_data = N_data

	gp_kwargs = {'regr':'linear','theta0':theta0,'thetaL':thetaL,'thetaU':thetaU,'random_start':3,'verbose':False,
					'optimizer':'fmin_cobyla','normalize':True,'corr':'squared_exponential','nugget':sample_noise}

	gp_kwargs_arr = np.array([gp_kwargs.copy() for i in range(N_modes)])
	insert(gp_kwargs_arr,'theta0',theta0)
	insert(gp_kwargs_arr,'thetaU',thetaU)
	insert(gp_kwargs_arr,'thetaL',thetaL)
	insert(gp_kwargs_arr,'nugget',sample_noise)

	kwargs_tr = {'use_pca':use_pca,'scale_by_std':False,'calc_noise':calc_noise,'norm_noise':norm_noise,
					'sample_noise':sample_noise,'verbose':False,'invL':W.E.invL,'solve_simul':solve_simul,
					'fast':fast,'compute_klt':compute_klt,'split_modes':split_modes,'save_chol':save_chol,
					'gp_kwargs_arr':gp_kwargs_arr}

	predict_kwargs = {'solve_simul':solve_simul,'fast':fast,'use_Nmodes':None,'use_pca':use_pca,'split_modes':split_modes}


	### Initialize Sampler Variables ###
	param_width = np.array([grid_tr.T[i].max() - grid_tr.T[i].min() for i in range(N_params)])

	eps = -0.05

	param_bounds = np.array([[grid_tr.T[i].min()+param_width[i]*eps,grid_tr.T[i].max()-param_width[i]*eps]\
								for i in range(N_params)])

	param_hypervol = reduce(operator.mul,map(lambda x: x[1] - x[0], param_bounds))

	use_Nmodes = None
	add_model_err = False
	calc_lnlike_emu_err = True
	emu_err_mc = False
	ndim = N_params
	nwalkers = 22

	sampler_init_kwargs = {'use_Nmodes':use_Nmodes,'param_bounds':param_bounds,'param_hypervol':param_hypervol,
						'nwalkers':nwalkers,'ndim':ndim,'N_params':ndim,'z_num':z_num}
	sampler_kwargs = {}

	lnprob_kwargs = {'add_model_err':add_model_err,'fast':fast,'kwargs_tr':kwargs_tr,
						'calc_lnlike_emu_err':calc_lnlike_emu_err,'predict_kwargs':predict_kwargs,'LAYG':LAYG,'k':k,
						'LAYG_pretrain':LAYG_pretrain,'emu_err_mc':emu_err_mc}

	W.samp_init(sampler_init_kwargs,lnprob_kwargs=lnprob_kwargs,sampler_kwargs=sampler_kwargs)














































	err_map = False
	if err_map == True:
		x = np.linspace(0.8,0.9,20)
		y = np.linspace(20000,60000,20)
		z = np.linspace(15,45,20)
		xx,yy,zz = np.meshgrid(x,y,z)
		theta=np.vstack([xx.ravel(),yy.ravel(),zz.ravel()])
		model,model_err = W.emu_forwardprop_weighterr(theta,use_Nmodes=15)
		model_err_fracs = np.abs(model_err/model)
		model_err_frac = np.array(map(np.min,model_err_fracs))
		model_err_frac = model_err_frac.reshape(xx.shape)

		marg_errmap = np.zeros((20,20))
		for i in range(9,10):
			marg_errmap += model_err_frac[:,i,:]

		fig,ax = mp.subplots()
		xmin,xmax	= y.min(),y.max()
		ymin,ymax	= z.min(),z.max()
		xdiff,ydiff	= xmax-xmin,ymax-ymin
		cax = ax.imshow(marg_errmap,origin='lower',extent=[xmin,xmax,ymin,ymax],cmap='pink_r',aspect=xdiff/ydiff)
		fig.colorbar(cax,label='fractional error')

		ax.plot(gridf.T[1],gridf.T[2],'bo',alpha=0.5)

		fig.savefig('err_map.png',bbox_inches='tight')
		mp.close()

	if plot_scree == True:
		# Plot Eigenvalues
		eig_sel = np.arange(30)         # Plot top 30
		fig = mp.figure()
		ax = fig.add_subplot(111)
		ax.plot(eig_sel+1,np.log10(KL.eig_vals[eig_sel]),'ko')
		ax.set_xlim(0,len(eig_sel)+1)
		ax.set_xlabel('Eigenvalue #',fontsize=16)
		ax.set_ylabel('log10 Eigenvalue (variance of PS units)',fontsize=16)
		ax.grid(True)
		fig.savefig('eigen_dispersion_Nsamp'+str(N_samples)+'.png',dpi=200,bbox_inches='tight')


	if plot_eigvecs_project == True:
		# Plot Eigenprojection Heat-Map Plot
		for i in range(5):
			# Get Eigenvector and kz data
			eig_vec = KL.eig_vecs[i]

			# reshape into kz_data shape
			eig_vec = eig_vec.reshape(z_len,y_len)

			# separate kz gz eig_vec
			kz_eig_vec = eig_vec[:,:k_len]
			gz_eig_vec = eig_vec[:,k_len:]

			# Make 2D Density plot showing degeneracy
			fig,ax = mp.subplots()

			# yz_eigvector plot
			cmap = mp.cm.get_cmap('seismic',41)
			im = ax.imshow(eig_vec,origin='lower',vmin=-0.5,vmax=0.5,cmap=cmap,interpolation='nearest')
			mp.xticks(list(np.arange(len(k_range))[::2])+list(np.arange(k_len,y_len)),list(np.round(k_range[::2],2))+list(g_array_tex))
			mp.yticks(np.arange(len(z_array)),z_array)
			ax.set_xlabel(r'$k$ (Mpc$^{-1}$)',fontsize=18)
			ax.set_ylabel(r'$z$',fontsize=18)

			ax.axvline(k_len-0.5,color='k')

			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="6%", pad=0.1)
			cbar = fig.colorbar(im,cax=cax,label='projection of unit eigen vector '+str(i+1))

			# Savefigure
			fig.savefig('eigvec_'+str(i+1)+'_components_Nsamp'+str(N_samples)+'.png',dpi=200,bbox_inches='tight')
			mp.close()

			# Are the fluctuations about the Fiducial PS Normally Distributed?
			#fig,ax = mp.subplots(4,4)
			#ax = ax.ravel()
			#j = 0
			#for i in np.arange(0,154,10):
			#        ax[j].hist(ps_data[i]-ps_fid[i],bins=25,histtype='step',color='b',linewidth=2)
			#        ax[j].tick_params(bottom='off',left='off',right='off',top='off',labelleft='off',labelbottom='off')
			#        j += 1
			#fig.savefig('ps_dist.png',dpi=200,bbox_inches='tight')

	if plot_constructions == True:
		# Choose discrete parameter values
		samps = [174] #rd.randint(0,N_samples,1)
		for jj in range(len(samps)):
			print jj
			ii = samps[jj]
			param_vals = map(lambda x: [float(x)],np.array(direcs[ii].split('_'))[[1,3,5]])
			true_data = data[ii]
			use_Nmodes = 5
			z_select = np.arange(z_len)[::2]		# Which redshifts to plot? Indexed from z_array
			data_select = map(lambda x: range(y_len*x,y_len*x+y_len),z_select)
			y_data = yz_data.T[0].T.ravel()
			z_data = yz_data.T[1].T.ravel()
			k_array = np.round(k_range,2)
			g_range = np.arange(g_len)

			use_true_a_ij = False

			# Construct signal from eigenmodes
			KL.calc_eigenmodes(param_vals)
			if use_Nmodes == 0:
				rec_data = fid_data.ravel()
			else:
				rec_data = fid_data.ravel() + reduce(operator.add,KL.eig_modes[0][:use_Nmodes])

			rec_true = fid_data.ravel() + sum(KL.a_ij[ii].reshape(N_modes,1) * KL.eig_vecs)
			if use_true_a_ij == True: rec_data = rec_true

			# vars
			xlim = (0.01,10)
			ylim = (0,300)
			ylim2 = (-.05,.05)

			# Plot
			fig = mp.figure(figsize=(12,10))
			gs = gridspec.GridSpec(z_len*2,k_len + g_len*2)
			ax1 = fig.add_subplot(gs[:z_len,:k_len])
			ax2 = fig.add_subplot(gs[z_len:,:k_len],sharex=ax1)
			axA = [fig.add_subplot(gs[:z_len,k_len+2*x:k_len+2*x+2]) for x in g_range]
			axB = [fig.add_subplot(gs[z_len:,k_len+2*x:k_len+2*x+2]) for x in g_range]
			fig.subplots_adjust(wspace=6,hspace=8)

			plots = []
			for i in range(len(z_select)):
				true_data_i = true_data[data_select[i]]
				rec_data_i = rec_data[data_select[i]]

				# Plot True and Recovered Power Spectra
				p0, = ax1.plot(k_array,true_data_i[:k_len],linewidth=6,alpha=0.2)
				p1, = ax1.plot(k_array,rec_data_i[:k_len],linewidth=1,alpha=0.9,color=p0.get_color())
				# Plot Fractional Error of Power Spectra
				p2, = ax2.plot(k_array,(rec_data_i[:k_len]-true_data_i[:k_len])/true_data_i[:k_len],color=p0.get_color(),marker='o',linestyle='-',alpha=0.4)

				# Plot True and Recovered Global Parameters, Fractional Error
				for j in range(g_len):
					p3, = axA[j].plot(g_range[j],true_data_i[k_len:][j],marker='o',markersize=7,linestyle='',alpha=0.25)
					p4, = axA[j].plot(g_range[j],rec_data_i[k_len:][j],marker='o',markersize=3,linestyle='',alpha=0.9,color=p3.get_color())
					axA[j].set_xticks([g_range[j]])
					axA[j].set_xticklabels([g_array_tex[j]])

					p5, = axB[j].plot(g_range[j],(rec_data_i[k_len:][j]-true_data_i[k_len:][j])/true_data_i[k_len:][j],marker='o',\
						markersize=3,linestyle='',alpha=0.6,color=p3.get_color())
					axB[j].set_xticks([g_range[j]])
					axB[j].set_xticklabels([g_array_tex[j]])
					axB[j].set_ylim(ylim2)

				plots.append(p1)

			ax1.set_ylabel(r'$\Delta^{2}(k)$ (mK$^{2}$)',fontsize=16)
			ax2.set_xlabel(r'$k$ ($h$ Mpc$^{-1}$)',fontsize=16)
			ax2.set_ylabel(r'Fractional Error',fontsize=16)
			ax1.legend(plots,map(lambda x: 'z='+str(x),z_array[z_select]),loc=2)

			if use_Nmodes == 0:
				ax1.annotate('ps_fid',xy=(0.4,0.1),xycoords='axes fraction',fontsize=13)
			else:
				ax1.annotate('ps_fid + EigenModes[1-'+str(use_Nmodes)+']',xy=(0.4,0.1),xycoords='axes fraction',fontsize=13)

			ax1.set_xscale('log')
			ax1.set_yscale('log')
			ax1.set_xlim(xlim)
			ax1.set_ylim(ylim)
			ax2.set_xscale('log')
			ax2.set_xlim(xlim)
			ax2.set_ylim(ylim2)

			fig.savefig('Nsamp'+str(N_samples)+'_samp'+str(ii)+'_eigmode'+str(use_Nmodes)+'_recon.png',dpi=100,bbox_inches='tight')
			mp.close()

	if plot_rms_err == True:
		# Draw from Training Set
		samps = np.arange(N_samples)
		param_vals = np.array(map(lambda x: np.array(x,float),map(lambda y: np.array(y.split('_'))[[1,3,5]],direcs))).T

		# Set reconstruction params
		use_Nmodes = 5

		# Iterate over poly_deg for polynomial regression
		error = []
		poly_degs = np.arange(0,10)
		for p in poly_degs:
			print 'poly_deg =',p
			KL.update({'poly_deg':p,'reg_meth':'poly'})
			KL.klinterp(data,gridf,fid_data=fid_data,scale_by_std=False)

			# Construct signal from eigenmodes
			KL.calc_eigenmodes(param_vals)
			rec_data = np.copy(KL.recon)

			# Estimate rms error across data for each sample
			rms = np.array(map(lambda x: np.sqrt(np.median(x)), ((rec_data-data)/data)**2))
			error.append(rms)

		# Do GP Regression
		print 'GP regress'
		KL.update({'reg_meth':'gaussian'})
		KL.klinterp(data,gridf,fid_data=fid_data,scale_by_std=False,calc_noise=True)		

		# Construct signal from eigenmodes
		KL.calc_eigenmodes(param_vals)
		rec_data = np.copy(KL.recon)

		# Estimate rms error across data for each sample
		rms = np.array(map(lambda x: np.sqrt(np.median(x)), ((rec_data-data)/data)**2))
		error.append(rms)

		### Get Mean Errors ###
		mean_err = np.array(map(np.median,error))

		## Plot ##
		fig,ax = mp.subplots()

		x_ax = np.concatenate([poly_degs,[poly_degs.max()+1]])
		ax.plot(x_ax,mean_err,'ko')
		ax.grid(True)
		ax.set_xlabel('polynomial degree',fontsize=15)
		ax.set_ylabel('mean of RMS error for TS samples',fontsize=15)
		ax.set_xlim(poly_degs.min()-1,poly_degs.max()+2)
		ax.set_yscale('log')
		ax.annotate('Nmodes='+str(use_Nmodes)+', Nsamples='+str(N_samples),xy=(0.6,0.1),xycoords='axes fraction')
		mp.xticks(np.arange(len(x_ax)),np.concatenate([np.arange(len(x_ax)-1),['GP']]))
		ax.set_ylim(1e-5,1)
	
		fig.savefig('rms_err_Nmodes'+str(use_Nmodes)+'_Nsamps'+str(N_samples)+'.png',dpi=100,bbox_inches='tight')
		mp.close()

	if plot_error_map == True:
		print '...plotting error map'
		def err_map(par1,par2):
			# Run Interp and Cross Valid
			theta0    = np.array([0.1,0.1,0.1])
			gp_kwargs = {'regr':'linear','theta0':theta0,'thetaL':theta0/10,'thetaU':theta0*10,\
					'random_start':1,'verbose':True,'corr':'squared_exponential'}
			KL.update({'gp_kwargs':gp_kwargs,'N_modes':5,'reg_meth':'gaussian'})
			KL.klinterp(data_tr,grid_tr,fid_data=fid_data,fid_params=fid_params,\
					scale_by_std=False,calc_noise=False)
			KL.cross_valid(data_cv,grid_cv,fid_data=fid_data,fid_params=fid_params)

			# Get errors
			errors = np.array(map(np.median,data_cv - KL.recon))
			resid = grid_cv - fid_params

			# Choose parameters to plot
			param1 = params[par1]
			param2 = params[par2]

			# Transform grid and fit kmeans
			L = KL.L[[par1,par2]].T[[par1,par2]].T
			invL = KL.invL[[par1,par2]].T[[par1,par2]].T
			norm_grid_cv = np.dot(invL,resid.T[[par1,par2]]).T
			n_clus = 100
			kmeans = KMeans(init='k-means++',n_clusters=n_clus,n_init=10)
			kmeans.fit(norm_grid_cv)

			# Initialize Grid, Normalize and Predict KMeans
			x_lim = np.mean(np.abs([resid.T[par1].min(),resid.T[par1].max()])) * 1.5
			y_lim = np.mean(np.abs([resid.T[par2].min(),resid.T[par2].max()])) * 1.5
			x_min,x_max=-x_lim,x_lim
			y_min,y_max=-y_lim,y_lim
			h = 100
			xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
			mesh_cv = np.c_[xx.ravel(), yy.ravel()]

			norm_mesh_cv = np.dot(invL,mesh_cv.T).T
			mesh_labels = kmeans.predict(norm_mesh_cv)
			shape = xx.shape
			mesh_labels = np.array(mesh_labels.reshape(shape),float)

			# Organize cells and assign cell error
			cell_errs = np.zeros(n_clus)
			mesh_errs = np.zeros(shape)
			for i in range(n_clus):
			    clus_sel = np.where(kmeans.labels_==i)[0]
			    mesh_sel = np.where(mesh_labels==i)
			    err=np.mean(errors[clus_sel])
			    cell_errs[i] = err
			    mesh_errs[mesh_sel]=err

			# Plot
			fig=mp.figure()
			ax = fig.add_subplot(111)
			xmin,xmax = x_min+fid_params[par1],x_max+fid_params[par1]
			ymin,ymax = y_min+fid_params[par2],y_max+fid_params[par2]
			aspect = x_lim/y_lim
			cax = ax.imshow(np.abs(mesh_errs),origin='lower',cmap='bone_r',interpolation='nearest',\
				    extent=(xmin,xmax,ymin,ymax),vmax=0.02,vmin=0,aspect=aspect)
			cbar = fig.colorbar(cax,label='standard error')
			ax.plot(grid_cv.T[par1],grid_cv.T[par2],color='r',marker='.',\
				    linestyle='',markersize=10,alpha=0.15)
			ax.set_xlim(xmin,xmax)
			ax.set_ylim(ymin,ymax)
			ax.set_xlabel(param1,fontsize=15)
			ax.set_ylabel(param2,fontsize=15)
			ax.grid(True)

			fig.savefig('ErrorMap_'+param2+'_'+param1+'_Ntrain'+str(N_samples)+'.png',dpi=100,bbox_inches='tight')
			mp.close()

		par1 = 1
		par2 = 0
		err_map(par1,par2)






