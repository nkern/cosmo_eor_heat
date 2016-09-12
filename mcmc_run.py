"""
==========
mcmc_run.py
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
import DictEZ as dez
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
import astropy.stats as astrostats
import scipy.stats as stats
import corner
import warnings
from pycape.toolbox import workspace
from pycape import common_priors
import time
warnings.filterwarnings('ignore',category=DeprecationWarning)


try:
	from IPython import get_ipython
	ipython = get_ipython()
except: pass

## Flags

## Program
if __name__ == "__main__":

	def print_message(string,type=1):
		str_len = len(string) + 5
		if type == 0:
			print '\n'+string
		elif type == 1:
			print '\n'+string+'\n'+'-'*str_len
		elif type == 2:
			print '-'*str_len+'\n'+string+'\n'+'-'*str_len

	def print_time():
		ellapsed_time = int(np.around(time.time() - start_time,0))
		print_message('...ellapsed time = '+str(ellapsed_time)+' seconds',type=0)

	start_time = time.time()

	############################################################
	### Load and Separate Training and Cross Validation Sets ###
	############################################################
	print_message('...loading and separating data')
	print_time()

	# Load Datasets
	file = open('gauss_hera127_data.pkl','rb')
	gauss_hera127_data = pkl.Unpickler(file).load()
	file.close()

	file = open('gauss_hera331_data.pkl','rb')
	gauss_hera331_data = pkl.Unpickler(file).load()
	file.close()

	file = open('cross_valid_data.pkl','rb')
	cross_valid_data = pkl.Unpickler(file).load()
	file.close()

	file = open('fiducial_data.pkl','rb')
	fiducial_data = pkl.Unpickler(file).load()
	file.close()

	# Separate and Draw Data
	def draw_data():
		make_globals = ['TS_data','tr_len','rando','data_tr','grid_tr','direcs_tr',\
					'CV_data','data_cv','grid_cv','direcs_cv','fid_params','fid_data']
		# Choose Training Set
		TS_data = gauss_hera127_data

		# Separate Data
		tr_len = 2000
		rando = np.random.choice(np.arange(tr_len),size=0,replace=False)
		rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))

		data_tr = TS_data['data'][np.argsort(TS_data['indices'])][rando]
		grid_tr = TS_data[ 'gridf'][np.argsort(TS_data['indices'])][rando]
		direcs_tr = TS_data['direcs'][np.argsort(TS_data['indices'])][rando]

		# Add other datasets
		add_other_data = True
		if add_other_data == True:
			# Choose Training Set
			TS_data = gauss_hera331_data

			# Separate Data
			tr_len = 2000
			rando = np.random.choice(np.arange(tr_len),size=2000,replace=False)
			rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))

			data_tr = np.concatenate([data_tr,TS_data['data'][np.argsort(TS_data['indices'])][rando]])
			grid_tr = np.concatenate([grid_tr,TS_data[ 'gridf'][np.argsort(TS_data['indices'])][rando]])
			direcs_tr = np.concatenate([direcs_tr,TS_data['direcs'][np.argsort(TS_data['indices'])][rando]])

		# Choose Cross Validation Set
		CV_data = cross_valid_data
		TS_remainder = False
		use_remainder = False

		# Separate Data
		if TS_remainder == True:
			if use_remainder == True:
				rando = ~rando
			else:
				remainder = np.where(rando==False)[0]
				rando = np.array([False for i in range(tr_len)])
				rando[np.random.choice(remainder,size=500,replace=False)] = True
			
		else:
			tr_len = 550
			rando = np.random.choice(np.arange(tr_len),size=550,replace=False)
			rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))

		data_cv = CV_data['data'][np.argsort(CV_data['indices'])][rando]
		grid_cv = CV_data[ 'gridf'][np.argsort(CV_data['indices'])][rando]
		direcs_cv = np.array(CV_data['direcs'])[np.argsort(CV_data['indices'])][rando]
		
		# Get Fiducial Data
		feed_fid = True
		if feed_fid == True:
			fid_params = fiducial_data['gridf']
			fid_data = fiducial_data['data']
		else:
			fid_params = np.array(map(np.median,grid_tr.T))
			fid_data = np.array(map(np.median,data_tr.T))
			
		globals().update(dez.create(make_globals,locals()))

	np.random.seed(1)
	draw_data()
	print_time()

	# Get minimum distance between any two points, assuming they have already been decorrelated and normalized
	def get_min_R(samples):
		minR = 100.0
		maxR = 0.01
		for i in np.arange(len(samples)-1):
			for j in np.arange(i+1,len(samples)):
				if j == i: continue
				this_R = la.norm(samples[i]-samples[j])
				if this_R < minR: minR = 1*this_R
				if this_R > maxR: maxR = 1*this_R
		return minR, maxR

	def sphere_minR(samples,cov_samp,fid_grid):
		X = cov_samp - fid_grid
		cov = np.cov(X.T)
		L = la.cholesky(cov)
		invL = la.inv(L)
		sph_X = np.dot(invL,samples.T).T
		minR,maxR = get_min_R(sph_X)
		return minR,maxR

	def sphere_getRad(samples,cov_samp,fid_grid):
		X = cov_samp - fid_grid
		cov = np.cov(X.T)
		L = la.cholesky(cov)
		invL = la.inv(L)
		sph_X = np.dot(invL,samples.T).T
		Rad = np.array(map(la.norm,sph_X))
		return Rad

	optimize_tr = False
	if optimize_tr == True:
		print_message('...optimizing ts')
		print_time()
		seed = []
		minR = []
		rando_seeds = np.random.choice(np.arange(100000),size=50,replace=False)
		for i in range(len(rando_seeds)):
			np.random.seed(rando_seeds[i])
			draw_data()
			minr,maxr = sphere_minR(grid_tr,gauss_data['gridf']-fid_params,fid_params)
			minR.append(minr)
			seed.append(rando_seeds[i])
			if i % 10 == 0: print '\r...'+str(i),

		print    
		seed = np.array(seed)
		minR = np.array(minR)
		index = np.where(minR==minR.max())[0][0]
		print 'maximum minR index: ',index
		print 'maximum minR: ',minR[index]
		print 'seed at max minR: ',seed[index]
		np.random.seed(seed[index])
		draw_data()
		print_time()

	# Separate Data
	separate_data = False
	if separate_data == True:
		# Separate Data By Dimensions

		#data_tr = data_tr.reshape(7,7,7,176)[::2,::2,::2,:].reshape(4**3,176)
		#grid_tr = grid_tr.reshape(7,7,7,3)[::2,::2,::2,:].reshape(4**3,3)
		#direcs_tr = direcs_tr.reshape(7,7,7)[::2,::2,::2].reshape(4**3)

		data_cv = np.vstack([data_cv[10:40],data_cv[60:90],data_cv[110:140]])
		grid_cv = np.vstack([grid_cv[10:40],grid_cv[60:90],grid_cv[110:140]])
		direcs_cv = np.hstack([direcs_cv[10:40],direcs_cv[60:90],direcs_cv[110:140]])

		#data_cv = TS_data['data'][TS_data['indices']][:tr_len][~rando]
		#grid_cv = TS_data[ 'gridf'][TS_data['indices']][:tr_len][~rando]
		#direcs_cv = TS_data['direcs'][TS_data['indices']][:tr_len][~rando]

	# Plot Training Set
	plot_tr = True
	if plot_tr == True:
		print_message('...plotting ts')
		print_time()
		fig = mp.figure(figsize=(15,8))
		fig.subplots_adjust(wspace=0.3)

		lims = [[None,None] for i in range(11)]

		j = 0
		for i in range(6):
			ax = fig.add_subplot(2,3,i+1)
			ax.plot(grid_tr.T[j],grid_tr.T[j+1],'k,',alpha=0.75)
			ax.plot(grid_cv.T[j],grid_cv.T[j+1],'r,')
			ax.set_xlim(lims[j])
			ax.set_ylim(lims[j+1])
			ax.set_xlabel(p_latex[j],fontsize=16)
			ax.set_ylabel(p_latex[j+1],fontsize=16)
			if i == 0:
				j += 1
			else:
				j += 2
		fig.savefig('ts.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()

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
	N_modes = 50
	N_params = len(params)
	N_data = 660
	N_samples = len(data_tr)
	poly_deg = 2
	reg_meth='gaussian'
	theta0 = np.array([0.1 for i in range(N_params)])
	thetaU = None#theta0*10.
	thetaL = None#theta0/10.

	gp_kwargs = {'regr':'linear','theta0':theta0,'thetaL':thetaL,'thetaU':thetaU,
					'random_start':1,'verbose':False,'corr':'squared_exponential'}

	variables.update({'params':params,'N_params':N_params,'N_modes':N_modes,'N_samples':N_samples,'N_data':N_data,
						'reg_meth':reg_meth,'poly_deg':poly_deg,'gp_kwargs':gp_kwargs})

	workspace_init = {'dir_pycape':'/global/homes/n/nkern/Software/pycape',
					  'dir_21cmSense':'/global/homes/n/nkern/Software/21cmSense'}


	###########################
	### INITIALIZE EMULATOR ###
	###########################

	# Initialize workspace, emulator
	W = workspace(workspace_init)
	W.emu_init(variables)

	# Initialize Cholesky
	W.E.sphere(grid_tr,fid_params=fid_params,save_chol=True)
	fid_params = W.E.fid_params

	# Initialize Tree
	W.E.create_tree(W.E.Xsph,metric='euclidean',leaf_size=5*N_params)

	# Initialize Observation Class
	obs_dic = dict(zip(workspace_init.keys(),workspace_init.values()))
	W.obs_init(obs_dic)


	print_message('...initializing mockobs')
	print_time()

	# Load Mock Observation and Configure
	file = open('mockObs_hera331_allz.pkl','rb')
	mock_data = pkl.Unpickler(file).load()
	file.close()
	p_true = params_fid

	### Variables for Mock Observation ###
	z_array         = np.arange(5.5,27.01,0.5)
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
				err_thresh = 1.0        # 200%
				small_errs = np.where(mock_data['sense_PSerrs'][i] / mock_data['sense_PSdata'][i] < err_thresh)[0]
				mock_data['sense_kbins'][i] = mock_data['sense_kbins'][i][small_errs]
				mock_data['sense_PSdata'][i] = mock_data['sense_PSdata'][i][small_errs]
				mock_data['sense_PSerrs'][i] = mock_data['sense_PSerrs'][i][small_errs]

	mock_data['sense_kbins'] = np.array( map(lambda x: np.array(x,float),mock_data['sense_kbins']))
	mock_data['sense_PSdata'] = np.array(np.concatenate(mock_data['sense_PSdata']),float)
	mock_data['sense_PSerrs'] = np.array(np.concatenate(mock_data['sense_PSerrs']),float)

	# If model data that comes with Mock Data does not conform to Predicted Model Data, interpolate!
	try:
		residual = k_range - mock_data['kbins'][1]
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
	update_obs_dic = {'N_data':sense_PSdata.size,'z_num':z_num,'model_lim':model_lim,'z_array':z_array,
						 'k_range':k_range}
	W.obs_update(update_obs_dic)
	W.obs_feed(kbins,sense_kbins,sense_PSdata,sense_PSerrs)
	print_time()

	# Interpolate Training Set onto Observational Basis
	print_message('...configuring data for emulation')
	print_time()
	data_tr = np.array(map(lambda x: W.obs_interp_mod2obs(x),data_tr))
	fid_data = W.obs_interp_mod2obs(fid_data)

	data_cv = np.array(map(lambda x: W.obs_interp_mod2obs(x),data_cv))

	# Initialize KLT
	W.E.klt(data_tr,fid_data=fid_data)

	# Make new yz_data matrix
	yz_data = []
	for i in range(z_num):
		kdat = np.array(np.around(W.Obs.x[i],3),str)
		zdat = np.array([z_array[i] for j in range(len(W.Obs.x[i]))],str)
		yz_data.append(np.array(zip(kdat,zdat)))
	yz_data = np.array(yz_data)

	print_time()

	###### Set Parameter values
	print_message('...configuring emulator and sampler')
	print_time()
	k = 5
	use_pca = True
	emode_variance_div = 30.0
	calc_noise = False
	norm_noise = False
	fast = False
	compute_klt = False
	save_chol = False
	LAYG = False
	LAYG_pretrain = False
	NN_hyper = False
	NN_hyper_k = 30

	sample_noise = 1e-5 * np.linspace(1,100,N_modes).reshape(N_modes,1)

	# Solve for GP Hyperparameters for every grid point
	#theta0 = np.array([1.0,1.0,1.0])
	#theta0 = np.array([0.1,0.1,0.1])
	#theta0 = np.array([0.17058725,  0.0139115,   0.10627892])
	theta0 = np.array([[0.1 for i in range(N_params)] for i in range(N_modes)]) *\
						np.linspace(1.0,10.0,N_modes).reshape(N_modes,1)
	thetaL = theta0/10
	thetaU = theta0*10
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

	W.E.group_eigenmodes(emode_variance_div=emode_variance_div)
		
	gp_kwargs = {'regr':'linear','theta0':theta0,'thetaL':thetaL,'thetaU':thetaU,'random_start':1,'verbose':False,
				 'optimizer':'fmin_cobyla','normalize':True,'corr':'squared_exponential','nugget':sample_noise}

	gp_kwargs_arr = np.array([gp_kwargs.copy() for i in range(W.E.N_modegroups)])
	insert(gp_kwargs_arr,'theta0',theta0)
	insert(gp_kwargs_arr,'thetaU',thetaU)
	insert(gp_kwargs_arr,'thetaL',thetaL)
	insert(gp_kwargs_arr,'nugget',sample_noise)

	kwargs_tr = {'use_pca':use_pca,'scale_by_std':False,'calc_noise':calc_noise,'norm_noise':norm_noise,
				 'sample_noise':sample_noise,'verbose':False,'invL':W.E.invL,'emode_variance_div':emode_variance_div,
				 'fast':fast,'compute_klt':compute_klt,'save_chol':save_chol,
				 'gp_kwargs_arr':gp_kwargs_arr}

	### Initialize Sampler Variables ###
	predict_kwargs = {'fast':fast,'use_Nmodes':None,'use_pca':use_pca}

	param_width = np.array([grid_tr.T[i].max() - grid_tr.T[i].min() for i in range(N_params)])

	eps = -0.1

	param_bounds = np.array([[grid_tr.T[i].min()+param_width[i]*eps,grid_tr.T[i].max()-param_width[i]*eps]\
								 for i in range(N_params)])

	param_hypervol = reduce(operator.mul,map(lambda x: x[1] - x[0], param_bounds))

	use_Nmodes = None
	add_model_err = False
	calc_lnlike_emu_err = True
	cut_high_fracerr = 10.0
	emu_err_mc = False
	ndim = N_params
	nwalkers = 40

	sampler_init_kwargs = {'use_Nmodes':use_Nmodes,'param_bounds':param_bounds,'param_hypervol':param_hypervol,
							'nwalkers':nwalkers,'ndim':ndim,'N_params':ndim,'z_num':z_num}
	sampler_kwargs = {}

	lnprob_kwargs = {'add_model_err':add_model_err,'fast':fast,'kwargs_tr':kwargs_tr,
					 'calc_lnlike_emu_err':calc_lnlike_emu_err,'predict_kwargs':predict_kwargs,'LAYG':LAYG,'k':k,
					 'LAYG_pretrain':LAYG_pretrain,'emu_err_mc':emu_err_mc,'cut_high_fracerr':cut_high_fracerr}

	W.samp_init(sampler_init_kwargs)
	print_time()


	train_emu = True
	if train_emu == True:
		print_message('...training emulator')
		print_time()
		W.emu_train(W.E.data_tr,W.E.grid_tr,fid_data=W.E.fid_data,fid_params=W.E.fid_params,kwargs_tr=kwargs_tr)
		print_time()

	def cross_validate():
		print_message('...cross validating power spectra')
		print_time()

		## Cross Validate
		recon = []
		recon_err = []
		weights = []
		true_w = []
		weights_err = []
		for i in range(len(data_cv)):
			W.E.N_samples = k
			W.samp_lnprob(grid_cv[i],**lnprob_kwargs)
			r = W.E.recon.ravel()
			rerr = W.E.recon_err.ravel()
			w = W.E.weights.ravel()
			werr = W.E.weights_err.ravel()
			tw = np.dot(data_cv[i]-W.E.fid_data,W.E.eig_vecs.T)
			recon.append(r)
			recon_err.append(rerr)
			weights.append(w)
			true_w.append(tw)
			weights_err.append(werr)
			#if i%100==0: print i

		recon = np.array(recon)
		recon_err = np.array(recon_err)
		weights = np.array(weights)
		true_w = np.array(true_w)
		weights_err = np.array(weights_err)
		print_time()
		return recon, recon_err, weights, true_w, weights_err

	cross_validate_ps = True
	if cross_validate_ps == True:
		print_message('...cross validating power spectra')
		print_time()

		recon, recon_err, weights, true_w, weights_err = cross_validate()

		fig = mp.figure(figsize=(16,4))

		# Scree Plot
		ax = fig.add_subplot(131)
		ax.plot(W.E.eig_vals,'ko')
		ax.set_yscale('log')
		ax.set_xlim(-1,N_modes+1)
		ax.set_xlabel('eigenmode #',fontsize=15)
		ax.set_ylabel('eigenvalue',fontsize=15)
		ax.grid(True)

		# Cross Validation Error
		#z_arr = np.array(yz_data.reshape(91,2).T[1],float)
		#good_z = (z_arr > 7) & (z_arr < 25)
		#W.E.model_lim[~good_z] = False

		err = (recon-data_cv)/data_cv
		pred_err = recon_err / data_cv
		mean_err = np.array(map(lambda x: np.median(x),err.T))
		central_err = err[np.where(np.abs(err)<0.2)]
		std = np.std(central_err)
		rms_err = np.sqrt(astrostats.biweight_location(err.ravel()**2))
		rms_std_err = astrostats.biweight_midvariance(err.ravel())

		ax = fig.add_subplot(132)
		p1 = ax.hist(np.abs(err.ravel()),histtype='step',color='b',linewidth=1,bins=75,range=(-0.01,1.5),normed=True,alpha=0.75)
		p2 = ax.hist(pred_err.ravel(),histtype='step',color='r',linewidth=1,bins=75,range=(-.01,1.5),normed=True,alpha=0.5)
		#ax.axvline(rms,color='r',alpha=0.5)
		#ax.axvline(-rms,color='r',alpha=0.5)
		#ax.hist(rms_obs_err,histtype='step',color='m',bins=50,range=(0,0.2),alpha=0.5,normed=True)
		ax.set_xlim(-0.001,1.5)
		ax.set_ylim(0,2)
		ax.set_xlabel('Fractional Error',fontsize=15)
		ax.annotate('rms err = '+str(np.around(rms_err*100,2))+'%\n rms standard error ='+str(np.around(rms_std_err*100,2))+'%',
			xy=(0.5,0.8),xycoords='axes fraction')

		# Get CV error at each redshift
		recon_mat = np.array(map(lambda x: W.obs_mat2row(x,mat2row=False), recon))
		cv_mat    = np.array(map(lambda x: W.obs_mat2row(x,mat2row=False), data_cv))
		frac_err = (recon_mat-cv_mat)/cv_mat
		frac_err_mat = []
		for i in range(z_num):
			avg_err_kbins = np.array(map(lambda x: np.sqrt(np.median(x**2)),np.array([frac_err.T[i][j] for j in range(len(data_cv))]).T))
			frac_err_mat.append(avg_err_kbins)
		frac_err_mat = np.array(frac_err_mat)

		zarr_vec = W.obs_mat2row(np.array([[z_array[i]]*len(W.Obs.x[i]) for i in range(z_num)]),mat2row=True)
		frac_err_vec = W.obs_mat2row(frac_err_mat,mat2row=True)

		ax = fig.add_subplot(133)
		im = ax.scatter(W.Obs.x_ext,zarr_vec,c=frac_err_vec*100,marker='o',s=35,edgecolor='',alpha=0.75,cmap='nipy_spectral_r',vmax=10)
		ax.set_xscale('log')
		ax.set_xlim(1e-1,3)
		ax.set_xlabel(r'$k$',fontsize=17)
		ax.set_ylabel(r'$z$',fontsize=17)
		fig.colorbar(im,label='Avg % Error')
		fig.savefig('cross_validate_ps.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()

	cross_validate_like = True
	add_lnlike_weight = True
	if cross_validate_like == True:
		print_message('...cross validating likelihoods')
		print_time()

		e_like,t_like,o_vars = W.samp_cross_valid(grid_cv,data_cv,lnlike_kwargs=lnprob_kwargs,also_record=['lnlike_emu_err'])
		fig = mp.figure(figsize=(5,5))
		ax = fig.add_subplot(111)
		frac_err = (e_like-t_like)/t_like
		patches = ax.hist(frac_err,bins=20,histtype='step',range=(-2.0,2.0),normed=True,color='b')
		ax.set_xlim(-2.0,2.0)
		ax.set_xlabel('likelihood fractional error',fontsize=14)
		ax.annotate('robust err std = '+str(np.around(astrostats.biweight_midvariance(frac_err)*100,2))+'%',xy=(0.5,0.8),xycoords='axes fraction')
		fig.savefig('cross_validate_like.png',dpi=100,bbox_inches='tight')

		if add_lnlike_weight == True:
			print_message('...adding lnlike cross validated errors to lnlike covariance as weights')
			print_time()
			## Add emulator error to lnlikelihood weight vector
			rel_weight_vec = (frac_err_vec/np.min(frac_err_vec))
			weight_constant = 0.5
			add_lnlike_cov = W.S.data_cov * np.eye(W.Obs.cov.shape[0])*(frac_err_vec*rel_weight_vec*weight_constant)**2
			lnprob_kwargs['add_lnlike_cov'] = add_lnlike_cov

			e_like,t_like,o_vars = W.samp_cross_valid(grid_cv,data_cv,lnlike_kwargs=lnprob_kwargs,also_record=['lnlike_emu_err'])
			fig = mp.figure(figsize=(5,5))
			ax = fig.add_subplot(111)
			frac_err = (e_like-t_like)/t_like
			patches = ax.hist(frac_err,bins=20,histtype='step',range=(-2.0,2.0),normed=True,color='r')
			ax.set_xlim(-2.0,2.0)
			ax.set_xlabel('likelihood fractional error',fontsize=14)
			ax.annotate('robust err std = '+str(np.around(astrostats.biweight_midvariance(frac_err)*100,2))+'%',xy=(0.5,0.8),xycoords='axes fraction')
			fig.savefig('cross_validate_like2.png',dpi=100,bbox_inches='tight')
		print_time()

	regress_for_hyperparams = False
	if regress_for_hyperparams == True:
		print_message('...regressing for optimal GP hyperparameters')
		print_time()

		ph_cv = np.dot(W.E.invL,(grid_cv-W.E.fid_params).T).T
		theta0_test = 10**np.linspace(-2.0,1.0,45)
		p_ind = 0

		# Edit gp_kwargs_arr
		for i in range(W.E.N_modegroups):
			gp_kwargs_arr[i]['thetaL'] = None
			gp_kwargs_arr[i]['thetaU'] = None
			
		# Get all the emulated and true weights from cv samples
		w_em = []
		w_cv = []
		precision = []

		# Iterate over theta0 choices
		for i in range(len(theta0_test)):
			# Assign theta0 test values for each GP
			for k in range(W.E.N_modegroups):
				gp_kwargs_arr[k]['theta0'][p_ind] = theta0_test[i]

			kwargs_tr['gp_kwargs_arr'] = gp_kwargs_arr

			# Train emulator with updated theta0 value
			W.emu_train(W.E.data_tr,W.E.grid_tr,fid_data=W.E.fid_data,fid_params=W.E.fid_params,kwargs_tr=kwargs_tr)

			# Get emulated and true weights over all cv samples
			w_em_theta0 = []
			for j in range(len(grid_cv[0:30])):
				W.E.cross_validate(grid_cv[j],data_cv[j],predict_kwargs=predict_kwargs)
				w_em_theta0.append(W.E.weights_cv[0])
				if i == 0: w_cv.append(W.E.a_ij_cv)
				if i == 0: precision.append(np.abs(X_sph_cv[j][p_ind]))
				
			w_em.append(w_em_theta0)

		w_em = np.array(w_em)
		w_cv = np.array(w_cv)
		precision = (np.array(precision)/np.min(precision))**(-1.0)

		def cost(w_em,w_cv,theta0,precision=None,reg_coeff=10.0):
			chisq = (w_em-w_cv)**2
			if precision is not None:
				chisq *= precision

			chisq = np.sum(chisq) + theta0*reg_coeff
			return chisq

		reg_coeff = 0.1
		chisq = []
		for k in range(W.E.N_modegroups):
			# iterate over the theta0 test values
			modegroup_chisq = []
			for i in range(len(theta0_test)):
				modegroup_chisq.append( cost(w_em[i,:,:].T[W.E.modegroups[k]].T.ravel(), 
							w_cv.T[W.E.modegroups[k]].T.ravel(),
							theta0_test[i],
							precision=np.array(list(precision)*len(W.E.modegroups[k])),
							reg_coeff=reg_coeff) )
				
			chisq.append(modegroup_chisq)

		chisq = np.array(chisq)

		# Get best-fit eigenmode hyperparameters
		theta0_bf = []
		for i in range(N_modes):
			theta0_bf.append( theta0_test[np.where(chisq[i]==chisq[i].min())[0][0]] )
		theta0_bf = np.array(theta0_bf)

		# Insert best-fit hyperparams into GP_kwargs, retrain
		theta0.T[p_ind] = theta0_bf
		insert(gp_kwargs_arr,'theta0',theta0)
		if LAYG == False:
			W.emu_train(W.E.data_tr,W.E.grid_tr,fid_data=W.E.fid_data,fid_params=W.E.fid_params,kwargs_tr=kwargs_tr)


		cmap = mp.cm.get_cmap('rainbow',len(chisq))
		colors = cmap(np.linspace(0,1,len(chisq)))
		norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=len(chisq)-0.5)

		fig = mp.figure(figsize=(6,6))

		ax = fig.add_subplot(111)
		for i in range(len(chisq))[::1]:
			ax.plot(theta0_test,chisq[i],alpha=0.75,color=colors[i])
			ax.axvline(theta0_bf[i],color=colors[i],alpha=0.5)
			
		ax.set_yscale('log')
		ax.set_xscale('log')
		ax.set_xlabel('theta0',fontsize=15)
		ax.set_ylabel(r'$\chi^{2}$',fontsize=16)
		#ax.set_ylim(0,500)

		cax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
		cb1 = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm,orientation='vertical')
		cb1.set_ticks(np.arange(len(chisq)))
		cb1.set_label('Eigenmode #',fontsize=13)
		fig.savefig('GPhyperparams.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()

	cross_validate_ps = False
	if cross_validate_ps == True:
		print_message('...cross validating power spectra')
		print_time()

		recon, recon_err, weights, true_w, weights_err = cross_validate()

		fig = mp.figure(figsize=(16,4))

		# Scree Plot
		ax = fig.add_subplot(131)
		ax.plot(W.E.eig_vals,'ko')
		ax.set_yscale('log')
		ax.set_xlim(-1,N_modes+1)
		ax.set_xlabel('eigenmode #',fontsize=15)
		ax.set_ylabel('eigenvalue',fontsize=15)
		ax.grid(True)

		# Cross Validation Error
		#z_arr = np.array(yz_data.reshape(91,2).T[1],float)
		#good_z = (z_arr > 7) & (z_arr < 25)
		#W.E.model_lim[~good_z] = False

		err = (recon-data_cv)/data_cv
		pred_err = recon_err / data_cv
		mean_err = np.array(map(lambda x: np.median(x),err.T))
		central_err = err[np.where(np.abs(err)<0.2)]
		std = np.std(central_err)
		rms_err = np.sqrt(astrostats.biweight_location(err.ravel()**2))
		rms_std_err = astrostats.biweight_midvariance(err.ravel())

		ax = fig.add_subplot(132)
		p1 = ax.hist(np.abs(err.ravel()),histtype='step',color='b',linewidth=1,bins=75,range=(-0.01,1.5),normed=True,alpha=0.75)
		p2 = ax.hist(pred_err.ravel(),histtype='step',color='r',linewidth=1,bins=75,range=(-.01,1.5),normed=True,alpha=0.5)
		#ax.axvline(rms,color='r',alpha=0.5)
		#ax.axvline(-rms,color='r',alpha=0.5)
		#ax.hist(rms_obs_err,histtype='step',color='m',bins=50,range=(0,0.2),alpha=0.5,normed=True)
		ax.set_xlim(-0.001,1.5)
		ax.set_ylim(0,2)
		ax.set_xlabel('Fractional Error',fontsize=15)
		ax.annotate('rms err = '+str(np.around(rms_err*100,2))+'%\n rms standard error ='+str(np.around(rms_std_err*100,2))+'%',
			xy=(0.5,0.8),xycoords='axes fraction')

		# Get CV error at each redshift
		recon_mat = np.array(map(lambda x: W.obs_mat2row(x,mat2row=False), recon))
		cv_mat    = np.array(map(lambda x: W.obs_mat2row(x,mat2row=False), data_cv))
		frac_err = (recon_mat-cv_mat)/cv_mat
		frac_err_mat = []
		for i in range(z_num):
			avg_err_kbins = np.array(map(lambda x: np.sqrt(np.median(x**2)),np.array([frac_err.T[i][j] for j in range(len(data_cv))]).T))
			frac_err_mat.append(avg_err_kbins)
		frac_err_mat = np.array(frac_err_mat)

		zarr_vec = W.obs_mat2row(np.array([[z_array[i]]*len(W.Obs.x[i]) for i in range(z_num)]),mat2row=True)
		frac_err_vec = W.obs_mat2row(frac_err_mat,mat2row=True)

		ax = fig.add_subplot(133)
		im = ax.scatter(W.Obs.x_ext,zarr_vec,c=frac_err_vec*100,marker='o',s=35,edgecolor='',alpha=0.75,cmap='nipy_spectral_r',vmax=10)
		ax.set_xscale('log')
		ax.set_xlim(1e-1,3)
		ax.set_xlabel(r'$k$',fontsize=17)
		ax.set_ylabel(r'$z$',fontsize=17)
		fig.colorbar(im,label='Avg % Error')
		fig.savefig('cross_validate_ps2.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()

	# Initialize Ensemble Sampler
	print_message('...initializing ensemble sampler')
	print_time()
	W.samp_emcee_init(lnprob_kwargs=lnprob_kwargs,sampler_kwargs=sampler_kwargs)

	# Initialize Walker positions
	pos = np.array(map(lambda x: x + x*stats.norm.rvs(0,0.05,nwalkers),p_true)).T

	# Add priors (other than flat priors)
	add_priors = True
	if add_priors == True:
		print_message('...adding non-flat priors',type=0)
		planck_cov = np.loadtxt('base_TTTEEE_lowl_plik.covmat')[[0,1,5]].T[[0,1,5]].T
		std_multiplier = 3.0

		# Add non-correlated Gaussian Priors
		prior_params = ['sigma8','H0']
		prior_indices = [0,1]
		priors = map(lambda x: common_priors.cmb_priors1[x+'_err'] * std_multiplier, prior_params)
		priors[1] /= 100.
		for i in range(len(priors)):
			W.S.lnprior_funcs[prior_indices[i]] = W.samp_gauss_lnprior(p_true[prior_indices[i]],priors[i],\
						index=prior_indices[i],return_func=True)
			
		# Add correlated Gaussian Priors
		prior_params = ['ombh2','omch2','ns']
		prior_indices = [2,3,4]
		Nindices = len(prior_indices)
		prior_cov = np.zeros((N_params,N_params))
		for i in range(Nindices):
			for j in range(Nindices):
				prior_cov[prior_indices[i],prior_indices[j]] = planck_cov[i,j] * std_multiplier**2
		for i in range(N_params):
			prior_cov[i,i] += 1e-12
		prior_prec = la.inv(prior_cov)            
		for i in range(len(prior_params)):
			W.S.lnprior_funcs[prior_indices[i]] = W.samp_cov_gauss_lnprior(p_true,prior_prec,\
					index=prior_indices[i],return_func=True)

	print_time()

	time_sampler = False
	if time_sampler == True:
		print_message('...timing sampler')
		ipython.magic("timeit -r 3 W.samp_drive(pos,step_num=1,burn_num=0)")

	drive_sampler = True
	if drive_sampler == True:
		print_message('...driving sampler',type=1)
		print_time()
		# Drive Sampler
		burn_num = 100
		step_num = 500
		print_message('...driving with burn_num='+str(burn_num)+', step_num='+str(step_num),type=0)
		W.samp_drive(pos,step_num=step_num,burn_num=burn_num)
		samples = W.S.sampler.chain[:, 0:, :].reshape((-1, W.S.ndim))
		print("Mean acceptance fraction: {0:.3f}".format(np.mean(W.S.sampler.acceptance_fraction)))
		print_time()

	trace_plots = True
	if trace_plots == True:
		print_message('...plotting trace plots')
		print_time()
		fig = mp.figure(figsize=(16,8))
		fig.subplots_adjust(wspace=0.4)

		for i in range(N_params):
			ax = fig.add_subplot(3,4,i+1)
			ax.set_ylabel(p_latex[i],fontsize=16)
			ax.axhline(p_true[i],color='k',alpha=0.25,linewidth=6)
			for j in range(nwalkers)[::nwalkers/10]:
				ax.plot(W.S.sampler.chain[j,:,i],alpha=0.3)
				ax.set_ylim(param_bounds[i])

		fig.savefig('trace_plots.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()

	tri_plots = True
	if tri_plots == True:
		print_message('...plotting triangle plots')
		print_time()
		# Triangle Plot
		levels = [0.68,0.95]

		p_eps = [0.1 for i in range(5)] + [0.2 for i in range(6)]
		p_lims = [[None,None] for i in range(N_params)]
		p_lims = [[params_fid[i]*(1-p_eps[i]),params_fid[i]*(1+p_eps[i])] for i in range(N_params)]

		label_kwargs = {'fontsize':18}

		print '...plotting triangle'
		fig = corner.corner(samples, labels=p_latex, label_kwargs=label_kwargs,
							truths=p_true, range=p_lims, levels=levels)

		plot_ts = False
		if plot_ts == True:
			print '...plotting TS'
			j = 1
			for i in range(N_params-1):
				xp_ind = i
				ax_ind = [N_params*j + i + k*N_params for k in range(N_params-j)]
				for l,h in zip(range(j,N_params),range(N_params-j)):
					yp_ind = l
					fig.axes[ax_ind[h]].plot(grid_tr.T[xp_ind],grid_tr.T[yp_ind],'r.',alpha=0.5)
					fig.axes[ax_ind[h]].plot(pos.T[xp_ind],pos.T[yp_ind],'c.',alpha=0.5)
				j += 1

		fig.savefig('tri_plot.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()










