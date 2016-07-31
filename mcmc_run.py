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

## Program
if __name__ == "__main__":

	def print_message(string,type=1):
		if type == 1:
			print '\n'+string+'-'*30
		elif type == 2:
			print '\n'+'-'*45+string+'-'*45

	############################################################
	### Load and Separate Training and Cross Validation Sets ###
	############################################################
	print_message('...loading and separating data')

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
		print_message('...plotting training set')

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


	# Initialize Gaussian Prior on Sigma8
	W.S.lnprior_funcs[0] = W.samp_gauss_lnprior(0.87,0.03,return_func=True)

	if LAYG == False:
		W.emu_train(W.E.data_tr,W.E.grid_tr,fid_data=W.E.fid_data,fid_params=W.E.fid_params,kwargs_tr=kwargs_tr)
		#print W.E.GP

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
		if i%100==0: print i
		#    print W.E.GP.theta0
		#    print W.E.GP.theta_
		#    print ''

	recon = np.array(recon)
	recon_err = np.array(recon_err)
	weights = np.array(weights)
	true_w = np.array(true_w)
	weights_err = np.array(weights_err)


	plot_err = False
	if plot_err == True:
		print_message('...plotting CV error')

		fig = mp.figure(figsize=(10,4))

		# Scree Plot
		ax = fig.add_subplot(121)
		ax.plot(W.E.eig_vals,'ko')
		ax.set_yscale('log')
		ax.set_xlim(-1,N_modes+1)
		ax.set_xlabel('eigenmode #',fontsize=15)
		ax.set_ylabel('eigenvalue',fontsize=15)
		ax.grid(True)

		# Cross Validation Error
		err = (recon.T[W.E.model_lim].T-data_cv.T[W.E.model_lim].T)/data_cv.T[W.E.model_lim].T
		pred_err = recon_err.T[W.E.model_lim].T / data_cv.T[W.E.model_lim].T
		mean_err = np.array(map(lambda x: np.median(x),err))
		central_err = err[np.where(np.abs(err)<0.2)]
		std = np.std(central_err)
		rms_obs_err = np.sqrt(np.array(map(lambda x: np.median(x**2),err)))
		rms = np.median(rms_obs_err)

		ax = fig.add_subplot(122)
		p1 = ax.hist(np.abs(err.ravel()),histtype='step',color='b',linewidth=1,bins=75,range=(-0.01,1.5),normed=True,alpha=0.75)
		p2 = ax.hist(pred_err.ravel(),histtype='step',color='r',linewidth=1,bins=75,range=(-.01,1.5),normed=True,alpha=0.5)
		#ax.axvline(rms,color='r',alpha=0.5)
		#ax.axvline(-rms,color='r',alpha=0.5)
		#ax.hist(rms_obs_err,histtype='step',color='m',bins=50,range=(0,0.2),alpha=0.5,normed=True)
		ax.set_xlim(-0.001,1.5)
		ax.set_xlabel('Fractional Error',fontsize=15)

		print 'rms_obs_err =', rms

		e_like,t_like,o_vars = W.samp_cross_valid(grid_cv,data_cv,lnlike_kwargs=lnprob_kwargs,also_record=['lnlike_emu_err'])
		fig = mp.figure(figsize=(5,5))
		ax = fig.add_subplot(111)
		frac_err = (e_like-t_like)/t_like
		#frac_err = o_vars['lnlike_emu_err']/np.abs(e_like)
		patches = ax.hist(frac_err,bins=20,histtype='step',range=(-0.19,0.19),normed=True)
		ax.set_xlim(-0.19,0.19)
		print 'robust std =',astrostats.biweight_midvariance(frac_err)
		print 'robust rms =',np.sqrt(np.median(frac_err**2))


	plot_lnslice = False
	if plot_lnslice == True:
		print_message('...plotting lnlike slice')

		## Look at likelihood surface
		xstep = 25
		x0step = 1
		x0range = [0.871019,0.871019]
		x1step = xstep
		x1range = [1000.,100000.]
		x2step = xstep
		x2range = [5.,70.]
		x0 = np.linspace(x0range[0],x0range[1],x0step)
		x1 = np.linspace(x1range[0],x1range[1],x1step)
		x2 = np.linspace(x2range[0],x2range[1],x2step)
		X = np.meshgrid(*[x0,x1,x2])
		X = np.vstack([X[0].ravel(),X[1].ravel(),X[2].ravel()]).T
		Xsph = np.dot(W.E.invL,(X-W.E.fid_params).T).T

		## Cross Validate
		recon = []
		recon_err = []
		weights = []
		weights_err = []
		lnlike = []
		lnlike_err = []
		lnprior = []
		lnpost = []
		for i in range(len(X)):
			W.E.N_samples = k
			W.samp_construct_model(X[i],**lnprob_kwargs)
			lnl = W.S.lnlike(W.Obs.y,W.S.model,W.S.data_invcov)
			lnp = W.S.lnprior(X[i])
			r = W.E.recon.ravel()
			rerr = W.E.recon_err.ravel()
			w = W.E.weights.ravel()
			werr = W.E.weights_err.ravel()
			lnlike.append(lnl)
			lnlike_err.append(W.S.lnlike_emu_err)
			lnprior.append(lnp)
			lnpost.append(lnl+lnp)
			recon.append(r)
			recon_err.append(rerr)
			weights.append(w)
			weights_err.append(werr)
			if i%500==0: print i

		recon = np.array(recon)
		recon_err = np.array(recon_err)
		weights = np.array(weights)
		weights_err = np.array(weights_err)
		lnlike = np.array(lnlike)
		lnlike_err = np.array(lnlike_err)
		lnprior = np.array(lnprior)
		lnpost = np.array(lnpost)

		lnlike = lnlike.reshape(xstep,xstep) 
		lnlike_err = lnlike_err.reshape(xstep,xstep)
		like_err = np.exp(lnlike)*lnlike_err
		like_err[np.isnan(like_err)] = 0.
		lnprior = lnprior.reshape(xstep,xstep)
		lnpost = lnpost.reshape(xstep,xstep)
		lnpost_err = lnlike_err
		post_err = np.exp(lnpost)*lnpost_err
		post_err[np.isnan(post_err)] = 0.

		# Plot Likelihood Surface
		fig = mp.figure(figsize=(21,5))
		fig.subplots_adjust(wspace=0.4,hspace=0.1)

		p1 = 1
		p2 = 2
		par1 = x1range
		par2 = x2range
		par1_range = np.linspace(par1[0],par1[1]+1e-4,xstep)
		par2_range = np.linspace(par2[0],par2[1]+1e-4,xstep)
		dp1 = par1_range[1] - par1_range[0]
		dp2 = par2_range[1] - par2_range[0]
		dp1_ticks = 10000.
		dp2_ticks = 10.

		levels=[-350,-150,-50,-20]
		vmax,vmin = None,None#-10,-2000

		# Plot Likelihood
		ax = fig.add_subplot(141)
		cax = ax.matshow(lnlike,origin='lower',cmap='nipy_spectral',
			extent=[par2[0],par2[1],par1[0],par1[1]],aspect=(par2[1]-par2[0])/(par1[1]-par1[0]),vmax=vmax,vmin=vmin)
		mp.xticks(np.arange(par2[0],par2[1]+dp2_ticks,dp2_ticks))
		mp.yticks(np.arange(par1[0],par1[1]+dp1_ticks,dp1_ticks))
		ax.xaxis.set_ticks_position('bottom')
		map(lambda x: x.set_rotation(35), ax.get_xticklabels())
		map(lambda x: x.set_rotation(35), ax.get_yticklabels())
		fig.colorbar(cax)
		ttl = ax.set_title('Log Likelihood')
		ax.set_xlabel(params[p2]);ax.set_ylabel(params[p1])
		ax.set_xlim(par2);ax.set_ylim(par1)
		ax.contour(lnlike,extent=[par2[0],par2[1],par1[0],par1[1]],levels=levels,cmap='magma_r')
		ax.plot(grid_tr.T[p2],grid_tr.T[p1],'b.')

		# Plot lnlike emulator error
		ax = fig.add_subplot(142)
		cax = ax.matshow(lnlike_err,origin='lower',cmap='nipy_spectral_r',
			extent=[par2[0],par2[1],par1[0],par1[1]],aspect=(par2[1]-par2[0])/(par1[1]-par1[0]),vmax=50,vmin=0)
		mp.xticks(np.arange(par2[0],par2[1]+dp2_ticks,dp2_ticks))
		mp.yticks(np.arange(par1[0],par1[1]+dp1_ticks,dp1_ticks))
		ax.xaxis.set_ticks_position('bottom')
		map(lambda x: x.set_rotation(35), ax.get_xticklabels())
		map(lambda x: x.set_rotation(35), ax.get_yticklabels())
		fig.colorbar(cax)
		ttl = ax.set_title('Log Likelihood Emulator Error')
		#ax.plot(grid_tr.T[p2],grid_tr.T[p1],'b.',alpha=0.2)
		H,xedges,yedges = np.histogram2d(grid_tr.T[p2],grid_tr.T[p1],bins=15)
		ax.contour(H,levels=[1,2,3],extent=[xedges.min(),xedges.max(),yedges.min(),yedges.max()],cmap='winter')
		ax.set_xlabel(params[p2]);ax.set_ylabel(params[p1])
		ax.set_xlim(par2);ax.set_ylim(par1)
		#ax.contour(lnlike_err,extent=[par2[0],par2[1],par1[0],par1[1]],
		#           levels=[0.05,0.1,0.2,0.3,0.4],cmap='magma_r')

		# Plot prior
		ax = fig.add_subplot(143)
		cax = ax.matshow(lnprior,origin='lower',cmap='nipy_spectral',
					extent=[par2[0],par2[1],par1[0],par1[1]],aspect=(par2[1]-par2[0])/(par1[1]-par1[0]))
		mp.xticks(np.arange(par2[0],par2[1]+dp2_ticks,dp2_ticks))
		mp.yticks(np.arange(par1[0],par1[1]+dp1_ticks,dp1_ticks))
		ax.xaxis.set_ticks_position('bottom')
		map(lambda x: x.set_rotation(35), ax.get_xticklabels())
		map(lambda x: x.set_rotation(35), ax.get_yticklabels())
		fig.colorbar(cax)
		ttl = ax.set_title('Log Prior')
		ax.set_xlabel(params[p2]);ax.set_ylabel(params[p1])
		ax.set_xlim(par2);ax.set_ylim(par1)

		# Plot Posterior
		ax = fig.add_subplot(144)
		cax = ax.matshow(lnpost,origin='lower',cmap='nipy_spectral',
			extent=[par2[0],par2[1],par1[0],par1[1]],aspect=(par2[1]-par2[0])/(par1[1]-par1[0]),vmax=vmax,vmin=vmin)
		mp.xticks(np.arange(par2[0],par2[1]+dp2_ticks,dp2_ticks))
		mp.yticks(np.arange(par1[0],par1[1]+dp1_ticks,dp1_ticks))
		ax.xaxis.set_ticks_position('bottom')
		map(lambda x: x.set_rotation(35), ax.get_xticklabels())
		map(lambda x: x.set_rotation(35), ax.get_yticklabels())
		fig.colorbar(cax)
		ttl = ax.set_title('Log Posterior')
		ax.set_xlabel(params[p2]);ax.set_ylabel(params[p1])
		ax.set_xlim(par2);ax.set_ylim(par1)
		ax.axvline(p_true[p2],color='b')
		ax.axhline(p_true[p1],color='b')
		ax.plot([p_true[p2]],[p_true[p1]],marker='s',markersize=8,color='b')
		ax.contour(lnpost,extent=[par2[0],par2[1],par1[0],par1[1]],
				levels=levels,cmap='magma_r')



	################
	### run MCMC ###
	################

	# Initialize Walkers
	#pos = np.array(map(lambda x: stats.uniform.rvs(x[0],x[1]-x[0],nwalkers),param_bounds)).T
	pos = np.array(map(lambda x: x*stats.norm.rvs(1,0.05,nwalkers),p_true)).T
	#pos = grid_cv[np.random.randint(0,len(grid_cv),nwalkers)]

	time_it = False
	if time_it == True:
		%timeit W.samp_lnprob(pos[0],**lnprob_kwargs)

		%timeit W.samp_drive(pos,step_num=1,burn_num=0)

	run_mcmc = False
	if run_mcmc == True:
		# Drive Sampler
		burn_num = 10
		step_num = 100
		W.samp_drive(pos,step_num=step_num,burn_num=burn_num)
		samples = W.S.sampler.chain[:, 0:, :].reshape((-1, W.S.ndim))
		print("Mean acceptance fraction: {0:.3f}".format(np.mean(W.S.sampler.acceptance_fraction)))

		plot_acor = False
		if plot_acor == True:
			print 'acor =',W.S.sampler.acor
			fig = mp.figure(figsize=(15,10))
			fig.subplots_adjust(wspace=0.3)

			for i in range(N_params):
				ax = fig.add_subplot(4,3,i+1)
				ax.set_ylabel(p_latex[i],fontsize=16)
				for j in range(nwalkers)[::nwalkers/20]:
					ax.plot(W.S.sampler.chain[j,:,i],alpha=0.5)

		plot_triangle = False
		if plot_triangle == True:
			# Triangle Plot
			levels = [0.34,0.68,0.90,0.95]

			fig = corner.corner(samples, labels=p_latex,
								truths=p_true, range=param_bounds, levels=levels)

			#p1_lim = (None,None); p2_lim = (None,None); p3_lim = (None,None)
			#p1_lim = (0.75,1.0); p2_lim = (20000,65000); p3_lim = (15,45)
			#p1_lim = (0.6,1.1); p2_lim = (0,80000); p3_lim = (0,95)

			#fig.axes[0].set_xlim(p1_lim);fig.axes[3].set_xlim(p1_lim);fig.axes[6].set_xlim(p1_lim)
			#fig.axes[4].set_xlim(p2_lim);fig.axes[7].set_xlim(p2_lim)
			#fig.axes[8].set_xlim(p3_lim)
			#fig.axes[3].set_ylim(p2_lim);fig.axes[6].set_ylim(p3_lim);fig.axes[7].set_ylim(p3_lim)

			add_ts = False
			if add_ts == True:
				fig.axes[3].plot(grid_tr.T[0],grid_tr.T[1],'r.',alpha=0.25)
				fig.axes[3].plot(pos.T[0],pos.T[1],'c.',alpha=0.8)
				fig.axes[6].plot(grid_tr.T[0],grid_tr.T[2],'r.',alpha=0.25)
				fig.axes[6].plot(pos.T[0],pos.T[2],'c.',alpha=0.8)
				fig.axes[7].plot(grid_tr.T[1],grid_tr.T[2],'r.',alpha=0.25)
				fig.axes[7].plot(pos.T[1],pos.T[2],'c.',alpha=0.8)











