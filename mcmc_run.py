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
import scipy.linalg as la
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
from sklearn import gaussian_process as gp
import astropy.stats as astats
import scipy.stats as stats
import corner
import warnings
from pycape import common_priors
import pycape
import time
from mpi4py import MPI
import emcee
import re
from round_to_n import round_to_n
import pathos
warnings.filterwarnings('ignore',category=DeprecationWarning)

try:
	from IPython import get_ipython
	ipython = get_ipython()
except: pass

mp.rcParams['font.family'] = 'sans-serif'
mp.rcParams['font.sans-serif'] = ['Helvetica']
mp.rcParams['text.usetex'] = True

## Flags


## Separate out multiple processes
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank > 0:
	sys.stdout = open('process'+str(rank)+'_stdout.out','w')
	sys.stderr = open('process'+str(rank)+'_stderr.out','w')

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

	def print_mem():
		print "...Memory Usage is "+str(memory_usage()[0]/1000.)+" GB"

	start_time = time.time()

	############################################################
	### Load and Separate Training and Cross Validation Sets ###
	############################################################
	print_message('...loading and separating data')
	print_time()
	print_mem()

	# Separate and Draw Data
	def draw_data():
		make_globals = ['tr_len','data_tr','grid_tr','data_cv','grid_cv','fid_params','fid_data']

		# Load Datasets
		file = open('lhs_data.pkl','rb')
		lhs_data = pkl.Unpickler(file).load()
		file.close()

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

		# Choose which data to keep
		keep_meta = ['ps','nf','Tb']

		# Choose Training Set
		TS_data = gauss_hera127_data
		#TS_data = lhs_data

		# Separate Data
		tr_len = 5000
		#tr_len = 16344
		rando = np.random.choice(np.arange(tr_len),size=4999,replace=False)
		rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))

		data_tr = TS_data['data'][np.argsort(TS_data['indices'])][rando]
		grid_tr = TS_data[ 'gridf'][np.argsort(TS_data['indices'])][rando]
		direcs_tr = TS_data['direcs'][np.argsort(TS_data['indices'])][rando]
		meta_tr = TS_data['metadata']
		keep = np.array(map(lambda x: x in keep_meta, meta_tr))
		data_tr = data_tr.T[keep].T

		# Add other datasets
		add_other_data = False
		if add_other_data == True:
			# Choose Training Set
			TS_data = gauss_hera331_data

			# Separate Data
			tr_len = 5000
			rando = np.random.choice(np.arange(tr_len),size=1500,replace=False)
			rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))

			data_tr2 = TS_data['data'][np.argsort(TS_data['indices'])][rando]
			grid_tr2 = TS_data[ 'gridf'][np.argsort(TS_data['indices'])][rando]
			direcs_tr2 = TS_data['direcs'][np.argsort(TS_data['indices'])][rando]
			meta_tr2 = TS_data['metadata']
			keep = np.array(map(lambda x: x in keep_meta, meta_tr2))
			data_tr2 = data_tr2.T[keep].T

			data_tr = np.concatenate([data_tr,data_tr2])
			grid_tr = np.concatenate([grid_tr,grid_tr2])
			direcs_tr = np.concatenate([direcs_tr,direcs_tr2])

		# Choose Cross Validation Set
		CV_data 		= gauss_hera127_data
		no_rando		= False
		TS_remainder	= True
		use_remainder	= True

		# Separate Data
		if TS_remainder == True:
			if use_remainder == True:
				rando = ~rando
			else:
				remainder = np.where(rando==False)[0]
				rando = np.array([False for i in range(tr_len)])
				rando[np.random.choice(remainder,size=1300,replace=False)] = True
			
		else:
			tr_len = 550
			rando = np.random.choice(np.arange(tr_len),size=550,replace=False)
			rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))

		data_cv = CV_data['data'][np.argsort(CV_data['indices'])][rando]
		grid_cv = CV_data[ 'gridf'][np.argsort(CV_data['indices'])][rando]
		direcs_cv = np.array(CV_data['direcs'])[np.argsort(CV_data['indices'])][rando]
		meta_cv = CV_data['metadata']
		keep = np.array(map(lambda x: x in keep_meta, meta_cv))
		data_cv = data_cv.T[keep].T

		# Get Fiducial Data
		feed_fid = True
		if feed_fid == True:
			fid_params = fiducial_data['fid_params']
			fid_data = fiducial_data['fid_data']
			fid_meta = fiducial_data['metadata']
			keep = np.array(map(lambda x: x in keep_meta, fid_meta))
			fid_data = fid_data[keep]
		else:
			fid_params = np.array(map(astats.biweight_location,grid_tr.T))
			fid_data = np.array([astats.biweight_location(data_tr.T[i]) if np.isnan(astats.biweight_location(data_tr.T[i])) == False \
									else np.median(data_tr.T[i]) for i in range(len(data_tr.T))])
			
		globals().update(dez.create(make_globals,locals()))

	np.random.seed(1)
	print "...Drawing data"
	draw_data()
	print_time()
	print_mem()

	# Get minimum distance between any two points, assuming they have already been decorrelated and normalized
	def get_min_R(samples,Ntest=None):
		if Ntest is None:
			Ntest = len(samples)
		minR = 100.0
		maxR = 0.01
		rando = np.random.choice(np.arange(len(samples)),replace=False,size=Ntest)
		for i in np.arange(Ntest-1):
			for j in np.arange(i+1,Ntest):
				if j == i: continue
				this_R = la.norm(samples[i]-samples[j])
				if this_R < minR: minR = 1*this_R
				if this_R > maxR: maxR = 1*this_R
		return minR, maxR

	def sphere_minR(samples,cov_samp,fid_grid,Ntest=None):
		X = cov_samp - fid_grid
		cov = np.cov(X.T)
		L = la.cholesky(cov)
		invL = la.inv(L)
		sph_X = np.dot(invL,samples.T).T
		minR,maxR = get_min_R(sph_X,Ntest=Ntest)
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
		Ntest = 500
		seed = []
		minR = []
		rando_seeds = np.random.choice(np.arange(100000),size=50,replace=False)
		for i in range(len(rando_seeds)):
			np.random.seed(rando_seeds[i])
			draw_data()
			minr,maxr = sphere_minR(grid_tr,gauss_hera331_data['gridf']-fid_params,fid_params,Ntest=Ntest)
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
	plot_tr = False
	if plot_tr == True:
		print_message('...plotting ts')
		print_time()

		lims = [[None,None] for i in range(11)]
		lims = [[grid_tr.T[i].min(), grid_tr.T[i].max()] for i in range(11)]

		fig = mp.figure(figsize=(15,8))
		fig.subplots_adjust(wspace=0.3)
		j = 0
		for i in range(6):
			ax = fig.add_subplot(2,3,i+1)
			ax.plot(grid_tr.T[j],grid_tr.T[j+1],'k,',alpha=0.75)
			ax.plot(fid_params[j], fid_params[j+1], color='m', marker='*', markersize=15)
			#ax.plot(grid_cv.T[j],grid_cv.T[j+1],'r.')
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

	### Variables for Emulator ###
	N_modes = 15
	N_params = len(params)
	N_data = 836
	N_samples = len(data_tr)
	poly_deg = 2
	reg_meth='gaussian'
	ell = np.array([10 for i in range(N_params)])
	recon_err_norm = 1.0
	kernel = gp.kernels.RBF(ell)
	scale_by_std = False
	scale_by_obs_errs = False

	gp_kwargs = {'kernel':kernel}

	variables.update({'params':params,'N_params':N_params,'N_modes':N_modes,'N_samples':N_samples,'N_data':N_data,
						'reg_meth':reg_meth,'poly_deg':poly_deg,'gp_kwargs':gp_kwargs,'scale_by_std':scale_by_std,
						'scale_by_obs_errs':scale_by_obs_errs,'recon_err_norm':recon_err_norm})

	workspace_init = {'dir_pycape':'/global/homes/n/nkern/Software/pycape',
					  'dir_21cmSense':'/global/homes/n/nkern/Software/21cmSense'}


	###########################
	### INITIALIZE EMULATOR ###
	###########################

	# Initialize workspace, emulator
	print_message('...initializing emulator')
	E = pycape.Emu(variables)
	print_mem()

	# Initialize Cholesky
	E.sphere(grid_tr,fid_params=fid_params,save_chol=True)
	fid_params = E.fid_params

	print_message('...initializing mockobs')
	print_time()
	print_mem()

	###########################################
	### Load Mock Observation and Configure ###
	###########################################

	file = open('mockObs_hera331_allz.pkl','rb')
	mock_data = pkl.Unpickler(file).load()
	file.close()
	p_true = fid_params

	# Variables of model data
	rest_freq		= 1.4204057517667	# GHz
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

	# Mock Obs variables
	freq_cent       = mock_data['freq']           # In GHz
	z_pick          = np.around(rest_freq/(freq_cent) - 1,2)
	z_num           = len(z_pick)
	z_lim = np.array(map(lambda x: x in z_pick, z_array))
	y_pick = k_range
	y_lim = np.array(map(lambda x: x in np.array(y_pick,str), y_array))
	model_lim = np.array([False]*N_data)
	for i in range(z_len):
		if z_lim[i] == True:
			model_lim[i*y_len:(i+1)*y_len] = np.copy(y_lim)

	model_lim = np.array([True]*N_data)

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
				err_thresh = 1.0        # 100%
				small_errs = np.where(mock_data['sense_PSerrs'][i] / mock_data['sense_PSdata'][i] < err_thresh)[0]
				mock_data['sense_kbins'][i] = mock_data['sense_kbins'][i][small_errs]
				mock_data['sense_PSdata'][i] = mock_data['sense_PSdata'][i][small_errs]
				mock_data['sense_PSerrs'][i] = mock_data['sense_PSerrs'][i][small_errs]

	mock_data['sense_kbins'] = np.array( map(lambda x: np.array(x,float),mock_data['sense_kbins']))

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

	model_x		= mock_data['kbins']
	obs_x		= mock_data['sense_kbins']
	obs_y		= mock_data['sense_PSdata']
	obs_y_errs	= mock_data['sense_PSerrs']
	obs_track	= np.array(map(lambda x: ['ps' for i in range(len(x))], obs_x))
	track_types = ['ps']

	# Add other information to mock dataset
	add_xh = True
	if add_xh == True:
		model_x		= np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(model_x,z_array)))
		obs_x		= np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_x,z_array)))
		obs_y		= np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y,np.zeros(z_num))))
		obs_y_errs	= np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y_errs,np.ones(z_num)*1e6)))
		obs_track	= np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_track,['xh' for i in range(z_num)])))
		track_types += ['xh']

	add_Tb = True
	if add_Tb == True:
		model_x     = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(model_x,z_array)))
		obs_x       = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_x,z_array)))
		obs_y       = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y,np.zeros(z_num))))
		obs_y_errs  = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y_errs,np.ones(z_num)*1e6)))
		obs_track   = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_track,['Tb' for i in range(z_num)])))
		track_types += ['Tb']

	obs_y = np.concatenate(obs_y)
	obs_y_errs = np.concatenate(obs_y_errs)
	obs_x_nums = np.array([len(x) for x in obs_x])
	obs_track = np.concatenate(obs_track.tolist())
	track_types = np.array(track_types)

	O = pycape.Obs(model_x,obs_x,obs_y,obs_y_errs,obs_track,track_types)

	# Feed Mock Observation to Workspace
	update_obs_dic = {'N_data':obs_y.size,'z_num':z_num,'z_array':z_array,
						 'k_range':k_range,'obs_x_nums':obs_x_nums}
	O.update(update_obs_dic)
	print_time()
	print_mem()

	## Interpolate Training Set onto Observational Basis ##
	print_message('...configuring data for emulation')
	print_time()

	# First interpolate onto observational redshift basis #
	redshift_interp = False
	if redshift_interp == True:
		# select out ps
		def z_interp(data):
			ps_select = np.array([[True if i < k_len else False for i in range(y_len)] for j in range(z_len)]).ravel()
			gp_select = np.array([[False if i < k_len else True for i in range(y_len)] for j in range(z_len)]).ravel()
			ps_data = np.array(map(lambda x: x[ps_select].reshape(z_len,k_len), data))
			gp_data = np.array(map(lambda x: x[gp_select].reshape(z_len,g_len), data))
			ps_pred = np.array([10**curve_interp(z_pick,z_array,np.log10(ps_data[:,:,i].T),n=3,degree=2).T for i in range(k_len)]).T
			gp_pred = np.array([curve_interp(z_pick,z_array,ps_data[:,:,i].T,n=3,degree=2).T for i in range(g_len)]).T
			data = [[] for i in range(len(ps_data))]
			for i in range(ps_pred.shape[0]):
				for j in range(len(ps_data)):
					data[j].extend(np.concatenate([ps_pred[i][j],gp_pred[i][j]]))
			return np.array(data)

		data_tr = z_interp(data_tr)
		data_cv = z_interp(data_cv)
		fid_data = z_interp(fid_data[np.newaxis,:]).ravel()

	# Second Interpolate P Spec onto observational k-mode basis #
	ps_interp = True
	if ps_interp == True:
		def ps_interp(data):
			# select out ps and other data
			ps_select = np.array([[True if i < k_len else False for i in range(y_len)] for j in range(z_len)]).ravel()
			ps_data = np.array(map(lambda x: x[ps_select].reshape(z_len,k_len), data))
			other_data = data.T[~ps_select].T.reshape(len(data),z_len,g_len)
			ps_track = O.track(['ps'])
			# interpolate (or make prediction)
			ps_pred = np.array([curve_interp(ps_track[i],O.k_range,ps_data[:,i,:].T,n=2,degree=1).T for i in range(z_len)])
			# reshape array
			data = [[] for i in range(len(ps_data))]
			for i in range(z_len):
				for j in range(len(ps_data)):
					try: data[j].extend(np.concatenate([ps_pred[i][j],other_data[j][i]]))
					except: data[j].extend(np.concatenate([np.array([]),other_data[j][i]]))
			return np.array(data)

		data_tr		= ps_interp(data_tr)
		data_cv		= ps_interp(data_cv)
		fid_data	= ps_interp(fid_data[np.newaxis,:]).ravel()

	# Transform data to likelihood?
	trans_lnlike = False
	if trans_lnlike == True:
		data_tr = np.array(map(lambda x: S.gauss_lnlike(x,O.y,O.invcov),data_tr))[:,np.newaxis]
		data_cv = np.array(map(lambda x: S.gauss_lnlike(x,O.y,O.invcov),data_cv))[:,np.newaxis]
		fid_data = np.array(map(np.median,data_tr.T))

	# Initialize KLT
	E.klt(data_tr,fid_data=fid_data)

	raise NameError
	# Make new yz_data matrix
	yz_data = []
	for i in range(z_num):
		kdat = np.array(np.around(O.xdata[i],3),str)
		zdat = np.array([z_array[i] for j in range(len(O.xdata[i]))],str)
		yz_data.append(np.array(zip(kdat,zdat)))
	yz_data = np.array(yz_data)

	print_time()

	###### Set Parameter values
	print_message('...configuring emulator and sampler')
	print_time()
	print_mem()
	k = 500
	use_pca = True
	emode_variance_div = 1.0
	calc_noise = False
	norm_noise = False
	fast = True
	compute_klt = False
	save_chol = False
	LAYG = False

	# First Guess of GP Hyperparameters
	ell = np.array([[10.0 for i in range(N_params)] for i in range(N_modes)]) / np.linspace(1.0,2.0,N_modes).reshape(N_modes,1)
	ell_bounds = np.array([[1e-2,1e2] for i in range(N_modes)])

	noise_var = 1e-8 * np.linspace(1,100,N_modes)
	noise_bounds = np.array([[1e-8,1e-3] for i in range(N_modes)])

	# Insert HP into GP
	kernels = map(lambda x: gp.kernels.RBF(*x[:2]) + gp.kernels.WhiteKernel(*x[2:]), zip(ell,ell_bounds,noise_var,noise_bounds))
#	kernels = map(lambda x: gp.kernels.RBF(x[0],x[1]), zip(ell,ell_bounds))

	if use_pca == False:
		if trans_lnlike == False:
			data_tr = data_tr.T[O.model_lim].T
			E.data_tr = data_tr
			E.fid_data = fid_data[O.model_lim]
			N_modes = len(O.model_lim==True)
		else:
			N_modes = 1
			N_data = 1
		N_data = N_modes
		E.N_modes = N_modes
		E.N_data = N_data
		emode_variance_div = 1.0

	# Separate regressors into modegroups
	E.group_eigenmodes(emode_variance_div=emode_variance_div)

	names       = ['kernel','copy_X_train','optimizer','n_restarts_optimizer']
	optimize    = 'fmin_l_bfgs_b'
	n_restarts  = 5
	gp_kwargs_arr = np.array([dict(zip(names,[kernels[i],False,optimize,n_restarts])) for i in map(lambda x: x[0],E.modegroups)])

	# Insert precomputed HP
	load_hype = True
	if load_hype == True:
		file = open('forecast_hyperparams2.pkl','rb')
		input = pkl.Unpickler(file)
		hp_dict = input.load()
		file.close()

		# check variables
		emode_variance_div = hp_dict['emode_variance_div']
		E.group_eigenmodes(emode_variance_div=emode_variance_div)
		optimize = None
		n_restarts = 0

		# insert kernels into gp_kwargs_arr
		try:
			gp_kwargs_arr = np.array([dict(zip(names,[hp_dict['fit_kernels'][i],False,optimize,n_restarts])) for i in range(E.N_modegroups)])
		except:
			E.modegroups = hp_dict['modegroups']
			E.N_modegroups = hp_dict['N_modegroups']
			gp_kwargs_arr = np.array([dict(zip(names,[hp_dict['fit_kernels'][i],False,optimize,n_restarts])) for i in range(E.N_modegroups)])

	# Create training kwargs
	kwargs_tr = {'use_pca':use_pca,'scale_by_std':scale_by_std,'calc_noise':calc_noise,'norm_noise':norm_noise,
				 'noise_var':noise_var,'verbose':False,'invL':E.invL,'emode_variance_div':emode_variance_div,
				 'fast':fast,'compute_klt':compute_klt,'save_chol':save_chol,'scale_by_obs_errs':scale_by_obs_errs,
				 'gp_kwargs_arr':gp_kwargs_arr}

	### Initialize Sampler Variables ###
	predict_kwargs = {'fast':fast,'use_Nmodes':None,'use_pca':use_pca}

	param_width = np.array([grid_tr.T[i].max() - grid_tr.T[i].min() for i in range(N_params)])

	eps = 0

	param_bounds = np.array([[grid_tr.T[i].min()+param_width[i]*eps,grid_tr.T[i].max()-param_width[i]*eps]\
								 for i in range(N_params)])

	param_hypervol = reduce(operator.mul,map(lambda x: x[1] - x[0], param_bounds))

	use_Nmodes = None
	add_model_cov = False
	add_overall_modeling_error = True
	modeling_error = 0.20
	cut_high_fracerr = 10.0
	ndim = N_params
	nwalkers = 100

	sampler_init_kwargs = {'use_Nmodes':use_Nmodes,'param_bounds':param_bounds,'param_hypervol':param_hypervol,
							'nwalkers':nwalkers,'ndim':ndim,'N_params':ndim,'z_num':z_num}

	lnprob_kwargs = {'add_model_cov':add_model_cov,'kwargs_tr':kwargs_tr,
					 'predict_kwargs':predict_kwargs,'LAYG':LAYG,'k':k,
					 'add_overall_modeling_error':add_overall_modeling_error,'modeling_error':modeling_error}

	print_message('...initializing sampler')
	S = pycape.Samp(N_params, param_bounds, Emu=E, Obs=O)
	print_time()
	print_mem()

	train_emu = True
	if train_emu == True:
		save_hype 		= False
		kfold_regress	= False
		if kfold_regress == True:
			kfold_Nsamp = 2000
			kfold_Nclus = 15
			E.sphere(E.grid_tr,save_chol=False,invL=E.invL)
			Rsph = np.array(map(la.norm, E.Xsph))
			kfold_cents = E.Xsph[np.random.choice(np.arange(0,N_samples),replace=False,size=kfold_Nclus)]
			# Iterate over kfold clusters
			def multiproc_train(kfold_cent,Nsamp=kfold_Nsamp):
				E.sphere(E.grid_tr,save_chol=False,invL=E.invL)
				distances = np.array(map(la.norm,E.Xsph-kfold_cent))
				nearest = np.argsort(distances)[:Nsamp]
				kfold_data_tr = E.data_tr[nearest]
				kfold_grid_tr = E.grid_tr[nearest]
				S.E.train(kfold_data_tr,kfold_grid_tr,fid_data=E.fid_data,fid_params=E.fid_params,**kwargs_tr)			
				return np.array(map(lambda x: [np.exp(x.kernel_.theta), x.log_marginal_likelihood_value_], E.GP))

			print_message('...running kfold training')
			print_time()
			pool = pathos.multiprocessing.Pool(kfold_Nclus)
			result = np.array(pool.map(multiproc_train, kfold_cents))
			pool.close()
			print_time()

			# Average over results
			# Put parameters into ln-space
			theta		= result[:,:,0]
			log_marg	= result[:,:,1]
			minima		= np.array(map(lambda x: np.where(x==x.min())[0][0], log_marg.T))
			optima		= np.array([theta.T[i][minima[i]] for i in range(E.N_modegroups)])

#			kernels			= map(lambda x: gp.kernels.RBF(x[:11]) + gp.kernels.WhiteKernel(x[-1]), optima)
			kernels			= map(lambda x: gp.kernels.RBF(x), optima)
			names       	= ['kernel','copy_X_train','optimizer','n_restarts_optimizer']
			optimize    	= None
			n_restarts		= 0
			gp_kwargs_arr	= np.array([dict(zip(names,[kernels[i],False,optimize,n_restarts])) for i in range(E.N_modegroups)])
			kwargs_tr['gp_kwargs_arr'] = gp_kwargs_arr
			save_hype		= True

		print_message('...training emulator')
		print_time()
		S.E.train(E.data_tr,E.grid_tr,fid_data=E.fid_data,fid_params=E.fid_params,**kwargs_tr)
		print_time()

		# Print out fitted hyperparameters
		if E.GP[0].optimizer is not None or save_hype == True:
			fit_kernels = []
			for i in range(len(E.GP)): fit_kernels.append(E.GP[i].kernel_)
			hyp_dict = {'N_modes':E.N_modes,'N_modegroups':E.N_modegroups,'modegroups':E.modegroups,\
							'emode_variance_div':emode_variance_div,'N_samples':E.N_samples,'fit_kernels':fit_kernels,\
							'data_tr':E.data_tr,'grid_tr':E.grid_tr,'data_cv':data_cv,'grid_cv':grid_cv,'fid_params':fid_params,\
							'fid_data':fid_data,'err_thresh':err_thresh,'mock_data':mock_data,'invL':E.invL}

			i = 0
			while True:
				param_filename = 'forecast_hyperparams'+str(i)+'.pkl'
				i += 1
				if os.path.isfile(param_filename) == False: break
			f = open(param_filename,'wb')
			output = pkl.Pickler(f)
			output.dump(hyp_dict)
			f.close()

	print_mem()
	#################
	### FUNCTIONS ###
	#################

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
			S.lnprob(grid_cv[i],**lnprob_kwargs)
			r = E.recon.ravel()
			rerr = E.recon_err.ravel()
			w = E.weights.ravel()
			werr = E.weights_err.ravel()
			tw = np.dot(data_cv[i]-E.fid_data,E.eig_vecs.T)
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

	def calc_errs(recon,recon_err):
		# Get cross validated reconstruction error
		err = (recon-data_cv)/data_cv
		pred_err = recon_err / data_cv
		mean_err = np.array(map(lambda x: np.median(x),err.T))
		central_err = err[np.where(np.abs(err)<np.std(err))]
		std = np.std(central_err)
		rms_err = np.sqrt(astats.biweight_location(err.ravel()**2))
		rms_std_err = astats.biweight_midvariance(err.ravel())
		# Get CV error at each redshift
		recon_mat = np.array(map(lambda x: O.mat2row(x,mat2row=False), recon))
		cv_mat    = np.array(map(lambda x: O.mat2row(x,mat2row=False), data_cv))
		frac_err = (recon_mat-cv_mat)/cv_mat
		frac_err_mat = []
		for i in range(z_num):
			avg_err_kbins = np.array(map(lambda x: np.sqrt(np.median(x**2)),np.array([frac_err.T[i][j] for j in range(len(data_cv))]).T))
			frac_err_mat.append(avg_err_kbins)
		frac_err_mat = np.array(frac_err_mat)

		zarr_vec = O.mat2row(np.array([[z_array[i]]*len(O.xdata[i]) for i in range(z_num)]),mat2row=True)
		frac_err_vec = O.mat2row(frac_err_mat,mat2row=True)

		names = ['err','pred_err','frac_err','frac_err_vec','zarr_vec','rms_err','rms_std_err']
		globals().update(dez.create(names,locals()))

	def plot_cross_valid(fname='cross_validate_ps.png'):
		fig = mp.figure(figsize=(16,12))

		# Scree Plot
		ax = fig.add_subplot(331)
		ax.plot(E.eig_vals,'ko')
		ax.set_yscale('log')
		ax.set_xlim(-1,N_modes+1)
		ax.set_xlabel(r'eigenmode #',fontsize=15)
		ax.set_ylabel(r'eigenvalue',fontsize=15)
		ax.grid(True)

		# Cross Validation Error
		#z_arr = np.array(yz_data.reshape(91,2).T[1],float)
		#good_z = (z_arr > 7) & (z_arr < 25)
		#E.model_lim[~good_z] = False

		ax = fig.add_subplot(332)
		try:
			p1 = ax.hist(np.abs(err.ravel()),histtype='step',color='b',linewidth=1,bins=75,range=(-0.01,1.5),normed=True,alpha=0.75)
			p2 = ax.hist(pred_err.ravel(),histtype='step',color='r',linewidth=1,bins=75,range=(-.01,1.5),normed=True,alpha=0.5)
		except UnboundLocalError:
			print 'UnboundLocalError on err or pred_err'
		#ax.axvline(rms,color='r',alpha=0.5)
		#ax.axvline(-rms,color='r',alpha=0.5)
		#ax.hist(rms_obs_err,histtype='step',color='m',bins=50,range=(0,0.2),alpha=0.5,normed=True)
		ax.set_xlim(-0.001,1.5)
		ax.set_ylim(0,2)
		ax.set_xlabel(r'Fractional Error',fontsize=15)
		ax.annotate(r'rms err = '+str(np.around(rms_err*100,2))+'%\n rms standard error ='+str(np.around(rms_std_err*100,2))+'%',
			xy=(0.2,0.8),xycoords='axes fraction')

		ax = fig.add_subplot(333)
		im = ax.scatter(O.x_ext,zarr_vec,c=frac_err_vec*100,marker='o',s=35,edgecolor='',alpha=0.75,cmap='nipy_spectral_r',vmax=50)
		ax.set_xscale('log')
		ax.set_xlim(1e-1,3)
		ax.set_ylim(4,25)
		ax.set_xlabel(r'$k\ cMpc^{-1}',fontsize=17)
		ax.set_ylabel(r'$z$',fontsize=17)
		fig.colorbar(im,label=r'Avg. Percent Error')

		for i in range(6):
			ax = fig.add_subplot(3,3,i+4)
			# Get Eigenvector and kz data
			eig_vec = E.eig_vecs[i]
			# yz_eigvector plot
			cmap = mp.cm.get_cmap('coolwarm',41)
			vavg = (eig_vec.max()+np.abs(eig_vec.min()))/2.0
			im = ax.scatter(O.x_ext,zarr_vec,c=eig_vec,marker='o',s=35,edgecolors='',alpha=0.9,cmap=cmap,vmin=-vavg,vmax=vavg)
			ax.set_xlabel(r'$k$ (Mpc$^{-1}$)',fontsize=12)
			ax.set_xscale('log')
			ax.set_xlim(1e-1,3)
			ax.set_ylabel(r'$z$',fontsize=12)
			fig.colorbar(im)

		fig.savefig(fname,dpi=200,bbox_inches='tight')
		mp.close()
		print_time()

	#####################
	### END FUNCTIONS ###
	#####################

	# Plotting and Cross Validating
	plot_eigenmodes		= False
	cross_validate_ps	= True
	cross_validate_like	= True
	plot_weight_pred	= False
	plot_ps_pred		= False
	plot_ps_recon_frac	= False
	plot_ps_recon		= False

	# Sampling
	make_fisher			= False
	add_priors			= True
	time_sampler		= False
	drive_sampler		= True
	save_chains			= True

	# Plotting
	trace_plots			= True
	autocorr_plots		= True
	tri_plots			= True
	plot_boxplots		= True
	ps_var_movie		= False


	if plot_eigenmodes == True:
		# Plot first few eigenmodes and scree plot
		gs = gridspec.GridSpec(3,2)
		fig = mp.figure(figsize=(9,12))
		fig.subplots_adjust(wspace=0.3,hspace=0.3)
		ax1 = fig.add_subplot(gs[0,:])
		ax2 = fig.add_subplot(gs[1,0])
		ax3 = fig.add_subplot(gs[1,1])
		ax4 = fig.add_subplot(gs[2,0])
		ax5 = fig.add_subplot(gs[2,1])

		# Make Scree Plot
		ax1.grid(True,which='major')
		ax1.set_yscale('log')
		ax1.scatter(np.arange(2,len(E.eig_vals))+1,E.eig_vals[2:],facecolor='k',s=40,marker='o',edgecolor='',alpha=0.75)
		ax1.scatter([1],E.eig_vals[0],facecolor='b',marker='o',s=50,edgecolor='',alpha=0.75)
		ax1.scatter([2],E.eig_vals[1],facecolor='r',marker='o',s=50,edgecolor='',alpha=0.75)
		#ax1.scatter([3],E.eig_vals[2],facecolor='r',marker='o',s=40,edgecolor='',alpha=0.75)
		ax1.set_xlim(0,len(E.eig_vals))
		ax1.set_xlabel(r'Eigenmode \#',fontsize=16)
		ax1.set_ylabel(r'Eigenvalue',fontsize=16)
		ax1.annotate(r'HERA Forecast Training Set',fontsize=14,xy=(0.61,0.73),xycoords='axes fraction')

		# Plot power spectra at 4 redshifts
		z1,z2,z3 = 5, 13, 19
		ax2.grid(True)
		ax2.set_xscale('log')
		ax2.set_yscale('log')
		kbins = O.track(['ps'],arr=O.xdata)
		fdat = O.track(['ps'],arr=O.mat2row(E.fid_data,mat2row=False))
		p0, = ax2.plot(kbins[z1],fdat[z1],linestyle='-',linewidth=2,color='k',alpha=0.75)
		p1, = ax2.plot(kbins[z2],fdat[z2],linestyle='--',linewidth=2,color='k',alpha=0.75)
		p2, = ax2.plot(kbins[z3],fdat[z3],linestyle=':',linewidth=2,color='k',alpha=0.75)
		ax2.legend([p0,p1,p2],[r'$z=8.0$',r'$z=12.0$',r'$z=15.0$'],loc=2)
		ax2.set_xlim(1e-1,4)
		ax2.set_ylim(8,2e4)
		ax2.set_xlabel(r'$k$ ($h$ Mpc$^{-1}$)',fontsize=15)
		ax2.set_ylabel(r'$\Delta^{2}$',fontsize=15)

		# Plot Eigenmodes
		ax4.grid(True)
		ax4.set_xscale('log')
		evecs = np.array(map(lambda x: O.track(['ps'],arr=O.mat2row(x,mat2row=False)),E.eig_vecs))
		p3, = ax4.plot(kbins[z1],evecs[0][z1]/np.abs(evecs[0][z1]).max(),linestyle='-',color='b',linewidth=1.5,alpha=0.5)
		p4, = ax4.plot(kbins[z1],evecs[1][z1]/np.abs(evecs[1][z1]).max(),linestyle='-',color='r',linewidth=1.5,alpha=0.5)
		p5, = ax4.plot(kbins[z2],evecs[0][z2]/np.abs(evecs[0][z2]).max(),linestyle='--',color='b',linewidth=1.5,alpha=0.5)
		p6, = ax4.plot(kbins[z2],evecs[1][z2]/np.abs(evecs[1][z2]).max(),linestyle='--',color='r',linewidth=1.5,alpha=0.5)
		p8, = ax4.plot(kbins[z3],evecs[0][z3]/np.abs(evecs[0][z3]).max(),linestyle=':',color='b',linewidth=1.5,alpha=0.5)
		p9, = ax4.plot(kbins[z3],evecs[1][z3]/np.abs(evecs[1][z3]).max(),linestyle=':',color='r',linewidth=1.5,alpha=0.5)
		ax4.axhline(0,color='k')
		ax4.set_xlabel(r'$k$ ($h$ Mpc$^{-1}$)',fontsize=15)
		ax4.set_ylabel(r'$\phi_{\Delta^{2}}$',fontsize=15)
		ax4.set_xlim(1e-1,4)
		ax4.set_ylim(-1.1,1.1)

		# Plot Neutral Fraction and Brightness Temperature
		ax3b = ax3.twinx()
		gdat = O.track(['xH','Tb'],arr=O.mat2row(E.fid_data,mat2row=False))
		p11, = ax3.plot(z_array,gdat.T[0],color='k',linestyle='-',linewidth=2,alpha=0.75)
		p12, = ax3b.plot(z_array,gdat.T[1],color='k',linestyle='--',linewidth=2,alpha=0.75)
		ax3.legend([p11,p12],[r'$\langle x_{HI}\rangle$',r'$\langle\delta T_{b}\rangle$'],loc=1)
		ax3.set_xlabel(r'$z$',fontsize=16)
		ax3.set_ylabel(r'$\langle x_{HI}\rangle$',fontsize=16)
		ax3b.set_ylabel(r'$\langle\delta T_{b}\rangle$ (mK)',fontsize=16)
		ax3.set_xlim(6,25)
		ax3.set_ylim(0,1.1)
		ax3b.set_ylim(-200,100)

		# Plot Eigenmodes
		ax5b = ax5.twinx()
		evecs = np.array(map(lambda x: O.track(['xH','Tb'],arr=O.mat2row(x,mat2row=False)),E.eig_vecs))
		evecs = np.array(map(lambda x: x/map(np.max,np.abs(x.T)), evecs) )
		p13, = ax5.plot(z_array,evecs[0].T[0],linestyle='-',color='b',linewidth=1.5,alpha=0.5)
		p14, = ax5.plot(z_array,evecs[1].T[0],linestyle='-',color='r',linewidth=1.5,alpha=0.5)
		p16, = ax5b.plot(z_array,evecs[0].T[1],linestyle='--',color='b',linewidth=1.5,alpha=0.5)
		p17, = ax5b.plot(z_array,evecs[1].T[1],linestyle='--',color='r',linewidth=1.5,alpha=0.5)
		ax5.axhline(0,color='k')
		ax5.set_xlabel(r"$z$",fontsize=16)
		ax5.set_ylabel(r'$\phi_{x_{HI}}$',fontsize=16)
		ax5b.set_ylabel(r'$\phi_{T_{B}}$',fontsize=16)
		ax5.set_xlim(6,25)
		ax5.set_ylim(-1.1,1.1)
		ax5b.set_ylim(-1.1,1.1)

		fig.savefig('data_compress.png',dpi=200,bbox_inches='tight')
		mp.close()

	calibrate_error = True
	add_lnlike_cov = True
	if cross_validate_ps == True:
		recon, recon_err, weights, true_w, weights_err = cross_validate()
		calc_errs(recon,recon_err)
		plot_cross_valid(fname='cross_validate_ps.png')

		if calibrate_error == True:
			resid_68 = np.array(map(astats.biweight_midvariance, data_cv.T - recon.T))
			resid_68 += np.array(map(astats.biweight_location, data_cv.T - recon.T))
			mean_pred_err = np.array(map(astats.biweight_location, recon_err.T))
			err_norm = resid_68 / mean_pred_err
			E.recon_err_norm = err_norm
			recon_err *= E.recon_err_norm

		if add_lnlike_cov == True:
			print_message('...adding pspec cross validated errors to lnlike covariance as weights',type=0)
			X = recon.T-data_cv.T
			ps_err_cov = np.inner(X,X) / len(recon)
			lnprob_kwargs['add_lnlike_cov'] = ps_err_cov

			fig = mp.figure(figsize=(5,5))
			ax = fig.add_subplot(111)
			ax.set_title('Emulator Error Covariance',fontsize=10)
			im1 = ax.matshow(ps_err_cov,origin='lower',cmap='seismic',vmin=-1,vmax=1)
			fig.colorbar(im1,label='covariance')
			ax.set_xlabel(r'$d$',fontsize=15)
			ax.set_ylabel(r'$d$',fontsize=15)
			fig.savefig('emu_cov.png',dpi=100,bbox_inches='tight')
			mp.close()


	if cross_validate_like == True:
		print_message('...cross validating likelihoods')
		print_time()

		e_like,t_like = S.cross_validate(grid_cv,data_cv,lnlike_kwargs=lnprob_kwargs)#,also_record=['lnlike_emu_err'])
		fig = mp.figure(figsize=(5,5))
		ax = fig.add_subplot(111)
		frac_err = (e_like-t_like)/t_like
		try: patches = ax.hist(frac_err,bins=40,histtype='step',range=(-1.0,1.0),normed=True,color='b')
		except UnboundLocalError: pass
		ax.set_xlim(-0.5,0.5)
		ax.set_xlabel('likelihood fractional error',fontsize=14)
		ax.annotate(r'$\sigma = '+str(np.around(astats.biweight_midvariance(frac_err)*50,2))+'\%$',xy=(0.2,0.8),xycoords='axes fraction',fontsize=18)
		fig.savefig('cross_validate_like.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()


	print_message('...making plots')
	print_time()


	# Plot Eigenmode Weight Prediction
	if plot_weight_pred == True:
		plot_modes = [0,1,2]
		plot_params = [1]

		gs = gridspec.GridSpec(4,4)
		gs1 = gs[:3,:]
		gs2 = gs[3,:]

		for p in plot_params:
			for plot_mode in plot_modes:
				# Plot regression of weights
				fig=mp.figure(figsize=(10,10))
				fig.subplots_adjust(hspace=0.1)

				sel = np.array(reduce(operator.mul,np.array([grid_cv.T[i]==params_fid[i] if i != p else np.ones(len(grid_cv.T[i])) for i in range(N_params)])),bool)
				sort = np.argsort(grid_cv.T[p][sel])
				grid_x = grid_cv.T[p][sel][sort]
				pred_weight = weights.T[plot_mode][sel][sort]
				true_weight = true_w.T[plot_mode][sel][sort]
				pred_weight_err = weights_err.T[plot_mode][sel][sort]

				ax1 = fig.add_subplot(gs1)
				ax1.grid(True)
				a0 = ax1.fill_between(grid_x,pred_weight+pred_weight_err,pred_weight-pred_weight_err,color='b',alpha=0.2)
				a1, = ax1.plot(grid_x,true_weight,'r.',markersize=14,alpha=0.3)
				a2, = ax1.plot(grid_x,pred_weight,'k',linewidth=2.5)
				ax1.set_ylabel(r'eigenmode weight',fontsize=15)
				ax1.legend([a1,a2],[r'True Weights',r'Prediction'])
				ylim = None,None
				ax1.set_ylim(ylim)
				ax1.tick_params(axis='x',which='both',bottom='off',labelbottom='off')
				ax1.set_title(r'Prediction of Eigenmode '+str(plot_mode),fontsize=15)

				ax2 = fig.add_subplot(gs2)
				ax2.grid(True)
				ax2.axhline(0,color='r',linewidth=1,alpha=0.75,linestyle='--')
				a0 = ax2.fill_between(grid_x,pred_weight+pred_weight_err-true_weight,pred_weight-pred_weight_err-true_weight,color='b',alpha=0.2)
				a2, = ax2.plot(grid_x,pred_weight-true_weight,'k',linewidth=2.5)
				[l.set_rotation(30) for l in ax2.get_xticklabels()]
				ylim = None,None
				ax2.set_xlabel(p_latex[p],fontsize=20)
				ax2.set_ylabel(r'$\hat{\mathbf{w}}-\mathbf{w}$',fontsize=18)

				fig.savefig('weight_pred_'+params[p]+'_eig'+str(plot_mode)+'.png',dpi=200,bbox_inches='tight')
				mp.close()

	# Plot PS Prediction
	if plot_ps_pred == True:
		plot_kbins = [50,100,150,200]
		plot_params = [0]

		gs = gridspec.GridSpec(4,4)
		gs1 = gs[:3,:]
		gs2 = gs[3,:]

		for p in plot_params:
			for kbin in plot_kbins:
				# Plot regression of weights
				fig=mp.figure(figsize=(8,8))
				fig.subplots_adjust(hspace=0.1)

				sel = np.array(reduce(operator.mul,np.array([grid_cv.T[i]==params_fid[i] if i != p else np.ones(len(grid_cv.T[i])) for i in range(N_params)])),bool)
				sort = np.argsort(grid_cv.T[p][sel])
				grid_x = grid_cv.T[p][sel][sort]
				pred_ps = recon.T[kbin][sel][sort]
				true_ps = data_cv.T[kbin][sel][sort]
				pred_ps_err = recon_err.T[kbin][sel][sort]
				yz = O.mat2row(yz_data,mat2row=True)[kbin]

				ax1 = fig.add_subplot(gs1)
				ax1.grid(True)
				a0 = ax1.fill_between(grid_x,pred_ps+pred_ps_err,pred_ps-pred_ps_err,color='b',alpha=0.2)
				a1, = ax1.plot(grid_x,true_ps,'r.',markersize=14,alpha=0.3)
				a2, = ax1.plot(grid_x,pred_ps,'k',linewidth=2.5)
				ax1.set_ylabel(r'$\Delta^{2}$',fontsize=18)
				ax1.legend([a1,a2],[r'True PS',r'Prediction'])
				ylim = None,None
				ax1.set_ylim(ylim)
				ax1.tick_params(axis='x',which='both',bottom='off',labelbottom='off')
				ax1.annotate(r'$k = '+str(np.round(float(yz[0])/0.7,2))+'\ h\ Mpc^{-1}$\n$z = '+yz[1]+'$',xy=(0.05,0.85),xycoords='axes fraction',fontsize=17)

				ax2 = fig.add_subplot(gs2)
				ax2.grid(True)
				ax2.axhline(1,color='r',linewidth=1,alpha=0.75,linestyle='--')
				a0 = ax2.fill_between(grid_x,(pred_ps+pred_ps_err)/true_ps,(pred_ps-pred_ps_err)/true_ps,color='b',alpha=0.2)
				a2, = ax2.plot(grid_x,pred_ps/true_ps,'k',linewidth=2.5)
				[l.set_rotation(30) for l in ax2.get_xticklabels()]
				ylim = None,None
				ax2.set_xlabel(p_latex[p],fontsize=20)
				ax2.set_ylabel(r'$\hat{\Delta^{2}}/\Delta^{2}$',fontsize=18)

				fig.savefig('ps_pred_'+params[p]+'_kbin'+str(kbin)+'.png',dpi=200,bbox_inches='tight')
				mp.close()

	# Plot Fractional PS Reconstruction fraction
	if plot_ps_recon == True:

		# Pick redshifts
		z_arr = np.arange(44)[2:-16][::2]

		# Calcualte error
		ps_frac_err = recon / data_cv

		# Iterate through redshifts
		for z in z_arr:
			# Plot
			fig = mp.figure(figsize=(8,8))
			ax = fig.add_subplot(111)
			ax.grid(True,which='both')

			# Iterate through cv
			pspec_ratio = []
			for i in range(len(grid_cv)):
				pspec_ratio.append(O.track(['ps'],arr=O.mat2row(recon[i]))[z]/O.track(['ps'],arr=O.mat2row(data_cv[i]))[z])
				ax.plot(O.track(['ps'],arr=O.xdata)[z],pspec_ratio[-1],color='b',alpha=0.2)

			pspec_ratio = np.array(pspec_ratio)
			loc = np.array(map(astats.biweight_location,pspec_ratio.T))
			stdev = np.array(map(astats.biweight_midvariance,pspec_ratio.T))

			ax.fill_between(O.track(['ps'],arr=O.xdata)[z],loc-stdev,loc+stdev,color='k',alpha=0.75)
			ax.plot(O.track(['ps'],arr=O.xdata)[z],loc,color='k',alpha=0.65,linewidth=2)

			ax.annotate(r'$z='+str(z_array[z])+'$',xy=(0.1,0.8),xycoords='axes fraction',fontsize=18)

			ax.set_xlim(1e-1,1)
			ax.set_ylim(0.5,1.5)
			ax.set_xscale('log')
			ax.set_xlabel(r'$k\ (Mpc^{-1})$',fontsize=18)
			ax.set_ylabel(r'$\hat{\Delta^{2}}/\Delta^{2}$',fontsize=18)

			fig.savefig('ps_frac_recon_z'+str(z_array[z])+'.png',dpi=200,bbox_inches='tight')
			mp.close()

	# Plot PS Reconstruction
	if plot_ps_recon == True:

		# Pick Redshifts
		z_arr = np.arange(44)[5:27][::4]

		recon_cut = np.array(map(lambda x: O.track(['ps'], arr=O.mat2row(x))[z_arr], recon))

		# Iterate through redshift
		for i in range(len(z_arr)):
			# Plot
			fig = mp.figure(figsize=(5,5))
			ax = fig.add_subplot(111)
			ax.grid(True, which='both')

			# Calculate standard dev
			recon_z = np.array(map(lambda x: x, recon_cut.T[i]))
			stand_dev = np.array(map(lambda x: astats.biweight_midvariance(x), recon_z.T))

			# Plot
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_xlabel(r'$k$ (Mpc$^{-1}$)', fontsize=16)
			ax.set_ylabel(r'$\Delta^{2}_{T_{b}}$ (mK$^{2}$)', fontsize=16)
			fid = O.track(['ps'], arr=O.mat2row(E.fid_data))[z_arr[i]]
			ax.fill_between(O.track(['ps'])[z_arr[i]], fid + stand_dev, fid-stand_dev, color='b', alpha=0.25)
			ax.plot(O.track(['ps'])[z_arr[i]], fid, color='k', linewidth=2, alpha=0.8)
			ax.set_ylim(1e1,1e4)
			ax.annotate(r'$z = '+str(z_array[z_arr[i]])+'$', xy=(0.2,0.7), xycoords='axes fraction', fontsize=12)

			fig.savefig('ps_recon_z'+str(z_array[z_arr[i]])+'.png', dpi=200, bbox_inches='tight')
			mp.close()


	# Make Fisher Info
	if make_fisher == True:
		pspecs = []
		epsilon = 0.01
		for i in range(11):
			ptemp = []
			for j in range(2):
				p = np.copy(p_true)
				if j == 0:
					p[i] += -epsilon*p[i]
				else:
					p[i] += epsilon*p[i]
				E.predict(p,predict_kwargs=predict_kwargs)
				ptemp.append(O.track(['ps'],arr=O.mat2row(E.recon[0]),mat=False))
			pspecs.append(ptemp)
		pspecs = np.array(pspecs)

		partial_PS = []
		partial_TH = []
		for i in range(11):
			partial_PS.append(pspecs[i][1] - pspecs[i][0])
			partial_TH.append(p_true[i]*0.02*2)

		partial_PS = np.array(partial_PS)
		partial_TH = np.array(partial_TH)
		yerrs = O.track(['ps'],arr=O.mat2row(O.yerrs),mat=False)

		F = np.zeros((N_params,N_params))
		for i in range(N_params):
			for j in range(N_params):
				F[i,j] = sum( (partial_PS[i]/partial_TH[i]).ravel() * (partial_PS[j]/partial_TH[j]).ravel() / yerrs**2)


	print_time()
	# Initialize Ensemble Sampler
	print_message('...initializing ensemble sampler')
	print_time()

	date = time.ctime()[4:].split()[:-1]
	date = ''.join(date[:2])+'_'+re.sub(':','_',date[-1])
	print_message('...date is '+date)
	sampler_kwargs = {}

	#pool = pathos.multiprocessing.Pool(5) #emcee.utils.MPIPool()
	#sampler_kwargs.update({'pool':pool})

	S.emcee_init(nwalkers, ndim, S.lnprob, lnprob_kwargs=lnprob_kwargs, sampler_kwargs=sampler_kwargs)

	# Initialize Walker positions
	#pos = np.array(map(lambda x: x + x*stats.norm.rvs(0,0.05,nwalkers),p_true)).T
	pos = np.copy(grid_tr[np.random.choice(np.arange(len(data_tr)), replace=False, size=nwalkers)])


	# Add priors (other than flat priors)
	if add_priors == True:
		print_message('...adding non-flat priors',type=0)
		#planck_cov = np.loadtxt('base_TTTEEE_lowl_plik.covmat')[[0,1,5]].T[[0,1,5]].T
		select_arr = np.array([5,6,0,1,4])
		planck_cov = np.loadtxt('new_planck_cov.tab')[select_arr[:,None],select_arr]
		std_multiplier = 1.0

		# Add non-correlated Gaussian Priors
		prior_params = []
		prior_indices = []
		priors = map(lambda x: common_priors.cmb_priors1[x+'_err'] * std_multiplier, prior_params)
		for i in range(len(priors)):
			S.lnprior_funcs[prior_indices[i]] = S.create_gauss_lnprior(p_true[prior_indices[i]],priors[i],\
						index=prior_indices[i],return_func=True)
			
		# Add correlated Gaussian Priors
		prior_params = ['sigma8','hlittle','ombh2','omch2','ns']
		prior_indices = [0,1,2,3,4]
		Nindices = len(prior_indices)
		prior_cov = np.zeros((N_params,N_params))
		for i in range(Nindices):
			for j in range(Nindices):
				prior_cov[prior_indices[i],prior_indices[j]] = planck_cov[i,j] * std_multiplier**2
		for i in range(N_params):
			prior_cov[i,i] += 1e-12
		prior_prec = la.inv(prior_cov)            
		for i in range(len(prior_params)):
			S.lnprior_funcs[prior_indices[i]] = S.create_covarying_gauss_lnprior(p_true,prior_prec,\
					index=prior_indices[i],return_func=True)

	print_time()

	if time_sampler == True:
		print_message('...timing sampler')
		ipython.magic("timeit -r 3 S.samp_drive(pos,step_num=1,burn_num=0)")

	if drive_sampler == True:
		print_message('...driving sampler',type=1)
		print_time()
		# Drive Sampler
		burn_num = 0
		step_num = 100
		print_message('...driving with burn_num='+str(burn_num)+', step_num='+str(step_num),type=0)
		S.samp_drive(pos,step_num=step_num,burn_num=burn_num)
		chain = S.sampler.chain
		samples = chain[:, 0:, :].reshape((-1, S.ndim))
		print("Mean acceptance fraction: {0:.3f}".format(np.mean(S.sampler.acceptance_fraction)))

		if save_chains == True:
			f = open('samp_chains_'+date+'.pkl','wb')
			output = pkl.Pickler(f)
			output.dump({'chain':S.sampler.chain,'burn_num':burn_num,'step_num':step_num,\
						'acceptance_frac':np.mean(S.sampler.acceptance_fraction),'ndim':S.ndim})
			f.close()

		print_time()

	load_chains = False
	if load_chains == True:
		fname = 'samp_chains_1planck.pkl'
		f = open(fname,'rb')
		input = pkl.Unpickler(f)
		chain_d = input.load()
		f.close()
		chain = chain_d['chain']
		thin = 10
		samples = chain[:,500::thin,:].reshape((-1,chain_d['ndim']))

	if trace_plots == True:
		print_message('...plotting trace plots')
		print_time()
		fig = mp.figure(figsize=(16,8))
		fig.subplots_adjust(wspace=0.4,hspace=0.1)

		for i in range(N_params):
			ax = fig.add_subplot(3,4,i+1)
			ax.set_ylabel(p_latex[i],fontsize=20)
			mp.tick_params(which='both',right='off',top='off')
			for j in range(len(chain))[::nwalkers/nwalkers]:
				ax.plot(chain[j,:,i],color='k',alpha=0.9)
				ax.set_ylim(param_bounds[i])
			ax.axhline(p_true[i],color='r',alpha=0.5,linewidth=3)

		fig.savefig('trace_plots_'+date+'.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()

	if autocorr_plots == True:
		print_message('...plotting autocorrelations')
		print_time()

		fig = mp.figure(figsize=(16,8))
		fig.subplots_adjust(wspace=0.4,hspace=0.2)

		maxlag = 350
		thin = 10
		for i in range(N_params):
			ax = fig.add_subplot(3,4,i+1)
			ax.set_ylim(-1,1)
			ax.axhline(0,color='k',alpha=0.5)
			mp.tick_params(which='both',right='off',top='off')
			ax.set_ylabel(p_latex[i],fontsize=20)
			series = chain[10,:,:].T[i][::thin]
			ax.acorr(series-astats.biweight_location(series),maxlags=maxlag)
			ax.set_xlim(-0,maxlag)

		fig.savefig('autocorrs_'+date+'.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()
		
	if tri_plots == True:
		print_message('...plotting triangle plots')
		print_time()

		# Triangle Plot
		levels = [0.68,0.95]

		p_eps = [0.1 for i in range(5)] + [0.4 for i in range(6)]
		p_eps = np.array(map(astats.biweight_midvariance,samples.T))*4
		p_lims = None #[[None,None] for i in range(N_params)]
		p_lims = [[fid_params[i]-p_eps[i],fid_params[i]+p_eps[i]] for i in range(N_params)]
#		p_lims = [[grid_tr.T[i].min(), grid_tr.T[i].max()] for i in range(N_params)]

		label_kwargs = {'fontsize':20}

		print '...plotting triangle'
		fig = corner.corner(samples, labels=p_latex, label_kwargs=label_kwargs,
							truths=p_true, range=p_lims, levels=levels, smooth=0.2,
							truth_color='red')

		add_fg_colors = True
		if add_fg_colors == True:
			axes = np.array(fig.axes).reshape(11,11)[1:5,0:4]
			axes = np.array([axes[i][:i+1] for i in range(len(axes))])
			axes = np.array([val for sublist in axes for val in sublist])
			for ax in axes:
				ax.patch.set_facecolor('purple')
				ax.patch.set_alpha(0.1)

			axes = np.array(fig.axes).reshape(11,11)[5:11,0:5].ravel()
			for ax in axes:
				ax.patch.set_facecolor('green')
				ax.patch.set_alpha(0.1)

			axes = np.array(fig.axes).reshape(11,11)[6:11,5:10]
			axes = np.array([axes[i][:i+1] for i in range(len(axes))])
			axes = np.array([val for sublist in axes for val in sublist])
			for ax in axes:
				ax.patch.set_facecolor('orange')
				ax.patch.set_alpha(0.1)

			p0 = matplotlib.patches.Rectangle([0,0],0,0,color='purple',alpha=0.1)
			p1 = matplotlib.patches.Rectangle([0,0],0,0,color='green',alpha=0.1)
			p2 = matplotlib.patches.Rectangle([0,0],0,0,color='orange',alpha=0.1)
			fig.legend([p0,p1,p2],['Cosmo-Cosmo','Cosmo-Astro','Astro-Astro'],fontsize=60,loc='upper center',frameon=False)

		add_fisher = False
		load_fisher = False
		if add_fisher == True:
			print'...adding fisher contours'

			if load_fisher == True:
				# Load fisher contours
				f = open('fisher_matrix.pkl','rb')
				input = pkl.Unpickler(f)
				fisher_d = input.load()
				f.close()
				F = fisher_d['F']
		
			# Load cosmological covariance
			select_arr = np.array([5,6,0,1,4])
			planck_cov = np.loadtxt('new_planck_cov.tab')[select_arr[:,None],select_arr]
			planck_len = len(planck_cov)

			# Add inverse of planck cov to F
			cov_mult = 1.0	
			planck_invcov = la.inv(cov_mult*planck_cov)
			F[:planck_len,:planck_len] += planck_invcov
			C = la.inv(F)

			# Plot
			axes = np.array(fig.axes).reshape(11,11).T
			for i in range(N_params):
				for j in range(i+1,N_params):
					# Construct marginalized covariance matrix between two parameters
					cov = np.array([[C[i,i],C[i,j]],[C[j,i],C[j,j]]])
					x_cent = p_true[i]
					y_cent = p_true[j]
					plot_kwargs = {'color':'b','linestyle':'-','linewidth':2,'alpha':0.8,'zorder':5}
					# Plot
					plot_ellipse(ax=axes[i,j],x_cent=p_true[i],y_cent=p_true[j],plot_kwargs=plot_kwargs,cov=cov,mass_level=0.95)


		plot_ts = False
		if plot_ts == True:
			print '...plotting TS'
			j = 1
			rando = np.random.choice(np.arange(0,len(grid_tr)),size=len(grid_tr),replace=False)
			for i in range(N_params-1):
				xp_ind = i
				ax_ind = [N_params*j + i + k*N_params for k in range(N_params-j)]
				for l,h in zip(range(j,N_params),range(N_params-j)):
					yp_ind = l
					fig.axes[ax_ind[h]].plot(grid_tr.T[xp_ind][rando],grid_tr.T[yp_ind][rando],'r.',alpha=0.1)
					#fig.axes[ax_ind[h]].plot(pos.T[xp_ind],pos.T[yp_ind],'c.',alpha=0.0)
				j += 1

		fig.savefig('tri_plot_'+date+'.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()

	if plot_boxplots == True:
		# Plot marginalized pdfs and compare to priors
		gs = gridspec.GridSpec(11,20)
		sub1 = np.array([np.arange(0,10,3) for i in range(3)]).T.ravel()
		sub2 = np.array([np.arange(2,12,3) for i in range(3)]).T.ravel()
		sub3 = np.array([np.arange(0,15,7) for i in range(4)]).ravel()
		sub4 = np.array([np.arange(6,21,7) for i in range(4)]).ravel()

		samps = samples[np.random.choice(np.arange(len(samples)),replace=False,size=len(samples)/5)]
		center = True

		fig = mp.figure(figsize=(10,7))
		fig.subplots_adjust(wspace=0.2,hspace=1.0)

		for i in range(N_params):
			# init axes
			ax = fig.add_subplot(gs[sub1[i+1]:sub2[i+1],sub3[i+1]:sub4[i+1]])
			mp.tick_params(which='both',left='off',right='off')
			ax.set_xlabel(p_latex[i],fontsize=20)
			if len(S.lnprior_funcs[i].func_defaults) == 5:
				# gaussian prior
				twosig = 2*np.sqrt(la.inv(S.lnprior_funcs[i].func_defaults[1]).diagonal()[i])
			else:
				# flat prior
				twosig = (param_bounds[i][1]-param_bounds[i][0])/2.0
				if twosig > p_true[i]: twosig = 1*p_true[i]
			# fill in prior
			ax.axvspan(p_true[i]-twosig,p_true[i]+twosig,color='red',alpha=0.25)
			perc1 = np.percentile(samps.T[i],2.2275)
			perc2 = np.percentile(samps.T[i],97.7725)
			perc50 = np.percentile(samps.T[i],50)
			if center == True:
				posterior_twosig = (perc2-perc1)/2.0
				mid = p_true[i]
			else:
				posterior_twosig = (perc2-perc1)/2.0
				mid = perc50
			ax.axvline(mid-posterior_twosig,ymin=0.2,ymax=0.8,color='blue',linewidth=3)
			ax.axvline(mid+posterior_twosig,ymin=0.2,ymax=0.8,color='blue',linewidth=3)
			ax.plot([mid-posterior_twosig,mid+posterior_twosig],[0,0],color='blue',linewidth=1.5)
			ax.tick_params('both',length=6)
			ax.set_yticklabels([])
			mp.xticks(ax.get_xticks()[::2],fontsize=15)
			ax.set_xlim(p_true[i]-1.2*twosig,p_true[i]+1.2*twosig)

		p0 = matplotlib.patches.Rectangle((0,0),0,0,color='red',alpha=0.25)
		p1 = matplotlib.lines.Line2D([0],[0],color='blue',linewidth=3)
		ax = fig.add_subplot(gs[sub1[0]:sub2[0],sub3[0]:sub4[0]])
		mp.axis('off')
		ax.legend([p0,p1],[r'$2\sigma$ Prior',r'$2\sigma$ Posterior'],fontsize=15,frameon=False)

		fig.savefig('boxplot_'+date+'.png',dpi=200,bbox_inches='tight')
		mp.close()


	if ps_var_movie == True:

		# Make directory structure
		outf = 'ps_movie_'
		outdir = 'ps_movie'
		os.system('mkdir '+outdir)

		# Load original data
		draw_data()

		# Iterate over "cross" cross-validation set
		spp = 50		# Samples per parameter

		num = 0
		for i in range(N_params):
			# Isolate samples along this axis
			sel = np.array(reduce(operator.mul,np.array([grid_cv.T[p]==params_fid[p] if p != i else np.ones(len(grid_cv.T[p])) for p in range(N_params)])),bool)
			sort = np.argsort(grid_cv[sel].T[i])
			ps_track = lambda datavec, z: datavec.reshape(z_len,y_len)[z,:k_len]
			xH_track = lambda datavec, z: datavec.reshape(z_len,y_len)[z,-2]
			tb_track = lambda datavec, z: datavec.reshape(z_len,y_len)[z,-1]
	
			for j in np.concatenate([np.arange(0,spp/2)[::-1],[0]*10,np.arange(0,spp),[spp-1]*10,np.arange(spp/2,spp)[::-1],[spp/2]*5]):
				num += 1
				if num <= 569: continue
				outfile = outf + "%04d" % num + ".png"

				# Make power spectrum variability movie
				gs = gridspec.GridSpec(16,30)

				sx = [np.array([8*i,8*i+6]) for i in range(4)]
				sx += [np.array([8*i,8*i+6]) for i in range(4)]
				sx += [np.array([5*i,5*i+4]) for i in range(6)]
				sx += [np.array([5*i,5*i+4]) for i in range(5)]

				sy = [np.array([0,4]) for i in range(4)]
				sy += [np.array([6,10]) for i in range(4)]
				sy += [np.array([12,13]) for i in range(6)]
				sy += [np.array([15,16]) for i in range(5)]

				fig = mp.figure(figsize=(10,5))
				fig.subplots_adjust(wspace=0.05,hspace=0.15)

				# Iterate over subplots
				for k in range(19):
					ax = fig.add_subplot(gs[sy[k][0]:sy[k][1],sx[k][0]:sx[k][1]])
					# Plot top row power spectra
					if k <= 3:
						z_plot = [5,13,21,29]
						ax.set_xlim(0.015,2.5)
						ax.set_ylim(1e0,1e4)
						ax.set_xscale('log')
						ax.set_yscale('log')
						ax.set_xlabel(r'$k\ ($Mpc$^{-1})$',fontsize=14,labelpad=1)
						ax.grid(True)
						mp.tick_params(which='both',right='off',top='off')
						ax.set_title(r'$z = '+str(z_array[z_plot[k]])+'$',fontsize=15)
						if k == 0: ax.set_ylabel(r'$\Delta^{2}_{\delta T_{b}}$ (mK$^{2}$)',fontsize=16,labelpad=0)
						ax.plot(k_range, ps_track(data_cv[sel][sort][j],z_plot[k]), color='k', linewidth=2)
					# Plot second row ps evol1 k = 0.15
					if k == 4:
						ax.set_xlim(6,25)
						ax.set_ylim(1e0,1e4)
						ax.set_yscale('log')
						ax.grid(True)
						ax.set_xlabel(r'$z$',fontsize=15,labelpad=0)
						ax.set_ylabel(r'$\Delta^{2}_{\delta T_{b}}$ (mK$^{2}$)',fontsize=16,labelpad=0)
						mp.tick_params(which='both',right='off',top='off')
						ax.plot(z_array,ps_track(data_cv[sel][sort][j],np.arange(z_len)).T[5], color='r', linewidth=2)
						ax.annotate(r'$k=0.15$ Mpc$^{-1}$',xy=(0.2,0.8),xycoords='axes fraction',fontsize=11)
					if k == 5:
						ax.set_xlim(6,25)
						ax.set_ylim(1e0,1e4)
						ax.set_yscale('log')
						ax.grid(True)
						ax.set_xlabel(r'$z$',fontsize=15,labelpad=0)
						ax.plot(z_array,ps_track(data_cv[sel][sort][j],np.arange(z_len)).T[9], color='r', linewidth=2)
						ax.annotate(r'$k=0.8$ Mpc$^{-1}$',xy=(0.2,0.8),xycoords='axes fraction',fontsize=11)
					# Plot Second row Xe evol
					if k == 6:
						ax.set_xlim(6,25)
						ax.set_ylim(0,1.1)
						ax.set_xlabel(r'$z$',fontsize=15,labelpad=0)
						ax.set_ylabel(r'$x_{HI}$',fontsize=16,labelpad=0)
						ax.grid(True)
						mp.tick_params(which='both',right='off',top='off')
						ax.plot(z_array,xH_track(data_cv[sel][sort][j],np.arange(z_len)), color='b', linewidth=2)
					# Plot Second row tb evol
					if k == 7:
						ax.set_xlim(6,25)
						ax.set_ylim(-200,50)
						ax.set_xlabel(r'$z$',fontsize=15,labelpad=0)
						ax.set_ylabel(r'$\delta T_{b}$',fontsize=16)
						ax.grid(True)
						mp.tick_params(which='both',left='off',top='off')
						ax.yaxis.tick_right()
						ax.yaxis.set_label_position('right')
						ax.plot(z_array,tb_track(data_cv[sel][sort][j],np.arange(z_len)), color='g', linewidth=2)
					# Plot first row of parameters
					if k >= 8 and k <= 13:
						mp.tick_params(which='both',left='off',right='off',top='off',labelleft='off')
						ax.set_xlabel(p_latex[k-8],fontsize=14,labelpad=0)
						ax.axvline(grid_cv[sel][sort].T[k-8][j],color='k',alpha=0.75,linewidth=2.5)
						ax.set_xlim(param_bounds[k-8][0],param_bounds[k-8][1])
						ax.set_xticks(ax.get_xticks()[::2])
						[l.set_rotation(25) for l in ax.get_xticklabels()]
						[l.set_fontsize(10) for l in ax.get_xticklabels()]
					# Plot second row of parameters
					if k > 13:
						mp.tick_params(which='both',left='off',right='off',top='off',labelleft='off')
						ax.set_xlabel(p_latex[k-8],fontsize=14,labelpad=0)
						ax.axvline(grid_cv[sel][sort].T[k-8][j],color='k',alpha=0.75,linewidth=2.5)
						ax.set_xlim(param_bounds[k-8][0],param_bounds[k-8][1])
						ax.set_xticks(ax.get_xticks()[::2])
						[l.set_rotation(25) for l in ax.get_xticklabels()]
						[l.set_fontsize(10) for l in ax.get_xticklabels()]

				fig.savefig(outdir+'/'+outfile,dpi=150,bbox_inches='tight')
				mp.close()










