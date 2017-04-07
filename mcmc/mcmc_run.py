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
from matplotlib import ticker
from fits_table import fits_table,fits_to_array,fits_data
from curve_poly_interp import curve_interp
from plot_ellipse import plot_ellipse
import astropy.io.fits as fits
import cPickle as pkl
from memory_profiler import memory_usage
import operator
from mpl_toolkits.mplot3d import Axes3D
from sklearn import gaussian_process as gp
from sklearn import covariance
import astropy.stats as astats
import scipy.stats as stats
import corner
import warnings
from pycape import common_priors
import pycape
import emupy
import time
from mpi4py import MPI
import emcee
import re
from round_to_n import round_to_n
import pathos
import dill
import py21cmsense
from biweight_midcovariance import biweight_midcovariance
warnings.filterwarnings('ignore',category=DeprecationWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

try:
	from IPython import get_ipython
	ipython = get_ipython()
except: pass

mp.rcParams['font.family'] = 'sans-serif'
mp.rcParams['font.sans-serif'] = ['Helvetica']
mp.rcParams['text.usetex'] = True


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
	def draw_data(keep_meta=['ps']):

		make_globals = ['tr_len','data_tr','grid_tr','data_cv','grid_cv','fid_params','fid_data',
						'keep_meta','data_od','grid_od','tr_name','cv_name']

		# Load Datasets
		file = open('lhs_data.pkl','rb')
		lhs_data = pkl.Unpickler(file).load()
		lhs_data['name'] = 'lhs_data'
		file.close()

		file = open('gauss_hera127_data.pkl','rb')
		gauss_hera127_data = pkl.Unpickler(file).load()
		gauss_hera127_data['name'] = 'gauss_hera127_data'
		file.close()

		file = open('gauss_hera331_data.pkl','rb')
		gauss_hera331_data = pkl.Unpickler(file).load()
		gauss_hera331_data['name'] = 'gauss_hera331_data'
		file.close()

		file = open('lhsfs_hera331_data.pkl','rb')
		lhsfs_hera331_data = pkl.Unpickler(file).load()
		lhsfs_hera331_data['name'] = 'lhsfs_hera331_data'
		file.close()

		file = open('cross_valid_data.pkl','rb')
		cross_valid_data = pkl.Unpickler(file).load()
		cross_valid_data['name'] = 'cross_valid_data'
		file.close()

		file = open('fiducial_data.pkl','rb')
		fiducial_data = pkl.Unpickler(file).load()
		file.close()

		# Set Random State
		RandomState = 1

		# Choose Training Set
		TS_data = lhsfs_hera331_data
		#TS_data = lhs_data

		# Separate Data
		tr_len = 582
		#tr_len = 16344
		rd = np.random.RandomState(RandomState)
		rando = rd.choice(np.arange(tr_len),size=500,replace=False)
		rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))

		tr_name = TS_data['name']
		data_tr = TS_data['data'][np.argsort(TS_data['indices'])][rando]
		grid_tr = TS_data[ 'gridf'][np.argsort(TS_data['indices'])][rando]
		direcs_tr = TS_data['direcs'][np.argsort(TS_data['indices'])][rando]
		meta_tr = TS_data['metadata']
		keep = np.array(map(lambda x: x in keep_meta, meta_tr))
		data_tr = data_tr.T[keep].T

		# Add other datasets
		add_other_data = True
		if add_other_data == True:
			# Choose Training Set
			TS_data = gauss_hera127_data

			# Separate Data
			rd = np.random.RandomState(RandomState)
			tr_len = 5000
			rando = rd.choice(np.arange(tr_len),size=3000,replace=False)
			rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))

			tr_name += '/'+TS_data['name']
			data_tr2 = TS_data['data'][np.argsort(TS_data['indices'])][rando]
			grid_tr2 = TS_data[ 'gridf'][np.argsort(TS_data['indices'])][rando]
			direcs_tr2 = TS_data['direcs'][np.argsort(TS_data['indices'])][rando]
			meta_tr2 = TS_data['metadata']
			keep = np.array(map(lambda x: x in keep_meta, meta_tr2))
			data_tr2 = data_tr2.T[keep].T

			data_tr = np.concatenate([data_tr,data_tr2])
			grid_tr = np.concatenate([grid_tr,grid_tr2])
			direcs_tr = np.concatenate([direcs_tr,direcs_tr2])

		# Add other datasets
		add_other_data = True
		if add_other_data == True:
			# Choose Training Set
			TS_data = gauss_hera331_data

			# Separate Data
			rd = np.random.RandomState(RandomState)
			tr_len = 5000
			rando = rd.choice(np.arange(tr_len),size=4000,replace=False)
			rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))

			tr_name += '/'+TS_data['name']
			data_tr2 = TS_data['data'][np.argsort(TS_data['indices'])][rando]
			grid_tr2 = TS_data[ 'gridf'][np.argsort(TS_data['indices'])][rando]
			direcs_tr2 = TS_data['direcs'][np.argsort(TS_data['indices'])][rando]
			meta_tr2 = TS_data['metadata']
			keep = np.array(map(lambda x: x in keep_meta, meta_tr2))
			data_tr2 = data_tr2.T[keep].T

			data_tr = np.concatenate([data_tr,data_tr2])
			grid_tr = np.concatenate([grid_tr,grid_tr2])
			direcs_tr = np.concatenate([direcs_tr,direcs_tr2])

		print "...added training set: "+tr_name+" of length: "+str(len(data_tr))

		# Choose Cross Validation Set
		CV_data 		= gauss_hera331_data
		TS_remainder	= True
		use_remainder	= True

		# Separate Data
		if TS_remainder == True:
			if use_remainder == True:
				rando = ~rando
			else:
				remainder = np.where(rando==False)[0]
				rando = np.array([False for i in range(tr_len)])
				rando[np.random.choice(remainder,size=2000,replace=False)] = True
			
		else:
			tr_len = 550
			rd = np.random.RandomState(RandomState)
			rando = rd.choice(np.arange(tr_len),size=550,replace=False)
			rando = np.array(map(lambda x: x in rando,np.arange(tr_len)))

		cv_name = CV_data['name']
		data_cv = CV_data['data'][np.argsort(CV_data['indices'])][rando]
		grid_cv = CV_data[ 'gridf'][np.argsort(CV_data['indices'])][rando]
		direcs_cv = np.array(CV_data['direcs'])[np.argsort(CV_data['indices'])][rando]
		meta_cv = CV_data['metadata']
		keep = np.array(map(lambda x: x in keep_meta, meta_cv))
		data_cv = data_cv.T[keep].T

		# Get Fiducial Data
		feed_fid = False
		if feed_fid == True:
			#fid_params = fiducial_data['fid_params']
			#fid_data = fiducial_data['fid_data']
			#fid_meta = fiducial_data['metadata']
			#keep = np.array(map(lambda x: x in keep_meta, fid_meta))
			#fid_data = fid_data[keep]
			fid_params = np.array(map(astats.biweight_location, grid_tr.T))
			param_rad = np.array([astats.biweight_midvariance(grid_tr.T[i]-fid_params[i]) for i in range(len(fid_params))])
			sample_R = np.array(map(la.norm, (grid_tr-fid_params)/param_rad))
			cent = np.where(sample_R == sample_R.min())[0][0]
			fid_params = grid_tr[cent]
			fid_data = data_tr[cent]
		else:
			fid_params = np.array(map(astats.biweight_location,grid_tr.T))
			fid_data = np.array([astats.biweight_location(data_tr.T[i]) if np.isnan(astats.biweight_location(data_tr.T[i])) == False \
									else np.median(data_tr.T[i]) for i in range(len(data_tr.T))])
		
		# Get 1D Sample
		ONE_D_data = cross_valid_data
		data_od1 = ONE_D_data['data'][np.argsort(ONE_D_data['indices'])]
		grid_od1 = ONE_D_data['gridf'][np.argsort(ONE_D_data['indices'])]
		meta_od = ONE_D_data['metadata']
		keep = np.array(map(lambda x: x in keep_meta, meta_od))
		data_od1 = data_od1.T[keep].T

		data_od = []
		grid_od = []
		order = np.arange(len(grid_od1))
		N_params = grid_od1.shape[1]
		for p in range(N_params):
			sel = order[np.array(reduce(operator.mul,np.array([grid_od1.T[i]==params_fid[i] if i != p else np.ones(len(grid_od1.T[i])) for i in range(N_params)])),bool)]
			sort = np.argsort(grid_od1.T[p][sel])
			data_od.append(data_od1[sel][sort])
			grid_od.append(grid_od1[sel][sort])
		data_od = np.array(data_od)
		grid_od = np.array(grid_od)

		globals().update(dez.create(make_globals,locals()))

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


	### Variables for Emulator ###
	N_modes = 40
	N_params = len(params)
	N_data = 660
	N_samples = len(data_tr)
	poly_deg = 6
	reg_meth='gaussian'
	ell = np.array([10 for i in range(N_params)])
	recon_err_calib = 1.0
	recon_calib	= 1.0
	kernel = gp.kernels.RBF(ell)
	scale_by_std = True
	scale_by_obs_errs = False
	scale_by_davg_ov_yerr = True
	davg_maxscale = 10
	norotate = True
	cov_est = lambda x: biweight_midcovariance(x)
	cov_est_name = 'biweight_midcovariance'
	lognorm = True
	norm_weights = True
	w_norm = None

	gp_kwargs = {'kernel':kernel}

	variables.update({'params':params,'N_params':N_params,'N_modes':N_modes,'N_samples':N_samples,'N_data':N_data,
						'reg_meth':reg_meth,'poly_deg':poly_deg,'gp_kwargs':gp_kwargs,'scale_by_std':scale_by_std,
						'scale_by_obs_errs':scale_by_obs_errs,'recon_err_calib':recon_err_calib,'recon_calib':recon_calib,
						'cov_est':cov_est,'lognorm':lognorm,'norm_weights':norm_weights,'w_norm':w_norm,
						'scale_by_davg_ov_yerr':scale_by_davg_ov_yerr,'davg_maxscale':davg_maxscale})


	###########################
	### INITIALIZE EMULATOR ###
	###########################

	# Initialize workspace, emulator
	print_message('...initializing emulator')
	E = emupy.Emu(variables)
	print_mem()

	# Initialize Cholesky
	E.sphere(grid_tr,fid_params=fid_params,save_chol=True,norotate=norotate)
	E.create_tree(E.Xsph)

	print_message('...initializing mockobs')
	print_time()
	print_mem()

	###########################################
	### Load Mock Observation and Configure ###
	###########################################

	# Variables of model data
	rest_freq       = 1.4204057517667   # GHz
	zbin			= 0.5
	z_array         = np.arange(5.5,27.01,zbin)
	data_zlen       = len(z_array)
	z_select        = np.arange(len(z_array))
	k_range         = np.loadtxt('k_range.tab')
	data_klen       = len(k_range)
	k_select        = np.arange(len(k_range))
	g_array         = np.array([])#r'nf',r'Tb'])
	g_array_tex     = np.array([])#r'$\chi_{HI}$',r'$T_{b}$'])
	g_select        = np.arange(len(g_array))

	# Limit to zlimits
	zmin = None
	zmax = None
	if zmin != None:
		select = np.where(z_array[z_select] >= zmin)
		z_select = z_select[select]
	if zmax != None:
		select = np.where(z_array[z_select] <= zmax)
		t_select = z_select[select]

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

	data_ylen = data_klen + g_len
	y_len   = k_len + g_len
	y_array = np.concatenate([k_range,g_array])
	x_len = y_len * z_len

	yz_data         = []
	map(lambda x: yz_data.extend(map(list,zip(y_array,[x]*y_len))),z_array)
	yz_data = np.array(yz_data)
	yz_data = yz_data.reshape(z_len,y_len,2)

	freq_low    = 1.4204057517667 / (z_array+zbin/2.0+1)
	freq_high   = 1.4204057517667 / (z_array-zbin/2.0+1)
	freq_cent   = np.round(1.4204057517667 / (z_array+1),8)
	bandwidth   = np.round(1.4204057517667 / (z_array-zbin/2.0+1) - 1.4204057517667 / (z_array+zbin/2.0+1), 4)

	# Load mock obs
	make_mock = False
	if make_mock == True:
		print_message('...making mock obs with 21cmSense')
		print_time()
		# Get fiducial parameters
		data_filename = 'mockObs_hera331_allz2.pkl'
		mock_direc = 'param_space/mock_obs/zeta_040.000_numin_300.000'
		p_true = np.loadtxt(mock_direc+'/param_vals.tab',usecols=(1,),unpack=True)

		# Initialize 21cmSense class
		CS = py21cmsense.Calc_Sense()

		# Make Array File
		CS.make_arrayfile('hera331', out_fname='hera331_af')

        # Collect ps filenames
		ps_files = np.array(map(lambda x: mock_direc+'/Output_files/Deldel_T_power_spec/ps_interp_z%06.2f.txt'%(x),z_array))
		ps_filenum = len(ps_files)
		lowk_cut = 0.1
		hlittle = fid_params[1]
		omega_m = fid_params[2]/hlittle**2 + fid_params[3]/hlittle**2
		cs_kwargs = {'model':'mod','buff':0.1,'ndays':180,'n_per_day':6,'nchan':82,'verbose':False,
						'hlittle':hlittle,'omega_m':omega_m}

		# Iterate over power spectra to Make 1D Sensitivities
		kbins = []
		PSdata = []
		sense_kbins = []
		sense_PSdata = []
		sense_PSerrs = []
		sense_Terrs = []
		valid = []
		for i in range(ps_filenum):
			# Calc Sense
			CS.calc_sense_1D('hera331_af.npz', out_fname='hera331_sense', freq=freq_cent[i], bwidth=bandwidth[i], eor=ps_files[i], **cs_kwargs)
			# Load Simulation Data
			model = np.loadtxt(ps_files[i])
			kb = model[:,0]
			PSdat = model[:,1]
			# Load 21cmSense errors
			sense = np.load('hera331_sense.npz')
			sense_kb = sense['ks']
			sense_PSerr = sense['errs']
			sense_Terr = sense['T_errs']
			valid.append( (sense_PSerr!=np.inf)&(np.isnan(sense_PSerr)!=True)&(sense_kb>lowk_cut) )
			# Interpolate Simulation onto 21cmSense k-basis
			sense_PSdat = np.interp(sense_kb,kb,PSdat)
			# Append to arrays
			kbins.append(kb)
			PSdata.append(PSdat)
			sense_kbins.append(sense_kb)
			sense_PSdata.append(sense_PSdat)
			sense_PSerrs.append(sense_PSerr)
			sense_Terrs.append(sense_Terr)

		kbins       	= np.array(kbins)
		PSdata      	= np.array(PSdata)
		sense_kbins 	= np.array(sense_kbins)
		sense_PSdata	= np.array(sense_PSdata)
		sense_PSerrs	= np.array(sense_PSerrs)
		sense_Terrs		= np.array(sense_Terrs)
		valid			= np.array(valid)

		# Save data
		with open(data_filename, 'wb') as f1:
			output = pkl.Pickler(f1)
			output.dump({'kbins':kbins,'PSdata':PSdata,'sense_kbins':sense_kbins,'freq':freq_cent,
						'sense_PSdata':sense_PSdata,'sense_PSerrs':sense_PSerrs,'valid':valid,
						'sense_Terrs':sense_Terrs,'p_true':p_true})
		print_time()

	#file = open('mockObs_hera331_outofbounds.pkl','rb')
	file = open('mockObs_hera331_allz.pkl','rb')
	mock_data = pkl.Unpickler(file).load()
	file.close()
	try:
		p_true = mock_data['p_true']
	except:
		p_true = np.array(params_fid)

	## Configure Mock Data and Append to Obs class
	prep_mock_data = mock_data.copy() 
	names = ['sense_kbins','sense_PSdata','sense_PSerrs']
	for n in names:
		prep_mock_data[n] = np.array(prep_mock_data[n],object)
		for i in range(z_len):
			# Cut out inf and nans
			try: prep_mock_data[n][i] = prep_mock_data[n][i].T[prep_mock_data['valid'][i]].T.ravel()
			except: prep_mock_data[n]=list(prep_mock_data[n]);prep_mock_data[n][i]=prep_mock_data[n][i].T[prep_mock_data['valid'][i]].T.ravel()
			if n == 'sense_PSerrs':
				# Cut out sense_PSerrs / sense_PSdata > x% and high k-modes
				err_thresh = 1e3        # 1000%
				hi_k_cut = 2.0
				small_errs = np.where(prep_mock_data['sense_PSerrs'][i] / prep_mock_data['sense_PSdata'][i] < err_thresh)[0]
				hi_k = np.where(prep_mock_data['sense_kbins'][i][small_errs] < hi_k_cut)[0]
				prep_mock_data['sense_kbins'][i] = prep_mock_data['sense_kbins'][i][small_errs][hi_k]
				prep_mock_data['sense_PSdata'][i] = prep_mock_data['sense_PSdata'][i][small_errs][hi_k]
				prep_mock_data['sense_PSerrs'][i] = prep_mock_data['sense_PSerrs'][i][small_errs][hi_k]

	prep_mock_data['sense_kbins'] = np.array( map(lambda x: np.array(x,float), prep_mock_data['sense_kbins']))

	# If model data that comes with Mock Data does not conform to Predicted Model Data, interpolate!
	try:
		residual = k_range - prep_mock_data['kbins'][1]
		if sum(residual) != 0:
			raise Exception
	except:
		# Interpolate onto k-bins of emulated power spectra
		new_kbins  = k_range*1
		new_PSdata = curve_interp(k_range, prep_mock_data['kbins'][0], prep_mock_data['PSdata'].T, n=2, degree=1 ).T
		prep_mock_data['kbins'] = np.array([new_kbins for i in range(z_len)])
		prep_mock_data['PSdata'] = new_PSdata

	model_x		= prep_mock_data['kbins'][z_select]
	obs_x		= prep_mock_data['sense_kbins'][z_select]
	obs_y		= prep_mock_data['sense_PSdata'][z_select]
	obs_y_errs	= prep_mock_data['sense_PSerrs'][z_select]
	obs_track	= np.array(map(lambda x: ['ps' for i in range(len(x))], obs_x))
	track_types = ['ps']

	# Add other information to mock dataset
	if 'nf' in keep_meta:
		model_x		= np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(model_x,z_array)))
		obs_x		= np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_x,z_array)))
		obs_y		= np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y,np.zeros(z_len))))
		obs_y_errs	= np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y_errs,np.ones(z_len)*1e6)))
		obs_track	= np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_track,['nf' for i in range(z_len)])))
		track_types += ['nf']

	if 'Tb' in keep_meta:
		model_x     = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(model_x,z_array)))
		obs_x       = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_x,z_array)))
		obs_y       = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y,np.zeros(z_len))))
		obs_y_errs  = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_y_errs,np.ones(z_len)*1e6)))
		obs_track   = np.array(map(lambda x: np.concatenate([x[0],[x[1]]]), zip(obs_track,['Tb' for i in range(z_len)])))
		track_types += ['Tb']

	obs_y = np.concatenate(obs_y)
	obs_y_errs = np.concatenate(obs_y_errs)
	obs_x_nums = np.array([len(x) for x in obs_x])
	obs_track = np.concatenate(obs_track.tolist())
	track_types = np.array(track_types)

	O = pycape.Obs(model_x,obs_x,obs_y,obs_y_errs,obs_track,track_types,p_true)

	# Feed Mock Observation to Workspace
	update_obs_dic = {'N_data':obs_y.size,'z_len':z_len,'z_array':z_array,
						 'k_range':k_range,'obs_x_nums':obs_x_nums}
	O.update(update_obs_dic)
	E.yerrs = O.yerrs
	print_time()
	print_mem()

	## Plot Mock ##
	plot_mock = True
	if plot_mock == True:
		print_message("...plotting mock obs")
		print_time()
		xdata = mock_data['sense_kbins']
		ydata = mock_data['sense_PSdata']
		yerrs = mock_data['sense_PSerrs']
		valid = mock_data['valid']

		fig = mp.figure(figsize=(8,2))
		fig.subplots_adjust(hspace=0.02)

		i = 0
		for z in np.arange(44)[5:29:6]:
			ax = fig.add_subplot(1,4,i+1)
			ax.grid(True)
			ax.errorbar(xdata[z][valid[z]], ydata[z][valid[z]], yerr=yerrs[z][valid[z]], color='k', fmt='s',
								alpha=0.8, markersize=2.5, ecolor='darkorange',markeredgecolor='None')
			ax.axvspan(6e-2,1e-1, color='grey', alpha=0.2)
			ax.axvspan(6e-2,1e-1, hatch='\\', color='None', alpha=1.0)
			ax.set_xlim(6e-2, 5)
			ax.set_ylim(1, 1e5)
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_title(r'$z = '+str(z_array[z])+'$', fontsize=14)
			ax.set_xlabel(r'$k\ (\mathrm{h\ Mpc}^{-1}$)', fontsize=12)
			if i == 0:
				ax.set_ylabel(r'$\Delta^{2}_{21}\ (\mathrm{mK}^{2}$)', fontsize=13)
			else:
				ax.set_yticklabels([])

			i += 1

		fig.savefig('mock_obs.png', dpi=200, bbox_inches='tight')
		mp.close()
		print_time()

	## Interpolate Training Set onto Observational Basis ##
	print_message('...configuring data for emulation')
	print_time()

	# First interpolate onto observational redshift basis #
	interp_z = False
	if interp_z == True:
		# select out ps
		def z_interp(data):
			ps_select = np.array([[True if i < data_klen else False for i in range(data_ylen)] for j in range(data_zlen)]).ravel()
			gp_select = np.array([[False if i < data_klen else True for i in range(data_ylen)] for j in range(data_zlen)]).ravel()
			ps_data = np.array(map(lambda x: x[ps_select].reshape(data_zlen,data_klen), data))
			gp_data = np.array(map(lambda x: x[gp_select].reshape(data_zlen,g_len), data))
			ps_pred = np.array([10**curve_interp(z_pick,z_array,np.log10(ps_data[:,:,i].T),n=3,degree=2).T for i in range(k_len)]).T
			gp_pred = np.array([curve_interp(z_pick,z_array,ps_data[:,:,i].T,n=3,degree=2).T for i in range(g_len)]).T
			data = [[] for i in range(len(ps_data))]
			for i in range(ps_pred.shape[0]):
				for j in range(len(ps_data)):
					data[j].extend(np.concatenate([ps_pred[i][j],gp_pred[i][j]]))
			return np.array(data)

		data_tr = z_interp(data_tr)
		data_cv = z_interp(data_cv)
		data_od = np.array(map(lambda x: z_interp(x), data_od))
		fid_data = z_interp(fid_data[np.newaxis,:]).ravel()

	# Second Interpolate P Spec onto observational k-mode basis #
	interp_ps = True
	if interp_ps == True:
		def ps_interp(data, logps=True):
			# select out ps and other data
			ps_select = np.array([[True if i < data_klen else False for i in range(data_ylen)] for j in range(data_zlen)]).ravel()
			ps_data = np.array(map(lambda x: x[ps_select].reshape(data_zlen,data_klen), data))
			if logps == True:
				ps_data = np.log10(ps_data)
			other_data = data.T[~ps_select].T.reshape(len(data),data_zlen,g_len)
			ps_track = O.track(['ps'])
			# interpolate (or make prediction)
			ps_pred = np.array([curve_interp(ps_track[i],O.k_range,ps_data[:,i,:].T,n=2,degree=1) for i in range(z_len)])
			if logps == True:
				ps_pred = 10**(ps_pred)
			# reshape array
			data = [[] for i in range(len(ps_data))]
			for i in range(z_len):
				for j in range(len(ps_data)):
					try: data[j].extend(np.concatenate([ps_pred[i].T[j],other_data[j][i]]))
					except: data[j].extend(np.concatenate([np.array([]),other_data[j][i]]))
			return np.array(data)

		data_tr		= ps_interp(data_tr)
		data_cv		= ps_interp(data_cv)
		data_od		= np.array(map(lambda x: ps_interp(x), data_od))
		fid_data	= ps_interp(fid_data[np.newaxis,:]).ravel()

	# Transform data to likelihood?
	trans_lnlike = False
	if trans_lnlike == True:
		data_tr = np.array(map(lambda x: S.gauss_lnlike(x,O.y,O.invcov),data_tr))[:,np.newaxis]
		data_cv = np.array(map(lambda x: S.gauss_lnlike(x,O.y,O.invcov),data_cv))[:,np.newaxis]
		data_od = np.array(map(lambda x: S.gauss_lnlike(x,O.y,O.invcov),data_od))[:,np.newaxis]
		fid_data = np.array(map(np.median,data_tr.T))

	# Initialize KLT
	E.klt(data_tr,fid_data=fid_data,normalize=norm_weights,w_norm=w_norm)

	# Plot Scree
	plot_scree = True
	if plot_scree == True:
		print_message('...plotting scree')
		print_time()

		fig = mp.figure(figsize=(6,4))
		ax = fig.add_subplot(111)
		ax.set_xlabel(r'eigenmode',fontsize=15)
		ax.set_ylabel(r'eigenvalue', fontsize=15)
		ax.grid(True)
		ax.set_xlim(-1,N_modes+1)
		ax.set_yscale('log')
		ax.scatter(np.arange(N_modes),E.eig_vals,color='k',s=40)

		fig.savefig('scree.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()

	# Make new yz_data matrix
	yz_data = []
	for i in range(z_len):
		kdat = np.array(np.around(O.xdata[i],3),str)
		zdat = np.array([z_array[i] for j in range(len(O.xdata[i]))],str)
		yz_data.append(np.array(zip(kdat,zdat)))
	yz_data = np.array(yz_data)
	yz_data_ext = O.row2mat(yz_data, row2mat=False)

	print_time()

	###### Set Parameter values
	print_message('...configuring emulator and sampler')
	print_time()
	print_mem()
	k = 80
	use_pca = True
	emode_variance_div = 1.0
	fast = True
	compute_klt = False
	save_chol = False
	LAYG = False

	# First Guess of GP Hyperparameters
	ell = np.array([[5.0 for i in range(N_params)] for i in range(N_modes)]) / np.linspace(1.0,5.0,N_modes).reshape(N_modes,1)
	ell_bounds = np.array([np.array([ell[i]*0.2,ell[i]*5.0]).T for i in range(N_modes)])
	ell_bounds = np.array([np.array([[0.1,100] for j in range(N_params)]) for i in range(N_modes)])

	alpha = 1e-6 * np.ones(N_modes)
	alpha_bounds = np.array([[1e-8,1e-2] for i in range(N_modes)])

	# Insert HP into GP
	kernels = map(lambda x: gp.kernels.RBF(*x[:2]) + gp.kernels.WhiteKernel(*x[2:]), zip(ell,ell_bounds,alpha,alpha_bounds))
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

	names       = ['kernel','copy_X_train','optimizer','n_restarts_optimizer','alpha']
	optimize    = 'fmin_l_bfgs_b'
	n_restarts  = np.array(np.linspace(10,3,E.N_modes),int)
	alpha		= 1e-8
	gp_kwargs_arr = np.array([dict(zip(names,[kernels[i],False,optimize,n_restarts[i],alpha])) for i in map(lambda x: x[0],E.modegroups)])

	### Load HyperParameters ###
	load_hype	= True
	load_obs	= True
	new_tr		= True
	if load_hype == True:
		hp_fname = 'forecast_hyperparams36.pkl'
#		hp_fname = 'hypersolve_1D.pkl'
		with open(hp_fname,'rb') as f:
			print("...loading previous hyperparameter file: "+hp_fname)
			input = pkl.Unpickler(f)
			hype_dic, global_dic, emulator_dic, obs_dic = input.load()

		# Find Different params
		print_diff = False
		if print_diff == True:
			master_d = dict(global_dic, **emulator_dic)
			for n in master_d.keys():
				if type(master_d[n]) in [float, int, bool, str]:
					if master_d[n] != dict(globals(), **E.__dict__)[n]:
						print("Old "+n+" : "+str(globals()[n])+", New "+n+" : "+str(master_d[n]))

		# Alter variables
		emulator_dic['N_modegroups'] = 30
		emulator_dic['N_modes'] = 30
		emulator_dic['w_norm'] = emulator_dic['w_norm'][:30]
		emulator_dic['eig_vecs'] = emulator_dic['eig_vecs'][:30]

		# Load Dictionaries
		globals().update(global_dic)
		globals().update(emulator_dic)
		E.update(emulator_dic)
		if load_obs == True:
			globals().update(obs_dic)
		else:
			E.yerrs = O.yerrs

		# insert kernels into gp_kwargs_arr
		gp_kwargs_arr = np.array([dict(zip(names,[hype_dic['fit_kernels'][i],False,None,0,alpha])) for i in range(E.N_modegroups)])

		# Use old hyperparameters and old Cholesky, but operate on new training set defined in draw_data()
		if new_tr == True:
			print_message('...using new training set')
			draw_data()
			if interp_z == True:
				data_tr = z_interp(data_tr)
				data_cv = z_interp(data_cv)
				data_od = np.array(map(lambda x: z_interp(x), data_od))
			if interp_ps == True:
				data_tr     = ps_interp(data_tr)
				data_cv     = ps_interp(data_cv)
				data_od     = np.array(map(lambda x: ps_interp(x), data_od))

			N_samples	= len(data_tr)
			fid_data	= E.fid_data
			fid_params	= E.fid_params

		Xsph		= np.dot(E.invL, (grid_tr-E.fid_params).T).T	
		E.create_tree(Xsph)
		E.update(dez.create(['grid_tr','data_tr','Xsph'], locals()))
		yz_data_ext = O.row2mat(yz_data, row2mat=False)

	# Create training kwargs
	kwargs_tr = {'use_pca':use_pca,'norotate':norotate,
				 'verbose':False,'invL':E.invL,'emode_variance_div':emode_variance_div,
				 'fast':fast,'compute_klt':compute_klt,'save_chol':save_chol,
				 'gp_kwargs_arr':gp_kwargs_arr}

	### Initialize Sampler Variables ###
	predict_kwargs = {'fast':fast,'use_Nmodes':None,'use_pca':use_pca,'LAYG':LAYG,'k':k,'kwargs_tr':kwargs_tr}

	param_width = np.array([grid_tr.T[i].max() - grid_tr.T[i].min() for i in range(N_params)])

	eps = -0.0001

	param_bounds = np.array([[grid_tr.T[i].min()+param_width[i]*eps,grid_tr.T[i].max()-param_width[i]*eps]\
								 for i in range(N_params)])

	param_hypervol = reduce(operator.mul,map(lambda x: x[1] - x[0], param_bounds))

	use_Nmodes = None
	add_model_cov = False
	ndim = N_params
	nwalkers = 300
	vectorize_predict = True

	sampler_init_kwargs = {'use_Nmodes':use_Nmodes,'param_bounds':param_bounds,'param_hypervol':param_hypervol,
							'nwalkers':nwalkers,'ndim':ndim,'N_params':ndim,'z_len':z_len}

	lnprob_kwargs = {'add_model_cov':add_model_cov,'predict_kwargs':predict_kwargs,'LAYG':LAYG,'k':k,
					 'vectorize':vectorize_predict}

	train_emu = True
	if train_emu == True:
		save_hype		= False
		kfold_regress	= False
		hypersolve_1D	= False
		SoD				= False
		hyper_filename	= None
		if kfold_regress == True:
			kfold_Nsamp = 50
			kfold_Nclus = 1
			E.sphere(E.grid_tr,save_chol=False,invL=E.invL)
			Rsph = np.array(map(la.norm, E.Xsph))
			kfold_cents = E.Xsph[np.random.choice(np.arange(0,N_samples),replace=False,size=kfold_Nclus)]
			kfold_cents = np.zeros(11)[np.newaxis,:]
			# Iterate over kfold clusters
			def multiproc_train(kfold_cent,Nsamp=kfold_Nsamp):
				E.sphere(E.grid_tr,save_chol=False,invL=E.invL)
				distances = np.array(map(la.norm,E.Xsph-kfold_cent))
				nearest = np.argsort(distances)[:Nsamp]
				kfold_data_tr = E.data_tr[nearest]
				kfold_grid_tr = E.grid_tr[nearest]
				E.train(kfold_data_tr,kfold_grid_tr,fid_data=E.fid_data,fid_params=E.fid_params,**kwargs_tr)			
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

		# 1D Approximation to hyperparameter regression
		if hypersolve_1D == True:
			print_message('...running hypersolve_1D')
			hyper_filename = 'hypersolve_1D.pkl'
			print_time()
			bounds = [0.1, 100]
			kernel = gp.kernels.RBF(length_scale=10.0, length_scale_bounds=(bounds[0], bounds[1]))
			ell = E.hypersolve_1D(grid_od, data_od, kernel=kernel, n_restarts=15, alpha=1e-5)
			kernels = np.array([gp.kernels.RBF(length_scale=ell[i], length_scale_bounds=(bounds[0],bounds[1])) for i in range(len(ell))])
			names       = ['kernel','copy_X_train','optimizer','n_restarts_optimizer','alpha']
			optimize    = None #'fmin_l_bfgs_b'
			n_restarts  = 0
			alpha       = 1e-5
			gp_kwargs_arr = np.array([dict(zip(names,[kernels[i],False,optimize,n_restarts,alpha])) for i in map(lambda x: x[0],E.modegroups)])
			kwargs_tr['gp_kwargs_arr'] = gp_kwargs_arr
			save_hype = True
			print_time()

		if SoD == True:
			# Sparse approximation to hyperparameter regression using Subset of Data approach
			print_message('...running Subset of Data regression')
			print_time()
			# Limit Grid
			within = np.where(np.array(map(la.norm, E.Xsph))<2.4)[0]
			data_tr = np.copy(data_tr)[within]
			grid_tr = np.copy(grid_tr)[within]
			E.data_tr, E.grid_tr = data_tr, grid_tr
			E.sphere(grid_tr, fid_params=E.fid_params, invL=E.invL)

		# Train!
		print_message('...training emulator')
		print_time()
		E.train(data_tr,grid_tr,fid_data=E.fid_data,fid_params=E.fid_params,**kwargs_tr)
		print_time()

		# Print out fitted hyperparameters
		if E.GP[0].optimizer is not None or save_hype == True:
			fit_kernels = []
			for i in range(len(E.GP)): fit_kernels.append(E.GP[i].kernel_)
			hype_dic		= ['fit_kernels']
			global_dic		= ['z_array','z_len','k_len','g_len','y_len','k_range','tr_name','yz_data','keep_meta',
								'kwargs_tr','predict_kwargs','lnprob_kwargs','sampler_init_kwargs','cov_est_name']
			obs_dic			= ['O', 'p_true', 'err_thresh', 'hi_k_cut']
			emulator_dic	= ['reg_meth','N_modes','N_modegroups','modegroups','emode_variance_div','N_samples','data_tr','grid_tr',
								'data_cv','grid_cv','fid_params','fid_data','invL','L','lognorm','eig_vecs','eig_vals','norotate',
								'scale_by_std','scale_by_obs_errs','norm_weights','w_norm','Dcov','D','Dstd','use_pca',
								'scale_by_davg_ov_yerr','obs_err_mult','Davg_ov_yerr','yerrs']

			i = 0
			while True:
				if hyper_filename is not None and i == 0: break
				hyper_filename = 'forecast_hyperparams'+str(i)+'.pkl'
				i += 1
				if os.path.isfile(hyper_filename) == False: break

			print_message('...saving '+hyper_filename)
			with open(hyper_filename,'wb') as f:
				output = pkl.Pickler(f)
				output.dump([dez.create(hype_dic, locals()), dez.create(global_dic, locals()),
								dez.create(emulator_dic, E.__dict__), dez.create(obs_dic, locals())])

			for i in range(len(gp_kwargs_arr)):
				gp_kwargs_arr[i]['optimizer']=None
			kwargs_tr['gp_kwargs_arr'] = gp_kwargs_arr

	print_mem()

	# Initialize Ensemble Sampler
	print_message('...initializing sampler')
	print_time()
	S = pycape.Samp(N_params, param_bounds, Emu=E, Obs=O)
	print_mem()

	date = time.ctime()[4:].split()[:-1]
	date = ''.join(date[:2])+'_'+re.sub(':','_',date[-1])

	#################
	### FUNCTIONS ###
	#################

	def calc_errs(recon, recon_err, sc_sig=3.5):
		# Get cross validated reconstruction error
		frac_err = (recon-data_cv)/data_cv
		log_frac_err = np.log(recon/data_cv)
		pred_frac_err = recon_err / data_cv
		std_err = astats.biweight_midvariance(frac_err.ravel())
		log_std_err = astats.biweight_midvariance(log_frac_err.ravel())
		frac_yerr = (recon-data_cv)/O.yerrs

		zarr_vec = O.row2mat(np.array([[z_array[i]]*len(O.xdata[i]) for i in range(z_len)]),row2mat=False)
		frac_err_vec = np.array(map(lambda x: astats.biweight_midvariance(x), frac_err.T))
		log_frac_err_vec = np.array(map(lambda x: astats.biweight_midvariance(x), log_frac_err.T))
		log_frac_err_sc_vec = np.array(map(lambda x: astats.biweight_midvariance(stats.sigmaclip(x,low=sc_sig,high=sc_sig)[0]), log_frac_err.T))
		exp_log_frac_err_vec = np.sqrt(np.exp(log_frac_err_vec**2)**2 - np.exp(log_frac_err_vec**2))
		exp_log_frac_err_sc_vec = np.sqrt(np.exp(log_frac_err_sc_vec**2)**2 - np.exp(log_frac_err_sc_vec**2))
		std_obserr_vec = np.array(map(lambda x: astats.biweight_midvariance(x), frac_yerr.T))
		frac_err_sc_vec = np.array(map(lambda x: astats.biweight_midvariance(stats.sigmaclip(x,low=sc_sig,high=sc_sig)[0]), frac_err.T))
		std_obserr_sc_vec = np.array(map(lambda x: astats.biweight_midvariance(stats.sigmaclip(x,low=sc_sig,high=sc_sig)[0]), frac_yerr.T))

		names = ['frac_err','log_frac_err','pred_frac_err','std_err','log_std_err','frac_yerr',
				'frac_err_vec','zarr_vec','std_obserr_vec','frac_err_sc_vec','std_obserr_sc_vec',
				'log_frac_err_vec','exp_log_frac_err_vec','log_frac_err_sc_vec','exp_log_frac_err_sc_vec']
		globals().update(dez.create(names,locals()))

	def plot_cross_valid(fname='cross_validate_ps.png'):
		fig = mp.figure(figsize=(16,12))

		# Cross Validation Error Histogram
		ax = fig.add_subplot(331)
		try:
			p1 = ax.hist(log_frac_err.ravel(),histtype='step',color='b',linewidth=1,bins=50,range=(-1.0,1.0),normed=True,alpha=0.75)
			p2 = ax.hist(pred_frac_err.ravel(),histtype='step',color='r',linewidth=1,bins=75,range=(-.01,1.5),normed=True,alpha=0.5)
		except UnboundLocalError:
			print 'UnboundLocalError on err or pred_frac_err'
		#ax.axvline(rms,color='r',alpha=0.5)
		#ax.axvline(-rms,color='r',alpha=0.5)
		#ax.hist(rms_obs_err,histtype='step',color='m',bins=50,range=(0,0.2),alpha=0.5,normed=True)
		ax.set_xlim(-1.0,1.0)
		ax.set_ylim(0,p1[0].max())
		ax.set_xlabel(r'Fractional Error',fontsize=15)
		ax.annotate(r'log std err = '+str(np.around(log_std_err*100,2))+'%',
			xy=(0.05,0.8),xycoords='axes fraction')

		# Cross Valid Error in k-z plane
		ax = fig.add_subplot(332)
		im = ax.scatter(O.x_ext,zarr_vec,c=exp_log_frac_err_sc_vec*100,marker='o',s=35,edgecolor='',alpha=0.75,cmap='nipy_spectral_r',vmin=0,vmax=20)
		ax.set_xscale('log')
		ax.set_xlim(1e-1,3)
		ax.set_ylim(4,25)
		ax.set_xlabel(r'$k\ cMpc^{-1}',fontsize=17)
		ax.set_ylabel(r'$z$',fontsize=17)
		fig.colorbar(im,label=r'Avg. Percent Error')

		# Plot CV error wrt HERA precision
		ax = fig.add_subplot(333)
		im = ax.scatter(O.x_ext,zarr_vec,c=std_obserr_sc_vec,marker='o',s=35,edgecolor='',alpha=0.75,cmap='nipy_spectral_r',vmin=0.0,vmax=2)
		ax.set_xscale('log')
		ax.set_xlim(1e-1,3)
		ax.set_ylim(4,25)
		ax.set_xlabel(r'$k\ cMpc^{-1}',fontsize=17)
		ax.set_ylabel(r'$z$',fontsize=17)
		fig.colorbar(im,label=r'Frac. Error Over HERA Error')

		# Plot eigenmodes
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
			ax.set_ylim(4,25)
			ax.set_ylabel(r'$z$',fontsize=12)
			fig.colorbar(im)

		fig.savefig(fname,dpi=200,bbox_inches='tight')
		mp.close()
		print_time()

	def plot_eigvecs(fname='eig_vecs.png'):
		fig = mp.figure(figsize=(10,5))
		for i in range(6):
			ax = fig.add_subplot(2,3,i+1)
			# Get Eigenvector and kz data
			eig_vec = E.eig_vecs[i]
			# yz_eigvector plot
			cmap = mp.cm.get_cmap('coolwarm',41)
			vavg = (eig_vec.max()+np.abs(eig_vec.min()))/2.0
			im = ax.scatter(O.x_ext,zarr_vec,c=eig_vec,marker='o',s=35,edgecolors='',alpha=0.9,cmap=cmap,vmin=-vavg,vmax=vavg)
			ax.set_xlabel(r'$k$ (Mpc$^{-1}$)',fontsize=12)
			ax.set_xscale('log')
			ax.set_xlim(1e-1,3)
			ax.set_ylim(4,25)
			ax.set_ylabel(r'$z$',fontsize=12)
			fig.colorbar(im)
		fig.savefig(fname,dpi=150,bbox_inches='tight')
		mp.close()

	def cv_plots(fname='cv_plots.png', frac_err_vec=None, frac_yerr_vec=None, cbmax1=1.0,
					inset=True, in_range=1.0, kbin=342):
		if frac_err_vec is None:
			frac_err_vec = log_frac_err_sc_vec
		if frac_yerr_vec is None:
			frac_yerr_vec = std_obserr_sc_vec

		fig = mp.figure(figsize=(4,8))
		fig.subplots_adjust(hspace=0.1)
		ax = fig.add_subplot(211)
		cmap = mp.cm.spectral_r
		cmaplist = [cmap(i) for i in range(cmap.N)]
		cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
		bounds = np.linspace(0,cbmax1,21)
		norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
		im = ax.scatter(O.x_ext,zarr_vec,c=frac_err_vec,marker='o',s=30,edgecolor='',alpha=0.75, cmap=cmap, norm=norm)
		ax.set_xscale('log')
		ax.set_xlim(0.08,2.5)
		ax.set_ylim(4,25)
		ax.set_ylabel(r'$z$',fontsize=20)
		cbax = fig.add_axes([0.92, 0.52, 0.03, 0.38])
		cb = matplotlib.colorbar.ColorbarBase(cbax, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')
		cb.set_label(r'$\sigma_{\mathrm{CV}}$', fontsize=16, labelpad=15, rotation=0)
		tick_locator = ticker.MaxNLocator(nbins=11)
		cb.locator = tick_locator
		cb.update_ticks()
		_ = [tl.set_size(10) for tl in cb.ax.get_yticklabels()]

		ax = fig.add_subplot(212)
		cmap = mp.cm.spectral_r
		cmaplist = [cmap(i) for i in range(cmap.N)]
		cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
		bounds = np.linspace(0,2,21)
		norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
		im = ax.scatter(O.x_ext,zarr_vec,c=frac_yerr_vec,marker='o',s=30,edgecolor='',alpha=0.75, cmap=cmap, norm=norm)
		ax.set_xscale('log')
		ax.set_xlim(0.08,2.5)
		ax.set_ylim(4,25)
		ax.set_xlabel(r'$k$ (h Mpc$^{-1}$)',fontsize=16)
		ax.set_ylabel(r'$z$',fontsize=20)
		cbax = fig.add_axes([0.92, 0.1, 0.03, 0.38])
		cb = matplotlib.colorbar.ColorbarBase(cbax, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.1f')
		cb.set_label(r'$\sigma_{\mathrm{S}}$', fontsize=16, labelpad=12, rotation=0)
		tick_locator = ticker.MaxNLocator(nbins=11)
		cb.locator = tick_locator
		cb.update_ticks()
		_ = [tl.set_size(10) for tl in cb.ax.get_yticklabels()]

		if inset == True:
			ax2 = fig.add_axes([0.65, 0.3, 0.2, 0.15], zorder=2)
			ax.plot([float(yz_data_ext[kbin][0])-.005,0.90],[float(yz_data_ext[kbin][1])+.2,22],linewidth=1,color='k',alpha=0.75)
			ax.plot([float(yz_data_ext[kbin][0])+.005,0.90],[float(yz_data_ext[kbin][1])-.2,16],linewidth=1,color='k',alpha=0.75)
			p = ax2.hist(frac_yerr.T[kbin], bins=35, histtype='step', color='b', linewidth=1.5, alpha=0.5, range=(-in_range,in_range), normed=True)
			ax2.axvline(-std_obserr_sc_vec[kbin], c='red', alpha=0.5, ymax=0.3)
			ax2.axvline(std_obserr_sc_vec[kbin], c='red', alpha=0.5, ymax=0.3)
			ax2.set_yticklabels([])
			ax2.set_yticks([])
			ax2.set_xticks(np.arange(-2,2.01,1))
			ax2.set_xlim(-in_range,in_range)
			ax2.set_ylim(0,p[0].max()*1.1)
			
		fig.savefig(fname, dpi=200, bbox_inches='tight')
		mp.close()

	def plot_error_dist(fname='cv_err_dists.png', frac_err=None, std_vec=None, xlim=(-1,1)):

		if frac_err is None:
			frac_err = log_frac_err
		if std_vec is None:
			std_vec = exp_log_frac_err_sc_vec

		fig = mp.figure(figsize=(12,12))
		fig.subplots_adjust(hspace=0.1,wspace=0.1)

		i = 0
		plot_kbins = np.arange(len(yz_data_ext))
		plot_kbins = plot_kbins[np.where(np.array(yz_data_ext.T[0],float)<0.5)]
		plot_kbins = plot_kbins[np.array(np.linspace(0,len(plot_kbins)-1,64),int)]
		for j in plot_kbins:
			ax = fig.add_subplot(8,8,i+1)
			p=ax.hist(frac_err.T[j], bins=30, range=xlim, histtype='step', color='k', linewidth=1.5, alpha=0.5, normed=True)
			ax.axvline(std_vec[j], color='r', alpha=0.5, ymax=0.3)
			ax.axvline(-std_vec[j], color='r', alpha=0.5, ymax=0.3)
			ax.set_xlim(xlim)
			ax.set_ylim(0,p[0].max()*1.1)
			ax.set_yticklabels([])
			ax.annotate(r'$z='+str(yz_data_ext[j][1])+'$\n$k='+str(yz_data_ext[j][0])+'$', xy=(0.05,0.75), xycoords='axes fraction', fontsize=6)
			if i < 56 :
				ax.set_xticklabels([])
			else:
				[tl.set_visible(False) for tl in ax.get_xticklabels()[::2]]
			i += 1	

		#ax = fig.add_axes([.4,0.9,.2,.01])
		#ax.axis('off')
		#ax.set_title('log frac data err', fontsize=16)
		fig.savefig(fname, dpi=200, bbox_inches='tight')
		mp.close()

	def plot_data_tr_dist(fname='data_tr_dists.png'):

		fig = mp.figure(figsize=(12,12))
		fig.subplots_adjust(hspace=0.1,wspace=0.1)

		Dloc = np.array(map(astats.biweight_location, E.D.T))
		Dstd = np.array(map(astats.biweight_midvariance, E.D.T))

		i = 0
		for j in np.array(np.linspace(0,len(yz_data_ext)-1,64),int):
			ax = fig.add_subplot(8,8,i+1)
			ax.axvspan(Dloc[j]-Dstd[j], Dloc[j]+Dstd[j], color='red', alpha=0.1)
			ax.hist(E.D.T[j], bins=40, range=(-3,3), histtype='step', color='k', linewidth=1.5, alpha=0.5, normed=True)
			ax.set_xlim(-3,3)
			ax.set_yticklabels([])
			ax.axvline(Dloc[j], color='green', alpha=0.3, linewidth=1.0)
			ax.axvline(0.0, color='dodgerblue', alpha=0.5, linewidth=1.0)
			ax.annotate(r'$z='+str(yz_data_ext[j][1])+'$\n$k='+str(yz_data_ext[j][0])+'$', xy=(0.05,0.75), xycoords='axes fraction', fontsize=6)
			if i < 56 :
				ax.set_xticklabels([])
			else:
				[tl.set_visible(False) for tl in ax.get_xticklabels()[::2]]
				[tl.set_size(8) for tl in ax.get_xticklabels()]
			i += 1

		fig.savefig(fname, dpi=200, bbox_inches='tight')
		mp.close()


	#####################
	### END FUNCTIONS ###
	#####################

	t = True
	f = False

	# Plotting and Cross Validating
	plot_eigenmodes		= f
	cross_validate_ps	= t
	cross_validate_like	= t
	plot_weight_pred	= f
	plot_ps_pred		= f
	plot_ps_recon_frac	= f
	plot_ps_recon		= f

	# Sampling
	make_fisher			= f
	add_priors			= t
	time_sampler		= f
	drive_sampler		= t
	parallel_temp		= f
	save_chains			= t
	edit_log			= t

	# Plotting
	trace_plots			= t
	autocorr_plots		= t
	tri_plots			= t
	plot_boxplots		= f
	plot_marghist		= t
	plot_recplot		= t
	plot_map_pspec		= t
	ps_var_movie		= f

	if plot_eigenmodes == True:
		# Plot first few eigenmodes and scree plot
		fig = mp.figure(figsize=(4,8))
		fig.subplots_adjust(hspace=0.5, wspace=0.1)

		# Scree
		ax = fig.add_subplot(4,1,1)
		ax.grid(True)
		ax.scatter(np.arange(1,30.5,1), E.eig_vals[:30], c='k', marker='o', s=30)
		ax.set_yscale('log')
		ax.set_xlim(0,31)
		ax.set_xlabel(r'eigenmode number', fontsize=15)
		ax.set_ylabel(r'eigenvalue', fontsize=15)

		# z = 8, 12, 16
		for i,z in enumerate([5,9,13]):
			ax = fig.add_subplot(4,1,i+2)
			pc1 = O.row2mat(E.eig_vecs[0])[z]
			pc2 = O.row2mat(E.eig_vecs[1])[z]
			pc3 = O.row2mat(E.eig_vecs[2])[z]
			ax.plot(O.track(['ps'])[z], pc1/np.abs(pc1).max(), linestyle='-', color='k', linewidth=1)
			ax.plot(O.track(['ps'])[z], pc2/np.abs(pc2).max(), linestyle='--', color='k', linewidth=1)
			ax.plot(O.track(['ps'])[z], pc3/np.abs(pc3).max(), linestyle='-.', color='k', linewidth=1)
			ax.set_xscale('log')
			ax.set_xlim(0.1,2.1)
			ax.set_ylim(-1.1,1.1)
			if i == 1: ax.set_ylabel(r'normalized $\ln\Delta_{21}^{2}$ eigenvector', fontsize=15)
			ax.set_xlabel(r'$k$ (h Mpc$^{-1}$)', fontsize=15)
			ax.annotate(r'$z='+yz_data[z][0][1]+'$',xy=(0.75,0.75),xycoords='axes fraction',fontsize=13,bbox=dict(fc="0.95"))

		fig.savefig('eigenmodes.png',dpi=150,bbox_inches='tight')
		mp.close()

	use_tr_for_cv = False
	if use_tr_for_cv == True:
		within = np.where(np.array(map(la.norm,E.Xsph))<2.0)[0]
		within = np.where(np.array(map(lambda x: np.abs(x).max(), E.Xsph))<1.375)[0]
		rando = np.random.choice(np.arange(len(within)), replace=False, size=1000)
		data_cv = np.copy(data_tr)[within[rando]]
		grid_cv = np.copy(grid_tr)[within[rando]]

	limit_cv_range = False
	if limit_cv_range == True:
		grid_cv_sph = np.dot(E.invL, (grid_cv-E.fid_params).T).T
		within = np.where(np.array(map(la.norm, grid_cv_sph))<4.5)[0]
		#within = np.where(np.array(map(lambda x: np.abs(x).max(), grid_cv_sph))<2.0)[0]
		data_cv = data_cv[within]
		grid_cv = grid_cv[within]

	kfold_cv = False
	calibrate = False
	add_lnlike_cov = True
	if cross_validate_ps == True:
		print_message('...cross validating power spectra')
		if kfold_cv == True:
			limit_range = True
			Nclus = 4
			Nsamp = 400
			if limit_range == True:
				within = np.where(np.array(map(la.norm,E.Xsph)) < 2.75)[0]
				Nclus_avail = len(within) / Nsamp
				rando = np.array([[False]*len(data_tr) for i in range(Nclus_avail)])
				rand_samp = within[np.random.choice(np.arange(len(within)), replace=False, size=Nsamp*Nclus_avail)].reshape(Nclus_avail, Nsamp)[:Nclus, :]
			else:
				Nclus_avail = len(data_tr) / Nsamp
				rando = np.array([[False]*len(data_tr) for i in range(Nclus_avail)])
				rand_samp = np.random.choice(np.arange(len(data_tr)), replace=False, size=Nclus_avail*Nsamp).reshape(Nclus_avail,Nsamp)[:Nclus, :]

			rando = rando[:Nclus]
			for i in range(len(rando)):
				rando[i][rand_samp[i]] = True

			recon_cv, recon_err_cv, recon_grid, recon_data, rando = E.kfold_cv(grid_tr, data_tr,
									predict_kwargs=predict_kwargs,kwargs_tr=kwargs_tr,kfold_Nclus=Nclus,kfold_Nsamp=Nsamp,rando=rando)
			recon = recon_cv
			recon_err = recon_err_cv
			data_cv = recon_data
			grid_cv = recon_grid
		else:
			E.cross_validate(grid_cv, data_cv, predict_kwargs=predict_kwargs, LAYG=LAYG, vectorize=vectorize_predict, use_tree=True)
			recon = E.recon_cv
			recon_err = E.recon_err_cv
			weights = E.weights_cv
			weights_err = E.weights_err_cv
			weights_true = E.weights_true_cv

		if calibrate == True:
			resid_loc = np.array(map(astats.biweight_location, recon.T/data_cv.T))**(-1)
			E.recon_calib = resid_loc
			recon *= E.recon_calib
			resid_std = np.array(map(astats.biweight_midvariance, recon.T - data_cv.T))
			avg_recon_err = np.array(map(astats.biweight_location, recon_err.T))
			err_calib = resid_std / avg_recon_err
			E.recon_err_calib = err_calib
			recon_err *= E.recon_err_calib

		if add_lnlike_cov == True:
			print_message('...adding pspec cross validated errors to lnlike covariance as weights',type=0)
			X = recon.T-data_cv.T
			ps_err_cov = cov_est(X)
			O.cov += np.abs(np.eye(len(X))*np.array(map(astats.biweight_location,X)))**2 + ps_err_cov

			fig = mp.figure(figsize=(6,4))
			fig.subplots_adjust(wspace=0.05,hspace=0.05)

			cmap = matplotlib.cm.YlGnBu_r
			cmap.set_bad('grey',0.1)
			hera_logcov = np.log10(np.abs(O.cov))
			masked_cov = np.ma.array(hera_logcov, mask=hera_logcov==-np.inf)
			ax1 = fig.add_subplot(121)
			im1 = ax1.matshow(masked_cov, origin='lower', cmap=cmap, vmin=-5, vmax=4)
			ax1.set_xticklabels([])
			ax1.set_yticklabels([])
			ax1.set_title(r'Telescope Sensitivity Covariance', fontsize=10)

			ax2 = fig.add_subplot(122)
			im2 = ax2.matshow(hera_logcov,origin='lower',cmap=cmap,vmin=-5,vmax=4)
			ax2.set_xticklabels([])
			ax2.set_yticklabels([])
			ax2.set_title(r'CV Residual Covariance', fontsize=10)

			cbaxes = fig.add_axes([0.1, 0.1, 0.8, 0.05]) 
			fig.colorbar(im1, cax=cbaxes, label=r'$\log_{10}\langle| r\cdot r^{T}|\rangle$', orientation='horizontal')
			fig.savefig('emu_cov_'+date+'.png',dpi=150,bbox_inches='tight')
			mp.close()

		calc_errs(recon, recon_err, sc_sig=3.0)
		plot_cross_valid(fname='cross_validate_ps.png')
		cv_plots(fname='cv_plots.png', frac_err_vec=log_frac_err_sc_vec, frac_yerr_vec=std_obserr_sc_vec, cbmax1=0.5)
		plot_error_dist(fname='cv_err_dists.png', frac_err=log_frac_err, std_vec=exp_log_frac_err_sc_vec, xlim=(-2.5,2.5))
		plot_data_tr_dist(fname='data_tr_dists.png')

	if cross_validate_like == True:
		print_message('...cross validating likelihoods')
		print_time()

		if kfold_cv == True:
			e_like, t_like, data_cv, grid_cv, rando = S.kfold_cross_validate(grid_tr, data_tr, predict_kwargs=predict_kwargs,
					kwargs_tr=kwargs_tr, lnlike_kwargs=lnprob_kwargs, kfold_Nclus=Nclus, kfold_Nsamp=Nsamp, rando=rando)
		else:
			e_like, t_like = S.cross_validate(grid_cv,data_cv,lnlike_kwargs=lnprob_kwargs)#,also_record=['lnlike_emu_err'])


		fig = mp.figure(figsize=(5,5))
		ax = fig.add_subplot(111)
		frac_err = (e_like-t_like)/t_like
		try: patches = ax.hist(frac_err,bins=100,histtype='step',range=(-5,5),normed=True,color='b')
		except UnboundLocalError: pass
		ax.set_xlim(-5,5)
		ax.set_xlabel('likelihood fractional error',fontsize=14)
		lnlike_sig = np.around(astats.biweight_midvariance(frac_err)*100,2)
		ax.annotate(r'$\sigma = '+str(lnlike_sig)+'\%$',xy=(0.2,0.8),xycoords='axes fraction',fontsize=18)
		fig.savefig('cross_validate_like.png',dpi=100,bbox_inches='tight')
		mp.close()
		print_time()

	if kfold_cv == True:
		print_message("...retraining emulator with full dataset")
		E.train(data_tr,grid_tr,fid_data=E.fid_data,fid_params=E.fid_params,**kwargs_tr)

	print_message('...making plots')
	print_time()

	# Plot Training Set
	plot_tr = True
	if plot_tr == True:
		print_message('...plotting ts')
		print_time()

		lims = [[None,None] for i in range(11)]
		pbound = np.array([grid_tr.T[i].max()-grid_tr.T[i].min() for i in range(11)])
		lims = [[grid_tr.T[i].min()-pbound[i]*0.05, grid_tr.T[i].max()+pbound[i]*0.05] for i in range(11)]

		fig = mp.figure(figsize=(15,8))
		fig.subplots_adjust(wspace=0.3)
		j = 0
		for i in range(6):
			ax = fig.add_subplot(2,3,i+1)
			ax.plot(grid_tr.T[j],grid_tr.T[j+1],'k,',alpha=0.75)
			ax.plot(p_true[j], p_true[j+1], color='m', marker='*', markersize=15)
			ax.plot(grid_cv.T[j], grid_cv.T[j+1], 'r.', markersize=5, alpha=0.5)
			#cax = ax.scatter(grid_cv.T[j],grid_cv.T[j+1],s=30,c=lnlike,cmap='spectral_r',alpha=0.75,vmin=-1000,vmax=-500)
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

	# Cross Validate cross set
	if plot_weight_pred == True or plot_ps_pred == True:
		recon_cross = []
		recon_err_cross = []
		weight_cross = []
		weight_err_cross = []
		weight_true_cross = []
		for i in range(N_params):
			E.cross_validate(grid_od[i], data_od[i], predict_kwargs=predict_kwargs, LAYG=LAYG, vectorize=vectorize_predict)
			recon_cross.append(E.recon_cv)
			recon_err_cross.append(E.recon_err_cv)
			weight_cross.append(E.weights_cv)
			weight_err_cross.append(E.weights_err_cv)
			weight_true_cross.append(E.weights_true_cv)

		recon_cross = np.array(recon_cross)
		recon_err_cross = np.array(recon_err_cross)
		weight_cross = np.array(weight_cross)
		weight_err_cross = np.array(weight_err_cross)
		weight_true_cross = np.array(weight_true_cross)

	# Plot Eigenmode Weight Prediction
	if plot_weight_pred == True:
		plot_modes = [0,1,10,20,30,35,38,39]
		plot_params = [0,2,5,7,10]

		gs = gridspec.GridSpec(4,4)
		gs1 = gs[:3,:]
		gs2 = gs[3,:]

		for p in plot_params:
			for plot_mode in plot_modes:
				# Plot regression of weights
				fig=mp.figure(figsize=(10,10))
				fig.subplots_adjust(hspace=0.1)

				#sel = np.array(reduce(operator.mul,np.array([grid_cv.T[i]==p_true[i] if i != p else np.ones(len(grid_cv.T[i])) for i in range(N_params)])),bool)
				grid_x = grid_od[p].T[p]
				pred_weight = weight_cross[p].T[plot_mode]
				true_weight = weight_true_cross[p].T[plot_mode]
				pred_weight_err = weight_err_cross[p].T[plot_mode]

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
		plot_kbins = [10,20,23,64,92,108,148]
		plot_params = [0,5,8]

		gs = gridspec.GridSpec(4,4)
		gs1 = gs[:3,:]
		gs2 = gs[3,:]

		p_cent = np.array(map(np.median, grid_cv.T))

		for p in plot_params:
			for kbin in plot_kbins:
				# Plot regression of weights
				fig=mp.figure(figsize=(8,8))
				fig.subplots_adjust(hspace=0.1)

				#sel = np.array(reduce(operator.mul,np.array([grid_cv.T[i]==p_cent[i] if i != p else np.ones(len(grid_cv.T[i])) for i in range(N_params)])),bool)
				grid_x = grid_od[p].T[p]
				pred_ps = recon_cross[p].T[kbin]
				true_ps = data_od[p].T[kbin]
				pred_ps_err = recon_err_cross[p].T[kbin]
				yz = O.row2mat(yz_data,row2mat=False)[kbin]

				ax1 = fig.add_subplot(gs1)
				ax1.grid(True)
				a0 = ax1.fill_between(grid_x,pred_ps+pred_ps_err,pred_ps-pred_ps_err,color='b',alpha=0.2)
				a1, = ax1.plot(grid_x,true_ps,'r.',markersize=14,alpha=0.3)
				a2, = ax1.plot(grid_x,pred_ps,'k',linewidth=2.5)
				#ax1.set_yscale('log')
				ax1.set_ylabel(r'$\Delta^{2}$',fontsize=18)
				ax1.legend([a1,a2],[r'True PS',r'Prediction'])
				ylim = None,None
				ax1.set_ylim(ylim)
				ax1.tick_params(axis='x',which='both',bottom='off',labelbottom='off')
				ax1.annotate(r'$k = '+str(np.round(float(yz[0]),2))+'\ \mathrm{h\ Mpc}^{-1}$\n$z = '+yz[1]+'$',xy=(0.05,0.85),xycoords='axes fraction',fontsize=17)

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
	if plot_ps_recon_frac == True:

		# Pick redshifts
		z_arr = np.arange(44)[0:6][::1]

		# Calcualte error
		ps_frac_err = recon / data_cv

		# Iterate through redshifts
		for z in z_arr:
			# Plot
			fig = mp.figure(figsize=(5,5))
			ax = fig.add_subplot(111)
			ax.grid(True,which='both')

			# Iterate through cv
			pspec_ratio = []
			for i in range(len(grid_cv)):
				pspec_ratio.append(O.track(['ps'],arr=O.row2mat(recon[i]))[z]/O.track(['ps'],arr=O.row2mat(data_cv[i]))[z])
				ax.plot(O.track(['ps'],arr=O.xdata)[z],pspec_ratio[-1],color='r',alpha=0.2,linewidth=1.5,zorder=1)

			pspec_ratio = np.array(pspec_ratio)
			loc = np.array(map(astats.biweight_location,pspec_ratio.T))
			stdev = np.array(map(astats.biweight_midvariance,pspec_ratio.T))

			ax.fill_between(O.track(['ps'],arr=O.xdata)[z],loc-stdev,loc+stdev,color='grey',alpha=0.5,zorder=2)
			ax.plot(O.track(['ps'],arr=O.xdata)[z],loc-stdev,color='k',alpha=0.75,linewidth=2,zorder=2)
			ax.plot(O.track(['ps'],arr=O.xdata)[z],loc+stdev,color='k',alpha=0.75,linewidth=2,zorder=2)
			ax.plot(O.track(['ps'],arr=O.xdata)[z],loc,color='k',alpha=0.75,linewidth=1,zorder=3)

			ax.annotate(r'$z='+str(z_array[z])+'$',xy=(0.1,0.8),xycoords='axes fraction',fontsize=18,bbox=dict(fc="0.95"))

			ax.set_xlim(1e-1,1)
			ax.set_ylim(0.5,1.5)
			ax.set_xscale('log')
			ax.set_xlabel(r'$k\ (\mathrm{h\ Mpc}^{-1})$',fontsize=18)
			ax.set_ylabel(r'$\widehat{\Delta^{2}}/\Delta^{2}$',fontsize=18)

			fig.savefig('ps_frac_recon_z'+str(z_array[z])+'.png',dpi=200,bbox_inches='tight')
			mp.close()

	# Plot PS Reconstruction
	if plot_ps_recon == True:

		# Pick Redshifts
		z_arr = np.arange(44)[5:27][::4]

		recon_cut = np.array(map(lambda x: O.track(['ps'], arr=O.row2mat(x))[z_arr], recon))

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
			fid = O.track(['ps'], arr=O.row2mat(E.fid_data))[z_arr[i]]
			ax.fill_between(O.track(['ps'])[z_arr[i]], fid + stand_dev, fid-stand_dev, color='b', alpha=0.25)
			ax.plot(O.track(['ps'])[z_arr[i]], fid, color='k', linewidth=2, alpha=0.8)
			ax.set_ylim(1e1,1e4)
			ax.annotate(r'$z = '+str(z_array[z_arr[i]])+'$', xy=(0.2,0.7), xycoords='axes fraction', fontsize=12)

			fig.savefig('ps_recon_z'+str(z_array[z_arr[i]])+'.png', dpi=200, bbox_inches='tight')
			mp.close()


	# Make Fisher Info
	if make_fisher == True:
		pspecs = []
		epsilon = 0.001
		for i in range(11):
			ptemp = []
			for j in range(2):
				p = np.copy(p_true)
				if j == 0:
					p[i] += -epsilon*p[i]
				else:
					p[i] += epsilon*p[i]
				E.predict(p,**predict_kwargs)
				ptemp.append(O.track(['ps'],arr=O.row2mat(E.recon[0]),mat=False))
			pspecs.append(ptemp)
		pspecs = np.array(pspecs)

		partial_PS = pspecs[:,1,:] - pspecs[:,0,:]
		partial_TH = p_true*epsilon*2
		partial_TH_norm = (np.dot(E.invL, (p_true*1.01 - fid_params).T) - np.dot(E.invL, (p_true - fid_params).T))*2
		ps_errs = O.track(['ps'],arr=O.row2mat(O.yerrs),mat=False)

		w = (partial_PS.T / partial_TH).T / ps_errs
		w_norm = (partial_PS.T / partial_TH_norm).T / ps_errs
		F = np.dot(w,w.T)

	plot_fisher_derivs = False
	if plot_fisher_derivs == True:
		# Get max w_norm at each bandpower
		w_norm_max = np.array(map(np.max, np.abs(w_norm).T))

		# Plot Pspec derivatives
		fig = mp.figure(figsize=(5,4))
		ax = fig.add_subplot(111)
		im = ax.scatter(O.x_ext,zarr_vec,c=w_norm_max,marker='o',s=35,edgecolor='',alpha=0.75,cmap='nipy_spectral_r',vmin=0,vmax=w_norm_max.max())
		ax.set_xscale('log')
		ax.set_xlim(1e-1,3.3)
		ax.set_ylim(4,23)
		ax.set_xlabel(r'$k\ h$ Mpc$^{-1}$',fontsize=17)
		ax.set_ylabel(r'$z$',fontsize=17)
		cbar = fig.colorbar(im)
		cbar.set_label(label=r'$w_{max}$',size=20) 

		fig.savefig('w_plot.png',dpi=150,bbox_inches='tight')
		mp.close()

	print_time()
	# Initialize Ensemble Sampler
	print_message('...initializing ensemble sampler')

	print_message('...date is '+date)
	sampler_kwargs = {'vectorize':False}
	ntemps = 3

	#pool = pathos.multiprocessing.Pool(5) #emcee.utils.MPIPool()
	#sampler_kwargs.update({'pool':pool})
	S.emcee_init(nwalkers, ndim, S.lnprob, lnprob_kwargs=lnprob_kwargs, sampler_kwargs=sampler_kwargs, PT=parallel_temp, ntemps=ntemps)

	# Add priors (other than flat priors)
	if add_priors == True:
		print_message('...adding non-flat priors',type=0)
		#planck_cov = np.loadtxt('base_TTTEEE_lowl_plik.covmat')[[0,1,5]].T[[0,1,5]].T
		select_arr = np.array([6,7,0,1,5])#,3])
		planck_cov = np.loadtxt('new_planck_cov.tab')[select_arr[:,None],select_arr]
		std_multiplier = 1
		#prior_cent = np.array(map(lambda x: common_priors.cmb_priors1[x], ['sigma8','H0','ombh2','omch2','ns']))
		#prior_cent[1] /= 100.0
		prior_cent = np.copy(p_true[:5])
		prior_cent = np.concatenate([prior_cent, map(astats.biweight_location, E.grid_tr.T[5:])])

		# Add non-correlated Gaussian Priors
		prior_params = []
		prior_indices = []
		priors = map(lambda x: common_priors.cmb_priors1[x+'_err'] * std_multiplier, prior_params)
		for i in range(len(priors)):
			S.lnprior_funcs[prior_indices[i]] = S.create_gauss_lnprior(prior_cent[prior_indices[i]],priors[i],\
						index=prior_indices[i],return_func=True)
			
		# Add correlated Gaussian Priors
		prior_params = ['sigma8','hlittle','ombh2','omch2','ns']
		prior_indices = [0,1,2,3,4]
		Nindices = len(prior_indices)
		prior_cov = np.zeros((N_params,N_params))
		for i in range(N_params):
			if i >= Nindices:
				prior_cov[i,i] = 1e20
			else:
				for j in range(Nindices):
					prior_cov[prior_indices[i],prior_indices[j]] = planck_cov[i,j] * std_multiplier**2
		prior_prec = la.inv(prior_cov)            
		for i in range(len(prior_params)):
			S.lnprior_funcs[prior_indices[i]] = S.create_covarying_gauss_lnprior(prior_cent,prior_prec,\
					index=prior_indices[i],return_func=True)

	# Initialize Walker positions
	if parallel_temp == True:
		pos = np.array([np.copy(grid_cv[np.random.choice(np.arange(len(grid_cv)), replace=False, size=nwalkers)]) for i in range(ntemps)])
	else:
		#pos = np.array(map(lambda x: x + x*stats.norm.rvs(0,0.05,nwalkers),p_true)).T
		#pos = np.copy(grid_tr[np.random.choice(np.arange(len(grid_tr)), replace=False, size=nwalkers)])
		pos = stats.multivariate_normal.rvs(mean=prior_cent[:5], cov=prior_cov[:5,:5], size=nwalkers)
		pos = np.concatenate([pos.T,\
			np.array([stats.uniform.rvs(loc=prior_cent[i]-param_width[i]/2.5,scale=param_width[i]/1.25,size=nwalkers) for i in range(5,11)])]).T

	if time_sampler == True:
		print_message('...timing sampler')
		ipython.magic("timeit -r 3 S.samp_drive(pos,step_num=1,burn_num=0)")

	if drive_sampler == True:
		print_message('...driving sampler',type=1)
		print_time()
		# Drive Sampler
		burn_num	= 0
		step_num	= 2000
		message		= "calibrate=True"

		print_message('...driving with burn_num='+str(burn_num)+', step_num='+str(step_num),type=0)
		S.samp_drive(pos,step_num=step_num,burn_num=burn_num)
		chain = S.sampler.chain
		samples = chain[:, 0:, :].reshape((-1, S.ndim))
		print("Mean acceptance fraction: {0:.3f}".format(np.mean(S.sampler.acceptance_fraction)))

		if save_chains == True:
			with open('samp_chains_'+date+'.pkl','wb') as f:
				output = pkl.Pickler(f)
				output.dump({'chain':S.sampler.chain,'burn_num':burn_num,'step_num':step_num,\
							'acceptance_frac':np.mean(S.sampler.acceptance_fraction),'ndim':S.ndim,
							'p_true':p_true, 'prior_cov':prior_cov, 'prior_cent':prior_cent})

		if edit_log == True:
			with open('mcmc_log.txt','a') as f:
				f.write('\n'+date+':\n'+'-'*17+'\n**'+message+'\n')



		print_time()

	load_chains = False
	if load_chains == True:
		fname = 'samp_chains_1planck.pkl'
		f = open(fname,'rb')
		input = pkl.Unpickler(f)
		chain_d = input.load()
		f.close()
		chain = chain_d['chain']
		thin = 50
		samples = chain[:,500::thin,:].reshape((-1,chain_d['ndim']))

	if trace_plots == True:
		print_message('...plotting trace plots')
		print_time()

		# Trace Plot 1
		fig = mp.figure(figsize=(16,8))
		fig.subplots_adjust(wspace=0.4,hspace=0.2)

		for i in range(N_params):
			ax = fig.add_subplot(3,4,i+1)
			ax.set_ylabel(p_latex[i],fontsize=20)
			mp.tick_params(which='both',right='off',top='off')
			for j in range(len(chain))[::nwalkers/200]:
				ax.plot(chain[j,:,i],color='k',alpha=0.1)
				ax.set_ylim(param_bounds[i])
			ax.axhline(p_true[i],color='r',alpha=0.5,linewidth=3)

		fig.savefig('trace_plots_'+date+'.png',dpi=100,bbox_inches='tight')
		mp.close()

		# Trace Plot 2
		fig = mp.figure(figsize=(6,6))
		fig.subplots_adjust(hspace=0.15)
		plot_params = [0,5,8]

		for i in range(len(plot_params)):
			p = plot_params[i]
			ax = fig.add_subplot(3,1,i+1)
			ax.set_ylabel(p_latex[p], fontsize=15)
			mp.tick_params(which='both', right='off')
			if i != 2:
				ax.set_xticklabels([])
			else:
				ax.set_xlabel(r'step number', fontsize=12)
			for j in range(len(chain))[::nwalkers/11]:
				ax.plot(chain[j,:,p],color='k',alpha=0.1)
				ax.set_ylim(param_bounds[p])
				ax.axhline(p_true[p],color='r',alpha=0.25,linewidth=2)

		fig.savefig('small_trace_plots_'+date+'.png', dpi=150, bbox_inches='tight',transparent=True)
		mp.close()
		print_time()

	if autocorr_plots == True:
		print_message('...plotting autocorrelations')
		print_time()

		fig = mp.figure(figsize=(16,8))
		fig.subplots_adjust(wspace=0.4,hspace=0.2)

		maxlag = 200
		thin = 1
		for i in range(N_params):
			ax = fig.add_subplot(3,4,i+1)
			ax.set_ylim(-1,1)
			ax.axhline(0,color='k',alpha=0.5)
			mp.tick_params(which='both',right='off',top='off')
			ax.set_ylabel(p_latex[i],fontsize=20)
			series = chain[235,1000:,:].T[i][::thin]
			if np.isnan(astats.biweight_location(series)) == True:
				trend = np.median(series)
			else:
				trend = astats.biweight_location(series)
			ax.acorr(series-trend,maxlags=maxlag)
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
		p_lims = [[p_true[i]-p_eps[i],p_true[i]+p_eps[i]] for i in range(N_params)]
		p_lims = [[grid_tr.T[i].min(), grid_tr.T[i].max()] for i in range(N_params)]

		label_kwargs = {'fontsize':26}

		print '...plotting triangle'
		fig = corner.corner(samples, labels=p_latex, label_kwargs=label_kwargs,
							truths=p_true, range=p_lims, levels=levels, smooth=0.2,
							truth_color='orangered')

		add_fg_colors = True
		if add_fg_colors == True:
			axes = np.array(fig.axes).reshape(11,11)[1:5,0:4]
			axes = np.array([axes[i][:i+1] for i in range(len(axes))])
			axes = np.array([val for sublist in axes for val in sublist])
			for ax in axes:
				ax.patch.set_facecolor('purple')
				ax.patch.set_alpha(0.075)

			axes = np.array(fig.axes).reshape(11,11)[5:11,0:5].ravel()
			for ax in axes:
				ax.patch.set_facecolor('green')
				ax.patch.set_alpha(0.075)

			axes = np.array(fig.axes).reshape(11,11)[6:11,5:10]
			axes = np.array([axes[i][:i+1] for i in range(len(axes))])
			axes = np.array([val for sublist in axes for val in sublist])
			for ax in axes:
				ax.patch.set_facecolor('orange')
				ax.patch.set_alpha(0.075)

			p0 = matplotlib.patches.Rectangle([0,0],0,0,color='purple',alpha=0.1)
			p1 = matplotlib.patches.Rectangle([0,0],0,0,color='green',alpha=0.1)
			p2 = matplotlib.patches.Rectangle([0,0],0,0,color='orange',alpha=0.1)
			fig.legend([p0,p1,p2],['Cosmo-Cosmo','Cosmo-Astro','Astro-Astro'],fontsize=60,loc='upper center',frameon=False)

		add_prior = False
		if add_prior == True:
			axes = np.array(fig.axes).reshape(11,11).T
			for i in range(N_params):
				for j in range(i+1, N_params):
					plot_ellipse(cov=prior_cov[[i,j]].T[[i,j]], x_cent=prior_cent[i], y_cent=prior_cent[j],
						ax=axes[i,j], plot_kwargs={'color':'m','zorder':5}, mass_level=0.95)

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
					plot_kwargs = {'color':'b','linestyle':'-','linewidth':2,'alpha':0.75,'zorder':5}
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
					fig.axes[ax_ind[h]].plot(grid_tr.T[xp_ind][rando],grid_tr.T[yp_ind][rando],'r,',alpha=0.1)
					fig.axes[ax_ind[h]].plot(pos.T[xp_ind],pos.T[yp_ind],'c.',alpha=0.8)
				j += 1

		marg_constraints = False
		if marg_constraints == True:
			gs = gridspec.GridSpec(11,11)
			for i in range(11):
				ax=fig.add_subplot(gs[0,2:4])
				xp = np.linspace(param_bounds[i][0],param_bounds[i][1],100)
				yp = stats.norm.pdf(xp, loc=prior_cent[i], scale=np.sqrt(prior_cov[i,i]))
				p = ax.hist(samples.T[i], range=(param_bounds[i][0],param_bounds[i][1]),
								bins=30, color='k', linewidth=2.5, alpha=0.75, normed=True, histtype='step')
				yp *= p[0].max()/yp.max()
				ax.plot(xp, yp, color='m', linestyle='-', linewidth=2.5, alpha=0.5)
				ax.set_ylim(0,yp.max()*1.2)
				ax.set_xlabel(p_latex[i], fontsize=30)
				adjust_spines(ax, ['bottom'])
				_ = [tick.label.set_fontsize(30) for tick in ax.xaxis.get_major_ticks()]
				if len(ax.get_xticks()) > 4:
					_ = ax.set_xticks(ax.get_xticks()[1:][::2])
				_ = [tl.set_rotation(30) for tl in ax.get_xticklabels()]

		fig.savefig('tri_plot_'+date+'.png',dpi=150,bbox_inches='tight')
		mp.close()
		print_time()

	plot_prior = False
	if plot_prior == True:
		fig = mp.figure(figsize=(12,3))
		fig.subplots_adjust(wspace=0.3,hspace=0.1)

		j = 0
		for i in range(3):
			ax = fig.add_subplot(1,3,i+1)
			ax.grid(True)
			plot_ellipse(cov=planck_cov[[j,j+1]].T[[j,j+1]], x_cent=prior_cent[j], y_cent=prior_cent[j+1],
							ax=ax, plot_kwargs={'color':'darkblue','linewidth':1.5,'alpha':0.75}, mass_level=0.68)
			plot_ellipse(cov=planck_cov[[j,j+1]].T[[j,j+1]], x_cent=prior_cent[j], y_cent=prior_cent[j+1],
							ax=ax, plot_kwargs={'color':'darkblue','linewidth':1.5,'alpha':0.75}, mass_level=0.95)
			plot_ellipse(cov=prior_cov[[j,j+1]].T[[j,j+1]], x_cent=prior_cent[j], y_cent=prior_cent[j+1],
							ax=ax, plot_kwargs={'color':'darkorange','linewidth':1.5,'alpha':0.75}, mass_level=0.68)
			plot_ellipse(cov=prior_cov[[j,j+1]].T[[j,j+1]], x_cent=prior_cent[j], y_cent=prior_cent[j+1],
							ax=ax, plot_kwargs={'color':'darkorange','linewidth':1.5,'alpha':0.75}, mass_level=0.95)

			ax.set_xlabel(p_latex[j])
			ax.set_ylabel(p_latex[j+1])
			ax.set_xlim(prior_cent[j]-param_width[j]/2.0, prior_cent[j]+param_width[j]/2.0)
			ax.set_ylim(prior_cent[j+1]-param_width[j+1]/2.0, prior_cent[j+1]+param_width[j+1]/2.0)	
			[tl.set_rotation(30) for tl in ax.get_xticklabels()]
			if i == 0:
				j += 1
			else:
				j += 2

			if i == 0:
				p1 = matplotlib.lines.Line2D([0],[0],color='darkblue',linewidth=3)
				p0 = matplotlib.lines.Line2D([0],[0],color='darkorange',linewidth=3)
				ax.legend([p0,p1],['Prior Covariance', 'Planck Covariance'],fontsize=11, loc=0)

		fig.savefig('cosmo_priors_'+date+'.png', dpi=150, bbox_inches='tight')
		mp.close()

	if plot_recplot == True:
		print_message('...making rec plot')
		print_time()
		pbound = np.array([grid_tr.T[i].max()-grid_tr.T[i].min() for i in range(11)])
		lims = [[grid_tr.T[i].min()-pbound[i]*0.1, grid_tr.T[i].max()+pbound[i]*0.1] for i in range(11)]

		gs = gridspec.GridSpec(9,14)
		sub1 = np.concatenate([[np.array([1,0,1]) for i in range(3)],[np.array([6,5,6]) for i in range(3)]])
		sub2 = np.concatenate([[np.array([4,1,4]) for i in range(3)],[np.array([9,6,9]) for i in range(3)]])
		sub3 = np.array([[np.array([0,0,3]),np.array([5,5,8]),np.array([10,10,13])] for i in range(2)]).reshape(6,3)
		sub4 = np.array([[np.array([3,3,4]),np.array([8,8,9]),np.array([13,13,14])] for i in range(2)]).reshape(6,3)
		levels = [0.68, 0.95]

		fig = mp.figure(figsize=(13,8))
		fig.subplots_adjust(wspace=0.0, hspace=0.0)
		bins = 50
		j = 0
		for i in range(6):
			for k in range(3):
				ax = fig.add_subplot(gs[sub1[i,k]:sub2[i,k],sub3[i,k]:sub4[i,k]])
				if k == 0:
					ax.plot(grid_tr.T[j],grid_tr.T[j+1],color='steelblue',marker=',',linestyle='',alpha=0.75,zorder=0)
					corner.hist2d(samples.T[j], samples.T[j+1], ax=ax, color='k', bins=bins, smooth=0.7, levels=levels,
								plot_datapoints=False, range=np.array([lims[j],lims[j+1]]), zorder=1)
					ax.scatter(p_true[j], p_true[j+1], color='orangered', marker='s', edgecolor='', s=60, zorder=2)
					ax.axvline(p_true[j], color='orangered', linewidth=1.5, alpha=0.75)
					ax.axhline(p_true[j+1], color='orangered', linewidth=1.5, alpha=0.75)
					ax.set_xlabel(p_latex[j], fontsize=17)
					ax.set_ylabel(p_latex[j+1], fontsize=17)
					ax.get_yaxis().set_label_coords(-0.2, 0.75)
					[l.set_rotation(30) for l in ax.get_xticklabels()]
					[l.set_rotation(30) for l in ax.get_yticklabels()]
					ax.set_xlim(lims[j])
					ax.set_ylim(lims[j+1])
					plot_ellipse(cov=prior_cov[[j,j+1]].T[[j,j+1]], x_cent=prior_cent[j], y_cent=prior_cent[j+1],
									ax=ax, plot_kwargs={'color':'m','zorder':5}, mass_level=0.95)
				if k == 1:
					ax.axis('off')
					hout = ax.hist(samples.T[j], histtype='step', color='k', linewidth=1.5, bins=bins, range=np.array(lims[j]))
					ax.axvline(p_true[j], color='orangered', linewidth=1.5, alpha=0.75, ymax=0.9)
					ax.set_xlim(lims[j])
					ax.set_ylim(0,hout[0].max()*1.15)
				if k == 2:
					ax.axis('off')
					hout = ax.hist(samples.T[j+1], histtype='step', color='k', linewidth=1.5, bins=bins, range=np.array(lims[j+1]), orientation='horizontal')
					ax.axhline(p_true[j+1], color='orangered', linewidth=1.5, alpha=0.75, xmax=0.9)
					ax.set_xlim(0,hout[0].max()*1.15)
					ax.set_ylim(lims[j+1])
			if i == 0:
				j += 1
			else:
				j += 2

		fig.savefig('recplot_'+date+'.png',dpi=150,bbox_inches='tight')
		mp.close()
		print_time()

	if plot_boxplots == True:
		# Plot marginalized pdfs and compare to priors
		gs = gridspec.GridSpec(11,9)
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

	if plot_marghist == True:
		gs = gridspec.GridSpec(11,9)
		sub1 = np.array([np.arange(0,10,3) for i in range(3)]).T.ravel()
		sub2 = np.array([np.arange(2,12,3) for i in range(3)]).T.ravel()
		sub3 = np.array([np.arange(0,7,3) for i in range(4)]).ravel()
		sub4 = np.array([np.arange(3,10,3) for i in range(4)]).ravel()

		fig = mp.figure(figsize=(9,11))
		fig.subplots_adjust(wspace=0.2,hspace=0.05)

		for i in range(N_params):
			# init axes
			ax = fig.add_subplot(gs[sub1[i+1]:sub2[i+1],sub3[i+1]:sub4[i+1]])
			mp.tick_params(which='both',left='off',right='off')
			ax.set_xlabel(p_latex[i],fontsize=20)

			# fill in prior and posterior
			pbound = np.array([prior_cent[i]-param_width[i]/2, prior_cent[i]+param_width[i]/2])
			x = np.linspace(pbound[0], pbound[1], 200)
			y = stats.norm.pdf(x, loc=prior_cent[i], scale=np.sqrt(prior_cov.diagonal()[i]))
			hist_cent = astats.biweight_location(samples.T[i])
			patches = ax.hist(samples.T[i]+(prior_cent[i]-hist_cent), color='dodgerblue', linewidth=1.5, alpha=0.8, histtype='step',
							range=(pbound[0],pbound[1]), bins=80, normed=True, zorder=2)
			y *= (patches[0].max()/y.max())
			ax.plot(x, y, color='darkred', linewidth=1.5, alpha=0.75, zorder=1)

			ax.tick_params('both',length=6)
			_ = [tl.set_size(14) for tl in ax.get_xticklabels()]
			_ = [tl.set_rotation(0) for tl in ax.get_xticklabels()]
			[tl.set_visible(False) for tl in ax.get_xticklabels()[::2]]
			ax.set_yticklabels([])
			ax.set_xlim(pbound[0] - param_width[i]*0.1, pbound[1] + param_width[i]*0.1)
			ax.set_ylim(0,patches[0].max()*1.2)

		p0 = matplotlib.lines.Line2D([0],[0],color='darkred',linewidth=3)
		p1 = matplotlib.lines.Line2D([0],[0],color='dodgerblue',linewidth=3)
		ax = fig.add_subplot(gs[sub1[0]:sub2[0],sub3[0]:sub4[0]])
		mp.axis('off')
		ax.legend([p0,p1],[r'Prior',r'Posterior'],fontsize=20,frameon=False)

		fig.savefig('marghist_'+date+'.png',dpi=200,bbox_inches='tight')
		mp.close()

	if plot_map_pspec == True:
		xdata = O.xdata
		ydata = O.row2mat(O.ydata)
		yerrs = O.row2mat(O.yerrs)

		theta_map = []
		for b in np.arange(20,50,5):
			hist = np.array(map(lambda x: np.histogram(x, bins=b, normed=True), samples.T))
			theta_map.append(map(lambda x: x[1][np.argmax(x[0])] + x[1][1]-x[1][0], hist))

		theta_map = np.array(map(astats.biweight_location, np.array(theta_map).T))
		E.predict(theta_map, **predict_kwargs)
		ypred = O.row2mat(E.recon[0])

		fig = mp.figure(figsize=(8,2))
		fig.subplots_adjust(hspace=0.02)

		i = 0
		for z in np.arange(44)[5:29:6]:
			ax = fig.add_subplot(1,4,i+1)
			ax.grid(True)
			ax.errorbar(xdata[z], ydata[z], yerr=yerrs[z], color='red', fmt='s', alpha=0.9, markersize=1, ecolor='None')
			#ax.plot(xdata[z], ypred[z], color='dodgerblue', linewidth=1.5, alpha=1.0)
			ax.axvspan(6e-2,1e-1, color='grey', alpha=0.2)
			ax.axvspan(6e-2,1e-1, hatch='\\', color='None', alpha=1.0)
			ax.set_xlim(6e-2, 5)
			ax.set_ylim(1, 1e5)
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_title(r'$z = '+str(z_array[z])+'$', fontsize=14)
			if i == 0:
				ax.set_xlabel(r'$k$ ($h$ Mpc$^{-1}$)', fontsize=13)
				ax.set_ylabel(r'$\Delta^{2}_{21}$ (mK$^{2}$)', fontsize=13)
			else:
				ax.set_yticklabels([])

			i += 1

		fig.savefig('MAP_pspec_'+date+'.png', dpi=200, bbox_inches='tight')
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
			sel = np.array(reduce(operator.mul,np.array([grid_cv.T[p]==p_true[p] if p != i else np.ones(len(grid_cv.T[p])) for p in range(N_params)])),bool)
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
						ax.set_ylabel(r'$\Delta^{2}_{21}$ (mK$^{2}$)',fontsize=16,labelpad=0)
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










