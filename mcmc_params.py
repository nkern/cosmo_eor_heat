'''
Parameter file for 11 parameter model MCMC emulation
'''
import numpy as np
import DictEZ

## Flags
sample_grid		= False							# Sample training set points
compile_direcs	= False							# Turn grid data into directory names
write_direcs	= False							# Write directory names to direcs.tab
build_direcs	= False							# Using samples, write 21cmFAST direcs
make_fiducial	= False							# Make fiducial run for Observation
send_slurm_jobs = True                       # Send jobs to PBS scheduler

## Astro / Cosmo Variables
sigma8          = 0.8159                        # Fiducial value for Sigma8
hlittle			= 0.6774						# Fiducial value for littleh
OMbh2			= 0.02230						# Fiducial value for omega b h**2
OMch2			= 0.11880						# Fiducial value for omega c h**2
ns				= 0.9667						# Fiducial value for spectral index

Tvir            = 5e4                      		# Fiducial value for Tvir
zeta            = 40.0                     		# Fiducial value for zeta
Rmfp            = 15.0                     		# Fiducial value for Rmfp

fX				= 2.0							# Fiducial value for fX
aX				= 1.5							# Fiducial value for aX
numin			= 300							# Fiducial value for numin

# Simulation Variables
z_start         = 25.00                         # starting redshift
z_end           = 4.5							# ending redshift
z_step          = -1.0                          # redshift steps
zlow			= 5.0							# Low redshift for Ts calculation
zprime			= 1.0245							# Redshift Coefficient for logzscroll

randomseed		= 111							# Seed for ICs
boxlen			= 400							# Box Length cMpc
dim				= 800							# Side dimension of High Res grid
HIIdim			= 200							# Side dimension of Low Res grid
computeRmfp		= 0								# Compute Rmfp 1=True, 0=False (i.e. feed it)
numcores		= 2								# Number of cores per simulation instatiation
ram				= 4								# Physical Memory in GB available per simulation
ram_needed		= (dim**3+4*HIIdim**3)*4/1e9	# In GB
use_Ts			= 1								# Use Ts in calculation

kmin			= None							# Lower k value for KL analysis
kmax			= None							# Upper k value for KL analysis
zmin			= None							# Lower z value for KL analysis
zmax			= None							# Upper z value for KL analysis
gmin			= None							# Limit to global params
gmax			= None							# Limit to global params

N_samples		= 4000							# Number of samples to draw from multi-Gaussian or to use in training
eval_samples	= np.arange(0,4000)				# Total number of samples in dataset, train + cv (excluding fiducial)
N_train			= 4000							# Samples to train on
N_cv			= 0								# Samples to cross validate on

## Organize Parameters
params          = ['sigma8','hlittle','OMbh2','OMch2','ns',
					'Tvir','zeta','Rmfp',
					'fX','aX','numin']         					     		# Cosmological and Astrophysical parameters
params_fid      = [sigma8,hlittle,OMbh2,OMch2,ns,
					Tvir,zeta,Rmfp,fX,aX,numin]								# Fiducial values
params_prefix   = map(lambda x: x +"_",params)          					# Strings to create directories etc.
p_latex         = ['$\sigma_{8}$','$h$','$\Omega_{b}h^{2}$',
					'$\Omega_{c}h^{2}$','$n_{s}$','$T_{vir}$','$\zeta$',
					'$R_{mfp}$','$f_{X}$','$\\alpha_{X}$','$\\nu_{min}$']		# params list but in LaTeX
p_fid_latex		= ['$\sigma_{8}^{fid}$','$h^{fid}$','$\Omega_{b}h^{2}^{fid}$',
                    '$\Omega_{c}h^{2}^{fid}$','$n_{s}^{fid}$','$T_{vir}^{fid}$','$\zeta^{fid}$',
                    '$R_{mfp}^{fid}$','$f_{X}^{fid}$','$\\alpha_{X}^{fid}$','$\\nu_{min}^{fid}$']     # params list but in LaTeX

variables       = ['z_start','z_end','z_step','zlow','zprime','randomseed','boxlen',
					'dim','HIIdim','computeRmfp','numcores','ram','use_Ts']         # Other variables to include in parameter files
variables		= DictEZ.create(variables,globals())

base_direc      = 'param_space/cross_valid/'                        # directory that opens up to 21cmFAST realizations

sim_root        = '/global/homes/n/nkern/Software/21cmFAST_v1'     # Where Home 21cmFAST directory lives
direc_root		= '/global/cscratch1/sd/nkern/EoR/cosmo_eor_heat/mcmc'	# Where this directory lives
command			= 'make;./drive_logZscroll_Ts'
