'''
Helper functions for placing and running 21cmFAST files
'''

import os

def insert_params(params,directory):
	''' 
	- params is a dictionary with param names as str and values as floats, directory is a string where 21cmFAST files to-be-edited live
	- it is assumed that the directory/ already has the 21cmFAST Programs, Parameter_files and Cosmo_c_files copied over
	'''

	## Redshifts ##
	if 'z_start' in params:	
		os.system("sed -e 's:ZSTART (.*): ZSTART ("+str(params['z_start'])+"):g;s:ZEND (.*):ZEND ("+str(params['z_end'])+"):g;s:ZSTEP (.*):ZSTEP ("+str(params['z_step'])+"):g;' < "+directory+"/Programs/drive_zscroll_noTs.c > "+directory+"/Programs/drive_zscroll_noTs_temp.c")
		os.system("mv "+directory+"/Programs/drive_zscroll_noTs_temp.c "+directory+"/Programs/drive_zscroll_noTs.c")

		os.system("sed -e 's:ZSTART (.*): ZSTART ("+str(params['z_start'])+"):g;s:ZEND (.*):ZEND ("+str(params['z_end'])+"):g;s:ZSTEP (.*):ZSTEP ("+str(params['z_step'])+"):g;' < "+directory+"/Programs/drive_zscroll_noTs_cosmo.c > "+directory+"/Programs/drive_zscroll_noTs_cosmo_temp.c")
		os.system("mv "+directory+"/Programs/drive_zscroll_noTs_cosmo_temp.c "+directory+"/Programs/drive_zscroll_noTs_cosmo.c")

		os.system("sed -e 's:ZSTART (.*): ZSTART ("+str(params['z_start'])+"):g;s:ZEND (.*):ZEND ("+str(params['z_end'])+"):g;s:ZSTEP (.*):ZSTEP ("+str(params['z_step'])+"):g;' < "+directory+"/Programs/drive_zscroll_noTs_astro.c > "+directory+"/Programs/drive_zscroll_noTs_astro_temp.c")
		os.system("mv "+directory+"/Programs/drive_zscroll_noTs_astro_temp.c "+directory+"/Programs/drive_zscroll_noTs_astro.c")

	if 'zlow' in params:
		os.system("sed -e 's:ZLOW (float) (.*): ZLOW (float) ("+str(params['zlow'])+"):g;' < "+directory+"/Programs/drive_logZscroll_Ts.c > "+directory+"/Programs/drive_logZscroll_Ts_temp.c")
		os.system("mv "+directory+"/Programs/drive_logZscroll_Ts_temp.c "+directory+"/Programs/drive_logZscroll_Ts.c")

	## Cosmological / Astrophysical Parameters  ##

	# Tvir
	if 'Tvir' in params:
		os.system("sed -e 's:ION_Tvir_MIN (double) (.*):ION_Tvir_MIN (double) ("+str(params['Tvir'])+"):g;' < "+directory+"/Parameter_files/ANAL_PARAMS.H > "+directory+"/Parameter_files/ANAL_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/ANAL_PARAMS_temp.H "+directory+"/Parameter_files/ANAL_PARAMS.H")

		os.system("sed -e 's:X_RAY_Tvir_MIN (double) (.*):X_RAY_Tvir_MIN (double) ("+str(params['Tvir'])+"):g;' < "+directory+"/Parameter_files/HEAT_PARAMS.H > "+directory+"/Parameter_files/HEAT_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/HEAT_PARAMS_temp.H "+directory+"/Parameter_files/HEAT_PARAMS.H")

	# zeta
	if 'zeta' in params:
		os.system("sed -e 's:HII_EFF_FACTOR (float) (.*):HII_EFF_FACTOR (float) ("+str(params['zeta'])+"):g' < "+directory+"/Parameter_files/ANAL_PARAMS.H > "+directory+"/Parameter_files/ANAL_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/ANAL_PARAMS_temp.H "+directory+"/Parameter_files/ANAL_PARAMS.H")

	print 'done editing parameter files...'

	# Rmfp
	if 'Rmfp' in params:
		os.system("sed -e 's:R_BUBBLE_MAX (float) (.*):R_BUBBLE_MAX (float) ("+str(params['Rmfp'])+"):g' < "+directory+"/Parameter_files/ANAL_PARAMS.H > "+directory+"/Parameter_files/ANAL_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/ANAL_PARAMS_temp.H "+directory+"/Parameter_files/ANAL_PARAMS.H")

	# Sigma8
	if 'sigma8' in params:
		os.system("sed -e 's:SIGMA8 (float) (.*):SIGMA8 (float) ("+str(params['sigma8'])+"):g;' < "+directory+"/Parameter_files/COSMOLOGY.H > "+directory+"/Parameter_files/COSMOLOGY_temp.H")
		os.system("mv "+directory+"/Parameter_files/COSMOLOGY_temp.H "+directory+"/Parameter_files/COSMOLOGY.H")

    # little h
	if 'hlittle' in params:
		os.system("sed -e 's:hlittle (float) (.*):hlittle (float) ("+str(params['hlittle'])+"):g;' < "+directory+"/Parameter_files/COSMOLOGY.H > "+directory+"/Parameter_files/COSMOLOGY_temp.H")
		os.system("mv "+directory+"/Parameter_files/COSMOLOGY_temp.H "+directory+"/Parameter_files/COSMOLOGY.H")

    # Omega baryon h^2
	if 'OMbh2' in params:
		os.system("sed -e 's:OMb (float) (.*):OMb (float) ("+str(params['OMbh2']/(params['hlittle']**2))+"):g;' < "+directory+"/Parameter_files/COSMOLOGY.H > "+directory+"/Parameter_files/COSMOLOGY_temp.H")
		os.system("mv "+directory+"/Parameter_files/COSMOLOGY_temp.H "+directory+"/Parameter_files/COSMOLOGY.H")

    # Omega CDM h^2
	if 'OMch2' in params:
		os.system("sed -e 's:OMc (float) (.*):OMc (float) ("+str(params['OMch2']/(params['hlittle']**2))+"):g;' < "+directory+"/Parameter_files/COSMOLOGY.H > "+directory+"/Parameter_files/COSMOLOGY_temp.H")
		os.system("mv "+directory+"/Parameter_files/COSMOLOGY_temp.H "+directory+"/Parameter_files/COSMOLOGY.H")

    # n_s
	if 'ns' in params:
		os.system("sed -e 's:POWER_INDEX (float) (.*):POWER_INDEX (float) ("+str(params['ns'])+"):g;' < "+directory+"/Parameter_files/COSMOLOGY.H > "+directory+"/Parameter_files/COSMOLOGY_temp.H")
		os.system("mv "+directory+"/Parameter_files/COSMOLOGY_temp.H "+directory+"/Parameter_files/COSMOLOGY.H")

	# f_X
	if 'fX' in params:
		os.system("sed -e 's:X_RAY_EFF_FACTOR (double) (.*):X_RAY_EFF_FACTOR (double) ("+str(params['fX'])+"):g' < "+directory+"/Parameter_files/HEAT_PARAMS.H > "+directory+"/Parameter_files/HEAT_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/HEAT_PARAMS_temp.H "+directory+"/Parameter_files/HEAT_PARAMS.H")

	# N_X
	if 'NX' in params:
		os.system("sed -e 's:N_X (double) (.*):N_X (double) ("+str(params['NX'])+"):g' < "+directory+"/Parameter_files/HEAT_PARAMS.H > "+directory+"/Parameter_files/HEAT_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/HEAT_PARAMS_temp.H "+directory+"/Parameter_files/HEAT_PARAMS.H")

	# a_X
	if 'aX' in params:
		os.system("sed -e 's:X_RAY_SPEC_INDEX (double) (.*):X_RAY_SPEC_INDEX (double) ("+str(params['aX'])+"):g' < "+directory+"/Parameter_files/HEAT_PARAMS.H > "+directory+"/Parameter_files/HEAT_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/HEAT_PARAMS_temp.H "+directory+"/Parameter_files/HEAT_PARAMS.H")

	# numin
	if 'numin' in params:
		os.system("sed -e 's:NU_X_THRESH (double) (.*):NU_X_THRESH (double) ("+str(params['numin'])+"*NU_over_EV):g' < "+directory+"/Parameter_files/HEAT_PARAMS.H > "+directory+"/Parameter_files/HEAT_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/HEAT_PARAMS_temp.H "+directory+"/Parameter_files/HEAT_PARAMS.H")


	## Box Initialization Parameters ##

	# Random Seed
	if 'randomseed' in params:
		os.system("sed -e 's:RANDOM_SEED (long) (.*) //:RANDOM_SEED (long) ("+str(params['randomseed'])+") //:g;' < "+directory+"/Parameter_files/INIT_PARAMS.H > "+directory+"/Parameter_files/INIT_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/INIT_PARAMS_temp.H "+directory+"/Parameter_files/INIT_PARAMS.H")

	# Box Len
	if 'boxlen' in params:
		os.system("sed -e 's:BOX_LEN (float) (.*) //:BOX_LEN (float) ("+str(params['boxlen'])+") //:g;' < "+directory+"/Parameter_files/INIT_PARAMS.H > "+directory+"/Parameter_files/INIT_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/INIT_PARAMS_temp.H "+directory+"/Parameter_files/INIT_PARAMS.H")

	# Dim
	if 'dim' in params:
		os.system("sed -e 's:DIM (int) (.*) //:DIM (int) ("+str(params['dim'])+") //:g;' < "+directory+"/Parameter_files/INIT_PARAMS.H > "+directory+"/Parameter_files/INIT_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/INIT_PARAMS_temp.H "+directory+"/Parameter_files/INIT_PARAMS.H")

	# HIIDim
	if 'HIIdim' in params:
		os.system("sed -e 's:HII_DIM (int) (.*) //:HII_DIM (int) ("+str(params['HIIdim'])+") //:g;' < "+directory+"/Parameter_files/INIT_PARAMS.H > "+directory+"/Parameter_files/INIT_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/INIT_PARAMS_temp.H "+directory+"/Parameter_files/INIT_PARAMS.H")

	## Other Parameters ##

	# compute_Rmfp
	if 'computeRmfp' in params:
		os.system("sed -e 's:COMPUTE_MFP (int) (.*):COMPUTE_MFP (int) ("+str(params['computeRmfp'])+"):g' < "+directory+"/Parameter_files/ANAL_PARAMS.H > "+directory+"/Parameter_files/ANAL_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/ANAL_PARAMS_temp.H "+directory+"/Parameter_files/ANAL_PARAMS.H")

	# zprime
	if 'zprime' in params:
		os.system("sed -e 's:ZPRIME_STEP_FACTOR (float) (.*):ZPRIME_STEP_FACTOR (float) ("+str(params['zprime'])+"):g' < "+directory+"/Parameter_files/HEAT_PARAMS.H > "+directory+"/Parameter_files/HEAT_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/HEAT_PARAMS_temp.H "+directory+"/Parameter_files/HEAT_PARAMS.H")

	# NUMCORES
	if 'numcores' in params:
		os.system("sed -e 's:NUMCORES (int) (.*) //:NUMCORES (int) ("+str(params['numcores'])+") //:g' < "+directory+"/Parameter_files/INIT_PARAMS.H > "+directory+"/Parameter_files/INIT_PARAMS_temp.H")	
		os.system("mv "+directory+"/Parameter_files/INIT_PARAMS_temp.H "+directory+"/Parameter_files/INIT_PARAMS.H")

	# RAM
	if 'ram' in params:
		os.system("sed -e 's:RAM (float) (.*) //:RAM (float) ("+str(params['ram'])+") //:g' < "+directory+"/Parameter_files/INIT_PARAMS.H > "+directory+"/Parameter_files/INIT_PARAMS_temp.H")      
		os.system("mv "+directory+"/Parameter_files/INIT_PARAMS_temp.H "+directory+"/Parameter_files/INIT_PARAMS.H")

    # Use_Ts
	if 'use_Ts' in params:
		os.system("sed -e 's:USE_TS_IN_21CM (int) (.*):USE_TS_IN_21CM (int) ("+str(params['use_Ts'])+"):g' < "+directory+"/Parameter_files/HEAT_PARAMS.H > "+directory+"/Parameter_files/HEAT_PARAMS_temp.H")
		os.system("mv "+directory+"/Parameter_files/HEAT_PARAMS_temp.H "+directory+"/Parameter_files/HEAT_PARAMS.H")
	
