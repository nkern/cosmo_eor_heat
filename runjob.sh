#!/bin/bash
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=runjob
#SBATCH --output=runjob_%j.out
#SBATCH --qos=premium

ipython mcmc_config.py


