#!/bin/bash
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --job-name=runjob
#SBATCH --output=runjob_%j.out
#SBATCH --qos=normal
#SBATCH --constraint=haswell

ipython mcmc_config.py

