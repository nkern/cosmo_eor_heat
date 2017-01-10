#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=30:00
#SBATCH --job-name=runjob
#SBATCH --output=runjob_%j.out
#SBATCH --qos=normal
#SBATCH --constraint=haswell

ipython run_tau.py

