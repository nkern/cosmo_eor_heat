#!/bin/bash
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --job-name=runjob
#SBATCH --output=runjob_%j.out
#SBATCH --qos=premium

bash param_space/lhs/zeta_060.850_numin_406.402/run_21cmFAST.sh >& param_space/lhs/zeta_060.850_numin_406.402/jobout.txt




