#!/bin/bash
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --time=30:00
#SBATCH --job-name=21cmFAST
#SBATCH --output=job_%j.out
#SBATCH --qos=normal

echo "--------------------------------"
echo "running slurm_21cmFAST.sh"
echo start: $(date)
echo ""

# SRUN Info
nodes=2
tasks=8
cpus=4
CPUMem=500

# Get Directories
IFS=$'\r\n' command eval 'direcs=($(<direcs.tab))'

# Slice direcs
begin=0
tot_length=16
direcs=("${direcs[@]:$begin:$tot_length}")

# Define Loop Variables
Nseq=2
begin=0
length=$tasks

# Iterate over Sequential runs, Edit Configuration Files and SRUN
for i in $(seq 1 $Nseq)
	do
	echo "...Starting Iteration $i"
	echo "==========================="
	dirs=("${direcs[@]:$begin:$length}")
	echo '...configuring multiprog'
	echo '-------------------------'
	python config_multiprog.py config "$SLURM_JOBID"_"$i" ${dirs[@]}
	echo ''
	echo '...submitting srun'
	echo '-------------------------'
	srun -N $nodes -n $tasks -c $cpus --mem-per-cpu=$CPUMem -o job"$SLURM_JOBID"_seq"$i"_task%2t.out --multi-prog MPMD"$SLURM_JOBID"_"$i".conf &
	echo ''
	echo '...sending jobout files to direcs'
	echo '-------------------------'
	(sleep 10; python config_multiprog.py jobout job"$SLURM_JOBID"_seq"$i"_task ${dirs[@]}) &
	wait
	begin=$((begin+length))
	echo "==========================="
	echo ""
	done

echo ""
echo "done with slurm_21cmFAST.sh"
echo end: $(date)
echo "--------------------------------"
