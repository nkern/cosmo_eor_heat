#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=25
#SBATCH --tasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=500
#SBATCH --time=30:00
#SBATCH --job-name=21cmFAST
#SBATCH --output=job_%j.out
#SBATCH --qos=normal

echo "--------------------------------"
echo "running slurm_21cmFAST3.sh"
echo start: $(date)
echo ""

# SRUN Info
nodes=4
tasks_per_node=2
cpus=4
CPUMem=500

# Get Directories
IFS=$'\r\n' command eval 'direcs=($(<direcs.tab))'

# Slice direcs
begin=0
tot_length=8
direcs=("${direcs[@]:$begin:$tot_length}")

# Define Loop Variables
Nseq=1
begin=0
length=$tasks_per_node

# Make MPMD Configuration files
echo ""
echo "...Making MPMD Config Files"
for i in $(seq 1 $Nseq)
	do
	for k in $(seq 0 $((nodes-1)))
		do
		echo "...Seq"$i" Node"$k
		dirs=("${direcs[@]:$begin:$length}")
		python config_multiprog.py config $SLURM_JOBID $i $k ${dirs[@]}
		begin=$((begin+length))
		done
	done
echo ""

# Iterate over Sequential runs, Edit Configuration Files and SRUN
for i in $(seq 1 $Nseq)
do
	echo "...Starting Sequential Iteration $i"
	echo "==========================="
	for k in $(seq 0 $((nodes-1)))
	do
		echo ""
		echo "...submitting srun for node $k"
		srun -N 1 -n $tasks_per_node -c $cpus --mem-per-cpu=$CPUMem -o job"$SLURM_JOBID"_seq"$i"_node"$k"_task%2t.out --exclusive --multi-prog MPMD_job"$SLURM_JOBID"_seq"$i"_node"$k".conf &
	done
	echo ''
	echo "...sending jobout files to direcs"
	(sleep 10; python config_multiprog.py jobout $SLURM_JOBID $i $nodes) &
	wait

	echo ''
	echo "==========================="
	echo ""
	done

echo ""
echo "done with slurm_21cmFAST3.sh"
echo end: $(date)
echo "--------------------------------"
