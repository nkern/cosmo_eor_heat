#!/bin/bash
#SBATCH --partition=@@partition@@
#SBATCH --nodes=@@Nnodes@@
#SBATCH --tasks=@@Ntasks@@
#SBATCH --cpus-per-task=@@cpus_per_task@@
#SBATCH --mem-per-cpu=@@mem_per_cpu@@
#SBATCH --time=@@walltime@@
#SBATCH --job-name=@@job_name@@
#SBATCH --output=job_%j.out
#SBATCH --qos=normal

echo "--------------------------------"
echo "running slurm_21cmFAST.sh"
echo start: $(date)
echo ""

# SRUN Info
nodes=@@Nnodes@@
tasks_per_node=@@tasks_per_node@@
cpus=@@cpus_per_task@@
CPUMem=@@mem_per_cpu@@

# Get Directories
IFS=$'\r\n' command eval 'direcs=($(<@@direc_file@@))'

# Slice direcs
begin=@@Nstart@@
tot_length=@@Nruns@@
direcs=("${direcs[@]:$begin:$tot_length}")

# Define Loop Variables
Nseq=@@Nseq@@
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
echo "done with slurm_21cmFAST.sh"
echo end: $(date)
echo "--------------------------------"
