#!/bin/bash
#SBATCH --partition=@@partition@@
#SBATCH --nodes=@@Nnodes@@
#SBATCH --time=@@walltime@@
#SBATCH --job-name=@@job_name@@
#SBATCH --output=job_%j.out
#SBATCH --qos=normal

echo "--------------------------------"
echo "running slurm_21cmFAST.sh"
echo start: $(date)
echo ""

# Get Directories
IFS=$'\r\n' command eval 'direcs=($(<@@direc_file@@))'

# Slice direcs
begin=@@Nstart@@
tot_length=@@Nruns@@
direcs=("${direcs[@]:$begin:$tot_length}")

# Define Loop Variables
Nseq=@@Nseq@@
begin=0
length=@@tasks_per_node@@

# Iterate over Sequential Runs
for i in $(seq 0 $((Nseq-1)))
do
	echo "...Starting Iteration $i"
	echo "----------------------------"
	dirs=("${direcs[@]:$begin:$length}")
	for j in ${dirs[@]}
	do
		echo $j
		bash $j/run_21cmFAST.sh >& $j/jobout.txt &
	done
	wait
	begin=$((begin+length))
	echo ""
done

echo ""
echo "done with slurm_21cmFAST.sh"
echo end: $(date)
echo "--------------------------------"

