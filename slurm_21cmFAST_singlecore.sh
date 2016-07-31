#!/bin/bash
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --job-name=21cmFAST
#SBATCH --output=job_%j.out
#SBATCH --qos=normal

echo "--------------------------------"
echo "running slurm_21cmFAST.sh"
echo start: $(date)
echo ""

# Get Directories
IFS=$'\r\n' command eval 'direcs=($(<rerun.tab))'
basedir='param_space/'

# Slice direcs
begin=0
tot_length=16
direcs=("${direcs[@]:$begin:$tot_length}")

# Define Loop Variables
Nseq=1
begin=0
length=12

# Iterate over Sequential Runs
for i in $(seq 0 $((Nseq-1)))
do
	echo "...Starting Iteration $i"
	echo "----------------------------"
	dirs=("${direcs[@]:$begin:$length}")
	for j in ${dirs[@]}
	do
		echo $j
		bash $basedir$j/run_21cmFAST.sh >& $basedir$j/jobout.txt &
	done
	wait
	begin=$((begin+length))
	echo ""
done

echo ""
echo "done with slurm_21cmFAST.sh"
echo end: $(date)
echo "--------------------------------"

