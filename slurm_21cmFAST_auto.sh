#!/bin/bash
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --time=44:00:00
#SBATCH --job-name=Small
#SBATCH --output=job_%j.out
#SBATCH --qos=normal

echo "--------------------------------"
echo "running slurm_21cmFAST.sh"
echo start: $(date)
echo ""

# Get Directories
IFS=$'\r\n' command eval 'direcs=($(<direcs.tab))'

# Slice direcs
begin=9968
tot_length=32
direcs=("${direcs[@]:$begin:$tot_length}")

# Define Loop Variables
Nseq=4
begin=0
length=8

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

