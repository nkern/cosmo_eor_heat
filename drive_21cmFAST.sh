#!/bin/bash

# Get time
echo '--------------------------------------------'
start_d=($(date))
echo ...Start Time: ${start_d[@]}
echo '--------------------------------------------'
echo ''

# Go to Working Directory
echo '--------------------------------------------'
cd @@working_direc@@/Programs
echo "working directory = @@working_direc@@"
echo '--------------------------------------------'
echo ''

# Attach Programs/ to PATH
echo '--------------------------------------------'
current_dir="$(pwd)"
export PATH="$current_dir":$PATH
echo "PATH = "$PATH
echo '--------------------------------------------'
echo ''

# Echo fisher params file
echo '--------------------------------------------'
cat ../mcmc_params.py
echo '--------------------------------------------'
echo ''

# Remove finish file
rm ../finish.txt

# Delete executables if present
make clean

# Run Makefile then run command
@@command@@

# Make finish file
touch ../finish.txt
end_d=($(date))
echo ...End Time: ${end_d[@]}

echo "Start Time: "${start_d[@]} >> ../finish.txt
echo "End Time  : "${end_d[@]} >> ../finish.txt

# Calculate Tau
python ../calc_tau.py --overwrite

# Extract Global Parameters
python ../global_params.py

# Take Programs/ out of the PATH
IFS=":" read -a path <<< "$PATH"
unset path[0]
function join { local IFS="$1"; shift; echo "$*"; }
PATH=$(join : "${path[@]}")

# Remove Boxes data
python ../rm_boxes.py

# Clean Executables
make clean

# Exit Program
echo "...done with drive_21cmFAST"
exit 0
