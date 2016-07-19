#!/bin/bash

# Get time
start_d=($(date))
echo ...Start Time: ${start_d[@]}

# Go to Working Directory
cd @@working_direc@@/Programs

# Attach Programs/ to PATH
current_dir="$(pwd)"
export PATH="$current_dir":$PATH
echo "PATH = "$PATH

# Echo fisher params file
cat ../fisher_params.py

# Remove finish file
rm ../finish.txt

# Delete executables if present
make clean

# Run Makefile then run command
@@command@@

# Make finish file
touch ../finish.txt
end_d=($(date))
echo ...End Tim: ${end_d[@]}

echo "Start Time: "${start_d[@]} >> ../finish.txt
echo "End Time  : "${end_d[@]} >> ../finish.txt

# Extract Global Parameters
python ../global_params.py

# Take Programs/ out of the PATH
IFS=":" read -a path <<< "$PATH"
unset path[0]
function join { local IFS="$1"; shift; echo "$*"; }
PATH=$(join : "${path[@]}")

# Remove Boxes data
rm -vrf ../Boxes/*

# Clean Executables
make clean
