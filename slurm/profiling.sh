#!/bin/bash
#SBATCH --job-name=profile
#SBATCH --mail-user=ajensen123@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=hpg-ai
#SBATCH --ntasks=1
#SBATCH --mem=2gb
#SBATCH --time=01:00:00
#SBATCH --account=nvidia-ai
#SBATCH --qos=nvidia-ai
#SBATCH --reservation=hackathon

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

ml nvhpc

export PATH=/red/nvidia-ai/miller-lab/env/venv/bin/:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/red/nvidia-ai/miller-lab/env/venv/lib/

nsys profile -t nvtx,osrt --force-overwrite=true --stats=true --output=first-profile python library-creator.py
