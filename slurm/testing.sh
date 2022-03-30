#!/bin/bash
#SBATCH --job-name=test_profile
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --time=00:30:00
#SBATCH --output=./slurm/logs/profile_test_%j.log
#SBATCH --account=nvidia-ai
#SBATCH --qos=nvidia-ai
#SBATCH --reservation=hackathon
#SBATCH --partition=hpg-ai
#SBATCH --gres=gpu:1


echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

ml nvhpc

export PATH=/red/nvidia-ai/miller-lab/env/venv/bin/:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/red/nvidia-ai/miller-lab/env/venv/lib/

#nsys profile -t nvtx,osrt --force-overwrite=true --stats=true --output=first-profile python /red/nvidia-ai/miller-lab/JTA_FFT/library-creator.py

python /red/nvidia-ai/miller-lab/JTA_FFT/library-creator.py