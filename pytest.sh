#!/bin/bash

#SBATCH --job-name=slurm_conda_python_example
#SBATCH --nodes=10
#SBATCH --time=00-00:05:00
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=2
#SBATCH --mem=2GB
#SBATCH --output=./out/test.txt

# see "Setup Conda" above or consult "module avail miniconda3" to get the right package name
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate py311f

python3 ./source/test.py
