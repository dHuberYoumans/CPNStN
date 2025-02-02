#!/bin/bash

# Slurm job options
#SBATCH --job-name=lattice_cpn_test
#SBATCH --time=00:01:00
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=dp358  

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

# Load modules
module load gcc/9.3.0
module load cuda/12.3
module load openmpi/4.1.5-cuda12.3 

source activate
conda activate pytorch2.5

# Name of script
application="./CPNStN/lattice/test.py"

# Run script 
srun torchrun \
	--nnodes=1 \
	--nproc_per_node=2 \
	${application}

