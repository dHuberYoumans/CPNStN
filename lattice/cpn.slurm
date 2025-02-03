#!/bin/bash

# Slurm job options
#SBATCH --job-name=lattice_cpn_unet
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=dp358  

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

# load modules
module load gcc/9.3.0
module load cuda/12.3
module load openmpi/4.1.5-cuda12.3 

source activate
conda activate pytorch2.5

# name of script
application="main.py"


# run script 
echo "run started on " `date`

srun torchrun \
	--nnodes=$SLURM_JOB_NUM_NODES \
	--nproc_per_node=4 \
	${application}

echo "run completed on " `date`
