#!/bin/bash

# Slurm job options
#SBATCH --job-name=lattice_cpn_unet
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=dp358  

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
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

# master-port and world size
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_ID | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "MASTER_ADDR"=$MASTER_ADDR
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

# run script 
echo "run started on " `date`

torchrun \
	--nnodes=$SLURM_JOB_NUM_NODES \
	--nproc_per_node=4 \
	--node_rank=$SLURM_NODEID \
	--master_addr=$MASTER_ADDR \
	--master_port=$MASTER_PORT \
	${application}

echo "run completed on " `date`
