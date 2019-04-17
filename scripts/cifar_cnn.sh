#!/bin/bash
#SBATCH -J cifar-cnn
#SBATCH -C knl
#SBATCH -N 1
#SBATCH --reservation=sc18
#SBATCH -q regular
#SBATCH -t 45
#SBATCH -o logs/%x-%j.out

. scripts/setup.sh
export HDF5_USE_FILE_LOCKING=FALSE

config=configs/cifar10_cnn.yaml
srun -N ${SLURM_NNODES} -c 272 -u python train.py $config
