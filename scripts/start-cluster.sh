#!/bin/bash
#SBATCH -C knl
#SBATCH -t 2:00:00
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -J ipyparallel_setup

#load modules
module load tensorflow/intel-1.13.0-py36-dev

#get master ip
head_ip=$(ip addr show ipogif0 | grep '10\.' | awk '{print $2}' | awk -F'/' '{print $1}')

# Unique cluster ID for this job
cluster_id=cori_${SLURM_JOB_ID}

# Cluster controller
ipcontroller --ip="$head_ip" --cluster-id=$cluster_id &
sleep 2

# Compute engines
srun -u -n $(( $SLURM_JOB_NUM_NODES )) ipengine --cluster-id=$cluster_id
