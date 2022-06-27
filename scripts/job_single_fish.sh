#!/bin/bash
#SBATCH --job-name=single_fish_pmap
#SBATCH --partition=hns
#SBATCH --cpus-per-task=4
#SBATCH --mincpus=4
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --chdir ~/killifish/scripts/
#SBATCH --output=/scratch/users/%u/kf_tmp/job.%j.out
#SBATCH --error=/scratch/users/%u/kf_tmp/job.%j.err

# NB: Important to specify `mincpus`` to be <= `cpus-per-task``.
# This is the minimum number of CPUs required on node, and a "hard-requirement"
# for SLURM to fulfill, where as `cpus-per-task` is more a "soft-requirement". 
# If unspecified, Slurm happily allocates a node with 1 CPU.

# ================================================
# Slurm job script for testing EM on a single fish
# ================================================

# Load modules and activate virtual environment
# NB: `venv activate` is a custom bash function. See Libby for code
source ~/.bashrc

module purge > /dev/null
module load python/3.9.0

venv activate kf-cpu

# Set XLA_FLAGS to match number of CPUs allocated for this job
echo 'CPUS for this job ' $SLURM_JOB_CPUS_PER_NODE, 'CPUS available on this node ' $SLURM_CPUS_ON_NODE
export XLA_FLAGS=--xla_force_host_platform_device_count=$SLURM_JOB_CPUS_PER_NODE

# ---------------------------------------------------------------------------
# HMM FIT PARAMETERS
# Memory req's determined by size of these 2 parameters: how much data is loaded
# per partial-EM iteration, and as a result, how much compute memory req'd
# 10 x 8h of data [f/s x s/min x min/hr] ~= 22 MB
# batch_size=10
# num_frames_per_batch=576000 # 8hrs of data=20 * 60 * 60 * 8
batch_size=2
num_frames_per_batch=72000 # 20 * 60 * 60 * 1 hr

# "Full"
# frac_train=0.9    # 185 files
# frac_test =0.1    # 20 files

# "Small"
frac_train=0.05   # 10 files
frac_test=0.005  # 1 file

num_hmm_states=20
num_em_iters=20

seed=20220627

# python -u flag: Force  stdin,  stdout and stderr to be totally unbuffered.
python -u script_pmap_single.py --method pmap --seed $seed --batch_size $batch_size --num_frames_per_batch $num_frames_per_batch --num_train $frac_train --num_test $frac_test --num_hmm_states $num_hmm_states --num_em_iters $num_em_iters