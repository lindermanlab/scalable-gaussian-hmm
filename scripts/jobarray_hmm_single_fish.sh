#!/bin/bash
#SBATCH --partition=hns
#SBATCH --mincpus=8
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --output=/scratch/users/%u/kf_tmp/prof.%j.out
#SBATCH --error=/scratch/users/%u/kf_tmp/prof.%j.err
#SBATCH --job-name=profile_fish
#SBATCH --array=[5,10,20,30,40,50]%3

# =========================================================
# Slurm array job script for profiling EM on a single fish
#
# Parameters to vary
#   SBATCH's --mincpus (MUST DO THIS MANUALLY)
#   num_frames_per_batch (with SBATCH's --mem)
#   num_hmm_states (with SBATCH's --mem)
#
# Here, we use the array parameter to vary over num_hmm_states
# =========================================================

# Load modules and activate virtual environment
# NB: `venv activate` is a custom bash function. See Libby for code
source ~/.bashrc

module purge > /dev/null
module load python/3.9.0

venv activate kf-cpu

echo 'JOB ID: ' $SLURM_JOB_ID, 'NUM_HMM_STATES = ' $SLURM_ARRAY_TASK_ID

# Set XLA_FLAGS to match number of CPUs allocated for this job
echo 'CPUS for this job: ' $SLURM_JOB_CPUS_PER_NODE, 'CPUS available on this node: ' $SLURM_CPUS_ON_NODE
export XLA_FLAGS=--xla_force_host_platform_device_count=$SLURM_JOB_CPUS_PER_NODE

# ---------------------------------------------------------------------------
# HMM FIT PARAMETERS
# Memory req's determined by size of these 2 parameters: how much data is loaded
# per partial-EM iteration, and as a result, how much compute memory req'd
batch_size=$SLURM_JOB_CPUS_PER_NODE
num_frames_per_batch=72000 # 20 * 60 * 60 * 1 = 1 hr

num_hmm_states=$SLURM_ARRAY_TASK_ID
num_em_iters=20

# "Small"
frac_train=0.05   # 10 files
frac_test=0.005  # 1 file

seed=20220627

cd ~/killifish/scripts

# Capture memory usage of main process and child processes (-C)
mprof run -C -o $TEMPDIR/prof.$SLURM_JOB_ID\.mprof ./script_pmap_single.py --seed $seed --batch_size $batch_size --num_frames_per_batch $num_frames_per_batch --num_train $frac_train --num_test $frac_test --num_hmm_states $num_hmm_states --num_em_iters $num_em_iters

# Print peak memory usage
mprof peak $TEMPDIR\prof.$SLURM_JOB_ID\.mprof

# Make mem usage plot with slope (-s)
mprof plot -s $TEMPDIR\prof.$SLURM_JOB_ID\.mprof -o $TEMPDIR\prof.$SLURM_JOB_ID\.mprof.png