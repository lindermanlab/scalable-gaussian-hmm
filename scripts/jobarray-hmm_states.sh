#!/bin/bash
#SBATCH --partition=hns
#SBATCH --time=03:00:00
#SBATCH --mem=20G
#SBATCH --output=/scratch/users/%u/kf_tmp/prof.%j.out
#SBATCH --error=/scratch/users/%u/kf_tmp/prof.%j.err
#SBATCH --job-name=profile_fish
#SBATCH --array=[5,10,20]


# nSBATCH --mincpus=50
# nSBATCH --mem-per-cpu=25G

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

# In addition to loading required packages, this also...
# - Sets environmental variables DATADIR and TEMPDIR
# - Reveals available CPUs (if >1) to JAX by setting XLA_FLAGS
venv activate kf-cpu

cd ~/killifish/scripts

echo 'JOB ID: ' $SLURM_JOB_ID, 'ARRAY ID = ' $SLURM_ARRAY_JOB_ID\_$SLURM_ARRAY_TASK_ID
echo 'MEM PER NODE: ' $SLURM_MEM_PER_NODE, 'MEM_PER_CPU: ' $SLURM_MEM_PER_CPU

# ---------------------------------------------------------------------------
# HMM FIT PARAMETERS
# Memory req's determined by size of these 2 parameters: how much data is loaded
# per partial-EM iteration, and as a result, how much compute memory req'd
# batch_size=$SLURM_JOB_CPUS_PER_NODE
batch_size=8
# frames_per_batch=72000 # 20 * 60 * 60 * 1 = 1 hr
frames_per_batch=144000 # 20 * 60 * 60 * 2 = 2 hr

num_hmm_states=$SLURM_ARRAY_TASK_ID
num_em_iters=20

# "Small"
frac_train=10
frac_test=0.005     # 1 file

seed=20220627

# Capture memory usage of main process and child processes (-C) or multiprocessing (-M)
mpfile=$TEMPDIR$SLURM_JOB_ID\.mprof

mprof run -M -o $mpfile \
    ./profile_pmap.py \
    --method vmap \
    --num_hmm_states $num_hmm_states \
    --batch_size $batch_size \
    --frames_per_batch $frames_per_batch \
    --num_train $frac_train \
    --num_test $frac_test \
    --num_em_iters $num_em_iters
    --seed $seed \

mprof peak $mpfile                      # Print peak memory usage
mprof plot $mpfile -o $mpfile.png -s    # Make figure (-o), with slope (-s)