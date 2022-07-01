#!/bin/bash
#SBATCH --job-name=hmm_fit
#SBATCH --partition=hns
#SBATCH --time=02:00:00
#SBATCH --mem=10G
#SBATCH --output=/scratch/users/%u/kf_tmp/job.%j.out
#SBATCH --error=/scratch/users/%u/kf_tmp/job.%j.err

# SBATCH --mincpus=8
# SBATCH --mem-per-cpu=2G

# NB: Apparently --chdir does not accept pattersn such as %u. So use `cd` below
# NB: Important to specify `--mincpus`  instead of `--cpus-per-task` since
#     Slurm seems to consider the former a "constraint"/"hard requirement"
#     whereas it considers the latter as more of a "soft requirement". 

# ================================================
# Slurm job script for testing EM on a single fish
# ================================================

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
echo 'JOB ID: ' $SLURM_JOB_ID
echo 'MEM PER NODE: ' $SLURM_MEM_PER_NODE, 'MEM_PER_CPU: ' $SLURM_MEM_PER_CPU

# ---------------------------------------------------------------------------
# HMM FIT PARAMETERS
# Memory req's determined by size of these 2 parameters: how much data is loaded
# per partial-EM iteration, and as a result, how much compute memory req'd
# 10 x 8h of data [f/s x s/min x min/hr] ~= 22 MB
batch_size=8
# num_frames_per_batch=576000 # 8hrs of data=20 * 60 * 60 * 8
frames_per_batch=72000 # 20 * 60 * 60 * 1 hr

# "Full"
# frac_train=0.9    # 185 files
# frac_test =0.1    # 20 files

# "Small"
frac_train=0.05   # 10 files
frac_test=0.005  # 1 file

num_hmm_states=20
num_em_iters=20

seed=20220627

# Capture memory usage of main process and child processes (-C) or multiprocessing (-M)
mpfile=$TEMPDIR$SLURM_JOB_ID\.mprof

mprof run -M -o $mpfile \
    ./profile_pmap.py \
    --method vmap \
    --seed $seed \
    --batch_size $batch_size \
    --frames_per_batch $frames_per_batch \
    --num_train $frac_train \
    --num_test $frac_test \
    --num_hmm_states $num_hmm_states \
    --num_em_iters $num_em_iters

mprof peak $mpfile                      # Print peak memory usage
mprof plot $mpfile -o $mpfile.png -s    # Make figure (-o), with slope (-s)