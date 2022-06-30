# Script for profiling HMM fit via pmap parallelization

import os
import argparse
from datetime import datetime

from memory_profiler import profile

import jax.numpy as np
import jax.random as jr
from ssm_jax.hmm.models import GaussianHMM
from kf.data_utils import FishPCDataset, FishPCDataloader, save_hmm
from kf.fit import fit_pmap

DATADIR = os.environ['DATADIR']
TEMPDIR = os.environ['TEMPDIR']
fish_id = 'fish0_137'

# -------------------------------------
# Suppress JAX/TFD warning: ...`check_dtypes` is deprecated... message
import logging
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())

# -------------------------------------

parser = argparse.ArgumentParser(description='Single-subject EM parallelization')
parser.add_argument(
    '--method', type=str, default='pmap',
    choices=['pmap','vmap'],
    help='Parallelization approach. NB: If using pmap, must run on node with multiple cores.')
parser.add_argument(
    '--profile', type=str, default='none',
    choices=['time', 'mem', 'none'],
    help='What aspect of code to profile')
parser.add_argument(
    '--log_dir', type=str, default=TEMPDIR,
    help='Directory to log profiles.')
parser.add_argument(
    '--log_prefix', type=str, default=None,
    help='Prefix for log files.')
parser.add_argument(
    '--seed', type=int, default=45212276,
    help='PRNG seed to split data and intialize HMM.')
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='Number of batches loaded per iteration.')
parser.add_argument(
    '--frames_per_batch', type=int, default=100000,
    help='Number of frames per batch. Also used to filter out files which do not have this many frames.')
parser.add_argument(
    '--max_frames_per_day', type=int, default=-1,
    help='Maximum number of frames available in each file/day. Useful for debugging.')
parser.add_argument(
    '--num_train', type=float, default=1,
    help='If >1, number of files in dataset to train over. If [0, 1), fraction of dataset to train over.')
parser.add_argument(
    '--num_test', type=float, default=1,
    help='If >1, number of files in dataset to test over. If [0, 1), fraction of dataset to test over.')
parser.add_argument(
    '--num_em_iters', type=int, default=10,
    help='Number of EM iterations to run')
parser.add_argument(
    '--num_hmm_states', type=int, default=20,
    help='Number of HMM states to fit')

# =============================================================================

# @profile(precision=2)
def main():
    # TODO Allow warm-starting of hmm fit code from saved file.
    # - Add argument to to start with random initialization or from file
    # - If warm-starting, extract last iteration and log likelihoods, same seed for splitting dataset
    args = parser.parse_args()
    method = args.method
    profile = args.profile
    log_dir = args.log_dir
    log_prefix = args.log_prefix        

    seed = jr.PRNGKey(args.seed)
    batch_size = args.batch_size
    frames_per_batch = args.frames_per_batch
    max_frames_per_day = args.max_frames_per_day

    num_train = args.num_train
    num_test = args.num_test
    
    num_hmm_states = args.num_hmm_states
    num_em_iters = args.num_em_iters

    if log_prefix is None:
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        log_prefix=f"{timestamp}-{num_hmm_states}st-{num_em_iters}it"
    # ==========================================================================
    seed_split_data, seed_init_hmm = jr.split(seed, 2)

    full_ds = FishPCDataset(fish_id, DATADIR,
                            data_subset='all',
                            min_frames_per_file=frames_per_batch,
                            max_frames_per_file=max_frames_per_day)
    train_ds, test_ds = full_ds.train_test_split(num_train=num_train,
                                                 num_test=num_test,
                                                 seed=seed_split_data)
    
    del full_ds

    # TODO Move train_test_split function to FishPCDataloader class
    # so that we can shuffle over batches (and not just over days)
    # Load all emissions, shape (num_days, num_frames_per_batch, dim)
    train_dl = FishPCDataloader(train_ds,
                                batch_size=batch_size,
                                num_frames_per_batch=frames_per_batch)
          
    test_dl  = FishPCDataloader(test_ds,
                                batch_size=batch_size,
                                num_frames_per_batch=frames_per_batch)
    
    print(f"Initialized datasets ({len(train_ds)} days training, {len(test_ds)} days testing)")
    print(f"\ttrain: {len(train_dl):3d} batches of shape {train_dl.batch_shape}")
    print(f"\ttest : {len(test_dl):3d} batches of shape {test_dl.batch_shape}")
        
    # --------------------------------------------------------------------------
    # Initialize hidden Markov model
    print(f'Initializing HMM with {num_hmm_states} states...')
    init_hmm = GaussianHMM.random_initialization(seed_init_hmm, num_hmm_states, train_ds.dim)

    # Run function
    fn_args = (train_dl, test_dl, init_hmm,)
    fn_kwargs = {'num_iters': num_em_iters, 'll_fmt':'.4f'}

    if profile == 'mem':
        from memory_profiler import memory_usage
        with open(log_prefix+'.mprof', 'a') as fstream:
            mem_usage, (hmm, train_lls, test_lls) = \
                        memory_usage((fit_pmap, fn_args, fn_kwargs),
                                      retval=True,
                                      include_children=False,
                                      multiprocess=True,
                                      stream=fstream)

    else:
        hmm, train_lls, test_lls = fit_pmap(*fn_args, **fn_kwargs)
        train_lls.block_until_ready()

    # --------------------------------------------------------------------------
    # Save likelihoods and hmm
    # save_hmm(log_prefix, hmm, train_lls=train_lls, test_lls=test_lls)
    return

if __name__ == '__main__':
    main()