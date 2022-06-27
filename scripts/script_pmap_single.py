# pmap/vmap script for running EM on a single fish
# WIP
#
# NB: If using PMAP, must
#   1. Request node with multiple CPUs, e.g.
#       - $ sdev -p dev -m 8G -c NUM_CPUS
#   2. Manually set XLA_FLAGS in shell...doesn't seem to register when setting from script
#       - $ export XLA_FLAGS=--xla_force_host_platform_device_count=NUM_CPUS
#   3. Now, we can finally run the script. From killifish directory,
#       - python scripts/script_pmap_single.py --method pmap --profile mem --batch_size NUM_CPUS --num_train 6 --num_test 1 --num_em_iters 10

import os
import argparse
import time
from datetime import datetime

from jax import jit, vmap, pmap, lax, local_device_count
import jax.numpy as np
import jax.random as jr

from functools import partial, reduce
from operator import mul

from ssm_jax.hmm.models import GaussianHMM
from kf.inference import (sharded_e_step, collective_m_step,
                          NormalizedGaussianHMMSuffStats as NGSS)
from kf.data_utils import FishPCDataset, FishPCDataloader, save_hmm
from tqdm import trange

DATADIR = os.environ['DATADIR']
TEMPDIR = os.environ['TEMPDIR']
fish_id = 'fish0_137'

# Calculate byte size of arrays, assuming float32 precision
get_nbytes_32 = lambda shp: reduce(mul, shp) * 4

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
    '--logdir', type=str, default=TEMPDIR,
    help='Directory to log profiles.')
parser.add_argument(
    '--seed', type=int, default=45212276,
    help='PRNG seed to split data and intialize HMM.')
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='Number of batches loaded per iteration.')
parser.add_argument(
    '--num_frames_per_batch', type=int, default=100000,
    help='Number of frames per batch.')
parser.add_argument(
    '--num_train', type=float, default=1,
    help='Fraction of dataset to train over')
parser.add_argument(
    '--num_test', type=float, default=1,
    help='Fraction of dataset to test over')
parser.add_argument(
    '--num_em_iters', type=int, default=10,
    help='Number of EM iterations to run')
parser.add_argument(
    '--num_hmm_states', type=int, default=20,
    help='Number of HMM states to fit')

# -------------------------------------

def fit_hmm_jmap(train_dataset, test_dataset, hmm,
                 num_iters, batch_size, num_frames_per_batch, method='vmap'):
    """Fit HMM to FishPCDataset by vmapping over batches using EM.

    Parameters
        train_dataset: FishPCDataset
        test_dataset: FishPCDataset
        hmm: GaussianHMM
        num_iters: int. Number of EM steps to run.
        batch_size: int. Number of days per batch
        num_frames_per_batch: int
        method: str. The jax parallelization method, either 'vmap' or 'pmap'.
    """

    jmap = vmap if method == 'vmap' else pmap
    
    emissions_dim = train_dataset.dim

    # Load all emissions, shape (num_days, num_frames_per_batch, dim)
    train_dl = FishPCDataloader(train_dataset,
                                batch_size=batch_size,
                                num_frames_per_batch=num_frames_per_batch)

    test_dl  = FishPCDataloader(test_dataset,
                                batch_size=1,
                                num_frames_per_batch=num_frames_per_batch)

    print('Anticipated train, test emissions array nbytes [MB]')
    print(get_nbytes_32(train_dl.batch_shape)/(1024**2),
          get_nbytes_32(test_dl.batch_shape)/(1024**2),)

    train_lls = -np.ones(num_iters) * np.inf
    test_lls  = -np.ones(num_iters) * np.inf

    pbar = trange(num_iters)
    pbar.set_description(f"Train: -inf, test: -inf")
    for itr in pbar:
        start_time = time.perf_counter()
        # --------------------------------------------------------
        def e_step(hmm, dl):
            _ngss = [jmap(partial(sharded_e_step, hmm))(ems) for ems in dl]    
            ngss = reduce(NGSS.concat, _ngss)
            return ngss, ngss.batch_marginal_loglik()
        
        train_ngss, train_ll = e_step(hmm, train_dl)
        hmm = collective_m_step(train_ngss)
        
        # --------------------------------------------------------
        # Evaluate on test set and save log-likelihoods
        _, test_ll = e_step(hmm, test_dl)

        train_lls = train_lls.at[itr].set(train_ll)
        test_lls  = test_lls.at[itr].set(test_ll)
    
        print(f"i{itr}: {time.perf_counter()-start_time:0.4f} s")
        pbar.set_description(f"Train: {train_ll:.2f}, test: {test_ll:.2f}")
        pbar.update()

    return hmm, train_lls, test_lls
    
FIT_HMM = dict(vmap=partial(fit_hmm_jmap, method='vmap'),
               pmap=partial(fit_hmm_jmap, method='pmap'),
               )

def main():
    args = parser.parse_args()
    method = args.method
    profile = args.profile
    logdir = args.logdir

    seed = jr.PRNGKey(args.seed)
    batch_size = args.batch_size
    num_frames_per_batch = args.num_frames_per_batch

    num_train = args.num_train
    num_test = args.num_test
    
    num_hmm_states = args.num_hmm_states
    num_em_iters = args.num_em_iters

    # ==========================================================================
    
    if method =='pmap':
        assert local_device_count() >= args.batch_size, \
               f"Found {local_device_count()} devices, expected {args.batch_size} devices."
        print(f'USING PMAP: found {local_device_count()} devices on node')
    # ==========================================================================
    seed_split_data, seed_init_hmm = jr.split(seed, 2)

    # full_ds = FishPCDataset(fish_id, DATADIR,
    #                         data_subset='all',
    #                         min_num_frames=num_frames_per_batch)
    full_ds = FishPCDataset(fish_id, DATADIR,
                            data_subset='all',
                            min_num_frames=num_frames_per_batch,
                            max_num_frames=600000) # used for debugging
    train_ds, test_ds = full_ds.train_test_split(num_train=num_train,
                                                 num_test=num_test,
                                                 seed=seed_split_data)
    
    print(f"Initializing training ({len(train_ds)} days) " 
          + f"and testing ({len(test_ds)} days) datasets, "
          + f"with {num_frames_per_batch} frames per batch...")

    # --------------------------------------------------------------------------
    # Start trace
    if profile == 'time':
        jax.profiler.start_trace(logdir)
    
    # --------------------------------------------------------------------------
    # Initialize hidden Markov model
    print(f'Initializing HMM with {num_hmm_states} states...')
    init_hmm = GaussianHMM.random_initialization(seed_init_hmm,
                                                 num_hmm_states,
                                                 train_ds.dim)

    # Fit function
    print(f'Running...')
    hmm, train_lls, test_lls = \
                FIT_HMM[method](train_ds, test_ds, init_hmm, num_em_iters,
                                batch_size, num_frames_per_batch)
    train_lls.block_until_ready()

    # --------------------------------------------------------------------------
    # Stop trace
    
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    print(f'Finished, saving results to file with prefix {timestamp}')
    if profile == 'time':
        jax.profiler.stop_trace()
    elif profile == 'mem':
        jax.profiler.save_device_memory_profile(
                    os.path.join(logdir, f"{timestamp}-memory_{method}.prof"))    

    # ----
    # Save likelihoods and hmm
    fpath = os.path.join(TEMPDIR, f"{timestamp}-{num_hmm_states}_states-{num_em_iters}_iters.npz")
    save_hmm(fpath, hmm, train_lls=train_lls, test_lls=test_lls)
    return

if __name__ == '__main__':
    main()