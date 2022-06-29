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

# python script_pmap_single.py --batch_size 4 --num_frames_per_batch 72000 --num_train 0.05 --num_test 0.005 --num_hmm_states 20 --num_em_iters 20

import os
import argparse
from datetime import datetime
from tqdm import tqdm
from sys import stdout

from memory_profiler import profile

from functools import partial, reduce
from operator import mul

from jax import jit, vmap, pmap, lax, local_device_count
import jax.numpy as np
import jax.random as jr

from ssm_jax.hmm.models import GaussianHMM
from kf.inference import (sharded_e_step, collective_m_step,
                          NormalizedGaussianHMMSuffStats as NGSS)
from kf.data_utils import FishPCDataset, FishPCDataloader, save_hmm

import pdb

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

# =============================================================================

@profile(precision=2)
def fit_hmm_jmap(train_data, test_data, hmm,
                 num_iters, batch_size, num_frames_per_batch, method='vmap'):
    """Fit HMM to FishPCDataset by vmapping over batches using EM.

    Parameters
        train_data: FishPCDataloader
        test_data: FishPCDataloader
        hmm: GaussianHMM
        num_iters: int. Number of EM steps to run.
        batch_size: int. Number of days per batch
        num_frames_per_batch: int
        method: str. The jax parallelization method, either 'vmap' or 'pmap'.
    """

    jmap = vmap if method == 'vmap' else pmap

    train_lls = -np.ones(num_iters) * np.inf
    test_lls  = -np.ones(num_iters) * np.inf

    pbar = tqdm(iterable=range(num_iters),
                desc=f"Epochs",
                file=stdout,
                initial=0,
                postfix=f'train={-np.inf}, test={-np.inf}',)

    train_ngss = NGSS.empty((train_data.num_minibatches, hmm.num_states, hmm.num_obs))
    test_ngss  = NGSS.empty((test_data.num_minibatches,  hmm.num_states, hmm.num_obs))
    for itr in pbar:
        def e_step_inplace(hmm, dl, ngss):
            _e_step = jmap(partial(sharded_e_step, hmm))
            for i_batch, emissions in enumerate(dl):
                bslice = np.s_[i_batch*dl.batch_size:(i_batch+1)*dl.batch_size]
                ngss.batch_set(bslice, _e_step(emissions))
            return

        e_step_inplace(hmm, train_data, train_ngss)
        train_ll = train_ngss.batch_marginal_loglik()
        hmm = collective_m_step(train_ngss)
        
        # --------------------------------------------------------
        # Evaluate on test set and save log-likelihoods
        e_step_inplace(hmm, test_data, test_ngss)

        train_lls = train_lls.at[itr].set(train_ngss.batch_marginal_loglik())
        test_lls  = test_lls.at[itr].set(test_ngss.batch_marginal_loglik())
    
        pbar.set_postfix_str(f'train={train_ll:0.3f}, test={test_ll:0.3f}')

        # if np.isnan(train_ll) or np.isnan(test_ll):
        #     pdb.set_trace()
        
    return hmm, train_lls, test_lls
    
FIT_HMM = dict(vmap=partial(fit_hmm_jmap, method='vmap'),
               pmap=partial(fit_hmm_jmap, method='pmap'),)

def main():
    # TODO Allow warm-starting of hmm fit code from saved file.
    # - Add argument to to start with random initialization or from file
    # - If warm-starting, extract last iteration and log likelihoods, same seed for splitting dataset
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
    
    del full_ds
    # TODO Move train_test_split function to FishPCDataloader class
    # so that we can shuffle over batches (and not just over days)
    # Load all emissions, shape (num_days, num_frames_per_batch, dim)
    train_dl = FishPCDataloader(train_ds,
                                batch_size=batch_size,
                                num_frames_per_batch=num_frames_per_batch)
          
    test_dl  = FishPCDataloader(test_ds,
                                batch_size=batch_size,
                                num_frames_per_batch=num_frames_per_batch)
    
    print(f"Initialized datasets ({len(train_ds)} days training, {len(test_ds)} days testing)")
    print(f"\ttrain: {len(train_dl):3d} batches of shape {train_dl.batch_shape}")
    print(f"\ttest : {len(test_dl):3d} batches of shape {test_dl.batch_shape}")
        
    # --------------------------------------------------------------------------
    # Initialize hidden Markov model
    print(f'Initializing HMM with {num_hmm_states} states...')
    init_hmm = GaussianHMM.random_initialization(seed_init_hmm,
                                                 num_hmm_states,
                                                 train_ds.dim)

    # Fit function
    print(f'Running...')
    hmm, train_lls, test_lls = \
                FIT_HMM[method](train_dl, test_dl, init_hmm, num_em_iters,
                                batch_size, num_frames_per_batch)
    train_lls.block_until_ready()

    # --------------------------------------------------------------------------
    # Save likelihoods and hmm
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    fpath = os.path.join(TEMPDIR, f"{timestamp}-{num_hmm_states}_states-{num_em_iters}_iters.npz")
    save_hmm(fpath, hmm, train_lls=train_lls, test_lls=test_lls)
    return

if __name__ == '__main__':
    main()