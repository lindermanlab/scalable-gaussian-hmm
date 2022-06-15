# Time EM on real data
# From killifish directory
#   python examples/script_em_real_data.py --method vmap --profile mem --batch_size 3 --num_train 6 --num_test 1 --num_em_iters 10

import os
import argparse
from datetime import datetime

from jax import jit, vmap, pmap, lax
import jax.numpy as np
import jax.random as jr

import jax.profiler

from functools import partial, reduce
from operator import mul

from ssm_jax.hmm.models import GaussianHMM
from kf.inference import (sharded_e_step, collective_m_step,
                          fullbatch_e_step, fullbatch_m_step,
                          NormalizedGaussianHMMSuffStats as NGSS)

from kf.data_utils import FishPCDataset, FishPCDataloader

import pdb

DATADIR = os.environ['DATADIR']
TEMPDIR = os.environ['TEMPDIR']

# -------------------------------------
# Suppress JAX/TFD warning: ...`check_dtypes` is deprecated... message
import logging
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger = logging.getLogger()
logger.addFilter(CheckTypesFilter())

# -------------------------------------

parser = argparse.ArgumentParser(description='Profiling of EM parallelization')
parser.add_argument(
    '--method', type=str, default='nomap',
    choices=['pmap','vmap','nomap'],
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
    '--batch_size', type=int, default=None,
    help='Number of days per batch.')
parser.add_argument(
    '--num_train', type=int, default=1,
    help='Number of days to train over')
parser.add_argument(
    '--num_test', type=int, default=1,
    help='Number of days to test over')
parser.add_argument(
    '--num_em_iters', type=int, default=10,
    help='Number of EM iterations to run')
parser.add_argument(
    '--num_hmm_states', type=int, default=20,
    help='Number of HMM states to fit')


# -------------------------------------

def fit_hmm_vmap(train_dataset, test_dataset, init_hmm,
                 num_iters, batch_size, num_frames_per_day):
    """Fit HMM to FishPCDataset by vmapping over batches using EM.

    Parameters
        train_dataset: FishPCDataset
        test_dataset: FishPCDataset
        init_hmm: GaussianHMM
        num_iters: int. Number of EM steps to run.
        batch_size: int. Number of days per batch
    """

    emissions_dim = train_dataset.dim

    # Load all emissions, shape (num_days, num_frames_per_day, dim)
    train_dl = FishPCDataloader(train_dataset,
                                batch_size=batch_size,
                                num_frames_per_day=num_frames_per_day)

    test_dl  = FishPCDataloader(test_dataset,
                                batch_size=1,
                                num_frames_per_day=num_frames_per_day)

    hmm = init_hmm

    print('Anticipated train, test emissions array nbytes [MB]')
    get_nbytes_32 = lambda shp: reduce(mul, shp) * 4
    print(get_nbytes_32(train_dl.data_batch_shape)/(1024**2),
          get_nbytes_32(test_dl.data_batch_shape)/(1024**2),)

    # Computed average marginal loglikelihood, weighted by number of obs
    # NB: If weird numbers appear, consider int32/float32 overflow issues...
    compute_avg_marginal_loglik = lambda ngss: \
        (ngss.num_emissions/ngss.num_emissions.sum() * ngss.marginal_loglik).sum()/ngss.num_emissions.sum()

    train_and_test_lls = -np.ones((num_iters, 2)) * np.inf
    for itr in range(num_iters):
        def e_step(hmm, dl):
            _ngss = [vmap(partial(sharded_e_step, hmm))(ems) for ems in dl]    
            ngss = reduce(NGSS.concat, _ngss)
            ll = compute_avg_marginal_loglik(ngss)
            return ngss, ll
        
        train_ngss, train_ll = e_step(hmm, train_dl)
        
        hmm = collective_m_step(train_ngss)
        
        # --------------------------------------------------------

        _, test_ll = e_step(hmm, test_dl)

        train_and_test_lls = train_and_test_lls.at[itr].set(np.array([train_ll, test_ll]))

    return train_and_test_lls[:,0], train_and_test_lls[:,1]
    

def fit_hmm_nomap(train_dataset, test_dataset, init_hmm,
                  num_iters, batch_size=None, num_frames_per_day=-1):
    """Fit HMM to FishPCDataset using full batch expectation maximization.

    Parameters
        train_dataset: FishPCDataset
        test_dataset: FishPCDataset
        init_hmm: GaussianHMM.
        num_iters: int. Number of EM steps to run.
        batch_size: int. Ignored in this function.
    """

    emissions_dim = train_dataset.dim

    # Load all emissions, shape (num_days*uniform_num_frames_per_day, dim)
    train_dl = FishPCDataloader(train_dataset,
                                batch_size=len(train_dataset),
                                num_frames_per_day=num_frames_per_day)
    train_emissions = next(iter(train_dl)).reshape(-1, emissions_dim)

    test_dl = FishPCDataloader(test_dataset,
                               batch_size=len(test_dataset),
                               num_frames_per_day=num_frames_per_day)
    test_emissions = next(iter(test_dl)).reshape(-1, emissions_dim)

    print('train, test emissions array nbytes [MB]')
    print(train_emissions.nbytes/(1024**2), test_emissions.nbytes/(1024**2))

    @jit
    def em_step(hmm, itr):
        train_posterior = fullbatch_e_step(hmm, train_emissions)
        new_hmm = fullbatch_m_step(train_posterior, train_emissions)

        avg_train_ll = train_posterior.marginal_loglik / len(train_emissions)

        # --------------------------------------------------------
        test_posterior = fullbatch_e_step(new_hmm, test_emissions)
        avg_test_ll = test_posterior.marginal_loglik / len(test_emissions)

        return new_hmm, np.array([avg_train_ll, avg_test_ll])

    fitted_hmm, train_and_test_lls = lax.scan(em_step, init_hmm, np.arange(num_iters))

    return train_and_test_lls[:,0], train_and_test_lls[:,1]

def fit_hmm_pmap(em_step, train_dataset, test_dataset, initial_hmm, seed, num_iters):
    """Fit HMM to FishPCDataset using expectation maximization.

    Parameters
        em_step: Callable[[GaussianHMM, np.ndarray] -> [GaussianHMM, NGSS]]
        train_dataset: FishPCDataset
        test_dataset: FishPCDataset
        initial_hmm: GaussianHMM
        num_iters: int
            Number of EM steps to run.
    """

    for itr in range(num_iters):
        # Run full-batch E-step
        pass

def main():
    args = parser.parse_args()
    method = args.method
    profile = args.profile
    logdir = args.logdir

    seed = jr.PRNGKey(args.seed)
    batch_size = args.batch_size
    num_train = args.num_train
    num_test = args.num_test
    num_em_iters = args.num_em_iters
    num_hmm_states = args.num_hmm_states
    num_frames_per_day = 100000 # 100K seems loadable interval
    
    seed_split_data, seed_init_hmm = jr.split(seed, 2)

    # Initialize training and testing datasets
    print(f"Initializing training ({num_train} days) " 
          + f"and testing ({num_test} days) datasets, "
          + f"with {num_frames_per_day} frames per day...")
    train_ds, test_ds = \
        FishPCDataset('fish0_137', DATADIR).train_test_split(num_train=num_train,
                                                             num_test=num_test,
                                                             seed=seed_split_data)

    # Initialize hidden Markov model
    print(f'Initializing HMM with {num_hmm_states} states...')
    emission_dim = train_ds.dim
    init_hmm = GaussianHMM.random_initialization(seed_init_hmm,
                                                 num_hmm_states,
                                                 emission_dim)

    # Fit
    if method == 'vmap':
        fit_fn = fit_hmm_vmap
    elif method =='pmap':
        raise NotImplementedError
    elif method == 'nomap':
        fit_fn = fit_hmm_nomap
    
    if profile == 'time':
        raise NotImplementedError
    elif profile == 'mem':
        train_lls, test_lls = \
                fit_fn(train_ds, test_ds, init_hmm,
                       num_em_iters, batch_size, num_frames_per_day)
        train_lls.block_until_ready()

        log_fname = datetime.now().strftime("%y%m%d_%H%M%S") \
                    + f"-memory_{method}.prof"
        jax.profiler.save_device_memory_profile(os.path.join(TEMPDIR, log_fname))
    else:
        train_lls, test_lls = fit_fn(train_ds, test_ds, init_hmm,
                                     num_em_iters, batch_size, num_frames_per_day)

    print('train log likelihoods')
    print(train_lls)
    print('\n---------\n')
    print('test log likelihoods')
    print(test_lls)
    return

if __name__ == '__main__':
    main()